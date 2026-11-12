"""
MRSD-C2 (Multiple Responses Subgroup Discovery, C2 = classification in both spaces, X and Y):
the ranking method.

For every starting subgroup G_i the membership classifiers in X space and in
Y space are cross-fitted (`classifiers.DualSpaceEvaluator`), which gives every
instance out-of-fold membership probabilities p_X(e) and p_Y(e).  Instances
are ranked by these probabilities and candidate subgroups are built as
top-n prefixes of the rankings for a set of sizes n in [n_min, n_max]:

* `rank_aggregation`:
    union          G_{X,n} ∪ G_{Y,n} (top-n of each ranking; the original
                   algorithm), plus the two individual prefixes;
    intersection   rank by the worse of the two ranks;
    borda          sum of Borda scores;
    rrf            reciprocal rank fusion (1 / (k + rank));
    weighted_borda weight_X * Borda_X + (1 - weight_X) * Borda_Y.
  For every method other than `union` the combined prefix is a candidate in
  addition to the union and the individual prefixes.
* `size_step_method`: all | linear | logarithmic (default) | custom sizes;
  the number of sizes is capped at `max_sizes` (evenly thinned).

Candidate index sets are deduplicated before evaluation.  Every candidate
is scored out-of-fold, q(G) = aggregate(q_X(G), q_Y(G)), and the final
subgroups are chosen by `selection_method`:
    best    the best candidate for every starting subgroup;
    pareto  the (q_X, q_Y) Pareto front with epsilon-dominance
            (`pareto_epsilon`), which removes near-duplicates on the front;
    top_k   the `top_k` best candidates overall.

Returned subgroups are re-scored with fresh folds (`final_rescoring`;
`quality` honest, `quality_search` optimistic), described by a shallow tree
in X space, pruned for redundancy and validated according to
`inference_statistic` ('y_shift': fixed index set, Y permuted, conditional on
the candidate set and anti-conservative; 'search_permutation': the whole
procedure re-run on permuted Y, honest but costly; see mrsdc2_swapping).

Earlier versions ranked instances with models evaluated on their own training
rows; with tree models every member then had probability 1 and the ranking
inside the subgroup was arbitrary.  All probabilities are now out-of-fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from .common import derive_seed, make_rng, mask_from_indices
from .data_loader import DataLoader, Dataset
from .mrsdc2_swapping import (build_c2_subgroup, finalize_c2_subgroups, make_c2_evaluator,
                             order_by_score, prepare_initial_subgroups, resolve_size_limits,
                             search_qualities)
from .core import SetOfSubgroups, Subgroup

logger = logging.getLogger("mrsd")


# =============================================================================
# ENUMS
# =============================================================================

class RankAggregationMethod(Enum):
    UNION = "union"
    INTERSECTION = "intersection"
    BORDA = "borda"
    RRF = "rrf"
    WEIGHTED_BORDA = "weighted_borda"


class SizeStepMethod(Enum):
    ALL = "all"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    CUSTOM = "custom"


class SelectionMethod(Enum):
    BEST = "best"
    PARETO = "pareto"
    TOP_K = "top_k"


# =============================================================================
# RANK AGGREGATION
# =============================================================================

class RankAggregator:
    """Combines the X and Y rankings of instances."""

    def __init__(self, method: Union[RankAggregationMethod, str] = RankAggregationMethod.UNION,
                 rrf_k: int = 60, weight_X: float = 0.5):
        self.method = RankAggregationMethod(method) if isinstance(method, str) else method
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if not 0.0 <= weight_X <= 1.0:
            raise ValueError("weight_X must be in [0, 1]")
        self.rrf_k = rrf_k
        self.weight_X = weight_X

    @staticmethod
    def _ranks(order: np.ndarray) -> np.ndarray:
        """rank[e] = position of e in `order` (0 = best)."""
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(len(order))
        return rank

    def get_combined_ranking(self, prob_X: np.ndarray, prob_Y: np.ndarray,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Instance indices sorted by combined rank (best first); ties broken at random."""
        rng = make_rng(rng)
        n = len(prob_X)
        rank_X = self._ranks(order_by_score(prob_X, rng))
        rank_Y = self._ranks(order_by_score(prob_Y, rng))
        if self.method == RankAggregationMethod.UNION:
            score = 0.5 * (prob_X + prob_Y)
        elif self.method == RankAggregationMethod.INTERSECTION:
            score = -np.maximum(rank_X, rank_Y).astype(float)
        elif self.method == RankAggregationMethod.BORDA:
            score = (n - rank_X) + (n - rank_Y)
        elif self.method == RankAggregationMethod.RRF:
            score = 1.0 / (self.rrf_k + rank_X) + 1.0 / (self.rrf_k + rank_Y)
        elif self.method == RankAggregationMethod.WEIGHTED_BORDA:
            score = self.weight_X * (n - rank_X) + (1 - self.weight_X) * (n - rank_Y)
        else:
            raise ValueError(self.method)
        return order_by_score(np.asarray(score, dtype=float), rng)

    @staticmethod
    def get_candidates_at_size(ranking: np.ndarray, size: int) -> Set[int]:
        return set(int(e) for e in ranking[:size])


# =============================================================================
# SIZE SELECTOR
# =============================================================================

class SizeSelector:
    """Candidate sizes between n_min and n_max."""

    def __init__(self, method: Union[SizeStepMethod, str] = SizeStepMethod.LOGARITHMIC,
                 linear_step: int = 10, custom_sizes: Optional[Sequence[int]] = None,
                 max_sizes: int = 15):
        self.method = SizeStepMethod(method) if isinstance(method, str) else method
        if linear_step < 1:
            raise ValueError("linear_step must be >= 1")
        if max_sizes < 1:
            raise ValueError("max_sizes must be >= 1")
        self.linear_step = linear_step
        self.custom_sizes = list(custom_sizes or [])
        self.max_sizes = max_sizes

    def get_sizes(self, n_min: int, n_max: int) -> List[int]:
        """Sorted distinct sizes in [n_min, n_max] (empty when n_min > n_max),
        thinned evenly to at most `max_sizes` values keeping the extremes."""
        if n_min > n_max:
            return []
        if self.method == SizeStepMethod.ALL:
            sizes = list(range(n_min, n_max + 1))
        elif self.method == SizeStepMethod.LINEAR:
            sizes = list(range(n_min, n_max + 1, self.linear_step))
            if sizes[-1] != n_max:
                sizes.append(n_max)
        elif self.method == SizeStepMethod.LOGARITHMIC:
            sizes = []
            size = max(n_min, 1)
            while size <= n_max:
                sizes.append(size)
                size = max(2 * size, size + 1)
            if not sizes or sizes[-1] != n_max:
                sizes.append(n_max)
        elif self.method == SizeStepMethod.CUSTOM:
            sizes = [int(s) for s in self.custom_sizes if n_min <= s <= n_max]
        else:
            raise ValueError(self.method)
        sizes = sorted(set(sizes))
        if len(sizes) > self.max_sizes:
            keep = np.unique(np.round(np.linspace(0, len(sizes) - 1, self.max_sizes)).astype(int))
            sizes = [sizes[i] for i in keep]
        return sizes


# =============================================================================
# PARETO FRONT
# =============================================================================

@dataclass
class RankingCandidate:
    indices: Set[int]
    q_X: float = 0.0
    q_Y: float = 0.0
    quality: float = 0.0
    source: str = ""
    target_size: int = 0
    init_indices: List[int] = field(default_factory=list)


def is_dominated(point: Tuple[float, float], other: Tuple[float, float], epsilon: float = 0.0) -> bool:
    """`point` is epsilon-dominated by `other`: `other` is at least as good up to
    epsilon in both coordinates and strictly better in at least one."""
    at_least = other[0] + epsilon >= point[0] and other[1] + epsilon >= point[1]
    strictly = other[0] > point[0] or other[1] > point[1]
    return at_least and strictly


def compute_pareto_front(candidates: List[RankingCandidate], epsilon: float = 0.0) -> List[RankingCandidate]:
    """Non-dominated candidates in (q_X, q_Y).  With epsilon > 0 candidates are
    visited by decreasing aggregated quality and one is kept only if no kept
    candidate epsilon-dominates it, which thins near-duplicates on the front."""
    front: List[RankingCandidate] = []
    for c in sorted(candidates, key=lambda c: (c.quality, c.q_X + c.q_Y), reverse=True):
        p = (c.q_X, c.q_Y)
        if any(is_dominated(p, (f.q_X, f.q_Y), epsilon) for f in front):
            continue
        front = [f for f in front if not is_dominated((f.q_X, f.q_Y), p, 0.0)]
        front.append(c)
    return front


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RankingMethodConfig:
    """Configuration of the ranking method."""
    # evaluation
    classifier_X: str = "random_forest"
    classifier_Y: str = "random_forest"
    n_folds: int = 5
    aggregation: str = "min"
    # size limits
    n_min: int = 5
    n_max: Optional[int] = None          # default n // 2
    # rank aggregation
    rank_aggregation: str = "union"
    rrf_k: int = 60
    weight_X: float = 0.5
    # candidate sizes
    size_step_method: str = "logarithmic"
    linear_step: int = 10
    custom_sizes: Optional[List[int]] = None
    max_sizes: int = 15
    # selection
    selection_method: str = "best"       # best | pareto | top_k
    top_k: int = 5
    pareto_epsilon: float = 0.02
    # starting subgroups
    initialization: str = "clustering"   # clustering | random
    n_initial_subgroups: int = 10
    # post-processing
    final_rescoring: bool = True
    redundancy_threshold: float = 0.7
    n_permutations: int = 50   # search permutations are full reruns of the search
    inference_statistic: str = "search_permutation"   # search_permutation (honest, slower) | y_shift (conditional) | none (see mrsdc2_swapping)
    # other
    random_state: Optional[int] = None
    verbose: bool = False


# =============================================================================
# ALGORITHM
# =============================================================================

class RankingMethod:
    """Ranking method of MRSD-C2 (see the module docstring)."""

    ALGORITHM = "MRSD-C2-ranking"

    def __init__(self, config: Optional[RankingMethodConfig] = None):
        self.config = config or RankingMethodConfig()
        self._all_subgroups: List[Subgroup] = []
        self.candidates: List[RankingCandidate] = []
        self.n_evaluations = 0

    # ------------------------------------------------------------------
    def fit(self, dataset: Dataset, initial_subgroups: Optional[Sequence] = None) -> SetOfSubgroups:
        """Build, evaluate and select ranking candidates for every starting subgroup."""
        cfg = self.config
        rng = make_rng(cfg.random_state)
        X = dataset.X_space.numeric
        Y = dataset.Y_space.numeric
        n = dataset.n_instances
        n_min, n_max = resolve_size_limits(n, cfg.n_min, cfg.n_max)
        aggregator = RankAggregator(cfg.rank_aggregation, cfg.rrf_k, cfg.weight_X)
        sizes = SizeSelector(cfg.size_step_method, cfg.linear_step, cfg.custom_sizes, cfg.max_sizes).get_sizes(n_min, n_max)
        SelectionMethod(cfg.selection_method)   # validate early
        starts_seed = derive_seed(rng)
        starts = prepare_initial_subgroups(cfg, X, Y, n, n_min, n_max, make_rng(starts_seed), initial_subgroups)
        if cfg.verbose:
            print(f"{self.ALGORITHM}: n={n}, sizes {sizes}, aggregation={aggregator.method.value}, "
                  f"{len(starts)} starting subgroups")

        seed = derive_seed(rng)

        def permuted_starts(Yp):
            # generated starts are re-derived from the permuted data (user-supplied stay fixed)
            if initial_subgroups is not None:
                return starts
            return prepare_initial_subgroups(cfg, X, Yp, n, n_min, n_max, make_rng(starts_seed), None)
        evaluator, rescorer, found = self._search(X, Y, n, n_min, n_max, seed, starts, aggregator, sizes)
        candidates = self.candidates
        if cfg.verbose:
            print(f"  {len(candidates)} distinct candidates evaluated, {len(found)} selected")

        subgroups = [build_c2_subgroup(self.ALGORITHM, G, dataset, evaluator, rescorer, extra=extra)
                     for G, extra in found]
        self.n_evaluations = evaluator.n_evaluations
        self._all_subgroups = subgroups
        result = finalize_c2_subgroups(
            subgroups, dataset, cfg, rng,
            search_fn=lambda Yp: search_qualities(
                self._search(X, Yp, n, n_min, n_max, seed, permuted_starts(Yp), aggregator, sizes), n))
        self.candidates = candidates   # the permutation re-runs overwrite it
        if cfg.verbose:
            print(result.summary())
        return result

    def _search(self, X: np.ndarray, Y: np.ndarray, n: int, n_min: int, n_max: int, seed: int,
                starts: Sequence[Set[int]], aggregator: RankAggregator, sizes: Sequence[int]):
        """Candidate generation, evaluation and selection for given data and seed
        (re-run on permuted Y by `search_permutation_test`).
        Returns (evaluator, rescorer, [(indices, extra_measures)])."""
        cfg = self.config
        rng = make_rng(seed)
        evaluator = make_c2_evaluator(cfg, X, Y, rng, fast=True)
        selection = SelectionMethod(cfg.selection_method)

        # candidates, deduplicated by index set before any evaluation
        unique: Dict[frozenset, RankingCandidate] = {}

        def add(indices: Set[int], source: str, size: int, k: int) -> None:
            if not n_min <= len(indices) <= n_max:
                return
            key = frozenset(indices)
            c = unique.get(key)
            if c is None:
                c = RankingCandidate(set(indices), source=source, target_size=size)
                unique[key] = c
            if k not in c.init_indices:
                c.init_indices.append(k)

        for k, G0 in enumerate(starts):
            ev = evaluator.evaluate(mask_from_indices(G0, n))
            ranking_X = order_by_score(ev.proba_X, rng)
            ranking_Y = order_by_score(ev.proba_Y, rng)
            combined = (aggregator.get_combined_ranking(ev.proba_X, ev.proba_Y, rng)
                        if aggregator.method != RankAggregationMethod.UNION else None)
            for size in sizes:
                G_X = aggregator.get_candidates_at_size(ranking_X, size)
                G_Y = aggregator.get_candidates_at_size(ranking_Y, size)
                add(G_X | G_Y, "union", size, k)
                add(G_X, "X", size, k)
                add(G_Y, "Y", size, k)
                if combined is not None:
                    add(aggregator.get_candidates_at_size(combined, size), aggregator.method.value, size, k)

        for c in unique.values():
            ev = evaluator.evaluate(mask_from_indices(c.indices, n))
            c.q_X, c.q_Y, c.quality = float(ev.q_X), float(ev.q_Y), float(ev.quality)
        candidates = list(unique.values())
        self.candidates = candidates

        if selection == SelectionMethod.BEST:
            selected = self._best_per_start(candidates, len(starts))
        elif selection == SelectionMethod.PARETO:
            selected = compute_pareto_front(candidates, cfg.pareto_epsilon)
        else:
            selected = sorted(candidates, key=lambda c: c.quality, reverse=True)[: cfg.top_k]

        rescorer = evaluator.fresh_copy(derive_seed(rng)) if cfg.final_rescoring else None
        found = [(c.indices, {"source": c.source, "target_size": c.target_size}) for c in selected]
        return evaluator, rescorer, found

    @staticmethod
    def _best_per_start(candidates: List[RankingCandidate], n_starts: int) -> List[RankingCandidate]:
        best: Dict[int, RankingCandidate] = {}
        for c in candidates:
            for k in c.init_indices:
                if k not in best or c.quality > best[k].quality:
                    best[k] = c
        seen: Set[int] = set()
        out = []
        for k in range(n_starts):
            c = best.get(k)
            if c is not None and id(c) not in seen:
                seen.add(id(c))
                out.append(c)
        return out


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_ranking_method(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
                       initial_subgroups: Optional[Sequence] = None, **config_kwargs) -> SetOfSubgroups:
    """Run the ranking method on a DataFrame or a data file."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    return RankingMethod(RankingMethodConfig(**config_kwargs)).fit(dataset, initial_subgroups)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from .mrsdc2_swapping import _demo_data

    df, planted = _demo_data()
    planted_idx = set(np.flatnonzero(planted))
    for agg, sel in (("union", "best"), ("rrf", "pareto")):
        print(f"\n--- rank_aggregation={agg}, selection={sel} ---")
        result = run_ranking_method(df, ["x1", "x2"], ["y1", "y2"], classifier_X="decision_tree",
                                    classifier_Y="decision_tree", n_folds=3, n_initial_subgroups=4,
                                    initialization="clustering", rank_aggregation=agg, selection_method=sel,
                                    n_permutations=100, random_state=1, verbose=True)
        best = max(result, key=lambda s: s.quality)
        jac = len(best.covered_instances & planted_idx) / len(best.covered_instances | planted_idx)
        print(f"planted subgroup has {len(planted_idx)} instances; best found: {best}")
        print(f"Jaccard with planted = {jac:.2f} -> {'recovered' if jac >= 0.5 else 'not recovered'}")
