"""
MRSD-C2 (Multiple Responses Subgroup Discovery, C2 = classification in both spaces, X and Y):
the semi-supervised learning method.

A starting subgroup G is treated as a noisy labelling.  Membership
classifiers in X space and in Y space are cross-fitted
(`classifiers.DualSpaceEvaluator`): every instance e receives out-of-fold
probabilities p_X(e), p_Y(e) from models that did not see e.  A combination
rule r (`CombinationRule`: and / or / average / product / min / max /
weighted, with a threshold) turns them into a binary prediction, and the
expanded subgroup is

    G' = { e : r(p_X(e), p_Y(e)) = 1 }.

If G' is empty or violates [n_min, n_max], the top-|G| instances by combined
probability are taken instead, so that an expansion step always proposes a
set of valid size.  Note that with out-of-fold probabilities and a minority
membership class, `and` / `product` at threshold 0.5 almost only remove
instances (non-members rarely reach p > 0.5 in both spaces), which drives
self-training towards tiny sets whose AUC is high by chance; `average`
(the default) is much more stable.

Phase 1 (per starting subgroup): self-training repeats the expansion and
accepts G' only if the out-of-fold quality q(G') = aggregate(q_X, q_Y)
improves by more than `min_improvement`; it stops otherwise, when G' = G, or
after `max_self_training_iterations`.  With `use_self_training=False` a
single expansion is applied.

Phase 2 (pairwise, optional): for every unordered pair (G_i, G_j) the
models are trained on G_i ∪ G_j; the top-m instances by combined
out-of-fold probability (m = |G_i ∪ G_j|, clamped to the size limits) form
a candidate that replaces G_i and/or G_j when it is better by more than
`min_improvement`.  Results are deduplicated.

Every returned subgroup is re-scored with fresh folds (`final_rescoring`;
`quality` honest, `quality_search` optimistic), described by a shallow tree
in X space, pruned for redundancy and validated according to
`inference_statistic` ('y_shift': fixed index set, Y permuted, conditional on
the candidate set and anti-conservative; 'search_permutation': the whole
procedure re-run on permuted Y, honest but costly; see mrsdc2_swapping).

Earlier versions trained and predicted on the same rows; with tree models
every member was then predicted as a member and the expansion never changed
a subgroup.  All probabilities are now out-of-fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd

from .classifiers import DualSpaceEvaluator
from .common import derive_seed, indices_from_mask, make_rng, mask_from_indices
from .data_loader import DataLoader, Dataset
from .mrsdc2_swapping import (build_c2_subgroup, combined_probability, finalize_c2_subgroups,
                             make_c2_evaluator, order_by_score, prepare_initial_subgroups,
                             resolve_size_limits, search_qualities)
from .core import SetOfSubgroups, Subgroup

logger = logging.getLogger("mrsd")


# =============================================================================
# COMBINATION RULES
# =============================================================================

class CombinationRule(Enum):
    """Rule r combining the X and Y membership probabilities."""
    AND = "and"            # both above threshold
    OR = "or"              # at least one above threshold
    AVERAGE = "average"    # mean above threshold
    PRODUCT = "product"    # product above threshold
    MIN = "min"            # minimum above threshold (same decision as AND)
    MAX = "max"            # maximum above threshold (same decision as OR)
    WEIGHTED = "weighted"  # weight_X * p_X + (1 - weight_X) * p_Y above threshold


class ProbabilityCombiner:
    """f(e) = r(p_X(e), p_Y(e)) and the underlying combined score."""

    def __init__(self, rule: Union[CombinationRule, str] = CombinationRule.AND,
                 threshold: float = 0.5, weight_X: float = 0.5):
        self.rule = CombinationRule(rule) if isinstance(rule, str) else rule
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not 0.0 <= weight_X <= 1.0:
            raise ValueError("weight_X must be in [0, 1]")
        self.threshold = threshold
        self.weight_X = weight_X

    def combine(self, prob_X: np.ndarray, prob_Y: np.ndarray) -> np.ndarray:
        """Binary prediction (1 = member)."""
        t = self.threshold
        if self.rule == CombinationRule.AND:
            return ((prob_X > t) & (prob_Y > t)).astype(int)
        if self.rule == CombinationRule.OR:
            return ((prob_X > t) | (prob_Y > t)).astype(int)
        return (self.get_combined_probability(prob_X, prob_Y) > t).astype(int)

    def get_combined_probability(self, prob_X: np.ndarray, prob_Y: np.ndarray) -> np.ndarray:
        """Combined score before thresholding (used for ranking instances)."""
        if self.rule == CombinationRule.AVERAGE:
            return 0.5 * (prob_X + prob_Y)
        if self.rule == CombinationRule.PRODUCT:
            return prob_X * prob_Y
        if self.rule in (CombinationRule.MIN, CombinationRule.AND):
            return np.minimum(prob_X, prob_Y)
        if self.rule in (CombinationRule.MAX, CombinationRule.OR):
            return np.maximum(prob_X, prob_Y)
        if self.rule == CombinationRule.WEIGHTED:
            return self.weight_X * prob_X + (1 - self.weight_X) * prob_Y
        raise ValueError(self.rule)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SemiSupervisedConfig:
    """Configuration of the semi-supervised method."""
    # evaluation
    classifier_X: str = "random_forest"
    classifier_Y: str = "random_forest"
    n_folds: int = 5
    aggregation: str = "min"
    # combination rule ('and' at threshold 0.5 can only shrink a minority-class
    # set when probabilities are out-of-fold; 'average' is the stable default)
    combination_rule: str = "average"
    threshold: float = 0.5
    weight_X: float = 0.5
    # size limits
    n_min: int = 5
    n_max: Optional[int] = None          # default n // 2
    # starting subgroups
    initialization: str = "clustering"   # clustering | random
    n_initial_subgroups: int = 10
    # self-training and pairwise phase
    use_self_training: bool = True
    max_self_training_iterations: int = 10
    min_improvement: float = 0.005
    use_pairwise: bool = True
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

class SemiSupervisedMethod:
    """Semi-supervised method of MRSD-C2 (see the module docstring)."""

    ALGORITHM = "MRSD-C2-semisupervised"

    def __init__(self, config: Optional[SemiSupervisedConfig] = None):
        self.config = config or SemiSupervisedConfig()
        self._all_subgroups: List[Subgroup] = []
        self.n_evaluations = 0

    # ------------------------------------------------------------------
    def fit(self, dataset: Dataset, initial_subgroups: Optional[Sequence] = None) -> SetOfSubgroups:
        """
        Expand every starting subgroup (phase 1), combine pairs (phase 2),
        then re-score, describe, prune and validate.
        """
        cfg = self.config
        rng = make_rng(cfg.random_state)
        X = dataset.X_space.numeric
        Y = dataset.Y_space.numeric
        n = dataset.n_instances
        n_min, n_max = resolve_size_limits(n, cfg.n_min, cfg.n_max)
        combiner = ProbabilityCombiner(cfg.combination_rule, cfg.threshold, cfg.weight_X)
        starts_seed = derive_seed(rng)
        starts = prepare_initial_subgroups(cfg, X, Y, n, n_min, n_max, make_rng(starts_seed), initial_subgroups)
        if cfg.verbose:
            print(f"{self.ALGORITHM}: n={n}, sizes in [{n_min}, {n_max}], rule={combiner.rule.value}, "
                  f"{len(starts)} starting subgroups")

        seed = derive_seed(rng)

        def permuted_starts(Yp):
            # generated starts are re-derived from the permuted data (user-supplied stay fixed)
            if initial_subgroups is not None:
                return starts
            return prepare_initial_subgroups(cfg, X, Yp, n, n_min, n_max, make_rng(starts_seed), None)
        evaluator, rescorer, found = self._search(X, Y, n, n_min, n_max, seed, starts, combiner)
        if cfg.verbose:
            for G, extra in found:
                print(f"  result: n={len(G)} q={evaluator.quality(mask_from_indices(G, n)):.3f}")

        subgroups = [build_c2_subgroup(self.ALGORITHM, G, dataset, evaluator, rescorer, extra=extra)
                     for G, extra in found]
        self.n_evaluations = evaluator.n_evaluations
        self._all_subgroups = subgroups
        result = finalize_c2_subgroups(
            subgroups, dataset, cfg, rng,
            search_fn=lambda Yp: search_qualities(self._search(X, Yp, n, n_min, n_max, seed, permuted_starts(Yp), combiner), n))
        if cfg.verbose:
            print(result.summary())
        return result

    def _search(self, X: np.ndarray, Y: np.ndarray, n: int, n_min: int, n_max: int, seed: int,
                starts: Sequence[Set[int]], combiner: ProbabilityCombiner):
        """Phase 1 + phase 2 + deduplication for given data and seed (re-run on
        permuted Y by `search_permutation_test`).
        Returns (evaluator, rescorer, [(indices, extra_measures)])."""
        cfg = self.config
        rng = make_rng(seed)
        evaluator = make_c2_evaluator(cfg, X, Y, rng, fast=True)
        processed: List[Set[int]] = []
        for G0 in starts:
            if cfg.use_self_training:
                processed.append(self._self_training(G0, evaluator, n, n_min, n_max, combiner, rng))
            else:
                processed.append(self._expand(G0, evaluator, n, n_min, n_max, combiner, rng))
        if cfg.use_pairwise and len(processed) > 1:
            processed = self._pairwise(processed, evaluator, n, n_min, n_max, combiner, rng)
        unique: Dict[frozenset, Set[int]] = {}
        for G in processed:
            unique.setdefault(frozenset(G), G)
        rescorer = evaluator.fresh_copy(derive_seed(rng)) if cfg.final_rescoring else None
        found = [(G, {"combination_rule": combiner.rule.value}) for G in unique.values()]
        return evaluator, rescorer, found

    # ------------------------------------------------------------------
    @staticmethod
    def _top_m(score: np.ndarray, m: int, rng: np.random.Generator) -> Set[int]:
        return set(int(e) for e in order_by_score(score, rng)[:m])

    def _expand(self, G: Set[int], evaluator: DualSpaceEvaluator, n: int, n_min: int, n_max: int,
                combiner: ProbabilityCombiner, rng: np.random.Generator) -> Set[int]:
        """G' = {e : r(p_X(e), p_Y(e)) = 1} from out-of-fold probabilities, or the
        top-|G| instances by combined probability when that set is invalid."""
        ev = evaluator.evaluate(mask_from_indices(G, n))
        G_new = indices_from_mask(combiner.combine(ev.proba_X, ev.proba_Y) == 1)
        if not n_min <= len(G_new) <= n_max:
            m = min(max(len(G), n_min), n_max)
            G_new = self._top_m(combiner.get_combined_probability(ev.proba_X, ev.proba_Y), m, rng)
        return G_new

    def _self_training(self, G0: Set[int], evaluator: DualSpaceEvaluator, n: int, n_min: int, n_max: int,
                       combiner: ProbabilityCombiner, rng: np.random.Generator) -> Set[int]:
        """Repeat the expansion while the out-of-fold quality improves."""
        G = set(G0)
        q = evaluator.quality(mask_from_indices(G, n))
        for _ in range(self.config.max_self_training_iterations):
            G_new = self._expand(G, evaluator, n, n_min, n_max, combiner, rng)
            if G_new == G:
                break
            q_new = evaluator.quality(mask_from_indices(G_new, n))
            if q_new - q > self.config.min_improvement:
                G, q = G_new, q_new
            else:
                break
        return G

    def _pairwise(self, subgroups: List[Set[int]], evaluator: DualSpaceEvaluator, n: int, n_min: int,
                  n_max: int, combiner: ProbabilityCombiner, rng: np.random.Generator) -> List[Set[int]]:
        """Phase 2: candidates from models trained on unions of pairs."""
        result = [set(G) for G in subgroups]
        quality = [evaluator.quality(mask_from_indices(G, n)) for G in subgroups]
        for i, j in combinations(range(len(subgroups)), 2):
            union = subgroups[i] | subgroups[j]
            if len(union) > n - 2:
                continue
            ev = evaluator.evaluate(mask_from_indices(union, n))
            m = min(max(len(union), n_min), n_max)
            cand = self._top_m(combined_probability(ev.proba_X, ev.proba_Y), m, rng)
            q_c = evaluator.quality(mask_from_indices(cand, n))
            for idx in (i, j):
                if q_c - quality[idx] > self.config.min_improvement:
                    result[idx] = set(cand)
                    quality[idx] = q_c
        return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_semisupervised_method(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
                              initial_subgroups: Optional[Sequence] = None, **config_kwargs) -> SetOfSubgroups:
    """Run the semi-supervised method on a DataFrame or a data file."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    return SemiSupervisedMethod(SemiSupervisedConfig(**config_kwargs)).fit(dataset, initial_subgroups)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from .mrsdc2_swapping import _demo_data

    df, planted = _demo_data()
    planted_idx = set(np.flatnonzero(planted))
    rng = np.random.default_rng(1)
    # start from a noisy version of the planted subgroup (60 % of it plus 20 random instances)
    noisy = set(rng.choice(sorted(planted_idx), int(0.6 * len(planted_idx)), replace=False)) | \
        set(rng.choice(len(df), 20, replace=False))
    result = run_semisupervised_method(df, ["x1", "x2"], ["y1", "y2"], initial_subgroups=[noisy],
                                       classifier_X="decision_tree", classifier_Y="decision_tree",
                                       n_folds=3, n_permutations=100, random_state=1, verbose=True)
    best = max(result, key=lambda s: s.quality)
    jac = len(best.covered_instances & planted_idx) / len(best.covered_instances | planted_idx)
    print(f"planted subgroup has {len(planted_idx)} instances; best found: {best}")
    print(f"Jaccard with planted = {jac:.2f} -> {'recovered' if jac >= 0.5 else 'not recovered'}")
