"""
MRSD-C2 (Multiple Responses Subgroup Discovery, C2 = classification in both spaces, X and Y):
the swapping method (local search by moving instances).

A subgroup G is scored by how well its membership can be predicted
out-of-fold from X and from Y (`classifiers.DualSpaceEvaluator`):
q(G) = aggregate(q_X(G), q_Y(G)) with q_* the out-of-fold AUC.  Folds are a
deterministic function of the seed and of the membership vector, so the
scores of neighbouring sets are paired and small differences are meaningful.

Search (per starting subgroup):
  1. evaluate G; the evaluation also returns the out-of-fold membership
     probabilities p_X(e), p_Y(e) of every instance;
  2. propose a batch of `swap_batch_size` (removal, addition) pairs: half of
     them guided (members with the lowest combined probability are removal
     candidates, non-members with the highest are addition candidates), the
     rest random (`random_fraction`);
  3. for every pair evaluate G \\ {e_i}, G ∪ {e_j} and (G \\ {e_i}) ∪ {e_j}
     (sizes kept inside [n_min, n_max]) and take the best candidate
     (best-improvement);
  4. accept it only if it improves q by more than `min_improvement`;
     otherwise count a non-improving iteration.
  5. stop after `max_iterations`, after `max_no_improvement` consecutive
     non-improving iterations or when `time_limit` seconds have elapsed.

The search is strictly improving, so no tabu list is needed.  Every returned
subgroup is re-scored with fresh folds (`final_rescoring`) so that
`quality_measures['quality']` is not the optimistic value chased by the search
(stored separately as `quality_search`), described by a shallow tree in X
space (`rule`, `description_precision`, `description_recall`), pruned for
redundancy and validated (`inference_statistic`).

Inference for the MRSD-C2 family
--------------------------------
The members of an MRSD-C2 subgroup are chosen freely using both X and Y, so
on independent data the search still finds index sets that are separable in
both spaces within the sample (an X-blob intersected with a Y-blob).  A
permutation test that keeps the found index set fixed and permutes Y
(`'y_shift'`, the package default via `inference.validate_subgroups`) is
therefore conditional on the candidate set and anti-conservative: it
rejects on pure noise.  The honest alternative is swap randomization of the
whole procedure (`'search_permutation'`): the search is re-run on every
permutation of Y and the quality of a found subgroup is compared with the
best quality the search achieves when X and Y are independent.  It costs
one full search per permutation, so it is opt-in (use cheap classifiers and
few starts).

This module also hosts the helpers shared by the whole MRSD-C2 family
(`build_c2_subgroup`, `finalize_c2_subgroups`, `search_permutation_test`,
`make_c2_evaluator`, `resolve_size_limits`, `prepare_initial_subgroups`,
`combined_probability`, `order_by_score`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from .classifiers import (AggregationMethod, ClassificationModel, DualSpaceEvaluator,  # noqa: F401 (re-exported)
                         initial_subgroups, make_classifier)
from .common import derive_seed, indices_from_mask, make_rng, mask_from_indices
from .data_loader import DataLoader, Dataset
from .inference import hotelling_t2, validate_subgroups
from .rules import describe_mask_with_tree
from .core import QualityMeasures, SetOfSubgroups, Subgroup

logger = logging.getLogger("mrsd")

SELECTION_NOTE = "candidates selected using X and Y; p-values conditional on the candidate set"


class InitializationMethod(Enum):
    """How the starting subgroups are generated when none are passed to fit()."""
    RANDOM = "random"
    CLUSTERING = "clustering"


# =============================================================================
# HELPERS SHARED BY THE MRSD-C2 FAMILY
# =============================================================================

def resolve_size_limits(n: int, n_min: int, n_max: Optional[int]) -> Tuple[int, int]:
    """Clamp [n_min, n_max] so that both the subgroup and its complement have
    at least two instances; default n_max is n // 2."""
    if n < 4:
        raise ValueError(f"MRSD-C2 needs at least 4 instances, got {n}")
    hi = n // 2 if n_max is None else int(n_max)
    hi = max(2, min(hi, n - 2))
    lo = max(2, int(n_min))
    if lo > hi:
        logger.warning("n_min=%d exceeds n_max=%d for n=%d; clamping n_min to %d", n_min, hi, n, hi)
        lo = hi
    return lo, hi


def make_c2_evaluator(config, X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
                      fast: bool = True) -> DualSpaceEvaluator:
    """Out-of-fold dual-space evaluator configured from a C2 config."""
    return DualSpaceEvaluator(X, Y, config.classifier_X, config.classifier_Y,
                              n_folds=config.n_folds, aggregation=config.aggregation,
                              random_state=derive_seed(rng), fast=fast)


def prepare_initial_subgroups(config, X: np.ndarray, Y: np.ndarray, n: int, n_min: int, n_max: int,
                              rng: np.random.Generator,
                              initial: Optional[Sequence] = None) -> List[Set[int]]:
    """User-supplied starting sets (validated) or `classifiers.initial_subgroups`."""
    if initial is None:
        method = InitializationMethod(config.initialization).value
        return initial_subgroups(n, config.n_initial_subgroups, n_min, n_max, rng, X, Y, method)
    out: List[Set[int]] = []
    for i, G in enumerate(initial):
        G = set(int(e) for e in G)
        if any(e < 0 or e >= n for e in G):
            raise ValueError(f"initial subgroup {i} contains indices outside [0, {n})")
        if len(G) < 2 or len(G) > n - 2:
            logger.warning("initial subgroup %d has size %d; skipped", i, len(G))
            continue
        out.append(G)
    return out


def combined_probability(proba_X: np.ndarray, proba_Y: np.ndarray) -> np.ndarray:
    """Per-instance membership evidence from both spaces (mean of the OOF probabilities)."""
    return 0.5 * (np.asarray(proba_X, dtype=float) + np.asarray(proba_Y, dtype=float))


def order_by_score(score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices sorted by decreasing score; ties broken at random (reproducibly)."""
    score = np.asarray(score, dtype=float)
    return np.lexsort((rng.random(len(score)), -score))


def build_c2_subgroup(algorithm: str, indices, dataset: Dataset, evaluator: DualSpaceEvaluator,
                      rescorer: Optional[DualSpaceEvaluator] = None,
                      extra: Optional[Dict[str, object]] = None) -> Subgroup:
    """
    Subgroup object for an index set found by an MRSD-C2 method.

    quality_search / q_X_search / q_Y_search: scores of the evaluator used
    during the search (optimistic).  quality / q_X / q_Y: scores of `rescorer`
    (fresh folds) when given, else the search scores.  The rule is a post-hoc
    tree description in X space.
    """
    n = dataset.n_instances
    indices = set(int(e) for e in indices)
    mask = mask_from_indices(indices, n)
    ev = evaluator.evaluate(mask)
    honest = rescorer.evaluate(mask) if rescorer is not None else ev
    desc = describe_mask_with_tree(dataset.X_space, mask)
    qm: Dict[str, object] = {
        "quality": float(honest.quality),
        "q_X": float(honest.q_X),
        "q_Y": float(honest.q_Y),
        "quality_search": float(ev.quality),
        "q_X_search": float(ev.q_X),
        "q_Y_search": float(ev.q_Y),
        "description_jaccard": float(desc.jaccard),
    }
    if extra:
        qm.update(extra)
    sg = Subgroup(
        description=f"{algorithm}: {desc.rule} (n={len(indices)})",
        covered_instances=indices,
        quality_measures=qm,
        algorithm=algorithm,
        rule=desc.rule,
        description_precision=float(desc.precision),
        description_recall=float(desc.recall),
    )
    sg.fill_target_statistics(dataset.Y)
    return sg


def search_qualities(search_output, n: int) -> List[float]:
    """Qualities of the subgroups returned by a `_search` call (honest ones when
    a rescorer was created)."""
    evaluator, rescorer, found = search_output
    scorer = rescorer if rescorer is not None else evaluator
    return [float(scorer.quality(mask_from_indices(G, n))) for G, _ in found]


def search_permutation_test(result: SetOfSubgroups, search_fn: Callable[[np.ndarray], Sequence[float]],
                            dataset: Dataset, n_permutations: int, rng: np.random.Generator) -> np.ndarray:
    """
    Swap randomization for the MRSD-C2 family: the WHOLE search is re-run on
    `n_permutations` row permutations of Y and the quality of every returned
    subgroup is compared with the best quality the same search finds when X
    and Y are independent (max-T across the found set, so the p-value is
    family-wise).  `search_fn(Y_perm)` returns the qualities found on the
    permuted data.  p_value == p_value_corrected (there is no per-subgroup
    counterpart under the null).  Exact but costly: one full search per
    permutation.
    """
    n = dataset.n_instances
    Y = dataset.Y_space.numeric
    sgs = list(result)
    observed = np.array([QualityMeasures.quality(sg) for sg in sgs])
    null_max = np.full(n_permutations, -np.inf)
    for b in range(n_permutations):
        qs = search_fn(Y[rng.permutation(n)])
        if len(qs):
            null_max[b] = max(qs)
    p_adj = ((null_max[:, None] >= observed[None, :]).sum(axis=0) + 1) / (n_permutations + 1)
    for sg, p in zip(sgs, p_adj):
        sg.p_value = float(p)
        sg.p_value_corrected = float(p)
        sg.correction_method = "search_permutation_maxt"
        sg.quality_measures["test_statistic"] = float(QualityMeasures.quality(sg))
        sg.quality_measures["p_bh"] = float(p)
        sg.quality_measures["inference"] = (f"search_permutation; permuted=Y; B={n_permutations}; "
                                            "statistic=quality of the whole search re-run on permuted Y")
        t2, _, pa = hotelling_t2(Y, mask_from_indices(sg.covered_instances, n))
        sg.quality_measures["t2"] = float(t2)
        sg.quality_measures["p_asymptotic"] = float(pa)
    return p_adj


def finalize_c2_subgroups(subgroups: List[Subgroup], dataset: Dataset, config, rng: np.random.Generator,
                          search_fn: Optional[Callable[[np.ndarray], Sequence[float]]] = None) -> SetOfSubgroups:
    """
    Prune (dedup + Jaccard) and validate.

    config.inference_statistic:
      'y_shift'            Hotelling T² of the fixed index set with Y permuted
                           (`inference.validate_subgroups`); cheap, but the
                           candidates were selected using Y, so the p-values
                           are conditional on the candidate set and
                           anti-conservative (see module docstring);
      'search_permutation' the whole search re-run on permuted Y
                           (`search_permutation_test`); honest, costly;
      'none'               no test.
    """
    result = SetOfSubgroups(subgroups).prune(QualityMeasures.quality, s_min=config.redundancy_threshold)
    statistic = getattr(config, "inference_statistic", "search_permutation")
    if statistic not in ("y_shift", "search_permutation", "none"):
        raise ValueError(f"inference_statistic must be 'y_shift', 'search_permutation' or 'none', got {statistic!r}")
    if config.n_permutations > 0 and len(result) > 0 and statistic != "none":
        if statistic == "y_shift":
            validate_subgroups(result, Y_space=dataset.Y_space, X_space=dataset.X_space,
                               statistic="y_shift", n_permutations=config.n_permutations,
                               random_state=derive_seed(rng), selection_note=SELECTION_NOTE,
                               verbose=False)
        else:
            if search_fn is None:
                raise ValueError("search_permutation needs a search_fn")
            search_permutation_test(result, search_fn, dataset, config.n_permutations, rng)
    return result


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SwappingMethodConfig:
    """Configuration of the swapping method."""
    # evaluation
    classifier_X: str = "random_forest"
    classifier_Y: str = "random_forest"
    n_folds: int = 5
    aggregation: str = "min"
    # size limits
    n_min: int = 5
    n_max: Optional[int] = None          # default n // 2
    # starting subgroups
    initialization: str = "clustering"   # clustering | random
    n_initial_subgroups: int = 10
    # local search
    swap_batch_size: int = 20
    random_fraction: float = 0.5
    min_improvement: float = 0.005
    max_iterations: int = 200
    max_no_improvement: int = 10
    time_limit: Optional[float] = None   # seconds per starting subgroup
    # post-processing
    final_rescoring: bool = True
    redundancy_threshold: float = 0.7
    n_permutations: int = 50   # search permutations are full reruns of the search
    inference_statistic: str = "search_permutation"   # search_permutation (honest, slower) | y_shift (conditional) | none
    # other
    random_state: Optional[int] = None
    verbose: bool = False


# =============================================================================
# ALGORITHM
# =============================================================================

class SwappingMethod:
    """Swapping method of MRSD-C2 (see the module docstring)."""

    ALGORITHM = "MRSD-C2-swapping"

    def __init__(self, config: Optional[SwappingMethodConfig] = None):
        self.config = config or SwappingMethodConfig()
        self._all_subgroups: List[Subgroup] = []
        self._iteration_history: List[List[Tuple[int, int, float]]] = []
        self.n_evaluations = 0

    # ------------------------------------------------------------------
    def fit(self, dataset: Dataset, initial_subgroups: Optional[Sequence] = None) -> SetOfSubgroups:
        """
        Optimize every starting subgroup by local search.

        Args:
            dataset: Dataset with X and Y defined
            initial_subgroups: optional starting index sets; generated according
                to `config.initialization` when None
        """
        cfg = self.config
        rng = make_rng(cfg.random_state)
        X = dataset.X_space.numeric
        Y = dataset.Y_space.numeric
        n = dataset.n_instances
        n_min, n_max = resolve_size_limits(n, cfg.n_min, cfg.n_max)
        if not 0.0 <= cfg.random_fraction <= 1.0:
            raise ValueError("random_fraction must be in [0, 1]")
        if cfg.swap_batch_size < 1:
            raise ValueError("swap_batch_size must be >= 1")

        starts_seed = derive_seed(rng)
        starts = prepare_initial_subgroups(cfg, X, Y, n, n_min, n_max, make_rng(starts_seed), initial_subgroups)
        if cfg.verbose:
            print(f"{self.ALGORITHM}: n={n}, sizes in [{n_min}, {n_max}], {len(starts)} starting subgroups")

        seed = derive_seed(rng)

        def permuted_starts(Yp):
            # Under the null the starting sets must be derived from the permuted
            # data too (generated starts); user-supplied starts stay fixed.
            if initial_subgroups is not None:
                return starts
            return prepare_initial_subgroups(cfg, X, Yp, n, n_min, n_max, make_rng(starts_seed), None)
        evaluator, rescorer, found = self._search(X, Y, n, n_min, n_max, seed, starts)
        self._iteration_history = [hist for _, hist in found]
        if cfg.verbose:
            for k, (G, hist) in enumerate(found):
                print(f"  start {k + 1}/{len(starts)}: n={len(starts[k])} q={hist[0][2]:.3f} -> "
                      f"n={len(G)} q={hist[-1][2]:.3f} after {hist[-1][0]} iterations")

        subgroups = [
            build_c2_subgroup(self.ALGORITHM, G, dataset, evaluator, rescorer,
                              extra={"n_iterations": hist[-1][0], "start_index": k})
            for k, (G, hist) in enumerate(found)
        ]
        self.n_evaluations = evaluator.n_evaluations
        self._all_subgroups = subgroups
        result = finalize_c2_subgroups(
            subgroups, dataset, cfg, rng,
            search_fn=lambda Yp: search_qualities(self._search(X, Yp, n, n_min, n_max, seed, permuted_starts(Yp)), n))
        if cfg.verbose:
            print(result.summary())
        return result

    def _search(self, X: np.ndarray, Y: np.ndarray, n: int, n_min: int, n_max: int, seed: int,
                starts: Sequence[Set[int]]):
        """The complete search for given data and seed (re-run on permuted Y by
        `search_permutation_test`).  Returns (evaluator, rescorer, [(indices, history)])."""
        cfg = self.config
        rng = make_rng(seed)
        evaluator = make_c2_evaluator(cfg, X, Y, rng, fast=True)
        found = [self._optimize(G0, evaluator, n, n_min, n_max, rng) for G0 in starts]
        rescorer = evaluator.fresh_copy(derive_seed(rng)) if cfg.final_rescoring else None
        return evaluator, rescorer, found

    # ------------------------------------------------------------------
    def _propose_pairs(self, mask: np.ndarray, score: np.ndarray, rng: np.random.Generator) -> List[Tuple[int, int]]:
        """(removal candidate, addition candidate) pairs: guided by the OOF
        membership evidence, mixed with random pairs."""
        cfg = self.config
        members = np.flatnonzero(mask)
        others = np.flatnonzero(~mask)
        k = cfg.swap_batch_size
        n_random = int(round(k * cfg.random_fraction))
        n_guided = k - n_random
        pairs: List[Tuple[int, int]] = []
        if n_guided > 0:
            rem = members[order_by_score(-score[members], rng)[:n_guided]]   # lowest evidence inside
            add = others[order_by_score(score[others], rng)[:n_guided]]      # highest evidence outside
            m = min(len(rem), len(add))
            pairs.extend((int(rem[i]), int(add[i])) for i in range(m))
        n_random = k - len(pairs)
        if n_random > 0:
            ri = rng.choice(members, size=n_random, replace=True)
            rj = rng.choice(others, size=n_random, replace=True)
            pairs.extend((int(a), int(b)) for a, b in zip(ri, rj))
        return pairs

    @staticmethod
    def _neighbours(mask: np.ndarray, pairs: Sequence[Tuple[int, int]], n_min: int, n_max: int) -> List[np.ndarray]:
        """Distinct masks reachable by remove / add / swap for the given pairs."""
        size = int(mask.sum())
        out: Dict[bytes, np.ndarray] = {}

        def add_candidate(m: np.ndarray) -> None:
            key = np.packbits(m).tobytes()
            if key not in out:
                out[key] = m

        for e_i, e_j in pairs:
            if size - 1 >= n_min:
                m = mask.copy(); m[e_i] = False
                add_candidate(m)
            if size + 1 <= n_max:
                m = mask.copy(); m[e_j] = True
                add_candidate(m)
            m = mask.copy(); m[e_i] = False; m[e_j] = True
            add_candidate(m)
        return list(out.values())

    def _optimize(self, G0: Set[int], evaluator: DualSpaceEvaluator, n: int, n_min: int, n_max: int,
                  rng: np.random.Generator) -> Tuple[Set[int], List[Tuple[int, int, float]]]:
        """Best-improvement local search from G0. Returns (indices, history)."""
        cfg = self.config
        mask = mask_from_indices(G0, n)
        ev = evaluator.evaluate(mask)
        best_q = ev.quality
        history: List[Tuple[int, int, float]] = [(0, int(mask.sum()), float(best_q))]
        iteration = 0
        no_improvement = 0
        start = time.time()
        while iteration < cfg.max_iterations and no_improvement < cfg.max_no_improvement:
            if cfg.time_limit is not None and time.time() - start >= cfg.time_limit:
                break
            iteration += 1
            score = combined_probability(ev.proba_X, ev.proba_Y)
            pairs = self._propose_pairs(mask, score, rng)
            best_cand = None
            best_cand_q = best_q
            for cand in self._neighbours(mask, pairs, n_min, n_max):
                q = evaluator.quality(cand)
                if q > best_cand_q:
                    best_cand, best_cand_q = cand, q
            if best_cand is not None and best_cand_q - best_q > cfg.min_improvement:
                mask = best_cand
                ev = evaluator.evaluate(mask)
                best_q = ev.quality
                no_improvement = 0
            else:
                no_improvement += 1
            history.append((iteration, int(mask.sum()), float(best_q)))
        return indices_from_mask(mask), history

    # ------------------------------------------------------------------
    def get_iteration_history(self, start: Optional[int] = None):
        """History of the local search: one list of (iteration, size, quality)
        tuples per starting subgroup, or the list of start `start`."""
        if start is None:
            return [list(h) for h in self._iteration_history]
        return list(self._iteration_history[start])


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_swapping_method(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
                        initial_subgroups: Optional[Sequence] = None, **config_kwargs) -> SetOfSubgroups:
    """Run the swapping method on a DataFrame or a data file."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    return SwappingMethod(SwappingMethodConfig(**config_kwargs)).fit(dataset, initial_subgroups)


# =============================================================================
# DEMO
# =============================================================================

def _demo_data(n: int = 200, seed: int = 0, effect: float = 2.0):
    """Planted subgroup: x1 > 1 shifts y1 up and y2 down."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y1, y2 = rng.normal(size=n), rng.normal(size=n)
    planted = x1 > 1.0
    y1[planted] += effect
    y2[planted] -= effect / 2
    return pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2}), planted


if __name__ == "__main__":
    df, planted = _demo_data()
    planted_idx = set(np.flatnonzero(planted))
    # random starts carry no out-of-fold signal; k-means starts on [X Y] do
    result = run_swapping_method(df, ["x1", "x2"], ["y1", "y2"], classifier_X="decision_tree",
                                 classifier_Y="decision_tree", n_folds=3, n_initial_subgroups=4,
                                 initialization="clustering", swap_batch_size=10, max_iterations=40,
                                 n_permutations=100, random_state=1, verbose=True)
    best = max(result, key=lambda s: s.quality)
    jac = len(best.covered_instances & planted_idx) / len(best.covered_instances | planted_idx)
    print(f"planted subgroup has {len(planted_idx)} instances; best found: {best}")
    print(f"Jaccard with planted = {jac:.2f} -> {'recovered' if jac >= 0.5 else 'not recovered'}")
