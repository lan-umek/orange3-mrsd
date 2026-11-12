"""
MRSD-Rules: CN2-SD style rule learner for MRSD (Multiple Responses Subgroup
Discovery).

What is implemented
-------------------
* Rules are conjunctions of conditions on the X attributes (``rules.Rule`` in
  original units: labels for categorical attributes, raw values for numeric
  ones).  Candidate conditions come from ``rules.ConditionGenerator`` (quantile
  cut points for numeric attributes, ``=`` for every level of a categorical
  attribute; ``!=`` optionally) and the rule space is explored with
  ``rules.BeamSearch`` (candidates deduplicated by covered set).
* The responses enter through ``Y_space.numeric``: numeric Y columns are
  standardized, categorical Y columns are one-hot encoded (all levels), so a
  categorical response contributes one indicator column per level and "shift"
  means a change of the level proportions.
* Quality of a rule G with complement C = D \\ G is a **signed multi-response
  WRAcc**: ``size(G) * (difference between G and its complement)`` with
  ``size(G) = (cov(G) * cov(C)) ** a``, ``cov(G) = n_G / N`` (sum of covering
  weights in G over the total weight when covering is active) and
  ``a = coverage_exponent``.  With ``a = 1`` the product
  ``cov(G) * cov(C) * (ȳ_G - ȳ_C)`` is algebraically the classical WRAcc
  ``cov(G) * (ȳ_G - ȳ_D)``, which is indifferent between a rule and any
  superset that merely dilutes it; with the default ``a = 0.5`` the measure is
  the two-sample test statistic on a per-instance scale
  (``sqrt(cov(G) cov(C)) * D_M = sqrt(T² / N)``), i.e. the shift normalized by
  its sampling error, which prefers the tight rule (DEV_NOTES item 8).  The
  empty rule and the whole data have quality 0, complements are not rewarded
  (the complement's own mean and weight are used; rules larger than ``n_max``
  are never candidates), and the size factor keeps tiny rules from winning.
  Three members of the family:

  - ``wracc_mahalanobis`` (default): ``size(G) * D_M(ȳ_G, ȳ_C)`` with the
    Mahalanobis distance under the pooled within-group covariance
    (pseudo-inverse).  Monotone in Hotelling's T² (exactly, for ``a = 0.5``);
    covariance aware (a shift along a direction with large variance counts
    less).
  - ``wracc_dispersion`` (Ženko's predictive clustering rules):
    ``size(G) * (1 - disp(G) / disp(D))`` where disp is the mean squared
    Euclidean distance to the centroid; negative when G is more dispersed than
    the population.
  - ``wracc_directed`` (classical multi-target WRAcc):
    ``size(G) * mean_j dir_j * (ȳ_Gj - ȳ_Cj) / sd_j`` for a user-specified
    direction per target (``target_direction``, default +1 for every column;
    a categorical target can be given a direction for one of its levels with
    the key ``"column=level"``).

* Weighted covering (CN2-SD, Lavrač et al. 2004): after a rule is accepted the
  weights of the covered instances are decreased, multiplicatively
  (``w <- w * gamma``) or additively (``w <- 1 / (k + 1)`` where k is the number
  of accepted rules covering the instance).  The weights enter both the
  coverage factor (``sum of weights in G / sum of all weights``) and the group
  means / covariance / dispersion (weighted statistics).  The search stops
  after ``n_rules`` rules, when the best rule's search quality drops below
  ``min_quality_fraction`` times the quality of the first rule, or when no
  rule with positive quality is left.  Candidates whose covered set is
  (nearly) that of an already accepted rule (Jaccard >= ``redundancy_threshold``)
  are skipped, so every accepted rule has a distinct covered set.
  Every returned subgroup carries the unweighted quality in
  ``quality_measures['quality']`` and the weighted search score in
  ``quality_measures['quality_search']``.

* Inference.  No p-value is derived from the quality.  Two modes:

  - ``inference_mode='rules'`` (default): ``inference.validate_subgroups``
    with the Y-shift statistic (Hotelling T² between G and C, rows of Y
    permuted) and Westfall-Young max-T over the **returned** rules.  Because
    the rules were selected by a search over many candidates, these p-values
    are conditional on the selection and are optimistic; this is recorded in
    ``quality_measures['inference']``.
  - ``inference_mode='search'``: swap randomization over the whole search.
    Y is permuted B times, the full (unweighted) beam search is rerun on every
    permutation and the best search quality is recorded; the p-value of a
    returned rule is the fraction of permutations whose best quality is at
    least the rule's quality.  This controls the family-wise error over
    everything the search looked at.  ``p_value`` and ``p_value_corrected``
    are both this value.

References
----------
Lavrač, N., Kavšek, B., Flach, P., Todorovski, L. (2004).  Subgroup discovery
with CN2-SD.  JMLR 5, 153-188.
Ženko, B. (2007).  Learning predictive clustering rules.  PhD thesis,
University of Ljubljana.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .common import FeatureSpace, derive_seed, make_rng
from .data_loader import DataLoader, Dataset
from .inference import hotelling_t2, validate_subgroups
from .rules import BeamSearch, Condition, ConditionGenerator, Rule, SearchResult
from .core import QualityMeasures, SetOfSubgroups, Subgroup

logger = logging.getLogger("mrsd")

__all__ = [
    "MRSD_Rules", "MRSDRulesConfig", "run_mrsd_rules",
    "Rule", "Condition", "ConditionGenerator", "BeamSearch",
    "RuleQualityEvaluator", "WeightedCovering",
    "QualityMeasureType", "CoveringStrategy", "InferenceMode",
    # backward-compatible aliases
    "CN2BeamSearch", "TargetMeasure",
]


# =============================================================================
# ENUMS
# =============================================================================

class QualityMeasureType(Enum):
    """Signed multi-response WRAcc variants (see module docstring)."""
    WRACC_MAHALANOBIS = "wracc_mahalanobis"
    WRACC_DISPERSION = "wracc_dispersion"
    WRACC_DIRECTED = "wracc_directed"


class CoveringStrategy(Enum):
    """Weighted covering scheme of CN2-SD."""
    NONE = "none"                      # one search; top non-redundant rules
    ADDITIVE = "additive"              # w = 1 / (k + 1)
    MULTIPLICATIVE = "multiplicative"  # w = gamma ** k


class InferenceMode(Enum):
    RULES = "rules"      # max-T over the returned rules (conditional on selection)
    SEARCH = "search"    # swap randomization of the whole beam search


# Backward-compatible aliases (the package __init__ imports these names).  The
# beam search now lives in rules.BeamSearch; ``TargetMeasure`` used to select a
# "target quality" that is now part of the quality measure itself.
CN2BeamSearch = BeamSearch
TargetMeasure = QualityMeasureType


# =============================================================================
# QUALITY MEASURES
# =============================================================================

class RuleQualityEvaluator:
    """
    Signed multi-response WRAcc of a covered set, with instance weights.

    ``Y`` is the numeric view of the responses (``Y_space.numeric``).  Call
    ``set_weights`` whenever the covering weights change; the evaluator keeps
    the weighted population statistics so that scoring one mask costs
    O(n_G * d + d^3).  ``evaluator(mask)`` returns the search score.
    """

    def __init__(self, Y: np.ndarray, measure: Union[str, QualityMeasureType] = QualityMeasureType.WRACC_MAHALANOBIS,
                 directions: Optional[np.ndarray] = None, coverage_exponent: float = 0.5):
        self.Y = np.asarray(Y, dtype=float)
        if not (0.0 < coverage_exponent <= 1.0):
            raise ValueError("coverage_exponent must be in (0, 1]")
        self.a = float(coverage_exponent)
        if self.Y.ndim != 2:
            raise ValueError("Y must be a 2-d array")
        self.n, self.d = self.Y.shape
        self.measure = QualityMeasureType(measure)
        self.sq_norm = np.einsum("ij,ij->i", self.Y, self.Y)
        if directions is None:
            directions = np.ones(self.d)
        self.directions = np.asarray(directions, dtype=float)
        if self.directions.shape != (self.d,):
            raise ValueError("directions must have one entry per numeric Y column")
        self.set_weights(np.ones(self.n))

    # ---- population statistics for the current weights ---------------------
    def set_weights(self, w: np.ndarray) -> None:
        w = np.asarray(w, dtype=float)
        if w.shape != (self.n,):
            raise ValueError("weights must have one entry per instance")
        self.w = w
        self.W = float(w.sum())
        self.wY = w[:, None] * self.Y                       # (n, d)
        self.sum_wY = self.wY.sum(axis=0)                   # (d,)
        self.ybar = self.sum_wY / self.W if self.W > 0 else np.zeros(self.d)
        Yc = self.Y - self.ybar
        self.T = (w[:, None] * Yc).T @ Yc                   # weighted total scatter (d, d)
        self.sd = np.sqrt(np.maximum(np.diag(self.T) / max(self.W, 1e-12), 0.0))
        self.sd = np.where(self.sd < 1e-12, 1.0, self.sd)
        self.w_sq = w * self.sq_norm
        self.disp_D = float(np.trace(self.T) / max(self.W, 1e-12))

    # ---- scoring -----------------------------------------------------------
    def group_statistics(self, mask: np.ndarray) -> Optional[Tuple[float, float, np.ndarray, np.ndarray]]:
        """(W_G, W_C, weighted mean of G, weighted mean of C) or None if degenerate."""
        n_G = int(mask.sum())
        if n_G < 2 or self.n - n_G < 2:
            return None
        W_G = float(self.w[mask].sum())
        W_C = self.W - W_G
        if W_G <= 1e-12 or W_C <= 1e-12:
            return None
        s_G = self.wY[mask].sum(axis=0)
        mean_G = s_G / W_G
        mean_C = (self.sum_wY - s_G) / W_C
        return W_G, W_C, mean_G, mean_C

    def size_factor(self, W_G: float, W_C: float) -> float:
        """(cov(G) * cov(C)) ** a with weighted coverages."""
        return float((W_G / self.W * W_C / self.W) ** self.a)

    def mahalanobis(self, mask: np.ndarray) -> float:
        st = self.group_statistics(mask)
        if st is None:
            return 0.0
        W_G, W_C, mean_G, mean_C = st
        dG = mean_G - self.ybar
        dC = mean_C - self.ybar
        within = self.T - W_G * np.outer(dG, dG) - W_C * np.outer(dC, dC)
        S = within / max(self.W - 2.0, 1e-12)
        diff = mean_G - mean_C
        Sinv = np.linalg.pinv(S, rcond=1e-10, hermitian=True)
        d2 = float(diff @ Sinv @ diff)
        return self.size_factor(W_G, W_C) * float(np.sqrt(max(d2, 0.0)))

    def dispersion(self, mask: np.ndarray) -> float:
        st = self.group_statistics(mask)
        if st is None or self.disp_D <= 1e-12:
            return 0.0
        W_G, W_C, mean_G, _ = st
        disp_G = float(self.w_sq[mask].sum() / W_G - mean_G @ mean_G)
        disp_G = max(disp_G, 0.0)
        return self.size_factor(W_G, W_C) * (1.0 - disp_G / self.disp_D)

    def directed(self, mask: np.ndarray) -> float:
        st = self.group_statistics(mask)
        if st is None:
            return 0.0
        W_G, W_C, mean_G, mean_C = st
        z = self.directions * (mean_G - mean_C) / self.sd
        return self.size_factor(W_G, W_C) * float(z.mean())

    def __call__(self, mask: np.ndarray) -> float:
        if self.measure == QualityMeasureType.WRACC_MAHALANOBIS:
            return self.mahalanobis(mask)
        if self.measure == QualityMeasureType.WRACC_DISPERSION:
            return self.dispersion(mask)
        return self.directed(mask)

    def coverage(self, mask: np.ndarray) -> float:
        """Weighted coverage sum(w_G) / sum(w)."""
        return float(self.w[mask].sum() / self.W) if self.W > 0 else 0.0


def direction_vector(Y_space: FeatureSpace, target_direction: Optional[Dict[str, int]]) -> np.ndarray:
    """Map ``{column: ±1}`` (or ``{"column=level": ±1}``) onto the columns of ``Y_space.numeric``."""
    dirs = np.ones(len(Y_space.numeric_names))
    if not target_direction:
        return dirs
    known = set(Y_space.columns) | set(Y_space.numeric_names)
    for key, val in target_direction.items():
        if key not in known:
            raise ValueError(f"target_direction key '{key}' is not a Y column or a 'column=level' name")
        if val not in (1, -1, 1.0, -1.0):
            raise ValueError(f"target_direction['{key}'] must be +1 or -1")
    for j, name in enumerate(Y_space.numeric_names):
        if name in target_direction:
            dirs[j] = float(target_direction[name])
        else:
            col = name.split("=", 1)[0]
            if col in target_direction:
                dirs[j] = float(target_direction[col])
    return dirs


# =============================================================================
# WEIGHTED COVERING
# =============================================================================

class WeightedCovering:
    """CN2-SD instance weights.  ``counts`` = number of accepted rules covering each instance."""

    def __init__(self, n: int, strategy: Union[str, CoveringStrategy] = CoveringStrategy.MULTIPLICATIVE,
                 gamma: float = 0.9):
        self.strategy = CoveringStrategy(strategy)
        if not (0.0 < gamma < 1.0):
            raise ValueError("covering_gamma must be in (0, 1)")
        self.gamma = gamma
        self.counts = np.zeros(n, dtype=int)

    def update(self, covered_mask: np.ndarray) -> None:
        self.counts[covered_mask] += 1

    @property
    def weights(self) -> np.ndarray:
        if self.strategy == CoveringStrategy.MULTIPLICATIVE:
            return self.gamma ** self.counts
        if self.strategy == CoveringStrategy.ADDITIVE:
            return 1.0 / (self.counts + 1.0)
        return np.ones(len(self.counts))


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MRSDRulesConfig:
    """Configuration of MRSD-Rules."""

    # beam search
    beam_width: int = 10
    max_depth: int = 3                  # maximum number of conditions per rule
    n_rules: int = 10                   # maximum number of rules
    n_bins: int = 5                     # quantile cut points for numeric attributes
    local_cutpoints: bool = False       # recompute cut points inside the current coverage
    allow_negation: bool = False        # '!=' conditions on categorical attributes

    # quality
    quality_measure: str = "wracc_mahalanobis"   # wracc_mahalanobis | wracc_dispersion | wracc_directed
    target_direction: Optional[Dict[str, int]] = None   # for wracc_directed: {Y column: +1/-1}
    coverage_exponent: float = 0.5      # size factor (cov(G) cov(C))**a; 0.5 = test-statistic scale, 1 = WRAcc

    # weighted covering
    covering_strategy: str = "multiplicative"    # none | additive | multiplicative
    covering_gamma: float = 0.9
    min_quality_fraction: float = 0.2   # stop when best quality < fraction * quality of first rule

    # size constraints
    n_min: int = 10
    n_max: Optional[int] = None         # default n // 2
    redundancy_threshold: float = 0.7   # Jaccard; covering stop + final pruning

    # inference
    n_permutations: int = 200           # 0 disables inference
    inference_mode: str = "search"      # search (honest, reruns the search under permutation) | rules (conditional, fast)
    n_search_permutations: int = 100    # B for inference_mode='search'

    random_state: Optional[int] = None
    verbose: bool = False


# =============================================================================
# ALGORITHM
# =============================================================================

@dataclass
class AcceptedRule:
    rule: Rule
    mask: np.ndarray
    quality: float          # unweighted
    quality_search: float   # with the covering weights in force when found
    depth: int
    n_evaluated: int


class MRSD_Rules:
    """
    CN2-SD style rule learner for multiple responses.

    1. weights <- 1
    2. repeat: beam search for the rule maximizing the (weighted) signed
       multi-response WRAcc; accept it; decrease the weights of the covered
       instances
    3. prune (dedup + greedy Jaccard), permutation inference, return.
    """

    def __init__(self, config: Optional[MRSDRulesConfig] = None):
        self.config = config or MRSDRulesConfig()
        self._check_config()
        self._rules: List[Rule] = []
        self._accepted: List[AcceptedRule] = []
        self.n_evaluated: int = 0

    def _check_config(self) -> None:
        c = self.config
        QualityMeasureType(c.quality_measure)
        CoveringStrategy(c.covering_strategy)
        InferenceMode(c.inference_mode)
        if c.n_min < 2:
            raise ValueError("n_min must be at least 2")
        if not (0.0 < c.redundancy_threshold <= 1.0):
            raise ValueError("redundancy_threshold must be in (0, 1]")
        if not (0.0 <= c.min_quality_fraction <= 1.0):
            raise ValueError("min_quality_fraction must be in [0, 1]")
        if not (0.0 < c.coverage_exponent <= 1.0):
            raise ValueError("coverage_exponent must be in (0, 1]")
        if c.n_permutations < 0 or c.n_search_permutations < 1:
            raise ValueError("n_permutations must be >= 0 and n_search_permutations >= 1")

    # ---- helpers -----------------------------------------------------------
    def _generator(self, fs_X: FeatureSpace) -> ConditionGenerator:
        c = self.config
        return ConditionGenerator(fs_X, n_bins=c.n_bins, min_support=c.n_min,
                                  local_cutpoints=c.local_cutpoints, allow_negation=c.allow_negation)

    def _evaluator(self, Y: np.ndarray, directions: np.ndarray) -> RuleQualityEvaluator:
        c = self.config
        return RuleQualityEvaluator(Y, c.quality_measure, directions, coverage_exponent=c.coverage_exponent)

    def _n_max(self, n: int) -> int:
        return self.config.n_max if self.config.n_max is not None else n // 2

    def _beam(self, gen: ConditionGenerator, quality_fn: Callable, n: int) -> BeamSearch:
        c = self.config
        return BeamSearch(gen, quality_fn, beam_width=c.beam_width, max_depth=c.max_depth,
                          n_min=c.n_min, n_max=self._n_max(n))

    @staticmethod
    def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
        union = int((a | b).sum())
        return float((a & b).sum() / union) if union else 0.0

    # ---- the covering loop -------------------------------------------------
    def _covering_search(self, gen: ConditionGenerator, Y: np.ndarray, directions: np.ndarray,
                         verbose: bool = False) -> List[AcceptedRule]:
        """Run the CN2-SD loop on numeric responses ``Y``; returns accepted rules in order."""
        c = self.config
        n = Y.shape[0]
        weighted = self._evaluator(Y, directions)     # search score (weights change)
        unweighted = self._evaluator(Y, directions)   # reported (unweighted) quality
        covering = WeightedCovering(n, c.covering_strategy, c.covering_gamma)
        accepted: List[AcceptedRule] = []
        self.n_evaluated = 0
        q_first: Optional[float] = None

        def redundant(mask: np.ndarray) -> bool:
            return any(self._jaccard(mask, a.mask) >= c.redundancy_threshold for a in accepted)

        if covering.strategy == CoveringStrategy.NONE:
            # Without covering every iteration would return the same rule, so
            # take the top non-redundant results of a single search instead.
            bs = self._beam(gen, weighted, n)
            results = bs.search()
            self.n_evaluated += bs.n_evaluated
            for res in results:
                if len(accepted) >= c.n_rules or res.quality <= 0.0:
                    break
                if q_first is not None and res.quality < c.min_quality_fraction * q_first:
                    break
                if redundant(res.mask):
                    continue
                if q_first is None:
                    q_first = res.quality
                accepted.append(AcceptedRule(res.rule, res.mask, res.quality, res.quality, res.depth, bs.n_evaluated))
            return accepted

        for r in range(c.n_rules):
            weighted.set_weights(covering.weights)
            bs = self._beam(gen, weighted, n)
            results = bs.search()
            self.n_evaluated += bs.n_evaluated
            # best candidate whose covered set is not (nearly) an accepted one
            best = next((res for res in results if res.quality > 0.0 and not redundant(res.mask)), None)
            if best is None:
                if verbose:
                    print(f"  rule {r + 1}: no non-redundant rule with positive quality; stopping")
                break
            if q_first is not None and best.quality < c.min_quality_fraction * q_first:
                if verbose:
                    print(f"  rule {r + 1}: search quality {best.quality:.4f} < "
                          f"{c.min_quality_fraction} * {q_first:.4f}; stopping")
                break
            if q_first is None:
                q_first = best.quality
            q_unw = float(unweighted(best.mask))
            accepted.append(AcceptedRule(best.rule, best.mask, q_unw, float(best.quality), best.depth, bs.n_evaluated))
            if verbose:
                print(f"  rule {r + 1}: {best.rule}  n={best.size}  quality={q_unw:.4f}  "
                      f"search={best.quality:.4f}  ({bs.n_evaluated} candidates)")
            covering.update(best.mask)
        return accepted

    # ---- main entry --------------------------------------------------------
    def fit(self, dataset: Dataset) -> SetOfSubgroups:
        c = self.config
        rng = make_rng(c.random_state)
        fs_X = dataset.X_space
        fs_Y = dataset.Y_space
        Y = fs_Y.numeric
        n = fs_X.n
        if Y.shape[1] == 0:
            raise ValueError("Y has no usable columns")
        directions = direction_vector(fs_Y, c.target_direction)
        gen = self._generator(fs_X)

        if c.verbose:
            print(f"MRSD-Rules: {n} instances, {fs_X.p} X attributes, {Y.shape[1]} numeric Y columns, "
                  f"measure={c.quality_measure}, covering={c.covering_strategy}")

        accepted = self._covering_search(gen, Y, directions, verbose=c.verbose)
        self._accepted = accepted

        subgroups = [self._make_subgroup(a, i, dataset, Y) for i, a in enumerate(accepted)]
        result = SetOfSubgroups(subgroups).prune(QualityMeasures.quality, s_min=c.redundancy_threshold)
        result = result.sort_by_quality(QualityMeasures.quality)
        self._rules = [sg.rule for sg in result]

        if c.n_permutations > 0 and len(result) > 0:
            mode = InferenceMode(c.inference_mode)
            if mode == InferenceMode.RULES:
                validate_subgroups(
                    result, Y_space=fs_Y, statistic="y_shift", n_permutations=c.n_permutations,
                    random_state=derive_seed(rng),
                    selection_note="max-T over the returned rules only; rules were selected by a beam "
                                   "search on Y, so p-values are conditional on the selection")
            else:
                self._search_inference(result, gen, Y, directions, rng)

        if c.verbose:
            print(result.summary())
            for sg in result:
                print(f"  {sg}")
        return result

    def _make_subgroup(self, a: AcceptedRule, idx: int, dataset: Dataset, Y: np.ndarray) -> Subgroup:
        c = self.config
        n = Y.shape[0]
        t2, _, p_asym = hotelling_t2(Y, a.mask)
        qm = {
            "quality": float(a.quality),
            "quality_search": float(a.quality_search),
            "coverage": float(a.mask.sum() / n),
            "n_conditions": len(a.rule),
            "rule_index": idx,
            "t2": float(t2),
            "p_asymptotic": float(p_asym),
            "n_candidates_examined": int(a.n_evaluated),
            "quality_measure": c.quality_measure,
        }
        sg = Subgroup(
            description=str(a.rule),
            covered_instances=set(int(i) for i in np.flatnonzero(a.mask)),
            quality_measures=qm,
            algorithm="mrsd_rules",
            rule=a.rule,
            description_precision=1.0,
            description_recall=1.0,
        )
        sg.fill_target_statistics(dataset.Y)
        return sg

    # ---- swap randomization over the whole search --------------------------
    def _search_inference(self, result: SetOfSubgroups, gen: ConditionGenerator, Y: np.ndarray,
                          directions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Permute Y, rerun the unweighted beam search, record its best quality.

        The weighted searches of the covering loop score on a different scale,
        and the first (unweighted) search is where the maximum over the whole
        run is attained, so one unweighted search per permutation gives the
        null distribution of the best search quality."""
        c = self.config
        B = c.n_search_permutations
        n = Y.shape[0]
        null_max = np.empty(B)
        for b in range(B):
            Yp = Y[rng.permutation(n)]
            bs = self._beam(gen, self._evaluator(Yp, directions), n)
            res = bs.search()
            null_max[b] = res[0].quality if res else -np.inf
            if c.verbose and (b + 1) % 20 == 0:
                print(f"  search permutation {b + 1}/{B}")
        for sg in result:
            q = float(sg.quality)
            p = float(((null_max >= q).sum() + 1) / (B + 1))
            sg.p_value = p
            sg.p_value_corrected = p
            sg.correction_method = "search_permutation_max_quality"
            sg.quality_measures["test_statistic"] = q
            sg.quality_measures["p_bh"] = p
            sg.quality_measures["inference"] = (f"search; permuted=Y; B={B}; whole beam search rerun per "
                                                f"permutation, p = P(best search quality >= quality)")
        return null_max

    # ---- API ---------------------------------------------------------------
    def get_rules(self) -> List[Rule]:
        """Rules of the returned subgroups (best first)."""
        return list(self._rules)

    def print_rules(self) -> None:
        print("=" * 60)
        print("DISCOVERED RULES (MRSD-Rules)")
        print("=" * 60)
        if not self._accepted:
            print("(none)")
        n = len(self._accepted[0].mask) if self._accepted else 0
        for i, a in enumerate(self._accepted):
            print(f"Rule {i + 1}: IF {a.rule}")
            print(f"   quality={a.quality:.4f}  search score={a.quality_search:.4f}  "
                  f"size={int(a.mask.sum())} ({a.mask.sum() / max(n, 1):.1%})  conditions={len(a.rule)}")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_mrsd_rules(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
                   **config_kwargs) -> SetOfSubgroups:
    """Run MRSD-Rules on a DataFrame or a file path; ``config_kwargs`` -> ``MRSDRulesConfig``."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    config = MRSDRulesConfig(**config_kwargs)
    return MRSD_Rules(config).fit(dataset)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    age = rng.uniform(20, 70, n)
    income = rng.uniform(20000, 100000, n)
    education = rng.choice(["low", "medium", "high"], n)
    y1 = rng.normal(size=n)
    y2 = rng.normal(size=n)
    planted = (age >= 50) & (income >= 60000)
    y1[planted] += 1.5
    y2[planted] += 1.0
    df = pd.DataFrame({"age": age, "income": income, "education": education, "y1": y1, "y2": y2})
    print(f"Planted rule: age >= 50 AND income >= 60000 ({planted.sum()} instances)")
    result = run_mrsd_rules(df, ["age", "income", "education"], ["y1", "y2"],
                            n_rules=5, n_permutations=200, random_state=0, verbose=True)
    best = result[0]
    inter = len(best.covered_instances & set(np.flatnonzero(planted)))
    union = len(best.covered_instances | set(np.flatnonzero(planted)))
    print(f"\nBest rule: {best.rule}   Jaccard with planted = {inter / union:.2f}   "
          f"adjusted p = {best.p_value_corrected:.3f}")
    print("Recovered" if inter / union >= 0.5 and best.p_value_corrected < 0.05 else "Not recovered")
