"""
MRSD-Ensemble: consensus of several Multiple Responses Subgroup Discovery
(MRSD) algorithms.

Idea
----
Every member algorithm (MRSD-PT, MRSD-HC, the three MRSD-C2 variants, MRSD-CCA,
MRSD-EMM, MRSD-MV, MRSD-MI, MRSD-Rules) searches for subgroups in its own way
and scores them on its own, incomparable scale.  The ensemble therefore never
compares raw scores across algorithms.  It works on *instances*:

1. Each member is run with small, fast settings (no member-level permutation
   test, cheap classifiers, few initial subgroups / k values), ``verbose=False``
   and a seed derived from the ensemble generator.  Configuration mistakes
   (unknown kwargs in ``algorithm_configs``) are detected before anything runs
   by checking the kwargs against ``dataclasses.fields`` of each ``*Config``.
   Data-dependent failures are logged with the full traceback, stored in
   ``get_errors()`` and (with ``skip_on_error=True``) skipped.
2. Inside each algorithm the subgroups are deduplicated and **ranked** (by
   ``p_value_corrected`` when every subgroup has one, otherwise by
   ``quality_measures['quality']``); only the best ``subgroups_per_algorithm``
   enter the fusion.
3. Co-association consensus: ``C[i, j]`` = fraction of successful algorithms
   that have at least one subgroup containing both instances *i* and *j*
   (each algorithm counts at most once per pair).  Average-linkage hierarchical
   clustering of ``1 - C`` cut at ``1 - consensus_threshold`` (default 0.5:
   "at least half of the algorithms put these two instances together") gives
   the consensus subgroups; clusters smaller than ``n_min`` are dropped.
4. For every consensus subgroup the *supporting* algorithms are those with a
   subgroup of Jaccard >= ``support_jaccard`` with it.  The fusion strategy
   decides the score:

   * ``voting``: weighted share of supporting algorithms,
   * ``rank_aggregation``: reciprocal-rank fusion of the within-algorithm rank
     of the best-matching subgroup of each supporting algorithm (normalised so
     that rank 1 in every algorithm gives 1.0),
   * ``union``: every distinct member subgroup is returned, scored by the share
     of algorithms that support it (no clustering),
   * ``intersection``: voting restricted to clusters supported by all
     successful algorithms.

   Weights are ``weights.get(alg, 0.0)`` with the denominator taken over the
   algorithms that actually ran, so every score lies in [0, 1].
   The representative member subgroup is the medoid of the supporting
   subgroups (highest mean Jaccard to the consensus set and to the other
   supporters); the consensus subgroup itself is described by a rule from
   ``rules.describe_mask_with_tree``.
5. No fake p-values.  ``inference_mode='candidates'`` (default) tests the
   consensus subgroups with ``inference.validate_subgroups(statistic='y_shift')``
   (Y rows permuted, Westfall-Young max-T over the consensus set) and records
   that the candidates were selected using both X and Y, so these p-values are
   conditional on the candidate set (optimistic).  ``inference_mode='search'``
   is the calibrated alternative: Y is permuted ``n_search_permutations``
   times, the whole pipeline (members + fusion) is rerun on every permutation
   and the largest Y-shift statistic of any consensus subgroup found is
   recorded; the p-value of a consensus subgroup is the share of permutations
   whose maximum is at least its own statistic (family-wise over everything
   the ensemble can find).  It costs one full ensemble run per permutation.
6. Optional stability (``n_bootstrap > 0``): the members are rerun on
   subsamples of ``bootstrap_fraction`` (default 80 %) of the rows drawn
   without replacement; ``quality_measures['stability']`` is the mean over
   replicates of the Jaccard between the consensus subgroup (restricted to the
   sampled rows) and the closest subgroup found in that replicate.

``quality_measures['quality']`` of the returned subgroups is the consensus
score; ``algorithm`` is ``"mrsd_ensemble"``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .common import derive_seed, make_rng, mask_from_indices
from .data_loader import DataLoader, Dataset
from .inference import hotelling_t2, validate_subgroups, y_shift_statistic
from .rules import describe_mask_with_tree
from .core import QualityMeasures, SetOfSubgroups, Subgroup

from .mrsdpt import MRSDPTConfig, run_mrsdpt
from .mrsdhc import MRSDHCConfig, run_mrsdhc
from .mrsdc2_swapping import SwappingMethodConfig, run_swapping_method
from .mrsdc2_semisupervised import SemiSupervisedConfig, run_semisupervised_method
from .mrsdc2_ranking import RankingMethodConfig, run_ranking_method
from .mrsdcca import MRSDCCAConfig, run_mrsdcca
from .mrsdemm import MRSDEMMConfig, run_mrsdemm
from .mrsd_mv import MRSDMVConfig, run_mrsd_mv
from .mrsd_mi import MRSDMIConfig, run_mrsd_mi
from .mrsd_rules import MRSDRulesConfig, run_mrsd_rules

logger = logging.getLogger("mrsd")

ALGORITHM_NAME = "mrsd_ensemble"


# =============================================================================
# ENUMS
# =============================================================================

class FusionStrategy(Enum):
    """How the support of the co-association clusters is turned into a score."""
    VOTING = "voting"
    RANK_AGGREGATION = "rank_aggregation"
    UNION = "union"
    INTERSECTION = "intersection"


class InferenceMode(Enum):
    """candidates: max-T over the consensus set (conditional); search: swap
    randomization over the whole ensemble pipeline."""
    CANDIDATES = "candidates"
    SEARCH = "search"


class WeightScheme(Enum):
    """How the member algorithms are weighted in the fusion."""
    EQUAL = "equal"
    CUSTOM = "custom"


# =============================================================================
# ALGORITHM REGISTRY
# =============================================================================

@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    name: str
    description: str
    runner: Any
    config_class: type
    category: str
    fast_defaults: Dict[str, Any]


# Small, fast member settings.  n_permutations=0 everywhere: the ensemble does
# its own inference on the consensus subgroups.
_C2_FAST = dict(classifier_X="decision_tree", classifier_Y="decision_tree", n_folds=3,
                n_initial_subgroups=3, n_permutations=0)

_SPECS: List[AlgorithmSpec] = [
    AlgorithmSpec("pt", "MRSD-PT", "Partitioning and Tables (partition pairs, contingency cells)", run_mrsdpt, MRSDPTConfig,
                  "statistical", dict(k_values=[2, 3], clustering_algorithms_X=["kmeans"],
                                      clustering_algorithms_Y=["kmeans"], n_permutations=0)),
    AlgorithmSpec("hc", "MRSD-HC", "Hierarchical Clustering (of Y) and Classification (from X)", run_mrsdhc, MRSDHCConfig,
                  "clustering", dict(n_folds=3, fast_classifiers=True, n_permutations=0)),
    AlgorithmSpec("swapping", "MRSD-C2 Swapping", "Local search with swapping", run_swapping_method,
                  SwappingMethodConfig, "classification", dict(_C2_FAST, swap_batch_size=5, max_iterations=15)),
    AlgorithmSpec("semisupervised", "MRSD-C2 Semi-supervised", "Semi-supervised membership learning",
                  run_semisupervised_method, SemiSupervisedConfig, "classification",
                  dict(_C2_FAST, max_self_training_iterations=3)),
    AlgorithmSpec("ranking", "MRSD-C2 Ranking", "Instance ranking", run_ranking_method, RankingMethodConfig,
                  "classification", dict(_C2_FAST, max_sizes=6)),
    AlgorithmSpec("cca", "MRSD-CCA", "Canonical Correlation Analysis", run_mrsdcca, MRSDCCAConfig,
                  "relational", dict(k_values=[2, 3], n_permutations=0)),
    AlgorithmSpec("emm", "MRSD-EMM", "Exceptional Model Mining", run_mrsdemm, MRSDEMMConfig,
                  "relational", dict(beam_width=5, max_depth=2, n_random=20, n_permutations=0)),
    AlgorithmSpec("mv", "MRSD-MV", "Multi-View Clustering", run_mrsd_mv, MRSDMVConfig,
                  "relational", dict(k_values=[2, 3], classifier_X="decision_tree", classifier_Y="decision_tree",
                                     n_permutations=0)),
    AlgorithmSpec("mi", "MRSD-MI", "Mutual Information", run_mrsd_mi, MRSDMIConfig,
                  "relational", dict(beam_width=5, max_depth=2, n_null_shuffles=5, n_permutations=0)),
    AlgorithmSpec("rules", "MRSD-Rules", "Rule-based (CN2-SD style)", run_mrsd_rules, MRSDRulesConfig,
                  "interpretable", dict(beam_width=5, max_depth=2, n_rules=5, n_permutations=0)),
]


class AlgorithmRegistry:
    """Registry of the member algorithms (id -> AlgorithmSpec)."""

    ALGORITHMS: Dict[str, AlgorithmSpec] = {s.id: s for s in _SPECS}

    @classmethod
    def get_all_ids(cls) -> List[str]:
        return list(cls.ALGORITHMS.keys())

    @classmethod
    def get_by_category(cls, category: str) -> List[str]:
        return [k for k, v in cls.ALGORITHMS.items() if v.category == category]

    @classmethod
    def get(cls, alg_id: str) -> AlgorithmSpec:
        if alg_id not in cls.ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{alg_id}'; known: {cls.get_all_ids()}")
        return cls.ALGORITHMS[alg_id]

    @classmethod
    def get_info(cls, alg_id: str) -> Dict[str, Any]:
        return dataclasses.asdict(cls.get(alg_id))

    @classmethod
    def config_fields(cls, alg_id: str) -> List[str]:
        return [f.name for f in dataclasses.fields(cls.get(alg_id).config_class)]

    @classmethod
    def check_kwargs(cls, alg_id: str, kwargs: Dict[str, Any], origin: str) -> None:
        """Raise ValueError if `kwargs` has keys that are not fields of the algorithm's Config."""
        known = set(cls.config_fields(alg_id))
        unknown = sorted(k for k in kwargs if k not in known)
        if unknown:
            raise ValueError(
                f"{origin} for algorithm '{alg_id}' contains unknown config kwargs {unknown}; "
                f"valid fields of {cls.get(alg_id).config_class.__name__}: {sorted(known)}")


# every fast default must be a real config field (checked at import time)
for _spec in _SPECS:
    AlgorithmRegistry.check_kwargs(_spec.id, _spec.fast_defaults, "fast_defaults")
    AlgorithmRegistry.check_kwargs(_spec.id, {"verbose": False, "random_state": 0}, "common settings")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MRSDEnsembleConfig:
    """Configuration of MRSD-Ensemble."""

    # which members to run (None = all) and per-member overrides {alg_id: {kwarg: value}}
    algorithms: Optional[List[str]] = None
    algorithm_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # fusion
    fusion_strategy: str = "voting"          # voting | rank_aggregation | union | intersection
    consensus_threshold: float = 0.5         # C[i,j] >= threshold -> i and j belong together
    support_jaccard: float = 0.5             # member subgroup supports a consensus subgroup if Jaccard >= this
    min_algorithms: int = 2                  # minimum number of supporting algorithms (voting / rank_aggregation)
    rrf_k: int = 60                          # reciprocal-rank-fusion constant

    # weights
    algorithm_weights: str = "equal"         # equal | custom
    custom_weights: Optional[Dict[str, float]] = None

    # per-member cap on the number of (ranked) subgroups used in the fusion
    subgroups_per_algorithm: int = 5

    # consensus subgroup constraints and output
    n_min: int = 10
    n_max: Optional[int] = None              # default n // 2
    top_k: int = 20
    redundancy_threshold: float = 0.7
    rule_max_depth: int = 3

    # inference on the consensus subgroups (n_permutations = 0 -> off)
    n_permutations: int = 200
    inference_mode: str = "search"           # search (honest swap randomization, default) | candidates (conditional, fast)
    n_search_permutations: int = 50          # B for inference_mode='search'

    # stability by subsampling (0 = off)
    n_bootstrap: int = 0
    bootstrap_fraction: float = 0.8

    # execution
    time_budget: Optional[float] = None      # seconds; members are skipped once the budget is used up
    skip_on_error: bool = True               # data-dependent failures of a member are logged and skipped

    random_state: Optional[int] = None
    verbose: bool = False

    def __post_init__(self):
        FusionStrategy(self.fusion_strategy)
        WeightScheme(self.algorithm_weights)
        InferenceMode(self.inference_mode)
        if self.n_search_permutations < 1:
            raise ValueError("n_search_permutations must be >= 1")
        if not 0.0 <= self.consensus_threshold <= 1.0:
            raise ValueError("consensus_threshold must be in [0, 1]")
        if not 0.0 <= self.support_jaccard <= 1.0:
            raise ValueError("support_jaccard must be in [0, 1]")
        if self.min_algorithms < 1:
            raise ValueError("min_algorithms must be >= 1")
        if self.subgroups_per_algorithm < 1:
            raise ValueError("subgroups_per_algorithm must be >= 1")
        if self.n_permutations < 0 or self.n_bootstrap < 0:
            raise ValueError("n_permutations and n_bootstrap must be >= 0")
        if not 0.0 < self.bootstrap_fraction < 1.0:
            raise ValueError("bootstrap_fraction must be in (0, 1)")
        if self.algorithm_weights == "custom" and not self.custom_weights:
            raise ValueError("algorithm_weights='custom' needs custom_weights")
        for alg_id in self.algorithms or []:
            AlgorithmRegistry.get(alg_id)
        for alg_id, overrides in self.algorithm_configs.items():
            AlgorithmRegistry.get(alg_id)
            if not isinstance(overrides, dict):
                raise TypeError(f"algorithm_configs['{alg_id}'] must be a dict")
            AlgorithmRegistry.check_kwargs(alg_id, overrides, "algorithm_configs")


# =============================================================================
# RUN RECORDS AND CONSENSUS SUBGROUPS
# =============================================================================

@dataclass
class AlgorithmRun:
    """What happened when a member algorithm was run."""
    alg_id: str
    seconds: float
    succeeded: bool
    n_subgroups: int = 0
    error: Optional[str] = None          # full traceback for data-dependent failures, or the skip reason
    skipped: bool = False


@dataclass
class MemberSubgroup:
    """A subgroup of a member algorithm with its within-algorithm rank (1 = best)."""
    alg_id: str
    rank: int
    subgroup: Subgroup
    mask: np.ndarray


@dataclass
class ConsensusSubgroup:
    """A cluster of instances supported by several member algorithms."""
    indices: np.ndarray                       # sorted instance indices
    algorithms: List[str]                     # supporting algorithms
    support: Dict[str, MemberSubgroup]        # best-matching member subgroup per supporting algorithm
    support_jaccard: Dict[str, float]         # its Jaccard with the consensus set
    consensus_score: float
    representative: Optional[MemberSubgroup]  # medoid of the supporters

    @property
    def n_algorithms(self) -> int:
        return len(self.algorithms)

    @property
    def coverage(self) -> int:
        return int(len(self.indices))


# =============================================================================
# HELPERS
# =============================================================================

def _rank_within_algorithm(result: SetOfSubgroups) -> List[Subgroup]:
    """Deduplicate and order a member's subgroups best first: by p_value_corrected
    when every subgroup has one (ties by quality), otherwise by 'quality'."""
    sgs = result.deduplicate(QualityMeasures.quality).to_list()
    if sgs and all(sg.p_value_corrected is not None for sg in sgs):
        return sorted(sgs, key=lambda sg: (sg.p_value_corrected, -QualityMeasures.quality(sg)))
    return sorted(sgs, key=lambda sg: -QualityMeasures.quality(sg))


def _jaccard_rows(M: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Jaccard between every row of the boolean matrix M and the boolean vector s."""
    if M.shape[0] == 0:
        return np.zeros(0)
    inter = M.astype(np.int32) @ s.astype(np.int32)
    union = M.sum(axis=1) + int(s.sum()) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def _jaccard_pairs(M: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard matrix between the rows of the boolean matrix M."""
    Mi = M.astype(np.int32)
    inter = Mi @ Mi.T
    sizes = Mi.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def coassociation_matrix(members_by_alg: Dict[str, List[MemberSubgroup]], n: int) -> np.ndarray:
    """C[i,j] = share of algorithms with a subgroup containing both i and j (one count per algorithm)."""
    C = np.zeros((n, n), dtype=np.float32)
    for alg_id, members in members_by_alg.items():
        together = np.zeros((n, n), dtype=bool)
        for m in members:
            idx = np.flatnonzero(m.mask)
            together[np.ix_(idx, idx)] = True
        C += together
    if members_by_alg:
        C /= float(len(members_by_alg))
    return C


def consensus_clusters(C: np.ndarray, consensus_threshold: float, n_min: int,
                       n_max: Optional[int] = None) -> List[np.ndarray]:
    """Average-linkage clustering of 1 - C cut at 1 - consensus_threshold; clusters
    with n_min <= size (<= n_max) as sorted index arrays, largest co-association first."""
    n = C.shape[0]
    if n < 2:
        return []
    D = 1.0 - np.asarray(C, dtype=float)
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2.0, 0.0, 1.0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - consensus_threshold + 1e-12, criterion="distance")
    clusters = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) < n_min or (n_max is not None and len(idx) > n_max):
            continue
        clusters.append(idx)
    # order by mean within-cluster co-association (ties: larger first) for determinism
    def strength(idx):
        sub = C[np.ix_(idx, idx)]
        return (float(sub.mean()), len(idx))
    clusters.sort(key=strength, reverse=True)
    return clusters


# =============================================================================
# FUSION
# =============================================================================

class ResultFusion:
    """Turns the ranked member subgroups into consensus subgroups."""

    def __init__(self, config: MRSDEnsembleConfig, n: int):
        self.cfg = config
        self.n = n
        self.strategy = FusionStrategy(config.fusion_strategy)
        self.C: Optional[np.ndarray] = None

    # -- weights --------------------------------------------------------------
    def weights(self, alg_ids: Sequence[str]) -> Dict[str, float]:
        if WeightScheme(self.cfg.algorithm_weights) == WeightScheme.CUSTOM:
            w = self.cfg.custom_weights or {}
            return {a: max(float(w.get(a, 0.0)), 0.0) for a in alg_ids}
        return {a: 1.0 for a in alg_ids}

    # -- support --------------------------------------------------------------
    def _support(self, s: np.ndarray, members_by_alg: Dict[str, List[MemberSubgroup]]
                 ) -> Tuple[Dict[str, MemberSubgroup], Dict[str, float]]:
        support, jac = {}, {}
        for alg_id, members in members_by_alg.items():
            if not members:
                continue
            M = np.stack([m.mask for m in members])
            j = _jaccard_rows(M, s)
            b = int(np.argmax(j))
            if j[b] >= self.cfg.support_jaccard:
                support[alg_id] = members[b]
                jac[alg_id] = float(j[b])
        return support, jac

    def _score(self, support: Dict[str, MemberSubgroup], w: Dict[str, float]) -> float:
        denom = sum(w.values())
        if denom <= 0:
            return 0.0
        if self.strategy == FusionStrategy.RANK_AGGREGATION:
            k = float(self.cfg.rrf_k)
            num = sum(w[a] / (k + m.rank) for a, m in support.items())
            return float(num / sum(w[a] / (k + 1.0) for a in w))
        return float(sum(w[a] for a in support) / denom)

    @staticmethod
    def _medoid(s: np.ndarray, support: Dict[str, MemberSubgroup]) -> Optional[MemberSubgroup]:
        if not support:
            return None
        members = list(support.values())
        M = np.stack([m.mask for m in members] + [s])
        J = _jaccard_pairs(M)
        # mean Jaccard to the consensus set and to the other supporters
        k = len(members)
        mean_j = (J[:k, :].sum(axis=1) - 1.0) / max(k, 1)
        return members[int(np.argmax(mean_j))]

    # -- strategies -----------------------------------------------------------
    def fuse(self, members_by_alg: Dict[str, List[MemberSubgroup]]) -> List[ConsensusSubgroup]:
        members_by_alg = {a: ms for a, ms in members_by_alg.items()}
        alg_ids = list(members_by_alg.keys())
        w = self.weights(alg_ids)
        if not alg_ids or not any(members_by_alg.values()):
            return []
        if self.strategy == FusionStrategy.UNION:
            return self._union(members_by_alg, w)

        self.C = coassociation_matrix(members_by_alg, self.n)
        n_max = self.cfg.n_max if self.cfg.n_max is not None else self.n // 2
        clusters = consensus_clusters(self.C, self.cfg.consensus_threshold, self.cfg.n_min, n_max)
        min_algs = len(alg_ids) if self.strategy == FusionStrategy.INTERSECTION else self.cfg.min_algorithms
        out = []
        for idx in clusters:
            s = mask_from_indices(idx, self.n)
            support, jac = self._support(s, members_by_alg)
            if len(support) < min_algs:
                continue
            out.append(ConsensusSubgroup(indices=idx, algorithms=sorted(support), support=support,
                                         support_jaccard=jac, consensus_score=self._score(support, w),
                                         representative=self._medoid(s, support)))
        out.sort(key=lambda c: (c.consensus_score, c.n_algorithms, c.coverage), reverse=True)
        return out

    def _union(self, members_by_alg: Dict[str, List[MemberSubgroup]], w: Dict[str, float]) -> List[ConsensusSubgroup]:
        seen: Dict[frozenset, ConsensusSubgroup] = {}
        for alg_id, members in members_by_alg.items():
            for m in members:
                key = m.subgroup.indices_key
                if key in seen:
                    continue
                idx = np.flatnonzero(m.mask)
                support, jac = self._support(m.mask, members_by_alg)
                support.setdefault(alg_id, m)
                jac.setdefault(alg_id, 1.0)
                seen[key] = ConsensusSubgroup(indices=idx, algorithms=sorted(support), support=support,
                                              support_jaccard=jac, consensus_score=self._score(support, w),
                                              representative=m)
        out = list(seen.values())
        out.sort(key=lambda c: (c.consensus_score, c.n_algorithms, c.coverage), reverse=True)
        return out


# =============================================================================
# MRSD-ENSEMBLE
# =============================================================================

class MRSD_Ensemble:
    """Runs the member algorithms, builds the co-association consensus, tests it."""

    def __init__(self, config: Optional[MRSDEnsembleConfig] = None):
        self.config = config or MRSDEnsembleConfig()
        self._results: Dict[str, SetOfSubgroups] = {}
        self._runs: Dict[str, AlgorithmRun] = {}
        self._members: Dict[str, List[MemberSubgroup]] = {}
        self._consensus: List[ConsensusSubgroup] = []
        self._coassociation: Optional[np.ndarray] = None

    # -- members --------------------------------------------------------------
    def _member_kwargs(self, alg_id: str, rng: np.random.Generator) -> Dict[str, Any]:
        spec = AlgorithmRegistry.get(alg_id)
        overrides = self.config.algorithm_configs.get(alg_id, {})
        AlgorithmRegistry.check_kwargs(alg_id, overrides, "algorithm_configs")
        kwargs = {**spec.fast_defaults, **overrides}
        kwargs["verbose"] = False
        kwargs["random_state"] = derive_seed(rng)
        return kwargs

    def _run_members(self, data: pd.DataFrame, X_columns: List[str], Y_columns: List[str],
                     alg_ids: List[str], rng: np.random.Generator, record: bool = True
                     ) -> Dict[str, SetOfSubgroups]:
        """Run the member algorithms; returns {alg_id: SetOfSubgroups} of the successful ones."""
        results: Dict[str, SetOfSubgroups] = {}
        t_start = time.perf_counter()
        for alg_id in alg_ids:
            kwargs = self._member_kwargs(alg_id, rng)   # always consume the seed (determinism)
            elapsed_total = time.perf_counter() - t_start
            if self.config.time_budget is not None and elapsed_total > self.config.time_budget:
                msg = f"skipped: time budget of {self.config.time_budget:.1f}s exceeded after {elapsed_total:.1f}s"
                logger.warning("mrsd_ensemble: %s %s", alg_id, msg)
                if record:
                    self._runs[alg_id] = AlgorithmRun(alg_id, 0.0, False, error=msg, skipped=True)
                continue
            spec = AlgorithmRegistry.get(alg_id)
            if self.config.verbose:
                print(f"  running {spec.name:26s}", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                res = spec.runner(data, X_columns, Y_columns, **kwargs)
            except TypeError:
                # configuration error (bad kwarg value / signature): never swallowed
                raise
            except Exception as exc:  # data-dependent failure
                dt = time.perf_counter() - t0
                tb = traceback.format_exc()
                logger.exception("mrsd_ensemble: member algorithm %s failed", alg_id)
                if record:
                    self._runs[alg_id] = AlgorithmRun(alg_id, dt, False, error=tb)
                if self.config.verbose:
                    print(f"FAILED ({type(exc).__name__}: {exc})")
                if not self.config.skip_on_error:
                    raise
                continue
            dt = time.perf_counter() - t0
            results[alg_id] = res
            if record:
                self._runs[alg_id] = AlgorithmRun(alg_id, dt, True, n_subgroups=len(res))
            if self.config.verbose:
                print(f"ok: {len(res):3d} subgroups, {dt:.1f}s")
        return results

    def _members_from_results(self, results: Dict[str, SetOfSubgroups], n: int,
                              index_map: Optional[np.ndarray] = None) -> Dict[str, List[MemberSubgroup]]:
        """Rank within each algorithm, keep the top subgroups_per_algorithm, build masks
        (optionally mapping subsample positions back to original indices)."""
        out: Dict[str, List[MemberSubgroup]] = {}
        for alg_id, res in results.items():
            ranked = _rank_within_algorithm(res)[: self.config.subgroups_per_algorithm]
            members = []
            for r, sg in enumerate(ranked, start=1):
                idx = np.fromiter(sg.covered_instances, dtype=int) if sg.covered_instances else np.zeros(0, dtype=int)
                if index_map is not None:
                    idx = index_map[idx]
                members.append(MemberSubgroup(alg_id, r, sg, mask_from_indices(idx, n)))
            out[alg_id] = members
        return out

    # -- main -----------------------------------------------------------------
    def fit(self, dataset: Dataset) -> SetOfSubgroups:
        cfg = self.config
        rng = make_rng(cfg.random_state)
        alg_ids = list(cfg.algorithms) if cfg.algorithms else AlgorithmRegistry.get_all_ids()
        # validate every override before anything runs
        for alg_id, overrides in cfg.algorithm_configs.items():
            AlgorithmRegistry.check_kwargs(alg_id, overrides, "algorithm_configs")

        n = len(dataset.X)
        data = pd.concat([dataset.X, dataset.Y], axis=1)
        X_columns, Y_columns = list(dataset.X_columns), list(dataset.Y_columns)

        if cfg.verbose:
            print(f"MRSD-Ensemble: {len(alg_ids)} member algorithms on n={n}")
        self._runs = {}
        self._results = self._run_members(data, X_columns, Y_columns, alg_ids, rng)
        if cfg.verbose:
            print(f"  {len(self._results)}/{len(alg_ids)} algorithms succeeded")

        self._members = self._members_from_results(self._results, n)
        fusion = ResultFusion(cfg, n)
        self._consensus = fusion.fuse(self._members)
        self._coassociation = fusion.C
        n_success = len(self._results)

        subgroups = [self._to_subgroup(c, dataset, n_success) for c in self._consensus]
        result = SetOfSubgroups(subgroups).prune(QualityMeasures.quality, s_min=cfg.redundancy_threshold)
        result = SetOfSubgroups(result.to_list()[: cfg.top_k])

        if cfg.n_bootstrap > 0 and len(result) > 0:
            self._stability(result, data, X_columns, Y_columns, alg_ids, n, rng)

        if cfg.n_permutations > 0 and len(result) > 0:
            if InferenceMode(cfg.inference_mode) == InferenceMode.SEARCH:
                self._search_inference(result, dataset, alg_ids, n, rng)
            else:
                validate_subgroups(
                    result, Y_space=dataset.Y_space, X_space=dataset.X_space, statistic="y_shift",
                    n_permutations=cfg.n_permutations, random_state=derive_seed(rng),
                    selection_note="consensus of subgroups selected using X and Y; "
                                   "p-values conditional on candidate set")
        if cfg.verbose:
            print(f"  {len(result)} consensus subgroups")
            for sg in result:
                print(f"    {sg.description}")
        return result

    # -- output ---------------------------------------------------------------
    def _to_subgroup(self, c: ConsensusSubgroup, dataset: Dataset, n_success: int) -> Subgroup:
        n = len(dataset.X)
        mask = mask_from_indices(c.indices, n)
        desc = describe_mask_with_tree(dataset.X_space, mask, max_depth=self.config.rule_max_depth)
        rep = c.representative
        algs = ", ".join(c.algorithms)
        description = (f"consensus of {c.n_algorithms}/{n_success} algorithms ({algs}); "
                       f"n={c.coverage}; rule: {desc.rule}")
        qm: Dict[str, Any] = {
            "quality": float(c.consensus_score),
            "consensus_score": float(c.consensus_score),
            "n_algorithms": int(c.n_algorithms),
            "n_algorithms_run": int(n_success),
            "algorithm_coverage": float(c.n_algorithms / n_success) if n_success else 0.0,
            "algorithms": algs,
            "mean_support_jaccard": float(np.mean(list(c.support_jaccard.values()))) if c.support_jaccard else 0.0,
            "fusion_strategy": self.config.fusion_strategy,
        }
        if rep is not None:
            qm["representative_algorithm"] = rep.alg_id
            qm["representative_rank"] = int(rep.rank)
            qm["representative_jaccard"] = float(c.support_jaccard.get(rep.alg_id, 0.0))
        sg = Subgroup(
            description=description,
            covered_instances=set(int(i) for i in c.indices),
            quality_measures=qm,
            algorithm=ALGORITHM_NAME,
            rule=desc.rule,
            description_precision=float(desc.precision),
            description_recall=float(desc.recall),
            clustering_X="ensemble",
            clustering_Y="consensus",
            clustering_metadata_X={
                "algorithms": list(c.algorithms),
                "support_jaccard": dict(c.support_jaccard),
                "member_ranks": {a: m.rank for a, m in c.support.items()},
                "representative": None if rep is None else {
                    "algorithm": rep.alg_id, "rank": rep.rank, "description": rep.subgroup.description,
                    "rule": None if rep.subgroup.rule is None else str(rep.subgroup.rule),
                    "coverage": rep.subgroup.coverage},
            },
        )
        sg.fill_target_statistics(dataset.Y)
        return sg

    # -- search-level inference (swap randomization) ---------------------------
    def _search_inference(self, result: SetOfSubgroups, dataset: Dataset, alg_ids: List[str], n: int,
                          rng: np.random.Generator) -> None:
        """Permute the rows of Y, rerun members + fusion, record the largest Y-shift
        statistic of any consensus subgroup found; p = share of permutations whose
        maximum reaches the observed statistic (family-wise over the whole search)."""
        cfg = self.config
        B = cfg.n_search_permutations
        Y_num = dataset.Y_space.numeric
        sgs = result.to_list()
        observed = np.array([y_shift_statistic(Y_num, sg.mask(n)) for sg in sgs])
        null_max = np.full(B, -np.inf)
        for b in range(B):
            perm = rng.permutation(n)
            data_p = pd.concat([dataset.X.reset_index(drop=True),
                                dataset.Y.iloc[perm].reset_index(drop=True)], axis=1)
            if cfg.verbose:
                print(f"  search permutation {b + 1}/{B}")
            res_p = self._run_members(data_p, list(dataset.X_columns), list(dataset.Y_columns), alg_ids, rng,
                                      record=False)
            cons_p = ResultFusion(cfg, n).fuse(self._members_from_results(res_p, n))
            if cons_p:
                Yp = Y_num[perm]
                null_max[b] = max(y_shift_statistic(Yp, mask_from_indices(c.indices, n)) for c in cons_p)
        for sg, obs in zip(sgs, observed):
            p = float((1 + np.sum(null_max >= obs)) / (B + 1))
            sg.p_value = p
            sg.p_value_corrected = p
            sg.correction_method = "search_permutation_maxT"
            t2, _, pa = hotelling_t2(Y_num, sg.mask(n))
            sg.quality_measures["test_statistic"] = float(obs)
            sg.quality_measures["t2"] = float(t2)
            sg.quality_measures["p_asymptotic"] = float(pa)
            sg.quality_measures["inference"] = (f"y_shift; permuted=Y; B={B}; swap randomization over the whole "
                                                f"ensemble (members + fusion), family-wise over the search")

    # -- stability ------------------------------------------------------------
    def _stability(self, result: SetOfSubgroups, data: pd.DataFrame, X_columns: List[str],
                   Y_columns: List[str], alg_ids: List[str], n: int, rng: np.random.Generator) -> None:
        """Rerun the members on subsamples (without replacement) and store, per consensus
        subgroup, the mean Jaccard with the closest subgroup found in each replicate."""
        cfg = self.config
        sgs = result.to_list()
        masks = np.stack([sg.mask(n) for sg in sgs])
        m = max(int(round(cfg.bootstrap_fraction * n)), 2)
        scores = np.zeros((cfg.n_bootstrap, len(sgs)))
        for b in range(cfg.n_bootstrap):
            sample = np.sort(rng.choice(n, size=m, replace=False))
            sub = data.iloc[sample].reset_index(drop=True)
            if cfg.verbose:
                print(f"  stability replicate {b + 1}/{cfg.n_bootstrap}")
            rep_results = self._run_members(sub, X_columns, Y_columns, alg_ids, rng, record=False)
            members = self._members_from_results(rep_results, n, index_map=sample)
            found = [mm.mask for ms in members.values() for mm in ms]
            if not found:
                continue
            F = np.stack(found)
            in_sample = mask_from_indices(sample, n)
            for i in range(len(sgs)):
                s = masks[i] & in_sample
                if s.sum() == 0:
                    continue
                scores[b, i] = float(_jaccard_rows(F, s).max())
        for i, sg in enumerate(sgs):
            sg.quality_measures["stability"] = float(scores[:, i].mean())
            sg.quality_measures["stability_n_bootstrap"] = int(cfg.n_bootstrap)

    # -- accessors ------------------------------------------------------------
    def get_results_by_algorithm(self) -> Dict[str, SetOfSubgroups]:
        """Raw results of the successful member algorithms."""
        return self._results

    def get_timings(self) -> Dict[str, float]:
        """Wall time (s) of every member algorithm that was attempted (skipped: 0.0)."""
        return {a: r.seconds for a, r in self._runs.items()}

    def get_run_records(self) -> Dict[str, AlgorithmRun]:
        """Full per-algorithm records: seconds, succeeded, n_subgroups, error, skipped."""
        return self._runs

    def get_errors(self) -> Dict[str, str]:
        """Full tracebacks (or skip reasons) of the member algorithms that did not succeed."""
        return {a: r.error for a, r in self._runs.items() if not r.succeeded}

    def get_consensus_subgroups(self) -> List[ConsensusSubgroup]:
        return self._consensus

    def get_coassociation_matrix(self) -> Optional[np.ndarray]:
        return self._coassociation


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_mrsd_ensemble(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
                      **config_kwargs) -> SetOfSubgroups:
    """Run MRSD-Ensemble on a DataFrame or a file path; see MRSDEnsembleConfig for kwargs."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    return MRSD_Ensemble(MRSDEnsembleConfig(**config_kwargs)).fit(dataset)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y1, y2 = rng.normal(size=n), rng.normal(size=n)
    planted = x1 > 1.0
    y1[planted] += 2.0
    y2[planted] -= 1.0
    df = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})

    ens = MRSD_Ensemble(MRSDEnsembleConfig(algorithms=["rules", "emm", "pt", "mi", "cca"],
                                           n_permutations=100, random_state=0, verbose=True))
    res = ens.fit(DataLoader.from_dataframe(df, X_columns=["x1", "x2"], Y_columns=["y1", "y2"]))
    planted_set = set(np.flatnonzero(planted))
    for sg in res:
        j = len(sg.covered_instances & planted_set) / len(sg.covered_instances | planted_set)
        print(f"score={sg.quality:.2f} p_adj={sg.p_value_corrected} jaccard_with_planted={j:.2f}  {sg.rule}")
    best = max((len(sg.covered_instances & planted_set) / len(sg.covered_instances | planted_set) for sg in res), default=0.0)
    print("planted subgroup recovered:", best >= 0.5)
    print("timings:", {a: round(t, 2) for a, t in ens.get_timings().items()})
