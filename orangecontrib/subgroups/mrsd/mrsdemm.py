"""
MRSD-EMM: Multiple Responses Subgroup Discovery with Exceptional Model Mining.

Question answered
-----------------
Is the *relationship* between the descriptive variables X and the responses Y
different inside a subgroup G than in its complement D \\ G?  (A mean shift of
Y alone is not exceptional for EMM; that is what the other MRSD methods test.)

Model classes and exceptionality statistics
-------------------------------------------
All statistics compare G with its complement and are normalized by their
sampling error so that they do not trivially favour small subgroups.  The
search score (`quality_measures['quality']`) is -log10 of the asymptotic
p-value of the chosen statistic; permutation inference then calibrates it.

* 'correlation' (default): Fisher-z difference of every pairwise correlation
  r(X_i, Y_j) between G and its complement,
      z_ij = (atanh r_G - atanh r_C) / sqrt(1/(n_G-3) + 1/(n_C-3)).
  Aggregation over the d_X * d_Y pairs (`correlation_aggregation`):
  'max_abs_z' (default; p-value by Sidak over the pairs) or 'mean_sq_z'
  (sum of z^2 ~ chi-square with d_X*d_Y df).
* 'regression': interaction F-test (Chow-type) of Y_j ~ X * 1[G] per
  response, i.e. the restricted model with a common slope vector and separate
  intercepts [1, 1[G], X] against separate regressions in G and in the
  complement.  Only slope changes count; mean shifts are absorbed by the
  group intercept.  Per-target p-values are combined with Fisher's method
  (`target_combination='fisher'`, default) or the minimum p with Sidak
  correction ('max').
* 'r2_difference': kept for backward compatibility; it is the same
  interaction F-test (an unnormalized R^2 difference is not a valid score),
  with R^2 inside / outside G reported in the quality measures.

Dropped measures: 'rv_coefficient' and 'distance_correlation' had no sampling
distribution attached (raw differences favour small subgroups), dCor needed
O(n^2) memory and was gated at |G| <= 100 which created a discontinuity in the
score; 'combined' averaged incomparable scales.  Model types PLS and
"separate regressions" were never used by the score.

Search
------
* 'beam' (default): rules.ConditionGenerator + rules.BeamSearch over
  conjunctions of conditions on X (original units; quantile cut points for
  numeric attributes, '=' for categorical ones).
* 'clustering': k-means (or another clusterer from clustering.py) on
  X_space.numeric; every cluster is a candidate and gets a post-hoc rule from
  rules.describe_mask_with_tree.
* 'random': random index sets (baseline), rules post hoc.

`use_Y_as_descriptive=True` adds the response columns to the descriptive
space.  This is selection on the outcome: the reported p-values are then
conditional on the candidate set (stated in quality_measures['inference']).

Inference
---------
The permutation test uses the EMM statistic itself: rows of Y are permuted
(breaking every X-Y relation), the exceptionality of every candidate is
recomputed, and raw / Westfall-Young max-T / Benjamini-Hochberg p-values are
filled exactly like inference.validate_subgroups does.  With
`family_wise='search'` (default) the max-T step runs over every mask the
search evaluated, so the adjusted p-values account for the search itself and
not only for the returned candidates ('candidates' restricts the family to
the returned set).  The correlation statistic is evaluated for all masks at
once with vectorized group sums, so the larger family is cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sps

from .common import FeatureSpace, derive_seed, make_rng
from .clustering import ClusteringFactory
from .data_loader import DataLoader, Dataset
from .inference import permutation_test_candidates
from .rules import BeamSearch, ConditionGenerator, Rule, describe_mask_with_tree
from .core import QualityMeasures, SetOfSubgroups, Subgroup

logger = logging.getLogger("mrsd")


def _k_values(k_values, n_clusters, default):
    """Cluster counts to try: explicit list, else [n_clusters], else the default."""
    if k_values:
        return list(k_values)
    if n_clusters is not None:
        return [int(n_clusters)]
    return list(default)

_P_FLOOR = 1e-300


# =============================================================================
# ENUMS
# =============================================================================

class ExceptionalityMeasure(Enum):
    CORRELATION = "correlation"
    REGRESSION = "regression"
    R2_DIFFERENCE = "r2_difference"   # alias of REGRESSION with R^2 reporting


class SearchStrategy(Enum):
    BEAM = "beam"
    CLUSTERING = "clustering"
    RANDOM = "random"

    @classmethod
    def _missing_(cls, value):
        aliases = {"beam_search": cls.BEAM, "random_sampling": cls.RANDOM}
        if isinstance(value, str) and value in aliases:
            return aliases[value]
        return None


# =============================================================================
# STATISTICS
# =============================================================================

def _standardize_columns(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Column z-scores (population sd); constant columns become 0 and are flagged."""
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    ok = sd > 1e-12
    Z = np.zeros_like(A, dtype=float)
    Z[:, ok] = (A[:, ok] - mu[ok]) / sd[ok]
    return Z, ok


def correlation_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(d_X, d_Y) Pearson correlations; pairs involving a constant column are 0."""
    Zx, okx = _standardize_columns(X)
    Zy, oky = _standardize_columns(Y)
    R = Zx.T @ Zy / X.shape[0]
    R[~okx, :] = 0.0
    R[:, ~oky] = 0.0
    return R


def fisher_z_matrix(X: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Fisher-z differences of all pairwise correlations, G vs complement.
    Returns None when a group is too small (< 4 rows)."""
    n = X.shape[0]
    n_g = int(mask.sum())
    n_c = n - n_g
    if n_g < 4 or n_c < 4:
        return None
    r_g = np.clip(correlation_matrix(X[mask], Y[mask]), -0.999999, 0.999999)
    r_c = np.clip(correlation_matrix(X[~mask], Y[~mask]), -0.999999, 0.999999)
    se = np.sqrt(1.0 / (n_g - 3) + 1.0 / (n_c - 3))
    return (np.arctanh(r_g) - np.arctanh(r_c)) / se


def _rss_and_rank(A: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, int]:
    beta, _, rank, _ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ beta
    return (resid ** 2).sum(axis=0), int(rank)


def interaction_f_test(X: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """
    Per-response interaction F-test of Y_j ~ X * 1[G].

    Restricted model: [1, 1[G], X] (separate intercepts, common slopes).
    Unrestricted: separate regressions inside G and inside the complement.
    Returns dict(F, p, df1, df2, r2_subgroup, r2_complement) or None when a
    group has fewer rows than parameters + 2.
    """
    n, p = X.shape
    n_g = int(mask.sum())
    n_c = n - n_g
    if n_g < p + 3 or n_c < p + 3:
        return None
    ones = np.ones((n, 1))
    A_full = np.hstack([ones, X])
    rss_g, r_g = _rss_and_rank(A_full[mask], Y[mask])
    rss_c, r_c = _rss_and_rank(A_full[~mask], Y[~mask])
    A_res = np.hstack([ones, mask.astype(float)[:, None], X])
    rss_r, r_r = _rss_and_rank(A_res, Y)
    df1 = r_g + r_c - r_r
    df2 = n - r_g - r_c
    if df1 <= 0 or df2 <= 0:
        return None
    rss_u = rss_g + rss_c
    with np.errstate(divide="ignore", invalid="ignore"):
        F = ((rss_r - rss_u) / df1) / (rss_u / df2)
    F = np.where(np.isfinite(F), np.maximum(F, 0.0), np.inf)
    pv = np.where(np.isinf(F), _P_FLOOR, sps.f.sf(F, df1, df2))
    tss_g = ((Y[mask] - Y[mask].mean(axis=0)) ** 2).sum(axis=0)
    tss_c = ((Y[~mask] - Y[~mask].mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_g = np.where(tss_g > 0, 1.0 - rss_g / tss_g, 0.0)
        r2_c = np.where(tss_c > 0, 1.0 - rss_c / tss_c, 0.0)
    return {"F": F, "p": np.maximum(pv, _P_FLOOR), "df1": df1, "df2": df2,
            "r2_subgroup": r2_g, "r2_complement": r2_c}


def _sidak(p_min: float, m: int) -> float:
    """P(min of m independent p-values <= p_min)."""
    p_min = min(max(float(p_min), _P_FLOOR), 1.0)
    return float(-np.expm1(m * np.log1p(-p_min)))


class EMMStatistic:
    """
    Exceptionality of a mask on the chosen model class; `score` is -log10 of
    the asymptotic p-value (higher = more exceptional), `details` adds the
    quantities reported in Subgroup.quality_measures.
    """

    def __init__(self, X: np.ndarray, measure: ExceptionalityMeasure,
                 correlation_aggregation: str = "max_abs_z",
                 target_combination: str = "fisher",
                 x_names: Optional[Sequence[str]] = None,
                 y_names: Optional[Sequence[str]] = None):
        if correlation_aggregation not in ("max_abs_z", "mean_sq_z"):
            raise ValueError(f"correlation_aggregation must be 'max_abs_z' or 'mean_sq_z', got {correlation_aggregation!r}")
        if target_combination not in ("fisher", "max"):
            raise ValueError(f"target_combination must be 'fisher' or 'max', got {target_combination!r}")
        self.X = np.asarray(X, dtype=float)
        self.measure = measure
        self.correlation_aggregation = correlation_aggregation
        self.target_combination = target_combination
        self.x_names = list(x_names) if x_names is not None else [f"x{i}" for i in range(self.X.shape[1])]
        self.y_names = list(y_names) if y_names is not None else None

    # ---- correlation ------------------------------------------------------
    def _corr_pvalue(self, z: np.ndarray) -> Tuple[float, float]:
        m = z.size
        if m == 0:
            return 1.0, 0.0
        if self.correlation_aggregation == "max_abs_z":
            zmax = float(np.max(np.abs(z)))
            p_one = 2.0 * float(sps.norm.sf(zmax))
            return _sidak(p_one, m), zmax
        s = float((z ** 2).sum())
        return float(sps.chi2.sf(s, m)), s

    # ---- regression -------------------------------------------------------
    def _reg_pvalue(self, res: Dict[str, np.ndarray]) -> Tuple[float, float]:
        p = res["p"]
        if self.target_combination == "fisher":
            stat = float(-2.0 * np.log(p).sum())
            return float(sps.chi2.sf(stat, 2 * len(p))), stat
        return _sidak(float(p.min()), len(p)), float(res["F"].max())

    # ---- public -----------------------------------------------------------
    def score(self, Y: np.ndarray, mask: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool)
        if self.measure == ExceptionalityMeasure.CORRELATION:
            z = fisher_z_matrix(self.X, Y, mask)
            if z is None:
                return 0.0
            p, _ = self._corr_pvalue(z)
        else:
            res = interaction_f_test(self.X, Y, mask)
            if res is None:
                return 0.0
            p, _ = self._reg_pvalue(res)
        return float(-np.log10(max(p, _P_FLOOR)))

    def _regression_batch(self, Y: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Vectorized interaction F-test for many masks through group sums.
        Separate intercepts are equivalent to within-group centering
        (Frisch-Waugh), so every model is an OLS on centered cross-products.
        Returns -log10 p per mask (0 where a group is too small).
        """
        X = self.X
        n, p = X.shape
        q = Y.shape[1]
        XX = (X[:, :, None] * X[:, None, :]).reshape(n, p * p)
        XY = (X[:, :, None] * Y[:, None, :]).reshape(n, p * q)
        YY = Y * Y

        def centered(w: np.ndarray, ng: np.ndarray):
            sx, sy = w @ X, w @ Y
            cxx = (w @ XX).reshape(-1, p, p) - sx[:, :, None] * sx[:, None, :] / ng[:, None, None]
            cxy = (w @ XY).reshape(-1, p, q) - sx[:, :, None] * sy[:, None, :] / ng[:, None, None]
            cyy = w @ YY - sy ** 2 / ng[:, None]
            return cxx, cxy, cyy

        def rss_rank(cxx, cxy, cyy):
            rank = np.linalg.matrix_rank(cxx, tol=1e-9)
            beta = np.linalg.pinv(cxx, rcond=1e-9) @ cxy                 # (m, p, q)
            rss = cyy - np.einsum("mpq,mpq->mq", cxy, beta)
            return np.maximum(rss, 0.0), rank

        n_g = M.sum(axis=1)
        n_c = n - n_g
        out = np.zeros(M.shape[0])
        valid = (n_g >= p + 3) & (n_c >= p + 3)
        if not valid.any():
            return out
        Mg = M[valid]
        ng, nc = n_g[valid], n_c[valid]
        g = centered(Mg, ng)
        c = centered(1.0 - Mg, nc)
        rss_g, r_g = rss_rank(*g)
        rss_c, r_c = rss_rank(*c)
        rss_r, r_r = rss_rank(g[0] + c[0], g[1] + c[1], g[2] + c[2])
        r_g, r_c, r_r = r_g + 1, r_c + 1, r_r + 2                        # intercept terms
        df1 = r_g + r_c - r_r
        df2 = n - r_g - r_c
        rss_u = rss_g + rss_c
        ok = (df1 > 0) & (df2 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            F = ((rss_r - rss_u) / df1[:, None]) / (rss_u / df2[:, None])
        F = np.where(np.isfinite(F), np.maximum(F, 0.0), np.inf)
        pv = np.where(np.isinf(F), _P_FLOOR,
                      sps.f.sf(np.where(np.isinf(F), 0.0, F), np.maximum(df1, 1)[:, None], np.maximum(df2, 1)[:, None]))
        pv = np.maximum(pv, _P_FLOOR)
        if self.target_combination == "fisher":
            p = sps.chi2.sf(-2.0 * np.log(pv).sum(axis=1), 2 * q)
        else:
            p_min = np.minimum(np.maximum(pv.min(axis=1), _P_FLOOR), 1.0)
            p = -np.expm1(q * np.log1p(-p_min))
        res = np.where(ok, -np.log10(np.maximum(p, _P_FLOOR)), 0.0)
        out[valid] = res
        return out

    def batch(self, Y: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray:
        """Scores of many masks, fully vectorized for both measures."""
        if self.measure != ExceptionalityMeasure.CORRELATION:
            M = np.array([np.asarray(mk, dtype=bool) for mk in masks], dtype=float)
            return self._regression_batch(Y, M)
        X = self.X
        n, dx = X.shape
        dy = Y.shape[1]
        M = np.array([np.asarray(mk, dtype=bool) for mk in masks], dtype=float)   # (m, n)
        XY = (X[:, :, None] * Y[:, None, :]).reshape(n, dx * dy)

        def corr(Mg: np.ndarray, ng: np.ndarray) -> np.ndarray:
            mx, my = (Mg @ X) / ng[:, None], (Mg @ Y) / ng[:, None]
            vx = (Mg @ (X * X)) / ng[:, None] - mx ** 2
            vy = (Mg @ (Y * Y)) / ng[:, None] - my ** 2
            cxy = ((Mg @ XY) / ng[:, None]).reshape(-1, dx, dy) - mx[:, :, None] * my[:, None, :]
            denom = np.sqrt(np.maximum(vx, 0.0))[:, :, None] * np.sqrt(np.maximum(vy, 0.0))[:, None, :]
            ok = (vx > 1e-24)[:, :, None] & (vy > 1e-24)[:, None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(ok, cxy / denom, 0.0)
            return np.clip(r, -0.999999, 0.999999)

        n_g = M.sum(axis=1)
        n_c = n - n_g
        out = np.zeros(len(masks))
        valid = (n_g >= 4) & (n_c >= 4)
        if not valid.any():
            return out
        r_g = corr(M[valid], n_g[valid])
        r_c = corr(1.0 - M[valid], n_c[valid])
        se = np.sqrt(1.0 / (n_g[valid] - 3) + 1.0 / (n_c[valid] - 3))
        z = (np.arctanh(r_g) - np.arctanh(r_c)) / se[:, None, None]
        m_pairs = dx * dy
        if self.correlation_aggregation == "max_abs_z":
            zmax = np.abs(z).reshape(len(z), -1).max(axis=1)
            p_one = np.minimum(np.maximum(2.0 * sps.norm.sf(zmax), _P_FLOOR), 1.0)
            p = -np.expm1(m_pairs * np.log1p(-p_one))
        else:
            p = sps.chi2.sf((z ** 2).reshape(len(z), -1).sum(axis=1), m_pairs)
        out[valid] = -np.log10(np.maximum(p, _P_FLOOR))
        return out

    def details(self, Y: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
        """Score plus the human-readable effect behind it."""
        mask = np.asarray(mask, dtype=bool)
        y_names = self.y_names or [f"y{j}" for j in range(Y.shape[1])]
        out: Dict[str, object] = {"quality": 0.0, "p_asymptotic_emm": 1.0}
        if self.measure == ExceptionalityMeasure.CORRELATION:
            z = fisher_z_matrix(self.X, Y, mask)
            if z is None:
                return out
            p, stat = self._corr_pvalue(z)
            i, j = np.unravel_index(int(np.argmax(np.abs(z))), z.shape)
            r_g = correlation_matrix(self.X[mask], Y[mask])
            r_c = correlation_matrix(self.X[~mask], Y[~mask])
            out.update({
                "quality": float(-np.log10(max(p, _P_FLOOR))),
                "p_asymptotic_emm": float(p),
                "max_abs_z": float(np.max(np.abs(z))),
                "sum_sq_z": float((z ** 2).sum()),
                "best_pair": f"{self.x_names[i]}~{y_names[j]}",
                "r_subgroup": float(r_g[i, j]),
                "r_complement": float(r_c[i, j]),
            })
            return out
        res = interaction_f_test(self.X, Y, mask)
        if res is None:
            return out
        p, stat = self._reg_pvalue(res)
        j = int(np.argmin(res["p"]))
        out.update({
            "quality": float(-np.log10(max(p, _P_FLOOR))),
            "p_asymptotic_emm": float(p),
            "F_max": float(res["F"].max()),
            "best_target": y_names[j],
            "r2_subgroup": float(np.mean(res["r2_subgroup"])),
            "r2_complement": float(np.mean(res["r2_complement"])),
            "r2_difference": float(np.mean(res["r2_subgroup"] - res["r2_complement"])),
        })
        return out


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MRSDEMMConfig:
    """Configuration of MRSD-EMM."""

    # exceptionality statistic
    exceptionality_measure: str = "correlation"     # correlation | regression | r2_difference
    correlation_aggregation: str = "max_abs_z"      # max_abs_z | mean_sq_z
    target_combination: str = "fisher"              # fisher | max  (regression measures)

    # search
    search_strategy: str = "beam"                   # beam | clustering | random
    beam_width: int = 20
    max_depth: int = 3
    n_bins: int = 4
    clustering_algorithm: str = "kmeans"
    k_values: Optional[List[int]] = None            # clustering search; default [n_clusters] or [2, 3, 4, 5]
    n_clusters: Optional[int] = None
    n_random: int = 100                             # random search

    # subgroup constraints
    n_min: int = 10
    n_max: Optional[int] = None                     # default n // 2
    min_exceptionality: float = 0.0                 # minimum -log10 p to be reported
    top_k: int = 50                                 # after pruning

    # descriptive space
    use_Y_as_descriptive: bool = False              # selection on the outcome (warns)

    # post-processing and inference
    redundancy_threshold: float = 0.7
    n_permutations: int = 200
    family_wise: str = "search"                     # max-T over 'search' (all evaluated masks) or 'candidates'

    random_state: Optional[int] = None
    verbose: bool = False


# =============================================================================
# ALGORITHM
# =============================================================================

class MRSDEMM:
    """
    MRSD-EMM (Multiple Responses Subgroup Discovery, Exceptional Model Mining).

    fit(dataset) -> SetOfSubgroups whose members carry a rule in original
    units, quality_measures['quality'] (-log10 asymptotic p of the EMM
    statistic vs the complement) and permutation p-values.
    """

    ALGORITHM = "MRSD-EMM"

    def __init__(self, config: Optional[MRSDEMMConfig] = None):
        self.config = config or MRSDEMMConfig()
        self._statistic: Optional[EMMStatistic] = None
        self._family: List[np.ndarray] = []
        self.n_evaluated: int = 0

    # ---- helpers ----------------------------------------------------------
    def _descriptive_space(self, dataset: Dataset) -> FeatureSpace:
        if self.config.use_Y_as_descriptive:
            logger.warning("MRSD-EMM: use_Y_as_descriptive=True selects candidates using the responses; "
                           "p-values are conditional on the selected candidate set")
            return FeatureSpace.from_dataframe(pd.concat([dataset.X, dataset.Y], axis=1))
        return dataset.X_space

    def _beam(self, fs_desc: FeatureSpace, Y: np.ndarray, n_max: int) -> List[Tuple[np.ndarray, Optional[Rule], float, str]]:
        cfg = self.config
        gen = ConditionGenerator(fs_desc, n_bins=cfg.n_bins, min_support=cfg.n_min)
        bs = BeamSearch(gen, lambda mk: self._statistic.score(Y, mk), beam_width=cfg.beam_width,
                        max_depth=cfg.max_depth, n_min=cfg.n_min, n_max=n_max)
        results = bs.search()
        self.n_evaluated = bs.n_evaluated
        return [(r.mask, r.rule, r.quality, "beam") for r in results]

    def _clustering(self, fs_desc: FeatureSpace, Y: np.ndarray, rng: np.random.Generator, n_max: int):
        cfg = self.config
        data = fs_desc.numeric
        out = []
        for k in _k_values(cfg.k_values, cfg.n_clusters, [2, 3, 4, 5]):
            if k < 2:
                continue
            clusterer = ClusteringFactory.get_clusterer(cfg.clustering_algorithm)
            try:
                part = clusterer.fit(data, k=k, random_state=derive_seed(rng))
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("MRSD-EMM: clustering with k=%d failed: %s", k, exc)
                continue
            for c in range(part.k):
                mask = part.labels == c
                size = int(mask.sum())
                if cfg.n_min <= size <= n_max:
                    out.append((mask, None, self._statistic.score(Y, mask), f"cluster k={k} c={c}"))
        self.n_evaluated = len(out)
        return out

    def _random(self, n: int, Y: np.ndarray, rng: np.random.Generator, n_max: int):
        cfg = self.config
        out = []
        hi = max(cfg.n_min, n_max)
        for i in range(cfg.n_random):
            size = int(rng.integers(cfg.n_min, hi + 1))
            mask = np.zeros(n, dtype=bool)
            mask[rng.choice(n, size, replace=False)] = True
            out.append((mask, None, self._statistic.score(Y, mask), f"random {i}"))
        self.n_evaluated = len(out)
        return out

    # ---- main -------------------------------------------------------------
    def fit(self, dataset: Dataset) -> SetOfSubgroups:
        cfg = self.config
        rng = make_rng(cfg.random_state)
        measure = ExceptionalityMeasure(cfg.exceptionality_measure)
        strategy = SearchStrategy(cfg.search_strategy)

        fs_X, fs_Y = dataset.X_space, dataset.Y_space
        X, Y = fs_X.numeric, fs_Y.numeric
        n = fs_X.n
        n_max = cfg.n_max if cfg.n_max is not None else n // 2
        self._statistic = EMMStatistic(X, measure, cfg.correlation_aggregation, cfg.target_combination,
                                       x_names=fs_X.numeric_names, y_names=fs_Y.numeric_names)
        fs_desc = self._descriptive_space(dataset)

        if cfg.verbose:
            print(f"{self.ALGORITHM}: n={n}, measure={measure.value}, search={strategy.value}")

        if strategy == SearchStrategy.BEAM:
            cands = self._beam(fs_desc, Y, n_max)
        elif strategy == SearchStrategy.CLUSTERING:
            cands = self._clustering(fs_desc, Y, rng, n_max)
        else:
            cands = self._random(n, Y, rng, n_max)
        if cfg.verbose:
            print(f"  {len(cands)} candidates, {self.n_evaluated} evaluations")
        self._family = [c[0] for c in cands]          # every mask the search evaluated

        subgroups: List[Subgroup] = []
        for mask, rule, q, origin in cands:
            if q < cfg.min_exceptionality:
                continue
            details = self._statistic.details(Y, mask)
            qm: Dict[str, object] = {"exceptionality_measure": measure.value, "search_origin": origin}
            qm.update(details)
            if rule is not None:
                sg = Subgroup(description=str(rule), covered_instances=set(np.flatnonzero(mask).tolist()),
                              quality_measures=qm, algorithm=self.ALGORITHM, rule=rule)
                sg.description_precision = 1.0
                sg.description_recall = 1.0
            else:
                desc = describe_mask_with_tree(fs_X, mask, random_state=derive_seed(rng))
                sg = Subgroup(description=f"EMM {origin}: {desc.rule}", covered_instances=set(np.flatnonzero(mask).tolist()),
                              quality_measures=qm, algorithm=self.ALGORITHM, rule=desc.rule,
                              description_precision=desc.precision, description_recall=desc.recall)
            sg.fill_target_statistics(dataset.Y)
            subgroups.append(sg)

        result = SetOfSubgroups(subgroups).prune(QualityMeasures.quality, s_min=cfg.redundancy_threshold)
        result = result.top_k(QualityMeasures.quality, cfg.top_k)
        if cfg.n_permutations > 0 and len(result) > 0:
            self._validate(result, Y, rng)
        if cfg.verbose:
            print(result.summary())
        return result

    # ---- inference --------------------------------------------------------
    def _validate(self, result: SetOfSubgroups, Y: np.ndarray, rng: np.random.Generator) -> None:
        """Permute rows of Y, recompute the EMM statistic for every candidate
        (and, with family_wise='search', for every other mask the search
        evaluated, so that the max-T step accounts for the search); fill
        p_value (raw), p_value_corrected (Westfall-Young max-T), p_bh."""
        cfg = self.config
        sgs = list(result)
        n = Y.shape[0]
        cand = [sg.mask(n) for sg in sgs]
        cand_keys = {np.packbits(m).tobytes() for m in cand}
        extra: List[np.ndarray] = []
        if cfg.family_wise == "search":
            seen = set(cand_keys)
            for mk in self._family:
                key = np.packbits(mk).tobytes()
                if key not in seen:
                    seen.add(key)
                    extra.append(mk)
        elif cfg.family_wise != "candidates":
            raise ValueError(f"family_wise must be 'search' or 'candidates', got {cfg.family_wise!r}")
        masks = cand + extra
        stat = self._statistic
        res = permutation_test_candidates(masks, stat.score, Y, n_permutations=cfg.n_permutations,
                                          random_state=derive_seed(rng), batch_fn=stat.batch)
        note = "candidates selected using Y; p-values conditional on candidate set" if cfg.use_Y_as_descriptive else ""
        m = len(sgs)
        for sg, obs, pr, pm, pb in zip(sgs, res.observed[:m], res.p_raw[:m], res.p_adj[:m], res.p_bh[:m]):
            sg.p_value = float(pr)
            sg.p_value_corrected = float(pm)
            sg.correction_method = "westfall_young_maxt"
            sg.quality_measures["test_statistic"] = float(obs)
            sg.quality_measures["p_bh"] = float(pb)
            sg.quality_measures["inference"] = (f"emm_{cfg.exceptionality_measure}; permuted=Y; B={cfg.n_permutations}; "
                                                f"family={len(masks)} ({cfg.family_wise})" + (f"; {note}" if note else ""))


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_mrsdemm(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
                **config_kwargs) -> SetOfSubgroups:
    """Run MRSD-EMM on a DataFrame or a file path."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    return MRSDEMM(MRSDEMMConfig(**config_kwargs)).fit(dataset)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 300
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    planted = x2 > 0.5
    y1 = x1 * np.where(planted, 2.0, 0.0) + rng.normal(size=n)
    y2 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y1": y1, "y2": y2})
    for measure in ("correlation", "regression"):
        res = run_mrsdemm(df, ["x1", "x2", "x3"], ["y1", "y2"], exceptionality_measure=measure,
                          n_permutations=100, random_state=0)
        best = res.top_k(QualityMeasures.quality, 1)[0]
        bm = best.mask(n)
        jac = (bm & planted).sum() / (bm | planted).sum()
        print(f"{measure}: best rule '{best.rule}' n={best.coverage} quality={best.quality:.2f} "
              f"p_adj={best.p_value_corrected:.3f} Jaccard with planted={jac:.2f} -> "
              f"{'recovered' if jac >= 0.5 and best.p_value_corrected < 0.05 else 'not recovered'}")
