"""
MRSD-CCA: Multiple Responses Subgroup Discovery with Canonical Correlation
Analysis.

Idea
----
Regularized CCA on (X_space.numeric, Y_space.numeric) finds the directions of
shared structure between the descriptive variables and the responses.
Clustering the instances in the canonical space yields candidate subgroups
in which the X-Y relationship is coherent; each candidate is scored by how
much stronger (or weaker) the first canonical correlation is inside the
subgroup than in its complement.

CCA
---
Solved with scipy, no sklearn CCA: with centred X, Y and covariances
S_XX, S_YY, S_XY, the whitening matrices (S_XX + lambda I)^{-1/2} and
(S_YY + lambda I)^{-1/2} are formed from eigen decompositions (rank-deficient
directions, e.g. one-hot blocks, are dropped), and the canonical weights come
from the SVD of the whitened cross-covariance.  `regularization` is the ridge
lambda; 0.0 (default) uses a 1e-6 ridge for numerical stability.  The former
"regularization = add random noise" was not regularization and made results
non-deterministic.

Clustering
----------
k-means (or agglomerative) directly on the canonical scores weighted by their
canonical correlation (`projection_space`: 'x_scores' U (default), 'y_scores'
V, or 'combined' [U V]), for every k in `k_values` (default [2, 3, 4]).
The weighted scores are passed to the clusterer as they are (not
re-standardized), so components with small canonical correlation matter less.

Quality
-------
With the population canonical weights fixed, rho_G = corr(U_1, V_1) inside G
and rho_C the same inside the complement; the Fisher-z difference
    z = (atanh rho_G - atanh rho_C) / sqrt(1/(n_G - 3) + 1/(n_C - 3))
is reported and quality = |z| (normalized by its sampling error, so it does
not favour small clusters).  The range-restricted average of canonical
correlations and the contrastive CCA refit of the earlier version were
removed (the refit overfits at small n and had no sampling distribution).
Optionally (`use_classification_quality=True`) out-of-fold q_X / q_Y from
classifiers.DualSpaceEvaluator are reported as well; they are off by default
because clusters in canonical space are near-tautologically separable.

Inference
---------
Candidates are fixed index sets; rows of Y are permuted (with the population
weights the permuted V is just a row permutation of V), |z| is recomputed
per candidate and raw / Westfall-Young max-T / Benjamini-Hochberg p-values
are filled; with `family_wise='search'` (default) the max-T step runs over
every cluster that was evaluated (all k), not only over the pruned result.
Candidates were selected using both views (the canonical weights use Y; with
'y_scores' / 'combined' the clustering itself does), so the p-values are
conditional on the candidate set (stated in quality_measures['inference']).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.cluster import AgglomerativeClustering, KMeans

from .classifiers import DualSpaceEvaluator
from .common import derive_seed, make_rng
from .data_loader import DataLoader, Dataset
from .inference import permutation_test_candidates
from .rules import describe_mask_with_tree
from .core import QualityMeasures, SetOfSubgroups, Subgroup

logger = logging.getLogger("mrsd")


def _k_values(k_values, n_clusters, default):
    """Cluster counts to try: explicit list, else [n_clusters], else the default."""
    if k_values:
        return list(k_values)
    if n_clusters is not None:
        return [int(n_clusters)]
    return list(default)


class CCAProjectionSpace(Enum):
    X_SCORES = "x_scores"
    Y_SCORES = "y_scores"
    COMBINED = "combined"


# =============================================================================
# REGULARIZED CCA
# =============================================================================

def _inv_sqrt(S: np.ndarray, ridge: float, rel_tol: float = 1e-10) -> np.ndarray:
    """(S + ridge I)^{-1/2} through eigh; directions with negligible variance are dropped."""
    w, V = linalg.eigh(S + ridge * np.eye(S.shape[0]))
    keep = w > max(w.max(), 0.0) * rel_tol
    if not keep.any():
        return np.zeros_like(S)
    return (V[:, keep] / np.sqrt(w[keep])) @ V[:, keep].T


class RegularizedCCA:
    """Regularized CCA: weights, canonical correlations and scores."""

    def __init__(self, n_components: int = 2, regularization: float = 0.0):
        if regularization < 0:
            raise ValueError("regularization must be >= 0")
        self.n_components = n_components
        self.regularization = regularization
        self.x_weights_: Optional[np.ndarray] = None
        self.y_weights_: Optional[np.ndarray] = None
        self.canonical_correlations_: Optional[np.ndarray] = None
        self.x_mean_: Optional[np.ndarray] = None
        self.y_mean_: Optional[np.ndarray] = None

    @property
    def fitted(self) -> bool:
        return self.x_weights_ is not None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RegularizedCCA":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n = X.shape[0]
        if n < 4 or X.shape[1] == 0 or Y.shape[1] == 0:
            raise ValueError("CCA needs at least 4 rows and one column per view")
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = Y.mean(axis=0)
        Xc = X - self.x_mean_
        Yc = Y - self.y_mean_
        ridge = self.regularization if self.regularization > 0 else 1e-6
        Sxx = Xc.T @ Xc / (n - 1)
        Syy = Yc.T @ Yc / (n - 1)
        Sxy = Xc.T @ Yc / (n - 1)
        Wx = _inv_sqrt(Sxx, ridge)
        Wy = _inv_sqrt(Syy, ridge)
        U, s, Vt = linalg.svd(Wx @ Sxy @ Wy, full_matrices=False)
        k = int(min(self.n_components, (s > 1e-12).sum()))
        if k < 1:
            raise ValueError("CCA found no non-zero canonical correlation")
        self.x_weights_ = Wx @ U[:, :k]
        self.y_weights_ = Wy @ Vt[:k].T
        self.canonical_correlations_ = np.clip(s[:k], 0.0, 1.0)
        return self

    def transform(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("CCA not fitted")
        return (X - self.x_mean_) @ self.x_weights_, (Y - self.y_mean_) @ self.y_weights_


# =============================================================================
# FISHER-Z STATISTIC ON THE FIRST CANONICAL PAIR
# =============================================================================

def _corr_in(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> float:
    a, b = u[mask], v[mask]
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.clip(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb), -0.999999, 0.999999))


def canonical_fisher_z(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """(z, rho_G, rho_C): Fisher-z difference of corr(u, v) inside mask vs its complement."""
    n = len(u)
    n_g = int(mask.sum())
    n_c = n - n_g
    if n_g < 4 or n_c < 4:
        return 0.0, 0.0, 0.0
    r_g = _corr_in(u, v, mask)
    r_c = _corr_in(u, v, ~mask)
    z = (np.arctanh(r_g) - np.arctanh(r_c)) / np.sqrt(1.0 / (n_g - 3) + 1.0 / (n_c - 3))
    return float(z), r_g, r_c


def canonical_fisher_z_batch(u: np.ndarray, v: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray:
    """|z| for many masks, vectorized through group sums."""
    n = len(u)
    M = np.array([mk.astype(float) for mk in masks])         # (m, n)
    n_g = M.sum(axis=1)
    n_c = n - n_g
    out = np.zeros(len(masks))
    valid = (n_g >= 4) & (n_c >= 4)
    if not valid.any():
        return out

    def corr(Mg, ng):
        su, sv = Mg @ u, Mg @ v
        suu, svv, suv = Mg @ (u * u), Mg @ (v * v), Mg @ (u * v)
        var_u = suu / ng - (su / ng) ** 2
        var_v = svv / ng - (sv / ng) ** 2
        cov = suv / ng - (su / ng) * (sv / ng)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((var_u > 1e-24) & (var_v > 1e-24), cov / np.sqrt(var_u * var_v), 0.0)
        return np.clip(r, -0.999999, 0.999999)

    r_g = corr(M[valid], n_g[valid])
    r_c = corr(1.0 - M[valid], n_c[valid])
    se = np.sqrt(1.0 / (n_g[valid] - 3) + 1.0 / (n_c[valid] - 3))
    out[valid] = np.abs(np.arctanh(r_g) - np.arctanh(r_c)) / se
    return out


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MRSDCCAConfig:
    """Configuration of MRSD-CCA."""

    # CCA
    n_components: int = 2
    regularization: float = 0.0            # ridge lambda; 0 -> 1e-6 for stability
    projection_space: str = "x_scores"     # x_scores | y_scores | combined

    # clustering in canonical space
    clustering_algorithm: str = "kmeans"   # kmeans | agglomerative
    k_values: Optional[List[int]] = None   # default [n_clusters] or [2, 3, 4]
    n_clusters: Optional[int] = None

    # optional OOF classification quality (reported, not used in the score)
    use_classification_quality: bool = False
    classifier_X: str = "random_forest"
    classifier_Y: str = "random_forest"
    cv_folds: int = 5

    # subgroup constraints
    n_min: int = 10
    n_max: Optional[int] = None            # default n // 2

    # post-processing and inference
    redundancy_threshold: float = 0.7
    n_permutations: int = 200
    family_wise: str = "search"            # max-T over 'search' (all evaluated clusters) or 'candidates'

    random_state: Optional[int] = None
    verbose: bool = False


# =============================================================================
# ALGORITHM
# =============================================================================

class MRSDCCA:
    """MRSD-CCA (Multiple Responses Subgroup Discovery, Canonical Correlation Analysis)."""

    ALGORITHM = "MRSD-CCA"

    def __init__(self, config: Optional[MRSDCCAConfig] = None):
        self.config = config or MRSDCCAConfig()
        self.cca_: Optional[RegularizedCCA] = None
        self._family: List[np.ndarray] = []

    def _cluster(self, Z: np.ndarray, k: int, seed: int) -> np.ndarray:
        if self.config.clustering_algorithm == "kmeans":
            return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        if self.config.clustering_algorithm == "agglomerative":
            return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
        raise ValueError(f"clustering_algorithm must be 'kmeans' or 'agglomerative', got {self.config.clustering_algorithm!r}")

    def fit(self, dataset: Dataset) -> SetOfSubgroups:
        cfg = self.config
        rng = make_rng(cfg.random_state)
        projection = CCAProjectionSpace(cfg.projection_space)
        fs_X, fs_Y = dataset.X_space, dataset.Y_space
        X, Y = fs_X.numeric, fs_Y.numeric
        n = fs_X.n
        n_max = cfg.n_max if cfg.n_max is not None else n // 2

        self.cca_ = RegularizedCCA(cfg.n_components, cfg.regularization)
        try:
            self.cca_.fit(X, Y)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("MRSD-CCA: CCA could not be fitted: %s", exc)
            return SetOfSubgroups([])
        U, V = self.cca_.transform(X, Y)
        rho = self.cca_.canonical_correlations_
        if cfg.verbose:
            print(f"{self.ALGORITHM}: n={n}, canonical correlations={np.round(rho, 3)}")

        if projection == CCAProjectionSpace.X_SCORES:
            Z = U * rho
        elif projection == CCAProjectionSpace.Y_SCORES:
            Z = V * rho
        else:
            Z = np.hstack([U * rho, V * rho])

        u1, v1 = U[:, 0], V[:, 0]
        evaluator = None
        if cfg.use_classification_quality:
            evaluator = DualSpaceEvaluator(X, Y, cfg.classifier_X, cfg.classifier_Y, n_folds=cfg.cv_folds,
                                           aggregation="min", random_state=derive_seed(rng), fast=True)

        subgroups: List[Subgroup] = []
        self._family = []
        for k in _k_values(cfg.k_values, cfg.n_clusters, [2, 3, 4]):
            if k < 2 or k > n // 2:
                continue
            try:
                labels = self._cluster(Z, k, derive_seed(rng))
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("MRSD-CCA: clustering with k=%d failed: %s", k, exc)
                continue
            for c in np.unique(labels):
                mask = labels == c
                size = int(mask.sum())
                if not (cfg.n_min <= size <= n_max):
                    continue
                self._family.append(mask)
                z, r_g, r_c = canonical_fisher_z(u1, v1, mask)
                qm: Dict[str, object] = {
                    "quality": abs(z), "fisher_z": z, "rho_subgroup": r_g, "rho_complement": r_c,
                    "rho_population": float(rho[0]), "k": int(k), "cluster": int(c),
                }
                if evaluator is not None:
                    ev = evaluator.evaluate(mask)
                    qm.update({"q_X": ev.q_X, "q_Y": ev.q_Y})
                d = describe_mask_with_tree(fs_X, mask, random_state=derive_seed(rng))
                sg = Subgroup(description=f"CCA cluster k={k} c={c}: {d.rule}",
                              covered_instances=set(np.flatnonzero(mask).tolist()), quality_measures=qm,
                              algorithm=self.ALGORITHM, rule=d.rule, description_precision=d.precision,
                              description_recall=d.recall, k=int(k), l=int(c), clustering_X=cfg.clustering_algorithm,
                              clustering_Y="cca")
                sg.fill_target_statistics(dataset.Y)
                subgroups.append(sg)

        result = SetOfSubgroups(subgroups).prune(QualityMeasures.quality, s_min=cfg.redundancy_threshold)
        if cfg.n_permutations > 0 and len(result) > 0:
            self._validate(result, u1, v1, rng)
        if cfg.verbose:
            print(result.summary())
        return result

    def _validate(self, result: SetOfSubgroups, u1: np.ndarray, v1: np.ndarray, rng: np.random.Generator) -> None:
        """Permute rows of Y (= rows of V under fixed weights) and recompute |z|
        per candidate; with family_wise='search' the max-T step runs over every
        cluster that was evaluated, not only over the returned ones."""
        cfg = self.config
        sgs = list(result)
        n = len(u1)
        cand = [sg.mask(n) for sg in sgs]
        seen = {np.packbits(m).tobytes() for m in cand}
        extra: List[np.ndarray] = []
        if cfg.family_wise == "search":
            for mk in self._family:
                key = np.packbits(mk).tobytes()
                if key not in seen:
                    seen.add(key)
                    extra.append(mk)
        elif cfg.family_wise != "candidates":
            raise ValueError(f"family_wise must be 'search' or 'candidates', got {cfg.family_wise!r}")
        masks = cand + extra
        data = v1[:, None]

        def stat(Vd, mk):
            return abs(canonical_fisher_z(u1, Vd[:, 0], mk)[0])

        def batch(Vd, mks):
            return canonical_fisher_z_batch(u1, Vd[:, 0], mks)

        res = permutation_test_candidates(masks, stat, data, n_permutations=cfg.n_permutations,
                                          random_state=derive_seed(rng), batch_fn=batch)
        note = "candidates selected using both views; p-values conditional on candidate set"
        m = len(sgs)
        for sg, obs, pr, pm, pb in zip(sgs, res.observed[:m], res.p_raw[:m], res.p_adj[:m], res.p_bh[:m]):
            sg.p_value = float(pr)
            sg.p_value_corrected = float(pm)
            sg.correction_method = "westfall_young_maxt"
            sg.quality_measures["test_statistic"] = float(obs)
            sg.quality_measures["p_bh"] = float(pb)
            sg.quality_measures["inference"] = (f"cca_fisher_z; permuted=Y; B={cfg.n_permutations}; "
                                                f"family={len(masks)} ({cfg.family_wise}); {note}")

    def get_canonical_correlations(self) -> List[float]:
        if self.cca_ is None or not self.cca_.fitted:
            return []
        return [float(r) for r in self.cca_.canonical_correlations_]


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_mrsdcca(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
                **config_kwargs) -> SetOfSubgroups:
    """Run MRSD-CCA on a DataFrame or a file path."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    return MRSDCCA(MRSDCCAConfig(**config_kwargs)).fit(dataset)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    planted = x1 > 0.3                           # a region of canonical (X-score) space ...
    y1 = np.where(planted, 2.0 * x1, 0.0) + rng.normal(size=n)   # ... in which X and Y are coupled
    y2 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
    res = run_mrsdcca(df, ["x1", "x2"], ["y1", "y2"], n_permutations=200, random_state=0)
    best = res.top_k(QualityMeasures.quality, 1)[0] if len(res) else None
    if best is None:
        print("no subgroup returned")
    else:
        bm = best.mask(n)
        jac = (bm & planted).sum() / (bm | planted).sum()
        print(f"best: {best.description} n={best.coverage} |z|={best.quality:.2f} rho_G={best.quality_measures['rho_subgroup']:.2f} "
              f"rho_C={best.quality_measures['rho_complement']:.2f} p_adj={best.p_value_corrected:.3f} Jaccard={jac:.2f} -> "
              f"{'recovered' if jac >= 0.5 and best.p_value_corrected < 0.05 else 'not recovered'}")
