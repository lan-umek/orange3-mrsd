"""
MRSD-MI: Multiple Responses Subgroup Discovery with Mutual Information.

Question answered
-----------------
Is the statistical dependence between the descriptive variables X and the
responses Y inside a subgroup G stronger than expected under independence,
and how does it compare with the complement D \\ G?

Mutual information estimators (all on X_space.numeric / Y_space.numeric)
-----------------------------------------------------------------------
* 'knn': Kraskov-Stoegbauer-Grassberger estimator (algorithm 1).
  Nearest neighbours are found with the max-norm in the joint space, the
  marginal counts use the same radius with the max-norm (strictly inside),
  a tiny jitter (1e-10 * sd, drawn from the algorithm's rng) breaks ties, and
  the marginal counts are obtained in one vectorized query_ball_point call.
* 'mixed' (default): the estimator of Gao, Kannan, Oh & Viswanath (2017) for
  mixtures of continuous and discrete variables.  X and Y are used with one
  column per variable (mixed_matrix: numeric standardized, categorical as
  level codes far apart); coinciding instances are counted explicitly instead
  of being jittered, so the estimate is consistent for numeric, categorical
  and mixed X / Y.  'pairwise' is then per pair of original variables.
* 'binning': np.histogram2d with n_bins = max(2, round(n ** (1/3))) per axis
  and the Miller-Madow bias correction (m_x + m_y - m_xy - 1) / (2n).
  For the joint (multivariate) MI each block is projected on its first
  principal component before binning.

MI aggregation (`mi_method`): 'joint' (default) I(X; Y) on the full vectors,
'pairwise' I(X_i; Y_j) aggregated with `aggregation` (mean / sum / max / min /
frobenius), or 'combined' (average of both).

Normalized measure: Linfoot's informational coefficient of correlation
1 - exp(-2 MI) in [0, 1] is reported as quality_measures['linfoot_subgroup']
and ['linfoot_complement'] (it equals |r| for a bivariate Gaussian).  The
earlier entropy-normalized NMI was dropped: its Kozachenko-Leonenko entropies
used an inconsistent volume and clipped at 0.

Search score (size-unbiased)
---------------------------
quality(G) = (MI_G - mean_null_G) / sd_null_G, where the null distribution for
candidate G is obtained by `n_null_shuffles` shuffles of the Y rows *within G*
(X of G and the Y point set of G are kept, only the pairing is destroyed).
The within-G null removes the dependence of the raw MI estimate on |G| and on
the local X/Y marginals.  The reported effect compares G with its complement
(quality_measures['mi_subgroup'], ['mi_complement'], ['mi_difference']).

Search: rules.BeamSearch over conjunctive rules on X (original units) or
k-means clustering on X_space.numeric with post-hoc rules
(rules.describe_mask_with_tree).

Inference
---------
Rows of Y are permuted (global independence) and (MI_G - mean_null_G) /
sd_null_G is recomputed, with mean_null_G and sd_null_G fixed at the values
estimated during the search: the KSG null under independence depends on n_G,
k and the dimensions, not on the pairing, so the within-G shuffle null and
the global-permutation null restricted to G coincide, and a common z-scale
across masks is what the Westfall-Young max-T step needs.  With
`family_wise='search'` (default) the maximum is taken over every mask the
search evaluated, so the adjusted p-values account for the search itself and
not only for the returned candidates ('candidates' restricts the family to
the returned set, as inference.validate_subgroups does).  Raw / max-T /
Benjamini-Hochberg p-values are filled like validate_subgroups does.  MI is
costly, so B defaults to 100.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.special import digamma

from .common import derive_seed, make_rng
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


# =============================================================================
# ENUMS
# =============================================================================

class MIMethod(Enum):
    PAIRWISE = "pairwise"
    JOINT = "joint"
    COMBINED = "combined"


class MIAggregation(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    FROBENIUS = "frobenius"


class MIEstimator(Enum):
    MIXED = "mixed"      # Gao et al. (2017): consistent for continuous, discrete and mixed variables
    KNN = "knn"
    BINNING = "binning"


class SearchMethod(Enum):
    BEAM = "beam"
    CLUSTERING = "clustering"

    @classmethod
    def _missing_(cls, value):
        if value == "beam_search":
            return cls.BEAM
        return None


# =============================================================================
# ESTIMATORS
# =============================================================================

def _jitter(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sd = A.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return A + rng.normal(size=A.shape) * (1e-10 * sd)


def ksg_mi(X: np.ndarray, Y: np.ndarray, k: int = 5, tree_X: Optional[KDTree] = None,
           tree_Y: Optional[KDTree] = None) -> float:
    """
    KSG (algorithm 1) estimate of I(X; Y) in nats with the max-norm.

    X, Y must already be jittered (ties broken).  Marginal trees may be
    passed when several pairings of the same point sets are evaluated.
    The estimate is not clipped at 0 (clipping would distort null values).
    """
    n = X.shape[0]
    k = int(min(k, n - 1))
    if n < 4 or k < 1:
        return 0.0
    Z = np.hstack([X, Y])
    dist, _ = KDTree(Z).query(Z, k=k + 1, p=np.inf)
    eps = dist[:, -1]
    r = np.nextafter(eps, 0.0)                     # strictly inside the k-th neighbour ball
    tx = tree_X if tree_X is not None else KDTree(X)
    ty = tree_Y if tree_Y is not None else KDTree(Y)
    n_x = tx.query_ball_point(X, r, p=np.inf, return_length=True) - 1
    n_y = ty.query_ball_point(Y, r, p=np.inf, return_length=True) - 1
    return float(digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n))


CATEGORY_GAP = 1e6     # distance between two different levels of a categorical variable


def mixed_matrix(fs) -> np.ndarray:
    """One column per variable for the mixed estimator: numeric variables
    standardized, categorical variables as level codes times CATEGORY_GAP, so
    that (max-norm) two instances are at distance 0 in that coordinate when
    the levels agree and far apart otherwise.  No one-hot coding, no jitter."""
    codes = np.asarray(fs.codes, dtype=float)
    out = np.zeros_like(codes)
    for j in range(codes.shape[1]):
        col = codes[:, j]
        if fs.is_categorical[j]:
            out[:, j] = np.where(np.isnan(col), -1.0, col) * CATEGORY_GAP
        else:
            mu, sd = np.nanmean(col), np.nanstd(col)
            out[:, j] = np.nan_to_num((col - mu) / (sd if sd > 0 else 1.0))
    return out


def mixed_ksg_mi(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """
    Estimate of I(X; Y) in nats for continuous, discrete or mixed variables
    (Gao, Kannan, Oh & Viswanath, NIPS 2017, "Estimating mutual information for
    discrete-continuous mixtures").

    For every instance the distance rho to its k-th nearest neighbour in the
    joint space (max-norm) is found.  If rho > 0 the instance is treated as in
    KSG: k_i = k and n_x, n_y count the instances within rho in each marginal.
    If rho = 0 (at least k other instances coincide with it, as happens with
    categorical variables) k_i is the number of coinciding instances in the
    joint space and n_x, n_y the numbers coinciding in each marginal.
        I = mean( digamma(k_i) + log n - digamma(n_x) - digamma(n_y) ),
    counts including the instance itself.  Not clipped at 0.
    """
    n = X.shape[0]
    k = int(min(k, n - 1))
    if n < 4 or k < 1:
        return 0.0
    Z = np.hstack([X, Y])
    tz, tx, ty = KDTree(Z), KDTree(X), KDTree(Y)
    rho = tz.query(Z, k=k + 1, p=np.inf)[0][:, -1]
    tied = rho <= 0
    r = np.where(tied, 1e-12, np.nextafter(rho, 0.0))
    kp = np.full(n, float(k))
    if tied.any():
        kp[tied] = tz.query_ball_point(Z[tied], 1e-12, p=np.inf, return_length=True) - 1
    n_x = tx.query_ball_point(X, r, p=np.inf, return_length=True)
    n_y = ty.query_ball_point(Y, r, p=np.inf, return_length=True)
    kp = np.maximum(kp, 1.0)
    return float(np.mean(digamma(kp) + np.log(n) - digamma(n_x) - digamma(n_y)))


def binning_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in MI from np.histogram2d with Miller-Madow bias correction (nats)."""
    n = len(x)
    if n < 4:
        return 0.0
    n_bins = max(2, int(round(n ** (1.0 / 3.0))))
    H, _, _ = np.histogram2d(x, y, bins=n_bins)
    P = H / n
    px = P.sum(axis=1, keepdims=True)
    py = P.sum(axis=0, keepdims=True)
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / (px @ py)[nz])).sum())
    m_xy = int(nz.sum())
    m_x = int((px > 0).sum())
    m_y = int((py > 0).sum())
    return mi + (m_x + m_y - m_xy - 1) / (2.0 * n)


def _first_pc(A: np.ndarray) -> np.ndarray:
    if A.shape[1] == 1:
        return A[:, 0]
    Ac = A - A.mean(axis=0)
    _, _, vt = np.linalg.svd(Ac, full_matrices=False)
    return Ac @ vt[0]


def linfoot(mi: float) -> float:
    """Linfoot's informational coefficient of correlation, 1 - exp(-2 MI) in [0, 1]."""
    return float(1.0 - np.exp(-2.0 * max(mi, 0.0)))


class MutualInformation:
    """MI of a (X, Y) pairing with the configured estimator and aggregation."""

    def __init__(self, estimator: MIEstimator, method: MIMethod, aggregation: MIAggregation, k_neighbors: int = 5):
        self.estimator = estimator
        self.method = method
        self.aggregation = aggregation
        self.k = k_neighbors

    def _aggregate(self, M: np.ndarray) -> float:
        a = self.aggregation
        if a == MIAggregation.MEAN:
            return float(M.mean())
        if a == MIAggregation.SUM:
            return float(M.sum())
        if a == MIAggregation.MAX:
            return float(M.max())
        if a == MIAggregation.MIN:
            return float(M.min())
        return float(np.sqrt((M ** 2).sum()))

    def _joint(self, X: np.ndarray, Y: np.ndarray, trees) -> float:
        if self.estimator == MIEstimator.MIXED:
            return mixed_ksg_mi(X, Y, self.k)
        if self.estimator == MIEstimator.KNN:
            return ksg_mi(X, Y, self.k, trees[0] if trees else None, trees[1] if trees else None)
        return binning_mi(_first_pc(X), _first_pc(Y))

    def _pairwise(self, X: np.ndarray, Y: np.ndarray, trees) -> float:
        dx, dy = X.shape[1], Y.shape[1]
        M = np.zeros((dx, dy))
        for i in range(dx):
            for j in range(dy):
                if self.estimator == MIEstimator.MIXED:
                    M[i, j] = mixed_ksg_mi(X[:, i:i + 1], Y[:, j:j + 1], self.k)
                elif self.estimator == MIEstimator.KNN:
                    tx = trees[2][i] if trees and trees[2] is not None else None
                    ty = trees[3][j] if trees and trees[3] is not None else None
                    M[i, j] = ksg_mi(X[:, i:i + 1], Y[:, j:j + 1], self.k, tx, ty)
                else:
                    M[i, j] = binning_mi(X[:, i], Y[:, j])
        return self._aggregate(M)

    def trees(self, X: np.ndarray, Y: np.ndarray):
        """Marginal KD-trees reused across pairings of the same point sets."""
        if self.estimator != MIEstimator.KNN:
            return None
        joint = (KDTree(X), KDTree(Y)) if self.method in (MIMethod.JOINT, MIMethod.COMBINED) else (None, None)
        if self.method in (MIMethod.PAIRWISE, MIMethod.COMBINED):
            tx = [KDTree(X[:, i:i + 1]) for i in range(X.shape[1])]
            ty = [KDTree(Y[:, j:j + 1]) for j in range(Y.shape[1])]
        else:
            tx, ty = None, None
        return (joint[0], joint[1], tx, ty)

    def value(self, X: np.ndarray, Y: np.ndarray, trees=None) -> float:
        if self.method == MIMethod.JOINT:
            return self._joint(X, Y, trees)
        if self.method == MIMethod.PAIRWISE:
            return self._pairwise(X, Y, trees)
        return 0.5 * (self._joint(X, Y, trees) + self._pairwise(X, Y, trees))


class MIExceptionality:
    """
    Within-G shuffle-standardized MI:
        z(G) = (MI_G - mean_null_G) / sd_null_G
    with the null from `n_null` shuffles of Y rows inside G.

    Every evaluated mask is recorded (`family`) together with its null mean /
    sd so that the permutation test can take the maximum over the whole
    family of hypotheses the search looked at.
    """

    def __init__(self, X: np.ndarray, mi: MutualInformation, n_null: int, rng: np.random.Generator):
        if n_null < 2:
            raise ValueError("n_null_shuffles must be >= 2")
        self.X = X
        self.mi = mi
        self.n_null = n_null
        self.rng = rng
        self.family: Dict[bytes, Tuple[np.ndarray, float, float, float, float]] = {}

    @property
    def n_small(self) -> int:
        return max(8, self.mi.k + 2)

    def evaluate(self, Y: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        mask = np.asarray(mask, dtype=bool)
        key = np.packbits(mask).tobytes()
        if key in self.family:
            _, q, obs, mu, sd = self.family[key]
            return {"quality": q, "mi_subgroup": obs, "null_mean": mu, "null_sd": sd}
        Xg, Yg = self.X[mask], Y[mask]
        n_g = Xg.shape[0]
        if n_g < self.n_small:
            out = {"quality": 0.0, "mi_subgroup": 0.0, "null_mean": 0.0, "null_sd": 1.0}
        else:
            trees = self.mi.trees(Xg, Yg)
            observed = self.mi.value(Xg, Yg, trees)
            null = np.empty(self.n_null)
            for b in range(self.n_null):
                null[b] = self.mi.value(Xg, Yg[self.rng.permutation(n_g)], trees)
            mu, sd = float(null.mean()), float(null.std(ddof=1))
            if not np.isfinite(sd) or sd < 1e-12:
                sd = 1e-12
            out = {"quality": float((observed - mu) / sd), "mi_subgroup": float(observed), "null_mean": mu, "null_sd": sd}
        self.family[key] = (mask.copy(), out["quality"], out["mi_subgroup"], out["null_mean"], out["null_sd"])
        return out

    def score(self, Y: np.ndarray, mask: np.ndarray) -> float:
        return self.evaluate(Y, mask)["quality"]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MRSDMIConfig:
    """Configuration of MRSD-MI."""

    # mutual information
    mi_method: str = "joint"                 # joint | pairwise | combined
    mi_estimator: str = "mixed"              # mixed (continuous / categorical / both) | knn | binning
    k_neighbors: int = 5
    aggregation: str = "mean"                # pairwise aggregation
    n_null_shuffles: int = 20                # within-G shuffles for the search score

    # search
    search_method: str = "beam"              # beam | clustering
    beam_width: int = 10
    max_depth: int = 3
    n_bins_discretize: int = 5
    clustering_algorithm: str = "kmeans"
    k_values: Optional[List[int]] = None     # default [n_clusters] or [2, 3, 4, 5]
    n_clusters: Optional[int] = None

    # subgroup constraints
    n_min: int = 20
    n_max: Optional[int] = None              # default n // 2
    top_k: int = 20                          # after pruning
    min_exceptionality: float = 0.0          # minimum z-score to be reported

    # post-processing and inference
    redundancy_threshold: float = 0.7
    n_permutations: int = 100
    family_wise: str = "search"              # max-T over 'search' (all evaluated masks) or 'candidates'

    random_state: Optional[int] = None
    verbose: bool = False


# =============================================================================
# ALGORITHM
# =============================================================================

class MRSD_MI:
    """MRSD-MI (Multiple Responses Subgroup Discovery, Mutual Information)."""

    ALGORITHM = "MRSD-MI"

    def __init__(self, config: Optional[MRSDMIConfig] = None):
        self.config = config or MRSDMIConfig()
        self._exc: Optional[MIExceptionality] = None
        self.n_evaluated: int = 0

    def fit(self, dataset: Dataset) -> SetOfSubgroups:
        cfg = self.config
        rng = make_rng(cfg.random_state)
        mi = MutualInformation(MIEstimator(cfg.mi_estimator), MIMethod(cfg.mi_method),
                               MIAggregation(cfg.aggregation), cfg.k_neighbors)
        search = SearchMethod(cfg.search_method)

        fs_X, fs_Y = dataset.X_space, dataset.Y_space
        n = fs_X.n
        n_max = cfg.n_max if cfg.n_max is not None else n // 2
        if MIEstimator(cfg.mi_estimator) == MIEstimator.MIXED:
            X, Y = mixed_matrix(fs_X), mixed_matrix(fs_Y)      # one column per variable, ties are meaningful
        else:
            X = _jitter(fs_X.numeric, rng)
            Y = _jitter(fs_Y.numeric, rng)
        self._exc = MIExceptionality(X, mi, cfg.n_null_shuffles, make_rng(derive_seed(rng)))

        if cfg.verbose:
            print(f"{self.ALGORITHM}: n={n}, mi={cfg.mi_method}/{cfg.mi_estimator}, search={search.value}")

        cands: List[Tuple[np.ndarray, Optional[Rule], float, str]] = []
        if search == SearchMethod.BEAM:
            gen = ConditionGenerator(fs_X, n_bins=cfg.n_bins_discretize, min_support=cfg.n_min)
            bs = BeamSearch(gen, lambda mk: self._exc.score(Y, mk), beam_width=cfg.beam_width,
                            max_depth=cfg.max_depth, n_min=cfg.n_min, n_max=n_max)
            cands = [(r.mask, r.rule, r.quality, "beam") for r in bs.search()]
            self.n_evaluated = bs.n_evaluated
        else:
            for k in _k_values(cfg.k_values, cfg.n_clusters, [2, 3, 4, 5]):
                if k < 2:
                    continue
                try:
                    part = ClusteringFactory.get_clusterer(cfg.clustering_algorithm).fit(
                        fs_X.numeric, k=k, random_state=derive_seed(rng))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    logger.warning("MRSD-MI: clustering with k=%d failed: %s", k, exc)
                    continue
                for c in range(part.k):
                    mask = part.labels == c
                    if cfg.n_min <= int(mask.sum()) <= n_max:
                        cands.append((mask, None, self._exc.score(Y, mask), f"cluster k={k} c={c}"))
            self.n_evaluated = len(cands)
        if cfg.verbose:
            print(f"  {len(cands)} candidates, {self.n_evaluated} evaluations")

        subgroups: List[Subgroup] = []
        for mask, rule, q, origin in cands:
            if q < cfg.min_exceptionality:
                continue
            ev = self._exc.evaluate(Y, mask)
            comp = ~mask
            mi_c = mi.value(X[comp], Y[comp]) if comp.sum() >= max(8, cfg.k_neighbors + 2) else 0.0
            qm: Dict[str, object] = {
                "quality": ev["quality"], "mi_subgroup": ev["mi_subgroup"], "mi_complement": float(mi_c),
                "mi_difference": float(ev["mi_subgroup"] - mi_c),
                "linfoot_subgroup": linfoot(ev["mi_subgroup"]), "linfoot_complement": linfoot(mi_c),
                "null_mean": ev["null_mean"], "null_sd": ev["null_sd"],
                "mi_method": cfg.mi_method, "mi_estimator": cfg.mi_estimator, "search_origin": origin,
            }
            idx = set(np.flatnonzero(mask).tolist())
            if rule is not None:
                sg = Subgroup(description=str(rule), covered_instances=idx, quality_measures=qm,
                              algorithm=self.ALGORITHM, rule=rule, description_precision=1.0, description_recall=1.0)
            else:
                d = describe_mask_with_tree(fs_X, mask, random_state=derive_seed(rng))
                sg = Subgroup(description=f"MI {origin}: {d.rule}", covered_instances=idx, quality_measures=qm,
                              algorithm=self.ALGORITHM, rule=d.rule, description_precision=d.precision,
                              description_recall=d.recall)
            sg.fill_target_statistics(dataset.Y)
            subgroups.append(sg)

        result = SetOfSubgroups(subgroups).prune(QualityMeasures.quality, s_min=cfg.redundancy_threshold)
        result = result.top_k(QualityMeasures.quality, cfg.top_k)
        if cfg.n_permutations > 0 and len(result) > 0:
            self._validate(result, Y, rng)
        if cfg.verbose:
            print(result.summary())
        return result

    def _validate(self, result: SetOfSubgroups, Y: np.ndarray, rng: np.random.Generator) -> None:
        """
        Permute rows of Y and recompute (MI_G - mean_null_G) / sd_null_G, with
        the null mean / sd of every mask fixed at the values estimated from the
        within-G shuffles (the KSG null under independence depends on n_G, k
        and the dimensions, not on the pairing).  With family_wise='search'
        the max-T step runs over every mask the search evaluated, so the
        adjusted p-values account for the search, not only for the returned
        candidates.
        """
        cfg = self.config
        sgs = list(result)
        n = Y.shape[0]
        cand = [sg.mask(n) for sg in sgs]
        cand_keys = {np.packbits(m).tobytes() for m in cand}
        extra = ([rec[0] for key, rec in self._exc.family.items() if key not in cand_keys]
                 if cfg.family_wise == "search" else [])
        masks = cand + extra
        fam = self._exc.family
        params = np.array([(fam[np.packbits(m).tobytes()][3], fam[np.packbits(m).tobytes()][4]) for m in masks])
        mu, sd = params[:, 0], params[:, 1]
        X, mi, n_small = self._exc.X, self._exc.mi, self._exc.n_small
        tx = [mi.trees(X[mk], Y[mk]) for mk in masks]            # X-side trees are permutation invariant

        def batch(Yd: np.ndarray, mks) -> np.ndarray:
            vals = np.empty(len(mks))
            for i, mk in enumerate(mks):
                if mk.sum() < n_small:
                    vals[i] = 0.0
                    continue
                t = tx[i]
                trees = (t[0], None, t[2], None) if t is not None else None
                vals[i] = mi.value(X[mk], Yd[mk], trees)
            return (vals - mu) / sd

        res = permutation_test_candidates(masks, None, Y, n_permutations=cfg.n_permutations,
                                          random_state=derive_seed(rng), batch_fn=batch)
        m = len(sgs)
        for sg, obs, pr, pm, pb in zip(sgs, res.observed[:m], res.p_raw[:m], res.p_adj[:m], res.p_bh[:m]):
            sg.p_value = float(pr)
            sg.p_value_corrected = float(pm)
            sg.correction_method = "westfall_young_maxt"
            sg.quality_measures["test_statistic"] = float(obs)
            sg.quality_measures["p_bh"] = float(pb)
            sg.quality_measures["inference"] = (f"mi_zscore; permuted=Y; B={cfg.n_permutations}; "
                                                f"family={len(masks)} ({cfg.family_wise})")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_mrsd_mi(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
                **config_kwargs) -> SetOfSubgroups:
    """Run MRSD-MI on a DataFrame or a file path."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    return MRSD_MI(MRSDMIConfig(**config_kwargs)).fit(dataset)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 300
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    planted = x2 > 0.5
    y1 = np.where(planted, x1, rng.normal(size=n)) + 0.3 * rng.normal(size=n)
    y2 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y1": y1, "y2": y2})
    res = run_mrsd_mi(df, ["x1", "x2", "x3"], ["y1", "y2"], n_permutations=100, max_depth=2,
                      n_bins_discretize=8, random_state=0)
    best = res.top_k(QualityMeasures.quality, 1)[0]
    bm = best.mask(n)
    jac = (bm & planted).sum() / (bm | planted).sum()
    print(f"best rule '{best.rule}' n={best.coverage} z={best.quality:.2f} MI_G={best.quality_measures['mi_subgroup']:.3f} "
          f"MI_C={best.quality_measures['mi_complement']:.3f} p_adj={best.p_value_corrected:.3f} "
          f"Jaccard={jac:.2f} -> {'recovered' if jac >= 0.5 and best.p_value_corrected < 0.05 else 'not recovered'}")
