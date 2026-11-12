"""
MRSD-MV: Multiple Responses Subgroup Discovery with Multi-View Clustering.

Idea
----
X (descriptive variables) and Y (responses) are two views of the same
instances.  A multi-view clustering yields candidate subgroups that are
coherent in both views; a candidate is interesting when the two views agree
on its internal structure more than they agree on the rest of the data, and
when the subgroup can be recognised out-of-fold both from X and from Y.

Affinities
----------
Kernel on X_space.numeric and on Y_space.numeric (`kernel_type`: rbf with
the median heuristic by default, linear, polynomial, cosine or laplacian).
For n > 2000 the kernel is sparsified with a symmetric k-nearest-neighbour
mask (AffinityComputer.compute_knn_affinity) so that the n x n matrices stay
manageable.

Multi-view methods (`method`)
-----------------------------
* 'coregularized' (default): centroid-based co-regularized spectral
  clustering (Kumar & Daume, ICML 2011).  With normalized affinities
  L_v = D^{-1/2} W_v D^{-1/2}, the k leading eigenvectors U_X of L_X are
  computed, then for a few iterations U_Y <- eigvecs(L_Y + lambda U_X U_X^T)
  and U_X <- eigvecs(L_X + lambda U_Y U_Y^T); the row-normalized
  concatenation [U_X U_Y] is clustered with k-means.  This is the one
  genuinely multi-view objective in the module: each view's spectral
  embedding is pulled towards the other view's.
* 'average': spectral clustering of (W_X + W_Y) / 2.  A sum of kernels is
  the kernel of the concatenated feature maps, so this is ordinary spectral
  clustering on [X Y]; kept as the baseline.
* 'co_training': alternate centroid-based assignments between the views
  (initial k-means in X, centroids in Y from the X labels, re-assign in Y,
  centroids in X from the Y labels, ...); empty clusters are dropped and
  `max_iterations` must be >= 1.
* 'late_fusion': k-means per view, vectorized co-association matrix,
  spectral clustering of the co-association.

* 'excess': no clustering.  Pairwise excess of joint similarity over
  independence, B = W_X * (H W_Y H) (see excess_matrix); a subgroup maximises
  Q(G) = sum_{i != j in G} B_ij / |G|**a.  Solved by the spectral relaxation
  (leading eigenvector of B), the best prefix of the ordered instances and
  single-instance moves; further subgroups by deflation.  quality = Q.
* 'dual_dendrogram': hierarchical clustering of X and of Y; a candidate is
  the intersection of a node of each dendrogram, scored by the adjusted
  standardized residual of that cell; only local maxima over parent / child
  moves in either tree are kept.  quality = residual.
For these two methods the classification qualities q_X, q_Y are reported but
not blended into the quality, and the scores are NOT calibrated against
selection: on pure noise the best Q / residual is of the same order as on the
planted test data, so use the permutation inference to judge them.

Removed: 'multi_kernel' / 'learned' fusion (its random weight search was
silently equivalent to the average), 'product' fusion (the product of two RBF
kernels is a single RBF kernel on [X Y]), 'weighted' / 'max' fusion (no
multi-view content beyond the average).  All randomness (spectral / k-means
seeds) comes from the algorithm's rng.

Quality
-------
(a) View agreement: with single-view k-means clusterings of X and of Y (same
    k as the candidate), ARI_G is the adjusted Rand index between the two
    labelings restricted to G and ARI_C the same on the complement;
    quality_measures['ari_subgroup'], ['ari_complement'],
    ['ari_difference'] = ARI_G - ARI_C.
(b) Out-of-fold separability q_X, q_Y via classifiers.DualSpaceEvaluator
    (fast=True, 3 folds; `use_classification_quality`, default True).
quality = 0.5 * clip(ARI_G - ARI_C, 0, 1) + 0.5 * min(q_X, q_Y), i.e. every
term is bounded in [0, 1]; without (b) quality = clip(ARI_G - ARI_C, 0, 1).
The former compactness / separation / affinity-correlation terms were
unbounded, size dependent and not comparable across datasets.

Inference (`inference_statistic`)
---------------------------------
* 'y_shift_rule' (default): a cluster built from both views is shifted in Y
  by construction, so a Y-shift test of the cluster itself is tautological
  (in pure noise every fused cluster would be "significant").  The X-only
  rule description of the cluster (rules.describe_mask_with_tree) is tested
  instead: rows of Y are permuted and Hotelling T^2 of the rule mask vs its
  complement is recomputed; with `family_wise='search'` the Westfall-Young
  maximum runs over the rule masks of every evaluated cluster.  The rule's
  precision / recall w.r.t. the cluster are stated in
  quality_measures['inference'].
* 'y_shift': inference.validate_subgroups on the cluster itself (the
  DEV_NOTES convention for candidates selected from both views); the
  p-values are conditional on the candidate set and anti-conservative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, pairwise_kernels
from sklearn.neighbors import kneighbors_graph

from .classifiers import DualSpaceEvaluator
from .common import derive_seed, make_rng
from .data_loader import DataLoader, Dataset
from .inference import permutation_test_candidates, validate_subgroups, y_shift_batch
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

KNN_SPARSIFY_ABOVE = 2000


# =============================================================================
# ENUMS
# =============================================================================

class MultiViewMethod(Enum):
    COREGULARIZED = "coregularized"
    AVERAGE = "average"
    CO_TRAINING = "co_training"
    LATE_FUSION = "late_fusion"
    EXCESS = "excess"                       # similarity excess over independence (modularity-like)
    DUAL_DENDROGRAM = "dual_dendrogram"     # pairs of nodes of the dendrograms of X and of Y

    @classmethod
    def _missing_(cls, value):
        aliases = {"spectral_fusion": cls.AVERAGE, "kernel_fusion": cls.AVERAGE}
        if isinstance(value, str) and value in aliases:
            return aliases[value]
        return None


class KernelType(Enum):
    RBF = "rbf"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    COSINE = "cosine"
    LAPLACIAN = "laplacian"


# =============================================================================
# AFFINITIES
# =============================================================================

class AffinityComputer:
    """Kernel / affinity matrix of one view."""

    def __init__(self, kernel_type: KernelType = KernelType.RBF, gamma: Optional[float] = None,
                 degree: int = 3, n_neighbors: int = 10):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree
        self.n_neighbors = n_neighbors

    def _gamma(self, X: np.ndarray) -> float:
        if self.gamma is not None:
            return self.gamma
        d = pdist(X, "euclidean")
        med = float(np.median(d)) if len(d) else 0.0
        return 1.0 / (2 * med ** 2) if med > 0 else 1.0

    def compute_kernel(self, X: np.ndarray) -> np.ndarray:
        kt = self.kernel_type
        if kt == KernelType.RBF:
            return pairwise_kernels(X, metric="rbf", gamma=self._gamma(X))
        if kt == KernelType.LINEAR:
            return pairwise_kernels(X, metric="linear")
        if kt == KernelType.POLYNOMIAL:
            return pairwise_kernels(X, metric="poly", degree=self.degree)
        if kt == KernelType.COSINE:
            return pairwise_kernels(X, metric="cosine")
        return np.exp(-self._gamma(X) * squareform(pdist(X, "cityblock")))

    def compute_knn_affinity(self, X: np.ndarray) -> np.ndarray:
        """Symmetric k-nearest-neighbour connectivity (dense 0/1 matrix)."""
        knn = kneighbors_graph(X, n_neighbors=min(self.n_neighbors, X.shape[0] - 1), mode="connectivity")
        return (0.5 * (knn + knn.T)).toarray()

    def compute_affinity(self, X: np.ndarray, use_knn: bool = False) -> np.ndarray:
        """Kernel matrix; with `use_knn` the kernel is restricted to the kNN graph."""
        K = np.maximum(self.compute_kernel(X), 0.0)
        if use_knn:
            K = K * (self.compute_knn_affinity(X) > 0)
            np.fill_diagonal(K, 1.0)
        return 0.5 * (K + K.T)


def _normalized_affinity(W: np.ndarray) -> np.ndarray:
    d = W.sum(axis=1)
    d = np.where(d > 1e-12, d, 1e-12)
    Dm = 1.0 / np.sqrt(d)
    return Dm[:, None] * W * Dm[None, :]


def _top_eigvecs(A: np.ndarray, k: int) -> np.ndarray:
    n = A.shape[0]
    _, V = linalg.eigh(A, subset_by_index=[n - k, n - 1])
    return V


def _relabel(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels, return_inverse=True)[1].astype(int)


# =============================================================================
# RELEVANCE WEIGHTS OF THE X VARIABLES
# =============================================================================

def relevance_weights(fs_X, W_Y: np.ndarray, power: float = 2.0) -> Tuple[np.ndarray, Dict[str, float]]:
    """Weight of every X variable = its kernel alignment with Y.

    For variable k with (standardized / one-hot) columns X_k and the centred Y
    kernel K = H W_Y H:  CKA_k = tr(X_k' K X_k) / (||X_k' X_k||_F ||K||_F), a
    number in [0, 1] that is 0 when the variable is unrelated to the
    similarity structure of Y.  weights = (CKA / max CKA) ** power, so that
    variables unrelated to Y do not dilute the X similarity.  Returns the
    weight of every column of fs_X.numeric and {variable: weight}."""
    X = np.nan_to_num(np.asarray(fs_X.numeric, dtype=float))
    n = X.shape[0]
    H = np.eye(n) - 1.0 / n
    K = H @ W_Y @ H
    k_norm = np.linalg.norm(K)
    names = list(fs_X.numeric_names)
    columns = list(fs_X.columns)
    owner = []
    for nm in names:
        base = nm.split("=", 1)[0] if nm not in columns else nm
        owner.append(base if base in columns else nm)
    cka: Dict[str, float] = {}
    for var in dict.fromkeys(owner):
        idx = [j for j, o in enumerate(owner) if o == var]
        Xk = X[:, idx] - X[:, idx].mean(axis=0)
        den = np.linalg.norm(Xk.T @ Xk) * k_norm
        cka[var] = float(np.trace(Xk.T @ K @ Xk) / den) if den > 1e-12 else 0.0
    top = max(max(cka.values()), 1e-12)
    w_var = {v: float(max(c, 0.0) / top) ** power for v, c in cka.items()}
    return np.array([w_var[o] for o in owner]), w_var


def weighted_X(fs_X, Y: np.ndarray, computer: "AffinityComputer", x_weighting: str = "relevance",
               power: float = 2.0) -> Tuple[np.ndarray, Dict[str, float]]:
    """fs_X.numeric with its columns scaled by sqrt(weight) (so squared
    distances are weighted sums), and the weights per variable."""
    X = np.nan_to_num(np.asarray(fs_X.numeric, dtype=float))
    if x_weighting == "none":
        return X, {}
    if x_weighting != "relevance":
        raise ValueError(f"x_weighting must be 'none' or 'relevance', got {x_weighting!r}")
    w, w_var = relevance_weights(fs_X, computer.compute_affinity(Y), power)
    return X * np.sqrt(w)[None, :], w_var


def dual_linkages(X: np.ndarray, Y: np.ndarray, linkage_method: str = "ward"):
    """Linkage matrices of X and of Y (Euclidean), as used by 'dual_dendrogram'."""
    from scipy.cluster.hierarchy import linkage
    return linkage(np.nan_to_num(X), method=linkage_method), linkage(np.nan_to_num(Y), method=linkage_method)


# =============================================================================
# SIMILARITY EXCESS  (method 'excess')
# =============================================================================

def excess_matrix(W_X: np.ndarray, W_Y: np.ndarray, gamma: float = 1.0, reference: str = "centered") -> np.ndarray:
    """Pairwise excess of joint similarity over independence, zero diagonal.

    reference='centered' (default): B = W_X * (H W_Y H), the Y kernel centred
        in feature space.  A pair scores high when both instances deviate from
        the centre of Y in the same direction, so a small shifted group stands
        out while pairs of "ordinary" instances score about zero.  With
        W_X = 1 and a linear kernel, sum_{i,j in G} B_ij = |G|^2 ||mean_G(Y) -
        mean(Y)||^2 (the mean-shift statistic); W_X restricts it to pairs
        that are also close in X.  The sum of all entries of the fully centred
        version is HSIC.
    reference='mean': B = W_X * (W_Y - gamma * mean off-diagonal W_Y), the
        expectation of W_Y under a permutation of the rows of Y.  Rewards any
        set that is dense in both views, including the majority, and on the
        planted test data it does not single out the planted subgroups."""
    n = W_X.shape[0]
    if reference == "centered":
        H = np.eye(n) - 1.0 / n
        B = W_X * (H @ W_Y @ H)
    elif reference == "mean":
        wbar = (W_Y.sum() - np.trace(W_Y)) / max(n * (n - 1), 1)
        B = W_X * (W_Y - gamma * wbar)
    else:
        raise ValueError(f"excess_reference must be 'centered' or 'mean', got {reference!r}")
    np.fill_diagonal(B, 0.0)
    return 0.5 * (B + B.T)


def excess_quality(B: np.ndarray, mask: np.ndarray, a: float = 1.0) -> float:
    """Q(G) = sum_{i != j in G} B_ij / |G|**a."""
    m = int(mask.sum())
    if m == 0:
        return 0.0
    return float(B[np.ix_(mask, mask)].sum() / m ** a)


def _best_prefix(B: np.ndarray, order: np.ndarray, n_min: int, n_max: int, a: float):
    Bo = B[np.ix_(order, order)]
    inc = 2.0 * np.tril(Bo, -1).sum(axis=1)          # gain of adding the m-th instance to the first m-1
    S = np.cumsum(inc)
    m = np.arange(1, len(order) + 1)
    Q = S / m ** a
    lo, hi = max(n_min, 1), min(n_max, len(order))
    if lo > hi:
        return None, -np.inf
    j = int(np.argmax(Q[lo - 1:hi])) + lo
    return j, float(Q[j - 1])


def excess_subgroup(B: np.ndarray, n_min: int, n_max: int, a: float = 1.0, max_moves: Optional[int] = None):
    """One subgroup maximising Q: leading eigenvector of B (spectral relaxation
    of max u'Bu / u'u), best prefix of the instances ordered by their
    component (both signs tried), then single-instance add / remove moves
    while Q increases.  Returns (mask, Q) or (None, -inf)."""
    n = B.shape[0]
    try:
        _, V = linalg.eigh(B, subset_by_index=[n - 1, n - 1])
    except (ValueError, np.linalg.LinAlgError):
        return None, -np.inf
    u = V[:, 0]
    best_mask, best_q = None, -np.inf
    for sign in (1.0, -1.0):
        order = np.argsort(-sign * u)
        m, q = _best_prefix(B, order, n_min, n_max, a)
        if m is not None and q > best_q:
            best_mask = np.zeros(n, dtype=bool)
            best_mask[order[:m]] = True
            best_q = q
    if best_mask is None:
        return None, -np.inf
    mask = best_mask
    S = float(B[np.ix_(mask, mask)].sum())
    d = 2.0 * B[:, mask].sum(axis=1)                 # change of S when an instance joins / leaves
    m = int(mask.sum())
    for _ in range(max_moves if max_moves is not None else 2 * n):
        q = S / m ** a
        q_add = np.where(~mask, (S + d) / (m + 1) ** a, -np.inf) if m < n_max else np.full(n, -np.inf)
        q_rem = np.where(mask, (S - d) / max(m - 1, 1) ** a, -np.inf) if m > n_min else np.full(n, -np.inf)
        i_add, i_rem = int(np.argmax(q_add)), int(np.argmax(q_rem))
        if max(q_add[i_add], q_rem[i_rem]) <= q + 1e-12:
            break
        if q_add[i_add] >= q_rem[i_rem]:
            mask[i_add] = True
            S += d[i_add]
            d += 2.0 * B[:, i_add]
            m += 1
        else:
            mask[i_rem] = False
            S -= d[i_rem]
            d -= 2.0 * B[:, i_rem]
            m -= 1
    return mask, float(S / m ** a)


def excess_subgroups(W_X: np.ndarray, W_Y: np.ndarray, n_min: int, n_max: int, gamma: float = 1.0,
                     a: float = 1.0, n_subgroups: int = 5, reference: str = "centered"
                     ) -> List[Tuple[np.ndarray, Dict[str, object]]]:
    """Subgroups by similarity excess, extracted one after another: after a
    subgroup is found its rows and columns of B are zeroed (deflation), so
    the next one is built from other instances."""
    B = excess_matrix(W_X, W_Y, gamma, reference)
    B_full = B.copy()
    out = []
    for r in range(int(n_subgroups)):
        mask, q = excess_subgroup(B, n_min, n_max, a)
        if mask is None or q <= 0:
            break
        m = int(mask.sum())
        out.append((mask.copy(), {"quality": float(excess_quality(B_full, mask, a)), "excess_sum":
                                  float(B_full[np.ix_(mask, mask)].sum()),
                                  "mean_pair_excess": float(B_full[np.ix_(mask, mask)].sum() / max(m * (m - 1), 1)),
                                  "extraction_order": r + 1, "gamma": float(gamma), "size_exponent": float(a),
                                  "reference": reference, "method": "excess"}))
        B[mask, :] = 0.0
        B[:, mask] = 0.0
    return out


# =============================================================================
# DUAL DENDROGRAM  (method 'dual_dendrogram')
# =============================================================================

def _dendrogram_nodes(Z: np.ndarray, n: int, n_min: int):
    """Internal nodes with n_min <= size < n: membership matrix (nodes x n),
    scipy node ids, and for every kept node the positions of its parent and
    children among the kept nodes (-1 if not kept)."""
    members: Dict[int, np.ndarray] = {}
    parent = {}
    kids = {}
    for t in range(Z.shape[0]):
        nid = n + t
        a, b = int(Z[t, 0]), int(Z[t, 1])
        ma = members[a] if a >= n else np.array([a])
        mb = members[b] if b >= n else np.array([b])
        members[nid] = np.concatenate([ma, mb])
        parent[a] = nid
        parent[b] = nid
        kids[nid] = (a, b)
    keep = [nid for nid, mem in members.items() if n_min <= len(mem) < n]
    pos = {nid: i for i, nid in enumerate(keep)}
    M = np.zeros((len(keep), n), dtype=np.float32)
    for nid, i in pos.items():
        M[i, members[nid]] = 1.0
    par = np.array([pos.get(parent.get(nid, -1), -1) for nid in keep], dtype=int)
    c1 = np.array([pos.get(kids[nid][0], -1) for nid in keep], dtype=int)
    c2 = np.array([pos.get(kids[nid][1], -1) for nid in keep], dtype=int)
    return M, np.array(keep, dtype=int), par, c1, c2


def dual_dendrogram_residuals(M_X: np.ndarray, M_Y: np.ndarray):
    """Observed intersections and adjusted standardized residuals of every
    pair (node of X, node of Y)."""
    n = M_X.shape[1]
    O = (M_X @ M_Y.T).astype(float)
    sa = M_X.sum(axis=1).astype(float)[:, None]
    sb = M_Y.sum(axis=1).astype(float)[None, :]
    E = sa * sb / n
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (O - E) / np.sqrt(E * (1 - sa / n) * (1 - sb / n))
    return O, E, np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)


def _neighbour_max(R: np.ndarray, idx: np.ndarray, axis: int) -> np.ndarray:
    """R re-indexed along `axis` by idx, -inf where idx == -1."""
    safe = np.where(idx >= 0, idx, 0)
    out = R[safe, :] if axis == 0 else R[:, safe]
    if axis == 0:
        out = np.where((idx >= 0)[:, None], out, -np.inf)
    else:
        out = np.where((idx >= 0)[None, :], out, -np.inf)
    return out


def dual_dendrogram_subgroups(X: np.ndarray, Y: np.ndarray, n_min: int, n_max: int, linkage_method: str = "ward",
                              max_candidates: int = 200) -> List[Tuple[np.ndarray, Dict[str, object]]]:
    """Candidates G = a ∩ b for a node a of the dendrogram of X and a node b
    of the dendrogram of Y.  Quality = adjusted standardized residual of the
    cell (a, b) of the 2 x 2 table.  Only local maxima are kept: pairs whose
    residual is not exceeded by the pairs obtained by moving a or b to its
    parent or to one of its children."""
    n = X.shape[0]
    Z_X, Z_Y = dual_linkages(X, Y, linkage_method)
    M_X, ids_X, par_X, c1_X, c2_X = _dendrogram_nodes(Z_X, n, n_min)
    M_Y, ids_Y, par_Y, c1_Y, c2_Y = _dendrogram_nodes(Z_Y, n, n_min)
    if len(ids_X) == 0 or len(ids_Y) == 0:
        return []
    O, E, R = dual_dendrogram_residuals(M_X, M_Y)
    ok = (O >= n_min) & (O <= n_max) & (R > 0)
    for idx, axis in ((par_X, 0), (c1_X, 0), (c2_X, 0), (par_Y, 1), (c1_Y, 1), (c2_Y, 1)):
        ok &= R >= _neighbour_max(R, idx, axis)
    cand = np.argwhere(ok)
    if len(cand) == 0:
        return []
    order = np.argsort(-R[cand[:, 0], cand[:, 1]])[:max_candidates]
    out, seen = [], set()
    sa, sb = M_X.sum(axis=1), M_Y.sum(axis=1)
    for i, j in cand[order]:
        mask = (M_X[i] > 0) & (M_Y[j] > 0)
        key = np.packbits(mask).tobytes()
        if key in seen:
            continue
        seen.add(key)
        union = float(sa[i] + sb[j] - O[i, j])
        jac = float(O[i, j] / union) if union else 0.0
        out.append((mask, {"quality": float(R[i, j]), "residual": float(R[i, j]), "observed": int(O[i, j]),
                           "expected": float(E[i, j]), "node_X": int(ids_X[i]), "node_Y": int(ids_Y[j]),
                           "size_node_X": int(sa[i]), "size_node_Y": int(sb[j]), "jaccard_nodes": jac,
                           "kind": "agreement" if jac >= 0.5 else "intersection",
                           "linkage_method": linkage_method, "method": "dual_dendrogram"}))
    return out


def restricted_ari(lab_a: np.ndarray, lab_b: np.ndarray, mask: np.ndarray) -> float:
    """Adjusted Rand index between two labelings restricted to `mask`; 0 when
    either labeling is constant on the subset (no evidence of agreement; sklearn
    would return 1 for two constant labelings)."""
    a, b = lab_a[mask], lab_b[mask]
    if len(a) < 2 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return 0.0
    return float(adjusted_rand_score(a, b))


# =============================================================================
# MULTI-VIEW CLUSTERINGS
# =============================================================================

def coregularized_spectral(W_X: np.ndarray, W_Y: np.ndarray, k: int, lam: float = 0.05,
                           n_iterations: int = 5, seed: Optional[int] = None) -> np.ndarray:
    """Centroid-based co-regularized spectral clustering (Kumar & Daume, 2011)."""
    L_X = _normalized_affinity(W_X)
    L_Y = _normalized_affinity(W_Y)
    U_X = _top_eigvecs(L_X, k)
    U_Y = _top_eigvecs(L_Y + lam * (U_X @ U_X.T), k)
    for _ in range(max(1, n_iterations) - 1):
        U_X = _top_eigvecs(L_X + lam * (U_Y @ U_Y.T), k)
        U_Y = _top_eigvecs(L_Y + lam * (U_X @ U_X.T), k)
    E = np.hstack([U_X, U_Y])
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.where(norms > 1e-12, norms, 1.0)
    return _relabel(KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(E))


def average_spectral(W_X: np.ndarray, W_Y: np.ndarray, k: int, seed: Optional[int] = None) -> np.ndarray:
    """Spectral clustering of the sum kernel (= kernel on the concatenation [X Y])."""
    W = 0.5 * (W_X + W_Y)
    labels = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed,
                                assign_labels="kmeans").fit_predict(W)
    return _relabel(labels)


def co_training(X: np.ndarray, Y: np.ndarray, k: int, max_iterations: int = 10,
                convergence_threshold: float = 0.01, seed: Optional[int] = None) -> np.ndarray:
    """Alternate centroid-based assignments between the two views."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    n = X.shape[0]
    labels_X = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    prev = labels_X.copy()
    dists_X = dists_Y = None
    for _ in range(max_iterations):
        ids = np.unique(labels_X)                      # empty clusters are dropped
        cent_Y = np.array([Y[labels_X == c].mean(axis=0) for c in ids])
        dists_Y = cdist(Y, cent_Y)
        labels_Y = ids[np.argmin(dists_Y, axis=1)]
        ids = np.unique(labels_Y)
        cent_X = np.array([X[labels_Y == c].mean(axis=0) for c in ids])
        dists_X = cdist(X, cent_X)
        labels_X = ids[np.argmin(dists_X, axis=1)]
        if np.mean(labels_X != prev) < convergence_threshold:
            break
        prev = labels_X.copy()
    ids = np.unique(labels_X)
    cent_X = np.array([X[labels_X == c].mean(axis=0) for c in ids])
    cent_Y = np.array([Y[labels_X == c].mean(axis=0) for c in ids])
    combined = 0.5 * cdist(X, cent_X) + 0.5 * cdist(Y, cent_Y)
    return _relabel(np.argmin(combined, axis=1))


def late_fusion(views: List[np.ndarray], k: int, seed: Optional[int] = None) -> np.ndarray:
    """k-means per view, co-association matrix, spectral clustering of it."""
    rng = make_rng(seed)
    n = views[0].shape[0]
    C = np.zeros((n, n))
    for V in views:
        lab = KMeans(n_clusters=k, n_init=10, random_state=derive_seed(rng)).fit_predict(V)
        C += (lab[:, None] == lab[None, :])
    C /= len(views)
    labels = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=derive_seed(rng),
                                assign_labels="kmeans").fit_predict(C)
    return _relabel(labels)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MRSDMVConfig:
    """Configuration of MRSD-MV."""

    method: str = "coregularized"          # coregularized | average | co_training | late_fusion
    kernel_type: str = "rbf"
    gamma: Optional[float] = None
    coreg_lambda: float = 0.05
    coreg_iterations: int = 5

    k_values: Optional[List[int]] = None   # default [n_clusters] or [2, 3, 4]
    n_clusters: Optional[int] = None
    max_iterations: int = 10               # co-training
    convergence_threshold: float = 0.01

    # method 'excess'
    excess_reference: str = "centered"     # centered: W_X * (H W_Y H) | mean: W_X * (W_Y - gamma * mean W_Y)
    excess_gamma: float = 1.0              # resolution of the 'mean' reference
    size_exponent: float = 1.0             # Q(G) = sum of B over pairs in G / |G|**a
    n_subgroups: int = 5                   # subgroups extracted one after another (deflation)
    # method 'dual_dendrogram'
    linkage_method: str = "ward"           # scipy linkage for both dendrograms (Euclidean distance)
    # methods 'excess' and 'dual_dendrogram': weights of the X variables in the X similarity / distance
    x_weighting: str = "relevance"         # none | relevance (kernel alignment of each X variable with Y)
    weight_power: float = 2.0              # weights = (alignment / max alignment) ** power

    use_classification_quality: bool = True
    classifier_X: str = "random_forest"
    classifier_Y: str = "random_forest"

    n_min: int = 10
    n_max: Optional[int] = None            # default n // 2
    top_k: int = 20

    redundancy_threshold: float = 0.7
    n_permutations: int = 200
    inference_statistic: str = "y_shift_rule"   # y_shift_rule | y_shift
    family_wise: str = "search"                 # max-T over 'search' (all evaluated clusters) or 'candidates'

    random_state: Optional[int] = None
    verbose: bool = False


# =============================================================================
# ALGORITHM
# =============================================================================

class MRSD_MV:
    """MRSD-MV (Multiple Responses Subgroup Discovery, Multi-View Clustering)."""

    ALGORITHM = "MRSD-MV"

    def __init__(self, config: Optional[MRSDMVConfig] = None):
        self.config = config or MRSDMVConfig()
        self._family: List[Tuple[np.ndarray, np.ndarray]] = []   # (cluster mask, rule mask)

    def _cluster(self, method: MultiViewMethod, X, Y, W_X, W_Y, k: int, rng) -> np.ndarray:
        cfg = self.config
        seed = derive_seed(rng)
        if method == MultiViewMethod.COREGULARIZED:
            return coregularized_spectral(W_X, W_Y, k, cfg.coreg_lambda, cfg.coreg_iterations, seed)
        if method == MultiViewMethod.AVERAGE:
            return average_spectral(W_X, W_Y, k, seed)
        if method == MultiViewMethod.CO_TRAINING:
            return co_training(X, Y, k, cfg.max_iterations, cfg.convergence_threshold, seed)
        return late_fusion([X, Y], k, seed)

    def fit(self, dataset: Dataset) -> SetOfSubgroups:
        cfg = self.config
        rng = make_rng(cfg.random_state)
        method = MultiViewMethod(cfg.method)
        kernel = KernelType(cfg.kernel_type)
        if cfg.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if cfg.inference_statistic not in ("y_shift_rule", "y_shift"):
            raise ValueError(f"inference_statistic must be 'y_shift_rule' or 'y_shift', got {cfg.inference_statistic!r}")
        fs_X, fs_Y = dataset.X_space, dataset.Y_space
        X, Y = fs_X.numeric, fs_Y.numeric
        n = fs_X.n
        n_max = cfg.n_max if cfg.n_max is not None else n // 2

        computer = AffinityComputer(kernel, cfg.gamma)
        use_knn = n > KNN_SPARSIFY_ABOVE
        W_X = W_Y = None
        if method in (MultiViewMethod.COREGULARIZED, MultiViewMethod.AVERAGE, MultiViewMethod.EXCESS):
            W_X = computer.compute_affinity(X, use_knn)
            W_Y = computer.compute_affinity(Y, use_knn)
        if cfg.verbose:
            print(f"{self.ALGORITHM}: n={n}, method={method.value}, kernel={kernel.value}, knn_sparsified={use_knn}")

        evaluator = None
        if cfg.use_classification_quality:
            evaluator = DualSpaceEvaluator(X, Y, cfg.classifier_X, cfg.classifier_Y, n_folds=3,
                                           aggregation="min", random_state=derive_seed(rng), fast=True)

        subgroups: List[Subgroup] = []
        self._family = []
        direct = None
        x_weights: Dict[str, float] = {}
        if method in (MultiViewMethod.EXCESS, MultiViewMethod.DUAL_DENDROGRAM):
            X_w, x_weights = weighted_X(fs_X, Y, computer, cfg.x_weighting, cfg.weight_power)
            if cfg.verbose and x_weights:
                print("  X variable weights:", {k: round(v, 3) for k, v in x_weights.items()})
        if method == MultiViewMethod.EXCESS:
            if x_weights:
                W_X = computer.compute_affinity(X_w, use_knn)
            direct = excess_subgroups(W_X, W_Y, cfg.n_min, n_max, cfg.excess_gamma, cfg.size_exponent,
                                      cfg.n_subgroups, cfg.excess_reference)
        elif method == MultiViewMethod.DUAL_DENDROGRAM:
            direct = dual_dendrogram_subgroups(X_w, Y, cfg.n_min, n_max, cfg.linkage_method)
        for mask, qm in (direct or []):
            for var, w in x_weights.items():
                qm[f"weight_{var}"] = float(w)
            # these methods have their own score; classification qualities are reported, not blended in
            if evaluator is not None:
                ev = evaluator.evaluate(mask)
                qm.update({"q_X": ev.q_X, "q_Y": ev.q_Y})
            d = describe_mask_with_tree(fs_X, mask, random_state=derive_seed(rng))
            self._family.append((mask, d.rule.evaluate(fs_X)))
            sg = Subgroup(description=f"MV {method.value}: {d.rule}",
                          covered_instances=set(np.flatnonzero(mask).tolist()), quality_measures=qm,
                          algorithm=self.ALGORITHM, rule=d.rule, description_precision=d.precision,
                          description_recall=d.recall, clustering_X=method.value, clustering_Y="multi_view")
            sg.fill_target_statistics(dataset.Y)
            subgroups.append(sg)
        for k in ([] if direct is not None else _k_values(cfg.k_values, cfg.n_clusters, [2, 3, 4])):
            if k < 2 or k > n // 2:
                continue
            try:
                labels = self._cluster(method, X, Y, W_X, W_Y, k, rng)
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("MRSD-MV: %s clustering with k=%d failed: %s", method.value, k, exc)
                continue
            # single-view clusterings for the view-agreement measure
            lab_X = KMeans(n_clusters=k, n_init=10, random_state=derive_seed(rng)).fit_predict(X)
            lab_Y = KMeans(n_clusters=k, n_init=10, random_state=derive_seed(rng)).fit_predict(Y)
            for c in np.unique(labels):
                mask = labels == c
                size = int(mask.sum())
                if not (cfg.n_min <= size <= n_max):
                    continue
                ari_g = restricted_ari(lab_X, lab_Y, mask)
                ari_c = restricted_ari(lab_X, lab_Y, ~mask)
                ari_diff = ari_g - ari_c
                qm: Dict[str, object] = {"ari_subgroup": ari_g, "ari_complement": ari_c, "ari_difference": ari_diff,
                                         "k": int(k), "cluster": int(c), "method": method.value}
                agreement = float(np.clip(ari_diff, 0.0, 1.0))
                if evaluator is not None:
                    ev = evaluator.evaluate(mask)
                    qm.update({"q_X": ev.q_X, "q_Y": ev.q_Y})
                    qm["quality"] = 0.5 * agreement + 0.5 * min(ev.q_X, ev.q_Y)
                else:
                    qm["quality"] = agreement
                d = describe_mask_with_tree(fs_X, mask, random_state=derive_seed(rng))
                self._family.append((mask, d.rule.evaluate(fs_X)))
                sg = Subgroup(description=f"MV {method.value} k={k} c={c}: {d.rule}",
                              covered_instances=set(np.flatnonzero(mask).tolist()), quality_measures=qm,
                              algorithm=self.ALGORITHM, rule=d.rule, description_precision=d.precision,
                              description_recall=d.recall, k=int(k), l=int(c), clustering_X=method.value,
                              clustering_Y="multi_view")
                sg.fill_target_statistics(dataset.Y)
                subgroups.append(sg)

        result = SetOfSubgroups(subgroups).prune(QualityMeasures.quality, s_min=cfg.redundancy_threshold)
        result = result.top_k(QualityMeasures.quality, cfg.top_k)
        if cfg.n_permutations > 0 and len(result) > 0:
            if cfg.inference_statistic == "y_shift":
                validate_subgroups(result, Y_space=fs_Y, X_space=fs_X, statistic="y_shift",
                                   n_permutations=cfg.n_permutations, random_state=derive_seed(rng),
                                   selection_note="candidates selected using both views; p-values conditional on candidate set")
            elif cfg.inference_statistic == "y_shift_rule":
                self._validate_rule(result, fs_X, fs_Y, rng)
            else:
                raise ValueError(f"inference_statistic must be 'y_shift_rule' or 'y_shift', got {cfg.inference_statistic!r}")
        if cfg.verbose:
            print(result.summary())
        return result

    def _validate_rule(self, result: SetOfSubgroups, fs_X, fs_Y, rng: np.random.Generator) -> None:
        """
        Y-shift test of the X-only rule description of every cluster.

        A cluster built from both views is shifted in Y by construction, so a
        Y-shift test of the cluster itself is tautological.  The rule
        (rules.describe_mask_with_tree, a function of X only) is tested
        instead: rows of Y are permuted, Hotelling T^2 of the rule mask vs its
        complement is recomputed, and raw / Westfall-Young max-T / BH p-values
        are filled.  With family_wise='search' the max is taken over the rule
        masks of every evaluated cluster.
        """
        cfg = self.config
        sgs = list(result)
        n = fs_X.n
        cand = [sg.rule.evaluate(fs_X) for sg in sgs]
        seen = {np.packbits(m).tobytes() for m in cand}
        extra: List[np.ndarray] = []
        if cfg.family_wise == "search":
            for _, rm in self._family:
                key = np.packbits(rm).tobytes()
                if key not in seen:
                    seen.add(key)
                    extra.append(rm)
        elif cfg.family_wise != "candidates":
            raise ValueError(f"family_wise must be 'search' or 'candidates', got {cfg.family_wise!r}")
        masks = cand + extra
        res = permutation_test_candidates(masks, None, fs_Y.numeric, n_permutations=cfg.n_permutations,
                                          random_state=derive_seed(rng), batch_fn=y_shift_batch)
        m = len(sgs)
        for sg, rm, obs, pr, pm, pb in zip(sgs, cand, res.observed[:m], res.p_raw[:m], res.p_adj[:m], res.p_bh[:m]):
            sg.p_value = float(pr)
            sg.p_value_corrected = float(pm)
            sg.correction_method = "westfall_young_maxt"
            sg.quality_measures["test_statistic"] = float(obs)
            sg.quality_measures["p_bh"] = float(pb)
            sg.quality_measures["rule_coverage"] = int(rm.sum())
            sg.quality_measures["inference"] = (
                f"y_shift of X-rule description (precision={sg.description_precision:.2f}, "
                f"recall={sg.description_recall:.2f}); permuted=Y; B={cfg.n_permutations}; "
                f"family={len(masks)} ({cfg.family_wise}); candidates selected using both views")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_mrsd_mv(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
                **config_kwargs) -> SetOfSubgroups:
    """Run MRSD-MV on a DataFrame or a file path."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    return MRSD_MV(MRSDMVConfig(**config_kwargs)).fit(dataset)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 300
    planted = rng.random(n) < 0.3                 # a group that is compact in both views
    x1 = np.where(planted, 3.0, 0.0) + rng.normal(size=n)
    x2 = rng.normal(size=n)
    y1 = np.where(planted, -3.0, 0.0) + rng.normal(size=n)
    y2 = rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
    res = run_mrsd_mv(df, ["x1", "x2"], ["y1", "y2"], n_permutations=200, random_state=0)
    best = res.top_k(QualityMeasures.quality, 1)[0] if len(res) else None
    if best is None:
        print("no subgroup returned")
    else:
        bm = best.mask(n)
        jac = (bm & planted).sum() / (bm | planted).sum()
        print(f"best: {best.description} n={best.coverage} q={best.quality:.2f} ARI_G-ARI_C={best.quality_measures['ari_difference']:.2f} "
              f"p_adj={best.p_value_corrected:.3f} Jaccard={jac:.2f} -> "
              f"{'recovered' if jac >= 0.5 and best.p_value_corrected < 0.05 else 'not recovered'}")
