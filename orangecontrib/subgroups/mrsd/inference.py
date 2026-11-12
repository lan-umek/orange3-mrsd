"""
Statistical inference shared by all MRSD algorithms.

Design
------
Every algorithm produces candidate subgroups (index sets G).  Each candidate
gets a test statistic T(G) that measures the association between X and Y that
the subgroup expresses, and a p-value obtained by permutation under the null
hypothesis "X and Y are independent" (rows of one view are shuffled relative
to the other).  Because all candidates are evaluated on the same permutations,
family-wise error over the whole candidate set is controlled with the
Westfall-Young max-T procedure; Benjamini-Hochberg FDR over raw permutation
p-values is available as a less conservative alternative.

Two canonical statistics are provided:

* `y_shift_statistic`: is the Y distribution inside G different from outside?
  Hotelling-type T² between G and D \\ G on the numeric view of Y (standardized
  numeric columns + one-hot categorical columns, pseudo-inverse of the pooled
  covariance).  Use it when the candidate was selected from X (rules, X-side
  clusters) and permute rows of Y.
* `x_separability_statistic`: can membership in G be predicted from X?
  Out-of-fold AUC of a classifier.  Use it when the candidate was selected from
  Y (dendrogram nodes of Y, Y-side clusters) and permute rows of X.

For candidates that were selected using both views, the Y-shift test with Y
permuted is reported; it is then conditional on the found candidate set (the
usual "swap randomization" approximation in subgroup discovery) and this is
stated in `Subgroup.quality_measures['inference']`.

Both statistics are mapped to a common scale (-log10 of an asymptotic
p-value for T²; AUC itself for separability) so that max-T across candidates
of different sizes is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats as sps

from .common import FeatureSpace, OutOfFoldClassifier, make_rng, mask_from_indices, derive_seed


# =============================================================================
# CANDIDATE STATISTICS
# =============================================================================

def hotelling_t2(Y: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Two-sample Hotelling T² between rows in `mask` and the rest.

    Returns (T², F statistic, asymptotic p-value).  Uses the pooled covariance
    with a pseudo-inverse so that one-hot blocks (rank deficient) are allowed;
    the degrees of freedom use the numerical rank.
    """
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]
    n1 = int(mask.sum())
    n2 = n - n1
    if n1 < 2 or n2 < 2 or Y.shape[1] == 0:
        return 0.0, 0.0, 1.0
    A = Y[mask]
    B = Y[~mask]
    d = A.mean(axis=0) - B.mean(axis=0)
    S = ((n1 - 1) * np.cov(A, rowvar=False, ddof=1).reshape(Y.shape[1], Y.shape[1]) +
         (n2 - 1) * np.cov(B, rowvar=False, ddof=1).reshape(Y.shape[1], Y.shape[1])) / (n1 + n2 - 2)
    Sinv = np.linalg.pinv(S, rcond=1e-10)
    rank = int(np.linalg.matrix_rank(S, tol=1e-10))
    if rank == 0:
        return 0.0, 0.0, 1.0
    t2 = float(n1 * n2 / (n1 + n2) * d @ Sinv @ d)
    p = rank
    df2 = n1 + n2 - p - 1
    if df2 <= 0:
        # not enough data for the F approximation; fall back to chi-square
        pval = float(sps.chi2.sf(t2, p))
        return t2, t2 / p, max(pval, 1e-300)
    F = t2 * df2 / (p * (n1 + n2 - 2))
    pval = float(sps.f.sf(F, p, df2))
    return t2, float(F), max(pval, 1e-300)


def y_shift_statistic(Y: np.ndarray, mask: np.ndarray) -> float:
    """-log10 asymptotic p-value of Hotelling T² (larger = more exceptional)."""
    _, _, p = hotelling_t2(Y, mask)
    return float(-np.log10(p))


def x_separability_statistic(X: np.ndarray, mask: np.ndarray, estimator, n_folds: int = 5,
                             random_state: Optional[int] = None) -> float:
    """Out-of-fold AUC of predicting membership from X (0.5 = chance)."""
    oof = OutOfFoldClassifier(estimator, n_folds=n_folds, random_state=random_state)
    return oof.auc(X, mask.astype(int))


# =============================================================================
# PERMUTATION MACHINERY
# =============================================================================

@dataclass
class PermutationResult:
    observed: np.ndarray          # (m,)
    p_raw: np.ndarray             # (m,) permutation p-values
    p_adj: np.ndarray             # (m,) Westfall-Young max-T adjusted (FWER)
    p_bh: np.ndarray              # (m,) Benjamini-Hochberg over p_raw (FDR)
    n_permutations: int
    null_max: np.ndarray          # (B,) max statistic per permutation (diagnostic)

    def summary(self, alpha: float = 0.05) -> str:
        return (f"{len(self.observed)} candidates, B={self.n_permutations}: "
                f"{int((self.p_adj < alpha).sum())} significant at FWER {alpha}, "
                f"{int((self.p_bh < alpha).sum())} at FDR {alpha}")


def pvalues_from_permutations(observed: np.ndarray, perm: np.ndarray) -> PermutationResult:
    """
    observed: (m,) statistics; perm: (B, m) statistics under permutations
    (larger = more extreme).  Implements raw p, Westfall-Young max-T  and BH.
    """
    observed = np.asarray(observed, dtype=float)
    perm = np.asarray(perm, dtype=float)
    B, m = perm.shape
    # raw p-values
    p_raw = ((perm >= observed[None, :]).sum(axis=0) + 1) / (B + 1)
    # Westfall-Young single-step max-T: compare each observed statistic with the
    # distribution of the maximum over candidates.  Statistics are on a common
    # scale (-log10 p or AUC), so max-T is appropriate and, unlike min-P from the
    # same permutations, not anti-conservative when B is small relative to m.
    null_max = perm.max(axis=1)
    p_adj = ((null_max[:, None] >= observed[None, :]).sum(axis=0) + 1) / (B + 1)
    # BH
    p_bh = bh_adjust(p_raw)
    return PermutationResult(observed, p_raw, p_adj, p_bh, B, null_max)


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    idx = np.argsort(p)
    ranked = p[idx] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[idx] = np.minimum(ranked, 1.0)
    return out


def permutation_test_candidates(
    masks: Sequence[np.ndarray],
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    data: np.ndarray,
    n_permutations: int = 500,
    random_state: Optional[int] = None,
    batch_fn: Optional[Callable[[np.ndarray, Sequence[np.ndarray]], np.ndarray]] = None,
    verbose: bool = False,
) -> PermutationResult:
    """
    Evaluate `stat_fn(data_perm, mask)` for every candidate mask under
    `n_permutations` row permutations of `data` (the view NOT used to select
    the candidates).

    batch_fn(data_perm, masks) -> (m,) may be supplied for a vectorized
    implementation; otherwise stat_fn is called per mask.
    """
    rng = make_rng(random_state)
    masks = [np.asarray(m, dtype=bool) for m in masks]
    m = len(masks)
    if m == 0:
        return PermutationResult(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), n_permutations, np.zeros(n_permutations))

    def evaluate(d):
        if batch_fn is not None:
            return np.asarray(batch_fn(d, masks), dtype=float)
        return np.array([stat_fn(d, mk) for mk in masks], dtype=float)

    observed = evaluate(data)
    n = data.shape[0]
    perm = np.empty((n_permutations, m))
    for b in range(n_permutations):
        idx = rng.permutation(n)
        perm[b] = evaluate(data[idx])
        if verbose and (b + 1) % 100 == 0:
            print(f"  permutation {b + 1}/{n_permutations}")
    return pvalues_from_permutations(observed, perm)


# =============================================================================
# VECTORIZED HOTELLING FOR MANY MASKS
# =============================================================================

def y_shift_batch(Y: np.ndarray, masks: Sequence[np.ndarray]) -> np.ndarray:
    """-log10 p of Hotelling T² for many masks; shares the pooled-covariance work.

    For speed the within-group covariance is replaced by the total covariance
    (identical under H0 up to O(1/n) and standard in multivariate screening);
    `hotelling_t2` is used for the exact per-subgroup report."""
    Y = np.asarray(Y, dtype=float)
    n, p = Y.shape
    if p == 0:
        return np.zeros(len(masks))
    mu = Y.mean(axis=0)
    Yc = Y - mu
    S = Yc.T @ Yc / (n - 1)
    Sinv = np.linalg.pinv(S, rcond=1e-10)
    rank = int(np.linalg.matrix_rank(S, tol=1e-10))
    M = np.array([mk.astype(float) for mk in masks])          # (m, n)
    n1 = M.sum(axis=1)
    n2 = n - n1
    sums = M @ Yc                                              # (m, p) sums of centered Y in G
    # mean difference between G and complement: sums/n1 - (-sums)/n2 = sums * (1/n1 + 1/n2)
    valid = (n1 >= 2) & (n2 >= 2)
    d = np.zeros_like(sums)
    d[valid] = sums[valid] * (1.0 / n1[valid] + 1.0 / n2[valid])[:, None]
    t2 = (n1 * n2 / n) * np.einsum("ij,jk,ik->i", d, Sinv, d)
    out = np.zeros(len(masks))
    df2 = n1 + n2 - rank - 1
    for i in range(len(masks)):
        if not valid[i] or rank == 0:
            continue
        if df2[i] <= 0:
            pv = sps.chi2.sf(t2[i], rank)
        else:
            F = t2[i] * df2[i] / (rank * (n - 2))
            pv = sps.f.sf(F, rank, df2[i])
        out[i] = -np.log10(max(float(pv), 1e-300))
    return out


# =============================================================================
# HIGH-LEVEL: VALIDATE A SET OF SUBGROUPS
# =============================================================================

def validate_subgroups(
    subgroups,
    Y_space: Optional[FeatureSpace] = None,
    X_space: Optional[FeatureSpace] = None,
    statistic: str = "y_shift",
    n_permutations: int = 500,
    random_state: Optional[int] = None,
    estimator=None,
    n_folds: int = 5,
    alpha: float = 0.05,
    selection_note: str = "",
    verbose: bool = False,
) -> PermutationResult:
    """
    Fill p_value / p_value_corrected / quality_measures of every subgroup in
    `subgroups` (an iterable of Subgroup) with permutation inference.

    statistic:
        'y_shift'  -> Hotelling T² on Y_space.numeric, rows of Y permuted
        'x_auc'    -> out-of-fold AUC on X_space.numeric, rows of X permuted
    Also stores the exact Hotelling T² and its asymptotic p-value per subgroup
    (quality_measures['t2'], ['p_asymptotic']) when Y_space is given.
    """
    sgs = list(subgroups)
    if not sgs:
        return PermutationResult(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), n_permutations, np.zeros(0))
    n = (Y_space or X_space).n
    masks = [mask_from_indices(sg.covered_instances, n) for sg in sgs]
    rng = make_rng(random_state)

    if statistic == "y_shift":
        if Y_space is None:
            raise ValueError("y_shift needs Y_space")
        res = permutation_test_candidates(masks, None, Y_space.numeric, n_permutations, derive_seed(rng),
                                          batch_fn=y_shift_batch, verbose=verbose)
        permuted = "Y"
    elif statistic == "x_auc":
        if X_space is None:
            raise ValueError("x_auc needs X_space")
        if estimator is None:
            from sklearn.tree import DecisionTreeClassifier
            estimator = DecisionTreeClassifier(max_depth=3, random_state=derive_seed(rng))
        seed = derive_seed(rng)

        def stat(Xd, mk):
            return x_separability_statistic(Xd, mk, estimator, n_folds, seed)

        res = permutation_test_candidates(masks, stat, X_space.numeric, n_permutations, derive_seed(rng), verbose=verbose)
        permuted = "X"
    else:
        raise ValueError(f"Unknown statistic {statistic}")

    for sg, mk, obs, pr, pm, pb in zip(sgs, masks, res.observed, res.p_raw, res.p_adj, res.p_bh):
        sg.p_value = float(pr)
        sg.p_value_corrected = float(pm)
        sg.correction_method = "westfall_young_maxt"
        sg.quality_measures["test_statistic"] = float(obs)
        sg.quality_measures["p_bh"] = float(pb)
        sg.quality_measures["inference"] = f"{statistic}; permuted={permuted}; B={n_permutations}" + (f"; {selection_note}" if selection_note else "")
        if Y_space is not None:
            t2, F, pa = hotelling_t2(Y_space.numeric, mk)
            sg.quality_measures["t2"] = float(t2)
            sg.quality_measures["p_asymptotic"] = float(pa)
    return res


# =============================================================================
# SEARCH-LEVEL PERMUTATION ("swap randomization")
# =============================================================================

def search_permutation_pvalues(
    qualities: Sequence[float],
    search_fn: Callable[[np.ndarray], float],
    Y: np.ndarray,
    n_permutations: int = 100,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Family-wise p-values that account for the WHOLE search, not only for the
    returned candidates: permute the rows of Y, rerun the complete search and
    record the best quality found; p_i = P(best permuted quality >= q_i).

    Args:
        qualities: search scores of the returned subgroups (on the original data)
        search_fn: Y_permuted -> best quality found by rerunning the search
        Y: the matrix whose rows are permuted (the view that was not used to
           define the candidate descriptions)
    Returns:
        (p_values, null_best) with p_values on the same scale as max-T adjusted
        p-values (they ARE family-wise adjusted) and the (B,) null distribution.
    """
    rng = make_rng(random_state)
    q = np.asarray(qualities, dtype=float)
    n = Y.shape[0]
    null_best = np.empty(n_permutations)
    for b in range(n_permutations):
        null_best[b] = float(search_fn(Y[rng.permutation(n)]))
        if verbose and (b + 1) % 20 == 0:
            print(f"  search permutation {b + 1}/{n_permutations}")
    p = ((null_best[:, None] >= q[None, :]).sum(axis=0) + 1) / (n_permutations + 1)
    return p, null_best
