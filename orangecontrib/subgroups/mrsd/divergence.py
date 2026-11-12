"""
f-Divergence functions and permutation testing for contingency table analysis.
Used in MRSD-PT for evaluating independence between partitions and for
cell-level (subgroup-level) statistics.

Notes on the generators (f convex, f(1)=0): two generators that differ by an
affine term c*(t-1) give IDENTICAL statistics sum_ij E_ij f(O_ij/E_ij), because
sum_ij E_ij (O_ij/E_ij - 1) = 0.  Hence 'squared_hellinger' == 'hellinger' and
'pearson' (f=(t-1)^2/t) == 'neyman'; they are kept as aliases only.
Generators with f(0+) = inf (reverse_kl, neyman) are clipped at `eps`; with
empty cells their value is dominated by the clipping and they should be avoided
for sparse tables (a warning is issued).
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings


class FDivergenceType(Enum):
    """Enumeration of available f-divergence functions."""
    TOTAL_VARIATION = "total_variation"
    KL_DIVERGENCE = "kl_divergence"
    REVERSE_KL = "reverse_kl"
    CHI_SQUARED = "chi_squared"
    HELLINGER = "hellinger"
    JENSEN_SHANNON = "jensen_shannon"
    PEARSON = "pearson"
    NEYMAN = "neyman"
    SQUARED_HELLINGER = "squared_hellinger"
    ALPHA_DIVERGENCE = "alpha_divergence"  # Parameterized by alpha


# =============================================================================
# F-DIVERGENCE FUNCTIONS
# All functions f must satisfy: f convex, f(1) = 0
# =============================================================================

def total_variation(t: np.ndarray) -> np.ndarray:
    """
    Total Variation distance.
    f(t) = |t - 1|
    """
    return np.abs(t - 1)


def kl_divergence(t: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Kullback-Leibler divergence.
    f(t) = t * ln(t)
    
    Note: f(0) = 0 by convention (limit as t -> 0+)
    """
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)
    mask = t > eps
    result[mask] = t[mask] * np.log(t[mask])
    return result


def reverse_kl(t: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Reverse KL divergence.
    f(t) = -ln(t)
    """
    t = np.asarray(t, dtype=float)
    t = np.clip(t, eps, None)
    return -np.log(t)


def chi_squared(t: np.ndarray) -> np.ndarray:
    """
    Chi-squared divergence (Pearson).
    f(t) = (t - 1)^2
    """
    return (t - 1) ** 2


def hellinger(t: np.ndarray) -> np.ndarray:
    """
    Hellinger distance.
    f(t) = (sqrt(t) - 1)^2
    """
    return (np.sqrt(np.maximum(t, 0)) - 1) ** 2


def squared_hellinger(t: np.ndarray) -> np.ndarray:
    """
    Squared Hellinger (alternative formulation).
    f(t) = 2 * (1 - sqrt(t))
    """
    return 2 * (1 - np.sqrt(np.maximum(t, 0)))


def jensen_shannon(t: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Jensen-Shannon divergence (symmetric).
    f(t) = t * ln(2t / (t + 1)) + ln(2 / (t + 1))
    
    This is symmetric: D_JS(P||Q) = D_JS(Q||P)
    """
    t = np.asarray(t, dtype=float)
    t = np.clip(t, eps, None)
    
    term1 = t * np.log(2 * t / (t + 1))
    term2 = np.log(2 / (t + 1))
    
    return term1 + term2


def pearson_divergence(t: np.ndarray) -> np.ndarray:
    """
    Pearson chi-squared divergence.
    f(t) = (t - 1)^2 / t
    """
    t = np.asarray(t, dtype=float)
    t = np.clip(t, 1e-10, None)
    return (t - 1) ** 2 / t


def neyman_divergence(t: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Neyman chi-squared divergence.
    f(t) = (1 - t)^2 / t = 1/t - 2 + t
    """
    t = np.asarray(t, dtype=float)
    t = np.clip(t, eps, None)
    return (1 - t) ** 2 / t


def alpha_divergence(t: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Alpha-divergence (Rényi family).
    f(t) = (t^alpha - alpha*t - 1 + alpha) / (alpha * (alpha - 1))
    
    Special cases:
        alpha -> 0: reverse KL
        alpha -> 1: KL divergence
        alpha = 0.5: Hellinger
        alpha = 2: Chi-squared
    """
    if np.isclose(alpha, 0) or np.isclose(alpha, 1):
        # Use limit forms
        if np.isclose(alpha, 1):
            return kl_divergence(t)
        else:
            return reverse_kl(t)
    
    t = np.asarray(t, dtype=float)
    t = np.clip(t, 1e-10, None)
    
    return (t ** alpha - alpha * t - 1 + alpha) / (alpha * (alpha - 1))


# =============================================================================
# F-DIVERGENCE REGISTRY
# =============================================================================

class FDivergenceRegistry:
    """Registry of f-divergence functions with metadata."""
    
    _functions: Dict[FDivergenceType, Callable] = {
        FDivergenceType.TOTAL_VARIATION: total_variation,
        FDivergenceType.KL_DIVERGENCE: kl_divergence,
        FDivergenceType.REVERSE_KL: reverse_kl,
        FDivergenceType.CHI_SQUARED: chi_squared,
        FDivergenceType.HELLINGER: hellinger,
        FDivergenceType.JENSEN_SHANNON: jensen_shannon,
        FDivergenceType.PEARSON: pearson_divergence,
        FDivergenceType.NEYMAN: neyman_divergence,
        FDivergenceType.SQUARED_HELLINGER: squared_hellinger,
        FDivergenceType.ALPHA_DIVERGENCE: alpha_divergence,
    }

    # generators that are affine-equivalent (same statistic)
    _aliases: Dict[FDivergenceType, FDivergenceType] = {
        FDivergenceType.SQUARED_HELLINGER: FDivergenceType.HELLINGER,
        FDivergenceType.PEARSON: FDivergenceType.NEYMAN,
    }
    # generators unbounded at 0 (sensitive to empty cells)
    _unbounded_at_zero = {FDivergenceType.REVERSE_KL, FDivergenceType.NEYMAN, FDivergenceType.PEARSON}
    
    _descriptions: Dict[FDivergenceType, str] = {
        FDivergenceType.TOTAL_VARIATION: "Total Variation: f(t) = |t - 1|",
        FDivergenceType.KL_DIVERGENCE: "KL Divergence: f(t) = t * ln(t)",
        FDivergenceType.REVERSE_KL: "Reverse KL: f(t) = -ln(t)",
        FDivergenceType.CHI_SQUARED: "Chi-squared: f(t) = (t - 1)²",
        FDivergenceType.HELLINGER: "Hellinger: f(t) = (√t - 1)²",
        FDivergenceType.JENSEN_SHANNON: "Jensen-Shannon: symmetric KL variant",
        FDivergenceType.PEARSON: "Pearson: f(t) = (t - 1)² / t",
        FDivergenceType.NEYMAN: "Neyman: f(t) = (1 - t)² / t",
        FDivergenceType.SQUARED_HELLINGER: "Squared Hellinger: f(t) = 2(1 - √t) (alias of hellinger)",
        FDivergenceType.ALPHA_DIVERGENCE: "Alpha divergence (alpha=0.5 unless partial applied)",
    }

    @classmethod
    def canonical(cls, divergence_type: FDivergenceType) -> FDivergenceType:
        """Canonical representative of affine-equivalent generators."""
        return cls._aliases.get(divergence_type, divergence_type)

    @classmethod
    def is_unbounded_at_zero(cls, divergence_type: FDivergenceType) -> bool:
        return divergence_type in cls._unbounded_at_zero

    @classmethod
    def distinct_divergences(cls) -> List[str]:
        """Names of generators that yield distinct statistics."""
        return [d.value for d in cls._functions if d not in cls._aliases]
    
    @classmethod
    def get_function(cls, divergence_type: FDivergenceType) -> Callable:
        """Get f-divergence function by type."""
        return cls._functions[divergence_type]
    
    @classmethod
    def get_all_functions(cls) -> Dict[FDivergenceType, Callable]:
        """Get all registered f-divergence functions."""
        return cls._functions.copy()
    
    @classmethod
    def available_divergences(cls) -> List[str]:
        """List available divergence names."""
        return [d.value for d in cls._functions.keys()]
    
    @classmethod
    def get_description(cls, divergence_type: FDivergenceType) -> str:
        """Get description of a divergence function."""
        return cls._descriptions.get(divergence_type, "No description available")


# =============================================================================
# CONTINGENCY TABLE ANALYSIS
# =============================================================================

@dataclass
class ContingencyTable:
    """
    Contingency table for two partitions.
    
    Attributes:
        table: 2D array of counts (k x l)
        row_marginals: Row sums
        col_marginals: Column sums
        total: Total count
        k: Number of rows (clusters in X)
        l: Number of columns (clusters in Y)
    """
    table: np.ndarray
    row_marginals: np.ndarray
    col_marginals: np.ndarray
    total: int
    k: int
    l: int
    
    @classmethod
    def from_labels(cls, labels_x: np.ndarray, labels_y: np.ndarray) -> 'ContingencyTable':
        """
        Create contingency table from two label arrays.
        
        Args:
            labels_x: Cluster assignments for X partition
            labels_y: Cluster assignments for Y partition
            
        Returns:
            ContingencyTable instance
        """
        labels_x = np.asarray(labels_x, dtype=int)
        labels_y = np.asarray(labels_y, dtype=int)
        k = int(labels_x.max()) + 1
        l = int(labels_y.max()) + 1
        
        # Build contingency table (vectorized)
        table = np.bincount(labels_x * l + labels_y, minlength=k * l).reshape(k, l)
        
        row_marginals = table.sum(axis=1)
        col_marginals = table.sum(axis=0)
        total = table.sum()
        
        return cls(
            table=table,
            row_marginals=row_marginals,
            col_marginals=col_marginals,
            total=total,
            k=k,
            l=l
        )
    
    def get_cell_count(self, i: int, j: int) -> int:
        """Get count for cell (i, j)."""
        return self.table[i, j]
    
    def get_expected_under_independence(self) -> np.ndarray:
        """
        Calculate expected frequencies under independence assumption.
        E[i,j] = (row_i * col_j) / total
        """
        expected = np.outer(self.row_marginals, self.col_marginals) / self.total
        return expected


def compute_f_divergence_statistic(
    contingency: ContingencyTable,
    f_func: Callable,
    eps: float = 1e-10
) -> float:
    """
    Compute f-divergence statistic for a contingency table.
    
    The statistic measures deviation from independence:
    D_f = sum_{i,j} E[i,j] * f(O[i,j] / E[i,j])
    
    where O is observed counts, E is expected counts under independence.
    
    Args:
        contingency: ContingencyTable instance
        f_func: f-divergence function
        eps: Small value to avoid division by zero
        
    Returns:
        f-divergence statistic value
    """
    observed = contingency.table.astype(float)
    expected = contingency.get_expected_under_independence()
    
    # Avoid division by zero
    expected = np.maximum(expected, eps)
    
    # Compute ratio t = O/E
    ratio = observed / expected
    
    # Compute f-divergence: sum of E * f(O/E)
    divergence = np.sum(expected * f_func(ratio))
    
    return divergence


def cell_f_divergence(contingency: ContingencyTable, f_func: Callable, eps: float = 1e-10) -> np.ndarray:
    """Per-cell contribution E_ij f(O_ij/E_ij) (the table statistic is their sum)."""
    observed = contingency.table.astype(float)
    expected = np.maximum(contingency.get_expected_under_independence(), eps)
    return expected * f_func(observed / expected)


def cell_standardized_residuals(contingency: ContingencyTable) -> np.ndarray:
    """Adjusted (Haberman) standardized residuals (O-E)/sqrt(E(1-r_i/N)(1-c_j/N))."""
    O = contingency.table.astype(float)
    E = contingency.get_expected_under_independence()
    N = float(contingency.total)
    r = contingency.row_marginals / N
    c = contingency.col_marginals / N
    denom = np.sqrt(np.maximum(E, 1e-12) * np.outer(1 - r, 1 - c))
    return (O - E) / np.maximum(denom, 1e-12)


def tables_for_permutations(labels_x: np.ndarray, labels_y: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """
    Contingency tables for many permutations at once.

    perms: (B, n) array of permuted indices applied to labels_y.
    Returns (B, k, l) integer array.  Uses one-hot matrix products: O(B*n*(k+l)).
    """
    labels_x = np.asarray(labels_x, dtype=int)
    labels_y = np.asarray(labels_y, dtype=int)
    k = int(labels_x.max()) + 1
    l = int(labels_y.max()) + 1
    B, n = perms.shape
    Ly = labels_y[perms]                                  # (B, n)
    codes = labels_x[None, :] * l + Ly                    # (B, n)
    out = np.zeros((B, k * l), dtype=int)
    for b in range(B):
        out[b] = np.bincount(codes[b], minlength=k * l)
    return out.reshape(B, k, l)


def cell_statistics_batch(tables: np.ndarray, f_func: Callable, eps: float = 1e-10) -> np.ndarray:
    """Per-cell E f(O/E) for a (B, k, l) batch of tables -> (B, k, l)."""
    O = tables.astype(float)
    N = O.sum(axis=(1, 2), keepdims=True)
    r = O.sum(axis=2, keepdims=True)
    c = O.sum(axis=1, keepdims=True)
    E = np.maximum(r * c / np.maximum(N, 1), eps)
    return E * f_func(O / E)


# =============================================================================
# PERMUTATION TEST
# =============================================================================

@dataclass
class PermutationTestResult:
    """
    Result of a permutation test.
    
    Attributes:
        observed_statistic: The observed f-divergence statistic
        p_value: Estimated p-value
        n_permutations: Number of permutations performed
        permutation_statistics: Array of statistics from permutations
        f_divergence_type: Type of f-divergence used
    """
    observed_statistic: float
    p_value: float
    n_permutations: int
    permutation_statistics: np.ndarray
    f_divergence_type: FDivergenceType
    
    @property
    def is_significant(self) -> bool:
        """Check if result is significant at 0.05 level."""
        return self.p_value < 0.05
    
    def is_significant_at(self, alpha: float) -> bool:
        """Check if result is significant at given alpha level."""
        return self.p_value < alpha


def permutation_test(
    labels_x: np.ndarray,
    labels_y: np.ndarray,
    f_divergence: FDivergenceType = FDivergenceType.CHI_SQUARED,
    n_permutations: int = 10000,
    random_state: Optional[int] = None
) -> PermutationTestResult:
    """
    Perform permutation test for independence between two partitions.
    
    Tests H0: X and Y partitions are independent
    vs H1: X and Y partitions are dependent
    
    Args:
        labels_x: Cluster assignments for X partition
        labels_y: Cluster assignments for Y partition
        f_divergence: Type of f-divergence to use
        n_permutations: Number of permutations (default: 10000)
        random_state: Random seed for reproducibility
        
    Returns:
        PermutationTestResult with p-value and statistics
    """
    rng = np.random.default_rng(random_state)
    
    # Get f-function
    f_func = FDivergenceRegistry.get_function(f_divergence)
    
    # Compute observed statistic
    contingency = ContingencyTable.from_labels(labels_x, labels_y)
    observed_stat = compute_f_divergence_statistic(contingency, f_func)
    
    # Permutation distribution (vectorized, in chunks to bound memory)
    n = len(labels_x)
    perm_statistics = np.zeros(n_permutations)
    chunk = max(1, min(n_permutations, 2_000_000 // max(n, 1)))
    done = 0
    while done < n_permutations:
        b = min(chunk, n_permutations - done)
        perms = np.argsort(rng.random((b, n)), axis=1)
        tables = tables_for_permutations(labels_x, labels_y, perms)
        perm_statistics[done:done + b] = cell_statistics_batch(tables, f_func).sum(axis=(1, 2))
        done += b
    
    # Compute p-value (proportion of permutation stats >= observed)
    p_value = (np.sum(perm_statistics >= observed_stat) + 1) / (n_permutations + 1)
    
    return PermutationTestResult(
        observed_statistic=observed_stat,
        p_value=p_value,
        n_permutations=n_permutations,
        permutation_statistics=perm_statistics,
        f_divergence_type=f_divergence
    )


def permutation_test_all_divergences(
    labels_x: np.ndarray,
    labels_y: np.ndarray,
    n_permutations: int = 10000,
    random_state: Optional[int] = None
) -> Dict[FDivergenceType, PermutationTestResult]:
    """
    Run permutation test with all available f-divergences.
    
    Shares the same permutations across all divergences for efficiency.
    
    Returns:
        Dictionary mapping divergence type to test result
    """
    rng = np.random.default_rng(random_state)
    
    results = {}
    n = len(labels_x)
    
    # Generate permutations once
    perms = np.argsort(rng.random((n_permutations, n)), axis=1)
    tables = tables_for_permutations(labels_x, labels_y, perms)
    contingency = ContingencyTable.from_labels(labels_x, labels_y)
    
    for div_type, f_func in FDivergenceRegistry.get_all_functions().items():
        if div_type in FDivergenceRegistry._aliases:
            continue
        observed_stat = compute_f_divergence_statistic(contingency, f_func)
        perm_stats = cell_statistics_batch(tables, f_func).sum(axis=(1, 2))
        
        # P-value
        p_value = (np.sum(perm_stats >= observed_stat) + 1) / (n_permutations + 1)
        
        results[div_type] = PermutationTestResult(
            observed_statistic=observed_stat,
            p_value=p_value,
            n_permutations=n_permutations,
            permutation_statistics=perm_stats,
            f_divergence_type=div_type
        )
    
    return results


# =============================================================================
# P-VALUE CORRECTIONS
# =============================================================================

class PValueCorrection(Enum):
    """Available p-value correction methods."""
    NONE = "none"
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BH = "benjamini_hochberg"  # FDR
    BY = "benjamini_yekutieli"  # FDR for dependent tests


def correct_pvalues(
    pvalues: np.ndarray,
    method: PValueCorrection = PValueCorrection.BH,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple testing correction to p-values.
    
    Args:
        pvalues: Array of p-values
        method: Correction method (default: Benjamini-Hochberg FDR)
        alpha: Significance level (used for some methods)
        
    Returns:
        Tuple of (corrected_pvalues, rejected_hypotheses)
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    
    if method == PValueCorrection.NONE:
        return pvalues.copy(), pvalues < alpha
    
    elif method == PValueCorrection.BONFERRONI:
        corrected = np.minimum(pvalues * n, 1.0)
        return corrected, corrected < alpha
    
    elif method == PValueCorrection.HOLM:
        # Holm step-down procedure
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        
        corrected = np.zeros(n)
        for i, p in enumerate(sorted_pvals):
            corrected[sorted_idx[i]] = min(p * (n - i), 1.0)
        
        # Ensure monotonicity
        corrected_sorted = corrected[sorted_idx]
        for i in range(1, n):
            corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i-1])
        corrected[sorted_idx] = corrected_sorted
        
        return corrected, corrected < alpha
    
    elif method == PValueCorrection.BH:
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        
        corrected = np.zeros(n)
        for i, p in enumerate(sorted_pvals):
            corrected[sorted_idx[i]] = p * n / (i + 1)
        
        # Ensure monotonicity (going backwards)
        corrected_sorted = corrected[sorted_idx]
        for i in range(n - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
        corrected[sorted_idx] = corrected_sorted
        corrected = np.minimum(corrected, 1.0)
        
        return corrected, corrected < alpha
    
    elif method == PValueCorrection.BY:
        # Benjamini-Yekutieli (conservative FDR for dependent tests)
        c_n = np.sum(1.0 / np.arange(1, n + 1))  # Harmonic number
        
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        
        corrected = np.zeros(n)
        for i, p in enumerate(sorted_pvals):
            corrected[sorted_idx[i]] = p * n * c_n / (i + 1)
        
        # Ensure monotonicity
        corrected_sorted = corrected[sorted_idx]
        for i in range(n - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
        corrected[sorted_idx] = corrected_sorted
        corrected = np.minimum(corrected, 1.0)
        
        return corrected, corrected < alpha
    
    else:
        raise ValueError(f"Unknown correction method: {method}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("F-Divergence Functions Test")
    print("=" * 50)
    
    # Test that all f(1) = 0
    t_test = np.array([1.0])
    for div_type in FDivergenceRegistry.available_divergences():
        div_enum = FDivergenceType(div_type)
        f = FDivergenceRegistry.get_function(div_enum)
        val = f(t_test)[0]
        print(f"{div_type}: f(1) = {val:.6f} (should be ~0)")
    
    print("\n" + "=" * 50)
    print("Permutation Test Example")
    print("=" * 50)
    
    # Create dependent partitions (non-independent)
    n = 200
    labels_x = np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])
    # Make Y dependent on X
    labels_y = np.where(
        labels_x == 0,
        np.random.choice([0, 1], n, p=[0.8, 0.2]),
        np.random.choice([0, 1], n, p=[0.2, 0.8])
    )
    
    # Run permutation test
    result = permutation_test(
        labels_x, labels_y,
        f_divergence=FDivergenceType.CHI_SQUARED,
        n_permutations=1000,
        random_state=42
    )
    
    print(f"Observed statistic: {result.observed_statistic:.4f}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Significant at 0.05: {result.is_significant}")
    
    # Contingency table
    ct = ContingencyTable.from_labels(labels_x, labels_y)
    print(f"\nContingency table:\n{ct.table}")
    
    print("\n" + "=" * 50)
    print("P-value Correction Example")
    print("=" * 50)
    
    # Multiple p-values
    pvals = np.array([0.001, 0.01, 0.03, 0.04, 0.05, 0.1, 0.2])
    
    for method in PValueCorrection:
        corrected, rejected = correct_pvalues(pvals, method)
        print(f"\n{method.value}:")
        print(f"  Original:  {pvals}")
        print(f"  Corrected: {np.round(corrected, 4)}")
        print(f"  Rejected:  {rejected}")
