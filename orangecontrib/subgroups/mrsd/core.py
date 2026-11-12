"""
Core data structures for MRSD (Multiple Responses Subgroup Discovery).

Conventions shared by all algorithms:
- `covered_instances` is the set of row indices of the subgroup.
- `rule` (rules.Rule) is the description in X space, in original units; for
  algorithms that produce index sets it is a post-hoc approximation and
  `description_precision` / `description_recall` say how well it matches.
- `quality_measures['quality']` is the algorithm's own search score (higher is
  better; scale is algorithm specific).
- `p_value` / `p_value_corrected` are set ONLY by a real statistical test
  (see inference.py); an algorithm that has not run a test leaves them None.
"""

from typing import List, Dict, Set, Callable, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Subgroup:
    """
    Represents a single subgroup in the data.
    
    A subgroup is defined by:
    - a description (rule/condition)
    - covered instances (row indices)
    - quality measures
    
    For MRSD-PT algorithm, additional attributes store:
    - clustering information (algorithm, k, l, cell indices)
    - f-divergence values
    - p-values (raw and corrected)
    - algorithm-specific metadata (medoids, centroids, etc.)
    """
    
    # Description of the subgroup (e.g., "X_cluster=2, Y_cluster=1")
    description: str
    
    # Indices of instances covered by this subgroup
    covered_instances: Set[int] = field(default_factory=set)
    
    # Quality measures (e.g., WRAcc, coverage, significance, ...)
    quality_measures: Dict[str, Any] = field(default_factory=dict)
    
    # Additional attributes for multi-target discovery
    target_variables: List[str] = field(default_factory=list)
    
    # Statistics for each target variable
    target_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Which algorithm produced the subgroup
    algorithm: Optional[str] = None

    # Description in X space (rules.Rule); for index-set methods a post-hoc
    # approximation with the precision/recall of that approximation
    rule: Optional[Any] = None
    description_precision: Optional[float] = None
    description_recall: Optional[float] = None
    
    # =========================================================================
    # MRSD-PT SPECIFIC ATTRIBUTES
    # =========================================================================
    
    # Clustering configuration for X (independent variables)
    clustering_X: Optional[str] = None  # Algorithm name (e.g., "kmeans", "kmedoids")
    k: Optional[int] = None  # Number of clusters in X partition
    cell_i: Optional[int] = None  # Row index in contingency table
    
    # Clustering configuration for Y (dependent variables)
    clustering_Y: Optional[str] = None  # Algorithm name
    l: Optional[int] = None  # Number of clusters in Y partition
    cell_j: Optional[int] = None  # Column index in contingency table
    
    # f-divergence information
    f_divergence_type: Optional[str] = None  # Type of f-divergence used
    f_divergence_value: Optional[float] = None  # Computed f-divergence statistic
    
    # P-values
    p_value: Optional[float] = None  # Raw p-value from permutation test
    p_value_corrected: Optional[float] = None  # Corrected p-value (FDR, Bonferroni, etc.)
    correction_method: Optional[str] = None  # P-value correction method used
    
    # Clustering metadata (algorithm-specific)
    clustering_metadata_X: Dict[str, Any] = field(default_factory=dict)
    clustering_metadata_Y: Dict[str, Any] = field(default_factory=dict)
    
    # For k-medoids: store medoid information
    medoid_X: Optional[Any] = None  # Medoid data point for X cluster
    medoid_Y: Optional[Any] = None  # Medoid data point for Y cluster
    medoid_index_X: Optional[int] = None  # Index of medoid in original data
    medoid_index_Y: Optional[int] = None  # Index of medoid in original data
    
    # For k-means: store centroid information
    centroid_X: Optional[Any] = None  # Centroid for X cluster
    centroid_Y: Optional[Any] = None  # Centroid for Y cluster
    
    # Contingency table cell information
    expected_count: Optional[float] = None  # Expected count under independence
    observed_count: Optional[int] = None  # Observed count in cell
    
    # Partition results (full clustering info if needed)
    partition_result_X: Optional[Any] = None  # PartitionResult for X
    partition_result_Y: Optional[Any] = None  # PartitionResult for Y
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.covered_instances, (list, tuple)):
            self.covered_instances = set(self.covered_instances)
    
    @property
    def coverage(self) -> int:
        """Number of covered instances."""
        return len(self.covered_instances)

    @property
    def quality(self) -> Optional[float]:
        """Canonical search score of the producing algorithm (None if not set)."""
        return self.quality_measures.get("quality")

    @property
    def indices_key(self) -> frozenset:
        return frozenset(self.covered_instances)

    def mask(self, n: int) -> np.ndarray:
        m = np.zeros(n, dtype=bool)
        if self.covered_instances:
            m[np.fromiter(self.covered_instances, dtype=int)] = True
        return m

    def fill_target_statistics(self, Y: pd.DataFrame) -> None:
        """Per-target summary inside the subgroup vs. the whole dataset."""
        mask = self.mask(len(Y))
        self.target_variables = list(Y.columns)
        for col in Y.columns:
            s = Y[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                v = s.to_numpy(dtype=float)
                sd_all = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                self.target_statistics[col] = {
                    "mean_subgroup": float(v[mask].mean()) if mask.any() else float("nan"),
                    "mean_all": float(v.mean()),
                    "std_subgroup": float(v[mask].std(ddof=1)) if mask.sum() > 1 else 0.0,
                    "std_all": sd_all,
                    "effect_size": float((v[mask].mean() - v.mean()) / sd_all) if sd_all > 0 and mask.any() else 0.0,
                }
            else:
                vc_all = s.value_counts(normalize=True)
                vc_sub = s[mask].value_counts(normalize=True)
                self.target_statistics[col] = {
                    f"p({lvl})_subgroup": float(vc_sub.get(lvl, 0.0)) for lvl in vc_all.index
                }
                self.target_statistics[col].update({f"p({lvl})_all": float(vc_all[lvl]) for lvl in vc_all.index})
    
    def get_quality(self, measure_name: str) -> float:
        """
        Get value of a quality measure.
        
        Args:
            measure_name: Name of the quality measure
            
        Returns:
            Value of measure or 0.0 if not exists
        """
        return self.quality_measures.get(measure_name, 0.0)
    
    def set_quality(self, measure_name: str, value: float) -> None:
        """Set value of a quality measure."""
        self.quality_measures[measure_name] = value
    
    def add_target_statistics(self, target: str, stats: Dict[str, float]) -> None:
        """
        Add statistics for a target variable.
        
        Args:
            target: Name of target variable
            stats: Dictionary with statistics (mean, std, ...)
        """
        self.target_statistics[target] = stats
    
    def get_cell_description(self) -> str:
        """Get description of the contingency table cell."""
        return f"Cell({self.cell_i}, {self.cell_j}) in {self.k}x{self.l} table"
    
    def get_clustering_summary(self) -> str:
        """Get summary of clustering configuration."""
        return (f"X: {self.clustering_X} (k={self.k}), "
                f"Y: {self.clustering_Y} (l={self.l})")
    
    def is_significant(self, alpha: float = 0.05, use_corrected: bool = True) -> bool:
        """
        Check if subgroup is statistically significant.
        
        Args:
            alpha: Significance level
            use_corrected: Use corrected p-value if available
            
        Returns:
            True if significant
        """
        if use_corrected and self.p_value_corrected is not None:
            return self.p_value_corrected < alpha
        elif self.p_value is not None:
            return self.p_value < alpha
        return False
    
    def __hash__(self):
        """For use in sets."""
        return hash((self.description, frozenset(self.covered_instances)))
    
    def __eq__(self, other):
        if not isinstance(other, Subgroup):
            return False
        return (self.description == other.description and 
                self.covered_instances == other.covered_instances)
    
    def __repr__(self):
        sig_str = ""
        if self.p_value is not None:
            sig_str = f", p={self.p_value:.4f}"
            if self.p_value_corrected is not None:
                sig_str += f", p_adj={self.p_value_corrected:.4f}"
        q = self.quality
        q_str = f", q={q:.3f}" if q is not None else ""
        desc = str(self.rule) if self.rule is not None and len(self.rule) > 0 else self.description
        return f"Subgroup('{desc}', n={self.coverage}{q_str}{sig_str})"


class SetOfSubgroups:
    """
    Collection of subgroups with methods for:
    - measuring similarity between subgroups
    - filtering/pruning the set
    - selecting best subgroups
    """
    
    def __init__(self, subgroups: Optional[List[Subgroup]] = None):
        """
        Initialize set of subgroups.
        
        Args:
            subgroups: List of subgroups (optional)
        """
        self._subgroups: List[Subgroup] = list(subgroups) if subgroups else []
    
    def __len__(self) -> int:
        return len(self._subgroups)
    
    def __iter__(self):
        return iter(self._subgroups)
    
    def __getitem__(self, index: int) -> Subgroup:
        return self._subgroups[index]
    
    def add(self, subgroup: Subgroup) -> None:
        """Add a subgroup to the set."""
        self._subgroups.append(subgroup)
    
    def remove(self, subgroup: Subgroup) -> None:
        """Remove a subgroup from the set."""
        self._subgroups.remove(subgroup)
    
    def copy(self) -> 'SetOfSubgroups':
        """Return a copy of the set."""
        return SetOfSubgroups([s for s in self._subgroups])
    
    def to_list(self) -> List[Subgroup]:
        """Return list of subgroups."""
        return list(self._subgroups)
    
    # =========================================================================
    # SIMILARITY MEASURES BETWEEN SUBGROUPS
    # =========================================================================
    
    @staticmethod
    def jaccard_similarity(sg1: Subgroup, sg2: Subgroup) -> float:
        """
        Jaccard index based on covered instances.
        
        s(G1, G2) = |G1 ∩ G2| / |G1 ∪ G2|
        
        Args:
            sg1: First subgroup
            sg2: Second subgroup
            
        Returns:
            Similarity between 0 and 1
        """
        intersection = len(sg1.covered_instances & sg2.covered_instances)
        union = len(sg1.covered_instances | sg2.covered_instances)
        
        if union == 0:
            return 0.0
        return intersection / union
    
    @staticmethod
    def overlap_coefficient(sg1: Subgroup, sg2: Subgroup) -> float:
        """
        Overlap coefficient (Szymkiewicz-Simpson).
        
        s(G1, G2) = |G1 ∩ G2| / min(|G1|, |G2|)
        """
        intersection = len(sg1.covered_instances & sg2.covered_instances)
        min_size = min(len(sg1.covered_instances), len(sg2.covered_instances))
        
        if min_size == 0:
            return 0.0
        return intersection / min_size
    
    @staticmethod
    def dice_coefficient(sg1: Subgroup, sg2: Subgroup) -> float:
        """
        Dice-Sørensen coefficient.
        
        s(G1, G2) = 2|G1 ∩ G2| / (|G1| + |G2|)
        """
        intersection = len(sg1.covered_instances & sg2.covered_instances)
        total = len(sg1.covered_instances) + len(sg2.covered_instances)
        
        if total == 0:
            return 0.0
        return 2 * intersection / total
    
    # =========================================================================
    # FILTERING/PRUNING ALGORITHMS
    # =========================================================================
    
    def filter_redundant(
        self,
        similarity_func: Callable[[Subgroup, Subgroup], float],
        quality_func: Callable[[Subgroup], float],
        s_min: float = 0.5
    ) -> 'SetOfSubgroups':
        """
        Default pruning procedure (Figure 7.1 in dissertation).
        
        Iteratively selects best subgroups and removes those
        that are too similar (greedy cover).
        
        Args:
            similarity_func: Function for measuring similarity s(G_i, G_j)
            quality_func: Function for measuring quality q(G)
            s_min: Similarity threshold (subgroups with similarity >= s_min are removed)
            
        Returns:
            Pruned set of subgroups G'
        """
        if not (0.0 < s_min <= 1.0):
            raise ValueError("s_min must be in (0, 1]")
        # G' ← ∅
        result = SetOfSubgroups()
        
        # Working copy G
        remaining = self.copy()
        
        # While |G| > 0
        while len(remaining) > 0:
            # G_i ← arg max_{G∈G} q(G)
            best_subgroup = max(remaining, key=quality_func)
            
            # G' ← G' ∪ {G_i}
            result.add(best_subgroup)
            
            # For G_j ∈ G do
            to_remove = []
            for sg_j in remaining:
                # If s(G_j, G_i) >= s_min, then G ← G \ {G_j}
                if similarity_func(sg_j, best_subgroup) >= s_min:
                    to_remove.append(sg_j)
            
            # Remove similar subgroups
            for sg in to_remove:
                remaining.remove(sg)
        
        # Return G'
        return result
    
    def deduplicate(self, quality_func: Optional[Callable[['Subgroup'], float]] = None) -> 'SetOfSubgroups':
        """Keep one subgroup per distinct set of covered instances (the best by
        quality_func, or the first one)."""
        best: Dict[frozenset, Subgroup] = {}
        for sg in self._subgroups:
            key = sg.indices_key
            if key not in best:
                best[key] = sg
            elif quality_func is not None and quality_func(sg) > quality_func(best[key]):
                best[key] = sg
        return SetOfSubgroups(list(best.values()))

    def prune(self, quality_func: Optional[Callable[['Subgroup'], float]] = None,
              s_min: float = 0.7) -> 'SetOfSubgroups':
        """Deduplicate, then greedy Jaccard pruning (default post-processing of every algorithm)."""
        qf = quality_func or (lambda sg: sg.quality if sg.quality is not None else 0.0)
        return self.deduplicate(qf).filter_redundant(self.jaccard_similarity, qf, s_min=s_min)

    def filter_by_quality(
        self,
        quality_func: Callable[[Subgroup], float],
        threshold: float
    ) -> 'SetOfSubgroups':
        """
        Filter subgroups by quality threshold.
        
        Args:
            quality_func: Function for measuring quality
            threshold: Minimum quality
            
        Returns:
            Filtered set
        """
        filtered = [sg for sg in self._subgroups if quality_func(sg) >= threshold]
        return SetOfSubgroups(filtered)
    
    def filter_significant(
        self,
        alpha: float = 0.05,
        use_corrected: bool = True
    ) -> 'SetOfSubgroups':
        """
        Filter to keep only statistically significant subgroups.
        
        Args:
            alpha: Significance level
            use_corrected: Use corrected p-values if available
            
        Returns:
            Filtered set with only significant subgroups
        """
        filtered = [sg for sg in self._subgroups 
                   if sg.is_significant(alpha, use_corrected)]
        return SetOfSubgroups(filtered)
    
    def filter_by_coverage(
        self,
        n_min: int,
        n_max: Optional[int] = None
    ) -> 'SetOfSubgroups':
        """
        Filter subgroups by coverage (number of instances).
        
        Args:
            n_min: Minimum coverage
            n_max: Maximum coverage (optional)
            
        Returns:
            Filtered set
        """
        filtered = [sg for sg in self._subgroups if sg.coverage >= n_min]
        if n_max is not None:
            filtered = [sg for sg in filtered if sg.coverage <= n_max]
        return SetOfSubgroups(filtered)
    
    def top_k(
        self,
        quality_func: Callable[[Subgroup], float],
        k: int
    ) -> 'SetOfSubgroups':
        """
        Return k best subgroups by quality.
        
        Args:
            quality_func: Function for measuring quality
            k: Number of subgroups
            
        Returns:
            Set of k best subgroups
        """
        sorted_subgroups = sorted(self._subgroups, key=quality_func, reverse=True)
        return SetOfSubgroups(sorted_subgroups[:k])
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_all_covered_instances(self) -> Set[int]:
        """Return all instances covered by at least one subgroup."""
        all_instances = set()
        for sg in self._subgroups:
            all_instances |= sg.covered_instances
        return all_instances
    
    def compute_pairwise_similarities(
        self,
        similarity_func: Callable[[Subgroup, Subgroup], float]
    ) -> np.ndarray:
        """
        Compute similarity matrix between all pairs of subgroups.
        
        Returns:
            Symmetric similarity matrix
        """
        n = len(self._subgroups)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    sim = similarity_func(self._subgroups[i], self._subgroups[j])
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        
        return sim_matrix
    
    def sort_by_quality(
        self,
        quality_func: Callable[[Subgroup], float],
        descending: bool = True
    ) -> 'SetOfSubgroups':
        """
        Return sorted copy of subgroups.
        
        Args:
            quality_func: Function for measuring quality
            descending: If True, best first
            
        Returns:
            Sorted set
        """
        sorted_sgs = sorted(self._subgroups, key=quality_func, reverse=descending)
        return SetOfSubgroups(sorted_sgs)
    
    def sort_by_pvalue(self, ascending: bool = True) -> 'SetOfSubgroups':
        """
        Sort subgroups by p-value.
        
        Args:
            ascending: If True, most significant first
            
        Returns:
            Sorted set
        """
        def get_pval(sg):
            if sg.p_value_corrected is not None:
                return sg.p_value_corrected
            elif sg.p_value is not None:
                return sg.p_value
            return 1.0
        
        sorted_sgs = sorted(self._subgroups, key=get_pval, reverse=not ascending)
        return SetOfSubgroups(sorted_sgs)
    
    def summary(self, alpha: float = 0.05) -> str:
        """Return brief summary of the set."""
        if not self._subgroups:
            return "Empty set of subgroups"
        
        coverages = [sg.coverage for sg in self._subgroups]
        tested = [sg for sg in self._subgroups if sg.p_value is not None]
        n_significant = sum(1 for sg in tested if sg.is_significant(alpha))
        
        lines = [
            f"Set of Subgroups:",
            f"  - Total subgroups: {len(self)}",
            (f"  - Significant (adjusted p < {alpha}): {n_significant} of {len(tested)} tested"
             if tested else "  - Significance: not tested (run inference.validate_subgroups)"),
            f"  - Average coverage: {np.mean(coverages):.2f}",
            f"  - Min/Max coverage: {min(coverages)}/{max(coverages)}"
        ]
        
        # P-value summary if available
        pvals = [sg.p_value for sg in self._subgroups if sg.p_value is not None]
        if pvals:
            lines.append(f"  - P-value range: {min(pvals):.4f} - {max(pvals):.4f}")
        
        return "\n".join(lines)
    
    def __repr__(self):
        return f"SetOfSubgroups(n={len(self)})"


# =============================================================================
# QUALITY MEASURES
# =============================================================================

class QualityMeasures:
    """
    Various quality measures for subgroups.
    Adapted for multi-target subgroup discovery.
    """
    
    @staticmethod
    def wracc(subgroup: Subgroup, total_instances: int, 
              target_mean_global: float, target_mean_subgroup: float) -> float:
        """
        Weighted Relative Accuracy.
        
        WRAcc = coverage/N * (mean_subgroup - mean_global)
        """
        coverage_ratio = subgroup.coverage / total_instances
        return coverage_ratio * (target_mean_subgroup - target_mean_global)
    
    @staticmethod
    def coverage_ratio(subgroup: Subgroup, total_instances: int) -> float:
        """Coverage ratio."""
        return subgroup.coverage / total_instances
    
    @staticmethod
    def aggregate_quality(subgroup: Subgroup, weights: Dict[str, float]) -> float:
        """
        Aggregate quality as weighted sum of measures.
        
        Args:
            subgroup: Subgroup
            weights: Weights for individual measures
            
        Returns:
            Aggregate quality value
        """
        total = 0.0
        for measure_name, weight in weights.items():
            total += weight * subgroup.get_quality(measure_name)
        return total
    
    @staticmethod
    def negative_pvalue(subgroup: Subgroup) -> float:
        """
        Return negative p-value (for maximization in filtering).
        Untested subgroups rank last.
        """
        if subgroup.p_value_corrected is not None:
            return -subgroup.p_value_corrected
        elif subgroup.p_value is not None:
            return -subgroup.p_value
        return -2.0

    @staticmethod
    def quality(subgroup: Subgroup) -> float:
        """Canonical search score; 0 if not set."""
        q = subgroup.quality
        return 0.0 if q is None else float(q)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create test subgroups with MRSD-PT attributes
    sg1 = Subgroup(
        description="X_cluster=0, Y_cluster=0",
        covered_instances={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        quality_measures={"wracc": 0.15},
        clustering_X="kmeans",
        clustering_Y="kmeans",
        k=3,
        l=2,
        cell_i=0,
        cell_j=0,
        f_divergence_type="chi_squared",
        f_divergence_value=12.5,
        p_value=0.001,
        p_value_corrected=0.01,
        correction_method="benjamini_hochberg"
    )
    
    sg2 = Subgroup(
        description="X_cluster=1, Y_cluster=0",
        covered_instances={0, 1, 2, 3, 4, 5, 10, 11, 12},
        quality_measures={"wracc": 0.12},
        clustering_X="kmedoids",
        clustering_Y="kmedoids",
        k=3,
        l=2,
        cell_i=1,
        cell_j=0,
        f_divergence_type="chi_squared",
        f_divergence_value=8.2,
        p_value=0.05,
        p_value_corrected=0.12,
        medoid_X=np.array([1.5, 2.3]),
        medoid_Y=np.array([0.5, 1.2]),
        medoid_index_X=5,
        medoid_index_Y=10
    )
    
    sg3 = Subgroup(
        description="X_cluster=2, Y_cluster=1",
        covered_instances={20, 21, 22, 23, 24},
        quality_measures={"wracc": 0.18},
        clustering_X="kmeans",
        clustering_Y="kmeans",
        k=3,
        l=2,
        cell_i=2,
        cell_j=1,
        p_value=0.02,
        p_value_corrected=0.06
    )
    
    # Create set
    subgroup_set = SetOfSubgroups([sg1, sg2, sg3])
    print(subgroup_set.summary())
    print()
    
    # Test significance filtering
    significant = subgroup_set.filter_significant(alpha=0.05)
    print(f"Significant subgroups: {len(significant)}")
    for sg in significant:
        print(f"  - {sg}")
    
    print()
    print("Subgroup details:")
    print(f"  {sg1.get_clustering_summary()}")
    print(f"  {sg1.get_cell_description()}")
    print(f"  Significant: {sg1.is_significant()}")
