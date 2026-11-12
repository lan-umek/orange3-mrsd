"""
Clustering algorithms for partitioning data (used by MRSD-PT and other
cluster-based MRSD algorithms).

Each algorithm returns a PartitionResult whose labels are always contiguous
0..k_actual-1 (empty clusters are removed and `k` reports the actual number),
so that downstream contingency tables never index a missing label.
Randomness goes through an explicit seed; the global numpy RNG is never touched.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score

from .common import FeatureSpace, make_rng


def _relabel(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make labels contiguous 0..k-1; returns (new_labels, old_ids_in_new_order)."""
    uniq, new = np.unique(labels, return_inverse=True)
    return new.astype(int), uniq


class ClusteringAlgorithm(Enum):
    """Enumeration of available clustering algorithms."""
    KMEANS = "kmeans"
    KMEDOIDS = "kmedoids"
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"
    NATURAL = "natural"  # Natural partitioning for categorical variables


@dataclass
class PartitionResult:
    """
    Result of a clustering/partitioning operation.
    
    Attributes:
        labels: Cluster assignment for each instance (array of length n)
        k: Number of clusters
        algorithm: Algorithm used for partitioning
        metadata: Algorithm-specific metadata (centroids, medoids, etc.)
        quality_score: Optional clustering quality metric (e.g., silhouette)
    """
    labels: np.ndarray
    k: int
    algorithm: ClusteringAlgorithm
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    
    def get_cluster_indices(self, cluster_id: int) -> np.ndarray:
        """Return indices of instances belonging to a specific cluster."""
        return np.where(self.labels == cluster_id)[0]
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """Return size of each cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def __repr__(self):
        q = f"{self.quality_score:.3f}" if self.quality_score is not None else "N/A"
        return f"PartitionResult(algorithm={self.algorithm.value}, k={self.k}, quality={q})"


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""
    
    algorithm: ClusteringAlgorithm
    
    @abstractmethod
    def fit(self, data: np.ndarray, k: int, **kwargs) -> PartitionResult:
        """
        Fit the clustering algorithm to data.
        
        Args:
            data: Data matrix (n_samples, n_features)
            k: Number of clusters
            **kwargs: Algorithm-specific parameters
            
        Returns:
            PartitionResult with cluster assignments and metadata
        """
        pass
    
    compute_silhouette: bool = False  # O(n^2); off by default

    def _compute_silhouette(self, data: np.ndarray, labels: np.ndarray) -> Optional[float]:
        """Compute silhouette score if requested and possible."""
        if not self.compute_silhouette:
            return None
        n_unique = len(np.unique(labels))
        if n_unique < 2 or n_unique >= len(labels):
            return None
        return float(silhouette_score(data, labels))


class KMeansClusterer(BaseClusterer):
    """K-Means clustering algorithm."""
    
    algorithm = ClusteringAlgorithm.KMEANS
    
    def fit(self, data: np.ndarray, k: int, 
            n_init: int = 10, 
            max_iter: int = 300,
            random_state: Optional[int] = None,
            standardize: bool = True,
            **kwargs) -> PartitionResult:
        """
        Fit K-Means clustering.
        
        Args:
            standardize: z-score the columns first (set False when the input is
                already a FeatureSpace.numeric matrix or weighted scores)
        Additional metadata:
            - centroids: Cluster centers
            - inertia: Within-cluster sum of squares
        """
        scaler = StandardScaler(with_mean=standardize, with_std=standardize)
        data_scaled = scaler.fit_transform(data)
        
        kmeans = KMeans(
            n_clusters=k, 
            n_init=n_init, 
            max_iter=max_iter,
            random_state=random_state
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ConvergenceWarning when fewer distinct points than k
            labels = kmeans.fit_predict(data_scaled)
        labels, kept = _relabel(labels)
        
        # Transform centroids back to original scale
        centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)[kept]
        
        metadata = {
            'centroids': centroids_original,
            'centroids_scaled': kmeans.cluster_centers_[kept],
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_
        }
        
        return PartitionResult(
            labels=labels,
            k=len(kept),
            algorithm=self.algorithm,
            metadata=metadata,
            quality_score=self._compute_silhouette(data_scaled, labels)
        )


class KMedoidsClusterer(BaseClusterer):
    """
    K-Medoids (PAM) clustering algorithm.
    Medoids are actual data points, unlike centroids in K-Means.
    """
    
    algorithm = ClusteringAlgorithm.KMEDOIDS
    
    def fit(self, data: np.ndarray, k: int,
            max_iter: int = 300,
            random_state: Optional[int] = None,
            **kwargs) -> PartitionResult:
        """
        Fit K-Medoids clustering using PAM algorithm.
        
        Additional metadata:
            - medoid_indices: Indices of medoid instances in original data
            - medoids: Actual medoid data points
            - total_cost: Sum of distances to medoids
        """
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        n_samples = data_scaled.shape[0]
        rng = make_rng(random_state)
        
        # Initialize medoids on distinct points (k-means++-like spread)
        distinct = np.unique(data_scaled, axis=0, return_index=True)[1]
        k_eff = min(k, len(distinct))
        medoid_indices = rng.choice(distinct, k_eff, replace=False)
        
        for iteration in range(max_iter):
            # Assign points to nearest medoid
            distances = self._compute_distances(data_scaled, data_scaled[medoid_indices])
            labels = np.argmin(distances, axis=1)
            
            # Update medoids
            new_medoid_indices = np.zeros(k_eff, dtype=int)
            for cluster_id in range(k_eff):
                cluster_mask = labels == cluster_id
                if not np.any(cluster_mask):
                    # Empty cluster - re-seed at the point farthest from its medoid
                    far = int(np.argmax(np.min(distances, axis=1)))
                    new_medoid_indices[cluster_id] = far
                    continue
                
                cluster_indices = np.where(cluster_mask)[0]
                cluster_data = data_scaled[cluster_mask]
                
                # Find point with minimum total distance to others in cluster
                intra_distances = self._compute_distances(cluster_data, cluster_data)
                total_distances = intra_distances.sum(axis=1)
                best_local_idx = np.argmin(total_distances)
                new_medoid_indices[cluster_id] = cluster_indices[best_local_idx]
            
            # Check for convergence
            if np.array_equal(medoid_indices, new_medoid_indices):
                break
            
            medoid_indices = new_medoid_indices
        
        # Final assignment
        distances = self._compute_distances(data_scaled, data_scaled[medoid_indices])
        labels = np.argmin(distances, axis=1)
        total_cost = np.min(distances, axis=1).sum()
        labels, kept = _relabel(labels)
        medoid_indices = medoid_indices[kept]
        
        metadata = {
            'medoid_indices': medoid_indices,
            'medoids': data[medoid_indices],  # Original scale
            'medoids_scaled': data_scaled[medoid_indices],
            'total_cost': total_cost,
            'n_iter': iteration + 1
        }
        
        return PartitionResult(
            labels=labels,
            k=len(kept),
            algorithm=self.algorithm,
            metadata=metadata,
            quality_score=self._compute_silhouette(data_scaled, labels)
        )
    
    @staticmethod
    def _compute_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Euclidean distance matrix between X and Y (O(n*m) memory)."""
        xx = (X ** 2).sum(axis=1)[:, None]
        yy = (Y ** 2).sum(axis=1)[None, :]
        d2 = np.maximum(xx + yy - 2 * X @ Y.T, 0.0)
        return np.sqrt(d2)


class AgglomerativeClusterer(BaseClusterer):
    """Agglomerative (hierarchical) clustering algorithm."""
    
    algorithm = ClusteringAlgorithm.AGGLOMERATIVE
    
    def fit(self, data: np.ndarray, k: int,
            linkage: str = 'ward',
            **kwargs) -> PartitionResult:
        """
        Fit agglomerative clustering.
        
        Additional metadata:
            - linkage: Linkage criterion used
            - n_leaves: Number of leaves in the hierarchical tree
            - n_connected_components: Number of connected components
        """
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        agg = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage
        )
        labels = agg.fit_predict(data_scaled)
        labels, kept = _relabel(labels)
        
        metadata = {
            'linkage': linkage,
            'n_leaves': agg.n_leaves_,
            'n_connected_components': agg.n_connected_components_
        }
        
        # Compute cluster centers (mean of points)
        centers = np.array([
            data[labels == i].mean(axis=0) 
            for i in range(len(kept))
        ])
        metadata['cluster_centers'] = centers
        
        return PartitionResult(
            labels=labels,
            k=len(kept),
            algorithm=self.algorithm,
            metadata=metadata,
            quality_score=self._compute_silhouette(data_scaled, labels)
        )


class SpectralClusterer(BaseClusterer):
    """Spectral clustering algorithm."""
    
    algorithm = ClusteringAlgorithm.SPECTRAL
    
    def fit(self, data: np.ndarray, k: int,
            affinity: str = 'rbf',
            random_state: Optional[int] = None,
            **kwargs) -> PartitionResult:
        """
        Fit spectral clustering.
        
        Additional metadata:
            - affinity: Affinity type used
        """
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        spectral = SpectralClustering(
            n_clusters=k,
            affinity=affinity,
            random_state=random_state,
            assign_labels='kmeans'
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = spectral.fit_predict(data_scaled)
        labels, kept = _relabel(labels)
        
        metadata = {
            'affinity': affinity
        }
        
        # Compute cluster centers
        centers = np.array([
            data[labels == i].mean(axis=0) 
            for i in range(len(kept))
        ])
        metadata['cluster_centers'] = centers
        
        return PartitionResult(
            labels=labels,
            k=len(kept),
            algorithm=self.algorithm,
            metadata=metadata,
            quality_score=self._compute_silhouette(data_scaled, labels)
        )


class NaturalPartitioner(BaseClusterer):
    """
    Natural partitioning based on unique value combinations.
    Suitable for categorical variables with limited unique combinations.
    """
    
    algorithm = ClusteringAlgorithm.NATURAL
    
    def fit(self, data: np.ndarray, k: int = None,
            column_names: Optional[List[str]] = None,
            **kwargs) -> PartitionResult:
        """
        Create natural partitions based on unique value combinations.
        
        Note: k parameter is ignored - number of clusters is determined by data.
        
        Additional metadata:
            - partition_values: Dictionary mapping cluster ID to value combination
            - column_names: Names of columns used (if provided)
        """
        # Convert to DataFrame for easier handling
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            if column_names:
                df = pd.DataFrame(data, columns=column_names)
            else:
                df = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
        
        # Create tuple of values for each row
        value_tuples = [tuple(row) for row in df.values]
        
        # Get unique combinations and create mapping (deterministic order)
        unique_combinations = sorted(set(value_tuples), key=lambda t: tuple(str(v) for v in t))
        combination_to_id = {comb: i for i, comb in enumerate(unique_combinations)}
        
        # Assign labels
        labels = np.array([combination_to_id[t] for t in value_tuples])
        actual_k = len(unique_combinations)
        
        # Create partition values dictionary
        partition_values = {
            i: {col: val for col, val in zip(df.columns, comb)}
            for comb, i in combination_to_id.items()
        }
        
        metadata = {
            'partition_values': partition_values,
            'column_names': list(df.columns),
            'n_unique_combinations': actual_k
        }
        
        return PartitionResult(
            labels=labels,
            k=actual_k,
            algorithm=self.algorithm,
            metadata=metadata,
            quality_score=None  # Not applicable for natural partitioning
        )


class ClusteringFactory:
    """Factory class for creating clusterers."""
    
    _clusterers = {
        ClusteringAlgorithm.KMEANS: KMeansClusterer,
        ClusteringAlgorithm.KMEDOIDS: KMedoidsClusterer,
        ClusteringAlgorithm.AGGLOMERATIVE: AgglomerativeClusterer,
        ClusteringAlgorithm.SPECTRAL: SpectralClusterer,
        ClusteringAlgorithm.NATURAL: NaturalPartitioner
    }
    
    @classmethod
    def get_clusterer(cls, algorithm: Union[ClusteringAlgorithm, str]) -> BaseClusterer:
        """
        Get clusterer instance by algorithm type.
        
        Args:
            algorithm: ClusteringAlgorithm enum or string name
            
        Returns:
            Clusterer instance
        """
        if isinstance(algorithm, str):
            algorithm = ClusteringAlgorithm(algorithm.lower())
        
        if algorithm not in cls._clusterers:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return cls._clusterers[algorithm]()
    
    @classmethod
    def available_algorithms(cls) -> List[str]:
        """Return list of available algorithm names."""
        return [alg.value for alg in cls._clusterers.keys()]


def prepare_data_for_clustering(
    data: pd.DataFrame,
    handle_categorical: str = 'onehot'
) -> Tuple[np.ndarray, FeatureSpace]:
    """
    Prepare a DataFrame for distance/model based methods.

    Args:
        data: Input DataFrame
        handle_categorical:
            'onehot' (default): standardized numeric columns + one-hot categorical
                      columns -> suitable for Euclidean distances, classifiers, CCA, kernels
            'codes':  integer codes for categorical columns (NOT suitable for
                      distances; only for tree models or rule conditions)

    Returns:
        (numeric matrix, FeatureSpace) -- the FeatureSpace keeps the column
        names, the categorical flags and the label <-> code mapping so that
        results can be reported in original units.
    """
    fs = FeatureSpace.from_dataframe(data)
    if handle_categorical == 'onehot':
        return fs.numeric, fs
    if handle_categorical == 'codes':
        return fs.codes, fs
    raise ValueError("handle_categorical must be 'onehot' or 'codes'")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n = 100
    
    # Numeric data
    data = np.column_stack([
        np.random.randn(n) * 2 + np.random.choice([0, 5, 10], n),
        np.random.randn(n) * 2 + np.random.choice([0, 5], n)
    ])
    
    print("Testing clustering algorithms:")
    print("=" * 50)
    
    for alg_name in ClusteringFactory.available_algorithms():
        if alg_name == 'natural':
            continue  # Skip for numeric data
            
        clusterer = ClusteringFactory.get_clusterer(alg_name)
        result = clusterer.fit(data, k=3, random_state=42)
        
        print(f"\n{alg_name.upper()}:")
        print(f"  Result: {result}")
        print(f"  Cluster sizes: {result.get_cluster_sizes()}")
        
        if 'centroids' in result.metadata:
            print(f"  Centroids shape: {result.metadata['centroids'].shape}")
        if 'medoid_indices' in result.metadata:
            print(f"  Medoid indices: {result.metadata['medoid_indices']}")
    
    # Test natural partitioning with categorical data
    print("\n" + "=" * 50)
    print("Testing NATURAL partitioning (categorical):")
    
    cat_data = pd.DataFrame({
        'color': np.random.choice(['red', 'blue', 'green'], n),
        'size': np.random.choice(['small', 'large'], n)
    })
    
    natural = ClusteringFactory.get_clusterer('natural')
    result = natural.fit(cat_data)
    print(f"  Result: {result}")
    print(f"  Partition values: {result.metadata['partition_values']}")
