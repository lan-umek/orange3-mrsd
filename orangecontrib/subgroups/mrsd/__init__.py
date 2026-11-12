"""
MRSD -- Multiple Responses Subgroup Discovery, version 2.0.0

Algorithms for discovering interesting subgroups in data with several
response (dependent) variables Y described by explanatory variables X.

Algorithms (11):
 1. MRSD-PT        partitioning and tables: partition pairs, contingency cells, permutation test with f-divergences
 2. MRSD-HC        hierarchical clustering (of Y) and classification (out-of-fold separability from X)
 3. MRSD-C2 (classification in both spaces) swapping: local search (instance moves) with classification in X and Y
 4. MRSD-C2 semi-supervised self-training with out-of-fold membership probabilities
 5. MRSD-C2 ranking         instance ranking by out-of-fold probabilities
 6. MRSD-CCA       clustering in regularized canonical-correlation space
 7. MRSD-EMM       exceptional model mining (interaction / Fisher-z tests)
 8. MRSD-MV        co-regularized multi-view spectral clustering
 9. MRSD-MI        mutual-information exceptionality (KSG, null-calibrated)
10. MRSD-Rules     CN2-SD style rules with multi-response WRAcc
11. MRSD-Ensemble  co-association consensus over the other algorithms

Every algorithm returns a `SetOfSubgroups` whose members carry a `Rule` in
original units, a canonical `quality`, per-target statistics, and p-values
from permutation inference (raw and family-wise adjusted).  See DEV_NOTES.md
for the shared conventions and `inference.py` for the testing framework.

Usage
-----
    import mrsd_subgroups as mrsd
    result = mrsd.run_mrsd_rules(df, X_columns=[...], Y_columns=[...])

or import an algorithm module directly
(`from mrsd_subgroups.mrsd_rules import run_mrsd_rules`).
"""

from .core import Subgroup, SetOfSubgroups, QualityMeasures

from .common import FeatureSpace, OutOfFoldClassifier, make_rng, derive_seed
from .data_loader import Dataset, DataLoader
from .rules import Condition, Rule, ConditionGenerator, BeamSearch, Description, describe_mask_with_tree
from .inference import (
    hotelling_t2, y_shift_statistic, x_separability_statistic, PermutationResult,
    pvalues_from_permutations, bh_adjust, permutation_test_candidates, y_shift_batch,
    validate_subgroups, search_permutation_pvalues,
)
from .classifiers import (
    ClassificationModel, AggregationMethod, make_classifier, ClassifierFactory,
    DualSpaceEvaluator, DualEvaluation, initial_subgroups,
)
from .clustering import (
    ClusteringAlgorithm, ClusteringFactory, PartitionResult,
    KMeansClusterer, KMedoidsClusterer, AgglomerativeClusterer, SpectralClusterer,
    NaturalPartitioner, prepare_data_for_clustering,
)
from .divergence import (
    FDivergenceType, FDivergenceRegistry, PValueCorrection, ContingencyTable,
    permutation_test, correct_pvalues, cell_f_divergence, cell_standardized_residuals,
)

# 1. MRSD-PT
from .mrsdpt import MRSDPT, MRSDPTConfig, run_mrsdpt
# 2. MRSD-HC
from .mrsdhc import (
    MRSDHC, MRSDHCConfig, run_mrsdhc, QualityMetric, EvaluationMethod,
    DistanceMetric, LinkageMethod, QualityEvaluator, DispersionMeasure,
)
# 3. MRSD-C2 Swapping
from .mrsdc2_swapping import SwappingMethod, SwappingMethodConfig, run_swapping_method, InitializationMethod
# 4. MRSD-C2 Semi-supervised
from .mrsdc2_semisupervised import (
    SemiSupervisedMethod, SemiSupervisedConfig, run_semisupervised_method, CombinationRule, ProbabilityCombiner,
)
# 5. MRSD-C2 Ranking
from .mrsdc2_ranking import (
    RankingMethod, RankingMethodConfig, run_ranking_method,
    RankAggregationMethod, SizeStepMethod, SelectionMethod, RankAggregator, SizeSelector, compute_pareto_front,
)
# 6. MRSD-CCA
from .mrsdcca import MRSDCCA, MRSDCCAConfig, run_mrsdcca, RegularizedCCA, CCAProjectionSpace
# 7. MRSD-EMM
from .mrsdemm import MRSDEMM, MRSDEMMConfig, run_mrsdemm, ExceptionalityMeasure, SearchStrategy, EMMStatistic
# 8. MRSD-MV
from .mrsd_mv import (
    MRSD_MV, MRSDMVConfig, run_mrsd_mv, MultiViewMethod, KernelType, AffinityComputer,
    coregularized_spectral, co_training, late_fusion,
)
# 9. MRSD-MI
from .mrsd_mi import (
    MRSD_MI, MRSDMIConfig, run_mrsd_mi, MIMethod, MIAggregation, MIEstimator,
    MutualInformation, MIExceptionality, ksg_mi, binning_mi, linfoot,
)
# 10. MRSD-Rules
from .mrsd_rules import (
    MRSD_Rules, MRSDRulesConfig, run_mrsd_rules, QualityMeasureType, CoveringStrategy,
    RuleQualityEvaluator, WeightedCovering,
)
from .mrsd_rules import InferenceMode as RulesInferenceMode
# 11. MRSD-Ensemble
from .mrsd_ensemble import (
    MRSD_Ensemble, MRSDEnsembleConfig, run_mrsd_ensemble, AlgorithmRegistry, AlgorithmSpec,
    ConsensusSubgroup, ResultFusion, FusionStrategy, coassociation_matrix, consensus_clusters,
)
from .mrsd_ensemble import InferenceMode as EnsembleInferenceMode

__version__ = "2.0.0"

ALGORITHMS = {
    "pt": run_mrsdpt,
    "hc": run_mrsdhc,
    "swapping": run_swapping_method,
    "semisupervised": run_semisupervised_method,
    "ranking": run_ranking_method,
    "cca": run_mrsdcca,
    "emm": run_mrsdemm,
    "mv": run_mrsd_mv,
    "mi": run_mrsd_mi,
    "rules": run_mrsd_rules,
    "ensemble": run_mrsd_ensemble,
}

__all__ = [n for n in dir() if not n.startswith("_")]
