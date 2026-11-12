"""
Declarative description of the parameters of every MRSD algorithm.

Each algorithm has a list of ``Param`` objects.  The algorithm widgets build
their GUI and their settings from these lists (see ``widgets/owbase.py``), and
``widget_config(alg_id, settings)`` turns the stored widget settings into the
keyword arguments of the algorithm's ``run_<name>`` function.

The lists mirror the ``<Name>Config`` dataclasses of the MRSD package; a test
(``tests/test_params.py``) checks that every config field is covered (or
deliberately listed in ``UNSUPPORTED``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Parameter kinds
# ---------------------------------------------------------------------------


@dataclass
class Param:
    name: str
    label: str
    default: Any = None
    group: str = "Settings"
    tip: str = ""

    def setting_defaults(self) -> Dict[str, Any]:
        """Widget settings (name -> default) that store this parameter."""
        return {self.name: self.default}

    def value(self, settings: Dict[str, Any]) -> Any:
        """Config value from the widget settings."""
        return settings[self.name]


@dataclass
class IntParam(Param):
    minimum: int = 0
    maximum: int = 1_000_000
    step: int = 1


@dataclass
class FloatParam(Param):
    minimum: float = 0.0
    maximum: float = 1e9
    step: float = 0.01
    decimals: int = 3


@dataclass
class BoolParam(Param):
    pass


@dataclass
class ChoiceParam(Param):
    choices: Sequence[Tuple[str, str]] = ()      # (value, label)

    @property
    def values(self):
        return [v for v, _ in self.choices]

    @property
    def labels(self):
        return [l for _, l in self.choices]


@dataclass
class MultiChoiceParam(Param):
    """A list of values chosen from `choices`; stored as a list of selected values."""
    choices: Sequence[Tuple[str, str]] = ()
    minimum_selected: int = 1

    @property
    def values(self):
        return [v for v, _ in self.choices]

    @property
    def labels(self):
        return [l for _, l in self.choices]

    def value(self, settings):
        sel = [v for v in settings[self.name] if v in dict(self.choices)]
        return sel if sel else [self.choices[0][0]]


@dataclass
class OptionalIntParam(Param):
    """An int or None ('automatic'); stored as <name> and <name>_enabled."""
    minimum: int = 0
    maximum: int = 1_000_000
    step: int = 1
    enabled_default: bool = False
    none_label: str = "automatic"

    def setting_defaults(self):
        return {self.name: self.default, f"{self.name}_enabled": self.enabled_default}

    def value(self, settings):
        return settings[self.name] if settings.get(f"{self.name}_enabled") else None


@dataclass
class OptionalFloatParam(Param):
    minimum: float = 0.0
    maximum: float = 1e9
    step: float = 0.01
    decimals: int = 3
    enabled_default: bool = False
    none_label: str = "automatic"

    def setting_defaults(self):
        return {self.name: self.default, f"{self.name}_enabled": self.enabled_default}

    def value(self, settings):
        return settings[self.name] if settings.get(f"{self.name}_enabled") else None


@dataclass
class IntListParam(Param):
    """Comma separated integers; empty text -> None (algorithm default)."""
    default: str = ""
    placeholder: str = "e.g. 2, 3, 4 (empty = default)"

    def value(self, settings):
        return parse_int_list(settings[self.name])


@dataclass
class DictParam(Param):
    """'key=value, key=value' text -> dict (values parsed as numbers); empty -> None."""
    default: str = ""
    placeholder: str = "e.g. y1=1, y2=-1"
    value_type: type = float

    def value(self, settings):
        return parse_dict(settings[self.name], self.value_type)


@dataclass
class JSONParam(Param):
    """JSON object text -> dict; empty -> {}."""
    default: str = ""
    placeholder: str = '{"rules": {"beam_width": 5}}'

    def value(self, settings):
        txt = (settings[self.name] or "").strip()
        if not txt:
            return {}
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            raise ValueError(f"{self.label}: a JSON object is expected")
        return obj


def parse_int_list(text: str) -> Optional[List[int]]:
    text = (text or "").strip()
    if not text:
        return None
    out = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out or None


def parse_dict(text: str, value_type=float) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    out = {}
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
        elif ":" in tok:
            k, v = tok.split(":", 1)
        else:
            raise ValueError(f"cannot parse '{tok}' (expected key=value)")
        out[k.strip()] = value_type(v.strip())
    return out or None


# ---------------------------------------------------------------------------
# Shared choice lists (values must match the enums of the MRSD package)
# ---------------------------------------------------------------------------

CLASSIFIERS = [
    ("random_forest", "Random forest"), ("decision_tree", "Decision tree"),
    ("logistic_regression", "Logistic regression"), ("svm", "SVM (RBF)"), ("svm_linear", "SVM (linear)"),
    ("naive_bayes", "Naive Bayes"), ("knn", "k-nearest neighbours"),
    ("gradient_boosting", "Gradient boosting"), ("adaboost", "AdaBoost"), ("mlp", "Neural network (MLP)"),
]
AGGREGATIONS = [("min", "minimum"), ("mean", "mean"), ("geometric_mean", "geometric mean"),
                ("harmonic_mean", "harmonic mean"), ("product", "product")]
CLUSTERING = [("kmeans", "k-means"), ("kmedoids", "k-medoids"), ("agglomerative", "agglomerative"),
              ("spectral", "spectral")]
F_DIVERGENCES = [
    ("chi_squared", "χ² (Pearson)"), ("total_variation", "total variation"), ("kl_divergence", "Kullback–Leibler"),
    ("reverse_kl", "reverse KL"), ("hellinger", "Hellinger"), ("jensen_shannon", "Jensen–Shannon"),
    ("neyman", "Neyman"), ("alpha_divergence", "α-divergence"),
]
INITIALIZATION = [("clustering", "clustering"), ("random", "random")]
C2_INFERENCE = [("search_permutation", "search permutation (honest, slower)"),
                ("y_shift", "Y-shift on candidates (conditional, fast)"), ("none", "none")]
FAMILY_WISE = [("search", "all evaluated candidates (search)"), ("candidates", "returned candidates only")]
SEARCH_RULES = [("search", "search re-run under permutation (honest)"), ("rules", "returned rules only (conditional)")]

G_SIZE = "Subgroup size"
G_INF = "Inference"
G_POST = "Post-processing"
G_OTHER = "Other"


def _size_params(n_min_default: int) -> List[Param]:
    return [
        IntParam("n_min", "Minimum size", n_min_default, G_SIZE, "Smallest subgroup reported", minimum=1, maximum=100000),
        OptionalIntParam("n_max", "Maximum size", 100, G_SIZE, "Largest subgroup (automatic = n / 2)",
                         minimum=2, maximum=10_000_000, none_label="automatic (n / 2)"),
    ]


def _redundancy() -> Param:
    return FloatParam("redundancy_threshold", "Redundancy threshold (Jaccard)", 0.7, G_POST,
                      "Subgroups more similar than this to a better one are pruned", 0.0, 1.0, 0.05, 2)


def _random_state() -> List[Param]:
    return [
        OptionalIntParam("random_state", "Random seed", 0, G_OTHER, "Fixed seed for reproducible results",
                         minimum=0, maximum=2**31 - 1, enabled_default=True, none_label="random"),
    ]


# ---------------------------------------------------------------------------
# Per-algorithm parameter lists
# ---------------------------------------------------------------------------

PARAMS: Dict[str, List[Param]] = {}

PARAMS["pt"] = [
    IntListParam("k_values", "Numbers of clusters k (both sides)", "2, 3, 4, 5", "Partitions",
                 "Cluster counts used for X and for Y when the side-specific lists are empty"),
    IntListParam("k_values_X", "k for X partitions", "", "Partitions", "Overrides the shared list for X",
                 placeholder="empty = same as above"),
    IntListParam("k_values_Y", "k for Y partitions", "", "Partitions", "Overrides the shared list for Y",
                 placeholder="empty = same as above"),
    OptionalIntParam("max_k", "Use k = 2 … max k", 5, "Partitions", "Shorthand for the shared list of k values",
                     minimum=2, maximum=50, none_label="off"),
    MultiChoiceParam("clustering_algorithms_X", "Clustering of X", ["kmeans", "kmedoids"], "Partitions",
                     choices=CLUSTERING),
    MultiChoiceParam("clustering_algorithms_Y", "Clustering of Y", ["kmeans", "kmedoids"], "Partitions",
                     choices=CLUSTERING),
    BoolParam("use_natural_partition_X", "Natural partition of X (categorical combinations)", True, "Partitions"),
    BoolParam("use_natural_partition_Y", "Natural partition of Y (categorical combinations)", True, "Partitions"),
    IntParam("max_natural_partitions", "Max. natural partition cells", 20, "Partitions", minimum=2, maximum=1000),
    ChoiceParam("f_divergence", "f-divergence", "chi_squared", "Statistic", choices=F_DIVERGENCES),
    ChoiceParam("cell_statistic", "Cell statistic", "f_divergence", "Statistic",
                choices=[("f_divergence", "cell f-divergence contribution"), ("residual", "squared adjusted residual")]),
    IntParam("n_permutations", "Permutations (0 = no test)", 1000, G_INF, minimum=0, maximum=100000, step=100),
    FloatParam("alpha", "Significance level α", 0.05, G_INF, minimum=0.0001, maximum=0.5, step=0.01, decimals=3),
    *_size_params(5),
    _redundancy(),
    *_random_state(),
]

PARAMS["hc"] = [
    ChoiceParam("distance_metric", "Distance on Y", "cityblock", "Clustering of Y",
                choices=[("cityblock", "Manhattan"), ("euclidean", "Euclidean"), ("cosine", "cosine"),
                         ("hamming", "Hamming"), ("chebyshev", "Chebyshev"), ("minkowski", "Minkowski")]),
    ChoiceParam("linkage_method", "Linkage", "ward", "Clustering of Y",
                choices=[("ward", "Ward"), ("complete", "complete"), ("average", "average"), ("single", "single"),
                         ("weighted", "weighted"), ("centroid", "centroid"), ("median", "median")]),
    MultiChoiceParam("classification_model", "Classifiers (quality = best)", ["decision_tree"], "Separability from X",
                     choices=CLASSIFIERS),
    BoolParam("fast_classifiers", "Fast classifier configuration", True, "Separability from X"),
    ChoiceParam("evaluation_method", "Evaluation", "cross_validation", "Separability from X",
                choices=[("cross_validation", "out-of-fold (cross-validation)"),
                         ("train_on_train", "train on train (diagnostic only)")]),
    IntParam("n_folds", "Folds", 3, "Separability from X", minimum=2, maximum=20),
    BoolParam("require_lower_dispersion", "Require lower dispersion than the whole data", False, "Separability from X"),
    OptionalFloatParam("q_min", "Minimum AUC (after inference)", 0.6, "Separability from X", minimum=0.0, maximum=1.0,
                       step=0.05, decimals=2, none_label="off"),
    IntParam("n_permutations", "Permutations of X (0 = no test)", 0, G_INF, minimum=0, maximum=100000, step=50),
    *_size_params(5),
    _redundancy(),
    *_random_state(),
]

_C2_COMMON = [
    ChoiceParam("classifier_X", "Classifier in X", "decision_tree", "Evaluation",
                "Every candidate is scored out-of-fold with this classifier – a tree is fast, a forest is slow",
                choices=CLASSIFIERS),
    ChoiceParam("classifier_Y", "Classifier in Y", "decision_tree", "Evaluation", choices=CLASSIFIERS),
    IntParam("n_folds", "Folds", 3, "Evaluation", minimum=2, maximum=20),
    ChoiceParam("aggregation", "Aggregation of q_X and q_Y", "min", "Evaluation", choices=AGGREGATIONS),
]
_C2_INIT = [
    ChoiceParam("initialization", "Starting subgroups", "clustering", "Initialization", choices=INITIALIZATION),
    IntParam("n_initial_subgroups", "Number of starting subgroups", 5, "Initialization", minimum=1, maximum=1000),
]
_C2_POST = [
    BoolParam("final_rescoring", "Final honest re-scoring", True, G_POST),
    _redundancy(),
    IntParam("n_permutations", "Permutations (0 = no test)", 0, G_INF,
             "Search permutations re-run the whole search – keep small", minimum=0, maximum=10000, step=10),
    ChoiceParam("inference_statistic", "Inference", "search_permutation", G_INF, choices=C2_INFERENCE),
]

PARAMS["swapping"] = [
    *_C2_COMMON,
    *_size_params(5),
    *_C2_INIT,
    IntParam("swap_batch_size", "Instances moved per step", 20, "Local search", minimum=1, maximum=10000),
    FloatParam("random_fraction", "Random fraction of candidates", 0.5, "Local search", minimum=0.0, maximum=1.0,
               step=0.05, decimals=2),
    FloatParam("min_improvement", "Minimum improvement", 0.005, "Local search", minimum=0.0, maximum=1.0,
               step=0.001, decimals=4),
    IntParam("max_iterations", "Maximum iterations", 50, "Local search", minimum=1, maximum=100000),
    IntParam("max_no_improvement", "Stop after iterations without improvement", 10, "Local search", minimum=1,
             maximum=10000),
    OptionalFloatParam("time_limit", "Time limit per start (s)", 30.0, "Local search", minimum=0.1, maximum=1e6,
                       step=10.0, decimals=1, enabled_default=True, none_label="none"),
    *_C2_POST,
    *_random_state(),
]

PARAMS["semisupervised"] = [
    *_C2_COMMON,
    ChoiceParam("combination_rule", "Combination of X and Y probabilities", "average", "Membership",
                choices=[("average", "average"), ("and", "AND"), ("or", "OR"), ("product", "product"),
                         ("min", "minimum"), ("max", "maximum"), ("weighted", "weighted")]),
    FloatParam("threshold", "Membership threshold", 0.5, "Membership", minimum=0.0, maximum=1.0, step=0.05, decimals=2),
    FloatParam("weight_X", "Weight of X (weighted rule)", 0.5, "Membership", minimum=0.0, maximum=1.0, step=0.05,
               decimals=2),
    *_size_params(5),
    *_C2_INIT,
    BoolParam("use_self_training", "Self-training", True, "Self-training"),
    IntParam("max_self_training_iterations", "Maximum self-training iterations", 10, "Self-training", minimum=1,
             maximum=1000),
    FloatParam("min_improvement", "Minimum improvement", 0.005, "Self-training", minimum=0.0, maximum=1.0, step=0.001,
               decimals=4),
    BoolParam("use_pairwise", "Pairwise phase", True, "Self-training"),
    *_C2_POST,
    *_random_state(),
]

PARAMS["ranking"] = [
    *_C2_COMMON,
    *_size_params(5),
    ChoiceParam("rank_aggregation", "Rank aggregation", "union", "Ranking",
                choices=[("union", "union"), ("intersection", "intersection"), ("borda", "Borda"),
                         ("rrf", "reciprocal rank fusion"), ("weighted_borda", "weighted Borda")]),
    IntParam("rrf_k", "RRF constant k", 60, "Ranking", minimum=1, maximum=10000),
    FloatParam("weight_X", "Weight of X (weighted Borda)", 0.5, "Ranking", minimum=0.0, maximum=1.0, step=0.05,
               decimals=2),
    ChoiceParam("size_step_method", "Candidate sizes", "logarithmic", "Candidate sizes",
                choices=[("logarithmic", "logarithmic"), ("linear", "linear"), ("all", "all"), ("custom", "custom")]),
    IntParam("linear_step", "Linear step", 10, "Candidate sizes", minimum=1, maximum=100000),
    IntListParam("custom_sizes", "Custom sizes", "", "Candidate sizes", placeholder="e.g. 10, 20, 50"),
    IntParam("max_sizes", "Maximum number of sizes", 15, "Candidate sizes", minimum=1, maximum=10000),
    ChoiceParam("selection_method", "Selection", "best", "Selection",
                choices=[("best", "best"), ("pareto", "Pareto front"), ("top_k", "top k")]),
    IntParam("top_k", "k (top k)", 5, "Selection", minimum=1, maximum=1000),
    FloatParam("pareto_epsilon", "Pareto ε", 0.02, "Selection", minimum=0.0, maximum=1.0, step=0.01, decimals=3),
    *_C2_INIT,
    *_C2_POST,
    *_random_state(),
]

PARAMS["cca"] = [
    IntParam("n_components", "Canonical components", 2, "CCA", minimum=1, maximum=100),
    FloatParam("regularization", "Ridge regularization λ", 0.0, "CCA", minimum=0.0, maximum=1e6, step=0.01,
               decimals=4),
    ChoiceParam("projection_space", "Clustering space", "x_scores", "CCA",
                choices=[("x_scores", "X scores"), ("y_scores", "Y scores"), ("combined", "combined")]),
    ChoiceParam("clustering_algorithm", "Clustering", "kmeans", "Clustering",
                choices=[("kmeans", "k-means"), ("agglomerative", "agglomerative")]),
    IntListParam("k_values", "Numbers of clusters", "2, 3, 4", "Clustering"),
    OptionalIntParam("n_clusters", "Single number of clusters", 3, "Clustering", minimum=2, maximum=100,
                     none_label="use list"),
    BoolParam("use_classification_quality", "Report out-of-fold classification quality", False, "Classification"),
    ChoiceParam("classifier_X", "Classifier in X", "random_forest", "Classification", choices=CLASSIFIERS),
    ChoiceParam("classifier_Y", "Classifier in Y", "random_forest", "Classification", choices=CLASSIFIERS),
    IntParam("cv_folds", "Folds", 5, "Classification", minimum=2, maximum=20),
    *_size_params(10),
    _redundancy(),
    IntParam("n_permutations", "Permutations (0 = no test)", 0, G_INF, minimum=0, maximum=100000, step=50),
    ChoiceParam("family_wise", "Family-wise correction over", "search", G_INF, choices=FAMILY_WISE),
    *_random_state(),
]

PARAMS["emm"] = [
    ChoiceParam("exceptionality_measure", "Exceptionality", "correlation", "Model",
                choices=[("correlation", "correlations (Fisher z)"), ("regression", "regression slopes"),
                         ("r2_difference", "R² difference")]),
    ChoiceParam("correlation_aggregation", "Correlation aggregation", "max_abs_z", "Model",
                choices=[("max_abs_z", "max |z|"), ("mean_sq_z", "mean z²")]),
    ChoiceParam("target_combination", "Combination over targets (regression)", "fisher", "Model",
                choices=[("fisher", "Fisher"), ("max", "maximum")]),
    BoolParam("use_Y_as_descriptive", "Also use Y as descriptive (selection on outcome!)", False, "Model"),
    ChoiceParam("search_strategy", "Search", "beam", "Search",
                choices=[("beam", "beam search"), ("clustering", "clustering"), ("random", "random")]),
    IntParam("beam_width", "Beam width", 20, "Search", minimum=1, maximum=1000),
    IntParam("max_depth", "Maximum conditions per rule", 3, "Search", minimum=1, maximum=20),
    IntParam("n_bins", "Bins for numeric attributes", 4, "Search", minimum=2, maximum=50),
    ChoiceParam("clustering_algorithm", "Clustering (clustering search)", "kmeans", "Search", choices=CLUSTERING),
    IntListParam("k_values", "Numbers of clusters (clustering search)", "", "Search",
                 placeholder="empty = 2, 3, 4, 5"),
    OptionalIntParam("n_clusters", "Single number of clusters", 3, "Search", minimum=2, maximum=100,
                     none_label="use list"),
    IntParam("n_random", "Random candidates (random search)", 100, "Search", minimum=1, maximum=100000),
    *_size_params(10),
    FloatParam("min_exceptionality", "Minimum exceptionality (−log10 p)", 0.0, G_POST, minimum=0.0, maximum=100.0,
               step=0.5, decimals=2),
    IntParam("top_k", "Report at most", 50, G_POST, minimum=1, maximum=10000),
    _redundancy(),
    IntParam("n_permutations", "Permutations (0 = no test)", 0, G_INF, minimum=0, maximum=100000, step=50),
    ChoiceParam("family_wise", "Family-wise correction over", "search", G_INF, choices=FAMILY_WISE),
    *_random_state(),
]

PARAMS["mv"] = [
    ChoiceParam("method", "Multi-view method", "coregularized", "Multi-view clustering",
                choices=[("coregularized", "co-regularized spectral"), ("average", "averaged affinities"),
                         ("co_training", "co-training"), ("late_fusion", "late fusion"),
                         ("excess", "similarity excess (W_X vs W_Y, no clustering)"),
                         ("dual_dendrogram", "dual dendrogram (nodes of X × nodes of Y)")]),
    ChoiceParam("kernel_type", "Kernel", "rbf", "Multi-view clustering",
                choices=[("rbf", "RBF"), ("linear", "linear"), ("polynomial", "polynomial"), ("cosine", "cosine"),
                         ("laplacian", "Laplacian")]),
    OptionalFloatParam("gamma", "Kernel γ", 1.0, "Multi-view clustering", minimum=1e-6, maximum=1e6, step=0.1,
                       decimals=4, none_label="automatic"),
    FloatParam("coreg_lambda", "Co-regularization λ", 0.05, "Multi-view clustering", minimum=0.0, maximum=100.0,
               step=0.01, decimals=3),
    IntParam("coreg_iterations", "Co-regularization iterations", 5, "Multi-view clustering", minimum=1, maximum=1000),
    IntListParam("k_values", "Numbers of clusters", "2, 3, 4", "Multi-view clustering"),
    OptionalIntParam("n_clusters", "Single number of clusters", 3, "Multi-view clustering", minimum=2, maximum=100,
                     none_label="use list"),
    IntParam("max_iterations", "Co-training iterations", 10, "Multi-view clustering", minimum=1, maximum=1000),
    FloatParam("convergence_threshold", "Convergence threshold", 0.01, "Multi-view clustering", minimum=0.0,
               maximum=1.0, step=0.005, decimals=4),
    ChoiceParam("excess_reference", "Reference for Y similarity", "centered", "Similarity excess",
                "centered: pairs deviating from the centre of Y in the same direction; mean: similarity above "
                "the average (rewards any dense set)",
                choices=[("centered", "centred kernel (H W_Y H)"), ("mean", "mean similarity (W_Y − γ·mean)")]),
    FloatParam("excess_gamma", "Resolution γ (mean reference)", 1.0, "Similarity excess", minimum=0.0, maximum=10.0,
               step=0.1, decimals=2),
    FloatParam("size_exponent", "Size exponent a in Q = Σ B / |G|^a", 1.0, "Similarity excess",
               "1 favours larger subgroups, 2 small tight ones", minimum=0.0, maximum=3.0, step=0.1, decimals=2),
    IntParam("n_subgroups", "Subgroups to extract", 5, "Similarity excess", minimum=1, maximum=100),
    ChoiceParam("linkage_method", "Linkage (both dendrograms)", "ward", "Dual dendrogram",
                choices=[("ward", "Ward"), ("complete", "complete"), ("average", "average"), ("single", "single")]),
    ChoiceParam("x_weighting", "Weights of X variables", "relevance", "X variable weights (excess, dual dendrogram)",
                "relevance: each X variable is weighted by its kernel alignment with Y, so that variables "
                "unrelated to Y do not dilute the X similarity",
                choices=[("relevance", "relevance for Y (kernel alignment)"), ("none", "none (all equal)")]),
    FloatParam("weight_power", "Sharpness of the weights", 2.0, "X variable weights (excess, dual dendrogram)",
               "weights = (alignment / max alignment) ^ power", minimum=0.1, maximum=10.0, step=0.5, decimals=1),
    BoolParam("use_classification_quality", "Report classification quality", True, "Classification"),
    ChoiceParam("classifier_X", "Classifier in X", "decision_tree", "Classification", choices=CLASSIFIERS),
    ChoiceParam("classifier_Y", "Classifier in Y", "decision_tree", "Classification", choices=CLASSIFIERS),
    *_size_params(10),
    IntParam("top_k", "Report at most", 20, G_POST, minimum=1, maximum=10000),
    _redundancy(),
    IntParam("n_permutations", "Permutations (0 = no test)", 0, G_INF, minimum=0, maximum=100000, step=50),
    ChoiceParam("inference_statistic", "Statistic", "y_shift_rule", G_INF,
                choices=[("y_shift_rule", "Y shift on the X rule of the cluster"), ("y_shift", "Y shift on the cluster")]),
    ChoiceParam("family_wise", "Family-wise correction over", "search", G_INF, choices=FAMILY_WISE),
    *_random_state(),
]

PARAMS["mi"] = [
    ChoiceParam("mi_method", "Mutual information", "joint", "Mutual information",
                choices=[("joint", "joint"), ("pairwise", "pairwise"), ("combined", "combined")]),
    ChoiceParam("mi_estimator", "Estimator", "mixed", "Mutual information",
                "mixed: consistent for numeric, categorical and mixed variables (Gao et al. 2017); "
                "k-NN (KSG): numeric variables only – categorical ones are one-hot coded and jittered",
                choices=[("mixed", "mixed types (k-NN, Gao et al.)"), ("knn", "k-NN (KSG, numeric)"),
                         ("binning", "binning")]),
    IntParam("k_neighbors", "Neighbours k", 5, "Mutual information", minimum=1, maximum=100),
    ChoiceParam("aggregation", "Pairwise aggregation", "mean", "Mutual information",
                choices=[("mean", "mean"), ("sum", "sum"), ("max", "maximum"), ("min", "minimum"),
                         ("frobenius", "Frobenius")]),
    IntParam("n_null_shuffles", "Null shuffles (calibration)", 20, "Mutual information", minimum=1, maximum=1000),
    ChoiceParam("search_method", "Search", "beam", "Search", choices=[("beam", "beam search"), ("clustering", "clustering")]),
    IntParam("beam_width", "Beam width", 10, "Search", minimum=1, maximum=1000),
    IntParam("max_depth", "Maximum conditions per rule", 3, "Search", minimum=1, maximum=20),
    IntParam("n_bins_discretize", "Bins for numeric attributes", 5, "Search", minimum=2, maximum=50),
    ChoiceParam("clustering_algorithm", "Clustering (clustering search)", "kmeans", "Search", choices=CLUSTERING),
    IntListParam("k_values", "Numbers of clusters (clustering search)", "", "Search", placeholder="empty = 2, 3, 4, 5"),
    OptionalIntParam("n_clusters", "Single number of clusters", 3, "Search", minimum=2, maximum=100,
                     none_label="use list"),
    *_size_params(20),
    FloatParam("min_exceptionality", "Minimum exceptionality (z)", 0.0, G_POST, minimum=0.0, maximum=100.0, step=0.5,
               decimals=2),
    IntParam("top_k", "Report at most", 20, G_POST, minimum=1, maximum=10000),
    _redundancy(),
    IntParam("n_permutations", "Permutations (0 = no test)", 0, G_INF, minimum=0, maximum=100000, step=50),
    ChoiceParam("family_wise", "Family-wise correction over", "search", G_INF, choices=FAMILY_WISE),
    *_random_state(),
]

PARAMS["rules"] = [
    IntParam("beam_width", "Beam width", 10, "Beam search", minimum=1, maximum=1000),
    IntParam("max_depth", "Maximum conditions per rule", 3, "Beam search", minimum=1, maximum=20),
    IntParam("n_rules", "Maximum number of rules", 10, "Beam search", minimum=1, maximum=1000),
    IntParam("n_bins", "Quantile cut points (numeric)", 5, "Beam search", minimum=2, maximum=50),
    BoolParam("local_cutpoints", "Recompute cut points inside the coverage", False, "Beam search"),
    BoolParam("allow_negation", "Allow '≠' conditions on categorical attributes", False, "Beam search"),
    ChoiceParam("quality_measure", "Quality", "wracc_mahalanobis", "Quality",
                choices=[("wracc_mahalanobis", "WRAcc – Mahalanobis"), ("wracc_dispersion", "WRAcc – dispersion"),
                         ("wracc_directed", "WRAcc – directed")]),
    DictParam("target_direction", "Target directions (directed)", "", "Quality",
              "For the directed measure: +1 = higher is interesting, −1 = lower", placeholder="e.g. y1=1, y2=-1",
              value_type=int),
    FloatParam("coverage_exponent", "Coverage exponent", 0.5, "Quality", "0.5 = test-statistic scale, 1 = WRAcc",
               minimum=0.0, maximum=2.0, step=0.1, decimals=2),
    ChoiceParam("covering_strategy", "Weighted covering", "multiplicative", "Covering",
                choices=[("multiplicative", "multiplicative"), ("additive", "additive"), ("none", "none")]),
    FloatParam("covering_gamma", "Covering γ (multiplicative)", 0.9, "Covering", minimum=0.0, maximum=1.0,
               step=0.05, decimals=2),
    FloatParam("min_quality_fraction", "Stop below fraction of the first rule's quality", 0.2, "Covering",
               minimum=0.0, maximum=1.0, step=0.05, decimals=2),
    *_size_params(10),
    _redundancy(),
    IntParam("n_permutations", "Permutations (0 = no test)", 0, G_INF, minimum=0, maximum=100000, step=50),
    ChoiceParam("inference_mode", "Inference", "search", G_INF, choices=SEARCH_RULES),
    IntParam("n_search_permutations", "Search permutations (search mode)", 100, G_INF, minimum=1, maximum=10000,
             step=10),
    *_random_state(),
]

MEMBER_ALGORITHMS = [("pt", "MRSD-PT"), ("hc", "MRSD-HC"), ("swapping", "C2 swapping"),
                     ("semisupervised", "C2 semi-supervised"), ("ranking", "C2 ranking"), ("cca", "MRSD-CCA"),
                     ("emm", "MRSD-EMM"), ("mv", "MRSD-MV"), ("mi", "MRSD-MI"), ("rules", "MRSD-Rules")]

PARAMS["ensemble"] = [
    MultiChoiceParam("algorithms", "Member algorithms", ["pt", "hc", "cca", "emm", "mv", "mi", "rules"], "Members",
                     choices=MEMBER_ALGORITHMS),
    JSONParam("algorithm_configs", "Member overrides (JSON)", "", "Members",
              "Per-member configuration overrides, e.g. {\"rules\": {\"beam_width\": 5}}"),
    IntParam("subgroups_per_algorithm", "Subgroups used per member", 5, "Members", minimum=1, maximum=1000),
    ChoiceParam("fusion_strategy", "Fusion", "voting", "Fusion",
                choices=[("voting", "voting"), ("rank_aggregation", "rank aggregation"), ("union", "union"),
                         ("intersection", "intersection")]),
    FloatParam("consensus_threshold", "Consensus threshold", 0.5, "Fusion", minimum=0.0, maximum=1.0, step=0.05,
               decimals=2),
    FloatParam("support_jaccard", "Support Jaccard", 0.5, "Fusion", minimum=0.0, maximum=1.0, step=0.05, decimals=2),
    IntParam("min_algorithms", "Minimum supporting algorithms", 2, "Fusion", minimum=1, maximum=10),
    IntParam("rrf_k", "RRF constant k", 60, "Fusion", minimum=1, maximum=10000),
    ChoiceParam("algorithm_weights", "Weights", "equal", "Fusion", choices=[("equal", "equal"), ("custom", "custom")]),
    DictParam("custom_weights", "Custom weights", "", "Fusion", placeholder="e.g. pt=1, rules=2"),
    *_size_params(10),
    IntParam("top_k", "Report at most", 20, G_POST, minimum=1, maximum=10000),
    _redundancy(),
    IntParam("rule_max_depth", "Max. depth of the describing rule", 3, G_POST, minimum=1, maximum=10),
    IntParam("n_permutations", "Permutations (0 = no test)", 0, G_INF, minimum=0, maximum=100000, step=50),
    ChoiceParam("inference_mode", "Inference", "search", G_INF,
                choices=[("search", "whole pipeline re-run under permutation (honest)"),
                         ("candidates", "consensus candidates only (conditional)")]),
    IntParam("n_search_permutations", "Search permutations (search mode)", 50, G_INF, minimum=1, maximum=10000,
             step=10),
    IntParam("n_bootstrap", "Bootstrap samples for stability (0 = off)", 0, "Stability", minimum=0, maximum=10000),
    FloatParam("bootstrap_fraction", "Bootstrap fraction", 0.8, "Stability", minimum=0.01, maximum=0.99, step=0.05,
               decimals=2),
    OptionalFloatParam("time_budget", "Time budget (s)", 600.0, "Execution", minimum=1.0, maximum=1e7, step=60.0,
                       decimals=0, none_label="unlimited"),
    BoolParam("skip_on_error", "Skip members that fail", True, "Execution"),
    *_random_state(),
]

# config fields that are intentionally not exposed in the GUI
UNSUPPORTED: Dict[str, List[str]] = {
    "hc": ["precomputed_distances", "feature_weights", "classification_models"],
}
# config fields the widgets set themselves
IMPLICIT = ["verbose"]


# ---------------------------------------------------------------------------
# Settings <-> config
# ---------------------------------------------------------------------------

def setting_defaults(alg_id: str) -> Dict[str, Any]:
    out = {}
    for p in PARAMS[alg_id]:
        out.update(p.setting_defaults())
    return out


def widget_config(alg_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword arguments for ``mrsd.ALGORITHMS[alg_id]`` from widget settings.

    Parameters whose value is None and whose config default is None are left
    out so that the algorithm applies its own default."""
    cfg = {}
    for p in PARAMS[alg_id]:
        v = p.value(settings)
        if v is None:
            continue
        cfg[p.name] = v
    cfg["verbose"] = False
    return cfg


def groups(alg_id: str) -> List[str]:
    """Parameter groups in order of first appearance."""
    seen = []
    for p in PARAMS[alg_id]:
        if p.group not in seen:
            seen.append(p.group)
    return seen
