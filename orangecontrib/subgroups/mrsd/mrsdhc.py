"""
MRSD-HC: Multiple Responses Subgroup Discovery with Hierarchical Clustering and
Classification (HC = hierarchical clustering of Y and classification from X).

What is implemented
-------------------
1. Y is encoded with `dataset.Y_space.numeric` (standardized numeric columns +
   one-hot categorical columns).  Optional `feature_weights` scale the columns
   by sqrt(weight) before linkage (so that squared Euclidean distances are
   weighted by `weight`); the dispersion measure uses the same scaled matrix.
2. Hierarchical clustering of the rows of Y (scipy `linkage`).  Ward linkage
   forces the Euclidean metric; the metric actually used is recorded in
   `quality_measures['distance_metric_used']`.  A user-supplied distance
   matrix (`precomputed_distances`) can be used with non-Ward linkages.
3. Every dendrogram node with n_min <= |node| <= n_max (default n // 2) is a
   candidate subgroup G.  Its dispersion q_d(G) (mean distance to the
   centroid) is compared with q_d(D) and reported
   (`quality_measures['lower_dispersion']`); nodes are dropped on that basis
   only when `require_lower_dispersion=True`.
4. Quality q(G) = out-of-fold AUC of predicting membership in G from
   `X_space.numeric` with `common.OutOfFoldClassifier` (never predicts on
   training rows).  One classifier per run by default (`classification_model`,
   a fast decision tree); a list of models gives one subgroup per node with
   quality = max over the models and the best model's name stored.
5. Inference: nodes were selected from Y, so `inference.validate_subgroups`
   with `statistic='x_auc'` (rows of X permuted, the same classifier and
   folds) gives for ALL candidate nodes the raw permutation p-value
   (`p_value`), the Westfall-Young max-T adjusted p-value
   (`p_value_corrected`, FWER over every node examined) and BH
   (`quality_measures['p_bh']`).
6. After inference: optional `q_min` filter, then deduplication and greedy
   Jaccard pruning (`redundancy_threshold`).  Every subgroup gets a `rule`
   from `rules.describe_mask_with_tree` with its precision / recall, the
   target statistics and `algorithm="mrsd_hc"`.

`EvaluationMethod.TRAIN_ON_TRAIN` (resubstitution) is kept only as a
diagnostic: it logs a warning and its AUC is not a valid search score.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist, squareform

from .classifiers import ClassificationModel, ClassifierFactory, make_classifier, resubstitution_auc
from .common import FeatureSpace, OutOfFoldClassifier, derive_seed, make_rng
from .data_loader import DataLoader, Dataset
from .inference import validate_subgroups
from .rules import describe_mask_with_tree
from .core import QualityMeasures, SetOfSubgroups, Subgroup

logger = logging.getLogger("mrsd")

__all__ = [
    "MRSDHC", "MRSDHCConfig", "run_mrsdhc",
    "ClassificationModel", "ClassifierFactory", "QualityMetric", "EvaluationMethod",
    "QualityEvaluator", "DistanceMetric", "LinkageMethod", "DispersionMeasure",
    "DendrogramNode", "build_dendrogram_tree", "get_candidate_nodes",
]


# =============================================================================
# ENUMS
# =============================================================================

class QualityMetric(Enum):
    """Quality of a candidate: out-of-fold AUC of membership prediction."""
    AUC = "auc"


class EvaluationMethod(Enum):
    """How membership prediction is evaluated."""
    CROSS_VALIDATION = "cross_validation"   # out-of-fold (default, the only valid search score)
    TRAIN_ON_TRAIN = "train_on_train"       # resubstitution; diagnostic only, logs a warning


class DistanceMetric(Enum):
    """Distance metrics for hierarchical clustering / dispersion (scipy names)."""
    MANHATTAN = "cityblock"
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    HAMMING = "hamming"
    CHEBYSHEV = "chebyshev"
    MINKOWSKI = "minkowski"


class LinkageMethod(Enum):
    WARD = "ward"
    COMPLETE = "complete"
    AVERAGE = "average"
    SINGLE = "single"
    WEIGHTED = "weighted"
    CENTROID = "centroid"
    MEDIAN = "median"


_EUCLIDEAN_ONLY = {LinkageMethod.WARD, LinkageMethod.CENTROID, LinkageMethod.MEDIAN}


# =============================================================================
# QUALITY EVALUATION (thin wrapper over common.OutOfFoldClassifier)
# =============================================================================

class QualityEvaluator:
    """
    Out-of-fold AUC of predicting subgroup membership from X.

    `evaluate(X, mask, classifier)` -> {"auc": value}.  With
    EvaluationMethod.TRAIN_ON_TRAIN the resubstitution AUC is returned instead
    (diagnostic only; a warning is logged because it rewards overfitting).
    """

    def __init__(self, evaluation_method: Union[EvaluationMethod, str] = EvaluationMethod.CROSS_VALIDATION,
                 n_folds: int = 3, random_state: Optional[int] = None):
        if isinstance(evaluation_method, str):
            evaluation_method = EvaluationMethod(evaluation_method)
        self.evaluation_method = evaluation_method
        self.n_folds = n_folds
        self.random_state = random_state
        self.metric = QualityMetric.AUC
        if evaluation_method == EvaluationMethod.TRAIN_ON_TRAIN:
            logger.warning("QualityEvaluator: TRAIN_ON_TRAIN (resubstitution) is a diagnostic only; "
                           "its AUC is optimistically biased and must not be used as a search score")

    def evaluate(self, X: np.ndarray, subgroup_mask: np.ndarray, classifier) -> Dict[str, float]:
        mask = np.asarray(subgroup_mask, dtype=bool)
        if self.evaluation_method == EvaluationMethod.TRAIN_ON_TRAIN:
            return {"auc": resubstitution_auc(X, mask, classifier)}
        oof = OutOfFoldClassifier(classifier, n_folds=self.n_folds, random_state=self.random_state)
        return {"auc": oof.auc(X, mask.astype(int))}

    def get_primary_score(self, scores: Dict[str, float]) -> float:
        return float(scores.get("auc", 0.5))


# =============================================================================
# DISPERSION MEASURE  q_d (mean distance to the centroid)
# =============================================================================

class DispersionMeasure:
    """
    Generalized dispersion of a set of rows: mean distance of the rows to their
    centroid under `distance_metric`, or (with a precomputed distance matrix)
    the mean pairwise distance.  Feature weights are expected to be applied to
    the data beforehand (sqrt(weight) scaling), the same as for the linkage.
    """

    def __init__(self, distance_metric: Union[DistanceMetric, str] = DistanceMetric.EUCLIDEAN,
                 precomputed_distances: Optional[np.ndarray] = None):
        if isinstance(distance_metric, str):
            distance_metric = DistanceMetric(distance_metric)
        self.distance_metric = distance_metric
        self.precomputed_distances = precomputed_distances

    def compute_dispersion(self, data: np.ndarray, indices: Optional[np.ndarray] = None) -> float:
        if indices is None:
            indices = np.arange(len(data))
        indices = np.asarray(indices, dtype=int)
        if len(indices) <= 1:
            return 0.0
        if self.precomputed_distances is not None:
            sub = self.precomputed_distances[np.ix_(indices, indices)]
            m = len(indices)
            return float(sub.sum() / (m * (m - 1)))
        subset = data[indices]
        prototype = subset.mean(axis=0, keepdims=True)
        d = cdist(subset, prototype, metric=self.distance_metric.value)[:, 0]
        return float(np.nanmean(d))


# =============================================================================
# DENDROGRAM
# =============================================================================

class DendrogramNode:
    """A node of the hierarchical clustering tree (leaf or merge)."""

    def __init__(self, node_id: int, indices: np.ndarray, left: Optional["DendrogramNode"] = None,
                 right: Optional["DendrogramNode"] = None, distance: float = 0.0):
        self.node_id = node_id
        self.indices = indices
        self.left = left
        self.right = right
        self.distance = distance

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def build_dendrogram_tree(linkage_matrix: np.ndarray, n_samples: int) -> Tuple[DendrogramNode, List[DendrogramNode]]:
    """(root, internal nodes in merge order) from a scipy linkage matrix."""
    nodes = {i: DendrogramNode(i, np.array([i])) for i in range(n_samples)}
    internal: List[DendrogramNode] = []
    for i, (a, b, dist, _) in enumerate(linkage_matrix):
        left, right = nodes[int(a)], nodes[int(b)]
        node = DendrogramNode(n_samples + i, np.concatenate([left.indices, right.indices]), left, right, float(dist))
        nodes[node.node_id] = node
        internal.append(node)
    return nodes[n_samples + len(linkage_matrix) - 1], internal


def get_candidate_nodes(internal_nodes: List[DendrogramNode], n_min: int, n_max: int) -> List[DendrogramNode]:
    return [nd for nd in internal_nodes if n_min <= nd.size <= n_max]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MRSDHCConfig:
    """
    Configuration of MRSD-HC.

    n_min, n_max: node size limits (n_max None -> n // 2)
    distance_metric, linkage_method: for scipy linkage on Y (Ward, centroid and
        median force 'euclidean'; the metric used is recorded in the result)
    precomputed_distances: optional (n, n) distance matrix on Y (not with Ward)
    feature_weights: optional non-negative weights, one per Y column (expanded
        to one-hot columns) or one per column of Y_space.numeric
    classification_model: one model name or a list (quality = max over models)
    fast_classifiers: use the cheap configuration of the model family
    evaluation_method: 'cross_validation' (out-of-fold) or 'train_on_train' (diagnostic)
    n_folds: folds of the out-of-fold evaluation (search and inference)
    require_lower_dispersion: drop nodes with q_d(G) > q_d(D) (otherwise only reported)
    q_min: optional AUC filter applied AFTER inference
    n_permutations: permutations of X rows for validate_subgroups (0 disables)
    redundancy_threshold: Jaccard threshold of the final pruning
    """
    n_min: int = 5
    n_max: Optional[int] = None

    distance_metric: str = "cityblock"
    linkage_method: str = "ward"
    precomputed_distances: Optional[np.ndarray] = None
    feature_weights: Optional[np.ndarray] = None

    classification_model: Union[str, List[str], object] = "decision_tree"   # name(s) or estimator object(s)
    classification_models: Optional[List[str]] = None   # legacy alias of classification_model
    fast_classifiers: bool = True
    evaluation_method: str = "cross_validation"
    n_folds: int = 3

    require_lower_dispersion: bool = False
    q_min: Optional[float] = None

    n_permutations: int = 100
    redundancy_threshold: float = 0.7

    random_state: Optional[int] = None
    verbose: bool = True

    def __post_init__(self):
        DistanceMetric(self.distance_metric)      # ValueError for unknown names
        LinkageMethod(self.linkage_method)
        EvaluationMethod(self.evaluation_method)
        if self.classification_models is not None:
            self.classification_model = list(self.classification_models)
        for m in self.models():
            if isinstance(m, str):
                ClassificationModel(m)          # names are validated; estimator objects are accepted as they are
        if self.n_min < 2:
            raise ValueError("n_min must be >= 2")
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if self.n_permutations < 0:
            raise ValueError("n_permutations must be >= 0")
        if self.feature_weights is not None:
            w = np.asarray(self.feature_weights, dtype=float)
            if w.ndim != 1 or (w < 0).any():
                raise ValueError("feature_weights must be a 1-d array of non-negative numbers")
            self.feature_weights = w

    def models(self) -> List[str]:
        m = self.classification_model
        return list(m) if isinstance(m, (list, tuple)) else [m]


# =============================================================================
# ALGORITHM
# =============================================================================

class MRSDHC:
    """MRSD-HC: dendrogram nodes of Y scored by out-of-fold separability in X."""

    def __init__(self, config: Optional[MRSDHCConfig] = None):
        self.config = config or MRSDHCConfig()
        self._linkage_matrix: Optional[np.ndarray] = None
        self._dendrogram_root: Optional[DendrogramNode] = None
        self._all_subgroups: List[Subgroup] = []
        self._result: Optional[SetOfSubgroups] = None
        self._dataset_dispersion: float = 0.0
        self.distance_metric_used: Optional[str] = None
        self.linkage_method_used: Optional[str] = None
        self.n_candidates_examined: int = 0
        self.timings: Dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, dataset: Dataset) -> SetOfSubgroups:
        cfg = self.config
        rng = make_rng(cfg.random_state)
        t0 = time.perf_counter()

        fs_X, fs_Y = dataset.X_space, dataset.Y_space
        n = fs_X.n
        n_max = cfg.n_max if cfg.n_max is not None else n // 2
        X_num = fs_X.numeric
        Y_num = self._weighted_Y(fs_Y)

        # ---- linkage ----------------------------------------------------
        method = LinkageMethod(cfg.linkage_method)
        metric = DistanceMetric(cfg.distance_metric)
        if cfg.precomputed_distances is not None:
            D = np.asarray(cfg.precomputed_distances, dtype=float)
            if D.shape != (n, n):
                raise ValueError(f"precomputed_distances must be ({n}, {n}), got {D.shape}")
            if method in _EUCLIDEAN_ONLY:
                logger.warning("MRSD-HC: %s linkage cannot use a precomputed distance matrix; using 'average'",
                               method.value)
                method = LinkageMethod.AVERAGE
            self._linkage_matrix = linkage(squareform(D, checks=False), method=method.value)
            metric_used = "precomputed"
            dispersion = DispersionMeasure(metric, precomputed_distances=D)
        else:
            if method in _EUCLIDEAN_ONLY and metric != DistanceMetric.EUCLIDEAN:
                logger.info("MRSD-HC: %s linkage requires Euclidean distance; '%s' ignored",
                            method.value, metric.value)
                metric = DistanceMetric.EUCLIDEAN
            self._linkage_matrix = linkage(Y_num, method=method.value, metric=metric.value)
            metric_used = metric.value
            dispersion = DispersionMeasure(metric)
        self.distance_metric_used = metric_used
        self.linkage_method_used = method.value
        self._dataset_dispersion = dispersion.compute_dispersion(Y_num)

        root, internal = build_dendrogram_tree(self._linkage_matrix, n)
        self._dendrogram_root = root
        candidates = get_candidate_nodes(internal, cfg.n_min, n_max)
        self.timings["linkage"] = time.perf_counter() - t0
        if cfg.verbose:
            print(f"MRSD-HC: n={n}, linkage={method.value}, metric={metric_used}, "
                  f"q_d(D)={self._dataset_dispersion:.3f}; {len(candidates)} candidate nodes "
                  f"({cfg.n_min} <= size <= {n_max}) [{self.timings['linkage']:.2f}s]")

        # ---- quality: out-of-fold AUC in X ------------------------------
        t1 = time.perf_counter()
        models = cfg.models()
        eval_seed = derive_seed(rng)
        classifiers = {m: make_classifier(m, derive_seed(rng), cfg.fast_classifiers) for m in models}
        evaluator = QualityEvaluator(cfg.evaluation_method, cfg.n_folds, eval_seed)

        subgroups: List[Subgroup] = []
        n_higher_dispersion = 0
        for node in candidates:
            mask = np.zeros(n, dtype=bool)
            mask[node.indices] = True
            q_d = dispersion.compute_dispersion(Y_num, node.indices)
            lower = q_d <= self._dataset_dispersion
            if not lower:
                n_higher_dispersion += 1
                if cfg.require_lower_dispersion:
                    continue
            aucs = {m: evaluator.get_primary_score(evaluator.evaluate(X_num, mask, clf))
                    for m, clf in classifiers.items()}
            best_model = max(aucs, key=aucs.get)
            sg = self._create_subgroup(node, mask, aucs, best_model, q_d, lower, fs_X, dataset)
            subgroups.append(sg)
        self.n_candidates_examined = len(subgroups)
        self.timings["quality"] = time.perf_counter() - t1
        if cfg.verbose:
            print(f"  quality (OOF AUC, {models}, {cfg.n_folds} folds) of {len(subgroups)} nodes "
                  f"[{self.timings['quality']:.2f}s]; {n_higher_dispersion} with q_d(G) > q_d(D)"
                  + (" dropped" if cfg.require_lower_dispersion else " (reported only)"))

        # ---- inference over ALL candidates examined ---------------------
        if cfg.n_permutations > 0 and subgroups:
            t2 = time.perf_counter()
            validate_subgroups(subgroups, X_space=fs_X, statistic="x_auc",
                               estimator=classifiers[models[0]], n_folds=cfg.n_folds,
                               n_permutations=cfg.n_permutations, random_state=derive_seed(rng),
                               selection_note=f"candidates = {len(subgroups)} dendrogram nodes of Y")
            self.timings["inference"] = time.perf_counter() - t2
            if cfg.verbose:
                n_sig = sum(1 for sg in subgroups if sg.p_value_corrected < 0.05)
                print(f"  inference: x_auc, B={cfg.n_permutations}, {n_sig} significant at FWER 0.05 "
                      f"[{self.timings['inference']:.2f}s]")

        # ---- post-processing: q_min filter, dedup + Jaccard pruning -----
        self._all_subgroups = subgroups
        kept = subgroups if cfg.q_min is None else [sg for sg in subgroups if sg.quality >= cfg.q_min]
        result = SetOfSubgroups(kept).prune(QualityMeasures.quality, s_min=cfg.redundancy_threshold)
        self._result = result
        self.timings["total"] = time.perf_counter() - t0
        if cfg.verbose:
            print(f"  {len(result)} subgroups after pruning [total {self.timings['total']:.2f}s]")
        return result

    # ------------------------------------------------------------- helpers
    def _weighted_Y(self, fs_Y: FeatureSpace) -> np.ndarray:
        Y = fs_Y.numeric
        w = self.config.feature_weights
        if w is None:
            return Y
        d = Y.shape[1]
        if len(w) == d:
            w_num = w
        elif len(w) == fs_Y.p:
            # expand per original column to the numeric columns it produced
            w_num = np.empty(d)
            pos = 0
            for j in range(fs_Y.p):
                width = len(fs_Y.categories[j]) if fs_Y.is_categorical[j] else 1
                w_num[pos:pos + width] = w[j]
                pos += width
        else:
            raise ValueError(f"feature_weights must have length {fs_Y.p} (Y columns) or {d} (numeric columns), "
                             f"got {len(w)}")
        return Y * np.sqrt(w_num)[None, :]

    def _create_subgroup(self, node: DendrogramNode, mask: np.ndarray, aucs: Dict[str, float],
                         best_model: str, q_d: float, lower: bool, fs_X: FeatureSpace,
                         dataset: Dataset) -> Subgroup:
        d = describe_mask_with_tree(fs_X, mask)
        sg = Subgroup(
            description=f"HC node {node.node_id} (n={node.size}, merge distance {node.distance:.3g})",
            covered_instances=set(int(i) for i in node.indices),
            algorithm="mrsd_hc",
            rule=d.rule,
            description_precision=float(d.precision),
            description_recall=float(d.recall),
            clustering_Y=self.linkage_method_used,
            k=node.node_id,
            l=node.size,
            clustering_metadata_Y={
                "linkage_method": self.linkage_method_used,
                "distance_metric_requested": self.config.distance_metric,
                "distance_metric_used": self.distance_metric_used,
                "feature_weights": None if self.config.feature_weights is None else list(map(float, self.config.feature_weights)),
                "merge_distance": float(node.distance),
            },
            clustering_metadata_X={"classifier": best_model, "evaluation_method": self.config.evaluation_method,
                                   "n_folds": self.config.n_folds},
        )
        sg.quality_measures.update({
            "quality": float(aucs[best_model]),
            "best_model": best_model,
            "dispersion": float(q_d),
            "dataset_dispersion": float(self._dataset_dispersion),
            "lower_dispersion": bool(lower),
            "merge_distance": float(node.distance),
            "distance_metric_used": self.distance_metric_used,
        })
        for m, a in aucs.items():
            sg.quality_measures[f"auc_{m}"] = float(a)
        sg.fill_target_statistics(dataset.Y)
        return sg

    # ---------------------------------------------------------------- access
    def get_linkage_matrix(self) -> Optional[np.ndarray]:
        return self._linkage_matrix

    def get_all_candidates(self) -> SetOfSubgroups:
        """Every candidate node examined (unpruned, unfiltered), with p-values."""
        return SetOfSubgroups(self._all_subgroups)

    def get_significant_subgroups(self, alpha: float = 0.05, all_candidates: bool = False) -> SetOfSubgroups:
        src = self.get_all_candidates() if all_candidates else (self._result or SetOfSubgroups())
        return SetOfSubgroups([sg for sg in src if sg.p_value_corrected is not None and sg.p_value_corrected < alpha])

    def get_high_quality_subgroups(self, min_quality: float = 0.5) -> SetOfSubgroups:
        return SetOfSubgroups([sg for sg in self._all_subgroups if sg.quality is not None and sg.quality >= min_quality])


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_mrsdhc(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
               **config_kwargs) -> SetOfSubgroups:
    """Run MRSD-HC on a DataFrame or a file path; returns the pruned, validated set."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    return MRSDHC(MRSDHCConfig(**config_kwargs)).fit(dataset)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    planted = x1 > 1.0
    y1 = rng.normal(size=n) + 2.0 * planted
    y2 = rng.normal(size=n) - 1.0 * planted
    df = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})

    algo = MRSDHC(MRSDHCConfig(n_min=10, n_permutations=100, random_state=0, verbose=True))
    result = algo.fit(DataLoader.from_dataframe(df, ["x1", "x2"], ["y1", "y2"]))
    best = min(result, key=lambda s: (s.p_value_corrected, -s.quality))
    planted_idx = set(np.flatnonzero(planted))
    jac = len(best.covered_instances & planted_idx) / len(best.covered_instances | planted_idx)
    print(f"\nBest node: {best}\n  rule: {best.rule} (prec={best.description_precision:.2f}, "
          f"rec={best.description_recall:.2f})\n  Jaccard with planted subgroup: {jac:.2f} -> "
          f"{'recovered' if jac >= 0.5 and best.p_value_corrected < 0.05 else 'NOT recovered'}")
