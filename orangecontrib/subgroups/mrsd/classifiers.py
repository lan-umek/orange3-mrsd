"""
Classifier factory and the shared "classification in two spaces" evaluator
used by MRSD-HC, the MRSD-C2 family, MRSD-CCA and MRSD-MV.

All evaluation is out-of-fold (see common.OutOfFoldClassifier): a membership
probability for an instance always comes from a model that did not see that
instance.  Resubstitution ("train_on_train") is available only as an explicit
diagnostic and is never the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .common import OutOfFoldClassifier, make_rng, derive_seed


class ClassificationModel(Enum):
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    SVM_LINEAR = "svm_linear"
    NAIVE_BAYES = "naive_bayes"
    KNN = "knn"
    GRADIENT_BOOSTING = "gradient_boosting"
    ADABOOST = "adaboost"
    MLP = "mlp"


class AggregationMethod(Enum):
    MIN = "min"
    MEAN = "mean"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"
    PRODUCT = "product"


def make_classifier(model: Union[ClassificationModel, str], random_state: Optional[int] = None,
                    fast: bool = False):
    """New estimator with an explicit seed. `fast=True` returns a cheaper
    configuration of the same family for use inside search loops."""
    if isinstance(model, str):
        model = ClassificationModel(model)
    if not isinstance(model, ClassificationModel):
        # a ready estimator object (scikit-learn compatible: fit / predict_proba);
        # used as a template: a fresh copy with the requested seed
        from sklearn.base import clone
        est = clone(model)
        if random_state is not None and "random_state" in est.get_params():
            est.set_params(random_state=random_state)
        return est
    rs = random_state
    if model == ClassificationModel.DECISION_TREE:
        return DecisionTreeClassifier(max_depth=4 if fast else None, min_samples_leaf=3, random_state=rs)
    if model == ClassificationModel.RANDOM_FOREST:
        return RandomForestClassifier(n_estimators=30 if fast else 100, min_samples_leaf=2, random_state=rs, n_jobs=1)
    if model == ClassificationModel.LOGISTIC_REGRESSION:
        return LogisticRegression(max_iter=1000, random_state=rs)
    if model == ClassificationModel.SVM:
        return SVC(kernel="rbf", probability=True, random_state=rs)
    if model == ClassificationModel.SVM_LINEAR:
        return SVC(kernel="linear", probability=True, random_state=rs)
    if model == ClassificationModel.NAIVE_BAYES:
        return GaussianNB()
    if model == ClassificationModel.KNN:
        return KNeighborsClassifier(n_neighbors=5)
    if model == ClassificationModel.GRADIENT_BOOSTING:
        return GradientBoostingClassifier(n_estimators=50 if fast else 100, random_state=rs)
    if model == ClassificationModel.ADABOOST:
        return AdaBoostClassifier(random_state=rs)
    if model == ClassificationModel.MLP:
        return MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=rs)
    raise ValueError(model)


class ClassifierFactory:
    """Backward-compatible facade."""

    @classmethod
    def get_classifier(cls, model_type, random_state: Optional[int] = None, fast: bool = False):
        return make_classifier(model_type, random_state, fast)

    @classmethod
    def available_models(cls):
        return [m.value for m in ClassificationModel]


def aggregate(q_X: float, q_Y: float, method: Union[AggregationMethod, str]) -> float:
    if isinstance(method, str):
        method = AggregationMethod(method)
    if method == AggregationMethod.MIN:
        return min(q_X, q_Y)
    if method == AggregationMethod.MEAN:
        return 0.5 * (q_X + q_Y)
    if method == AggregationMethod.GEOMETRIC_MEAN:
        return float(np.sqrt(max(q_X, 0) * max(q_Y, 0)))
    if method == AggregationMethod.HARMONIC_MEAN:
        return 0.0 if q_X <= 0 or q_Y <= 0 else 2 * q_X * q_Y / (q_X + q_Y)
    if method == AggregationMethod.PRODUCT:
        return q_X * q_Y
    raise ValueError(method)


@dataclass
class DualEvaluation:
    q_X: float
    q_Y: float
    quality: float
    proba_X: np.ndarray
    proba_Y: np.ndarray


class DualSpaceEvaluator:
    """
    Out-of-fold separability of a subgroup in X space and in Y space.

    q_X = OOF AUC of classifier_X predicting membership from X
    q_Y = OOF AUC of classifier_Y predicting membership from Y
    quality = aggregate(q_X, q_Y)

    Fold ids are a deterministic function of (seed, membership vector), so
    evaluations of nearby masks are paired.  Results are cached by mask.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 classifier_X: Union[str, ClassificationModel] = "random_forest",
                 classifier_Y: Union[str, ClassificationModel] = "random_forest",
                 n_folds: int = 5, aggregation: Union[str, AggregationMethod] = "min",
                 random_state: Optional[int] = None, fast: bool = False):
        self.X = X
        self.Y = Y
        rng = make_rng(random_state)
        self.seed = derive_seed(rng)
        self.clf_X = make_classifier(classifier_X, derive_seed(rng), fast)
        self.clf_Y = make_classifier(classifier_Y, derive_seed(rng), fast)
        self.n_folds = n_folds
        self.aggregation = aggregation
        self._cache: Dict[bytes, DualEvaluation] = {}
        self.n_evaluations = 0

    def evaluate(self, mask: np.ndarray) -> DualEvaluation:
        mask = np.asarray(mask, dtype=bool)
        key = np.packbits(mask).tobytes()
        if key in self._cache:
            return self._cache[key]
        n = len(mask)
        n_in = int(mask.sum())
        if n_in < 2 or n - n_in < 2:
            ev = DualEvaluation(0.5, 0.5, aggregate(0.5, 0.5, self.aggregation), np.full(n, 0.5), np.full(n, 0.5))
        else:
            y = mask.astype(int)
            ox = OutOfFoldClassifier(self.clf_X, self.n_folds, self.seed)
            oy = OutOfFoldClassifier(self.clf_Y, self.n_folds, self.seed)
            qx, px = ox.auc_and_proba(self.X, y)
            qy, py = oy.auc_and_proba(self.Y, y)
            ev = DualEvaluation(qx, qy, aggregate(qx, qy, self.aggregation), px, py)
        self._cache[key] = ev
        self.n_evaluations += 1
        return ev

    def quality(self, mask: np.ndarray) -> float:
        return self.evaluate(mask).quality

    def fresh_copy(self, random_state: Optional[int]) -> "DualSpaceEvaluator":
        """Same configuration, new folds/seeds and empty cache (for honest re-scoring)."""
        ev = DualSpaceEvaluator.__new__(DualSpaceEvaluator)
        ev.X, ev.Y, ev.n_folds, ev.aggregation = self.X, self.Y, self.n_folds, self.aggregation
        rng = make_rng(random_state)
        ev.seed = derive_seed(rng)
        ev.clf_X = clone(self.clf_X)
        ev.clf_Y = clone(self.clf_Y)
        for c in (ev.clf_X, ev.clf_Y):
            if "random_state" in c.get_params():
                c.set_params(random_state=derive_seed(rng))
        ev._cache = {}
        ev.n_evaluations = 0
        return ev


def resubstitution_auc(X: np.ndarray, mask: np.ndarray, estimator) -> float:
    """Diagnostic only: AUC of a model evaluated on its own training rows."""
    y = mask.astype(int)
    if y.sum() < 1 or (len(y) - y.sum()) < 1:
        return 0.5
    clf = clone(estimator).fit(X, y)
    p = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    return float(roc_auc_score(y, p))


def initial_subgroups(n: int, n_subgroups: int, n_min: int, n_max: int, rng,
                      X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None,
                      method: str = "random") -> list:
    """
    Shared initializer for the MRSD-C2 family.

    'random': sets of random size in [n_min, min(n_max, n//2)]
    'clustering': k-means clusters of [X Y] (k = n_subgroups), padded with random sets
    """
    rng = make_rng(rng)
    hi = max(n_min, min(n_max, n // 2))
    out = []
    if method == "clustering" and X is not None and Y is not None:
        from sklearn.cluster import KMeans
        Z = np.hstack([X, Y])
        k = max(2, min(n_subgroups, n // max(n_min, 1)))
        labels = KMeans(n_clusters=k, n_init=5, random_state=derive_seed(rng)).fit_predict(Z)
        for c in range(k):
            idx = set(int(i) for i in np.flatnonzero(labels == c))
            if n_min <= len(idx) <= n_max:
                out.append(idx)
    while len(out) < n_subgroups:
        size = int(rng.integers(n_min, hi + 1))
        out.append(set(int(i) for i in rng.choice(n, size, replace=False)))
    return out[:n_subgroups]
