"""
Shared utilities for the MRSD (Multiple Responses Subgroup Discovery) package.

Provides:
- FeatureSpace: a typed view of a DataFrame (X or Y) that keeps the original
  values, knows which columns are categorical (by dtype, not by cardinality),
  can decode integer codes back to labels and can produce a numeric matrix
  suitable for distance-based / model-based methods (standardized numeric
  columns + one-hot encoded categorical columns).
- OutOfFoldClassifier: out-of-fold (cross-fitted) membership probabilities and
  AUC, the common building block for all "classification in X / Y space"
  quality measures.  Never predicts on rows it was trained on.
- make_rng: a single place to turn a seed into a numpy Generator without
  touching global state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold


def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC via ranks (ties averaged); identical to roc_auc_score."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

logger = logging.getLogger("mrsd")


# =============================================================================
# RANDOM STATE
# =============================================================================

def make_rng(seed: Optional[Union[int, np.random.Generator, np.random.RandomState]]) -> np.random.Generator:
    """Return a numpy Generator; never seeds the global numpy RNG."""
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.RandomState):
        return np.random.default_rng(seed.randint(0, 2**31 - 1))
    return np.random.default_rng(seed)


def derive_seed(rng: np.random.Generator) -> int:
    """Draw an integer seed for sklearn estimators from a Generator."""
    return int(rng.integers(0, 2**31 - 1))


# =============================================================================
# FEATURE SPACE
# =============================================================================

def _is_categorical_dtype(s: pd.Series) -> bool:
    """Categorical by dtype (object / string / category / bool), never by cardinality."""
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(dt):
        return True
    if pd.api.types.is_string_dtype(dt) or dt == object:
        return True
    return False


@dataclass
class FeatureSpace:
    """
    Typed numeric view of a DataFrame.

    Attributes:
        columns: original column names
        is_categorical: per-column flag (decided by dtype)
        codes: (n, p) float matrix; numeric columns as-is, categorical columns
               as integer codes (0..K-1) -- use for rule conditions ("=" on codes)
        categories: per categorical column, list of labels indexed by code
        numeric: (n, d) matrix for distance/model based methods: standardized
                 numeric columns + one-hot categorical columns (all levels)
        numeric_names: names of the columns of `numeric`
        raw: (n, p) matrix of original (unscaled) numeric values; categorical as codes
    """
    columns: List[str]
    is_categorical: List[bool]
    codes: np.ndarray
    categories: Dict[int, List] = field(default_factory=dict)
    numeric: np.ndarray = None
    numeric_names: List[str] = field(default_factory=list)
    _means: np.ndarray = None
    _stds: np.ndarray = None

    # ---- construction -------------------------------------------------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, categorical: Optional[Sequence[str]] = None,
                       standardize: bool = True) -> "FeatureSpace":
        """
        Build a FeatureSpace.

        Args:
            df: DataFrame (rows = instances)
            categorical: optional explicit list of categorical columns; columns not
                         listed are decided by dtype
            standardize: standardize numeric columns in `numeric` (z-scores)
        """
        columns = list(df.columns)
        n = len(df)
        p = len(columns)
        if p == 0:
            raise ValueError("FeatureSpace needs at least one column")
        cat_flags = []
        codes = np.zeros((n, p), dtype=float)
        categories: Dict[int, List] = {}
        num_blocks = []
        num_names = []
        means = []
        stds = []
        for j, col in enumerate(columns):
            s = df[col]
            is_cat = (categorical is not None and col in categorical) or _is_categorical_dtype(s)
            cat_flags.append(bool(is_cat))
            if is_cat:
                s_str = s.astype("object").where(~s.isna(), None)
                labels = sorted({v for v in s_str.tolist() if v is not None}, key=lambda v: str(v))
                lookup = {v: i for i, v in enumerate(labels)}
                col_codes = np.array([lookup[v] if v is not None else np.nan for v in s_str.tolist()], dtype=float)
                codes[:, j] = col_codes
                categories[j] = labels
                # one-hot (all levels; NaN -> all zeros)
                for i, lab in enumerate(labels):
                    num_blocks.append((col_codes == i).astype(float))
                    num_names.append(f"{col}={lab}")
                    means.append(0.0)
                    stds.append(1.0)
            else:
                vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
                if np.isnan(vals).any():
                    raise ValueError(f"Column '{col}' contains missing values; drop or impute them first")
                codes[:, j] = vals
                mu = float(vals.mean())
                sd = float(vals.std())
                if not np.isfinite(sd) or sd < 1e-12:
                    sd = 1.0
                means.append(mu)
                stds.append(sd)
                num_blocks.append((vals - mu) / sd if standardize else vals.copy())
                num_names.append(col)
        if np.isnan(codes).any():
            raise ValueError("Missing values in categorical columns are not supported; drop or impute them first")
        numeric = np.column_stack(num_blocks) if num_blocks else np.zeros((n, 0))
        return cls(columns=columns, is_categorical=cat_flags, codes=codes, categories=categories,
                   numeric=numeric, numeric_names=num_names,
                   _means=np.array(means), _stds=np.array(stds))

    # ---- accessors ----------------------------------------------------
    @property
    def n(self) -> int:
        return self.codes.shape[0]

    @property
    def p(self) -> int:
        return self.codes.shape[1]

    @property
    def all_categorical(self) -> bool:
        return all(self.is_categorical)

    @property
    def any_categorical(self) -> bool:
        return any(self.is_categorical)

    def decode(self, j: int, code) -> object:
        """Map an integer code of categorical column j back to its label."""
        if not self.is_categorical[j]:
            return code
        return self.categories[j][int(round(code))]

    def column_index(self, name: str) -> int:
        return self.columns.index(name)

    def raw_dataframe(self) -> pd.DataFrame:
        """Original values (labels for categorical) as a DataFrame."""
        data = {}
        for j, col in enumerate(self.columns):
            if self.is_categorical[j]:
                data[col] = [self.decode(j, c) for c in self.codes[:, j]]
            else:
                data[col] = self.codes[:, j]
        return pd.DataFrame(data)

    def numeric_unscaled(self) -> np.ndarray:
        """`numeric` with numeric columns on their original scale."""
        return self.numeric * self._stds + self._means


# =============================================================================
# OUT-OF-FOLD CLASSIFICATION
# =============================================================================

class OutOfFoldClassifier:
    """
    Out-of-fold (cross-fitted) membership prediction.

    Given a feature matrix and a binary membership vector, returns for every
    instance the probability of membership predicted by a model that did NOT
    see that instance.  Fold assignment is fixed per instance (derived from the
    seed and n), so two evaluations of slightly different masks are paired and
    the difference reflects the mask, not the split.
    """

    def __init__(self, estimator, n_folds: int = 5, random_state: Optional[int] = None):
        self.estimator = estimator
        self.n_folds = n_folds
        self.random_state = random_state

    def fold_ids(self, y: np.ndarray) -> np.ndarray:
        """Stratified fold id per instance; deterministic for given seed and y."""
        n = len(y)
        n_pos = int(y.sum())
        n_neg = n - n_pos
        k = max(2, min(self.n_folds, n_pos, n_neg))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        ids = np.empty(n, dtype=int)
        for f, (_, test_idx) in enumerate(skf.split(np.zeros((n, 1)), y)):
            ids[test_idx] = f
        return ids

    def predict_proba(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Out-of-fold P(member) for every instance. Requires >=2 of each class."""
        y = np.asarray(y).astype(int)
        n = len(y)
        if y.sum() < 2 or (n - y.sum()) < 2:
            raise ValueError("Need at least two instances in and outside the subgroup")
        ids = self.fold_ids(y)
        proba = np.zeros(n)
        for f in np.unique(ids):
            tr = ids != f
            te = ~tr
            clf = clone(self.estimator)
            clf.fit(X[tr], y[tr])
            if hasattr(clf, "predict_proba"):
                pr = clf.predict_proba(X[te])
                cls_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else None
                proba[te] = pr[:, cls_idx] if cls_idx is not None else 0.0
            elif hasattr(clf, "decision_function"):
                proba[te] = clf.decision_function(X[te])
            else:
                proba[te] = clf.predict(X[te])
        return proba

    def auc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Out-of-fold AUC of predicting membership (0.5 = chance)."""
        try:
            proba = self.predict_proba(X, y)
        except ValueError:
            return 0.5
        return fast_auc(y, proba)

    def auc_and_proba(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        try:
            proba = self.predict_proba(X, y)
        except ValueError:
            return 0.5, np.full(len(y), 0.5)
        return fast_auc(y, proba), proba


def mask_from_indices(indices, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[np.fromiter(indices, dtype=int)] = True
    return m


def indices_from_mask(mask: np.ndarray) -> set:
    return set(int(i) for i in np.flatnonzero(mask))
