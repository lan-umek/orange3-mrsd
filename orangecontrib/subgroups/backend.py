"""
Bridge between Orange and the MRSD (Multiple Responses Subgroup Discovery)
package.

* ``mrsd`` – the vendored MRSD package (``orangecontrib/subgroups/mrsd``),
  loaded once and shared by all widgets;
* ``SubgroupResults`` – the object that travels between widgets on the
  *Subgroups* signal: the ``SetOfSubgroups`` plus everything needed to map the
  subgroups back to the Orange table (X / Y column names, row mapping, the
  table itself, the configuration used);
* conversions Orange ``Table`` -> ``pandas.DataFrame`` (X = attributes,
  Y = class variables) and results -> ``Table`` / annotated ``Table``.

Everything that does not need Orange is kept importable without it so that
the backend can be unit-tested outside of an Orange installation.
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Loading the MRSD package
# ---------------------------------------------------------------------------

MRSD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mrsd")
_mrsd = None


def load_mrsd():
    """Import the vendored copy of the MRSD package (``mrsd_subgroups`` on PyPI),
    which lives in ``orangecontrib/subgroups/mrsd`` (see tools/sync_mrsd.py)."""
    global _mrsd
    if _mrsd is None:
        try:
            _mrsd = importlib.import_module("orangecontrib.subgroups.mrsd")
        except ImportError as ex:
            raise ImportError(
                f"The MRSD package could not be imported from {MRSD_DIR} ({ex}). Run "
                "tools/sync_mrsd.py to copy it from the 'subgroups' repository.") from ex
    return _mrsd


# ---------------------------------------------------------------------------
# Algorithm catalogue
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AlgorithmInfo:
    id: str                 # key of mrsd.ALGORITHMS
    name: str               # short display name
    title: str              # long display name
    config_class: str       # name of the Config dataclass in mrsd
    description: str


ALGORITHMS: List[AlgorithmInfo] = [
    AlgorithmInfo("pt", "MRSD-PT", "MRSD-PT: Partitioning and Tables", "MRSDPTConfig",
                  "Partitions of X × partitions of Y; cells of the contingency table that are "
                  "over-represented are candidate subgroups (cell-level f-divergence, max-T permutation test)."),
    AlgorithmInfo("hc", "MRSD-HC", "MRSD-HC: Hierarchical Clustering and Classification", "MRSDHCConfig",
                  "Dendrogram of Y; nodes that are separable from the rest using X "
                  "(out-of-fold AUC) are candidate subgroups."),
    AlgorithmInfo("swapping", "MRSD-C2 Swapping", "MRSD-C2 (classification in both spaces): Swapping", "SwappingMethodConfig",
                  "Local search that moves instances in and out of the subgroup to maximise the "
                  "out-of-fold classification quality in X and in Y."),
    AlgorithmInfo("semisupervised", "MRSD-C2 Semi-supervised", "MRSD-C2 (classification in both spaces): Semi-supervised", "SemiSupervisedConfig",
                  "Self-training on out-of-fold membership probabilities in X and Y."),
    AlgorithmInfo("ranking", "MRSD-C2 Ranking", "MRSD-C2 (classification in both spaces): Ranking", "RankingMethodConfig",
                  "Top-s instances of aggregated out-of-fold probability rankings in X and Y."),
    AlgorithmInfo("cca", "MRSD-CCA", "MRSD-CCA: Canonical Correlation", "MRSDCCAConfig",
                  "Clusters in the regularized canonical-correlation space; exceptionality of the "
                  "first canonical correlation inside vs. outside the cluster (Fisher z)."),
    AlgorithmInfo("emm", "MRSD-EMM", "MRSD-EMM: Exceptional Model Mining", "MRSDEMMConfig",
                  "Rules in X whose X–Y model (correlations, regression slopes, R²) differs from the complement."),
    AlgorithmInfo("mv", "MRSD-MV", "MRSD-MV: Multi-View Clustering", "MRSDMVConfig",
                  "Co-regularized multi-view spectral clustering of the X and Y views."),
    AlgorithmInfo("mi", "MRSD-MI", "MRSD-MI: Mutual Information", "MRSDMIConfig",
                  "Rules in X with exceptional (null-calibrated) mutual information between X and Y."),
    AlgorithmInfo("rules", "MRSD-Rules", "MRSD-Rules: CN2-SD", "MRSDRulesConfig",
                  "CN2-SD style beam search of rules in X with a multi-response WRAcc quality."),
    AlgorithmInfo("ensemble", "MRSD-Ensemble", "MRSD-Ensemble: Consensus", "MRSDEnsembleConfig",
                  "Runs several of the other algorithms and fuses their subgroups by co-association consensus."),
]
ALGORITHM_BY_ID: Dict[str, AlgorithmInfo] = {a.id: a for a in ALGORITHMS}


def config_class(alg_id: str):
    return getattr(load_mrsd(), ALGORITHM_BY_ID[alg_id].config_class)


def config_fields(alg_id: str) -> Dict[str, dataclasses.Field]:
    return {f.name: f for f in dataclasses.fields(config_class(alg_id))}


# ---------------------------------------------------------------------------
# Results object (the "Subgroups" signal)
# ---------------------------------------------------------------------------

class SubgroupResults:
    """Subgroups found by one (or several) algorithm(s) on one Orange table.

    Attributes
    ----------
    subgroups : mrsd.SetOfSubgroups
    frame : pandas.DataFrame   the data the algorithm saw (rows without missing
                               values, in the same order as `row_indices`)
    row_indices : np.ndarray   row index into `data` (the Orange table) for
                               each row of `frame`
    data : Orange.data.Table or None
    X_columns, Y_columns : list of str
    algorithm : str            algorithm id ('rules', ...) or 'combined'
    config : dict              configuration used
    runtime : float            seconds
    names : dict               user-given names, key = position in `subgroups`
    """

    def __init__(self, subgroups, frame: pd.DataFrame, X_columns: Sequence[str], Y_columns: Sequence[str],
                 algorithm: str = "", config: Optional[Dict[str, Any]] = None, data=None,
                 row_indices: Optional[np.ndarray] = None, runtime: float = 0.0,
                 names: Optional[Dict[int, str]] = None):
        self.subgroups = subgroups
        self.frame = frame
        self.X_columns = list(X_columns)
        self.Y_columns = list(Y_columns)
        self.algorithm = algorithm
        self.config = dict(config or {})
        self.data = data
        self.row_indices = (np.arange(len(frame)) if row_indices is None
                            else np.asarray(row_indices, dtype=int))
        self.runtime = float(runtime)
        self.names = dict(names or {})

    # -- basics -------------------------------------------------------------
    def __len__(self):
        return len(self.subgroups)

    def __iter__(self):
        return iter(self.subgroups)

    def __getitem__(self, i):
        return self.subgroups[i]

    @property
    def n_rows(self) -> int:
        """Number of rows of the Orange table (if known) or of the frame."""
        if self.data is not None:
            return len(self.data)
        return int(self.row_indices.max()) + 1 if len(self.row_indices) else len(self.frame)

    @property
    def algorithm_name(self) -> str:
        info = ALGORITHM_BY_ID.get(self.algorithm)
        return info.name if info else (self.algorithm or "?")

    def name(self, i: int) -> str:
        """Display name of the i-th subgroup (user name or 'SG i')."""
        return self.names.get(i) or f"SG{i + 1}"

    def mask(self, i: int) -> np.ndarray:
        """Boolean mask of the i-th subgroup over the rows of the Orange table."""
        sg = self.subgroups[i]
        m = np.zeros(self.n_rows, dtype=bool)
        idx = np.fromiter(sg.covered_instances, dtype=int, count=len(sg.covered_instances))
        if len(idx):
            m[self.row_indices[idx]] = True
        return m

    def rule_text(self, i: int) -> str:
        sg = self.subgroups[i]
        if sg.rule is not None and len(sg.rule) > 0:
            return str(sg.rule)
        return sg.description or ""

    def y_prototype_text(self, i: int, digits: int = 3) -> str:
        """Prototype of the subgroup in Y: the mean of every numeric dependent
        variable and the modal level (with its share) of every categorical one,
        e.g. 'y1 = 1.42, y2 = -0.61, y3 = u (62%)'."""
        sg = self.subgroups[i]
        parts = []
        for target in (sg.target_variables or list((sg.target_statistics or {}).keys())):
            stats = (sg.target_statistics or {}).get(target)
            if not stats:
                continue
            if "mean_subgroup" in stats:
                parts.append(f"{target} = {stats['mean_subgroup']:.{digits}g}")
            else:
                levels = {k[2:-len(")_subgroup")]: v for k, v in stats.items() if k.endswith("_subgroup")}
                if levels:
                    lv = max(levels, key=levels.get)
                    parts.append(f"{target} = {lv} ({100 * levels[lv]:.0f}%)")
        return ", ".join(parts)

    def subset(self, indices: Sequence[int], algorithm: Optional[str] = None) -> "SubgroupResults":
        """New results object with the chosen subgroups (same order as `indices`)."""
        mrsd = load_mrsd()
        sgs = mrsd.SetOfSubgroups([self.subgroups[i] for i in indices])
        names = {k: self.names[i] for k, i in enumerate(indices) if i in self.names}
        return SubgroupResults(sgs, self.frame, self.X_columns, self.Y_columns,
                               algorithm if algorithm is not None else self.algorithm, self.config, self.data,
                               self.row_indices, self.runtime, names)

    def copy(self) -> "SubgroupResults":
        return self.subset(range(len(self)))

    # -- tabular summaries ----------------------------------------------------
    def summary_frame(self) -> pd.DataFrame:
        """One row per subgroup: name, rule, size, quality, p-values, per-target statistics."""
        rows = []
        for i, sg in enumerate(self.subgroups):
            qm = sg.quality_measures or {}
            row = {
                "name": self.name(i),
                "algorithm": sg.algorithm or self.algorithm,
                "rule": self.rule_text(i),
                "prototype_Y": self.y_prototype_text(i),
                "description": sg.description,
                "size": sg.coverage,
                "coverage": sg.coverage / max(len(self.frame), 1),
                "quality": sg.quality,
                "p_value": sg.p_value,
                "p_adjusted": sg.p_value_corrected,
                "p_bh": qm.get("p_bh"),
                "test_statistic": qm.get("test_statistic"),
                "description_precision": sg.description_precision,
                "description_recall": sg.description_recall,
            }
            for key, label in (("common_distinct_Y", "Y distinctness"), ("common_describe_X", "X describability"),
                               ("common_relation", "relation exceptionality")):
                if key in qm:
                    row[label] = qm[key]
            for target, stats in (sg.target_statistics or {}).items():
                for key, val in stats.items():
                    row[f"{target}: {key}"] = val
            rows.append(row)
        return pd.DataFrame(rows)

    def target_effects(self) -> pd.DataFrame:
        """Effect sizes (numeric targets) / proportion differences (categorical
        targets, per level) – subgroup vs. all: rows = subgroups, columns = targets."""
        cols: Dict[str, List[float]] = {}
        for i, sg in enumerate(self.subgroups):
            for target, stats in (sg.target_statistics or {}).items():
                if "effect_size" in stats:
                    cols.setdefault(target, [np.nan] * len(self))[i] = stats["effect_size"]
                else:
                    for key, val in stats.items():
                        if key.endswith("_subgroup"):
                            level = key[2:-len(")_subgroup")]
                            p_all = stats.get(f"p({level})_all", np.nan)
                            cols.setdefault(f"{target}={level}", [np.nan] * len(self))[i] = val - p_all
        return pd.DataFrame(cols, index=[self.name(i) for i in range(len(self))])

    def membership_frame(self) -> pd.DataFrame:
        """Rows = rows of the Orange table, one 0/1 column per subgroup."""
        return pd.DataFrame({self.name(i): self.mask(i).astype(int) for i in range(len(self))})

    def jaccard_matrix(self) -> np.ndarray:
        n = len(self)
        J = np.eye(n)
        sets = [set(sg.covered_instances) for sg in self.subgroups]
        for i in range(n):
            for j in range(i + 1, n):
                u = len(sets[i] | sets[j])
                J[i, j] = J[j, i] = len(sets[i] & sets[j]) / u if u else 0.0
        return J

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable description of the results."""
        out = {
            "algorithm": self.algorithm,
            "algorithm_name": self.algorithm_name,
            "X_columns": self.X_columns,
            "Y_columns": self.Y_columns,
            "n_rows": int(len(self.frame)),
            "runtime_seconds": self.runtime,
            "config": {k: _jsonable(v) for k, v in self.config.items()},
            "subgroups": [],
        }
        for i, sg in enumerate(self.subgroups):
            out["subgroups"].append({
                "name": self.name(i),
                "algorithm": sg.algorithm,
                "rule": self.rule_text(i),
                "prototype_Y": self.y_prototype_text(i),
                "description": sg.description,
                "size": int(sg.coverage),
                "quality": _jsonable(sg.quality),
                "p_value": _jsonable(sg.p_value),
                "p_adjusted": _jsonable(sg.p_value_corrected),
                "quality_measures": {k: _jsonable(v) for k, v in (sg.quality_measures or {}).items()},
                "target_statistics": {t: {k: _jsonable(v) for k, v in s.items()}
                                      for t, s in (sg.target_statistics or {}).items()},
                "description_precision": _jsonable(sg.description_precision),
                "description_recall": _jsonable(sg.description_recall),
                "rows": [int(r) for r in np.flatnonzero(self.mask(i))],
            })
        return out


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def combine_results(results: Sequence[SubgroupResults]) -> Optional[SubgroupResults]:
    """Concatenate results computed on the same rows (same frame length)."""
    results = [r for r in results if r is not None and len(r)]
    if not results:
        return None
    mrsd = load_mrsd()
    base = results[0]
    sgs, names = [], {}
    for r in results:
        if len(r.frame) != len(base.frame) or not np.array_equal(r.row_indices, base.row_indices):
            raise ValueError("Subgroup results were computed on different data and cannot be combined.")
        for i, sg in enumerate(r.subgroups):
            if i in r.names:
                names[len(sgs)] = r.names[i]
            elif len(results) > 1:
                names[len(sgs)] = f"{r.algorithm_name}: {r.name(i)}"
            sgs.append(sg)
    algs = sorted({r.algorithm for r in results})
    return SubgroupResults(mrsd.SetOfSubgroups(sgs), base.frame, base.X_columns, base.Y_columns,
                           algs[0] if len(algs) == 1 else "combined",
                           {r.algorithm: r.config for r in results}, base.data, base.row_indices,
                           sum(r.runtime for r in results), names)


# ---------------------------------------------------------------------------
# Running an algorithm on a data frame
# ---------------------------------------------------------------------------

def prepare_frame(frame: pd.DataFrame, X_columns: Sequence[str], Y_columns: Sequence[str],
                  drop_missing: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """Keep X and Y columns; optionally drop rows with missing values.

    Returns the frame (index reset) and the original row positions."""
    cols = list(X_columns) + list(Y_columns)
    sub = frame[cols]
    keep = np.ones(len(sub), dtype=bool)
    if drop_missing:
        keep = ~sub.isna().any(axis=1).to_numpy()
    sub = sub[keep].reset_index(drop=True)
    return sub, np.flatnonzero(keep)


ACCEPTS_INITIAL = ("swapping", "semisupervised", "ranking")


def run_algorithm(alg_id: str, frame: pd.DataFrame, X_columns: Sequence[str], Y_columns: Sequence[str],
                  config: Dict[str, Any], data=None, drop_missing: bool = True,
                  row_indices: Optional[np.ndarray] = None,
                  initial_masks: Optional[Sequence[np.ndarray]] = None) -> SubgroupResults:
    """Run one MRSD algorithm and wrap the result.

    `frame` is the full data frame (all columns); rows with missing X / Y
    values are dropped when `drop_missing` is set and `row_indices` is None.
    `initial_masks` (boolean masks over the rows of `frame`) are passed as
    starting subgroups to the algorithms that accept them (C2 family)."""
    mrsd = load_mrsd()
    if row_indices is None:
        sub, row_indices = prepare_frame(frame, X_columns, Y_columns, drop_missing)
    else:
        sub = frame
    t = time.time()
    runner = mrsd.ALGORITHMS[alg_id]
    kwargs = dict(config)
    if initial_masks and alg_id in ACCEPTS_INITIAL:
        position = {int(orig): k for k, orig in enumerate(row_indices)}
        initial = []
        for m in initial_masks:
            rows = {position[int(r)] for r in np.flatnonzero(np.asarray(m, dtype=bool)) if int(r) in position}
            if len(rows) >= 2:
                initial.append(rows)
        if initial:
            kwargs["initial_subgroups"] = initial
    sgs = runner(sub, list(X_columns), list(Y_columns), **kwargs)
    return SubgroupResults(sgs, sub, X_columns, Y_columns, alg_id, config, data, row_indices, time.time() - t)


# ---------------------------------------------------------------------------
# Orange conversions (imported lazily; the module stays importable without Orange)
# ---------------------------------------------------------------------------

def table_roles(table) -> Tuple[List[str], List[str]]:
    """X = attributes, Y = class variables of an Orange table's domain."""
    dom = table.domain
    return [v.name for v in dom.attributes], [v.name for v in dom.class_vars]


def table_to_dataframe(table) -> pd.DataFrame:
    """Orange table -> DataFrame with X and Y columns (metas excluded).

    Discrete variables become pandas 'category' columns (MRSD treats them as
    nominal), continuous ones float columns, time variables float seconds."""
    from Orange.data import DiscreteVariable, TimeVariable
    dom = table.domain
    X = np.asarray(table.X, dtype=float)
    Y = np.asarray(table.Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    cols = {}
    for j, var in enumerate(dom.attributes):
        cols[var.name] = _column(var, X[:, j], DiscreteVariable, TimeVariable)
    for j, var in enumerate(dom.class_vars):
        cols[var.name] = _column(var, Y[:, j], DiscreteVariable, TimeVariable)
    return pd.DataFrame(cols)


def _column(var, values, DiscreteVariable, TimeVariable):
    if isinstance(var, DiscreteVariable):
        codes = np.where(np.isnan(values), -1, values).astype(int)
        return pd.Categorical.from_codes(codes, categories=list(var.values))
    return pd.Series(values, dtype=float)


def results_to_table(results: SubgroupResults):
    """One row per subgroup as an Orange table (name, rule, size, quality, p-values, effects)."""
    from Orange.data import Domain, Table, ContinuousVariable, StringVariable, DiscreteVariable
    df = results.summary_frame()
    if df.empty:
        return None
    attrs, metas, Xcols, Mcols = [], [], [], []
    algs = sorted({str(a) for a in df["algorithm"].fillna("")})
    for col in df.columns:
        s = df[col]
        if col == "algorithm":
            var = DiscreteVariable("algorithm", values=algs)
            metas.append(var)
            Mcols.append(np.array([algs.index(str(a)) if a is not None else np.nan for a in s.fillna("")], dtype=object))
        elif col in ("name", "rule", "prototype_Y", "description"):
            metas.append(StringVariable(col))
            Mcols.append(s.fillna("").astype(str).to_numpy(dtype=object))
        else:
            attrs.append(ContinuousVariable(col))
            Xcols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))
    X = np.column_stack(Xcols) if Xcols else np.empty((len(df), 0))
    M = np.column_stack(Mcols) if Mcols else np.empty((len(df), 0), dtype=object)
    return Table.from_numpy(Domain(attrs, metas=metas), X, metas=M)


def annotate_table(results: SubgroupResults, indices: Optional[Sequence[int]] = None,
                   mode: str = "columns", column_name: str = "Subgroup"):
    """Return the Orange table with subgroup membership added as meta column(s).

    mode='columns': one Yes/No column per subgroup (`indices` or all);
    mode='single':  one discrete column with the name of the first covering
                    subgroup (or 'none')."""
    from Orange.data import DiscreteVariable, Domain, Table
    data = results.data
    if data is None:
        return None
    indices = list(range(len(results))) if indices is None else list(indices)
    dom = data.domain
    new_metas, new_cols = [], []
    if mode == "single":
        names = ["none"] + [results.name(i) for i in indices]
        col = np.zeros(len(data), dtype=float)
        for pos, i in reversed(list(enumerate(indices))):   # first covering wins
            col[results.mask(i)] = pos + 1
        new_metas.append(DiscreteVariable(_unique_name(column_name, dom), values=names))
        new_cols.append(col)
    else:
        for i in indices:
            var = DiscreteVariable(_unique_name(results.name(i), dom), values=("No", "Yes"))
            new_metas.append(var)
            new_cols.append(results.mask(i).astype(float))
    if not new_metas:
        return data
    domain = Domain(dom.attributes, dom.class_vars, list(dom.metas) + new_metas)
    table = data.transform(domain)
    with table.unlocked(table.metas):
        for k, col in enumerate(new_cols):
            table.metas[:, len(dom.metas) + k] = col
    return table


def membership_table(results: SubgroupResults, index: int, class_name: Optional[str] = None,
                     include_y: bool = False, features: str = "X"):
    """The rows the algorithm saw as an Orange table whose class is the
    membership ('no' / 'yes') of the given subgroup; the attributes are X
    (features='X'), Y ('Y') or both ('XY' / include_y=True); original metas
    kept.  Ready for Test & Score, learners, Rank …"""
    from Orange.data import DiscreteVariable, Domain, Table
    if results.data is None:
        return None
    data = results.data[results.row_indices]
    dom = data.domain
    if include_y:
        features = "XY"
    attrs = (list(dom.attributes) if "X" in features else []) + (list(dom.class_vars) if "Y" in features else [])
    cls = DiscreteVariable(_unique_name(class_name or f"in {results.name(index)}", dom), values=("no", "yes"))
    base = data.transform(Domain(attrs, [], dom.metas))
    mask = results[index].mask(len(results.frame)).astype(float)
    table = Table.from_numpy(Domain(attrs, cls, dom.metas), base.X, mask, base.metas)
    table.name = f"{results.name(index)} membership"
    return table


def selected_rows_table(results: SubgroupResults, indices: Sequence[int]):
    """Rows of the Orange table covered by any of the chosen subgroups."""
    if results.data is None or not indices:
        return None
    m = np.zeros(results.n_rows, dtype=bool)
    for i in indices:
        m |= results.mask(i)
    return results.data[np.flatnonzero(m)]


def _unique_name(name: str, domain) -> str:
    existing = {v.name for v in domain.variables} | {v.name for v in domain.metas}
    if name not in existing:
        return name
    k = 1
    while f"{name} ({k})" in existing:
        k += 1
    return f"{name} ({k})"


# ---------------------------------------------------------------------------
# Orange learners applied to subgroup membership
# ---------------------------------------------------------------------------

def orange_classifier_report(results: SubgroupResults, index: int, learner, include_y: bool = False,
                             n_folds: int = 5, random_state: int = 0, features: str = "X"):
    """Cross-validated AUC / CA of an Orange learner predicting membership of
    the subgroup from X (optionally X and Y), the training AUC and the model
    fitted on all rows.  Returns dict(auc_cv, ca_cv, auc_train, folds, model,
    data, results) or None when the results carry no data."""
    from Orange.evaluation import AUC, CA, CrossValidation, TestOnTrainingData
    data = membership_table(results, index, include_y=include_y, features=features)
    if data is None:
        return None
    n_pos = int(np.nansum(data.Y))
    folds = max(2, min(n_folds, n_pos, len(data) - n_pos))
    cv = CrossValidation(k=folds, stratified=True, random_state=random_state)(data, [learner])
    tr = TestOnTrainingData()(data, [learner])
    model = learner(data)
    model.name = f"{getattr(learner, 'name', type(learner).__name__)}: {results.name(index)}"
    return {"auc_cv": float(AUC(cv)[0]), "ca_cv": float(CA(cv)[0]), "auc_train": float(AUC(tr)[0]),
            "folds": folds, "model": model, "data": data, "results": cv}


def describe_orange_model(model) -> str:
    """Short HTML description of an Orange model (coefficients, tree size, importances)."""
    try:
        if hasattr(model, "coefficients") and hasattr(model, "intercept"):
            coef = np.asarray(model.coefficients)
            coef = coef[0] if coef.ndim == 2 and coef.shape[0] == 1 else coef[-1] if coef.ndim == 2 else coef
            names = [v.name for v in model.domain.attributes]
            order = np.argsort(-np.abs(coef))
            rows = "".join(f"<tr><td>{names[j]}</td><td align='right'>{coef[j]:+.3f}</td></tr>" for j in order)
            return (f"<b>Coefficients (log-odds of membership)</b><table cellpadding='2'>"
                    f"<tr><td>intercept</td><td align='right'>{float(np.ravel(model.intercept)[-1]):.3f}</td></tr>{rows}"
                    "</table><small>Connect the Model output to Nomogram.</small>")
        if hasattr(model, "node_count") and hasattr(model, "depth"):
            return (f"Tree with {model.node_count()} nodes, depth {model.depth()}. "
                    "<small>Connect the Model output to Tree Viewer.</small>")
        if hasattr(model, "skl_model") and hasattr(model.skl_model, "feature_importances_"):
            imp = model.skl_model.feature_importances_
            names = [v.name for v in model.domain.attributes]
            rows = "".join(f"<tr><td>{names[j]}</td><td align='right'>{imp[j]:.3f}</td></tr>"
                           for j in np.argsort(-imp)[:15])
            return f"<b>Feature importances</b><table cellpadding='2'>{rows}</table>"
        return f"{type(model).__name__}: connect the Model output to a viewer widget (Nomogram, Tree Viewer …)."
    except Exception as ex:          # noqa: BLE001 - description is best effort
        return f"<i>Cannot summarise the model: {ex}</i>"


# ---------------------------------------------------------------------------
# Orange learners inside the MRSD algorithms
# ---------------------------------------------------------------------------

_LEARNER_REGISTRY: Dict[str, Any] = {}


class OrangeEstimator:
    """A scikit-learn-compatible wrapper of an Orange learner, so that the MRSD
    algorithms (which score candidates with `fit` / `predict_proba` on numeric
    arrays) can use any learner connected to the widget: Naive Bayes, SVM,
    logistic regression, a tree with the user's settings …

    The learner itself is kept in a module registry and referenced by a string
    key, so that scikit-learn's `clone` (which the algorithms use to get fresh
    copies) only copies the key and never touches the Orange object."""
    _estimator_type = "classifier"

    def __init__(self, learner=None, key=None):
        if learner is not None:
            key = key or f"{getattr(learner, 'name', type(learner).__name__)}#{id(learner)}"
            _LEARNER_REGISTRY[key] = learner
        self.key = key

    @property
    def learner(self):
        return _LEARNER_REGISTRY[self.key]

    @property
    def name(self) -> str:
        return (self.key or "?").split("#")[0]

    # scikit-learn protocol -------------------------------------------------
    def get_params(self, deep=False):
        return {"key": self.key}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        attrs = [ContinuousVariable(f"f{j}") for j in range(X.shape[1])]
        self.domain_ = Domain(attrs, DiscreteVariable("member", values=("0", "1")))
        self.classes_ = np.array([0, 1])
        self.model_ = self.learner(Table.from_numpy(self.domain_, X, y.astype(float)))
        return self

    def predict_proba(self, X):
        from Orange.data import Domain, Table
        table = Table.from_numpy(Domain(self.domain_.attributes), np.asarray(X, dtype=float))
        probs = np.asarray(self.model_(table, self.model_.Probs), dtype=float)
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def __repr__(self):
        return f"Orange:{self.name}"

    __str__ = __repr__


CLASSIFIER_PARAMS = {
    "hc": ("classification_model",),
    "swapping": ("classifier_X", "classifier_Y"),
    "semisupervised": ("classifier_X", "classifier_Y"),
    "ranking": ("classifier_X", "classifier_Y"),
    "cca": ("classifier_X", "classifier_Y"),
    "mv": ("classifier_X", "classifier_Y"),
}


def apply_learners(alg_id: str, config: Dict[str, Any], learners: Sequence, x_index: int = 0,
                   y_index: int = 1) -> Dict[str, Any]:
    """Replace the classifier parameters of `config` with wrappers of the
    connected Orange learners: MRSD-HC gets all of them (quality = best),
    the two-space algorithms get `learners[x_index]` for X and
    `learners[y_index]` for Y (indices are clipped).  Other algorithms are
    returned unchanged."""
    if not learners or alg_id not in CLASSIFIER_PARAMS:
        return config
    cfg = dict(config)
    wrapped = [OrangeEstimator(l) for l in learners]
    if alg_id == "hc":
        cfg["classification_model"] = wrapped
    else:
        cfg["classifier_X"] = wrapped[min(max(x_index, 0), len(wrapped) - 1)]
        cfg["classifier_Y"] = wrapped[min(max(y_index, 0), len(wrapped) - 1)]
    return cfg


# ---------------------------------------------------------------------------
# Running an algorithm in a separate process (cancellable, does not block Orange)
# ---------------------------------------------------------------------------

class Cancelled(Exception):
    """The run was stopped by the user."""


def log(*parts):
    """Progress message to the console Orange was started from (python -m Orange.canvas)."""
    stream = sys.stderr or sys.stdout
    if stream is None:          # pythonw: no console
        return
    try:
        print(time.strftime("[MRSD %H:%M:%S]"), *parts, file=stream, flush=True)
    except Exception:           # noqa: BLE001 - logging must never break a run
        pass


def _picklable_estimator_state(self):
    return {"key": self.key, "learner": _LEARNER_REGISTRY.get(self.key)}


def _restore_estimator_state(self, state):
    self.key = state["key"]
    if state.get("learner") is not None:
        _LEARNER_REGISTRY[self.key] = state["learner"]


OrangeEstimator.__getstate__ = _picklable_estimator_state
OrangeEstimator.__setstate__ = _restore_estimator_state


def run_algorithm_in_process(alg_id: str, frame: pd.DataFrame, X_columns: Sequence[str], Y_columns: Sequence[str],
                             config: Dict[str, Any], drop_missing: bool = True,
                             initial_masks: Optional[Sequence[np.ndarray]] = None,
                             should_stop=lambda: False, poll: float = 0.2,
                             use_process: bool = True) -> SubgroupResults:
    """Like run_algorithm, but in a child Python process that is killed when
    `should_stop()` becomes true (raises Cancelled).  Falls back to running
    in the current thread if the process cannot be started."""
    import pickle
    import subprocess
    import tempfile
    load_mrsd()
    kwargs = dict(alg_id=alg_id, frame=frame, X_columns=list(X_columns), Y_columns=list(Y_columns), config=config,
                  drop_missing=drop_missing, initial_masks=initial_masks)
    if not use_process:
        log(f"{alg_id}: running in the Orange process ({len(frame)} rows); Stop cannot interrupt it")
        t0 = time.time()
        try:
            result = run_algorithm(**kwargs)
        except Exception:
            import traceback
            log(f"{alg_id}: failed:\n{traceback.format_exc()}")
            raise
        log(f"{alg_id}: finished in {time.time() - t0:.1f} s, {len(result)} subgroups")
        return result
    tmp = tempfile.mkdtemp(prefix="mrsd_")
    inp, out, logfile = (os.path.join(tmp, n) for n in ("in.pkl", "out.pkl", "log.txt"))
    try:
        try:
            with open(inp, "wb") as f:
                pickle.dump(kwargs, f)
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([root] + [p for p in sys.path if p] +
                                                ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
            exe = sys.executable
            if os.name == "nt" and os.path.basename(exe).lower() == "pythonw.exe":
                cand = os.path.join(os.path.dirname(exe), "python.exe")
                exe = cand if os.path.exists(cand) else exe
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            log(f"{alg_id}: starting worker process: {exe} -m orangecontrib.subgroups.worker "
                f"({len(frame)} rows, X={len(kwargs['X_columns'])}, Y={len(kwargs['Y_columns'])})")
            log(f"{alg_id}: config = { {k: v for k, v in config.items() if k != 'verbose'} }")
            with open(logfile, "wb") as lf:
                proc = subprocess.Popen([exe, "-m", "orangecontrib.subgroups.worker", inp, out], env=env,
                                        stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        creationflags=flags)
        except (OSError, pickle.PicklingError, TypeError, AttributeError) as ex:
            log(f"{alg_id}: cannot start a worker process ({type(ex).__name__}: {ex}); running in this process")
            result = run_algorithm(**kwargs)            # cannot start a process: run here
            log(f"{alg_id}: finished in-process, {len(result)} subgroups")
            return result
        log(f"{alg_id}: worker pid {proc.pid} started")
        t = time.time()
        last = t
        while proc.poll() is None:
            if should_stop():
                proc.kill()
                proc.wait()
                log(f"{alg_id}: stopped by the user after {time.time() - t:.0f} s (worker killed)")
                raise Cancelled()
            if time.time() - last >= 10:
                last = time.time()
                log(f"{alg_id}: still running, {last - t:.0f} s")
            time.sleep(poll)
        try:
            with open(logfile, "r", errors="replace") as lf:
                worker_output = lf.read()
        except OSError:
            worker_output = ""
        log(f"{alg_id}: worker exited with code {proc.returncode} after {time.time() - t:.1f} s")
        if worker_output.strip():
            log(f"{alg_id}: worker output:\n" + worker_output.strip()[-4000:])
        if not os.path.exists(out):
            tail = worker_output[-1500:]
            raise RuntimeError(f"the worker process ended without a result (exit code {proc.returncode}). {tail}")
        with open(out, "rb") as f:
            status, payload = pickle.load(f)
        if status != "ok":
            log(f"{alg_id}: the algorithm failed in the worker:\n{payload}")
            raise RuntimeError(payload.split("\n")[0])
        payload.runtime = time.time() - t
        log(f"{alg_id}: finished, {len(payload)} subgroups")
        return payload
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
