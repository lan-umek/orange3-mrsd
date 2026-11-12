"""
Algorithm-specific views of a subgroup – the computations behind the
inspector tabs of the algorithm widgets (``widgets/inspectors.py``).

Everything here works on a ``SubgroupResults`` and pandas / numpy only, so
it is testable without Orange.

* prototypes:          means / modes of all variables in X and Y, subgroup vs
                       complement vs all (every algorithm)
* MRSD-PT:             the contingency table of the partition pair a cell
                       belongs to (observed, expected, residuals), cluster
                       prototypes and medoids of both partitions
* MRSD-HC:             the dendrogram (re-computed exactly as the algorithm
                       did) and the node of the subgroup
* classification:      out-of-fold membership classifier in X and in Y with a
                       readable model (tree text, coefficients, importances)
* MRSD-CCA:            canonical variates and weights
* MRSD-EMM:            X–Y correlations inside vs. outside, best pair
* MRSD-MI:             pairwise mutual information inside vs. outside
* MRSD-Rules:          conditions with their individual and cumulative coverage
* MRSD-Ensemble:       supporting member algorithms
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from orangecontrib.subgroups import backend
from orangecontrib.subgroups.backend import SubgroupResults


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_cat(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def dataset(results: SubgroupResults):
    """The MRSD Dataset (typed feature spaces) of the frame the algorithm saw."""
    mrsd = backend.load_mrsd()
    return mrsd.DataLoader.from_dataframe(results.frame, X_columns=results.X_columns, Y_columns=results.Y_columns)


def membership(results: SubgroupResults, index: int) -> np.ndarray:
    return results[index].mask(len(results.frame))


# ---------------------------------------------------------------------------
# prototypes (all algorithms)
# ---------------------------------------------------------------------------

def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    vals = np.empty(m)
    vals[order] = np.clip(ranked, 0, 1)
    out[ok] = vals
    return out


def _permutation_chi2(table: np.ndarray, n_perm: int = 2000, seed: int = 0) -> float:
    """Monte-Carlo p-value of the chi-square statistic (row/column margins fixed)."""
    from scipy.stats import chi2_contingency
    rng = np.random.default_rng(seed)
    obs = chi2_contingency(table, correction=False)[0]
    rows = np.repeat(np.arange(table.shape[0]), table.sum(1).astype(int))
    cols = np.repeat(np.arange(table.shape[1]), table.sum(0).astype(int))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(cols)
        t = np.zeros_like(table)
        np.add.at(t, (rows, perm), 1)
        try:
            stat = chi2_contingency(t, correction=False)[0]
        except ValueError:
            stat = 0.0
        count += stat >= obs - 1e-12
    return (count + 1) / (n_perm + 1)


def prototype_table(results: SubgroupResults, index: int, n_perm: int = 500) -> pd.DataFrame:
    """One row per variable of X and Y with descriptive statistics of the
    subgroup and of its complement and two tests of their difference.

    Numeric variables: n, mean, SD, median, IQR for both groups, difference
    of means, Cohen's d (pooled SD), Welch's t-test (parametric) and the
    Mann-Whitney U test with Cliff's delta (non-parametric).
    Categorical variables: modal level with its share, the distribution of
    levels in both groups, the largest difference of a level's share,
    Cramer's V, the chi-square test of independence (asymptotic) and Fisher's
    exact test (2 levels) or a Monte-Carlo chi-square test (more levels).
    'p adj.' columns hold the Benjamini-Hochberg adjustment over the
    variables of the same test family.  The p-values are descriptive: the
    subgroup was selected on the same data."""
    from scipy import stats
    frame = results.frame
    mask = membership(results, index)
    comp = ~mask
    medoid = medoid_row(results, index)
    rows = []
    for space, cols in (("X", results.X_columns), ("Y", results.Y_columns)):
        for col in cols:
            s = frame[col]
            row: Dict[str, Any] = {"space": space, "variable": col}
            if _is_cat(s):
                levels = list(s.astype("category").cat.categories)
                a, b = s[mask], s[comp]
                vc_in, vc_out = a.value_counts(normalize=True), b.value_counts(normalize=True)
                table = np.array([[(a == lv).sum() for lv in levels], [(b == lv).sum() for lv in levels]], dtype=float)
                keep = table.sum(0) > 0
                table = table[:, keep]
                row.update({
                    "type": "categorical", "n subgroup": int(mask.sum()), "n complement": int(comp.sum()),
                    "centre subgroup": f"{vc_in.idxmax()} ({100 * vc_in.iloc[0]:.0f}%)" if len(vc_in) else "",
                    "spread subgroup": ", ".join(f"{lv}: {100 * vc_in.get(lv, 0):.0f}%" for lv in levels),
                    "centre complement": f"{vc_out.idxmax()} ({100 * vc_out.iloc[0]:.0f}%)" if len(vc_out) else "",
                    "spread complement": ", ".join(f"{lv}: {100 * vc_out.get(lv, 0):.0f}%" for lv in levels),
                    "difference": float(max((vc_in.get(lv, 0) - vc_out.get(lv, 0) for lv in levels),
                                            key=abs, default=np.nan)),
                })
                try:
                    chi2, p_chi, dof, _ = stats.chi2_contingency(table, correction=False)
                    n = table.sum()
                    v = float(np.sqrt(chi2 / (n * max(min(table.shape) - 1, 1))))
                except ValueError:
                    chi2, p_chi, v = np.nan, np.nan, np.nan
                row.update({"effect size": v, "parametric test": "chi-square", "statistic": float(chi2),
                            "p": float(p_chi)})
                if table.shape[1] == 2:
                    try:
                        odds, p_f = stats.fisher_exact(table.astype(int))
                    except ValueError:
                        odds, p_f = np.nan, np.nan
                    row.update({"nonparametric test": "Fisher exact", "statistic (np)": float(odds), "p (np)": float(p_f)})
                elif table.shape[1] > 2 and n_perm:
                    row.update({"nonparametric test": "chi-square (Monte Carlo)", "statistic (np)": float(chi2),
                                "p (np)": _permutation_chi2(table, n_perm)})
                else:
                    row.update({"nonparametric test": "", "statistic (np)": np.nan, "p (np)": np.nan})
            else:
                v = s.to_numpy(dtype=float)
                a, b = v[mask], v[comp]
                a, b = a[~np.isnan(a)], b[~np.isnan(b)]
                sd_a = a.std(ddof=1) if len(a) > 1 else np.nan
                sd_b = b.std(ddof=1) if len(b) > 1 else np.nan
                pooled = np.sqrt(((len(a) - 1) * sd_a ** 2 + (len(b) - 1) * sd_b ** 2) / max(len(a) + len(b) - 2, 1)) \
                    if len(a) > 1 and len(b) > 1 else np.nan
                diff = float(a.mean() - b.mean()) if len(a) and len(b) else np.nan
                row.update({
                    "type": "numeric", "n subgroup": int(len(a)), "n complement": int(len(b)),
                    "centre subgroup": float(a.mean()) if len(a) else np.nan, "spread subgroup": float(sd_a),
                    "median subgroup": float(np.median(a)) if len(a) else np.nan,
                    "IQR subgroup": float(np.subtract(*np.percentile(a, [75, 25]))) if len(a) else np.nan,
                    "centre complement": float(b.mean()) if len(b) else np.nan, "spread complement": float(sd_b),
                    "median complement": float(np.median(b)) if len(b) else np.nan,
                    "IQR complement": float(np.subtract(*np.percentile(b, [75, 25]))) if len(b) else np.nan,
                    "difference": diff, "effect size": diff / pooled if pooled and pooled > 0 else np.nan,
                })
                if len(a) > 1 and len(b) > 1:
                    t, p_t = stats.ttest_ind(a, b, equal_var=False)
                    try:
                        u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
                        delta = 2 * u / (len(a) * len(b)) - 1
                    except ValueError:
                        u, p_u, delta = np.nan, np.nan, np.nan
                else:
                    t, p_t, u, p_u, delta = np.nan, np.nan, np.nan, np.nan, np.nan
                row.update({"parametric test": "Welch t", "statistic": float(t), "p": float(p_t),
                            "nonparametric test": "Mann-Whitney U", "statistic (np)": float(u), "p (np)": float(p_u),
                            "Cliff delta": float(delta)})
            row["medoid"] = "" if medoid is None else (f"{frame.loc[medoid, col]:.3g}" if row["type"] == "numeric"
                                                       else str(frame.loc[medoid, col]))
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["p adj."] = _bh(df["p"].to_numpy(dtype=float))
        df["p adj. (np)"] = _bh(df["p (np)"].to_numpy(dtype=float))
        order = ["space", "variable", "type", "n subgroup", "n complement", "centre subgroup", "spread subgroup",
                 "median subgroup", "IQR subgroup", "centre complement", "spread complement", "median complement",
                 "IQR complement", "difference", "effect size", "Cliff delta", "parametric test", "statistic", "p",
                 "p adj.", "nonparametric test", "statistic (np)", "p (np)", "p adj. (np)", "medoid"]
        df = df[[c for c in order if c in df.columns]]
    return df


def profile_values(results: SubgroupResults, index: int) -> pd.DataFrame:
    """Per variable / level: mean (numeric) or share (categorical level) in the
    subgroup, in the complement and in all data, their difference and the
    standardized difference – the data behind the prototype chart."""
    frame = results.frame
    mask = membership(results, index)
    comp = ~mask
    rows = []
    for space, cols in (("X", results.X_columns), ("Y", results.Y_columns)):
        for col in cols:
            s = frame[col]
            if _is_cat(s):
                for lv in s.astype("category").cat.categories:
                    p_in = float((s[mask] == lv).mean()) if mask.any() else np.nan
                    p_out = float((s[comp] == lv).mean()) if comp.any() else np.nan
                    p_all = float((s == lv).mean())
                    sd = np.sqrt(p_all * (1 - p_all))
                    rows.append({"space": space, "name": f"{col}={lv}", "kind": "share", "subgroup": p_in,
                                 "complement": p_out, "all": p_all, "difference": p_in - p_out,
                                 "standardized": (p_in - p_out) / sd if sd > 0 else np.nan})
            else:
                v = s.to_numpy(dtype=float)
                sd = np.nanstd(v, ddof=1)
                m_in, m_out = np.nanmean(v[mask]) if mask.any() else np.nan, np.nanmean(v[comp]) if comp.any() else np.nan
                rows.append({"space": space, "name": col, "kind": "mean", "subgroup": m_in, "complement": m_out,
                             "all": float(np.nanmean(v)), "difference": m_in - m_out,
                             "standardized": (m_in - m_out) / sd if sd > 0 else np.nan})
    return pd.DataFrame(rows)


def medoid_row(results: SubgroupResults, index: int, space: str = "XY") -> Optional[int]:
    """Row (position in the frame) of the subgroup member closest to the
    others in the standardized / one-hot space of X, Y or both."""
    mask = membership(results, index)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    if len(idx) == 1:
        return int(idx[0])
    ds = dataset(results)
    parts = []
    if "X" in space:
        parts.append(ds.X_space.numeric)
    if "Y" in space:
        parts.append(ds.Y_space.numeric)
    M = np.hstack(parts)[idx]
    M = np.nan_to_num(M)
    D = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(-1))
    return int(idx[int(np.argmin(D.sum(1)))])


def standardized_profile(results: SubgroupResults, index: int) -> pd.DataFrame:
    """Rows = variables / levels with the space and the standardized difference."""
    df = profile_values(results, index)
    return df[["space", "name", "standardized"]].rename(columns={"standardized": "value"})


# ---------------------------------------------------------------------------
# MRSD-PT: contingency tables and partitions
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ContingencyView:
    key: Tuple[str, int, str, int]            # (clustering_X, k, clustering_Y, l)
    labels_x: np.ndarray
    labels_y: np.ndarray
    observed: np.ndarray                      # k × l
    expected: np.ndarray
    residuals: np.ndarray                     # adjusted standardized residuals
    subgroup_at: Dict[Tuple[int, int], int]   # (i, j) -> index of the reported subgroup for that cell
    cell: Optional[Tuple[int, int]]           # the selected subgroup's cell


def pt_partition_keys(results: SubgroupResults) -> List[Tuple[str, int, str, int]]:
    keys = []
    for sg in results.subgroups:
        if sg.partition_result_X is None or sg.partition_result_Y is None:
            continue
        key = (sg.clustering_X, int(sg.k), sg.clustering_Y, int(sg.l))
        if key not in keys:
            keys.append(key)
    return keys


def pt_contingency(results: SubgroupResults, index: Optional[int] = None,
                   key: Optional[Tuple[str, int, str, int]] = None) -> Optional[ContingencyView]:
    """Contingency table of the partition pair of subgroup `index` (or of `key`)."""
    if key is None:
        if index is None:
            return None
        sg = results[index]
        if sg.partition_result_X is None or sg.partition_result_Y is None:
            return None
        key = (sg.clustering_X, int(sg.k), sg.clustering_Y, int(sg.l))
    ref = None
    for cand in results.subgroups:
        if cand.partition_result_X is not None and (cand.clustering_X, int(cand.k), cand.clustering_Y,
                                                    int(cand.l)) == key:
            ref = cand
            break
    if ref is None:
        return None
    lx = np.asarray(ref.partition_result_X.labels, dtype=int)
    ly = np.asarray(ref.partition_result_Y.labels, dtype=int)
    k, l = int(lx.max()) + 1, int(ly.max()) + 1
    obs = np.zeros((k, l))
    np.add.at(obs, (lx, ly), 1)
    n = obs.sum()
    rs, cs = obs.sum(1, keepdims=True), obs.sum(0, keepdims=True)
    exp = rs @ cs / n
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (obs - exp) / np.sqrt(exp * (1 - rs / n) * (1 - cs / n))
    resid = np.nan_to_num(resid)
    at = {}
    for i, sg in enumerate(results.subgroups):
        if sg.partition_result_X is not None and (sg.clustering_X, int(sg.k), sg.clustering_Y, int(sg.l)) == key:
            at[(int(sg.cell_i), int(sg.cell_j))] = i
    cell = None
    if index is not None:
        sg = results[index]
        if sg.partition_result_X is not None and (sg.clustering_X, int(sg.k), sg.clustering_Y, int(sg.l)) == key:
            cell = (int(sg.cell_i), int(sg.cell_j))
    return ContingencyView(key, lx, ly, obs, exp, resid, at, cell)


def cluster_prototypes(results: SubgroupResults, labels: np.ndarray, columns: List[str],
                       highlight: Optional[int] = None) -> pd.DataFrame:
    """Rows = clusters of a partition; columns = variables (mean or modal level
    with its share), plus size and the medoid row in that space."""
    frame = results.frame
    labels = np.asarray(labels, dtype=int)
    ds = dataset(results)
    space = ds.X_space if columns == results.X_columns else ds.Y_space
    num = np.nan_to_num(space.numeric)
    rows = []
    for c in range(int(labels.max()) + 1):
        idx = np.flatnonzero(labels == c)
        row: Dict[str, Any] = {"cluster": c, "size": int(len(idx)), "selected": highlight is not None and c == highlight}
        for col in columns:
            s = frame[col].iloc[idx]
            if _is_cat(s):
                vc = s.value_counts(normalize=True)
                row[col] = f"{vc.idxmax()} ({100 * vc.iloc[0]:.0f}%)" if len(vc) else ""
            else:
                row[col] = float(np.nanmean(s.to_numpy(dtype=float))) if len(idx) else np.nan
        if len(idx):
            M = num[idx]
            D = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(-1))
            row["medoid row"] = int(idx[int(np.argmin(D.sum(1)))])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MRSD-HC: dendrogram
# ---------------------------------------------------------------------------

def hc_linkage(results: SubgroupResults) -> Optional[np.ndarray]:
    """Re-compute the linkage matrix of Y exactly as MRSD-HC did (same
    method / metric, taken from the subgroups' metadata)."""
    from scipy.cluster.hierarchy import linkage
    sg = next((s for s in results.subgroups if s.clustering_metadata_Y.get("linkage_method")), None)
    if sg is None:
        return None
    meta = sg.clustering_metadata_Y
    method = meta.get("linkage_method", "ward")
    metric = meta.get("distance_metric_used", "euclidean")
    if metric == "precomputed":
        return None
    Y = dataset(results).Y_space.numeric
    w = meta.get("feature_weights")
    if w is not None and len(w) == Y.shape[1]:
        Y = Y * np.sqrt(np.asarray(w, dtype=float))
    return linkage(np.nan_to_num(Y), method=method, metric=metric)


def hc_node_of(results: SubgroupResults, index: int) -> Optional[int]:
    sg = results[index]
    return int(sg.k) if sg.k is not None and sg.clustering_metadata_Y.get("linkage_method") else None


def dendrogram_segments(Z: np.ndarray, n: int, highlight_node: Optional[int] = None,
                        max_leaves: int = 400) -> Dict[str, Any]:
    """Line segments of a dendrogram for plotting.

    Returns {'segments': list of (x0, y0, x1, y1), 'highlight': list of segments
    of the subtree under `highlight_node`, 'node_xy': {node_id: (x, height)},
    'leaf_order': leaf ids left to right}.  Leaves are placed at 0 … n-1.
    If n > max_leaves the leaves are still drawn (as positions only)."""
    from scipy.cluster.hierarchy import to_tree
    root = to_tree(Z)
    order: List[int] = []
    x_of: Dict[int, float] = {}
    h_of: Dict[int, float] = {}
    segments: List[Tuple[float, float, float, float]] = []
    owner: Dict[int, int] = {}      # node id -> parent id (to find the subtree)

    def walk(node):
        if node.is_leaf():
            x_of[node.id] = float(len(order))
            h_of[node.id] = 0.0
            order.append(node.id)
            return
        walk(node.left)
        walk(node.right)
        owner[node.left.id] = node.id
        owner[node.right.id] = node.id
        xl, xr = x_of[node.left.id], x_of[node.right.id]
        hl, hr = h_of[node.left.id], h_of[node.right.id]
        x_of[node.id] = (xl + xr) / 2
        h_of[node.id] = float(node.dist)
        segments.append((xl, hl, xl, node.dist))
        segments.append((xr, hr, xr, node.dist))
        segments.append((xl, node.dist, xr, node.dist))

    import sys
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10 * n + 100))
    try:
        walk(root)
    finally:
        sys.setrecursionlimit(old)

    highlight: List[Tuple[float, float, float, float]] = []
    if highlight_node is not None and highlight_node in h_of:
        # segments whose top is at or below the highlighted node and inside its subtree
        def in_subtree(nid):
            while nid in owner:
                if nid == highlight_node:
                    return True
                nid = owner[nid]
            return nid == highlight_node
        for nid, h in h_of.items():
            if nid >= n and in_subtree(nid):
                x = x_of[nid]
                # re-create the three segments of this merge
                # (find children by scanning: cheap enough for a few hundred nodes)
                kids = [c for c, p in owner.items() if p == nid]
                if len(kids) == 2:
                    a, b = kids
                    highlight.append((x_of[a], h_of[a], x_of[a], h))
                    highlight.append((x_of[b], h_of[b], x_of[b], h))
                    highlight.append((x_of[a], h, x_of[b], h))
    return {"segments": segments, "highlight": highlight, "node_xy": {k: (x_of[k], h_of[k]) for k in h_of},
            "leaf_order": order}


# ---------------------------------------------------------------------------
# classification-based algorithms: out-of-fold membership classifier
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ClassifierReport:
    space: str
    model_name: str
    auc_oof: float
    auc_train: float
    n_folds: int
    text: str                       # readable description of the model fitted on all rows
    importances: Optional[pd.DataFrame]   # variable, importance / coefficient (when available)


def classifier_report(results: SubgroupResults, index: int, space: str = "X", model_name: str = "decision_tree",
                      n_folds: int = 5, random_state: int = 0, fast: bool = True) -> ClassifierReport:
    """Membership (in / out of the subgroup) predicted from X or from Y with
    one of the MRSD classifiers: honest out-of-fold AUC, training AUC and a
    readable model (decision-tree rules, coefficients, feature importances)."""
    from sklearn.metrics import roc_auc_score
    mrsd = backend.load_mrsd()
    ds = dataset(results)
    fs = ds.X_space if space == "X" else ds.Y_space
    M = np.nan_to_num(fs.numeric)
    names = list(fs.numeric_names)
    y = membership(results, index).astype(int)
    est = mrsd.make_classifier(model_name, random_state=random_state, fast=fast)
    oof = mrsd.OutOfFoldClassifier(est, n_folds=n_folds, random_state=random_state)
    try:
        auc_oof = float(oof.auc(M, y))
    except ValueError:
        auc_oof = float("nan")
    k = max(2, min(n_folds, int(y.sum()), int(len(y) - y.sum())))
    est.fit(M, y)
    if hasattr(est, "predict_proba"):
        pr = est.predict_proba(M)
        score = pr[:, list(est.classes_).index(1)] if 1 in est.classes_ else np.zeros(len(y))
    elif hasattr(est, "decision_function"):
        score = est.decision_function(M)
    else:
        score = est.predict(M)
    auc_train = float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")
    text, imp = describe_sklearn(est, names)
    return ClassifierReport(space, model_name, auc_oof, auc_train, k, text, imp)


def describe_sklearn(est, names: List[str]) -> Tuple[str, Optional[pd.DataFrame]]:
    """Readable form of a fitted sklearn estimator."""
    imp = None
    try:
        from sklearn.tree import export_text
        if hasattr(est, "tree_"):
            text = export_text(est, feature_names=[n[:40] for n in names], decimals=3, show_weights=True)
            imp = pd.DataFrame({"variable": names, "importance": est.feature_importances_})
            return text, imp.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:              # noqa: BLE001 - fall through to generic descriptions
        pass
    if hasattr(est, "coef_"):
        coef = np.ravel(est.coef_) if np.asarray(est.coef_).shape[0] == 1 else np.asarray(est.coef_)[-1]
        imp = pd.DataFrame({"variable": names, "coefficient": coef}).iloc[np.argsort(-np.abs(coef))]
        intercept = float(np.ravel(est.intercept_)[0]) if hasattr(est, "intercept_") else 0.0
        text = f"{type(est).__name__}: linear score = {intercept:+.3f} " + " ".join(
            f"{c:+.3f}·{n}" for n, c in zip(imp["variable"], imp["coefficient"]))
        return text, imp.reset_index(drop=True)
    if hasattr(est, "feature_importances_"):
        imp = pd.DataFrame({"variable": names, "importance": est.feature_importances_})
        imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
        return f"{type(est).__name__} with {getattr(est, 'n_estimators', '?')} estimators; " \
               f"most important: {', '.join(imp['variable'].head(5))}", imp
    if hasattr(est, "theta_"):      # Gaussian naive Bayes
        theta = est.theta_
        imp = pd.DataFrame({"variable": names, "mean in": theta[-1], "mean out": theta[0]})
        return f"{type(est).__name__}: class-conditional means", imp
    return f"{type(est).__name__} (no readable summary; see the Subgroup Classifier widget)", None


def classifier_names_for(results: SubgroupResults, index: int) -> Tuple[str, str]:
    """Classifier names used by the algorithm for X and Y (from config / metadata)."""
    cfg = results.config or {}
    sg = results[index]
    cx = cfg.get("classifier_X") or sg.clustering_metadata_X.get("classifier") or \
        (sg.quality_measures or {}).get("best_model") or "decision_tree"
    if isinstance(cx, (list, tuple)):
        cx = (sg.quality_measures or {}).get("best_model") or cx[0]
    cy = cfg.get("classifier_Y") or cx
    # estimator objects (e.g. Orange learners) have no MRSD name: fall back to a tree
    cx = cx if isinstance(cx, str) else "decision_tree"
    cy = cy if isinstance(cy, str) else "decision_tree"
    return cx, cy


# ---------------------------------------------------------------------------
# MRSD-CCA
# ---------------------------------------------------------------------------

def cca_view(results: SubgroupResults, index: int) -> Dict[str, Any]:
    """First canonical variates (u, v) of all rows, the subgroup mask,
    correlations inside / outside and the canonical weights."""
    mrsd = backend.load_mrsd()
    cfg = results.config or {}
    ds = dataset(results)
    X, Y = np.nan_to_num(ds.X_space.numeric), np.nan_to_num(ds.Y_space.numeric)
    cca = mrsd.RegularizedCCA(n_components=int(cfg.get("n_components", 2)),
                              regularization=float(cfg.get("regularization", 0.0))).fit(X, Y)
    U, V = cca.transform(X, Y)
    mask = membership(results, index)
    comp = ~mask

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")
    weights = pd.concat([
        pd.DataFrame({"space": "X", "variable": list(ds.X_space.numeric_names),
                      **{f"w{c + 1}": cca.x_weights_[:, c] for c in range(cca.x_weights_.shape[1])}}),
        pd.DataFrame({"space": "Y", "variable": list(ds.Y_space.numeric_names),
                      **{f"w{c + 1}": cca.y_weights_[:, c] for c in range(cca.y_weights_.shape[1])}}),
    ], ignore_index=True)
    return {"u": U[:, 0], "v": V[:, 0], "mask": mask,
            "rho_all": float(cca.canonical_correlations_[0]),
            "rho_subgroup": corr(U[mask, 0], V[mask, 0]), "rho_complement": corr(U[comp, 0], V[comp, 0]),
            "weights": weights, "n_components": int(cca.x_weights_.shape[1])}


# ---------------------------------------------------------------------------
# MRSD-EMM: correlations
# ---------------------------------------------------------------------------

def emm_correlations(results: SubgroupResults, index: int) -> pd.DataFrame:
    """Pearson correlation of every numeric X with every numeric Y inside the
    subgroup and in the complement, with the Fisher-z of the difference."""
    frame = results.frame
    mask = membership(results, index)
    comp = ~mask
    rows = []
    num_x = [c for c in results.X_columns if not _is_cat(frame[c])]
    num_y = [c for c in results.Y_columns if not _is_cat(frame[c])]

    def r(a, b):
        if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    for x in num_x:
        for y in num_y:
            a_in, b_in = frame[x].to_numpy(float)[mask], frame[y].to_numpy(float)[mask]
            a_out, b_out = frame[x].to_numpy(float)[comp], frame[y].to_numpy(float)[comp]
            r_in, r_out = r(a_in, b_in), r(a_out, b_out)
            n_in, n_out = mask.sum(), comp.sum()
            if np.isnan(r_in) or np.isnan(r_out) or n_in < 4 or n_out < 4:
                z = np.nan
            else:
                z = (np.arctanh(np.clip(r_in, -0.999999, 0.999999)) - np.arctanh(np.clip(r_out, -0.999999, 0.999999))) \
                    / np.sqrt(1 / (n_in - 3) + 1 / (n_out - 3))
            rows.append({"X": x, "Y": y, "r subgroup": r_in, "r complement": r_out, "z": z})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.iloc[np.argsort(-np.abs(df["z"].fillna(0).to_numpy()))].reset_index(drop=True)
    return df


def best_pair(results: SubgroupResults, index: int) -> Optional[Tuple[str, str]]:
    qm = results[index].quality_measures or {}
    bp = qm.get("best_pair")
    if isinstance(bp, str) and "~" in bp:
        x, y = bp.split("~", 1)
        return x.strip(), y.strip()
    df = emm_correlations(results, index)
    return (df.iloc[0]["X"], df.iloc[0]["Y"]) if len(df) else None


# ---------------------------------------------------------------------------
# MRSD-MI
# ---------------------------------------------------------------------------

def mi_pairs(results: SubgroupResults, index: int, k: int = 5, random_state: int = 0) -> pd.DataFrame:
    """Mutual information of every X variable with every Y variable (and the
    joint X-Y MI) inside the subgroup vs. in the complement, with the
    estimator for mixed variable types (one column per original variable)."""
    mrsd = backend.load_mrsd()
    from orangecontrib.subgroups.mrsd import mrsd_mi as mi_mod
    ds = dataset(results)
    X, Y = mi_mod.mixed_matrix(ds.X_space), mi_mod.mixed_matrix(ds.Y_space)
    mask = membership(results, index)
    comp = ~mask
    rows = []

    def mi(a, b, m):
        if m.sum() < max(k + 2, 8):
            return np.nan
        return float(mi_mod.mixed_ksg_mi(a[m], b[m], k=k))
    rows.append({"X": "(all X)", "Y": "(all Y)", "MI subgroup": mi(X, Y, mask), "MI complement": mi(X, Y, comp)})
    for i, x in enumerate(ds.X_space.columns):
        for j, y in enumerate(ds.Y_space.columns):
            rows.append({"X": x, "Y": y, "MI subgroup": mi(X[:, [i]], Y[:, [j]], mask),
                         "MI complement": mi(X[:, [i]], Y[:, [j]], comp)})
    df = pd.DataFrame(rows)
    df["difference"] = df["MI subgroup"] - df["MI complement"]
    return df


# ---------------------------------------------------------------------------
# MRSD-Rules: conditions
# ---------------------------------------------------------------------------

def rule_conditions(results: SubgroupResults, index: int) -> pd.DataFrame:
    """Each condition of the rule with the coverage of the condition alone and
    the cumulative coverage of the conjunction up to it."""
    sg = results[index]
    if sg.rule is None or len(sg.rule) == 0:
        return pd.DataFrame(columns=["condition", "alone", "cumulative"])
    fs = dataset(results).X_space
    n = fs.n
    cum = np.ones(n, dtype=bool)
    rows = []
    for c in sorted(sg.rule.conditions, key=lambda c: c.attribute_idx):
        m = c.evaluate(fs)
        cum &= m
        rows.append({"condition": str(c), "alone": int(m.sum()), "alone %": 100 * m.mean(),
                     "cumulative": int(cum.sum()), "cumulative %": 100 * cum.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MRSD-Ensemble: support
# ---------------------------------------------------------------------------

def ensemble_support(results: SubgroupResults, index: int) -> pd.DataFrame:
    meta = results[index].clustering_metadata_X or {}
    algs = meta.get("algorithms") or []
    jac = meta.get("support_jaccard") or {}
    ranks = meta.get("member_ranks") or {}
    rep = (meta.get("representative") or {}).get("algorithm")
    rows = [{"algorithm": backend.ALGORITHM_BY_ID[a].name if a in backend.ALGORITHM_BY_ID else a,
             "rank within algorithm": ranks.get(a), "Jaccard with consensus": jac.get(a),
             "representative": a == rep} for a in algs]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# generic: quality measures as a table
# ---------------------------------------------------------------------------

def quality_measures_table(results: SubgroupResults, index: int) -> pd.DataFrame:
    qm = results[index].quality_measures or {}
    rows = [{"measure": k, "value": v} for k, v in qm.items()
            if isinstance(v, (int, float, str, bool, np.integer, np.floating))]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ad-hoc subgroups chosen interactively (dendrogram node, contingency cell)
# ---------------------------------------------------------------------------

def adhoc_subgroup(results: SubgroupResults, mask: np.ndarray, description: str,
                   extra: Optional[Dict[str, Any]] = None, quality: Optional[float] = None, **fields):
    """Build a Subgroup object for an interactively chosen set of rows (mask
    over the analysed frame): rule from a decision tree, target statistics;
    no p-value (it was not tested)."""
    mrsd = backend.load_mrsd()
    ds = dataset(results)
    mask = np.asarray(mask, dtype=bool)
    desc = mrsd.describe_mask_with_tree(ds.X_space, mask, max_depth=3)
    qm = {"quality": quality, "interactive": True}
    qm.update(extra or {})
    sg = mrsd.Subgroup(description=description, covered_instances=set(int(i) for i in np.flatnonzero(mask)),
                       quality_measures=qm, algorithm=results.algorithm, rule=desc.rule,
                       description_precision=float(desc.precision), description_recall=float(desc.recall), **fields)
    sg.fill_target_statistics(results.frame[results.Y_columns])
    return sg


def with_adhoc(results: SubgroupResults, subgroup, name: str) -> Tuple[SubgroupResults, int]:
    """A copy of the results with the ad-hoc subgroup appended; returns it and the index."""
    view = results.copy()
    view.subgroups.add(subgroup)
    idx = len(view) - 1
    view.names[idx] = name
    return view, idx


def hc_node_mask(Z: np.ndarray, n: int, node_id: int) -> np.ndarray:
    """Rows under a node of the dendrogram (node ids as in scipy: leaves 0..n-1, merges n..2n-2)."""
    mask = np.zeros(n, dtype=bool)
    stack = [int(node_id)]
    while stack:
        k = stack.pop()
        if k < n:
            mask[k] = True
        else:
            left, right = Z[k - n, 0], Z[k - n, 1]
            stack.append(int(left))
            stack.append(int(right))
    return mask


def hc_node_quality(results: SubgroupResults, mask: np.ndarray, n_folds: int = 3, random_state: int = 0) -> float:
    """Out-of-fold AUC of membership from X with the classifier of the run (as MRSD-HC scores nodes)."""
    mrsd = backend.load_mrsd()
    cfg = results.config or {}
    name = cfg.get("classification_model", "decision_tree")
    if isinstance(name, (list, tuple)):
        name = name[0]
    est = mrsd.make_classifier(name, random_state=random_state, fast=bool(cfg.get("fast_classifiers", True)))
    X = np.nan_to_num(dataset(results).X_space.numeric)
    try:
        return float(mrsd.OutOfFoldClassifier(est, n_folds=int(cfg.get("n_folds", n_folds)),
                                              random_state=random_state).auc(X, mask.astype(int)))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# MRSD-MV: the two dendrograms (X and Y)
# ---------------------------------------------------------------------------

def mv_linkages(results: SubgroupResults):
    """Linkage matrices of X and of Y computed as the 'dual_dendrogram' method
    of MRSD-MV does (same linkage, same weights of the X variables), and the
    weights per X variable ({} when unweighted)."""
    backend.load_mrsd()
    from orangecontrib.subgroups.mrsd import mrsd_mv as mv
    cfg = results.config or {}
    ds = dataset(results)
    Y = np.nan_to_num(ds.Y_space.numeric)
    computer = mv.AffinityComputer(mv.KernelType(cfg.get("kernel_type", "rbf")), cfg.get("gamma"))
    X_w, weights = mv.weighted_X(ds.X_space, Y, computer, cfg.get("x_weighting", "relevance"),
                                 float(cfg.get("weight_power", 2.0)))
    Z_X, Z_Y = mv.dual_linkages(X_w, Y, cfg.get("linkage_method", "ward"))
    return Z_X, Z_Y, weights


def node_pair_statistics(mask_a: np.ndarray, mask_b: np.ndarray) -> Dict[str, float]:
    """Sizes, intersection, expectation under independence, adjusted residual and Jaccard of two row sets."""
    n = len(mask_a)
    a, b = int(mask_a.sum()), int(mask_b.sum())
    o = int((mask_a & mask_b).sum())
    e = a * b / n
    den = np.sqrt(e * (1 - a / n) * (1 - b / n))
    union = a + b - o
    return {"size_a": a, "size_b": b, "observed": o, "expected": e,
            "residual": float((o - e) / den) if den > 0 else 0.0, "jaccard": o / union if union else 0.0}


def best_matching_node(Z: np.ndarray, n: int, mask: np.ndarray, n_min: int = 2) -> Tuple[Optional[int], float]:
    """The node of a dendrogram whose row set has the largest Jaccard with `mask`."""
    members: Dict[int, np.ndarray] = {}
    best, best_j = None, -1.0
    target = int(mask.sum())
    for t in range(Z.shape[0]):
        nid = n + t
        a, b = int(Z[t, 0]), int(Z[t, 1])
        ma = members.pop(a) if a >= n else np.array([a])
        mb = members.pop(b) if b >= n else np.array([b])
        mem = np.concatenate([ma, mb])
        members[nid] = mem
        if len(mem) < n_min:
            continue
        inter = int(mask[mem].sum())
        j = inter / (len(mem) + target - inter) if (len(mem) + target - inter) else 0.0
        if j > best_j:
            best, best_j = nid, j
    return best, float(best_j)
