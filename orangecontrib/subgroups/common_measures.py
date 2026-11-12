"""
Common quality measures, computed from the subgroup itself (its set of rows)
in the same way whatever algorithm produced it, so that subgroups of
different algorithms can be compared.  All are in [0, 1], larger is better,
and 1 need not be attainable.

* distinct_Y   2 * AUC - 1 of predicting membership from Y (out-of-fold):
               how well the subgroup is told apart by the dependent variables.
* describe_X   the same from X: how well the independent variables describe
               the subgroup (close to 1 for rule-based subgroups).
* relation     share of the subgroup's own variation of Y that is explained
               only when the model of Y from X may differ inside the subgroup
               (cross-validated ridge regression: Y ~ X  versus  Y ~ X + g
               [+ g*X], errors on the subgroup's rows).  Captures a shifted
               level as well as a changed slope / correlation, which
               distinct_Y does not see; does not depend on the subgroup's size.

The measures are computed on the data the subgroup was found on, so they are
optimistic; they are meant for comparing subgroups, not as estimates of
replicability.  Stored in ``Subgroup.quality_measures`` under MEASURES keys.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from orangecontrib.subgroups import backend, inspection
from orangecontrib.subgroups.backend import SubgroupResults

MEASURES = {
    "common_distinct_Y": "Y distinctness",
    "common_describe_X": "X describability",
    "common_relation": "Relation exceptionality",
}


def _gini(M: np.ndarray, y: np.ndarray, classifier: str, n_folds: int, seed: int) -> float:
    mrsd = backend.load_mrsd()
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        return float("nan")
    est = mrsd.make_classifier(classifier, random_state=seed, fast=True)
    try:
        auc = float(mrsd.OutOfFoldClassifier(est, n_folds=n_folds, random_state=seed).auc(M, y))
    except ValueError:
        return float("nan")
    return float(np.clip(2.0 * auc - 1.0, 0.0, 1.0))


def relation_exceptionality(X: np.ndarray, Y: np.ndarray, mask: np.ndarray, n_folds: int = 5, seed: int = 0,
                            ridge: float = 1.0) -> float:
    """How much better Y of the subgroup's rows is predicted when the model of
    Y from X may differ inside the subgroup.

    Cross-validated ridge regressions on all rows: A: Y ~ X (one model for
    everybody), B1: Y ~ X + g (own level), B2: Y ~ X + g + g*X (own level and
    slopes), g = membership.  With the squared errors summed over the test
    rows *inside the subgroup* and over the (standardized) Y columns,
        relation = (SSE_A - min(SSE_B1, SSE_B2)) / SS of Y in the subgroup about the overall mean,
    clipped to [0, 1]: the share of the subgroup's own Y variation that only a
    subgroup-specific model explains.  It does not depend on the subgroup's
    share of the data."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import StratifiedKFold
    g = mask.astype(float)
    n_g = int(g.sum())
    folds = min(n_folds, n_g, len(g) - n_g)
    if folds < 2:
        return float("nan")
    Xc = X - X.mean(axis=0)
    designs = [Xc, np.hstack([Xc, g[:, None]]), np.hstack([Xc, g[:, None], Xc * g[:, None]])]
    Ys = Y - Y.mean(axis=0)
    sst = float((Ys[mask] ** 2).sum())
    if sst <= 0:
        return float("nan")
    sse = [0.0, 0.0, 0.0]
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xc, mask):
        mu = Ys[train].mean(axis=0)
        inside = test[mask[test]]
        for d, D in enumerate(designs):
            pred = Ridge(alpha=ridge).fit(D[train], Ys[train] - mu).predict(D[inside]) + mu
            sse[d] += float(((Ys[inside] - pred) ** 2).sum())
    return float(np.clip((sse[0] - min(sse[1], sse[2])) / sst, 0.0, 1.0))


def compute(results: SubgroupResults, index: int, classifier: str = "decision_tree", n_folds: int = 3,
            seed: int = 0, dataset=None) -> Dict[str, float]:
    """The three common measures of one subgroup (also written into its quality_measures)."""
    ds = dataset if dataset is not None else inspection.dataset(results)
    X = np.nan_to_num(ds.X_space.numeric)
    Y = np.nan_to_num(ds.Y_space.numeric)
    mask = inspection.membership(results, index)
    out = {
        "common_distinct_Y": _gini(Y, mask.astype(int), classifier, n_folds, seed),
        "common_describe_X": _gini(X, mask.astype(int), classifier, n_folds, seed),
        "common_relation": relation_exceptionality(X, Y, mask, seed=seed),
    }
    results[index].quality_measures.update(out)
    return out


def compute_all(results: SubgroupResults, classifier: str = "decision_tree", n_folds: int = 3, seed: int = 0,
                cache: Optional[dict] = None, progress=None) -> None:
    """Common measures of every subgroup (skipping those already computed / cached)."""
    if results is None or len(results) == 0:
        return
    ds = inspection.dataset(results)
    for i, sg in enumerate(results.subgroups):
        key = (frozenset(sg.covered_instances), classifier, n_folds, seed)
        if cache is not None and key in cache:
            sg.quality_measures.update(cache[key])
        else:
            vals = compute(results, i, classifier, n_folds, seed, ds)
            if cache is not None:
                cache[key] = vals
        if progress is not None:
            progress(100.0 * (i + 1) / len(results))
