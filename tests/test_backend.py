"""Backend: running algorithms and wrapping results (no Orange needed)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from orangecontrib.subgroups import backend, params

from orangecontrib.subgroups.mrsd import synthetic  # vendored copy of the MRSD data generators


@pytest.fixture(scope="module")
def planted():
    df, X, Y, mask = synthetic.planted_shift(n=200, seed=0, categorical=True)
    df.loc[3, "x1"] = np.nan
    return df, X, Y, mask


@pytest.fixture(scope="module")
def rules_result(planted):
    df, X, Y, _ = planted
    s = params.setting_defaults("rules")
    s["n_search_permutations"] = 10
    return backend.run_algorithm("rules", df, X, Y, params.widget_config("rules", s))


def test_missing_rows_are_dropped_and_mapped(rules_result, planted):
    df, X, Y, mask = planted
    r = rules_result
    assert len(r.frame) == len(df) - 1
    assert 3 not in r.row_indices
    assert len(r.mask(0)) == len(df)
    assert r.mask(0).sum() == r[0].coverage


def test_summary_and_effects(rules_result):
    r = rules_result
    sf = r.summary_frame()
    assert list(sf.columns[:6]) == ["name", "algorithm", "rule", "prototype_Y", "description", "size"]
    assert "y1 = " in r.y_prototype_text(0) and "y3 = " in r.y_prototype_text(0)
    eff = r.target_effects()
    assert "y1" in eff.columns and "y3=u" in eff.columns
    assert len(eff) == len(r)
    J = r.jaccard_matrix()
    assert np.allclose(np.diag(J), 1.0)
    json.dumps(r.to_dict())


def test_subset_and_combine(rules_result):
    r = rules_result
    r.names[0] = "my group"
    sub = r.subset([2, 0])
    assert len(sub) == 2 and sub.name(1) == "my group" and sub[1] is r[0]
    both = backend.combine_results([r, sub])
    assert len(both) == len(r) + 2
    assert both.algorithm == "rules"
    other = r.subset([0])
    other.row_indices = other.row_indices[:-1]
    with pytest.raises(ValueError):
        backend.combine_results([r, other])


def test_recovers_planted_subgroup(planted):
    df, X, Y, mask = planted
    s = params.setting_defaults("rules")
    s.update(n_search_permutations=20, random_state=0, random_state_enabled=True)
    r = backend.run_algorithm("rules", df, X, Y, params.widget_config("rules", s))
    planted_rows = set(np.flatnonzero(mask))
    best = max((len(set(np.flatnonzero(r.mask(i))) & planted_rows) /
                len(set(np.flatnonzero(r.mask(i))) | planted_rows) for i in range(len(r))), default=0)
    assert best > 0.5


def test_save_formats(rules_result, tmp_path):
    from orangecontrib.subgroups import export as sv
    r = rules_result
    for fmt, ext in [("csv", ".csv"), ("json", ".json"), ("pkl", ".pkl"), ("txt", ".txt")]:
        path = str(tmp_path / f"out{ext}")
        sv.save_results(r, path, fmt, membership=True)
        assert os.path.getsize(path) > 0
    assert os.path.exists(tmp_path / "out_membership.csv")
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return
    sv.save_results(r, str(tmp_path / "out.xlsx"), "xlsx")
    assert pd.read_excel(tmp_path / "out.xlsx", sheet_name=None).keys() >= {"subgroups", "membership"}


def test_consolidation(rules_result):
    from orangecontrib.subgroups import consensus
    r = rules_result
    c = consensus.consolidate(r, threshold=0.5)
    assert sum(len(cl.members) for cl in c.clusters + c.dropped) == len(r)
    assert all(cl.representative in cl.members for cl in c.clusters)
    k3 = consensus.consolidate(r, n_clusters=3)
    assert len(k3.clusters) == 3
    med = consensus.consolidate(r, threshold=0.5, representative="medoid")
    assert len(med.clusters) == len(c.clusters)
    greedy = consensus.greedy_selection(r, threshold=0.5)
    S = consensus.similarity_matrix(r)
    for a in range(len(greedy)):
        for b in range(a):
            assert S[greedy[a], greedy[b]] < 0.5
    assert consensus.within_algorithm_ranks(r).min() == 1


def test_cleaning_matches_package_filter_redundant(rules_result):
    from orangecontrib.subgroups import consensus
    r = rules_result
    mrsd = backend.load_mrsd()
    own = r.subgroups.filter_redundant(mrsd.SetOfSubgroups.jaccard_similarity, mrsd.QualityMeasures.quality,
                                       s_min=0.5)
    ours = consensus.clean(r, s_min=0.5, quality="quality")
    assert [list(r.subgroups).index(sg) for sg in own] == ours
    S = consensus.similarity_matrix(r)
    for a in range(len(ours)):
        for b in range(a):
            assert S[ours[a], ours[b]] < 0.5


def test_inspection_views(planted):
    from orangecontrib.subgroups import inspection as I
    df, X, Y, _ = planted
    df = df.dropna()
    def run(alg, **over):
        s = params.setting_defaults(alg)
        s.update(over)
        return backend.run_algorithm(alg, df, X, Y, params.widget_config(alg, s))
    pt = run("pt", n_permutations=20, k_values="2, 3")
    proto = I.prototype_table(pt, 0)
    assert set(proto["space"]) == {"X", "Y"} and len(proto) == len(X) + len(Y)
    cv = I.pt_contingency(pt, 0)
    assert cv is not None and cv.observed.sum() == len(pt.frame) and cv.cell in cv.subgroup_at
    assert cv.subgroup_at[cv.cell] == 0
    protos = I.cluster_prototypes(pt, cv.labels_x, pt.X_columns, cv.cell[0])
    assert protos["size"].sum() == len(pt.frame)
    hc = run("hc", n_permutations=5)
    Z = I.hc_linkage(hc)
    node = I.hc_node_of(hc, 0)
    seg = I.dendrogram_segments(Z, len(hc.frame), node)
    assert node in seg["node_xy"] and len(seg["highlight"]) > 0
    rep = I.classifier_report(hc, 0, "X", "decision_tree", n_folds=3)
    assert 0 <= rep.auc_oof <= 1 and "x1" in rep.text or rep.importances is not None
    cca = run("cca", n_permutations=5)
    v = I.cca_view(cca, 0)
    assert len(v["u"]) == len(cca.frame) and len(v["weights"]) > 0
    emm = run("emm", n_permutations=5)
    corr = I.emm_correlations(emm, 0)
    assert {"X", "Y", "z"} <= set(corr.columns) and I.best_pair(emm, 0) is not None
    rules = run("rules", n_search_permutations=5)
    conds = I.rule_conditions(rules, 0)
    assert conds["cumulative"].iloc[-1] == rules[0].coverage
    assert len(I.standardized_profile(rules, 0)) > 0


def test_estimator_objects_in_algorithms(planted):
    from sklearn.naive_bayes import GaussianNB
    df, X, Y, _ = planted
    df = df.dropna()
    mrsd = backend.load_mrsd()
    r = mrsd.run_mrsdhc(df, X, Y, classification_model=["decision_tree", GaussianNB()], n_permutations=0,
                        verbose=False)
    assert len(r) and "auc_GaussianNB()" in r[0].quality_measures
    r = mrsd.run_swapping_method(df, X, Y, classifier_X=GaussianNB(), classifier_Y="decision_tree",
                                 n_permutations=0, n_folds=3, n_initial_subgroups=2, max_iterations=3, verbose=False)
    assert len(r)
    cfg = backend.apply_learners("hc", {"classification_model": ["decision_tree"]}, [object()])
    assert isinstance(cfg["classification_model"][0], backend.OrangeEstimator)


def test_mv_direct_methods(planted):
    df, X, Y, mask = planted
    df = df.dropna()
    mrsd = backend.load_mrsd()
    for method in ("excess", "dual_dendrogram"):
        r = mrsd.run_mrsd_mv(df, X, Y, method=method, n_permutations=0, use_classification_quality=False,
                             random_state=0, verbose=False)
        assert len(r) > 0 and all(sg.quality_measures["method"] == method for sg in r)
        assert all(10 <= sg.coverage <= len(df) // 2 for sg in r)
    from orangecontrib.subgroups.mrsd import mrsd_mv as mv
    import numpy as np
    rng = np.random.default_rng(0)
    W = rng.random((30, 30)); W = (W + W.T) / 2
    B = mv.excess_matrix(W, W, reference="centered")
    assert np.allclose(B, B.T) and np.allclose(np.diag(B), 0)


def test_mv_dendrogram_inspection(planted):
    from orangecontrib.subgroups import inspection as I
    df, X, Y, _ = planted
    df = df.dropna()
    s = params.setting_defaults("mv")
    s["method"] = "dual_dendrogram"
    r = backend.run_algorithm("mv", df, X, Y, params.widget_config("mv", s))
    Z_X, Z_Y, w = I.mv_linkages(r)
    n = len(r.frame)
    qm = r[0].quality_measures
    a, b = I.hc_node_mask(Z_X, n, qm["node_X"]), I.hc_node_mask(Z_Y, n, qm["node_Y"])
    assert set(np.flatnonzero(a & b)) == r[0].covered_instances
    st = I.node_pair_statistics(a, b)
    assert st["observed"] == qm["observed"] and abs(st["residual"] - qm["residual"]) < 1e-6
    node, j = I.best_matching_node(Z_Y, n, a)
    assert node is not None and 0 <= j <= 1
    assert set(w) == set(X)


def test_mixed_mi_estimator():
    backend.load_mrsd()
    from orangecontrib.subgroups.mrsd import mrsd_mi as mi
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=n)
    y = 0.8 * x + 0.6 * rng.normal(size=n)
    c = rng.integers(0, 3, n)
    assert abs(mi.mixed_ksg_mi(x[:, None], y[:, None]) - 0.511) < 0.06            # bivariate Gaussian, r = 0.8
    assert abs(mi.mixed_ksg_mi(x[:, None], rng.normal(size=(n, 1)))) < 0.05          # independent
    assert abs(mi.mixed_ksg_mi(c[:, None] * 1e6, c[:, None] * 1e6) - np.log(3)) < 0.05   # identical categorical
    assert abs(mi.mixed_ksg_mi(c[:, None] * 1e6, (2.0 * c + rng.normal(size=n))[:, None]) - 0.619) < 0.07


def test_common_measures(rules_result):
    from orangecontrib.subgroups import common_measures as cm, consensus, inspection as I
    r = rules_result.copy()
    rng = np.random.default_rng(0)
    r.subgroups.add(I.adhoc_subgroup(r, rng.random(len(r.frame)) < 0.2, "random"))
    cm.compute_all(r)
    for sg in r.subgroups:
        for key in cm.MEASURES:
            v = sg.quality_measures[key]
            assert np.isnan(v) or 0.0 <= v <= 1.0
    assert r[0].quality_measures["common_describe_X"] > 0.9            # a rule is describable in X
    assert r[len(r) - 1].quality_measures["common_distinct_Y"] < 0.3    # a random set is not distinct in Y
    chosen = consensus.clean(r, s_min=0.5, quality="common_best")
    assert chosen and chosen[0] != len(r) - 1
    assert "Y distinctness" in r.summary_frame().columns
