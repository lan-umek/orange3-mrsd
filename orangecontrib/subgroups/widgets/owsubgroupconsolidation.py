"""
Subgroup Consolidation: from many candidate subgroups (usually the union of
several algorithms' outputs) choose a non-redundant set on the basis of their
pairwise similarity – hierarchical clustering of the candidates with one
representative per cluster, or greedy selection in rank order.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QAbstractItemView, QHeaderView, QTableView

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.subgroups import backend, common_measures, consensus
from orangecontrib.subgroups.backend import SubgroupResults
from orangecontrib.subgroups.widgets.utils import guard_wheel_later

CRITERIA_PRESETS = [
    ("support, adjusted p, rank, size", ("support", "p_adj", "rank", "size")),
    ("adjusted p, support, rank, size", ("p_adj", "support", "rank", "size")),
    ("rank within algorithm, support, size", ("rank", "support", "size")),
    ("size, support, adjusted p", ("size", "support", "p_adj")),
]


QUALITY_CHOICES = [
    ("algorithm's quality score", "quality"),
    ("1 − adjusted p-value", "p_adj"),
    ("1 − raw p-value", "p"),
    ("rank within algorithm", "rank"),
    ("size", "size"),
    ("ranking criteria below", "criteria"),
    ("common: Y distinctness", "common_distinct_Y"),
    ("common: relation exceptionality", "common_relation"),
    ("common: better of Y distinctness and relation", "common_best"),
    ("common: X describability", "common_describe_X"),
]


def _f(v, d=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


class OWSubgroupConsolidation(OWWidget):
    name = "Subgroup Consolidation"
    description = ("Clean a set of candidate subgroups (keep the best, drop the similar ones) or cluster them "
                   "on their overlap and keep one representative per cluster.")
    icon = "icons/consolidation.svg"
    priority = 310
    keywords = ["subgroup discovery", "MRSD", "consolidation", "cleaning", "redundancy", "similarity", "jaccard",
                "consensus", "select", "optimal"]

    want_main_area = True
    resizing_enabled = True

    class Inputs:
        subgroups = Input("Subgroups", SubgroupResults, multiple=True)

    class Outputs:
        selected = Output("Selected Subgroups", SubgroupResults, default=True)
        clustered = Output("Clustered Subgroups", SubgroupResults)
        table = Output("Clusters Table", Table)
        similarity = Output("Similarity Matrix", Table)

    class Error(OWWidget.Error):
        incompatible = Msg("The connected results were computed on different data and cannot be combined.")

    method = Setting(0)              # 0 cleaning, 1 clustering
    quality = Setting(8)             # index into QUALITY_CHOICES (default: better of the common measures)
    compute_common = Setting(True)
    measure = Setting(0)
    linkage = Setting(0)
    cut_mode = Setting(0)            # 0 threshold, 1 number of clusters
    threshold = Setting(0.5)
    n_clusters = Setting(5)
    representative = Setting(0)
    criteria = Setting(0)
    min_support = Setting(1)
    max_selected = Setting(0)        # 0 = all
    auto_commit = Setting(True)

    def __init__(self):
        super().__init__()
        guard_wheel_later(self)
        self._inputs: Dict[Any, SubgroupResults] = {}
        self.results: Optional[SubgroupResults] = None
        self.consolidation: Optional[consensus.ConsolidationResult] = None
        self._common_cache: dict = {}
        self.greedy: List[int] = []

        box = gui.vBox(self.controlArea, "Similarity")
        gui.comboBox(box, self, "measure", items=["Jaccard |A∩B| / |A∪B|", "overlap |A∩B| / min(|A|,|B|)",
                                                 "Dice 2|A∩B| / (|A|+|B|)"],
                     label="Measure", orientation=Qt.Horizontal, callback=self._changed)
        box = gui.vBox(self.controlArea, "Method")
        gui.radioButtons(box, self, "method",
                         ["Cleaning: take the best by q, remove all at least s_min similar, repeat",
                          "Clustering with one representative per cluster"], callback=self._changed)
        self.clean_box = gui.vBox(box)
        gui.comboBox(self.clean_box, self, "quality", items=[lbl for lbl, _ in QUALITY_CHOICES],
                     label="Quality measure q", orientation=Qt.Horizontal, callback=self._changed,
                     tooltip="The algorithms' own quality scores are comparable only within one algorithm; "
                             "with candidates from several algorithms prefer the adjusted p-value or the rank.")
        gui.checkBox(box, self, "compute_common", "Compute common measures (comparable across algorithms)",
                     callback=self._inputs_changed,
                     tooltip="Y distinctness and X describability (2·AUC − 1 of membership, out-of-fold, decision "
                             "tree) and relation exceptionality (what a subgroup-specific model of Y from X explains "
                             "of the subgroup's Y variation). All in [0, 1]; computed from the rows of the subgroup, "
                             "so they do not depend on the algorithm that found it.")
        self.cluster_box = gui.vBox(box)
        gui.comboBox(self.cluster_box, self, "linkage", items=["average", "complete", "single"], label="Linkage",
                     orientation=Qt.Horizontal, callback=self._changed)
        gui.radioButtons(self.cluster_box, self, "cut_mode", ["Cut at similarity threshold", "Number of clusters"],
                         callback=self._changed)
        self.n_clusters_spin = gui.spin(self.cluster_box, self, "n_clusters", 1, 1000, label="Number of clusters",
                                        callback=self._changed)
        gui.comboBox(self.cluster_box, self, "representative",
                     items=["best ranked member", "medoid (most similar to the others)", "largest member",
                            "smallest member"], label="Representative", orientation=Qt.Horizontal,
                     callback=self._changed)
        gui.doubleSpin(box, self, "threshold", 0.0, 1.0, 0.05, decimals=2, label="Similarity threshold s_min",
                       callback=self._changed,
                       tooltip="Cleaning: candidates at least this similar to a chosen subgroup are removed.\n"
                               "Clustering: candidates at least this similar form one cluster.")
        box = gui.vBox(self.controlArea, "Ranking")
        gui.comboBox(box, self, "criteria", items=[lbl for lbl, _ in CRITERIA_PRESETS], label="Criteria (priority order)",
                     orientation=Qt.Vertical, callback=self._changed,
                     tooltip="support = number of algorithms in the cluster; rank = rank within its own algorithm "
                             "(quality scales differ between algorithms, ranks do not)")
        gui.spin(box, self, "min_support", 1, 20, label="Minimum supporting algorithms", callback=self._changed)
        gui.spin(box, self, "max_selected", 0, 1000, label="At most subgroups (0 = all)", callback=self._changed)
        gui.rubber(self.controlArea)
        gui.auto_commit(self.buttonsArea, self, "auto_commit", "Send", box=False)

        self.info_label = gui.widgetLabel(self.mainArea, "No subgroups on input.")
        self.model = PyTableModel(parent=self)
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.setSortingEnabled(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.verticalHeader().setVisible(False)
        self.mainArea.layout().addWidget(self.view)
        self._update_enabled()

    # ------------------------------------------------------------------ input
    @Inputs.subgroups
    def set_subgroups(self, results: Optional[SubgroupResults], key):
        if results is None:
            self._inputs.pop(key, None)
        else:
            self._inputs[key] = results

    def handleNewSignals(self):
        self._inputs_changed()

    def _inputs_changed(self):
        self.Error.incompatible.clear()
        try:
            self.results = backend.combine_results(list(self._inputs.values()))
        except ValueError:
            self.Error.incompatible()
            self.results = None
        if self.results is not None and self.compute_common and len(self.results):
            self.progressBarInit()
            try:
                common_measures.compute_all(self.results, cache=self._common_cache, progress=self.progressBarSet)
            except Exception as ex:          # noqa: BLE001 - the widget still works with the other measures
                backend.log(f"{self.name}: common measures failed: {type(ex).__name__}: {ex}")
            finally:
                self.progressBarFinished()
        self._changed()

    def _update_enabled(self):
        self.clean_box.setEnabled(self.method == 0)
        self.cluster_box.setEnabled(self.method == 1)
        self.n_clusters_spin.setEnabled(self.cut_mode == 1)

    def _changed(self):
        self._update_enabled()
        self._compute()
        self.commit.deferred()

    # ------------------------------------------------------------------ computation
    def _compute(self):
        self.consolidation = None
        self.greedy = []
        r = self.results
        if r is None or len(r) == 0:
            self.model.clear()
            self.info_label.setText("No subgroups on input.")
            return
        measure = consensus.SIMILARITIES[self.measure]
        criteria = CRITERIA_PRESETS[self.criteria][1]
        if self.method == 1:
            self.consolidation = consensus.consolidate(
                r, measure=measure, linkage_method=consensus.LINKAGES[self.linkage],
                threshold=None if self.cut_mode == 1 else self.threshold,
                n_clusters=self.n_clusters if self.cut_mode == 1 else None,
                representative=consensus.REPRESENTATIVES[self.representative], criteria=criteria,
                min_support=self.min_support)
            self._fill_cluster_table()
        else:
            self.greedy = consensus.clean(r, measure=measure, s_min=self.threshold,
                                          quality=QUALITY_CHOICES[self.quality][1], criteria=criteria,
                                          max_subgroups=self.max_selected or None)
            self._fill_greedy_table()

    def _selected_indices(self) -> List[int]:
        if self.method == 1:
            if self.consolidation is None:
                return []
            reps = self.consolidation.representatives
        else:
            reps = list(self.greedy)
        if self.max_selected:
            reps = reps[:self.max_selected]
        return reps

    def _fill_cluster_table(self):
        r, c = self.results, self.consolidation
        header = ["Cluster", "Representative", "Rule / description (X)", "Prototype (Y)", "Algorithm", "Support", "Supporting algorithms",
                  "Members", "Mean similarity", "Size", "Y distinct.", "X describ.", "Relation", "p adj.", "Kept"]
        rows = []
        n_alg = len({sg.algorithm or r.algorithm for sg in r.subgroups})
        for k, cl in enumerate(c.clusters + c.dropped):
            sg = r[cl.representative]
            rows.append([k + 1, r.name(cl.representative), r.rule_text(cl.representative),
                         r.y_prototype_text(cl.representative),
                         backend.ALGORITHM_BY_ID[sg.algorithm].name if sg.algorithm in backend.ALGORITHM_BY_ID
                         else (sg.algorithm or ""), cl.support, ", ".join(cl.algorithms), len(cl.members),
                         _f(cl.mean_similarity, 2), sg.coverage, *self._common(sg), _f(cl.best_p_adj),
                         "yes" if cl in c.clusters else "no"])
        self.model.wrap(rows)
        self.model.setHorizontalHeaderLabels(header)
        self.view.resizeColumnsToContents()
        self.view.horizontalHeader().setStretchLastSection(True)
        self.info_label.setText(
            f"{len(r)} candidates from {len(self._inputs)} input(s) and {n_alg} algorithm(s) → "
            f"{len(c.clusters) + len(c.dropped)} clusters; {len(c.clusters)} kept with support ≥ {self.min_support}"
            + (f", {min(len(c.clusters), self.max_selected)} sent" if self.max_selected else "") + ".")

    @staticmethod
    def _common(sg):
        qm = sg.quality_measures or {}
        return [_f(qm.get(k), 2) for k in ("common_distinct_Y", "common_describe_X", "common_relation")]

    def _fill_greedy_table(self):
        r = self.results
        S = consensus.similarity_matrix(r, consensus.SIMILARITIES[self.measure])
        header = ["Order", "Name", "Rule / description (X)", "Prototype (Y)", "Algorithm", "Size", "Quality", "Y distinct.", "X describ.",
                  "Relation", "p adj.", "Max similarity to kept"]
        rows = []
        for k, i in enumerate(self.greedy):
            sg = r[i]
            prev = self.greedy[:k]
            rows.append([k + 1, r.name(i), r.rule_text(i), r.y_prototype_text(i),
                         backend.ALGORITHM_BY_ID[sg.algorithm].name if sg.algorithm in backend.ALGORITHM_BY_ID
                         else (sg.algorithm or ""), sg.coverage, _f(sg.quality), *self._common(sg),
                         _f(sg.p_value_corrected),
                         _f(float(S[i, prev].max()), 2) if prev else ""])
        self.model.wrap(rows)
        self.model.setHorizontalHeaderLabels(header)
        self.view.resizeColumnsToContents()
        self.view.horizontalHeader().setStretchLastSection(True)
        self.info_label.setText(f"{len(r)} candidates → {len(self.greedy)} kept by cleaning "
                                f"(q = {QUALITY_CHOICES[self.quality][0]}, s_min = {self.threshold:.2f}).")

    # ------------------------------------------------------------------ output
    @gui.deferred
    def commit(self):
        r = self.results
        if r is None or len(r) == 0:
            for out in (self.Outputs.selected, self.Outputs.clustered, self.Outputs.table, self.Outputs.similarity):
                out.send(None)
            return
        sel = self._selected_indices()
        selected = r.subset(sel) if sel else None
        if selected is not None and self.method == 1:
            for k, cl in enumerate(self.consolidation.clusters[:len(sel)]):
                selected.names[k] = f"C{k + 1} ({cl.support} alg.): {r.name(cl.representative)}"
        self.Outputs.selected.send(selected)

        if self.method == 1 and self.consolidation is not None:
            c = self.consolidation
            order = [i for cl in c.clusters + c.dropped for i in cl.members]
            clustered = r.subset(order)
            pos = 0
            for k, cl in enumerate(c.clusters + c.dropped):
                for i in cl.members:
                    clustered.names[pos] = f"C{k + 1}: {r.name(i)}"
                    pos += 1
            self.Outputs.clustered.send(clustered)
            self.Outputs.table.send(self._cluster_table())
        else:
            self.Outputs.clustered.send(None)
            self.Outputs.table.send(None)

        S = consensus.similarity_matrix(r, consensus.SIMILARITIES[self.measure])
        names = [r.name(i) for i in range(len(r))]
        dom = Domain([ContinuousVariable(n) for n in names], metas=[StringVariable("subgroup")])
        self.Outputs.similarity.send(Table.from_numpy(dom, S, metas=np.array(names, dtype=object).reshape(-1, 1)))

    def _cluster_table(self):
        r, c = self.results, self.consolidation
        attrs = [ContinuousVariable(n) for n in ("cluster", "support", "members", "mean similarity", "size",
                                                 "p adjusted", "kept")]
        metas = [StringVariable("representative"), StringVariable("rule"), StringVariable("algorithms")]
        X, M = [], []
        for k, cl in enumerate(c.clusters + c.dropped):
            sg = r[cl.representative]
            X.append([k + 1, cl.support, len(cl.members), cl.mean_similarity, sg.coverage,
                      cl.best_p_adj if cl.best_p_adj is not None else np.nan, float(cl in c.clusters)])
            M.append([r.name(cl.representative), r.rule_text(cl.representative), ", ".join(cl.algorithms)])
        return Table.from_numpy(Domain(attrs, metas=metas), np.array(X, dtype=float),
                                metas=np.array(M, dtype=object))

    def send_report(self):
        self.report_items("Settings", [("Method", ["cleaning", "clustering"][self.method]),
                                       ("Quality q", QUALITY_CHOICES[self.quality][0]),
                                       ("Similarity", consensus.SIMILARITIES[self.measure]),
                                       ("Threshold", self.threshold), ("Min. support", self.min_support)])
        self.report_table("Clusters", self.view)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    from orangecontrib.subgroups.params import setting_defaults, widget_config
    _data = Table(dataset_path("customers"))
    _res = []
    for _alg, _over in [("rules", {"n_search_permutations": 10}), ("emm", {"n_permutations": 20})]:
        _cfg = widget_config(_alg, {**setting_defaults(_alg), **_over})
        _res.append(backend.run_algorithm(_alg, backend.table_to_dataframe(_data), *backend.table_roles(_data), _cfg,
                                          data=_data))
    WidgetPreview(OWSubgroupConsolidation).run(set_subgroups=[(_res[0], 0), (_res[1], 1)])
