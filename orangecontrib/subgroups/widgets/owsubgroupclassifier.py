"""
Subgroup Classifier: describe a subgroup (membership yes / no) with any Orange
classifier – a tree, logistic regression, random forest … – and judge how
well the independent variables X separate the subgroup from the rest.

* AUC / CA are estimated with stratified k-fold cross-validation (the
  classifier never scores the rows it was trained on).  The training-set AUC
  is shown separately and labelled as optimistic.
* The model trained on all rows goes to the *Model* output so that Orange's
  own viewers (Tree Viewer, Nomogram, …) can show it; *Membership Data* is
  the table with the membership class, ready for Test & Score, Rank, etc.;
  *Evaluation Results* feed ROC Analysis / Confusion Matrix.

Caveat: the subgroup itself was found on the same data, so even a
cross-validated AUC describes how *describable* the subgroup is in X, not
whether it would be found again in new data.  Rule-based subgroups are
trivially separable (AUC ≈ 1); the measure is informative for subgroups
defined through Y (clustering-based algorithms).
"""
from __future__ import annotations

from html import escape as _esc

from typing import Any, Dict, List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QAbstractItemView, QHeaderView, QListWidget, QListWidgetItem, QSplitter, QTableView, \
    QTextBrowser

from Orange.base import Learner, Model
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.evaluation import AUC, CA, CrossValidation, Results, TestOnTrainingData
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.subgroups import backend
from orangecontrib.subgroups.backend import SubgroupResults
from orangecontrib.subgroups.widgets.utils import guard_wheel_later


def builtin_learners() -> List[Learner]:
    from Orange.classification import (
        TreeLearner, LogisticRegressionLearner, RandomForestLearner, NaiveBayesLearner, KNNLearner,
    )
    learners = [TreeLearner(max_depth=4), LogisticRegressionLearner(), RandomForestLearner(n_estimators=100),
                NaiveBayesLearner(), KNNLearner()]
    for l, n in zip(learners, ["Tree (depth ≤ 4)", "Logistic Regression", "Random Forest", "Naive Bayes", "kNN"]):
        l.name = n
    return learners


membership_table = backend.membership_table


class OWSubgroupClassifier(OWWidget):
    name = "Subgroup Classifier"
    description = ("Describe a subgroup with an Orange classifier (tree, logistic regression, …) "
                   "and estimate how well X separates it, with cross-validated AUC.")
    icon = "icons/classifier.svg"
    priority = 320
    keywords = ["subgroup discovery", "MRSD", "classifier", "describe", "tree", "logistic regression", "AUC",
                "membership", "model"]

    want_main_area = True
    resizing_enabled = True

    class Inputs:
        subgroups = Input("Subgroups", SubgroupResults)
        learner = Input("Learner", Learner, multiple=True)

    class Outputs:
        model = Output("Model", Model, default=True)
        membership = Output("Membership Data", Table)
        evaluation = Output("Evaluation", Table)
        results = Output("Evaluation Results", Results)

    class Error(OWWidget.Error):
        no_data = Msg("The subgroups carry no data table.")
        failed = Msg("Training / evaluation failed: {}")

    class Warning(OWWidget.Warning):
        tiny = Msg("The subgroup has fewer members than folds; folds were reduced.")

    selected_index = Setting(0, schema_only=True)
    learner_index = Setting(0)
    n_folds = Setting(5)
    random_seed = Setting(0)
    include_y = Setting(False)
    evaluate_all = Setting(False)
    show_train = Setting(True)

    def __init__(self):
        super().__init__()
        guard_wheel_later(self)
        self.results: Optional[SubgroupResults] = None
        self._learners: Dict[Any, Learner] = {}
        self.learners: List[Learner] = []
        self.model_out: Optional[Model] = None
        self.membership_out: Optional[Table] = None
        self.cv_results: Optional[Results] = None
        self.eval_rows: List[List[Any]] = []

        box = gui.vBox(self.controlArea, "Subgroup")
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.currentRowChanged.connect(self._subgroup_changed)
        box.layout().addWidget(self.list)

        box = gui.vBox(self.controlArea, "Classifier")
        self.learner_combo = gui.comboBox(box, self, "learner_index", items=[], callback=self._run,
                                          tooltip="Connect Learner widgets (Tree, Logistic Regression, …) to use "
                                                  "their settings; otherwise the built-in defaults are offered.")
        gui.checkBox(box, self, "include_y", "Also use dependent variables (Y) as features", callback=self._run,
                     tooltip="Normally the subgroup is described in X only")

        box = gui.vBox(self.controlArea, "Evaluation")
        gui.spin(box, self, "n_folds", 2, 20, label="Cross-validation folds", callback=self._run)
        gui.spin(box, self, "random_seed", 0, 10_000, label="Random seed", callback=self._run)
        gui.checkBox(box, self, "show_train", "Also show training-set AUC (optimistic)", callback=self._run)
        gui.checkBox(box, self, "evaluate_all", "Evaluate all subgroups with all classifiers", callback=self._run,
                     tooltip="Fills the table for every subgroup; the model output is still the selected one")
        gui.rubber(self.controlArea)
        gui.button(self.controlArea, self, "Run", callback=self._run, default=True)

        splitter = QSplitter(Qt.Vertical)
        self.mainArea.layout().addWidget(splitter)
        self.table_model = PyTableModel(parent=self)
        self.view = QTableView()
        self.view.setModel(self.table_model)
        self.view.setSortingEnabled(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.verticalHeader().setVisible(False)
        splitter.addWidget(self.view)
        self.summary = QTextBrowser()
        splitter.addWidget(self.summary)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        self._refresh_learners()

    # ------------------------------------------------------------------ inputs
    @Inputs.subgroups
    def set_subgroups(self, results: Optional[SubgroupResults]):
        self.results = results
        self.list.blockSignals(True)
        self.list.clear()
        if results is not None:
            for i in range(len(results)):
                item = QListWidgetItem(f"{results.name(i)}  ({results.rule_text(i)[:50]}, n={results[i].coverage})")
                item.setToolTip(results.rule_text(i))
                self.list.addItem(item)
            self.list.setCurrentRow(min(self.selected_index, len(results) - 1) if len(results) else -1)
        self.list.blockSignals(False)

    @Inputs.learner
    def set_learner(self, learner: Optional[Learner], key):
        if learner is None:
            self._learners.pop(key, None)
        else:
            self._learners[key] = learner

    def handleNewSignals(self):
        self._refresh_learners()
        self._run()

    def _refresh_learners(self):
        self.learners = list(self._learners.values()) or builtin_learners()
        self.learner_combo.blockSignals(True)
        self.learner_combo.clear()
        self.learner_combo.addItems([getattr(l, "name", type(l).__name__) for l in self.learners])
        self.learner_index = max(0, min(self.learner_index, len(self.learners) - 1))
        self.learner_combo.setCurrentIndex(self.learner_index)
        self.learner_combo.blockSignals(False)

    def _subgroup_changed(self, row):
        if row >= 0:
            self.selected_index = row
            self._run()

    # ------------------------------------------------------------------ computation
    def _evaluate(self, data: Table, learners: List[Learner]):
        n_pos = int(np.nansum(data.Y))
        n_neg = len(data) - n_pos
        folds = max(2, min(self.n_folds, n_pos, n_neg))
        if folds < self.n_folds:
            self.Warning.tiny()
        cv = CrossValidation(k=folds, stratified=True, random_state=self.random_seed)(data, learners)
        auc, ca = AUC(cv), CA(cv)
        train_auc = None
        if self.show_train:
            tr = TestOnTrainingData()(data, learners)
            train_auc = AUC(tr)
        return cv, auc, ca, train_auc, folds

    def _run(self):
        self.Error.clear()
        self.Warning.clear()
        self.table_model.clear()
        self.summary.clear()
        self.model_out, self.cv_results, self.eval_rows = None, None, []
        r = self.results
        if r is None or len(r) == 0 or self.list.currentRow() < 0 or not self.learners:
            self._send()
            return
        if r.data is None:
            self.Error.no_data()
            self._send()
            return
        sel = self.list.currentRow()
        learner = self.learners[self.learner_index]
        header = ["Subgroup", "Classifier", "n members", "Folds", "AUC (CV)", "CA (CV)"] + \
                 (["AUC (train)"] if self.show_train else [])
        try:
            if self.evaluate_all:
                self.progressBarInit()
                for i in range(len(r)):
                    data = membership_table(r, i, include_y=self.include_y)
                    cv, auc, ca, tr, folds = self._evaluate(data, self.learners)
                    for k, l in enumerate(self.learners):
                        row = [r.name(i), getattr(l, "name", type(l).__name__), r[i].coverage, folds,
                               f"{auc[k]:.3f}", f"{ca[k]:.3f}"] + ([f"{tr[k]:.3f}"] if tr is not None else [])
                        self.eval_rows.append(row)
                    if i == sel:
                        self.cv_results = cv
                    self.progressBarSet(100 * (i + 1) / len(r))
                self.progressBarFinished()
            else:
                data = membership_table(r, sel, include_y=self.include_y)
                cv, auc, ca, tr, folds = self._evaluate(data, [learner])
                self.cv_results = cv
                self.eval_rows.append([r.name(sel), getattr(learner, "name", type(learner).__name__),
                                       r[sel].coverage, folds, f"{auc[0]:.3f}", f"{ca[0]:.3f}"]
                                      + ([f"{tr[0]:.3f}"] if tr is not None else []))
            data = membership_table(r, sel, include_y=self.include_y)
            self.model_out = learner(data)
            self.model_out.name = f"{getattr(learner, 'name', type(learner).__name__)}: {r.name(sel)}"
            self.membership_out = data
        except Exception as ex:            # noqa: BLE001 - shown to the user
            self.progressBarFinished()
            self.Error.failed(f"{type(ex).__name__}: {ex}")
            self.model_out, self.cv_results, self.membership_out = None, None, None
        self.table_model.wrap(self.eval_rows)
        self.table_model.setHorizontalHeaderLabels(header)
        self.view.resizeColumnsToContents()
        self.view.horizontalHeader().setStretchLastSection(True)
        self._describe_model()
        self._send()

    def _describe_model(self):
        m = self.model_out
        if m is None:
            return
        r = self.results
        sel = self.list.currentRow()
        html = [f"<h3>{m.name}</h3>", f"<b>Rule from the algorithm:</b> {_esc(r.rule_text(sel))}<br>",
                "<small>AUC (CV) is estimated on rows the classifier did not train on; "
                "AUC (train) re-scores the training rows and is optimistic. Both describe how well X "
                "separates this subgroup, not whether it generalises to new data.</small>"]
        try:
            if hasattr(m, "coefficients") and hasattr(m, "intercept"):
                coef = np.asarray(m.coefficients)
                coef = coef[0] if coef.ndim == 2 and coef.shape[0] == 1 else coef[-1] if coef.ndim == 2 else coef
                names = [v.name for v in m.domain.attributes]
                order = np.argsort(-np.abs(coef))
                html.append("<h4>Coefficients (log-odds of membership)</h4><table cellpadding='3'>"
                            f"<tr><td>intercept</td><td align='right'>{float(np.ravel(m.intercept)[-1]):.3f}</td></tr>")
                for j in order:
                    html.append(f"<tr><td>{names[j]}</td><td align='right'>{coef[j]:+.3f}</td></tr>")
                html.append("</table><small>Connect the Model output to Nomogram for a graphical view.</small>")
            elif hasattr(m, "node_count") and hasattr(m, "depth"):
                html.append(f"<p>Tree with {m.node_count()} nodes, depth {m.depth()}. "
                            "Connect the Model output to Tree Viewer.</p>")
            elif hasattr(m, "skl_model") and hasattr(m.skl_model, "feature_importances_"):
                imp = m.skl_model.feature_importances_
                names = [v.name for v in m.domain.attributes]
                html.append("<h4>Feature importances</h4><table cellpadding='3'>")
                for j in np.argsort(-imp)[:15]:
                    html.append(f"<tr><td>{names[j]}</td><td align='right'>{imp[j]:.3f}</td></tr>")
                html.append("</table>")
            else:
                html.append(f"<p>{type(m).__name__}: connect the Model output to a viewer widget "
                            "(Nomogram for Naive Bayes / logistic regression).</p>")
        except Exception as ex:            # noqa: BLE001 - description is best effort
            html.append(f"<p><i>Cannot summarise the model: {ex}</i></p>")
        self.summary.setHtml("".join(html))

    # ------------------------------------------------------------------ outputs
    def _send(self):
        self.Outputs.model.send(self.model_out)
        self.Outputs.membership.send(self.membership_out if self.model_out is not None else None)
        self.Outputs.results.send(self.cv_results)
        if self.eval_rows:
            n_num = 3 + (1 if self.show_train else 0)
            attrs = [ContinuousVariable(n) for n in (["n members", "folds", "AUC (CV)", "CA (CV)"]
                                                     + (["AUC (train)"] if self.show_train else []))]
            metas = [StringVariable("subgroup"), StringVariable("classifier")]
            X = np.array([[float(v) for v in row[2:]] for row in self.eval_rows], dtype=float)
            M = np.array([row[:2] for row in self.eval_rows], dtype=object)
            self.Outputs.evaluation.send(Table.from_numpy(Domain(attrs, metas=metas), X, metas=M))
        else:
            self.Outputs.evaluation.send(None)

    def send_report(self):
        if self.results is None:
            return
        self.report_items("Settings", [("Folds", self.n_folds), ("Seed", self.random_seed),
                                       ("Features", "X and Y" if self.include_y else "X")])
        self.report_table("Evaluation", self.view)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups import backend
    from orangecontrib.subgroups.datasets import dataset_path
    from orangecontrib.subgroups.params import setting_defaults, widget_config
    _data = Table(dataset_path("customers"))
    _cfg = widget_config("pt", {**setting_defaults("pt"), "n_permutations": 50, "k_values": "2, 3"})
    _res = backend.run_algorithm("pt", backend.table_to_dataframe(_data), *backend.table_roles(_data), _cfg,
                                 data=_data)
    WidgetPreview(OWSubgroupClassifier).run(set_subgroups=_res)
