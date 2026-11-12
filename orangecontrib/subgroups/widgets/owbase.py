"""
Base class of the eleven algorithm widgets.

A subclass only declares ``ALGORITHM`` (the id in ``backend.ALGORITHMS``) and
the usual widget metadata; the settings and the parameter GUI are generated
from ``params.PARAMS[ALGORITHM]`` by the metaclass / ``_build_parameters``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHeaderView, QLabel, QScrollArea, QSizePolicy,
    QTableView, QTabWidget, QVBoxLayout, QWidget, QTextBrowser,
)

from Orange.base import Learner, Model
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.subgroups import backend, params
from orangecontrib.subgroups import inspection
from orangecontrib.subgroups.backend import SubgroupResults
from orangecontrib.subgroups.widgets import inspectors as insp_widgets
from orangecontrib.subgroups.widgets.utils import guard_wheel


try:   # link summaries ("12 subgroups") for the Subgroups signal; silences Orange's warning
    from orangewidget.utils.signals import PartialSummary, summarize

    @summarize.register(SubgroupResults)
    def summarize_subgroups(results: SubgroupResults):
        n_sig = sum(1 for sg in results.subgroups if sg.p_value_corrected is not None and sg.p_value_corrected < 0.05)
        return PartialSummary(f"{len(results)} subgroups",
                              f"{results.algorithm_name}: {len(results)} subgroups on {len(results.frame)} rows, "
                              f"{n_sig} significant (adjusted p < 0.05)")
except Exception:     # noqa: BLE001 - older Orange without summarize
    pass


class MRSDWidgetMeta(type(OWWidget)):
    """Adds one ``Setting`` per parameter of ``ALGORITHM`` to the class namespace."""

    def __new__(mcs, name, bases, namespace, **kwargs):
        alg = namespace.get("ALGORITHM")
        algs = namespace.get("ALGORITHMS") or ([alg] if alg else [])
        for a in algs:
            for key, default in params.setting_defaults(a).items():
                namespace.setdefault(key, Setting(default))
        if any(a in OWMRSDBase.AUC_QUALITY for a in algs):   # AUC-type quality: show AUC >= 0.7 by default
            namespace.setdefault("filter_measure", Setting(True))
            namespace.setdefault("measure_threshold", Setting(0.7))
        return super().__new__(mcs, name, bases, namespace, **kwargs)


def _run_task(alg_id: str, frame, X_columns, Y_columns, config, drop_missing, initial_masks, use_process,
              state: TaskState):
    """Runs in a worker thread, which only waits for a separate Python process:
    Stop kills that process, and a long run cannot slow down other widgets."""
    state.set_status(f"Running {backend.ALGORITHM_BY_ID[alg_id].name} …")
    return backend.run_algorithm_in_process(alg_id, frame, X_columns, Y_columns, config, drop_missing=drop_missing,
                                            initial_masks=initial_masks,
                                            should_stop=state.is_interruption_requested,
                                            use_process=use_process)


def _fmt(v, digits=3) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return ""
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.2e}"
        return f"{v:.{digits}f}"
    return str(v)


class OWMRSDBase(OWWidget, ConcurrentWidgetMixin, metaclass=MRSDWidgetMeta, openclass=True):
    """Common machinery: data input, parameter GUI, background run, outputs."""

    name = None                 # subclasses set name, description, icon, priority, ALGORITHM
    ALGORITHM: Optional[str] = None            # the algorithm id (default one when ALGORITHMS is set)
    ALGORITHMS: Optional[List[str]] = None     # several algorithms in one widget (a method selector chooses)

    @property
    def algorithm_id(self) -> str:
        """The algorithm currently chosen (subclasses with several methods override this)."""
        return self.ALGORITHM

    @property
    def hosted_algorithms(self) -> List[str]:
        return list(self.ALGORITHMS) if self.ALGORITHMS else [self.ALGORITHM]
    category = "Subgroups"
    want_main_area = True
    resizing_enabled = True

    drop_missing = Setting(True)
    auto_run = Setting(False)
    use_process = Setting(True)
    use_learners = Setting(True)
    include_builtin = Setting(False)
    learner_x = Setting(0)
    learner_y = Setting(1)
    measure = Setting("quality")
    filter_measure = Setting(False)      # per-algorithm default set in __init__ (AUC-type qualities: on, 0.7)
    measure_threshold = Setting(0.7)
    only_significant = Setting(False)
    filter_alpha = Setting(0.05)
    sort_by = Setting(0)             # index into SORT_KEYS

    AUC_QUALITY = ("hc", "swapping", "semisupervised", "ranking")   # quality is an AUC in [0, 1]
    SORT_KEYS = ["measure (descending)", "adjusted p-value", "raw p-value", "size (descending)",
                 "order of discovery"]

    class Inputs:
        data = Input("Data", Table, default=True)
        subgroups = Input("Subgroups", SubgroupResults)
        learner = Input("Learner", Learner, multiple=True)

    class Outputs:
        subgroups = Output("Subgroups", SubgroupResults)
        data = Output("Data", Table)
        annotated = Output("Annotated Data", Table)
        table = Output("Subgroups Table", Table)
        selected_data = Output("Selected Subgroup Data", Table)
        membership = Output("Membership Data (X)", Table, replaces=["Membership Data"])
        membership_y = Output("Membership Data (Y)", Table)
        model = Output("Model (X)", Model, replaces=["Model"])
        model_y = Output("Model (Y)", Model)

    class Error(OWWidget.Error):
        no_targets = Msg("The data has no dependent (target) variables.\n"
                         "Use the 'Variable Roles' widget (or Select Columns) to set them.")
        no_features = Msg("The data has no independent variables (attributes).")
        no_rows = Msg("No rows remain after removing rows with missing values.")
        invalid_params = Msg("Invalid parameters: {}")
        run_failed = Msg("The algorithm failed: {}")

    class Warning(OWWidget.Warning):
        missing_dropped = Msg("{} row(s) with missing values were removed.")
        missing_kept = Msg("The data contains missing values; the algorithm may fail or misbehave.")
        empty_result = Msg("No subgroups were found.")
        slow_inference = Msg("{} search permutations: the whole search is repeated that many times. "
                             "Set Permutations to 0 (or a small number) for a quick run.")
        waiting_for_subgroups = Msg("No data yet: the link carries Subgroups from a widget that has not "
                                    "finished. Connect its 'Data' output to 'Data' here instead.")

    class Information(OWWidget.Information):
        data_from_subgroups = Msg("Data taken from the connected subgroups ({}).")
        learners_used = Msg("The algorithm will use the connected Orange learner(s): {}.")
        initial_from_subgroups = Msg("{} connected subgroup(s) are used as starting subgroups.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.data: Optional[Table] = None
        self.frame = None
        self.X_columns: List[str] = []
        self.Y_columns: List[str] = []
        self.results: Optional[SubgroupResults] = None
        self._param_controls: Dict[str, Any] = {}
        self._learners: Dict[Any, Learner] = {}
        self._subgroups_link_seen = False
        self.input_data: Optional[Table] = None             # the Data input as such
        self.input_results: Optional[SubgroupResults] = None  # the Subgroups input
        self._inspected: Optional[SubgroupResults] = None     # results shown by the inspectors (may hold an ad-hoc subgroup)

        self._run_buttons, self._stop_buttons = [], []
        self._build_run_box(top=True)
        self._build_header()
        self._build_parameters()
        self._build_results_box()
        self._build_run_box()
        self._build_main_area()
        guard_wheel(self)

    # ------------------------------------------------------------------ GUI
    def _build_header(self):
        box = gui.vBox(self.controlArea, "Data")
        self.info_label = gui.widgetLabel(box, "No data on input.")
        self.info_label.setWordWrap(True)
        gui.checkBox(box, self, "drop_missing", "Remove rows with missing values",
                     tooltip="Rows with a missing X or Y value are left out of the analysis",
                     callback=self._settings_changed)
        self.algorithm_box = gui.vBox(self.controlArea, "Algorithm")
        self._build_method_selector(self.algorithm_box)
        self.description_label = gui.widgetLabel(self.algorithm_box,
                                                 backend.ALGORITHM_BY_ID[self.algorithm_id].description)
        self.description_label.setWordWrap(True)
        if any(a in backend.CLASSIFIER_PARAMS for a in self.hosted_algorithms):
            box = gui.vBox(self.controlArea, "Classifier")
            self.learner_label = gui.widgetLabel(
                box, "No Orange learner connected – the built-in classifier(s) chosen below are used.")
            self.learner_label.setWordWrap(True)
            self.use_learners_check = gui.checkBox(
                box, self, "use_learners", "Use the connected Orange learner(s)", callback=self._learners_changed,
                tooltip="Learners on the Learner input (Naive Bayes, SVM, Logistic Regression, Tree …). "
                        "Permutation inference re-fits the learner many times – keep permutations low.")
            self.include_builtin_check = gui.checkBox(
                box, self, "include_builtin", "Also use the built-in classifiers ticked below",
                callback=self._learners_changed,
                tooltip="Only for MRSD-HC, which can score every node with several classifiers")
            self.learner_x_combo = QComboBox()
            self.learner_y_combo = QComboBox()
            self.learner_rows = []
            for lbl, combo in (("Learner for X", self.learner_x_combo), ("Learner for Y", self.learner_y_combo)):
                row = gui.hBox(box)
                gui.widgetLabel(row, lbl)
                row.layout().addWidget(combo, 1)
                self.learner_rows.append(row)
            self.learner_x_combo.currentIndexChanged.connect(lambda i: setattr(self, "learner_x", max(i, 0)))
            self.learner_y_combo.currentIndexChanged.connect(lambda i: setattr(self, "learner_y", max(i, 0)))
            self.use_learners_check.hide()
            self.include_builtin_check.hide()
            for row in self.learner_rows:
                row.hide()

    def _build_method_selector(self, box):
        """Hook for widgets that host several algorithms (a method chooser)."""

    def _build_parameters(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        container.setLayout(QVBoxLayout())
        container.layout().setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(container)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.controlArea.layout().addWidget(scroll, 1)

        # one control per parameter name; parameters shared by several methods are created once
        self._group_boxes: Dict[str, Any] = {}
        self._group_algs: Dict[str, set] = {}
        self._param_rows: Dict[str, Any] = {}
        self._param_algs: Dict[str, set] = {}
        for alg in self.hosted_algorithms:
            for p in params.PARAMS[alg]:
                if p.group not in self._group_boxes:
                    self._group_boxes[p.group] = gui.vBox(container, p.group)
                    self._group_algs[p.group] = set()
                self._group_algs[p.group].add(alg)
                if p.name in self._param_rows:
                    self._param_algs[p.name].add(alg)
                    continue
                self._param_rows[p.name] = self._add_control(self._group_boxes[p.group], p)
                self._param_algs[p.name] = {alg}
        container.layout().addStretch(1)
        self._update_param_visibility()

    def _update_param_visibility(self):
        """Show only the parameters (and groups) of the current algorithm."""
        if len(self.hosted_algorithms) == 1:
            return
        alg = self.algorithm_id
        for name, row in self._param_rows.items():
            if row is not None:
                row.setVisible(alg in self._param_algs[name])
        for group, box in self._group_boxes.items():
            box.setVisible(alg in self._group_algs[group])

    def _add_control(self, box, p: params.Param):
        tip = p.tip or None
        if isinstance(p, params.OptionalIntParam):
            ctrl = gui.spin(box, self, p.name, p.minimum, p.maximum, step=p.step, label=p.label,
                            checked=f"{p.name}_enabled", tooltip=tip or f"Unchecked: {p.none_label}",
                            callback=self._settings_changed, checkCallback=self._settings_changed)
            ctrl = self._optional_tooltip(ctrl, p)
        elif isinstance(p, params.OptionalFloatParam):
            ctrl = gui.doubleSpin(box, self, p.name, p.minimum, p.maximum, step=p.step, decimals=p.decimals,
                                  label=p.label, checked=f"{p.name}_enabled",
                                  tooltip=tip or f"Unchecked: {p.none_label}",
                                  callback=self._settings_changed, checkCallback=self._settings_changed)
            ctrl = self._optional_tooltip(ctrl, p)
        elif isinstance(p, params.IntParam):
            ctrl = gui.spin(box, self, p.name, p.minimum, p.maximum, step=p.step, label=p.label, tooltip=tip,
                            callback=self._settings_changed)
        elif isinstance(p, params.FloatParam):
            ctrl = gui.doubleSpin(box, self, p.name, p.minimum, p.maximum, step=p.step, decimals=p.decimals,
                                  label=p.label, tooltip=tip, callback=self._settings_changed)
        elif isinstance(p, params.BoolParam):
            ctrl = gui.checkBox(box, self, p.name, p.label, tooltip=tip, callback=self._settings_changed)
        elif isinstance(p, params.ChoiceParam):
            row = gui.hBox(box)
            lbl = gui.widgetLabel(row, p.label)
            if tip:
                lbl.setToolTip(tip)
            ctrl = QComboBox()
            ctrl.addItems(list(p.labels))
            values = list(p.values)
            current = getattr(self, p.name)
            ctrl.setCurrentIndex(values.index(current) if current in values else 0)
            if current not in values:
                setattr(self, p.name, values[0])
            ctrl.currentIndexChanged.connect(
                lambda i, name=p.name, vals=values: (setattr(self, name, vals[i]), self._settings_changed()))
            if tip:
                ctrl.setToolTip(tip)
            row.layout().addWidget(ctrl, 1)
        elif isinstance(p, params.MultiChoiceParam):
            sub = gui.vBox(box, p.label)
            if tip:
                sub.setToolTip(tip)
            selected = set(getattr(self, p.name) or [])
            checks = []
            for value, label in p.choices:
                cb = QCheckBox(label)
                cb.setChecked(value in selected)
                cb.toggled.connect(lambda _, name=p.name, par=p, cs=checks: self._multi_changed(name, par, cs))
                sub.layout().addWidget(cb)
                checks.append((value, cb))
            ctrl = sub
        elif isinstance(p, (params.IntListParam, params.DictParam, params.JSONParam)):
            ctrl = gui.lineEdit(box, self, p.name, label=p.label, orientation=Qt.Horizontal, tooltip=tip,
                                callback=self._settings_changed)
            ctrl.setPlaceholderText(p.placeholder)
        else:   # pragma: no cover - unknown kind
            raise TypeError(f"Unsupported parameter kind {type(p).__name__}")
        self._param_controls[p.name] = ctrl
        if isinstance(p, (params.BoolParam, params.MultiChoiceParam)):
            return ctrl
        return ctrl.parentWidget() if hasattr(ctrl, "parentWidget") else ctrl

    @staticmethod
    def _optional_tooltip(ctrl, p):
        """gui.spin/doubleSpin with `checked` return (spin, checkbox) in newer Orange, the spin alone in older."""
        widgets = ctrl if isinstance(ctrl, tuple) else (ctrl,)
        for w in widgets:
            if hasattr(w, "setToolTip"):
                w.setToolTip(p.tip or f"Unchecked: {p.none_label}")
        return widgets[0]

    def _multi_changed(self, name, par, checks):
        setattr(self, name, [v for v, cb in checks if cb.isChecked()])
        self._settings_changed()

    def _build_results_box(self):
        box = gui.vBox(self.controlArea, "Results")
        row = gui.hBox(box)
        gui.widgetLabel(row, "Measure")
        self.measure_combo = QComboBox()
        self.measure_combo.addItem("quality" + (" (AUC)" if self.algorithm_id in self.AUC_QUALITY else ""), "quality")
        self.measure_combo.currentIndexChanged.connect(self._measure_changed)
        self.measure_combo.setToolTip("The algorithm's quality or any numeric measure it records per subgroup "
                                      "(e.g. AUC per classifier, dispersion, q_X / q_Y)")
        row.layout().addWidget(self.measure_combo, 1)
        gui.doubleSpin(box, self, "measure_threshold", -1e9, 1e9, 0.05, decimals=3, label="Show only measure ≥",
                       checked="filter_measure", callback=self._results_filter_changed,
                       checkCallback=self._results_filter_changed,
                       tooltip="Applies to the table and to the outputs")
        gui.doubleSpin(box, self, "filter_alpha", 0.0001, 0.5, 0.01, decimals=3, label="Show only adjusted p <",
                       checked="only_significant", callback=self._results_filter_changed,
                       checkCallback=self._results_filter_changed,
                       tooltip="Applies to the table and to the outputs; subgroups without a test are kept")
        gui.comboBox(box, self, "sort_by", items=list(self.SORT_KEYS), label="Sort by",
                     orientation=Qt.Horizontal, callback=self._results_filter_changed)
        gui.widgetLabel(box, "Selecting a subgroup (table, dendrogram node or\n"
                             "contingency cell) sends its rows, the data with a\n"
                             "0/1 membership class and – with a connected Orange\n"
                             "learner – the fitted membership model.")

    def _build_run_box(self, top: bool = False):
        """Run / Stop buttons; built twice (top and bottom of the control area)."""
        box = gui.vBox(self.controlArea, True)
        if not top:
            gui.checkBox(box, self, "auto_run", "Run automatically on new data")
            gui.checkBox(box, self, "use_process", "Run in a separate process",
                         tooltip="On: Stop really stops the run and other widgets stay responsive. "
                                 "Off: run inside Orange (as before) – use it if the separate process fails.")
        row = gui.hBox(box)
        run = gui.button(row, self, "Run", callback=self.run, default=top)
        stop = gui.button(row, self, "Stop", callback=self._stop)
        stop.setEnabled(False)
        self._run_buttons.append(run)
        self._stop_buttons.append(stop)

    def _set_running(self, running: bool):
        for b in self._run_buttons:
            b.setEnabled(not running)
        for b in self._stop_buttons:
            b.setEnabled(running)

    def _build_main_area(self):
        self.tabs = QTabWidget()
        self.mainArea.layout().addWidget(self.tabs)
        page = QWidget()
        page.setLayout(QVBoxLayout())
        page.layout().setContentsMargins(0, 0, 0, 0)
        self.tabs.addTab(page, "Subgroups")

        self.summary_box = QTextBrowser()
        self.summary_box.setOpenExternalLinks(False)
        self.summary_box.setMaximumHeight(110)
        self.summary_box.setFrameShape(QTextBrowser.NoFrame)
        page.layout().addWidget(self.summary_box)

        self.result_model = PyTableModel(parent=self)
        self.result_view = QTableView()
        self.result_view.setModel(self.result_model)
        self.result_view.setSortingEnabled(True)
        self.result_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.result_view.horizontalHeader().setStretchLastSection(True)
        self.result_view.verticalHeader().setVisible(False)
        self.result_view.setWordWrap(False)
        self.result_view.setSelectionMode(QAbstractItemView.SingleSelection)
        page.layout().addWidget(self.result_view, 1)
        self.result_view.selectionModel().selectionChanged.connect(self._row_selected)
        self._show_summary("<i>No results yet. Connect data and press <b>Run</b>.</i>")

        # algorithm-specific inspectors of the selected subgroup
        self.inspectors = []
        for cls in insp_widgets.INSPECTORS.get(self.algorithm_id, [insp_widgets.PrototypeInspector]):
            inspector = cls(self)
            inspector.select_requested.connect(self.select_subgroup)
            inspector.adhoc_requested.connect(self._show_adhoc)
            inspector.keep_requested.connect(self._keep_adhoc)
            inspector.model_changed.connect(self._send_models)
            self.tabs.addTab(inspector, inspector.title)
            self.inspectors.append(inspector)
            inspector.set_learners(list(self._learners.values()))

    # ------------------------------------------------------------------ selection / inspectors
    def _selected_index(self) -> Optional[int]:
        rows = self.result_view.selectionModel().selectedRows()
        if not rows or self.results is None:
            return None
        try:
            return int(self.result_model[rows[0].row()][0]) - 1      # column '#' (sorting reorders rows)
        except (IndexError, TypeError, ValueError):
            return None

    def _row_selected(self, *_):
        idx = self._selected_index()
        if idx is None:
            return
        if self._inspected is not self.results:
            self._inspected = self.results
            for inspector in self.inspectors:
                inspector.set_results(self.results)
        for inspector in self.inspectors:
            inspector.set_index(idx)
        self._send_selected(idx)

    def _show_adhoc(self, payload):
        """An inspector chose rows interactively (dendrogram node, contingency cell):
        show them in all inspectors and send them, without touching the results."""
        subgroup, name = payload
        if self.results is None:
            return
        view, idx = inspection.with_adhoc(self.results, subgroup, name)
        self._inspected = view
        self.result_view.clearSelection()
        for inspector in self.inspectors:
            inspector.set_results(view)
            inspector.set_index(idx)
        self._send_selected(idx, view)

    def _keep_adhoc(self, payload):
        """Add an interactively chosen subgroup to the results (no p-value)."""
        subgroup, name = payload
        if self.results is None:
            return
        self.results.subgroups.add(subgroup)
        self.results.names[len(self.results) - 1] = name
        self._fill_results()
        self._send_outputs()
        self.select_subgroup(len(self.results) - 1)

    def _send_models(self, models):
        """Membership models of the selected subgroup: X -> membership and Y -> membership."""
        models = models or {}
        self.Outputs.model.send(models.get("X"))
        self.Outputs.model_y.send(models.get("Y"))
        if models:
            backend.log(f"{self.name}: sent membership model(s): "
                        + ", ".join(f"{sp}: {m.name}" for sp, m in models.items()))

    def _send_selected(self, idx: Optional[int], results: Optional[SubgroupResults] = None):
        r = results if results is not None else self.results
        if r is None or idx is None or r.data is None:
            self.Outputs.selected_data.send(None)
            self.Outputs.membership.send(None)
            self.Outputs.membership_y.send(None)
            return
        self.Outputs.selected_data.send(backend.selected_rows_table(r, [idx]))
        self.Outputs.membership.send(backend.membership_table(r, idx, features="X"))
        self.Outputs.membership_y.send(backend.membership_table(r, idx, features="Y"))

    def measure_value(self, sg) -> Optional[float]:
        if self.measure == "quality":
            v = sg.quality
        else:
            v = (sg.quality_measures or {}).get(self.measure)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(v) else v

    def visible_indices(self) -> List[int]:
        """Indices of the subgroups that pass the filters, in the chosen order."""
        r = self.results
        if r is None:
            return []
        idx = list(range(len(r)))
        if self.filter_measure:
            idx = [i for i in idx if (self.measure_value(r[i]) is not None and
                                      self.measure_value(r[i]) >= self.measure_threshold)]
        if self.only_significant:
            idx = [i for i in idx if r[i].p_value_corrected is None or r[i].p_value_corrected < self.filter_alpha]
        keys = [
            lambda sg: (self.measure_value(sg) is None, -(self.measure_value(sg) or 0.0)),
            lambda sg: (sg.p_value_corrected is None, sg.p_value_corrected or 0.0),
            lambda sg: (sg.p_value is None, sg.p_value or 0.0),
            lambda sg: -sg.coverage,
            lambda sg: 0,
        ]
        key = keys[self.sort_by]
        return sorted(idx, key=lambda i: (key(r[i]), i))

    def _measure_changed(self):
        if self.measure_combo.currentIndex() >= 0:
            self.measure = self.measure_combo.currentData()
        self._results_filter_changed()

    def _refresh_measures(self):
        """Offer every numeric measure the algorithm recorded, keeping the current choice."""
        r = self.results
        keys = []
        if r is not None:
            seen = set()
            for sg in r.subgroups:
                for k, v in (sg.quality_measures or {}).items():
                    if k != "quality" and k not in seen and isinstance(v, (int, float, np.integer, np.floating)) \
                            and not isinstance(v, bool):
                        seen.add(k)
                        keys.append(k)
        self.measure_combo.blockSignals(True)
        self.measure_combo.clear()
        self.measure_combo.addItem("quality" + (" (AUC)" if self.algorithm_id in self.AUC_QUALITY else ""), "quality")
        for k in keys:
            self.measure_combo.addItem(k, k)
        k = self.measure_combo.findData(self.measure)
        if k < 0:
            self.measure = "quality"
            k = 0
        self.measure_combo.setCurrentIndex(k)
        self.measure_combo.blockSignals(False)

    def _results_filter_changed(self):
        if self.results is None:
            return
        self._fill_results()
        self._send_outputs()

    def select_subgroup(self, index: int):
        """Select the subgroup with the given index in the results table (e.g. from an inspector)."""
        for row in range(self.result_model.rowCount()):
            if int(self.result_model[row][0]) - 1 == index:
                self.result_view.selectRow(row)
                return
        # not in the (filtered) table: still show and send it
        self.result_view.clearSelection()
        if self._inspected is not self.results:
            self._inspected = self.results
            for inspector in self.inspectors:
                inspector.set_results(self.results)
        for inspector in self.inspectors:
            inspector.set_index(index)
        self._send_selected(index)

    # ------------------------------------------------------------------ input
    @Inputs.learner
    def set_learner(self, learner: Optional[Learner], key):
        if learner is None:
            self._learners.pop(key, None)
        else:
            self._learners[key] = learner
        if self.algorithm_id not in backend.CLASSIFIER_PARAMS:
            for inspector in getattr(self, "inspectors", []):
                inspector.set_learners(list(self._learners.values()))
        self._learners_changed()

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.input_data = data

    @Inputs.subgroups
    def set_input_subgroups(self, results: Optional[SubgroupResults]):
        self.input_results = results
        self._subgroups_link_seen = True

    def handleNewSignals(self):
        self.Information.data_from_subgroups.clear()
        self.Information.initial_from_subgroups.clear()
        data = self.input_data
        if data is None and self.input_results is not None and self.input_results.data is not None:
            data = self.input_results.data
            self.Information.data_from_subgroups(self.input_results.algorithm_name)
        if self.input_results is not None and len(self.input_results) and self.algorithm_id in backend.ACCEPTS_INITIAL:
            self.Information.initial_from_subgroups(len(self.input_results))
        self.Warning.waiting_for_subgroups.clear()
        if data is None and self.input_data is None and self.input_results is None \
                and self._subgroups_link_seen:
            self.Warning.waiting_for_subgroups()
        self._set_data(data)

    def _initial_masks(self):
        r = self.input_results
        if r is None or not len(r) or self.algorithm_id not in backend.ACCEPTS_INITIAL or self.data is None:
            return None
        if r.n_rows != len(self.data):
            return None
        return [r.mask(i) for i in range(len(r))]

    def _set_data(self, data: Optional[Table]):
        self.cancel()
        self.Error.clear()
        self.Warning.clear()
        self.data = data
        self.frame = None
        self.X_columns, self.Y_columns = [], []
        self.results = None
        if data is None:
            self.info_label.setText("No data on input.")
            self._clear_outputs()
            return
        self.X_columns, self.Y_columns = backend.table_roles(data)
        n_meta = len(data.domain.metas)
        txt = (f"{len(data)} rows<br><b>Independent (X):</b> {len(self.X_columns)}: "
               f"{', '.join(self.X_columns[:8])}{' …' if len(self.X_columns) > 8 else ''}<br>"
               f"<b>Dependent (Y):</b> {len(self.Y_columns)}: {', '.join(self.Y_columns) or '—'}"
               + (f"<br><i>{n_meta} meta attribute(s) are ignored.</i>" if n_meta else ""))
        self.info_label.setText(txt)
        if not self.Y_columns:
            self.Error.no_targets()
            self._clear_outputs()
            return
        if not self.X_columns:
            self.Error.no_features()
            self._clear_outputs()
            return
        self.frame = backend.table_to_dataframe(data)
        self._check_missing()
        self._clear_outputs()
        if self.auto_run:
            self.run()

    def _check_missing(self):
        self.Warning.missing_dropped.clear()
        self.Warning.missing_kept.clear()
        if self.frame is None:
            return
        n_missing = int(self.frame.isna().any(axis=1).sum())
        if n_missing and self.drop_missing:
            self.Warning.missing_dropped(n_missing)
        elif n_missing:
            self.Warning.missing_kept()

    def _settings_changed(self):
        self._check_missing()

    # ------------------------------------------------------------------ running
    def settings_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in params.setting_defaults(self.algorithm_id)}

    def current_config(self) -> Dict[str, Any]:
        cfg = params.widget_config(self.algorithm_id, self.settings_dict())
        if self.use_learners and self._learners:
            builtin = cfg.get("classification_model")
            cfg = backend.apply_learners(self.algorithm_id, cfg, list(self._learners.values()),
                                         self.learner_x, self.learner_y)
            if self.algorithm_id == "hc" and self.include_builtin and builtin:
                cfg["classification_model"] = list(cfg["classification_model"]) + \
                    (list(builtin) if isinstance(builtin, (list, tuple)) else [builtin])
        return cfg

    def _learners_changed(self):
        if not hasattr(self, "use_learners_check"):
            return
        alg = self.algorithm_id
        names = [getattr(l, "name", type(l).__name__) for l in self._learners.values()]
        connected = bool(names) and alg in backend.CLASSIFIER_PARAMS
        self.use_learners_check.setVisible(connected)
        self.include_builtin_check.setVisible(connected and alg == "hc")
        orange_active = connected and self.use_learners
        two_space = orange_active and alg != "hc"
        for combo, setting in ((self.learner_x_combo, "learner_x"), (self.learner_y_combo, "learner_y")):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            k = min(getattr(self, setting), max(len(names) - 1, 0))
            setattr(self, setting, k)
            combo.setCurrentIndex(k)
            combo.blockSignals(False)
        for row in self.learner_rows:
            row.setVisible(two_space)
        if not connected:
            self.learner_label.setText("No Orange learner connected – the built-in classifier(s) chosen below are used.")
        elif orange_active:
            self.learner_label.setText("Connected: " + ", ".join(names) + ". The algorithm uses "
                                       + ("them" if len(names) > 1 else "it") + ".")
        else:
            self.learner_label.setText("Connected: " + ", ".join(names) + " (not used).")
        # the built-in classifier controls are available only when they will be used
        builtin_enabled = (not orange_active) or (alg == "hc" and self.include_builtin)
        for name in backend.CLASSIFIER_PARAMS.get(alg, ()):
            ctrl = self._param_controls.get(name)
            if ctrl is None:
                continue
            target = ctrl.parentWidget() if name in ("classifier_X", "classifier_Y") else ctrl
            target.setVisible(builtin_enabled)
        self.Information.learners_used.clear()
        if orange_active:
            self.Information.learners_used(", ".join(names))
        for inspector in getattr(self, "inspectors", []):
            inspector.set_learners(list(self._learners.values()), builtin_enabled)

    def run(self):
        self.cancel()
        self.Error.invalid_params.clear()
        self.Error.run_failed.clear()
        self.Error.no_rows.clear()
        self.Warning.empty_result.clear()
        if self.frame is None:
            backend.log(f"{self.name}: Run pressed but there is no usable data "
                        f"(data: {'yes' if self.data is not None else 'none'}, X: {len(self.X_columns)}, "
                        f"Y: {len(self.Y_columns)})")
            return
        try:
            config = self.current_config()
            backend.config_class(self.algorithm_id)(**config)      # validate early, in the GUI thread
        except Exception as ex:        # noqa: BLE001 - shown to the user
            self.Error.invalid_params(str(ex))
            return
        if self.drop_missing and self.frame.dropna().empty:
            self.Error.no_rows()
            return
        backend.log(f"{self.name}: Run pressed (method {self.algorithm_id}, separate process: {self.use_process})")
        self.Warning.slow_inference.clear()
        n_perm = int(config.get("n_permutations", 0) or 0)
        if n_perm > 20 and (config.get("inference_statistic") == "search_permutation"
                            or (config.get("inference_mode") == "search" and self.algorithm_id == "ensemble")):
            self.Warning.slow_inference(n_perm)
        self._set_running(True)
        self._show_summary(f"<i>Running {backend.ALGORITHM_BY_ID[self.algorithm_id].name} … "
                           "(permutation inference can take a while)</i>")
        self.start(_run_task, self.algorithm_id, self.frame, self.X_columns, self.Y_columns, config,
                   self.drop_missing, self._initial_masks(), self.use_process)

    def _stop(self):
        self.cancel()
        self._finished_ui()
        self._show_summary("<i>Stopped.</i>")

    def _finished_ui(self):
        self._set_running(False)

    def on_done(self, result: SubgroupResults):
        self._finished_ui()
        backend.log(f"{self.name}: result received, {len(result)} subgroups; filling the table")
        result.data = self.data
        self.results = result
        if len(result) == 0:
            self.Warning.empty_result()
        try:
            self._fill_results()
            self._send_outputs()
            backend.log(f"{self.name}: {len(self.visible_indices())} of {len(result)} subgroups shown and sent")
        except Exception as ex:          # noqa: BLE001 - show instead of failing silently
            import traceback
            backend.log(f"{self.name}: error while showing the results:\n{traceback.format_exc()}")
            self.Error.run_failed(f"showing the results failed: {type(ex).__name__}: {ex}")

    def on_exception(self, ex: Exception):
        self._finished_ui()
        if isinstance(ex, backend.Cancelled):
            return
        import traceback
        backend.log(f"{self.name}: error:\n" + "".join(traceback.format_exception(type(ex), ex, ex.__traceback__)))
        self.Error.run_failed(f"{type(ex).__name__}: {ex}")
        self._show_summary(f"<b>Error:</b> {type(ex).__name__}: {ex}")
        self._clear_outputs()

    def on_partial_result(self, result):
        pass

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    # ------------------------------------------------------------------ outputs
    def _clear_outputs(self):
        self.results = None
        self.result_model.clear()
        for inspector in getattr(self, "inspectors", []):
            inspector.set_results(None)
        self.Outputs.data.send(self.data)
        self.Outputs.subgroups.send(None)
        self.Outputs.annotated.send(None)
        self.Outputs.table.send(None)

    def _send_outputs(self):
        r = self.results
        self.Outputs.data.send(self.data)
        if r is None or len(r) == 0:
            self.Outputs.subgroups.send(r)
            self.Outputs.annotated.send(self.data)
            self.Outputs.table.send(None)
            return
        vis = self.visible_indices()
        out = r.subset(vis) if vis else None
        if out is not None:
            out.names = {k: r.name(i) for k, i in enumerate(vis)}
        self.Outputs.subgroups.send(out)
        self.Outputs.annotated.send(backend.annotate_table(out) if out is not None else self.data)
        self.Outputs.table.send(backend.results_to_table(out) if out is not None else None)

    def _show_summary(self, html: str):
        self.summary_box.setHtml(html)

    def _fill_results(self):
        r = self.results
        if r is None:
            self.result_model.clear()
            return
        sgs = list(r.subgroups)
        tested = [sg for sg in sgs if sg.p_value_corrected is not None]
        n_sig = sum(sg.p_value_corrected < 0.05 for sg in tested)
        inference = next((sg.quality_measures.get("inference") for sg in sgs
                          if sg.quality_measures and sg.quality_measures.get("inference")), None)
        self._refresh_measures()
        vis = self.visible_indices()
        conds = []
        if self.filter_measure:
            conds.append(f"{self.measure} ≥ {self.measure_threshold:g}")
        if self.only_significant:
            conds.append(f"adjusted p &lt; {self.filter_alpha:g}")
        summary = (f"<b>{r.algorithm_name}</b> found <b>{len(sgs)}</b> subgroup(s) in {r.runtime:.1f} s "
                   f"on {len(r.frame)} rows; <b>{len(vis)}</b> shown"
                   + (f" ({', '.join(conds)})" if conds else "") + ". ")
        if tested:
            summary += f"Significant at adjusted p &lt; 0.05: <b>{n_sig}</b> of {len(tested)}. "
        else:
            summary += "No permutation test was run (permutations = 0). "
        if inference:
            summary += f"<br><small>Inference: {inference}</small>"
        self._show_summary(summary)

        effects = r.target_effects()
        extra_measure = self.measure if self.measure != "quality" else None
        header = ["#", "Name", "Rule / description (X)", "Prototype (Y)", "Size", "Quality"] + ([extra_measure] if extra_measure else []) + \
                 ["p", "p adj.", "p BH"] + [f"Δ {c}" for c in effects.columns]
        rows = []
        for i in vis:
            sg = sgs[i]
            qm = sg.quality_measures or {}
            row = [i + 1, r.name(i), r.rule_text(i), r.y_prototype_text(i), sg.coverage, _fmt(sg.quality)] + \
                  ([_fmt(self.measure_value(sg))] if extra_measure else []) + \
                  [_fmt(sg.p_value), _fmt(sg.p_value_corrected), _fmt(qm.get("p_bh"))]
            row += [_fmt(effects.iloc[i][c], 2) for c in effects.columns]
            rows.append(row)
        self.result_model.wrap(rows)
        self.result_model.setHorizontalHeaderLabels(header)
        self.result_view.resizeColumnsToContents()
        self.result_view.horizontalHeader().setStretchLastSection(True)
        self._inspected = r
        for inspector in self.inspectors:
            inspector.set_results(r)
        if rows:
            self.result_view.selectRow(0)
        else:
            self._send_selected(None)
            self._send_models({})
            if len(sgs):
                backend.log(f"{self.name}: no subgroup passes the filters (Results box), so nothing is selected "
                            "and no model / membership data is sent")

    # ------------------------------------------------------------------ report
    def send_report(self):
        self.report_items("Data", [("Rows", len(self.data) if self.data is not None else 0),
                                   ("Independent", ", ".join(self.X_columns)),
                                   ("Dependent", ", ".join(self.Y_columns))])
        try:
            cfg = self.current_config()
        except Exception:      # noqa: BLE001
            cfg = {}
        self.report_items("Parameters", [(k, str(v)) for k, v in cfg.items() if k != "verbose"])
        if self.results is not None:
            self.report_table("Subgroups", self.result_view)
