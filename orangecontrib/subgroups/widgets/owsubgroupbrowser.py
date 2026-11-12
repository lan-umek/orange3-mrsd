"""
Subgroup Browser: inspect, filter, rename, remove and select the subgroups
found by one or several algorithm widgets; outputs the edited set, the
selected subgroups, the covered rows and an annotated data table.
"""
from __future__ import annotations

import math
from html import escape as _esc
from typing import Any, Dict, List, Optional

import numpy as np
from AnyQt.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QAbstractItemView, QHeaderView, QSplitter, QTableView, QTextBrowser

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.subgroups import backend
from orangecontrib.subgroups.backend import SubgroupResults
from orangecontrib.subgroups.widgets.utils import guard_wheel_later


def _f(v, digits=3):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return ""
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.{digits}f}"
    return str(v)


def _num(v):
    if v is None:
        return float("inf")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("inf")
    return float("inf") if math.isnan(f) else f


FIXED_COLUMNS = ["Include", "Name", "Algorithm", "Rule / description (X)", "Prototype (Y)", "Size", "Coverage", "Quality",
                 "p", "p adj.", "p BH", "Rule precision", "Rule recall"]
(COL_INCLUDE, COL_NAME, COL_ALG, COL_RULE, COL_PROTO, COL_SIZE, COL_COV, COL_Q, COL_P, COL_PADJ, COL_PBH, COL_PREC,
 COL_REC) = range(13)
TEXT_COLUMNS = (COL_NAME, COL_ALG, COL_RULE, COL_PROTO)


class SubgroupTableModel(QAbstractTableModel):
    """Rows = subgroups of a SubgroupResults; first column checkable, name editable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: Optional[SubgroupResults] = None
        self.rows: List[Dict[str, Any]] = []
        self.effect_columns: List[str] = []
        self.alpha = 0.05

    def set_results(self, results: Optional[SubgroupResults]):
        self.beginResetModel()
        self.results = results
        self.rows = []
        self.effect_columns = []
        if results is not None and len(results):
            effects = results.target_effects()
            self.effect_columns = list(effects.columns)
            n = len(results.frame)
            for i, sg in enumerate(results.subgroups):
                qm = sg.quality_measures or {}
                self.rows.append({
                    "idx": i, "include": True, "name": results.name(i),
                    "algorithm": backend.ALGORITHM_BY_ID[sg.algorithm].name
                    if sg.algorithm in backend.ALGORITHM_BY_ID else (sg.algorithm or results.algorithm_name),
                    "rule": results.rule_text(i), "proto": results.y_prototype_text(i), "size": int(sg.coverage), "coverage": sg.coverage / max(n, 1),
                    "quality": sg.quality, "p": sg.p_value, "p_adj": sg.p_value_corrected, "p_bh": qm.get("p_bh"),
                    "precision": sg.description_precision, "recall": sg.description_recall,
                    "effects": [effects.iloc[i][c] for c in self.effect_columns],
                })
        self.endResetModel()

    # -- Qt model API -------------------------------------------------------
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(FIXED_COLUMNS) + len(self.effect_columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if section < len(FIXED_COLUMNS):
                return FIXED_COLUMNS[section]
            return "Δ " + self.effect_columns[section - len(FIXED_COLUMNS)]
        return str(section + 1)

    def flags(self, index):
        f = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == COL_INCLUDE:
            f |= Qt.ItemIsUserCheckable
        elif index.column() == COL_NAME:
            f |= Qt.ItemIsEditable
        return f

    def _value(self, row, col):
        r = self.rows[row]
        if col == COL_INCLUDE:
            return r["include"]
        if col == COL_NAME:
            return r["name"]
        if col == COL_ALG:
            return r["algorithm"]
        if col == COL_RULE:
            return r["rule"]
        if col == COL_PROTO:
            return r["proto"]
        if col == COL_SIZE:
            return r["size"]
        if col == COL_COV:
            return r["coverage"]
        if col == COL_Q:
            return r["quality"]
        if col == COL_P:
            return r["p"]
        if col == COL_PADJ:
            return r["p_adj"]
        if col == COL_PBH:
            return r["p_bh"]
        if col == COL_PREC:
            return r["precision"]
        if col == COL_REC:
            return r["recall"]
        return r["effects"][col - len(FIXED_COLUMNS)]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        v = self._value(row, col)
        if col == COL_INCLUDE:
            if role == Qt.CheckStateRole:
                return Qt.Checked if v else Qt.Unchecked
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            if col in TEXT_COLUMNS:
                return v
            if col == COL_SIZE:
                return str(v)
            if col == COL_COV:
                return f"{100 * v:.1f} %"
            if col >= len(FIXED_COLUMNS):
                return _f(v, 2)
            return _f(v)
        if role == Qt.UserRole:                     # sorting
            if col in TEXT_COLUMNS:
                return str(v)
            return _num(v)
        if role == Qt.TextAlignmentRole and col not in TEXT_COLUMNS:
            return Qt.AlignRight | Qt.AlignVCenter
        if role == Qt.ForegroundRole and col == COL_PADJ and v is not None and not math.isnan(v):
            return QColor("#1a7f37") if v < self.alpha else QColor("#8c8c8c")
        if role == Qt.BackgroundRole and col >= len(FIXED_COLUMNS) and v is not None and not math.isnan(v):
            a = min(abs(float(v)) / 1.5, 1.0)
            return QColor(255 - int(80 * a), 255 - int(20 * a), 255 - int(80 * a)) if v > 0 \
                else QColor(255 - int(20 * a), 255 - int(80 * a), 255 - int(80 * a))
        if role == Qt.ToolTipRole and col == COL_RULE:
            sg = self.results.subgroups[self.rows[row]["idx"]]
            return sg.description or v
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False
        row, col = index.row(), index.column()
        if col == COL_INCLUDE and role == Qt.CheckStateRole:
            self.rows[row]["include"] = value == Qt.Checked
        elif col == COL_NAME and role == Qt.EditRole:
            value = str(value).strip()
            if not value:
                return False
            self.rows[row]["name"] = value
        else:
            return False
        self.dataChanged.emit(index, index, [role])
        return True

    # -- editing helpers ----------------------------------------------------
    def remove_rows(self, rows: List[int]):
        for r in sorted(rows, reverse=True):
            self.beginRemoveRows(QModelIndex(), r, r)
            del self.rows[r]
            self.endRemoveRows()

    def set_include(self, rows: List[int], value: bool):
        for r in rows:
            self.rows[r]["include"] = value
            idx = self.index(r, COL_INCLUDE)
            self.dataChanged.emit(idx, idx, [Qt.CheckStateRole])


class SubgroupFilterProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.only_significant = False
        self.alpha = 0.05
        self.min_size = 0
        self.max_size = 0            # 0 = no limit
        self.text = ""
        self.setSortRole(Qt.UserRole)
        self.setDynamicSortFilter(True)

    def filterAcceptsRow(self, source_row, parent):
        model: SubgroupTableModel = self.sourceModel()
        r = model.rows[source_row]
        if self.only_significant:
            p = r["p_adj"]
            if p is None or math.isnan(p) or p >= self.alpha:
                return False
        if r["size"] < self.min_size:
            return False
        if self.max_size and r["size"] > self.max_size:
            return False
        if self.text and self.text.lower() not in (r["rule"] + " " + r["name"]).lower():
            return False
        return True

    def lessThan(self, left, right):
        a, b = left.data(Qt.UserRole), right.data(Qt.UserRole)
        if a is None or b is None:                      # the Include column: by check state
            a, b = left.data(Qt.CheckStateRole) == Qt.Checked, right.data(Qt.CheckStateRole) == Qt.Checked
            return bool(a) < bool(b)
        if isinstance(a, str) or isinstance(b, str):
            return str(a) < str(b)
        return a < b


class OWSubgroupBrowser(OWWidget):
    name = "Subgroup Browser"
    description = "Inspect, filter, rename, remove and select discovered subgroups."
    icon = "icons/browser.svg"
    priority = 300
    keywords = ["subgroup discovery", "MRSD", "results", "browse", "edit", "select", "filter"]

    want_main_area = True
    resizing_enabled = True

    class Inputs:
        subgroups = Input("Subgroups", SubgroupResults, multiple=True)

    class Outputs:
        subgroups = Output("Subgroups", SubgroupResults, default=True)
        selected = Output("Selected Subgroups", SubgroupResults)
        selected_data = Output("Selected Data", Table)
        annotated = Output("Annotated Data", Table)
        table = Output("Subgroups Table", Table)

    class Error(OWWidget.Error):
        incompatible = Msg("The connected results were computed on different data and cannot be shown together.")

    class Information(OWWidget.Information):
        no_data = Msg("The results carry no data table; only the subgroup list is available.")

    only_significant = Setting(False)
    alpha = Setting(0.05)
    min_size = Setting(0)
    max_size = Setting(0)
    annotation_mode = Setting(0)          # 0 = one column per subgroup, 1 = single column
    rule_text = Setting("")
    auto_commit = Setting(True)

    def __init__(self):
        super().__init__()
        guard_wheel_later(self)
        self._inputs: Dict[Any, SubgroupResults] = {}
        self.results: Optional[SubgroupResults] = None
        self.model = SubgroupTableModel(self)
        self.proxy = SubgroupFilterProxy(self)
        self.proxy.setSourceModel(self.model)

        # ---- control area
        box = gui.vBox(self.controlArea, "Filter")
        gui.checkBox(box, self, "only_significant", "Only significant (adjusted p < α)", callback=self._filter_changed)
        gui.doubleSpin(box, self, "alpha", 0.0001, 0.5, 0.01, decimals=3, label="α", callback=self._filter_changed)
        gui.spin(box, self, "min_size", 0, 10_000_000, label="Minimum size", callback=self._filter_changed)
        gui.spin(box, self, "max_size", 0, 10_000_000, label="Maximum size (0 = none)", callback=self._filter_changed)
        self.text_filter = gui.lineEdit(box, self, "rule_text", label="Rule contains", orientation=Qt.Horizontal,
                                        callback=self._filter_changed, callbackOnType=True)

        box = gui.vBox(self.controlArea, "Edit")
        gui.button(box, self, "Remove selected", callback=self._remove_selected,
                   tooltip="Drop the selected subgroups from this list (Restore brings them back)")
        gui.button(box, self, "Keep only selected", callback=self._keep_selected)
        gui.button(box, self, "Include selected", callback=lambda: self._set_include(True))
        gui.button(box, self, "Exclude selected", callback=lambda: self._set_include(False))
        gui.button(box, self, "Restore all", callback=self._restore,
                   tooltip="Reload the subgroups from the input (names and removals are reset)")
        gui.widgetLabel(box, "Double-click a name to rename it;\nuntick 'Include' to leave a subgroup out of the outputs.")

        box = gui.vBox(self.controlArea, "Annotated data")
        gui.radioButtons(box, self, "annotation_mode",
                         ["One Yes/No column per included subgroup", "One column: first covering subgroup"],
                         callback=self._commit_deferred)

        gui.rubber(self.controlArea)
        gui.auto_commit(self.buttonsArea, self, "auto_commit", "Send", box=False)

        # ---- main area
        splitter = QSplitter(Qt.Vertical)
        self.mainArea.layout().addWidget(splitter)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.setWordWrap(False)
        self.view.selectionModel().selectionChanged.connect(self._selection_changed)
        self.model.dataChanged.connect(self._data_changed)
        splitter.addWidget(self.view)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        splitter.addWidget(self.details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.summary_label = gui.widgetLabel(self.mainArea, "")
        self._update_status_line()

    # ------------------------------------------------------------------ input
    @Inputs.subgroups
    def set_subgroups(self, results: Optional[SubgroupResults], key):
        if results is None:
            self._inputs.pop(key, None)
        else:
            self._inputs[key] = results

    def handleNewSignals(self):
        self._rebuild()

    def _rebuild(self):
        self.Error.incompatible.clear()
        self.Information.no_data.clear()
        try:
            self.results = backend.combine_results(list(self._inputs.values()))
        except ValueError:
            self.Error.incompatible()
            self.results = None
        if self.results is not None and self.results.data is None:
            self.Information.no_data()
        self.model.alpha = self.alpha
        self.model.set_results(self.results)
        self.view.resizeColumnsToContents()
        self.view.horizontalHeader().setStretchLastSection(True)
        self._filter_changed()
        self.details.clear()
        self.commit.now()

    # ------------------------------------------------------------------ editing
    def _selected_source_rows(self) -> List[int]:
        return sorted({self.proxy.mapToSource(i).row() for i in self.view.selectionModel().selectedRows()})

    def _remove_selected(self):
        self.model.remove_rows(self._selected_source_rows())
        self._after_edit()

    def _keep_selected(self):
        keep = set(self._selected_source_rows())
        self.model.remove_rows([r for r in range(len(self.model.rows)) if r not in keep])
        self._after_edit()

    def _set_include(self, value):
        self.model.set_include(self._selected_source_rows(), value)
        self._after_edit()

    def _restore(self):
        self.model.set_results(self.results)
        self._after_edit()

    def _after_edit(self):
        self._update_status_line()
        self.commit.deferred()

    def _data_changed(self, *_):
        self._after_edit()

    def _filter_changed(self):
        self.proxy.only_significant = self.only_significant
        self.proxy.alpha = self.alpha
        self.model.alpha = self.alpha
        self.proxy.min_size = self.min_size
        self.proxy.max_size = self.max_size
        self.proxy.text = self.rule_text or ""
        self.proxy.invalidateFilter()
        self.view.viewport().update()
        self._after_edit()

    def _selection_changed(self, *_):
        rows = self._selected_source_rows()
        self._show_details(rows[0] if rows else None)
        self.commit.deferred()

    def _commit_deferred(self):
        self.commit.deferred()

    # ------------------------------------------------------------------ details
    def _show_details(self, row: Optional[int]):
        if row is None or self.results is None:
            self.details.clear()
            return
        r = self.model.rows[row]
        sg = self.results.subgroups[r["idx"]]
        html = [f"<h3>{_esc(str(r['name']))}</h3>",
                f"<b>Rule:</b> {_esc(str(r['rule']))}<br>",
                f"<b>Description:</b> {_esc(str(sg.description))}<br>",
                f"<b>Algorithm:</b> {r['algorithm']} &nbsp; <b>Size:</b> {sg.coverage} "
                f"({100 * r['coverage']:.1f} %)<br>"]
        if sg.p_value is not None:
            html.append(f"<b>p:</b> {_f(sg.p_value)} &nbsp; <b>adjusted p:</b> {_f(sg.p_value_corrected)} "
                        f"&nbsp; <b>correction:</b> {sg.correction_method or ''}<br>")
        if sg.target_statistics:
            html.append("<h4>Dependent variables</h4><table border='0' cellspacing='0' cellpadding='3'>")
            for target, stats in sg.target_statistics.items():
                if "mean_subgroup" in stats:
                    html.append(f"<tr><td><b>{target}</b></td>"
                                f"<td>mean in subgroup {_f(stats['mean_subgroup'])} (SD {_f(stats['std_subgroup'])})</td>"
                                f"<td>mean in all data {_f(stats['mean_all'])} (SD {_f(stats['std_all'])})</td>"
                                f"<td>effect size {_f(stats['effect_size'], 2)}</td></tr>")
                else:
                    levels = [k[2:-len(')_subgroup')] for k in stats if k.endswith("_subgroup")]
                    cells = ", ".join(f"{lv}: {100 * stats[f'p({lv})_subgroup']:.0f} % "
                                      f"(all {100 * stats.get(f'p({lv})_all', float('nan')):.0f} %)" for lv in levels)
                    html.append(f"<tr><td><b>{target}</b></td><td colspan='3'>{cells}</td></tr>")
            html.append("</table>")
        qm = {k: v for k, v in (sg.quality_measures or {}).items()}
        if qm:
            html.append("<h4>Quality measures</h4><table border='0' cellspacing='0' cellpadding='2'>")
            for k, v in qm.items():
                if isinstance(v, (int, float, str, np.integer, np.floating)):
                    html.append(f"<tr><td>{k}</td><td>{_f(v, 4) if isinstance(v, (float, np.floating)) else v}</td></tr>")
            html.append("</table>")
        self.details.setHtml("".join(html))

    def _update_status_line(self):
        if self.results is None:
            self.summary_label.setText("No subgroups on input.")
            return
        n_all = len(self.model.rows)
        n_shown = self.proxy.rowCount()
        n_incl = sum(r["include"] for r in self.model.rows)
        self.summary_label.setText(
            f"{n_all} subgroup(s) from {len(self._inputs)} input(s); {n_shown} shown after filtering; "
            f"{n_incl} included in the outputs.")

    # ------------------------------------------------------------------ output
    def _output_indices(self) -> List[int]:
        """Original indices of the included subgroups that pass the filters, in the view's order."""
        out = []
        for prow in range(self.proxy.rowCount()):
            srow = self.proxy.mapToSource(self.proxy.index(prow, 0)).row()
            r = self.model.rows[srow]
            if r["include"]:
                out.append(srow)
        return out

    def _subset(self, source_rows: List[int], algorithm=None) -> Optional[SubgroupResults]:
        if self.results is None or not source_rows:
            return None
        indices = [self.model.rows[r]["idx"] for r in source_rows]
        sub = self.results.subset(indices)
        sub.names = {k: self.model.rows[r]["name"] for k, r in enumerate(source_rows)}
        return sub

    @gui.deferred
    def commit(self):
        self._update_status_line()
        if self.results is None:
            for out in (self.Outputs.subgroups, self.Outputs.selected, self.Outputs.selected_data,
                        self.Outputs.annotated, self.Outputs.table):
                out.send(None)
            return
        included = self._output_indices()
        out = self._subset(included)
        self.Outputs.subgroups.send(out)
        self.Outputs.table.send(backend.results_to_table(out) if out is not None else None)
        if out is not None and out.data is not None:
            mode = "single" if self.annotation_mode == 1 else "columns"
            self.Outputs.annotated.send(backend.annotate_table(out, mode=mode))
        else:
            self.Outputs.annotated.send(self.results.data)
        sel_rows = self._selected_source_rows()
        sel = self._subset(sel_rows)
        self.Outputs.selected.send(sel)
        self.Outputs.selected_data.send(
            backend.selected_rows_table(sel, list(range(len(sel)))) if sel is not None and sel.data is not None
            else None)

    def send_report(self):
        if self.results is None:
            return
        self.report_items("Subgroups", [("Inputs", len(self._inputs)), ("Shown", self.proxy.rowCount()),
                                        ("Only significant", self.only_significant), ("α", self.alpha)])
        self.report_table("Subgroups", self.view)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    from orangecontrib.subgroups.params import setting_defaults, widget_config
    _data = Table(dataset_path("planted_rule"))
    _cfg = widget_config("rules", {**setting_defaults("rules"), "n_search_permutations": 10})
    _res = backend.run_algorithm("rules", backend.table_to_dataframe(_data), *backend.table_roles(_data), _cfg,
                                 data=_data)
    WidgetPreview(OWSubgroupBrowser).run(set_subgroups=[(_res, 0)])
