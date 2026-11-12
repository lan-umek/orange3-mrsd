"""
Inspector tabs of the algorithm widgets: algorithm-specific views of the
subgroup selected in the results table (see ``inspection.py`` for the
computations).

Every inspector is a QWidget with ``set_results(results)`` and
``set_index(index)``; ``select_requested`` asks the hosting widget to select
another subgroup (e.g. when a cell of a contingency table is clicked).
"""
from __future__ import annotations

from html import escape as _esc

import math
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import pyqtgraph as pg
from AnyQt.QtCore import QRectF, Qt, Signal
from AnyQt.QtGui import QBrush, QColor, QCursor, QFont
from AnyQt.QtWidgets import (
    QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSpinBox,
    QSplitter, QTableView, QTableWidget, QTableWidgetItem, QTextBrowser, QToolTip, QVBoxLayout, QWidget,
)

from Orange.widgets.utils.itemmodels import PyTableModel

from orangecontrib.subgroups import backend, inspection as insp, params
from orangecontrib.subgroups.backend import SubgroupResults

IN_COLOR = QColor("#d95f02")
OUT_COLOR = QColor("#7570b3")


def _fmt(v, digits=3):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return ""
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.{digits}f}"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return str(v)


def make_table() -> QTableView:
    view = QTableView()
    view.setModel(PyTableModel(parent=view))
    view.setSelectionBehavior(QAbstractItemView.SelectRows)
    view.setEditTriggers(QAbstractItemView.NoEditTriggers)
    view.setSortingEnabled(True)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    view.horizontalHeader().setStretchLastSection(True)
    view.verticalHeader().setVisible(False)
    view.setWordWrap(False)
    return view


def fill_table(view: QTableView, df: Optional[pd.DataFrame], digits=3, bold_rows: Optional[List[int]] = None):
    model: PyTableModel = view.model()
    if df is None or len(df) == 0:
        model.clear()
        return
    rows = [[_fmt(v, digits) for v in row] for row in df.itertuples(index=False)]
    model.wrap(rows)
    model.setHorizontalHeaderLabels([str(c) for c in df.columns])
    if bold_rows:
        font = QFont()
        font.setBold(True)
        for r in bold_rows:
            for c in range(len(df.columns)):
                model.setData(model.index(r, c), font, Qt.FontRole)
    view.resizeColumnsToContents()
    view.horizontalHeader().setStretchLastSection(True)


def _diverging(v: float, scale: float = 1.0) -> QColor:
    if v is None or math.isnan(v):
        return QColor(255, 255, 255)
    a = min(abs(v) / scale, 1.0)
    return QColor(255 - int(90 * a), 255 - int(20 * a), 255 - int(90 * a)) if v > 0 \
        else QColor(255 - int(20 * a), 255 - int(90 * a), 255 - int(90 * a))


class Inspector(QWidget):
    title = "Inspector"
    select_requested = Signal(int)          # select a reported subgroup (index)
    adhoc_requested = Signal(object)        # (Subgroup, name): show / send an interactively chosen subgroup
    keep_requested = Signal(object)         # (Subgroup, name): add it to the results permanently
    model_changed = Signal(object)          # {space: Orange model} for the Model (X) / Model (Y) outputs

    def set_learners(self, learners, builtin_allowed: bool = True):
        """Orange learners connected to the widget (used by the classifier inspector);
        `builtin_allowed` says whether the built-in MRSD classifiers may be offered too."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: Optional[SubgroupResults] = None
        self.index: Optional[int] = None
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(4, 4, 4, 4)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.RichText)
        self.layout().addWidget(self.note)

    def set_results(self, results: Optional[SubgroupResults]):
        self.results = results
        self.index = None
        self.refresh()

    def set_index(self, index: Optional[int]):
        self.index = index
        self.refresh()

    def ready(self) -> bool:
        return self.results is not None and self.index is not None and 0 <= self.index < len(self.results)

    def refresh(self):
        if not self.ready():
            self.note.setText("<i>Select a subgroup in the Subgroups tab.</i>")
            self.clear()
            return
        try:
            self.show_subgroup()
        except Exception as ex:          # noqa: BLE001 - an inspector must never kill the widget
            import traceback
            backend.log(f"{self.title} view failed:\n{traceback.format_exc()}")
            self.clear()
            self.note.setText(f"<b>Cannot show this view:</b> {type(ex).__name__}: {ex}")

    def clear(self):
        pass

    def show_subgroup(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Prototype (all algorithms)
# ---------------------------------------------------------------------------

class RotatedAxis(pg.AxisItem):
    """Bottom axis with vertical tick labels (for many / long category names)."""

    def __init__(self, orientation="bottom", **kwargs):
        super().__init__(orientation, **kwargs)
        self.setStyle(textFillLimits=[(0, 10.0)])     # never drop major-tick labels
        self._labels: List[str] = []

    def set_labels(self, labels: List[str]):
        self._labels = list(labels)
        self.setTicks([[(k, lab) for k, lab in enumerate(labels)]])
        longest = max((len(lab) for lab in labels), default=0)
        self.setHeight(min(int(6.5 * longest) + 24, 260))

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        super().drawPicture(p, axisSpec, tickSpecs, [])
        p.setPen(self.textPen() if hasattr(self, "textPen") else pg.mkPen("k"))
        for rect, flags, text in textSpecs:
            p.save()
            p.translate(rect.center().x(), rect.top())
            p.rotate(-90)
            p.drawText(QRectF(-250, -8, 250, 16), Qt.AlignRight | Qt.AlignVCenter, text)
            p.restore()


class PrototypeInspector(Inspector):
    title = "Prototype"
    DISPLAYS = [
        ("Standardized difference (subgroup − complement) / SD", "standardized"),
        ("Difference of means / shares (subgroup − complement)", "difference"),
        ("Means / shares: subgroup and complement", "pair"),
        ("Means / shares: subgroup, complement and all data", "triple"),
    ]
    positive_color = QColor("#1b9e77")
    negative_color = QColor("#d95f02")
    complement_color = QColor("#7570b3")
    all_color = QColor("#999999")

    def __init__(self, parent=None):
        super().__init__(parent)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Show:"))
        self.display_combo = QComboBox()
        for lbl, _ in self.DISPLAYS:
            self.display_combo.addItem(lbl)
        self.display_combo.currentIndexChanged.connect(self._replot)
        controls.addWidget(self.display_combo, 1)
        self.space_combo = QComboBox()
        self.space_combo.addItems(["X and Y", "X only", "Y only"])
        self.space_combo.currentIndexChanged.connect(self._replot)
        controls.addWidget(self.space_combo)
        self.values_check = QCheckBox("Values on bars")
        self.values_check.toggled.connect(self._replot)
        controls.addWidget(self.values_check)
        self.pos_button = QPushButton("Positive colour")
        self.pos_button.clicked.connect(lambda: self._pick_color("positive_color"))
        self.neg_button = QPushButton("Negative colour")
        self.neg_button.clicked.connect(lambda: self._pick_color("negative_color"))
        controls.addWidget(self.pos_button)
        controls.addWidget(self.neg_button)
        self.layout().addLayout(controls)

        split = QSplitter(Qt.Vertical)
        self.table = make_table()
        split.addWidget(self.table)
        self.axis = RotatedAxis("bottom")
        self.plot = pg.PlotWidget(background="w", axisItems={"bottom": self.axis})
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=False, y=False)
        self.plot.addLegend(offset=(10, 10))
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        split.addWidget(self.plot)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 3)
        self.layout().addWidget(split, 1)
        self._bars: List[Tuple[float, float, float, str]] = []     # (x, height, width, tooltip)
        self._profile: Optional[pd.DataFrame] = None
        self._update_color_buttons()

    # -- colours -----------------------------------------------------------
    def _pick_color(self, attr):
        color = QColorDialog.getColor(getattr(self, attr), self, "Choose a colour")
        if color.isValid():
            setattr(self, attr, color)
            self._update_color_buttons()
            self._replot()

    def _update_color_buttons(self):
        for btn, col in ((self.pos_button, self.positive_color), (self.neg_button, self.negative_color)):
            btn.setStyleSheet(f"background-color: {col.name()}; color: white; padding: 2px 8px;")

    # -- content -------------------------------------------------------------
    def clear(self):
        fill_table(self.table, None)
        self.plot.clear()
        self._bars = []
        self._profile = None

    def show_subgroup(self):
        r, i = self.results, self.index
        df = insp.prototype_table(r, i)
        fill_table(self.table, df, digits=3)
        model = self.table.model()
        for colname, scale in (("effect size", 1.0), ("difference", None)):
            if colname in df.columns:
                col = list(df.columns).index(colname)
                vals = pd.to_numeric(df[colname], errors="coerce").to_numpy(dtype=float)
                sc = scale or max(float(np.nanmax(np.abs(vals))), 1e-9) if len(vals) else 1.0
                for row, v in enumerate(vals):
                    model.setData(model.index(row, col), QBrush(_diverging(float(v), sc)), Qt.BackgroundRole)
        for colname in ("p", "p (np)", "p adj.", "p adj. (np)"):
            if colname in df.columns:
                col = list(df.columns).index(colname)
                for row, v in enumerate(pd.to_numeric(df[colname], errors="coerce")):
                    if not np.isnan(v) and v < 0.05:
                        model.setData(model.index(row, col), QBrush(QColor("#1a7f37")), Qt.ForegroundRole)
        med = insp.medoid_row(r, i)
        self.note.setText(
            f"<b>{r.name(i)}</b>: {_esc(r.rule_text(i))} &nbsp;(n = {r[i].coverage} vs. {len(r.frame) - r[i].coverage} in the "
            "complement). Numeric variables: mean, SD, median, IQR, difference of means, Cohen's d (pooled SD), "
            "Welch's t-test and Mann–Whitney U with Cliff's δ. Categorical: modal level, distribution of levels, "
            "largest difference of a share, Cramér's V, χ² test and Fisher's exact test (2 levels) or a Monte-Carlo "
            "χ² test. 'p adj.' = Benjamini–Hochberg over the variables of the same test. The p-values are "
            "descriptive – the subgroup was selected on these same rows." +
            (f" Medoid: row {med} of the analysed data." if med is not None else ""))
        self._profile = insp.profile_values(r, i)
        self._replot()

    def _replot(self):
        self.plot.clear()
        self.plot.plotItem.legend.clear()
        self._bars = []
        prof = self._profile
        if prof is None or not self.ready():
            return
        space = self.space_combo.currentIndex()
        if space == 1:
            prof = prof[prof["space"] == "X"]
        elif space == 2:
            prof = prof[prof["space"] == "Y"]
        prof = prof.reset_index(drop=True)
        if prof.empty:
            return
        mode = self.DISPLAYS[self.display_combo.currentIndex()][1]
        names = [f"{n} ({sp})" if space == 0 else n for n, sp in zip(prof["name"], prof["space"])]
        xs = np.arange(len(prof), dtype=float)
        r, i = self.results, self.index
        show_vals = self.values_check.isChecked()

        def add_bars(x, heights, width, brush_for, label, tip):
            heights = np.asarray(heights, dtype=float)
            ok = ~np.isnan(heights)
            if not ok.any():
                return
            brushes = [brush_for(h) for h in heights[ok]]
            item = pg.BarGraphItem(x=x[ok], height=heights[ok], width=width, brushes=brushes,
                                   pen=pg.mkPen("k", width=0.5))
            self.plot.addItem(item)
            if label:
                self.plot.plotItem.legend.addItem(item, label)
            for xx, h, nm in zip(x[ok], heights[ok], np.asarray(names)[ok]):
                self._bars.append((float(xx), float(h), width, f"{nm}\n{tip}: {h:.3g}"))
                if show_vals:
                    t = pg.TextItem(f"{h:.2f}", color="#333", anchor=(0.5, 1.0 if h >= 0 else 0.0))
                    t.setPos(xx, h)
                    self.plot.addItem(t)

        sign_brush = lambda h: pg.mkBrush(self.positive_color if h >= 0 else self.negative_color)
        if mode == "standardized":
            add_bars(xs, prof["standardized"], 0.7, sign_brush, None, "standardized difference")
            self.plot.setLabel("left", "standardized difference (subgroup − complement)")
        elif mode == "difference":
            add_bars(xs, prof["difference"], 0.7, sign_brush, None, "difference")
            self.plot.setLabel("left", "difference of means / shares (subgroup − complement)")
        elif mode == "pair":
            w = 0.38
            add_bars(xs - w / 2, prof["subgroup"], w, lambda h: pg.mkBrush(self.positive_color), r.name(i), "subgroup")
            add_bars(xs + w / 2, prof["complement"], w, lambda h: pg.mkBrush(self.complement_color), "complement",
                     "complement")
            self.plot.setLabel("left", "mean (numeric) / share (categorical level)")
        else:
            w = 0.27
            add_bars(xs - w, prof["subgroup"], w, lambda h: pg.mkBrush(self.positive_color), r.name(i), "subgroup")
            add_bars(xs, prof["complement"], w, lambda h: pg.mkBrush(self.complement_color), "complement", "complement")
            add_bars(xs + w, prof["all"], w, lambda h: pg.mkBrush(self.all_color), "all data", "all data")
            self.plot.setLabel("left", "mean (numeric) / share (categorical level)")
        self.plot.addLine(y=0, pen=pg.mkPen("#444"))
        self.axis.set_labels(names)
        self.plot.setXRange(-0.7, len(prof) - 0.3, padding=0)
        self.plot.enableAutoRange(axis="y")
        self.plot.autoRange(padding=0.08)

    def _mouse_moved(self, pos):
        if not self._bars:
            return
        vb = self.plot.plotItem.vb
        if not self.plot.sceneBoundingRect().contains(pos):
            QToolTip.hideText()
            return
        pt = vb.mapSceneToView(pos)
        x, y = pt.x(), pt.y()
        for bx, h, w, tip in self._bars:
            lo, hi = min(0.0, h), max(0.0, h)
            if abs(x - bx) <= w / 2 and lo - 1e-9 <= y <= hi + 1e-9:
                QToolTip.showText(QCursor.pos(), tip, self.plot)
                return
        QToolTip.hideText()


# ---------------------------------------------------------------------------
# Quality measures (all algorithms)
# ---------------------------------------------------------------------------

class MeasuresInspector(Inspector):
    title = "Measures"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = make_table()
        self.layout().addWidget(self.table)

    def clear(self):
        fill_table(self.table, None)

    def show_subgroup(self):
        r, i = self.results, self.index
        sg = r[i]
        df = insp.quality_measures_table(r, i)
        extra = pd.DataFrame([{"measure": k, "value": v} for k, v in (
            ("size", sg.coverage), ("p_value", sg.p_value), ("p_value_corrected", sg.p_value_corrected),
            ("correction", sg.correction_method), ("description_precision", sg.description_precision),
            ("description_recall", sg.description_recall), ("description", sg.description)) if v is not None])
        fill_table(self.table, pd.concat([extra, df], ignore_index=True), digits=4)
        self.note.setText(f"<b>{r.name(i)}</b>: {_esc(r.rule_text(i))}")


# ---------------------------------------------------------------------------
# MRSD-PT: contingency table and partitions
# ---------------------------------------------------------------------------

class ContingencyInspector(Inspector):
    title = "Contingency table"

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout()
        row.addWidget(QLabel("Partition pair (X clustering, k × Y clustering, l):"))
        self.pair_combo = QComboBox()
        self.pair_combo.currentIndexChanged.connect(self._pair_changed)
        row.addWidget(self.pair_combo, 1)
        self.keep_button = QPushButton("Keep this cell as a subgroup")
        self.keep_button.setEnabled(False)
        self.keep_button.clicked.connect(self._keep)
        row.addWidget(self.keep_button)
        self.layout().addLayout(row)
        split = QSplitter(Qt.Vertical)
        self.grid = QTableWidget()
        self.grid.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.grid.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grid.cellClicked.connect(self._cell_clicked)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.grid.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        split.addWidget(self.grid)
        lower = QWidget()
        lower.setLayout(QHBoxLayout())
        lower.layout().setContentsMargins(0, 0, 0, 0)
        self.x_table = make_table()
        self.y_table = make_table()
        for lbl, tbl in (("Clusters of X (row prototypes)", self.x_table), ("Clusters of Y (column prototypes)",
                                                                              self.y_table)):
            box = QWidget()
            box.setLayout(QVBoxLayout())
            box.layout().setContentsMargins(0, 0, 0, 0)
            box.layout().addWidget(QLabel(f"<b>{lbl}</b>"))
            box.layout().addWidget(tbl)
            lower.layout().addWidget(box)
        split.addWidget(lower)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 2)
        self.layout().addWidget(split)
        self._keys = []
        self._updating = False

    def set_results(self, results):
        self._updating = True
        self.pair_combo.clear()
        self._keys = insp.pt_partition_keys(results) if results is not None else []
        self.pair_combo.addItems([f"{cx}, k={k}  ×  {cy}, l={l}" for cx, k, cy, l in self._keys])
        self._updating = False
        super().set_results(results)

    def clear(self):
        self.grid.clear()
        self.grid.setRowCount(0)
        self.grid.setColumnCount(0)
        fill_table(self.x_table, None)
        fill_table(self.y_table, None)

    def _pair_changed(self):
        if not self._updating and self.results is not None:
            self._show(self.pair_combo.currentIndex())

    def show_subgroup(self):
        r, i = self.results, self.index
        sg = r[i]
        key = (sg.clustering_X, int(sg.k), sg.clustering_Y, int(sg.l)) if sg.partition_result_X is not None else None
        if key in self._keys and self.pair_combo.currentIndex() != self._keys.index(key):
            self._updating = True
            self.pair_combo.setCurrentIndex(self._keys.index(key))
            self._updating = False
        self._show(self.pair_combo.currentIndex())

    def _show(self, pair_index: int):
        r = self.results
        if pair_index < 0 or pair_index >= len(self._keys):
            self.clear()
            return
        key = self._keys[pair_index]
        cv = insp.pt_contingency(r, self.index, key)
        if cv is None:
            self.clear()
            return
        k, l = cv.observed.shape
        self.grid.clear()
        self.grid.setRowCount(k)
        self.grid.setColumnCount(l)
        self.grid.setHorizontalHeaderLabels([f"Y cluster {j}" for j in range(l)])
        self.grid.setVerticalHeaderLabels([f"X cluster {i}" for i in range(k)])
        scale = max(3.0, float(np.abs(cv.residuals).max()))
        for i in range(k):
            for j in range(l):
                idx = cv.subgroup_at.get((i, j))
                txt = f"{int(cv.observed[i, j])}\nexp. {cv.expected[i, j]:.1f}\nresid. {cv.residuals[i, j]:+.2f}"
                if idx is not None:
                    sg = r[idx]
                    txt += f"\n{r.name(idx)}" + (f", p adj. {sg.p_value_corrected:.3f}" if sg.p_value_corrected is not None else "")
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QBrush(_diverging(float(cv.residuals[i, j]), scale)))
                if cv.cell == (i, j):
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                    item.setForeground(QBrush(QColor("#b30000")))
                elif idx is None:
                    item.setForeground(QBrush(QColor("#777")))
                self.grid.setItem(i, j, item)
        if cv.cell is not None:
            self.grid.setCurrentCell(*cv.cell)
        sg = r[self.index]
        qm = sg.quality_measures or {}
        self.note.setText(
            f"<b>{r.name(self.index)}</b> = cell (X cluster {sg.cell_i}, Y cluster {sg.cell_j}) of the "
            f"<b>{key[0]} k={key[1]} × {key[2]} l={key[3]}</b> table: observed {sg.observed_count}, expected "
            f"{_fmt(sg.expected_count, 1)}, {qm.get('cell_statistic_type', 'cell statistic')} = "
            f"{_fmt(qm.get('cell_statistic'))}, f-divergence of the whole table = {_fmt(qm.get('f_divergence_table'))}"
            f" (p = {_fmt(qm.get('p_table'))}). Cells: observed, expected under independence, adjusted residual; "
            "cells that are reported subgroups carry their name – click one to select it; clicking any other cell "
            "shows and sends that cell as an ad-hoc subgroup (no p-value), which can be kept with the button.")
        self.keep_button.setEnabled(bool(qm.get("interactive")))
        hx = cv.cell[0] if cv.cell else None
        hy = cv.cell[1] if cv.cell else None
        dx = insp.cluster_prototypes(r, cv.labels_x, r.X_columns, hx)
        dy = insp.cluster_prototypes(r, cv.labels_y, r.Y_columns, hy)
        fill_table(self.x_table, dx.drop(columns=["selected"]), bold_rows=[hx] if hx is not None else None)
        fill_table(self.y_table, dy.drop(columns=["selected"]), bold_rows=[hy] if hy is not None else None)

    def _cell_clicked(self, i, j):
        r = self.results
        if r is None or self.pair_combo.currentIndex() < 0:
            return
        key = self._keys[self.pair_combo.currentIndex()]
        cv = insp.pt_contingency(r, self.index, key)
        if cv is None:
            return
        idx = cv.subgroup_at.get((i, j))
        if idx is not None:
            if idx != self.index:
                self.select_requested.emit(idx)
            return
        ref = next(sg for sg in r.subgroups if sg.partition_result_X is not None and
                   (sg.clustering_X, int(sg.k), sg.clustering_Y, int(sg.l)) == key)
        mask = (cv.labels_x == i) & (cv.labels_y == j)
        if mask.sum() == 0:
            return
        sg = insp.adhoc_subgroup(
            r, mask, f"X: {key[0]} k={key[1]} cluster {i} | Y: {key[2]} l={key[3]} cluster {j} (chosen cell)",
            extra={"observed": int(cv.observed[i, j]), "expected": float(cv.expected[i, j]),
                   "residual": float(cv.residuals[i, j])},
            clustering_X=key[0], k=key[1], cell_i=i, clustering_Y=key[2], l=key[3], cell_j=j,
            observed_count=int(cv.observed[i, j]), expected_count=float(cv.expected[i, j]),
            partition_result_X=ref.partition_result_X, partition_result_Y=ref.partition_result_Y,
            clustering_metadata_X=dict(ref.clustering_metadata_X), clustering_metadata_Y=dict(ref.clustering_metadata_Y))
        self.adhoc_requested.emit((sg, f"cell ({i},{j}) of {key[0]} k={key[1]} × {key[2]} l={key[3]}"))

    def _keep(self):
        if self.ready() and (self.results[self.index].quality_measures or {}).get("interactive"):
            self.keep_requested.emit((self.results[self.index], self.results.name(self.index)))


# ---------------------------------------------------------------------------
# MRSD-HC: dendrogram
# ---------------------------------------------------------------------------

class DendrogramInspector(Inspector):
    title = "Dendrogram"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setMenuEnabled(False)
        self.plot.getAxis("bottom").setTicks([[]])
        self.plot.setLabel("left", "merge distance")
        self.plot.setLabel("bottom", "instances (leaves of the dendrogram of Y)")
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        row = QHBoxLayout()
        self.keep_button = QPushButton("Keep this node as a subgroup")
        self.keep_button.setEnabled(False)
        self.keep_button.clicked.connect(self._keep)
        row.addWidget(QLabel("Click a merge point to choose that node as a subgroup."))
        row.addStretch(1)
        row.addWidget(self.keep_button)
        self.layout().addLayout(row)
        self.layout().addWidget(self.plot, 1)
        self._cache_key = None
        self._Z = None
        self._node_xy: Dict[int, Tuple[float, float]] = {}

    def clear(self):
        self.plot.clear()

    def _linkage(self):
        key = id(self.results.frame)
        if self._cache_key != key:
            self._Z = insp.hc_linkage(self.results)
            self._cache_key = key
        return self._Z

    def show_subgroup(self):
        r, i = self.results, self.index
        Z = self._linkage()
        if Z is None:
            self.clear()
            self.note.setText("The dendrogram cannot be reproduced (precomputed distances).")
            return
        n = len(r.frame)
        node = insp.hc_node_of(r, i)
        seg = insp.dendrogram_segments(Z, n, node)
        self._node_xy = seg["node_xy"]
        self.plot.clear()

        def draw(segments, pen):
            if not segments:
                return
            arr = np.array(segments, dtype=float)
            x = np.column_stack([arr[:, 0], arr[:, 2]]).ravel()
            y = np.column_stack([arr[:, 1], arr[:, 3]]).ravel()
            self.plot.addItem(pg.PlotCurveItem(x, y, pen=pen, connect="pairs"))
        draw(seg["segments"], pg.mkPen("#999", width=1))
        draw(seg["highlight"], pg.mkPen(IN_COLOR, width=2.5))
        if node is not None and node in seg["node_xy"]:
            x, h = seg["node_xy"][node]
            self.plot.addItem(pg.ScatterPlotItem([x], [h], size=12, brush=pg.mkBrush(IN_COLOR), pen=pg.mkPen("k")))
            txt = pg.TextItem(f"{r.name(i)}: node {node}, n={r[i].coverage}", color=IN_COLOR.name(), anchor=(0.5, 1.2))
            txt.setPos(x, h)
            self.plot.addItem(txt)
        self.plot.autoRange()
        meta = r[i].clustering_metadata_Y or {}
        qm = r[i].quality_measures or {}
        self.keep_button.setEnabled(bool(qm.get("interactive")))
        chosen = " (chosen interactively, not tested)" if qm.get("interactive") else ""
        self.note.setText(
            f"<b>{r.name(i)}</b> is node {node} of the hierarchical clustering of Y{chosen} "
            f"({meta.get('linkage_method', '?')} linkage, {meta.get('distance_metric_used', '?')} distance), merged at "
            f"distance {_fmt(meta.get('merge_distance'))}; dispersion in Y {_fmt(qm.get('dispersion'))} vs. "
            f"{_fmt(qm.get('dataset_dispersion'))} for all data. Quality = out-of-fold AUC of predicting membership "
            f"from X: {_fmt(qm.get('quality'))} with {qm.get('best_model', '?')}. "
            "The p-value (if permutations > 0) is the share of permutations of the rows of X – the node is fixed, "
            "it was chosen from Y – with an AUC at least this large; adjusted p = Westfall–Young max-T over all "
            "nodes examined.")

    def _clicked(self, event):
        if not self._node_xy or self.results is None:
            return
        vb = self.plot.plotItem.vb
        pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        pt = vb.mapSceneToView(pos)
        px = vb.viewPixelSize()
        n = len(self.results.frame)
        best, best_d = None, None
        for nid, (x, h) in self._node_xy.items():
            if nid < n:
                continue
            d = max(abs(pt.x() - x) / max(px[0], 1e-12), abs(pt.y() - h) / max(px[1], 1e-12))
            if d < 12 and (best_d is None or d < best_d):
                best, best_d = nid, d
        if best is None:
            return
        r = self.results
        for idx in range(len(r)):
            if insp.hc_node_of(r, idx) == best and not (r[idx].quality_measures or {}).get("interactive"):
                self.select_requested.emit(idx)
                return
        Z = self._linkage()
        mask = insp.hc_node_mask(Z, n, best)
        ref = next((sg for sg in r.subgroups if sg.clustering_metadata_Y.get("linkage_method")), None)
        meta = dict(ref.clustering_metadata_Y) if ref is not None else {}
        meta["merge_distance"] = float(Z[best - n, 2])
        q = insp.hc_node_quality(r, mask)
        sg = insp.adhoc_subgroup(r, mask, f"HC node {best} (n={int(mask.sum())}, chosen)", quality=q,
                                 extra={"best_model": (r.config or {}).get("classification_model", "decision_tree")},
                                 k=best, l=int(mask.sum()), clustering_Y=meta.get("linkage_method"),
                                 clustering_metadata_Y=meta,
                                 clustering_metadata_X=dict(ref.clustering_metadata_X) if ref is not None else {})
        self.adhoc_requested.emit((sg, f"node {best}"))

    def _keep(self):
        if self.ready() and (self.results[self.index].quality_measures or {}).get("interactive"):
            self.keep_requested.emit((self.results[self.index], self.results.name(self.index)))


# ---------------------------------------------------------------------------
# classification-based algorithms (HC, C2 family, MV, CCA)
# ---------------------------------------------------------------------------

class ClassifierInspector(Inspector):
    """Membership classifier in X and / or Y: the MRSD (scikit-learn)
    classifiers, or Orange learners connected to the widget's Learner input –
    then the fitted Orange model goes to the widget's Model output."""
    title = "Classifier"

    def __init__(self, parent=None, spaces=("X", "Y")):
        super().__init__(parent)
        self.spaces = spaces
        self.learners = []
        self.builtin_allowed = True
        controls = QHBoxLayout()
        self.combos: Dict[str, QComboBox] = {}
        for sp in spaces:
            controls.addWidget(QLabel(f"Classifier in {sp}:"))
            cb = QComboBox()
            cb.currentIndexChanged.connect(self._choice_changed)
            controls.addWidget(cb)
            self.combos[sp] = cb
        controls.addWidget(QLabel("Folds:"))
        self.folds = QSpinBox()
        self.folds.setRange(2, 20)
        self.folds.setValue(5)
        self.folds.valueChanged.connect(self.refresh)
        controls.addWidget(self.folds)
        controls.addStretch(1)
        self.layout().addLayout(controls)
        split = QSplitter(Qt.Horizontal)
        self.panels: Dict[str, QTextBrowser] = {}
        self.tables: Dict[str, QTableView] = {}
        for sp in spaces:
            box = QWidget()
            box.setLayout(QVBoxLayout())
            box.layout().setContentsMargins(0, 0, 0, 0)
            tb = QTextBrowser()
            tv = make_table()
            box.layout().addWidget(tb, 3)
            box.layout().addWidget(tv, 2)
            split.addWidget(box)
            self.panels[sp] = tb
            self.tables[sp] = tv
        self.layout().addWidget(split, 1)
        self._defaults_set_for = None
        self._fill_combos()

    # -- choices -------------------------------------------------------------
    def _fill_combos(self):
        for sp, cb in self.combos.items():
            current = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            for k, learner in enumerate(self.learners):
                cb.addItem(f"Orange: {getattr(learner, 'name', type(learner).__name__)}", ("orange", k))
            if self.builtin_allowed or not self.learners:
                for value, label in params.CLASSIFIERS:
                    cb.addItem(f"MRSD: {label}", ("mrsd", value))
            idx = next((i for i in range(cb.count()) if cb.itemData(i) == current), 0)
            cb.setCurrentIndex(idx)
            cb.blockSignals(False)

    def set_learners(self, learners, builtin_allowed: bool = True):
        self.learners = list(learners or [])
        self.builtin_allowed = builtin_allowed
        self._fill_combos()            # Orange learners come first, so a new learner is selected by default
        self.refresh()

    def _choice_changed(self):
        self.refresh()

    def clear(self):
        for sp in self.spaces:
            self.panels[sp].clear()
            fill_table(self.tables[sp], None)
        self.model_changed.emit({})

    def set_results(self, results):
        self._defaults_set_for = None
        super().set_results(results)

    # -- content -------------------------------------------------------------
    def show_subgroup(self):
        r, i = self.results, self.index
        if self._defaults_set_for != id(r.frame) and not self.learners:
            cx, cy = insp.classifier_names_for(r, i)
            for sp, name in (("X", cx), ("Y", cy)):
                if sp in self.combos:
                    k = self.combos[sp].findData(("mrsd", name))
                    self.combos[sp].blockSignals(True)
                    self.combos[sp].setCurrentIndex(max(k, 0))
                    self.combos[sp].blockSignals(False)
            self._defaults_set_for = id(r.frame)
        qm = r[i].quality_measures or {}
        extra = ""
        if "q_X" in qm:
            extra = (f" The algorithm's own out-of-fold qualities: q_X = {_fmt(qm['q_X'])}, q_Y = {_fmt(qm['q_Y'])}"
                     f" (aggregated: {_fmt(qm.get('quality'))}).")
        self.note.setText(
            f"<b>{r.name(i)}</b>: {_esc(r.rule_text(i))} (n = {r[i].coverage}). Membership predicted from X and from Y "
            "with the chosen classifier – one of the MRSD classifiers or an Orange learner connected to the "
            "<i>Learner</i> input (its models are sent to the <i>Model (X)</i> / <i>Model (Y)</i> outputs, e.g. for "
            "Tree Viewer or Nomogram). "
            "<b>AUC (OOF/CV)</b> is estimated on rows the classifier did not train on with stratified folds; "
            "<b>AUC (train)</b> re-scores the training rows and is optimistic. "
            "The subgroup was selected on these same rows." + extra)
        models = {}
        for sp in self.spaces:
            kind, value = self.combos[sp].currentData()
            if kind == "mrsd":
                rep = insp.classifier_report(r, i, sp, value, self.folds.value())
                html = (f"<h4>{sp} → membership: {dict(params.CLASSIFIERS)[value]}</h4>"
                        f"AUC (OOF, {rep.n_folds} folds) = <b>{_fmt(rep.auc_oof)}</b> &nbsp; "
                        f"AUC (train) = {_fmt(rep.auc_train)}<pre>{_escape(rep.text)}</pre>")
                self.panels[sp].setHtml(html)
                fill_table(self.tables[sp], rep.importances, digits=3)
            else:
                learner = self.learners[value]
                rep = backend.orange_classifier_report(r, i, learner, n_folds=self.folds.value(), features=sp)
                if rep is None:
                    self.panels[sp].setHtml("<i>The results carry no data table; Orange learners need one.</i>")
                    fill_table(self.tables[sp], None)
                    continue
                html = (f"<h4>{sp} → membership: {getattr(learner, 'name', type(learner).__name__)}</h4>"
                        f"AUC (CV, {rep['folds']} folds) = <b>{_fmt(rep['auc_cv'])}</b> &nbsp; CA (CV) = "
                        f"{_fmt(rep['ca_cv'])} &nbsp; AUC (train) = {_fmt(rep['auc_train'])}<br>"
                        + backend.describe_orange_model(rep["model"]))
                self.panels[sp].setHtml(html)
                fill_table(self.tables[sp], None)
                models[sp] = rep["model"]
        self.model_changed.emit(models)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class ClassifierXInspector(ClassifierInspector):
    """X only (for algorithms that select candidates from Y: MRSD-HC)."""

    def __init__(self, parent=None):
        super().__init__(parent, spaces=("X",))


# ---------------------------------------------------------------------------
# MRSD-CCA
# ---------------------------------------------------------------------------

class CCAInspector(Inspector):
    title = "Canonical space"

    def __init__(self, parent=None):
        super().__init__(parent)
        split = QSplitter(Qt.Horizontal)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "first canonical variate of X (u₁)")
        self.plot.setLabel("left", "first canonical variate of Y (v₁)")
        self.plot.addLegend(offset=(10, 10))
        split.addWidget(self.plot)
        self.table = make_table()
        split.addWidget(self.table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.layout().addWidget(split, 1)

    def clear(self):
        self.plot.clear()
        fill_table(self.table, None)

    def show_subgroup(self):
        r, i = self.results, self.index
        v = insp.cca_view(r, i)
        self.plot.clear()
        self.plot.plotItem.legend.clear()
        m = v["mask"]
        out = pg.ScatterPlotItem(v["u"][~m], v["v"][~m], size=6, brush=pg.mkBrush(117, 112, 179, 90), pen=None)
        inn = pg.ScatterPlotItem(v["u"][m], v["v"][m], size=8, brush=pg.mkBrush(IN_COLOR), pen=pg.mkPen("k", width=0.5))
        self.plot.addItem(out)
        self.plot.addItem(inn)
        self.plot.plotItem.legend.addItem(out, f"complement (ρ₁ = {v['rho_complement']:.2f})")
        self.plot.plotItem.legend.addItem(inn, f"{r.name(i)} (ρ₁ = {v['rho_subgroup']:.2f})")
        for mask, color in ((m, IN_COLOR), (~m, OUT_COLOR)):
            if mask.sum() > 2 and np.std(v["u"][mask]) > 0:
                b, a = np.polyfit(v["u"][mask], v["v"][mask], 1)
                xs = np.array([v["u"][mask].min(), v["u"][mask].max()])
                self.plot.addItem(pg.PlotCurveItem(xs, a + b * xs, pen=pg.mkPen(color, width=2)))
        self.plot.autoRange()
        qm = r[i].quality_measures or {}
        self.note.setText(
            f"<b>{r.name(i)}</b>: {_esc(r.rule_text(i))} (n = {r[i].coverage}). Regularized CCA of X and Y on all rows: "
            f"ρ₁ = {v['rho_all']:.3f} overall, <b>{v['rho_subgroup']:.3f}</b> inside the subgroup vs. "
            f"{v['rho_complement']:.3f} in the complement (Fisher z = {_fmt(qm.get('fisher_z'))}). "
            "The table lists the canonical weights of the standardized / one-hot variables.")
        fill_table(self.table, v["weights"], digits=3)


# ---------------------------------------------------------------------------
# MRSD-EMM
# ---------------------------------------------------------------------------

class CorrelationInspector(Inspector):
    title = "Correlations"

    def __init__(self, parent=None):
        super().__init__(parent)
        split = QSplitter(Qt.Horizontal)
        self.table = make_table()
        self.table.selectionModel().selectionChanged.connect(self._pair_selected)
        split.addWidget(self.table)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setMenuEnabled(False)
        self.plot.addLegend(offset=(10, 10))
        split.addWidget(self.plot)
        self.layout().addWidget(split, 1)
        self._df = None

    def clear(self):
        fill_table(self.table, None)
        self.plot.clear()

    def show_subgroup(self):
        r, i = self.results, self.index
        self._df = insp.emm_correlations(r, i)
        fill_table(self.table, self._df)
        qm = r[i].quality_measures or {}
        self.note.setText(
            f"<b>{r.name(i)}</b>: {_esc(r.rule_text(i))} (n = {r[i].coverage}). Exceptionality: {qm.get('exceptionality_measure', '')}, "
            f"best pair {qm.get('best_pair', '')}, z = {_fmt(qm.get('quality'))}. Pearson correlations of numeric X–Y pairs "
            "inside the subgroup vs. in the complement, with the Fisher-z of their difference; select a pair to plot it.")
        pair = insp.best_pair(r, i)
        if pair is not None and len(self._df):
            rows = self._df.index[(self._df["X"] == pair[0]) & (self._df["Y"] == pair[1])]
            if len(rows):
                self.table.selectRow(int(rows[0]))
                return
        self._plot_pair(pair)

    def _pair_selected(self, *_):
        rows = self.table.selectionModel().selectedRows()
        if rows and self._df is not None:
            vals = [self.table.model().index(rows[0].row(), c).data() for c in (0, 1)]
            self._plot_pair((vals[0], vals[1]))

    def _plot_pair(self, pair):
        self.plot.clear()
        self.plot.plotItem.legend.clear()
        if pair is None or not self.ready():
            return
        r, i = self.results, self.index
        x, y = pair
        if x not in r.frame.columns or y not in r.frame.columns:
            return
        m = insp.membership(r, i)
        a, b = r.frame[x].to_numpy(dtype=float), r.frame[y].to_numpy(dtype=float)
        out = pg.ScatterPlotItem(a[~m], b[~m], size=6, brush=pg.mkBrush(117, 112, 179, 90), pen=None)
        inn = pg.ScatterPlotItem(a[m], b[m], size=8, brush=pg.mkBrush(IN_COLOR), pen=pg.mkPen("k", width=0.5))
        self.plot.addItem(out)
        self.plot.addItem(inn)
        self.plot.plotItem.legend.addItem(out, "complement")
        self.plot.plotItem.legend.addItem(inn, r.name(i))
        for mask, color in ((m, IN_COLOR), (~m, OUT_COLOR)):
            ok = mask & ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() > 2 and np.std(a[ok]) > 0:
                slope, icpt = np.polyfit(a[ok], b[ok], 1)
                xs = np.array([a[ok].min(), a[ok].max()])
                self.plot.addItem(pg.PlotCurveItem(xs, icpt + slope * xs, pen=pg.mkPen(color, width=2)))
        self.plot.setLabel("bottom", x)
        self.plot.setLabel("left", y)
        self.plot.autoRange()


# ---------------------------------------------------------------------------
# MRSD-MI
# ---------------------------------------------------------------------------

class MutualInformationInspector(Inspector):
    title = "Mutual information"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = make_table()
        self.layout().addWidget(self.table, 1)

    def clear(self):
        fill_table(self.table, None)

    def show_subgroup(self):
        r, i = self.results, self.index
        qm = r[i].quality_measures or {}
        k = int((r.config or {}).get("k_neighbors", 5))
        df = insp.mi_pairs(r, i, k=k)
        fill_table(self.table, df, digits=4)
        model = self.table.model()
        col = list(df.columns).index("difference")
        scale = max(float(np.nanmax(np.abs(df["difference"].to_numpy(dtype=float)))), 1e-9)
        for row, v in enumerate(df["difference"]):
            model.setData(model.index(row, col), QBrush(_diverging(float(v), scale)), Qt.BackgroundRole)
        self.note.setText(
            f"<b>{r.name(i)}</b>: {_esc(r.rule_text(i))} (n = {r[i].coverage}). The algorithm's score: MI(X;Y) inside = "
            f"{_fmt(qm.get('mi_subgroup'), 4)} vs. {_fmt(qm.get('mi_complement'), 4)} in the complement "
            f"(null mean {_fmt(qm.get('null_mean'), 4)}, SD {_fmt(qm.get('null_sd'), 4)}, z = {_fmt(qm.get('quality'))}; "
            f"Linfoot inside {_fmt(qm.get('linfoot_subgroup'))}). Below: estimates for mixed variable types (k = {k}) of "
            "the joint and of every pairwise X–Y mutual information inside vs. outside the subgroup (nats; small "
            "samples are noisy).")


# ---------------------------------------------------------------------------
# MRSD-Rules
# ---------------------------------------------------------------------------

class RuleInspector(Inspector):
    title = "Rule"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = make_table()
        self.layout().addWidget(self.table, 1)

    def clear(self):
        fill_table(self.table, None)

    def show_subgroup(self):
        r, i = self.results, self.index
        df = insp.rule_conditions(r, i)
        fill_table(self.table, df, digits=1)
        qm = r[i].quality_measures or {}
        self.note.setText(
            f"<b>{r.name(i)}</b>: <code>{_esc(r.rule_text(i))}</code> covers {r[i].coverage} rows "
            f"({100 * r[i].coverage / len(r.frame):.1f} %). Quality ({qm.get('quality_measure', '')}) = "
            f"{_fmt(qm.get('quality'))}, Hotelling T² = {_fmt(qm.get('t2'), 2)}, asymptotic p = "
            f"{_fmt(qm.get('p_asymptotic'))}; {qm.get('n_candidates_examined', '?')} candidate rules were examined. "
            "Each condition with the coverage of the condition alone and of the conjunction up to it.")


# ---------------------------------------------------------------------------
# MRSD-Ensemble
# ---------------------------------------------------------------------------

class EnsembleInspector(Inspector):
    title = "Consensus"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = make_table()
        self.layout().addWidget(self.table, 1)

    def clear(self):
        fill_table(self.table, None)

    def show_subgroup(self):
        r, i = self.results, self.index
        fill_table(self.table, insp.ensemble_support(r, i))
        qm = r[i].quality_measures or {}
        rep = (r[i].clustering_metadata_X or {}).get("representative") or {}
        self.note.setText(
            f"<b>{r.name(i)}</b>: {_esc(r.rule_text(i))} (n = {r[i].coverage}). Consensus of "
            f"{qm.get('n_algorithms', '?')} of {qm.get('n_algorithms_run', '?')} member algorithms "
            f"({qm.get('fusion_strategy', '')} fusion, score {_fmt(qm.get('consensus_score'))}, mean Jaccard with the "
            f"members {_fmt(qm.get('mean_support_jaccard'))}). Representative member: {rep.get('algorithm', '—')} "
            f"rank {rep.get('rank', '')}: <i>{rep.get('rule') or rep.get('description', '')}</i>"
            + (f"; stability {_fmt(qm.get('stability'))}" if 'stability' in qm else "") + ".")


# ---------------------------------------------------------------------------
# MRSD-MV: the dendrograms of X and of Y side by side
# ---------------------------------------------------------------------------

class DualDendrogramInspector(Inspector):
    """Dendrogram of X (left) and of Y (right).  Click a merge point in one
    tree to select that node: its subtree is highlighted there and its rows
    are marked under the leaves of the other tree, so one sees where the same
    instances lie in the other space.  With a node selected in each tree the
    intersection can be sent on as a subgroup."""
    title = "Dendrograms"
    COLORS = {"X": QColor("#1b9e77"), "Y": QColor("#7570b3")}

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout()
        self.clear_button = QPushButton("Clear selection")
        self.clear_button.clicked.connect(self._clear_selection)
        self.send_button = QPushButton("Send selection as subgroup")
        self.send_button.clicked.connect(self._send)
        self.keep_button = QPushButton("Keep as subgroup")
        self.keep_button.clicked.connect(self._keep)
        self.selection_label = QLabel("")
        self.selection_label.setWordWrap(True)
        row.addWidget(self.selection_label, 1)
        for b in (self.clear_button, self.send_button, self.keep_button):
            row.addWidget(b)
        self.layout().addLayout(row)
        split = QSplitter(Qt.Horizontal)
        self.plots: Dict[str, pg.PlotWidget] = {}
        for sp in ("X", "Y"):
            plot = pg.PlotWidget(background="w")
            plot.setMenuEnabled(False)
            plot.getAxis("bottom").setTicks([[]])
            plot.setLabel("left", "merge distance")
            plot.setLabel("bottom", f"dendrogram of {sp}  (instances)")
            plot.scene().sigMouseClicked.connect(lambda ev, space=sp: self._clicked(space, ev))
            split.addWidget(plot)
            self.plots[sp] = plot
        self.layout().addWidget(split, 1)
        self._cache_key = None
        self._Z: Dict[str, np.ndarray] = {}
        self._weights: Dict[str, float] = {}
        self._node_xy: Dict[str, Dict[int, Tuple[float, float]]] = {"X": {}, "Y": {}}
        self.sel: Dict[str, Optional[int]] = {"X": None, "Y": None}
        self._shown_key = None

    # -- data ----------------------------------------------------------------
    def clear(self):
        for plot in self.plots.values():
            plot.clear()
        self.selection_label.setText("")

    def _linkages(self):
        r = self.results
        key = (id(r.frame), str((r.config or {}).get("linkage_method")), str((r.config or {}).get("x_weighting")),
               str((r.config or {}).get("weight_power")))
        if self._cache_key != key:
            Z_X, Z_Y, self._weights = insp.mv_linkages(r)
            self._Z = {"X": Z_X, "Y": Z_Y}
            self._cache_key = key
            self.sel = {"X": None, "Y": None}
        return self._Z

    def _mask(self, space: str) -> Optional[np.ndarray]:
        node = self.sel[space]
        if node is None:
            return None
        return insp.hc_node_mask(self._Z[space], len(self.results.frame), node)

    # -- drawing ---------------------------------------------------------------
    def show_subgroup(self):
        r, i = self.results, self.index
        self._linkages()
        qm = r[i].quality_measures or {}
        key = (self._cache_key, i, len(r))
        if key != self._shown_key:          # another subgroup: follow its nodes (if it has any)
            self._shown_key = key
            self.sel = {"X": qm.get("node_X"), "Y": qm.get("node_Y")}
        self.keep_button.setEnabled(bool(qm.get("interactive")))
        w = ", ".join(f"{k}: {v:.2f}" for k, v in self._weights.items())
        self.note.setText(
            f"<b>{r.name(i)}</b>: {_esc(r.rule_text(i))} (n = {r[i].coverage}). Hierarchical clustering of X (left) and of "
            "Y (right). Orange marks under the leaves: the rows of this subgroup. Click a merge point to select a "
            "node; its rows are marked (in that tree's colour) under the leaves of the other tree. With a node "
            "selected in both trees their intersection can be sent on as a subgroup."
            + (f"<br><small>Weights of the X variables in the X distances: {w}</small>" if w else ""))
        self._redraw()

    def _redraw(self):
        r, i = self.results, self.index
        n = len(r.frame)
        member = insp.membership(r, i)
        masks = {sp: self._mask(sp) for sp in ("X", "Y")}
        for sp, other in (("X", "Y"), ("Y", "X")):
            plot = self.plots[sp]
            plot.clear()
            seg = insp.dendrogram_segments(self._Z[sp], n, self.sel[sp])
            self._node_xy[sp] = seg["node_xy"]
            self._draw(plot, seg["segments"], pg.mkPen("#999", width=1))
            self._draw(plot, seg["highlight"], pg.mkPen(self.COLORS[sp], width=2.5))
            hmax = max((h for _, h in seg["node_xy"].values()), default=1.0) or 1.0
            leaf_x = np.array([seg["node_xy"][leaf][0] for leaf in range(n)])
            if member.any():
                plot.addItem(pg.ScatterPlotItem(leaf_x[member], np.full(int(member.sum()), -0.03 * hmax), size=6,
                                                symbol="s", brush=pg.mkBrush(IN_COLOR), pen=None))
            if masks[other] is not None:
                m = masks[other]
                plot.addItem(pg.ScatterPlotItem(leaf_x[m], np.full(int(m.sum()), -0.07 * hmax), size=6, symbol="s",
                                                brush=pg.mkBrush(self.COLORS[other]), pen=None))
            if self.sel[sp] is not None and self.sel[sp] in seg["node_xy"]:
                x, h = seg["node_xy"][self.sel[sp]]
                plot.addItem(pg.ScatterPlotItem([x], [h], size=12, brush=pg.mkBrush(self.COLORS[sp]),
                                                pen=pg.mkPen("k")))
            plot.autoRange()
        self._update_selection_label(masks)

    @staticmethod
    def _draw(plot, segments, pen):
        if not segments:
            return
        arr = np.array(segments, dtype=float)
        x = np.column_stack([arr[:, 0], arr[:, 2]]).ravel()
        y = np.column_stack([arr[:, 1], arr[:, 3]]).ravel()
        plot.addItem(pg.PlotCurveItem(x, y, pen=pen, connect="pairs"))

    def _update_selection_label(self, masks):
        a, b = masks["X"], masks["Y"]
        n = len(self.results.frame)
        if a is None and b is None:
            self.selection_label.setText("No node selected.")
            self.send_button.setEnabled(False)
            return
        self.send_button.setEnabled(True)
        if a is not None and b is not None:
            st = insp.node_pair_statistics(a, b)
            self.selection_label.setText(
                f"X node {self.sel['X']} (n = {st['size_a']}) ∩ Y node {self.sel['Y']} (n = {st['size_b']}): "
                f"<b>{st['observed']}</b> rows, expected {st['expected']:.1f}, adjusted residual "
                f"<b>{st['residual']:+.2f}</b>, Jaccard of the nodes {st['jaccard']:.2f}.")
            return
        sp, other = ("X", "Y") if a is not None else ("Y", "X")
        mask = a if a is not None else b
        node, jac = insp.best_matching_node(self._Z[other], n, mask)
        self.selection_label.setText(
            f"{sp} node {self.sel[sp]}: {int(mask.sum())} rows; the best matching node of the {other} tree is "
            f"node {node} (Jaccard {jac:.2f}). Click a node in the {other} tree to intersect.")

    # -- interaction -------------------------------------------------------------
    def _clicked(self, space: str, event):
        if not self.ready() or not self._node_xy[space]:
            return
        plot = self.plots[space]
        pos = event.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        vb = plot.plotItem.vb
        pt = vb.mapSceneToView(pos)
        px = vb.viewPixelSize()
        n = len(self.results.frame)
        best, best_d = None, None
        for nid, (x, h) in self._node_xy[space].items():
            if nid < n:
                continue
            d = max(abs(pt.x() - x) / max(px[0], 1e-12), abs(pt.y() - h) / max(px[1], 1e-12))
            if d < 12 and (best_d is None or d < best_d):
                best, best_d = nid, d
        if best is None:
            return
        self.sel[space] = best
        self._redraw()

    def _clear_selection(self):
        self.sel = {"X": None, "Y": None}
        if self.ready():
            self._redraw()

    def _send(self):
        if not self.ready():
            return
        r = self.results
        a, b = self._mask("X"), self._mask("Y")
        if a is None and b is None:
            return
        extra = {"method": "dual_dendrogram"}
        if a is not None and b is not None:
            mask = a & b
            st = insp.node_pair_statistics(a, b)
            extra.update({"node_X": int(self.sel["X"]), "node_Y": int(self.sel["Y"]), "observed": st["observed"],
                          "expected": st["expected"], "residual": st["residual"], "jaccard_nodes": st["jaccard"]})
            quality, name = st["residual"], f"X node {self.sel['X']} ∩ Y node {self.sel['Y']}"
        else:
            sp = "X" if a is not None else "Y"
            mask = a if a is not None else b
            extra[f"node_{sp}"] = int(self.sel[sp])
            quality, name = None, f"{sp} node {self.sel[sp]}"
        if mask.sum() == 0:
            self.selection_label.setText("The intersection is empty.")
            return
        sg = insp.adhoc_subgroup(r, mask, f"MV dendrograms: {name} (chosen)", extra=extra, quality=quality,
                                 clustering_X="dual_dendrogram", clustering_Y="multi_view")
        self._shown_key = None
        self.adhoc_requested.emit((sg, name))

    def _keep(self):
        if self.ready() and (self.results[self.index].quality_measures or {}).get("interactive"):
            self.keep_requested.emit((self.results[self.index], self.results.name(self.index)))


# ---------------------------------------------------------------------------
# which inspectors for which algorithm
# ---------------------------------------------------------------------------

INSPECTORS: Dict[str, List[Type[Inspector]]] = {
    "pt": [ContingencyInspector, PrototypeInspector, MeasuresInspector],
    "hc": [DendrogramInspector, ClassifierXInspector, PrototypeInspector, MeasuresInspector],
    "swapping": [ClassifierInspector, PrototypeInspector, MeasuresInspector],
    "semisupervised": [ClassifierInspector, PrototypeInspector, MeasuresInspector],
    "ranking": [ClassifierInspector, PrototypeInspector, MeasuresInspector],
    "cca": [CCAInspector, ClassifierInspector, PrototypeInspector, MeasuresInspector],
    "emm": [CorrelationInspector, PrototypeInspector, MeasuresInspector],
    "mv": [DualDendrogramInspector, ClassifierInspector, PrototypeInspector, MeasuresInspector],
    "mi": [MutualInformationInspector, PrototypeInspector, MeasuresInspector],
    "rules": [RuleInspector, PrototypeInspector, MeasuresInspector],
    "ensemble": [EnsembleInspector, PrototypeInspector, MeasuresInspector],
}
