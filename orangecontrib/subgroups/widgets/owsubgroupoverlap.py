"""
Subgroup Overlap: heat map of the pairwise Jaccard similarity (or overlap
coefficient) of the subgroups, plus a bar of their sizes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from AnyQt.QtCore import Qt

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.subgroups.backend import SubgroupResults
from orangecontrib.subgroups.widgets.utils import guard_wheel_later


def overlap_matrix(results: SubgroupResults, measure: str) -> np.ndarray:
    sets = [set(sg.covered_instances) for sg in results.subgroups]
    n = len(sets)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            if measure == "jaccard":
                u = len(sets[i] | sets[j])
                v = inter / u if u else 0.0
            elif measure == "overlap":
                m = min(len(sets[i]), len(sets[j]))
                v = inter / m if m else 0.0
            else:  # dice
                s = len(sets[i]) + len(sets[j])
                v = 2 * inter / s if s else 0.0
            M[i, j] = M[j, i] = v
    return M


class OWSubgroupOverlap(OWWidget):
    name = "Subgroup Overlap"
    description = "Heat map of the pairwise overlap (Jaccard) between subgroups."
    icon = "icons/overlap.svg"
    priority = 410
    keywords = ["subgroup discovery", "MRSD", "overlap", "jaccard", "heat map", "similarity", "visualize"]

    want_main_area = True
    resizing_enabled = True

    class Inputs:
        subgroups = Input("Subgroups", SubgroupResults)

    class Outputs:
        intersection = Output("Intersection", Table, default=True)
        first_only = Output("First Only", Table)
        second_only = Output("Second Only", Table)
        matrix = Output("Overlap Matrix", Table)

    class Information(OWWidget.Information):
        no_data = Msg("The subgroups carry no data table; a clicked cell cannot send rows.")

    measure = Setting(0)
    show_values = Setting(True)
    max_subgroups = Setting(30)

    MEASURES = [("jaccard", "Jaccard  |A∩B| / |A∪B|"), ("overlap", "Overlap  |A∩B| / min(|A|,|B|)"),
                ("dice", "Dice  2|A∩B| / (|A|+|B|)")]

    def __init__(self):
        super().__init__()
        guard_wheel_later(self)
        self.results: Optional[SubgroupResults] = None
        self.matrix: Optional[np.ndarray] = None

        box = gui.vBox(self.controlArea, "Measure")
        gui.radioButtons(box, self, "measure", [lbl for _, lbl in self.MEASURES], callback=self._replot)
        box = gui.vBox(self.controlArea, "Display")
        gui.checkBox(box, self, "show_values", "Show values in cells", callback=self._replot)
        gui.spin(box, self, "max_subgroups", 2, 200, label="At most subgroups", callback=self._replot,
                 tooltip="Only the first subgroups (in the order of the input) are shown")
        gui.rubber(self.controlArea)
        self.info_label = gui.widgetLabel(self.controlArea, "")
        self.cell_label = gui.widgetLabel(
            gui.vBox(self.controlArea, "Selected cell"),
            "Click a cell (row A, column B) to send\nA ∩ B, A without B and B without A.")
        self.cell_label.setWordWrap(True)
        self.sub: Optional[SubgroupResults] = None
        self.cell = None
        self.marker = None

        self.plot = pg.PlotWidget(background="w")
        self.plot.setMenuEnabled(False)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.getPlotItem().setContentsMargins(5, 5, 15, 5)
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.colormap = pg.ColorMap(pos=np.linspace(0, 1, 5),
                                    color=[(255, 255, 255), (198, 219, 239), (107, 174, 214), (33, 113, 181),
                                           (8, 48, 107)])
        self.image.setLookupTable(self.colormap.getLookupTable(0.0, 1.0, 256))
        self.image.setLevels([0, 1])
        self.mainArea.layout().addWidget(self.plot)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self.texts = []

    @Inputs.subgroups
    def set_subgroups(self, results: Optional[SubgroupResults]):
        self.results = results
        self.cell = None
        self.Information.no_data(shown=results is not None and results.data is None)
        self._replot()
        self._send_cell()

    # ------------------------------------------------------------------ cell selection
    def _clicked(self, event):
        if self.sub is None:
            return
        pos = event.scenePos()
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        pt = self.plot.plotItem.vb.mapSceneToView(pos)
        n = len(self.sub)
        col, row = int(np.floor(pt.x())), int(np.floor(pt.y()))
        if not (0 <= row < n and 0 <= col < n):
            return
        self.cell = (row, col)
        self._mark_cell()
        self._send_cell()

    def _mark_cell(self):
        if self.marker is not None:
            self.plot.removeItem(self.marker)
            self.marker = None
        if self.cell is None or self.sub is None:
            return
        i, j = self.cell
        if i >= len(self.sub) or j >= len(self.sub):
            self.cell = None
            return
        xs = [j, j + 1, j + 1, j, j]
        ys = [i, i, i + 1, i + 1, i]
        self.marker = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#d95f02", width=3))
        self.plot.addItem(self.marker)

    def _send_cell(self):
        sub = self.sub
        if sub is None or self.cell is None or sub.data is None:
            for out in (self.Outputs.intersection, self.Outputs.first_only, self.Outputs.second_only):
                out.send(None)
            if self.cell is None:
                self.cell_label.setText("Click a cell (row A, column B) to send\nA ∩ B, A without B and B without A.")
            return
        i, j = self.cell
        a, b = sub.mask(i), sub.mask(j)
        parts = {"intersection": a & b, "first": a & ~b, "second": b & ~a}

        def rows(mask, name):
            if not mask.any():
                return None
            t = sub.data[np.flatnonzero(mask)]
            t.name = name
            return t
        A, B = sub.name(i), sub.name(j)
        self.Outputs.intersection.send(rows(parts["intersection"], f"{A} and {B}"))
        self.Outputs.first_only.send(rows(parts["first"], f"{A} without {B}"))
        self.Outputs.second_only.send(rows(parts["second"], f"{B} without {A}"))
        self.cell_label.setText(
            f"<b>A = {A}</b> ({int(a.sum())} rows)<br><b>B = {B}</b> ({int(b.sum())} rows)<br>"
            f"A ∩ B: <b>{int(parts['intersection'].sum())}</b> → Intersection<br>"
            f"A without B: <b>{int(parts['first'].sum())}</b> → First Only<br>"
            f"B without A: <b>{int(parts['second'].sum())}</b> → Second Only")

    def _replot(self):
        for t in self.texts:
            self.plot.removeItem(t)
        self.texts = []
        if self.results is None or len(self.results) == 0:
            self.image.clear()
            self.matrix = None
            self.sub = None
            self._mark_cell()
            self.info_label.setText("")
            self.Outputs.matrix.send(None)
            return
        n = min(len(self.results), self.max_subgroups)
        sub = self.results.subset(range(n))
        self.sub = sub
        key = self.MEASURES[self.measure][0]
        M = overlap_matrix(sub, key)
        self.matrix = M
        names = [sub.name(i) for i in range(n)]
        # pyqtgraph images are (x, y) indexed -> transpose so rows go down
        self.image.setImage(M.T, autoLevels=False)
        self.image.setLevels([0, 1])
        ticks = [[(i + 0.5, names[i]) for i in range(n)]]
        self.plot.getAxis("bottom").setTicks(ticks)
        self.plot.getAxis("left").setTicks(ticks)
        self.plot.getAxis("bottom").setStyle(tickTextOffset=4)
        self.plot.setXRange(0, n, padding=0.02)
        self.plot.setYRange(0, n, padding=0.02)
        if self.show_values:
            for i in range(n):
                for j in range(n):
                    v = M[i, j]
                    color = "w" if v > 0.6 else "#222"
                    t = pg.TextItem(f"{v:.2f}", color=color, anchor=(0.5, 0.5))
                    t.setPos(j + 0.5, i + 0.5)
                    self.plot.addItem(t)
                    self.texts.append(t)
        off = M[np.triu_indices(n, 1)]
        self.info_label.setText(f"{n} subgroups; mean pairwise {key}: {off.mean():.2f}" if len(off) else
                                f"{n} subgroup")
        # output as a table
        attrs = [ContinuousVariable(nm) for nm in names]
        domain = Domain(attrs, metas=[StringVariable("subgroup")])
        table = Table.from_numpy(domain, M, metas=np.array(names, dtype=object).reshape(-1, 1))
        table.name = f"{key} overlap"
        self.Outputs.matrix.send(table)
        self._mark_cell()

    def send_report(self):
        self.report_plot(self.plot)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups import backend
    from orangecontrib.subgroups.datasets import dataset_path
    from orangecontrib.subgroups.params import setting_defaults, widget_config
    _data = Table(dataset_path("customers"))
    _cfg = widget_config("rules", {**setting_defaults("rules"), "n_search_permutations": 10})
    _res = backend.run_algorithm("rules", backend.table_to_dataframe(_data), *backend.table_roles(_data), _cfg,
                                 data=_data)
    WidgetPreview(OWSubgroupOverlap).run(_res)
