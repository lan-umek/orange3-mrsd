"""
Subgroup Profile: bar chart of how each dependent variable behaves inside the
chosen subgroups (effect sizes vs. all data, standardized differences vs. the
complement, or raw means), one group of bars per dependent variable.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from orangecontrib.subgroups.backend import SubgroupResults
from orangecontrib.subgroups.widgets.utils import guard_wheel_later

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]

MODES = ["Effect size vs. all data (from the results)",
         "Standardized difference vs. complement",
         "Mean in subgroup and in complement"]


class OWSubgroupProfile(OWWidget):
    name = "Subgroup Profile"
    description = "Bar chart of the dependent variables inside the chosen subgroups."
    icon = "icons/profile.svg"
    priority = 400
    keywords = ["subgroup discovery", "MRSD", "profile", "effect size", "bar chart", "visualize"]

    want_main_area = True
    resizing_enabled = True

    class Inputs:
        subgroups = Input("Subgroups", SubgroupResults)

    class Outputs:
        selected = Output("Selected Subgroups", SubgroupResults)

    mode = Setting(0)
    max_initial = Setting(5)
    show_values = Setting(True)

    def __init__(self):
        super().__init__()
        guard_wheel_later(self)
        self.results: Optional[SubgroupResults] = None

        box = gui.vBox(self.controlArea, "Subgroups")
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self._replot)
        box.layout().addWidget(self.list)
        gui.spin(box, self, "max_initial", 1, 50, label="Preselect first", callback=self._preselect)

        box = gui.vBox(self.controlArea, "Display")
        gui.radioButtons(box, self, "mode", MODES, callback=self._replot)
        gui.checkBox(box, self, "show_values", "Show values", callback=self._replot)
        gui.widgetLabel(box, "Categorical dependent variables are shown\nas differences of level proportions.")
        gui.rubber(self.controlArea)

        self.plot = pg.PlotWidget(background="w")
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=False, y=False)
        self.plot.getPlotItem().setContentsMargins(5, 5, 15, 5)
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.mainArea.layout().addWidget(self.plot)

    # ------------------------------------------------------------------ input
    @Inputs.subgroups
    def set_subgroups(self, results: Optional[SubgroupResults]):
        self.results = results
        self.list.blockSignals(True)
        self.list.clear()
        if results is not None:
            for i in range(len(results)):
                sg = results[i]
                item = QListWidgetItem(f"{results.name(i)}  ({results.rule_text(i)[:60]}, n={sg.coverage})")
                item.setData(Qt.UserRole, i)
                item.setToolTip(results.rule_text(i))
                self.list.addItem(item)
        self.list.blockSignals(False)
        self._preselect()

    def _preselect(self):
        self.list.blockSignals(True)
        for k in range(self.list.count()):
            self.list.item(k).setSelected(k < self.max_initial)
        self.list.blockSignals(False)
        self._replot()

    def _selected(self) -> List[int]:
        return sorted(it.data(Qt.UserRole) for it in self.list.selectedItems())

    # ------------------------------------------------------------------ values
    def _profile(self, indices: List[int]) -> pd.DataFrame:
        """Rows = subgroups (and 'complement' pairs in mode 2), columns = targets / target levels."""
        r = self.results
        if self.mode == 0:
            eff = r.target_effects()
            return eff.iloc[indices]
        frame = r.frame
        cols = {}
        for i in indices:
            sg = r[i]
            mask = sg.mask(len(frame))
            comp = ~mask
            row_in, row_diff = {}, {}
            for col in r.Y_columns:
                s = frame[col]
                if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
                    v = s.to_numpy(dtype=float)
                    m_in = np.nanmean(v[mask]) if mask.any() else np.nan
                    m_out = np.nanmean(v[comp]) if comp.any() else np.nan
                    sd = np.nanstd(v[comp], ddof=1) if comp.sum() > 1 else np.nan
                    row_diff[col] = (m_in - m_out) / sd if sd and sd > 0 else np.nan
                    row_in[col] = (m_in, m_out)
                else:
                    levels = list(s.astype("category").cat.categories)
                    for lv in levels:
                        p_in = float((s[mask] == lv).mean()) if mask.any() else np.nan
                        p_out = float((s[comp] == lv).mean()) if comp.any() else np.nan
                        row_diff[f"{col}={lv}"] = p_in - p_out
                        row_in[f"{col}={lv}"] = (p_in, p_out)
            cols[f"{i}"] = row_diff if self.mode == 1 else row_in
        return pd.DataFrame(cols).T

    # ------------------------------------------------------------------ plotting
    def _replot(self):
        self.plot.clear()
        self.legend.clear()
        indices = self._selected()
        if self.results is None or not indices:
            self._send_selected(indices)
            return
        df = self._profile(indices)
        targets = list(df.columns)
        K = len(indices)
        names = [self.results.name(i) for i in indices]
        ticks = []
        if self.mode == 2:
            # pairs of bars (subgroup, complement) per target per subgroup
            width = 0.8 / (2 * K)
            for k, (name, i) in enumerate(zip(names, indices)):
                color = QColor(PALETTE[k % len(PALETTE)])
                xs_in = np.array([j + (2 * k - K + 0.5) * width for j in range(len(targets))])
                vals_in = np.array([df.iloc[k][t][0] if isinstance(df.iloc[k][t], tuple) else np.nan
                                    for t in targets], dtype=float)
                vals_out = np.array([df.iloc[k][t][1] if isinstance(df.iloc[k][t], tuple) else np.nan
                                     for t in targets], dtype=float)
                bar_in = pg.BarGraphItem(x=xs_in, height=np.nan_to_num(vals_in), width=width, brush=color,
                                         pen=pg.mkPen("k", width=0.5))
                light = QColor(color)
                light.setAlpha(90)
                bar_out = pg.BarGraphItem(x=xs_in + width, height=np.nan_to_num(vals_out), width=width,
                                          brush=light, pen=pg.mkPen("k", width=0.5))
                self.plot.addItem(bar_in)
                self.plot.addItem(bar_out)
                self.legend.addItem(bar_in, f"{name} (subgroup / complement)")
                if self.show_values:
                    for x, v in list(zip(xs_in, vals_in)) + list(zip(xs_in + width, vals_out)):
                        self._label(x, v)
            self.plot.setLabel("left", "mean")
        else:
            width = 0.8 / K
            for k, (name, i) in enumerate(zip(names, indices)):
                color = QColor(PALETTE[k % len(PALETTE)])
                xs = np.array([j + (k - (K - 1) / 2) * width for j in range(len(targets))])
                vals = df.iloc[k].to_numpy(dtype=float)
                bar = pg.BarGraphItem(x=xs, height=np.nan_to_num(vals), width=width, brush=color,
                                      pen=pg.mkPen("k", width=0.5))
                self.plot.addItem(bar)
                self.legend.addItem(bar, name)
                if self.show_values:
                    for x, v in zip(xs, vals):
                        self._label(x, v)
            self.plot.addLine(y=0, pen=pg.mkPen("#444", width=1))
            self.plot.setLabel("left", "effect size (SD units) / proportion difference" if self.mode == 0
                               else "standardized difference vs. complement")
        ticks = [(j, t) for j, t in enumerate(targets)]
        ax = self.plot.getAxis("bottom")
        ax.setTicks([ticks])
        self.plot.setLabel("bottom", "dependent variables")
        self.plot.autoRange()
        self._send_selected(indices)

    def _label(self, x, v):
        if v is None or np.isnan(v):
            return
        txt = pg.TextItem(f"{v:.2f}", color="#333", anchor=(0.5, 1.0 if v >= 0 else 0.0))
        txt.setPos(x, v)
        self.plot.addItem(txt)

    def _send_selected(self, indices):
        if self.results is None or not indices:
            self.Outputs.selected.send(None)
        else:
            self.Outputs.selected.send(self.results.subset(indices))

    def send_report(self):
        self.report_plot(self.plot)


if __name__ == "__main__":  # pragma: no cover
    from Orange.data import Table
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups import backend
    from orangecontrib.subgroups.datasets import dataset_path
    from orangecontrib.subgroups.params import setting_defaults, widget_config
    _data = Table(dataset_path("customers"))
    _cfg = widget_config("rules", {**setting_defaults("rules"), "n_search_permutations": 10})
    _res = backend.run_algorithm("rules", backend.table_to_dataframe(_data), *backend.table_roles(_data), _cfg,
                                 data=_data)
    WidgetPreview(OWSubgroupProfile).run(_res)
