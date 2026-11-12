"""
Variable Roles: choose which variables are independent (X), which are
dependent (Y) and which are ignored.  The roles are encoded in the domain of
the output table (X = attributes, Y = class variables, ignored = metas), which
is what the algorithm widgets expect.
"""
from __future__ import annotations

from typing import List

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QAbstractItemView, QSizePolicy

from Orange.data import Domain, Table, Variable
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.listfilter import VariablesListItemView
from Orange.widgets.widget import Input, Msg, Output, OWWidget
from orangecontrib.subgroups.widgets.utils import guard_wheel_later


def _selected_rows(view) -> List[int]:
    return sorted({i.row() for i in view.selectionModel().selectedRows()}, reverse=True)


class OWVariableRoles(OWWidget):
    name = "Variable Roles"
    description = "Set the independent (X), dependent (Y) and ignored variables for subgroup discovery."
    icon = "icons/roles.svg"
    priority = 20
    keywords = ["subgroup discovery", "MRSD", "roles", "targets", "dependent", "independent", "select columns"]

    want_main_area = False
    resizing_enabled = True

    settingsHandler = DomainContextHandler()
    x_vars = ContextSetting([])
    y_vars = ContextSetting([])
    ignored_vars = ContextSetting([])
    auto_commit = Setting(True)

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class Warning(OWWidget.Warning):
        no_y = Msg("No dependent variables are selected.")
        no_x = Msg("No independent variables are selected.")
        one_y = Msg("Only one dependent variable – MRSD is meant for several, but will run.")

    def __init__(self):
        super().__init__()
        guard_wheel_later(self)
        self.data = None
        self._updating = False

        self.ignored_model = VariableListModel(enable_dnd=True)
        self.x_model = VariableListModel(enable_dnd=True)
        self.y_model = VariableListModel(enable_dnd=True)

        layout = gui.hBox(self.controlArea)
        # --- ignored
        box = gui.vBox(layout, "Ignored (meta)")
        self.ignored_view = self._view(self.ignored_model)
        box.layout().addWidget(self.ignored_view)
        # --- middle buttons
        mid = gui.vBox(layout)
        mid.layout().addStretch(1)
        gui.button(mid, self, "→ X", callback=lambda: self._move(self.ignored_view, self.ignored_model, self.x_model),
                   tooltip="Move selected variables to the independent variables")
        gui.button(mid, self, "← X", callback=lambda: self._move(self.x_view, self.x_model, self.ignored_model),
                   tooltip="Ignore selected independent variables")
        gui.separator(mid, 12)
        gui.button(mid, self, "→ Y", callback=lambda: self._move(self.ignored_view, self.ignored_model, self.y_model),
                   tooltip="Move selected variables to the dependent variables")
        gui.button(mid, self, "← Y", callback=lambda: self._move(self.y_view, self.y_model, self.ignored_model),
                   tooltip="Ignore selected dependent variables")
        gui.separator(mid, 12)
        gui.button(mid, self, "X ⇄ Y", callback=self._swap, tooltip="Swap the independent and dependent variables")
        gui.button(mid, self, "Reset", callback=self._reset, tooltip="Roles as they are in the input data")
        mid.layout().addStretch(1)
        # --- X and Y
        right = gui.vBox(layout)
        box = gui.vBox(right, "Independent variables (X)")
        self.x_view = self._view(self.x_model)
        box.layout().addWidget(self.x_view)
        box = gui.vBox(right, "Dependent variables (Y)")
        self.y_view = self._view(self.y_model)
        box.layout().addWidget(self.y_view)

        for model in (self.ignored_model, self.x_model, self.y_model):
            model.rowsInserted.connect(self._changed)
            model.rowsRemoved.connect(self._changed)

        gui.auto_commit(self.buttonsArea, self, "auto_commit", "Send", box=False)
        self.setMinimumWidth(620)

    @staticmethod
    def _view(model):
        view = VariablesListItemView(acceptedType=Variable)
        view.setModel(model)
        view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view.setMinimumHeight(120)
        return view

    # ------------------------------------------------------------------ moves
    def _move(self, src_view, src_model, dst_model):
        rows = _selected_rows(src_view)
        moved = [src_model[r] for r in rows]
        for r in rows:
            del src_model[r]
        dst_model.extend(reversed(moved))

    def _swap(self):
        xs, ys = list(self.x_model), list(self.y_model)
        self.x_model[:] = ys
        self.y_model[:] = xs
        self._changed()

    def _reset(self):
        if self.data is None:
            return
        dom = self.data.domain
        self.x_model[:] = list(dom.attributes)
        self.y_model[:] = list(dom.class_vars)
        self.ignored_model[:] = list(dom.metas)
        self._changed()

    def _changed(self, *_):
        if self._updating:
            return
        self.x_vars = list(self.x_model)
        self.y_vars = list(self.y_model)
        self.ignored_vars = list(self.ignored_model)
        self.commit.deferred()

    # ------------------------------------------------------------------ input / output
    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.data = data
        if data is None:
            self.ignored_model[:] = []
            self.x_model[:] = []
            self.y_model[:] = []
            self.Warning.clear()
            self.Outputs.data.send(None)
            return
        dom = data.domain
        self.x_vars = list(dom.attributes)
        self.y_vars = list(dom.class_vars)
        self.ignored_vars = list(dom.metas)
        self.openContext(data)
        # variables of the domain only (context may be stale)
        known = set(dom.variables) | set(dom.metas)
        used = set()
        for lst in (self.x_vars, self.y_vars, self.ignored_vars):
            lst[:] = [v for v in lst if v in known and v not in used]
            used.update(lst)
        self.ignored_vars += [v for v in list(dom.variables) + list(dom.metas) if v not in used]
        self._updating = True
        try:
            self.x_model[:] = self.x_vars
            self.y_model[:] = self.y_vars
            self.ignored_model[:] = self.ignored_vars
        finally:
            self._updating = False
        self.commit.now()

    @gui.deferred
    def commit(self):
        self.Warning.clear()
        if self.data is None:
            self.Outputs.data.send(None)
            return
        xs, ys, ms = list(self.x_model), list(self.y_model), list(self.ignored_model)
        if not ys:
            self.Warning.no_y()
        elif len(ys) == 1:
            self.Warning.one_y()
        if not xs:
            self.Warning.no_x()
        domain = Domain(xs, ys, ms)
        self.Outputs.data.send(self.data.transform(domain))

    def send_report(self):
        self.report_items("Roles", [("Independent (X)", ", ".join(v.name for v in self.x_model)),
                                    ("Dependent (Y)", ", ".join(v.name for v in self.y_model)),
                                    ("Ignored", ", ".join(v.name for v in self.ignored_model))])


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    WidgetPreview(OWVariableRoles).run(Table(dataset_path("customers")))
