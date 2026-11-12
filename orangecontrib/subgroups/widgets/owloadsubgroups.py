"""
Load Subgroups: reload a SubgroupResults object saved as a pickle by
'Save Subgroups' and, optionally, re-attach the data table it was computed on.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

from AnyQt.QtWidgets import QFileDialog

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.subgroups import backend, export
from orangecontrib.subgroups.backend import SubgroupResults
from orangecontrib.subgroups.widgets.utils import guard_wheel_later


class OWLoadSubgroups(OWWidget):
    name = "Load Subgroups"
    description = "Load subgroups saved as a pickle file; optionally attach the data they were computed on."
    icon = "icons/load.svg"
    priority = 510
    keywords = ["subgroup discovery", "MRSD", "load", "open", "pickle"]

    want_main_area = False
    resizing_enabled = False

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        subgroups = Output("Subgroups", SubgroupResults, default=True)
        annotated = Output("Annotated Data", Table)

    class Error(OWWidget.Error):
        load_failed = Msg("Cannot load the file: {}")

    class Warning(OWWidget.Warning):
        data_mismatch = Msg("The data table has {} rows but the subgroups were computed on {} rows; "
                            "the table was not attached.")

    filename = Setting("", schema_only=True)
    last_dir = Setting("")

    def __init__(self):
        super().__init__()
        guard_wheel_later(self)
        self.data: Optional[Table] = None
        self.results: Optional[SubgroupResults] = None
        box = gui.vBox(self.controlArea, "File")
        self.file_label = gui.widgetLabel(box, "")
        self.file_label.setWordWrap(True)
        row = gui.hBox(box)
        gui.button(row, self, "Open …", callback=self.browse)
        gui.button(row, self, "Reload", callback=self.load)
        self.info_label = gui.widgetLabel(self.controlArea, "")
        self.setMinimumWidth(380)
        self._update_label()
        if self.filename and os.path.exists(self.filename):
            self.load()

    def _update_label(self):
        self.file_label.setText(f"<b>{self.filename}</b>" if self.filename else "<i>no file chosen</i>")

    @Inputs.data
    def set_data(self, data):
        self.data = data
        self._send()

    def browse(self):
        start = self.filename or self.last_dir or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Load subgroups", start, "Pickle (*.pkl);;All files (*)")
        if not path:
            return
        self.filename = path
        self.last_dir = os.path.dirname(path)
        self._update_label()
        self.load()

    def load(self):
        self.Error.clear()
        self.results = None
        if not self.filename:
            self._send()
            return
        try:
            backend.load_mrsd()               # classes must be importable for unpickling
            obj = export.load_pickle(self.filename)
            if not isinstance(obj, SubgroupResults):
                raise TypeError("the file does not contain subgroup results")
            self.results = obj
        except Exception as ex:            # noqa: BLE001 - shown to the user
            self.Error.load_failed(f"{type(ex).__name__}: {ex}")
        self._send()

    def _send(self):
        self.Warning.clear()
        r = self.results
        if r is None:
            self.info_label.setText("")
            self.Outputs.subgroups.send(None)
            self.Outputs.annotated.send(None)
            return
        r = r.copy()
        r.data = None
        if self.data is not None:
            if len(self.data) >= r.n_rows and len(self.data) >= int(r.row_indices.max(initial=-1)) + 1:
                r.data = self.data
            else:
                self.Warning.data_mismatch(len(self.data), r.n_rows)
        self.info_label.setText(f"{len(r)} subgroup(s) from {r.algorithm_name}"
                                + ("" if r.data is not None else " (no data table attached)"))
        self.Outputs.subgroups.send(r)
        self.Outputs.annotated.send(backend.annotate_table(r) if r.data is not None else None)

    def send_report(self):
        self.report_items("File", [("Path", self.filename)])


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWLoadSubgroups).run()
