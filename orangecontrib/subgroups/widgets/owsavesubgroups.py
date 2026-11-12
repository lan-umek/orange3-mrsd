"""
Save Subgroups: write the discovered subgroups to a file
(CSV / Excel / JSON / pickle), optionally with the per-row membership.
"""
from __future__ import annotations

import os
from typing import Optional

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFileDialog, QSizePolicy

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Msg, OWWidget

from orangecontrib.subgroups.backend import SubgroupResults

from orangecontrib.subgroups.export import FORMATS, save_results, text_report  # noqa: F401
from orangecontrib.subgroups.widgets.utils import guard_wheel_later


class OWSaveSubgroups(OWWidget):
    name = "Save Subgroups"
    description = "Save the discovered subgroups to a CSV, Excel, JSON, pickle or text file."
    icon = "icons/save.svg"
    priority = 500
    keywords = ["subgroup discovery", "MRSD", "save", "export", "csv", "excel", "json"]

    want_main_area = False
    resizing_enabled = False

    class Inputs:
        subgroups = Input("Subgroups", SubgroupResults)

    class Error(OWWidget.Error):
        save_failed = Msg("Saving failed: {}")
        no_file = Msg("Choose a file name first.")

    class Information(OWWidget.Information):
        saved = Msg("Saved to {}")

    filename = Setting("", schema_only=True)
    last_dir = Setting("")
    format_index = Setting(0)
    membership = Setting(True)
    auto_save = Setting(False)

    def __init__(self):
        super().__init__()
        guard_wheel_later(self)
        self.results: Optional[SubgroupResults] = None

        box = gui.vBox(self.controlArea, "File")
        self.file_label = gui.widgetLabel(box, "")
        self.file_label.setWordWrap(True)
        self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        gui.comboBox(box, self, "format_index", items=[lbl for _, lbl, _ in FORMATS], label="Format",
                     orientation=Qt.Horizontal, callback=self._format_changed)
        gui.checkBox(box, self, "membership", "Include per-row membership (CSV: extra file; Excel/JSON: sheet / rows)")
        gui.checkBox(box, self, "auto_save", "Save automatically when new subgroups arrive")
        row = gui.hBox(box)
        gui.button(row, self, "Save as …", callback=self.save_as)
        self.save_button = gui.button(row, self, "Save", callback=self.save)
        self.info_label = gui.widgetLabel(self.controlArea, "No subgroups on input.")
        self._update_file_label()
        self.setMinimumWidth(420)

    def _format_changed(self):
        if self.filename:
            root, _ = os.path.splitext(self.filename)
            self.filename = root + FORMATS[self.format_index][2]
            self._update_file_label()

    def _update_file_label(self):
        self.file_label.setText(f"<b>{self.filename}</b>" if self.filename else "<i>no file chosen</i>")

    @Inputs.subgroups
    def set_subgroups(self, results: Optional[SubgroupResults]):
        self.results = results
        self.Information.saved.clear()
        if results is None:
            self.info_label.setText("No subgroups on input.")
            return
        self.info_label.setText(f"{len(results)} subgroup(s) from {results.algorithm_name}.")
        if self.auto_save and self.filename:
            self.save()

    def save_as(self):
        fmt_key, fmt_label, ext = FORMATS[self.format_index]
        start = self.filename or os.path.join(self.last_dir or os.path.expanduser("~"), "subgroups" + ext)
        filters = ";;".join(lbl for _, lbl, _ in FORMATS)
        path, chosen = QFileDialog.getSaveFileName(self, "Save subgroups", start, filters, fmt_label)
        if not path:
            return
        for k, (_, lbl, e) in enumerate(FORMATS):
            if lbl == chosen:
                self.format_index = k
                if not path.lower().endswith(e):
                    path += e
        self.filename = path
        self.last_dir = os.path.dirname(path)
        self._update_file_label()
        self.save()

    def save(self):
        self.Error.clear()
        self.Information.saved.clear()
        if not self.filename:
            self.Error.no_file()
            return
        if self.results is None:
            return
        try:
            save_results(self.results, self.filename, FORMATS[self.format_index][0], self.membership)
        except Exception as ex:          # noqa: BLE001 - shown to the user
            self.Error.save_failed(f"{type(ex).__name__}: {ex}")
            return
        self.Information.saved(os.path.basename(self.filename))

    def send_report(self):
        self.report_items("Save", [("File", self.filename), ("Format", FORMATS[self.format_index][1])])


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWSaveSubgroups).run()
