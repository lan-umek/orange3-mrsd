"""Widget that loads the example datasets bundled with the add-on."""
from __future__ import annotations

import os

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QListWidget, QListWidgetItem

from Orange.data import Domain, Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Msg, Output, OWWidget

from orangecontrib.subgroups import datasets
from orangecontrib.subgroups.widgets.utils import guard_wheel_later


class OWSubgroupDatasets(OWWidget):
    name = "Subgroup Datasets"
    description = "Example datasets with several dependent variables and planted subgroups."
    icon = "icons/datasets.svg"
    priority = 10
    keywords = ["subgroup discovery", "MRSD", "datasets", "examples", "load"]

    want_main_area = False
    resizing_enabled = True

    class Outputs:
        data = Output("Data", Table)

    class Error(OWWidget.Error):
        load_failed = Msg("Cannot load the dataset: {}")

    selected = Setting("planted_shift")
    keep_ground_truth = Setting(True)

    def __init__(self):
        super().__init__()
        guard_wheel_later(self)
        self.data = None

        box = gui.vBox(self.controlArea, "Datasets")
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SingleSelection)
        for name in datasets.available():
            item = QListWidgetItem(datasets.DATASETS[name][0])
            item.setData(Qt.UserRole, name)
            self.list.addItem(item)
            if name == self.selected:
                self.list.setCurrentItem(item)
        if self.list.currentItem() is None and self.list.count():
            self.list.setCurrentRow(0)
        self.list.currentItemChanged.connect(self._selection_changed)
        box.layout().addWidget(self.list)

        self.description_label = gui.widgetLabel(gui.vBox(self.controlArea, "Description"), "")
        self.description_label.setWordWrap(True)

        gui.checkBox(self.controlArea, self, "keep_ground_truth",
                     "Keep the 'planted' ground-truth meta column",
                     tooltip="The algorithms ignore meta columns; the column lets you check the results "
                             "(e.g. in the Subgroup Browser or Scatter Plot).", callback=self.commit)
        gui.widgetLabel(
            self.controlArea,
            "The files are also available by name in the File widget and as\n"
            "Table('planted_shift') etc.  Dependent variables are flagged as\n"
            "class variables, so the roles are set automatically.")
        gui.button(self.controlArea, self, "Reload", callback=self.commit)
        self._selection_changed()

    def _selection_changed(self):
        item = self.list.currentItem()
        if item is None:
            return
        self.selected = item.data(Qt.UserRole)
        self.description_label.setText(datasets.DATASETS[self.selected][1])
        self.commit()

    def commit(self):
        self.Error.load_failed.clear()
        path = datasets.dataset_path(self.selected)
        if not os.path.exists(path):
            self.Error.load_failed(f"{path} does not exist")
            self.Outputs.data.send(None)
            return
        try:
            data = Table(path)
        except Exception as ex:           # noqa: BLE001 - shown to the user
            self.Error.load_failed(str(ex))
            self.Outputs.data.send(None)
            return
        data.name = self.selected
        if not self.keep_ground_truth:
            dom = data.domain
            data = data.transform(Domain(dom.attributes, dom.class_vars))
        self.data = data
        self.Outputs.data.send(data)

    def send_report(self):
        self.report_items("Dataset", [("Name", self.selected),
                                      ("Description", datasets.DATASETS.get(self.selected, ("", ""))[1])])


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWSubgroupDatasets).run()
