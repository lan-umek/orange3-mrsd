"""MRSD-C2 widget: the three C2 methods (swapping, semi-supervised, ranking) in one widget."""
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting

from orangecontrib.subgroups import backend
from orangecontrib.subgroups.widgets.owbase import OWMRSDBase


class OWMRSDC2(OWMRSDBase):
    name = "MRSD-C2"
    description = ("Classification in both spaces: subgroups separable both in X and in Y (out-of-fold AUC): "
                   "swapping local search, semi-supervised self-training or instance ranking.")
    icon = "icons/c2.svg"
    priority = 120
    keywords = ["subgroup discovery", "MRSD", "C2", "swapping", "semi-supervised", "ranking", "classification"]

    METHODS = [("swapping", "Swapping – local search moving instances"),
               ("semisupervised", "Semi-supervised – self-training on membership probabilities"),
               ("ranking", "Ranking – top instances of aggregated rankings")]
    ALGORITHM = "swapping"
    ALGORITHMS = [m for m, _ in METHODS]

    method = Setting(0)

    @property
    def algorithm_id(self) -> str:
        return self.METHODS[self.method][0]

    def _build_method_selector(self, box):
        gui.radioButtons(box, self, "method", [lbl for _, lbl in self.METHODS], callback=self._method_changed)

    def _method_changed(self):
        self.description_label.setText(backend.ALGORITHM_BY_ID[self.algorithm_id].description)
        self._update_param_visibility()
        self._learners_changed()
        self.Information.initial_from_subgroups.clear()
        if self.input_results is not None and len(self.input_results):
            self.Information.initial_from_subgroups(len(self.input_results))


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    WidgetPreview(OWMRSDC2).run(Table(dataset_path("planted_shift")))
