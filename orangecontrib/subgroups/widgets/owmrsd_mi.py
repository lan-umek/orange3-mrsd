"""MRSD-MI widget."""
from Orange.data import Table

from orangecontrib.subgroups.widgets.owbase import OWMRSDBase


class OWMRSDMI(OWMRSDBase):
    name = "MRSD-MI"
    description = "Rules in X with exceptional null-calibrated mutual information between X and Y."
    icon = "icons/mi.svg"
    priority = 180
    keywords = ["subgroup discovery", "MRSD", "mutual information", "KSG"]

    ALGORITHM = "mi"


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    WidgetPreview(OWMRSDMI).run(Table(dataset_path("planted_shift")))
