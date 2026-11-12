"""MRSD-EMM widget."""
from Orange.data import Table

from orangecontrib.subgroups.widgets.owbase import OWMRSDBase


class OWMRSDEMM(OWMRSDBase):
    name = "MRSD-EMM"
    description = "Exceptional model mining: rules in X whose X–Y model differs from the complement."
    icon = "icons/emm.svg"
    priority = 160
    keywords = ["subgroup discovery", "MRSD", "exceptional model mining", "EMM"]

    ALGORITHM = "emm"


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    WidgetPreview(OWMRSDEMM).run(Table(dataset_path("planted_shift")))
