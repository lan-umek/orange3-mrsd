"""MRSD-HC widget."""
from Orange.data import Table

from orangecontrib.subgroups.widgets.owbase import OWMRSDBase


class OWMRSDHC(OWMRSDBase):
    name = "MRSD-HC"
    description = ("Hierarchical clustering and classification: nodes of the dendrogram of Y that a "
                   "classifier separates from the rest using X.")
    icon = "icons/hc.svg"
    priority = 110
    keywords = ["subgroup discovery", "MRSD", "hierarchical", "clustering", "dendrogram"]

    ALGORITHM = "hc"


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    WidgetPreview(OWMRSDHC).run(Table(dataset_path("planted_shift")))
