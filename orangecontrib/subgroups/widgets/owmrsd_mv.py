"""MRSD-MV widget."""
from Orange.data import Table

from orangecontrib.subgroups.widgets.owbase import OWMRSDBase


class OWMRSDMV(OWMRSDBase):
    name = "MRSD-MV"
    description = "Co-regularized multi-view spectral clustering of the X and Y views."
    icon = "icons/mv.svg"
    priority = 170
    keywords = ["subgroup discovery", "MRSD", "multi-view", "spectral clustering"]

    ALGORITHM = "mv"


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    WidgetPreview(OWMRSDMV).run(Table(dataset_path("planted_shift")))
