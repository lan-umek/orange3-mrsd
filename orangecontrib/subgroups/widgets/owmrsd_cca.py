"""MRSD-CCA widget."""
from Orange.data import Table

from orangecontrib.subgroups.widgets.owbase import OWMRSDBase


class OWMRSDCCA(OWMRSDBase):
    name = "MRSD-CCA"
    description = "Clusters in the regularized canonical-correlation space with exceptional canonical correlation."
    icon = "icons/cca.svg"
    priority = 150
    keywords = ["subgroup discovery", "MRSD", "CCA", "canonical correlation"]

    ALGORITHM = "cca"


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    WidgetPreview(OWMRSDCCA).run(Table(dataset_path("planted_shift")))
