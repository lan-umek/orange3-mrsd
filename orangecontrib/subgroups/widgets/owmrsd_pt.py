"""MRSD-PT widget."""
from Orange.data import Table

from orangecontrib.subgroups.widgets.owbase import OWMRSDBase


class OWMRSDPT(OWMRSDBase):
    name = "MRSD-PT"
    description = "Partitioning and tables: subgroups from pairs of partitions of X and Y (contingency cells, f-divergence permutation test)."
    icon = "icons/pt.svg"
    priority = 100
    keywords = ["subgroup discovery", "MRSD", "partition", "contingency", "f-divergence", "permutation"]

    ALGORITHM = "pt"


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    WidgetPreview(OWMRSDPT).run(Table(dataset_path("planted_shift")))
