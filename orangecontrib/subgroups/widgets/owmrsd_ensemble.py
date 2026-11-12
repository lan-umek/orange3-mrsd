"""MRSD-Ensemble widget."""
from Orange.data import Table

from orangecontrib.subgroups.widgets.owbase import OWMRSDBase


class OWMRSDEnsemble(OWMRSDBase):
    name = "MRSD-Ensemble"
    description = "Consensus of several MRSD algorithms by co-association clustering."
    icon = "icons/ensemble.svg"
    priority = 200
    keywords = ["subgroup discovery", "MRSD", "ensemble", "consensus"]

    ALGORITHM = "ensemble"


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    WidgetPreview(OWMRSDEnsemble).run(Table(dataset_path("planted_shift")))
