"""MRSD-Rules widget."""
from Orange.data import Table

from orangecontrib.subgroups.widgets.owbase import OWMRSDBase


class OWMRSDRules(OWMRSDBase):
    name = "MRSD-Rules"
    description = "CN2-SD style beam search of rules with a multi-response WRAcc quality."
    icon = "icons/rules.svg"
    priority = 190
    keywords = ["subgroup discovery", "MRSD", "rules", "CN2-SD", "WRAcc", "beam search"]

    ALGORITHM = "rules"


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.subgroups.datasets import dataset_path
    WidgetPreview(OWMRSDRules).run(Table(dataset_path("planted_shift")))
