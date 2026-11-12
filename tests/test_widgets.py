"""Widget tests – need Orange and a Qt platform (set QT_QPA_PLATFORM=offscreen)."""
import pytest

Orange = pytest.importorskip("Orange")
from Orange.data import Table                               # noqa: E402
from Orange.widgets.tests.base import WidgetTest             # noqa: E402

from orangecontrib.subgroups import backend                  # noqa: E402
from orangecontrib.subgroups.datasets import dataset_path    # noqa: E402
from orangecontrib.subgroups.widgets.owmrsd_rules import OWMRSDRules          # noqa: E402
from orangecontrib.subgroups.widgets.owsubgroupbrowser import OWSubgroupBrowser  # noqa: E402
from orangecontrib.subgroups.widgets.owvariableroles import OWVariableRoles   # noqa: E402
from orangecontrib.subgroups.widgets.owsubgroupdatasets import OWSubgroupDatasets  # noqa: E402


class TestDatasets(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSubgroupDatasets)

    def test_output(self):
        out = self.get_output(self.widget.Outputs.data)
        self.assertIsNotNone(out)
        self.assertGreaterEqual(len(out.domain.class_vars), 2)


class TestVariableRoles(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWVariableRoles)
        self.data = Table(dataset_path("planted_rule"))

    def test_roles_roundtrip(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        out = self.get_output(self.widget.Outputs.data)
        self.assertEqual([v.name for v in out.domain.class_vars], ["y1", "y2"])
        self.widget._swap()
        out = self.get_output(self.widget.Outputs.data)
        self.assertEqual([v.name for v in out.domain.attributes], ["y1", "y2"])


class TestRulesWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWMRSDRules)
        self.data = Table(dataset_path("planted_rule"))

    def test_run(self):
        w = self.widget
        w.n_search_permutations = 5
        self.send_signal(w.Inputs.data, self.data)
        w.run()
        self.wait_until_finished(timeout=120000)
        res = self.get_output(w.Outputs.subgroups)
        self.assertIsNotNone(res)
        self.assertGreater(len(res), 0)
        annotated = self.get_output(w.Outputs.annotated)
        self.assertEqual(len(annotated.domain.metas), len(self.data.domain.metas) + len(res))

    def test_no_targets(self):
        from Orange.data import Domain
        d = self.data.transform(Domain(self.data.domain.variables))
        self.send_signal(self.widget.Inputs.data, d)
        self.assertTrue(self.widget.Error.no_targets.is_shown())


class TestBrowser(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSubgroupBrowser)
        data = Table(dataset_path("planted_rule"))
        from orangecontrib.subgroups.params import setting_defaults, widget_config
        cfg = widget_config("rules", {**setting_defaults("rules"), "n_search_permutations": 5})
        self.results = backend.run_algorithm("rules", backend.table_to_dataframe(data), *backend.table_roles(data),
                                             cfg, data=data)

    def test_outputs(self):
        w = self.widget
        self.send_signal(w.Inputs.subgroups, self.results, 1)
        out = self.get_output(w.Outputs.subgroups)
        self.assertEqual(len(out), len(self.results))
        w.model.set_include([0], False)
        w.commit.now()
        out = self.get_output(w.Outputs.subgroups)
        self.assertEqual(len(out), len(self.results) - 1)
        self.assertIsNotNone(self.get_output(w.Outputs.annotated))
