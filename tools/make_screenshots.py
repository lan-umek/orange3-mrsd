"""
Make the screenshots of the widgets for the documentation.

Run it with the Python that runs Orange, from the root of the add-on:

    python tools/make_screenshots.py            # all widgets
    python tools/make_screenshots.py rules hc   # only some (names as in the list printed at the start)

The script creates every widget with its default settings (your saved
settings are not used), feeds it a bundled dataset, runs it, and saves
pictures of the widget (for the algorithm widgets one per tab) to
doc/widgets/images/.  At the end doc/ is regenerated so that the pages
include the pictures.  Nothing is shown on screen for long; do not close the
windows that flash up.  A widget that fails is reported and skipped.
"""
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "doc", "widgets", "images")

from AnyQt.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv[:1])

from Orange.data import Table  # noqa: E402

from orangecontrib.subgroups import backend, params  # noqa: E402
from orangecontrib.subgroups.datasets import dataset_path  # noqa: E402


def pump(seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def create(cls):
    """A widget with default settings (not the user's saved ones)."""
    w = cls.__new__(cls, signal_manager=None, stored_settings={})
    w.__init__()
    return w


def grab(widget, name, size=(1150, 720)):
    widget.resize(*size)
    widget.show()
    widget.raise_()
    pump(0.6)
    path = os.path.join(OUT, name + ".png")
    widget.grab().save(path)
    print("   saved", os.path.relpath(path, ROOT))


def slug(text):
    return text.lower().replace(" ", "-")


def run_algorithm_widget(cls, file, data, settings=None, timeout=900):
    w = create(cls)
    w.use_process = False            # simpler here: run inside this process
    w.filter_measure = False         # show all subgroups in the pictures
    for k, v in (settings or {}).items():
        setattr(w, k, v)
    if hasattr(w, "_method_changed") and settings and "method" in settings:
        w._method_changed()
    w.set_data(data)
    w.handleNewSignals()
    w.run()
    start = time.time()
    while w.results is None and time.time() - start < timeout:
        pump(0.2)
        if w.Error.run_failed.is_shown() or w.Error.invalid_params.is_shown():
            raise RuntimeError("the algorithm reported an error (see the console)")
    if w.results is None:
        raise RuntimeError(f"no result after {timeout} s")
    pump(0.5)
    w.tabs.setCurrentIndex(0)
    grab(w, file)
    for k in range(1, w.tabs.count()):
        w.tabs.setCurrentIndex(k)
        pump(1.0)
        grab(w, f"{file}-{slug(w.tabs.tabText(k))}")
    results = w.results
    w.close()
    w.onDeleteWidget()
    return results


def main(only):
    os.makedirs(OUT, exist_ok=True)
    from orangecontrib.subgroups.widgets.owmrsd_pt import OWMRSDPT
    from orangecontrib.subgroups.widgets.owmrsd_hc import OWMRSDHC
    from orangecontrib.subgroups.widgets.owmrsd_c2 import OWMRSDC2
    from orangecontrib.subgroups.widgets.owmrsd_cca import OWMRSDCCA
    from orangecontrib.subgroups.widgets.owmrsd_emm import OWMRSDEMM
    from orangecontrib.subgroups.widgets.owmrsd_mv import OWMRSDMV
    from orangecontrib.subgroups.widgets.owmrsd_mi import OWMRSDMI
    from orangecontrib.subgroups.widgets.owmrsd_rules import OWMRSDRules
    from orangecontrib.subgroups.widgets.owmrsd_ensemble import OWMRSDEnsemble
    from orangecontrib.subgroups.widgets.owsubgroupdatasets import OWSubgroupDatasets
    from orangecontrib.subgroups.widgets.owvariableroles import OWVariableRoles
    from orangecontrib.subgroups.widgets.owsubgroupbrowser import OWSubgroupBrowser
    from orangecontrib.subgroups.widgets.owsubgroupconsolidation import OWSubgroupConsolidation
    from orangecontrib.subgroups.widgets.owsubgroupclassifier import OWSubgroupClassifier
    from orangecontrib.subgroups.widgets.owsubgroupprofile import OWSubgroupProfile
    from orangecontrib.subgroups.widgets.owsubgroupoverlap import OWSubgroupOverlap
    from orangecontrib.subgroups.widgets.owsavesubgroups import OWSaveSubgroups
    from orangecontrib.subgroups.widgets.owloadsubgroups import OWLoadSubgroups

    data = Table(dataset_path("customers"))
    small = Table(dataset_path("planted_rule"))
    results = {}

    algorithms = [
        ("rules", OWMRSDRules, "mrsd-rules", data, {}),
        ("pt", OWMRSDPT, "mrsd-pt", data, {"n_permutations": 200, "k_values": "2, 3, 4"}),
        ("hc", OWMRSDHC, "mrsd-hc", data, {}),
        ("c2", OWMRSDC2, "mrsd-c2", small, {"method": 2}),
        ("cca", OWMRSDCCA, "mrsd-cca", data, {}),
        ("emm", OWMRSDEMM, "mrsd-emm", data, {}),
        ("mv", OWMRSDMV, "mrsd-mv", small, {"method": "dual_dendrogram"}),
        ("mi", OWMRSDMI, "mrsd-mi", small, {}),
        ("ensemble", OWMRSDEnsemble, "mrsd-ensemble", small, {"algorithms": ["pt", "hc", "rules"]}),
    ]
    names = [a[0] for a in algorithms] + ["datasets", "roles", "browser", "consolidation", "classifier", "profile",
                                         "overlap", "save", "load"]
    print("widgets:", ", ".join(names))
    wanted = set(only) if only else set(names)
    failed = []

    def attempt(name, func):
        if name not in wanted:
            return
        print(f"[{name}]")
        try:
            func()
        except Exception:      # noqa: BLE001 - report and continue with the next widget
            failed.append(name)
            traceback.print_exc()

    for name, cls, file, table, settings in algorithms:
        def job(cls=cls, file=file, table=table, settings=settings, name=name):
            results[name] = run_algorithm_widget(cls, file, table, settings)
        attempt(name, job)

    def simple(cls, file, feed=None, size=(700, 520)):
        def job():
            w = create(cls)
            if feed is not None:
                feed(w)
            pump(0.5)
            grab(w, file, size)
            w.close()
        return job

    def some_results(table=data):
        """Results for the result widgets: reuse a run from above or compute one."""
        for key in ("rules", "hc", "pt"):
            if key in results and results[key] is not None and results[key].data is not None \
                    and len(results[key].data) == len(table):
                return results[key]
        cfg = params.widget_config("rules", params.setting_defaults("rules"))
        return backend.run_algorithm("rules", backend.table_to_dataframe(table), *backend.table_roles(table), cfg,
                                     data=table)

    def second_results(table=data):
        if "hc" in results and results["hc"] is not None and len(results["hc"].data) == len(table):
            return results["hc"]
        cfg = params.widget_config("emm", params.setting_defaults("emm"))
        return backend.run_algorithm("emm", backend.table_to_dataframe(table), *backend.table_roles(table), cfg,
                                     data=table)

    attempt("datasets", simple(OWSubgroupDatasets, "subgroup-datasets", size=(520, 560)))

    def feed_roles(w):
        w.set_data(data)
        w.handleNewSignals()
    attempt("roles", simple(OWVariableRoles, "variable-roles", feed_roles, (720, 520)))

    def feed_browser(w):
        w.set_subgroups(some_results(), 1)
        w.handleNewSignals()
        w.view.selectRow(0)
    attempt("browser", simple(OWSubgroupBrowser, "subgroup-browser", feed_browser, (1150, 720)))

    def feed_consolidation(w):
        w.set_subgroups(some_results(), 1)
        w.set_subgroups(second_results(), 2)
        w.handleNewSignals()
    attempt("consolidation", simple(OWSubgroupConsolidation, "subgroup-consolidation", feed_consolidation, (1150, 720)))

    def feed_classifier(w):
        w.set_subgroups(some_results())
        w.handleNewSignals()
    attempt("classifier", simple(OWSubgroupClassifier, "subgroup-classifier", feed_classifier, (1000, 650)))

    attempt("profile", simple(OWSubgroupProfile, "subgroup-profile", lambda w: w.set_subgroups(some_results()),
                              (1000, 650)))
    attempt("overlap", simple(OWSubgroupOverlap, "subgroup-overlap",
                              lambda w: w.set_subgroups(some_results().subset(range(min(10, len(some_results()))))),
                              (900, 700)))
    attempt("save", simple(OWSaveSubgroups, "save-subgroups", lambda w: w.set_subgroups(some_results()), (520, 300)))
    attempt("load", simple(OWLoadSubgroups, "load-subgroups", size=(480, 220)))

    print()
    if failed:
        print("FAILED:", ", ".join(failed), "- send the tracebacks above")
    try:
        import make_docs
        make_docs.main()
    except Exception:      # noqa: BLE001
        traceback.print_exc()
    print("done; pictures are in", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1:])
