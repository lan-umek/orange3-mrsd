# Orange3-MRSD

An [Orange](https://orangedatamining.com) add-on for **Multiple Responses
Subgroup Discovery (MRSD)**: finding groups of instances, described by the
independent variables **X**, in which several dependent variables **Y** – or
their relation to X – behave differently from the rest of the data.

The add-on wraps the MRSD Python package (vendored in
`orangecontrib/subgroups/mrsd`) and adds widgets for setting variable roles,
running the algorithms, inspecting, combining, visualising and saving the
subgroups.

## Widgets (category *Subgroups*)

| | |
|---|---|
| **Data** | Subgroup Datasets, Variable Roles |
| **Algorithms** | MRSD-PT (partitioning and tables), MRSD-HC (hierarchical clustering and classification), MRSD-C2 (classification in both spaces: swapping, semi-supervised, ranking), MRSD-CCA, MRSD-EMM, MRSD-MV (multi-view), MRSD-MI, MRSD-Rules, MRSD-Ensemble |
| **Results** | Subgroup Browser, Subgroup Consolidation, Subgroup Classifier, Subgroup Profile, Subgroup Overlap, Save Subgroups, Load Subgroups |

Every algorithm widget shows the subgroups with their rule in X and prototype
in Y, has algorithm-specific views of the selected subgroup (contingency
table, dendrograms, classifiers, correlations …), accepts Orange learners and
sends the selected subgroup's rows, membership data and membership model on
to other Orange widgets.

Documentation: [doc/index.md](doc/index.md) – start with
[Getting started](doc/getting-started.md).

## Installation

Orange 3.34 or newer is required.

```
pip install Orange3-MRSD        # into the environment that runs Orange
```

or from a clone: `pip install -e .`.  Excel export additionally needs
`openpyxl`.

## Development

```
python -m pytest tests -q        # backend tests run without Orange; widget tests need Orange and QT_QPA_PLATFORM=offscreen
python tools/make_docs.py        # regenerate doc/ (parameter tables come from params.py)
python tools/make_datasets.py    # regenerate the bundled datasets
python tools/sync_mrsd.py        # copy the MRSD package from ../subgroups into orangecontrib/subgroups/mrsd
```

* The widgets build their parameter GUI from `orangecontrib/subgroups/params.py`;
  `tests/test_params.py` fails if a field of an algorithm's `Config` dataclass is
  not covered or a parameter name clashes with a widget attribute.
* Orange caches the widget descriptions of an add-on and refreshes them only
  when `orangecontrib/subgroups/widgets/__init__.py` changes: after changing a
  widget's inputs, outputs or metadata bump `WIDGET_REGISTRY_STAMP` there.

## Status

The algorithms and the non-GUI part of the add-on are covered by tests.  Most widgets have been tried interactively on Orange 3.40 (Windows); some of
the newer views (e.g. the dendrograms of MRSD-MV, cell selection in Subgroup
Overlap) and other Orange versions and platforms have not been checked yet.

## License

MIT – see [LICENSE](LICENSE).
