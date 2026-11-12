# Getting started

## Installation

The add-on needs Orange 3.34 or newer.  Install it into the Python
environment that runs Orange:

```
pip install Orange3-Subgroups
```

or, from a clone of the repository, `pip install -e .`.  With the standalone
Orange installer on Windows use Orange's own Python, e.g.

```
"%LOCALAPPDATA%\Programs\Orange\python.exe" -m pip install Orange3-Subgroups
```

Restart Orange; the widgets are in the category **Subgroups**.  Excel export
needs `openpyxl`.

## A first workflow

1. **Subgroup Datasets** → choose *Planted rule*.  The data has three
   independent variables (age, edu, inc) and two dependent variables (y1, y2);
   in the subgroup `edu = high AND age ≥ 50` both dependent variables are
   shifted.
2. Connect it to **MRSD-Rules** and press *Run*.  The first rule in the table
   should be close to the planted one.  Select a row and look at the tabs
   *Rule*, *Prototype* and *Measures*.
3. Connect the *Subgroups* output to **Subgroup Browser** to filter, rename
   and select subgroups, and its *Annotated Data* output to Orange's *Scatter
   Plot*; colour the points by a subgroup column and compare with the meta
   column `planted`.

## Your own data

The algorithm widgets take the **independent variables X from the attributes**
and the **dependent variables Y from the class variables** of the data;
meta attributes are ignored.  Set the roles with **Variable Roles** (or
Orange's *Select Columns*).  In a `.tab` file several columns can be flagged
`class`, and then *File* sets the roles by itself.

## Combining algorithms

Connect several algorithm widgets to **Subgroup Consolidation**.  It computes
measures that are comparable across algorithms and removes near-duplicates;
its output goes to the Browser, Profile, Overlap and Save widgets.

```
Data ─► MRSD-Rules ─┐
     ├► MRSD-HC   ──┼─► Subgroup Consolidation ─► Subgroup Browser ─► Subgroup Profile / Overlap / Save
     └► MRSD-PT   ──┘
```

## Describing a subgroup with a model

Connect an Orange learner (Tree, Naive Bayes, Logistic Regression) to the
*Learner* input of an algorithm widget.  For the selected subgroup the widget
then sends the fitted membership model: connect *Model (X)* to *Tree Viewer*
or to *Nomogram* (Nomogram shows Naive Bayes and logistic regression models;
add *Membership Data (X)* → *Data* to see individual instances).

## Speed

* Permutation tests are off by default except in MRSD-PT.  Search-level
  permutations repeat the whole search; start with a small number.
* Classification-based methods (HC, C2, MV, CCA) are fast with a decision
  tree and slow with a random forest or SVM.
* A widget placed before an update keeps its saved settings; new defaults
  apply to newly placed widgets.

## Troubleshooting

Start Orange from a terminal (`python -m Orange.canvas`); the add-on writes
its progress and any error there in lines beginning with `[MRSD`.  If a widget
lacks inputs or outputs described here, Orange is using a cached description
of the add-on: close Orange, delete the files `*.pck` in Orange's cache folder
(`%LOCALAPPDATA%\Orange\cache\<version>` on Windows) and start it again.
