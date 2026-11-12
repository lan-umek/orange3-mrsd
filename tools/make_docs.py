"""
Generate the documentation pages of the add-on (doc/*.md, doc/widgets/*.md).

The parameter tables of the algorithm widgets are produced from
``orangecontrib/subgroups/params.py`` so that they cannot drift from the GUI;
the prose is kept in this file.

    python tools/make_docs.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orangecontrib.subgroups import datasets, params  # noqa: E402

DOC = os.path.join(ROOT, "doc")
WID = os.path.join(DOC, "widgets")

# ---------------------------------------------------------------------------
# shared text of the algorithm widgets
# ---------------------------------------------------------------------------

ALG_INPUTS = """\
| input | type | meaning |
|---|---|---|
| **Data** | Table | independent variables X = attributes, dependent variables Y = class variables; meta attributes are ignored |
| **Subgroups** | Subgroups | optional. If no Data is connected, the data of these subgroups is used. In MRSD-C2 the subgroups are also the starting subgroups of the search |
| **Learner** | Learner (several) | optional Orange learners (Tree, Naive Bayes, Logistic Regression …). In widgets with a *Classifier* tab they describe the selected subgroup and provide the *Model* outputs; in the classification-based algorithms they also replace the built-in classifiers of the search |
"""

ALG_OUTPUTS = """\
| output | type | meaning |
|---|---|---|
| **Subgroups** | Subgroups | the subgroups shown in the table (after the filters of the *Results* box) |
| **Data** | Table | the input data, unchanged – for chaining algorithm widgets |
| **Annotated Data** | Table | the data with one Yes/No meta column per shown subgroup |
| **Subgroups Table** | Table | one row per subgroup: rule, prototype in Y, size, quality, p-values, statistics of the dependent variables |
| **Selected Subgroup Data** | Table | the rows of the subgroup selected in the table (or chosen in an inspector) |
| **Membership Data (X)** / **(Y)** | Table | all rows with a class `in <subgroup>` (no / yes) and X, respectively Y, as features |
| **Model (X)** / **(Y)** | Model | the membership model of the selected subgroup fitted with a connected Orange learner (for Tree Viewer, Nomogram …) |
"""

ALG_COMMON = """\
## Running

*Run* (top and bottom of the control area) starts the computation in a
separate process: Orange stays responsive, other widgets can run at the same
time and *Stop* really stops the run.  *Run in a separate process* can be
switched off if the process cannot be started on your system.  Progress is
written to the console Orange was started from (lines beginning with `[MRSD`).

Rows with missing values in X or Y are removed when *Remove rows with missing
values* is ticked (default).

## Results box

* **Measure** – the algorithm's quality or any numeric measure it records.
* **Show only measure ≥** – filter on the chosen measure.
* **Show only adjusted p <** – filter on the family-wise adjusted p-value
  (only meaningful when permutations > 0).
* **Sort by** – the measure (default), p-values, size or order of discovery.

The filters apply to the table and to the outputs.

## Permutation inference

Permutation tests are **off by default** (Permutations = 0){pt_note}.  When
switched on, the raw p-value, the family-wise adjusted p-value (Westfall–Young
max-T, or a re-run of the whole search on permuted data) and the
Benjamini–Hochberg value are added.  Search-level permutations repeat the
whole search and can be slow.

## Tabs of the main area

* **Subgroups** – the table: rule / description in X, prototype in Y (means
  and modal levels), size, quality, p-values and the difference of every
  dependent variable from the whole data.
{tabs}* **Prototype** – descriptive statistics of every X and Y variable in the
  subgroup and in its complement (mean, SD, median, IQR or level shares),
  effect sizes, a parametric and a non-parametric test per variable
  (Welch t / Mann–Whitney U; χ² / Fisher exact or Monte-Carlo χ²) and a bar
  chart with several display variants.  The p-values are descriptive: the
  subgroup was selected on the same data.
* **Measures** – everything the algorithm recorded about the subgroup.
"""

CLASSIFIERS = """\
## Classifiers

Without a connected learner the built-in classifiers chosen in the widget are
used.  When at least one Orange learner is connected to the *Learner* input,
*Use the connected Orange learner(s)* appears (ticked) and the built-in
choice is hidden{extra}.  Slow learners (random forest, SVM) make the search
slow, because every candidate is scored by cross-validation.
"""

ALGORITHMS = {
    "pt": dict(
        title="MRSD-PT", file="mrsd-pt", name="Partitioning and Tables",
        lead="Subgroups are cells of the contingency tables of pairs of partitions of X and of Y.",
        how="""\
Several partitions of X and of Y are built (clustering with several numbers of
clusters, and the natural partition by the combinations of categorical
variables).  For every pair (partition of X, partition of Y) the contingency
table is formed; a cell that holds more instances than expected under
independence is a candidate subgroup: instances that are alike in X and alike
in Y.  The cell statistic is the cell's contribution to an f-divergence (χ² by
default) or the squared adjusted residual; significance is assessed by
permuting the rows (max-T over all cells).""",
        tabs="""\
* **Contingency table** – the table of the partition pair of the selected
  subgroup: observed count, expected count and adjusted residual in every
  cell; below it the prototypes (means / modal levels, medoid) of the clusters
  of X and of Y.  Click a cell that is a reported subgroup to select it; click
  any other cell to send it on as an ad-hoc subgroup (*Keep this cell as a
  subgroup* adds it to the results).
""", classifiers=None, pt=True),
    "hc": dict(
        title="MRSD-HC", file="mrsd-hc", name="Hierarchical Clustering and Classification",
        lead="Subgroups are nodes of the dendrogram of Y that a classifier can tell apart from the rest using X.",
        how="""\
The instances are clustered hierarchically on the dependent variables Y.
Every node of the dendrogram (within the size limits) is a candidate: a group
of instances with similar responses.  Its quality is the out-of-fold AUC of
predicting membership of the node from X – a group that is homogeneous in Y
and recognisable from X.  With several classifiers the quality is the best of
them.  The optional test permutes the rows of X (the nodes come from Y only)
and compares the AUC; the adjustment runs over all examined nodes.""",
        tabs="""\
* **Dendrogram** – the dendrogram of Y with the node of the selected subgroup
  highlighted.  Click a merge point to choose another node; a node that is not
  a reported subgroup is sent on as an ad-hoc subgroup and can be kept.
* **Classifier** – the membership classifier in X: out-of-fold AUC, training
  AUC (optimistic) and a readable model (tree rules, coefficients,
  importances).
""", classifiers="; MRSD-HC scores every node with all connected learners and takes the best. *Also use the built-in classifiers ticked below* adds the built-in ones"),
    "c2": dict(
        title="MRSD-C2", file="mrsd-c2", name="Classification in Both Spaces",
        lead="Subgroups whose membership can be predicted both from X and from Y.",
        how="""\
A subgroup is scored by two out-of-fold AUCs: predicting its membership from
X (q_X) and from Y (q_Y), aggregated (minimum by default).  Three search
methods share this score:

* **Swapping** – local search from starting subgroups, moving instances in
  and out while the score improves.
* **Semi-supervised** – self-training: membership probabilities from X and Y
  are combined and thresholded to give the next version of the subgroup.
* **Ranking** – instances are ranked by their membership probabilities in X
  and in Y; the top-s instances of the aggregated ranking form candidates of
  several sizes.

Starting subgroups come from clustering or at random, or from the *Subgroups*
input (e.g. the output of another algorithm).""",
        tabs="""\
* **Classifier** – the membership classifiers in X and in Y side by side.
""", classifiers=": *Learner for X* and *Learner for Y* choose which connected learner is used in which space"),
    "cca": dict(
        title="MRSD-CCA", file="mrsd-cca", name="Canonical Correlation Analysis",
        lead="Subgroups in which the canonical correlation between X and Y differs from the rest.",
        how="""\
A regularized canonical correlation analysis of X and Y is computed on all
rows and the instances are clustered in the canonical space.  A cluster is
scored by Fisher's z of the difference between the first canonical correlation
inside the cluster and in its complement.  The cluster is described post hoc
by a rule in X (a decision tree).  The classification qualities are reported
only; they are not part of the score.""",
        tabs="""\
* **Canonical space** – scatter of the first canonical variates of X and Y
  with the subgroup and the complement, their regression lines and the table
  of canonical weights.
* **Classifier** – membership classifiers in X and in Y.
""", classifiers=": *Learner for X* and *Learner for Y* choose which connected learner is used in which space"),
    "emm": dict(
        title="MRSD-EMM", file="mrsd-emm", name="Exceptional Model Mining",
        lead="Rules in X under which the relation between X and Y differs from the complement.",
        how="""\
Candidates are conjunctive rules in X found by beam search (or clusters, or
random subsets).  A candidate is scored by how exceptional its X–Y model is
compared with the complement: differences of correlations (Fisher z),
of regression slopes or of R².  Scores are normalized by their sampling error
so that small subgroups are not favoured.""",
        tabs="""\
* **Correlations** – correlations of every numeric X–Y pair inside the
  subgroup and in the complement with the z of their difference; the selected
  pair is plotted with both regression lines.
""", classifiers=None),
    "mv": dict(
        title="MRSD-MV", file="mrsd-mv", name="Multi-View",
        lead="X and Y are two views of the same instances; subgroups are coherent in both.",
        how="""\
Six methods are available.

* **Co-regularized spectral** (default), **averaged affinities**,
  **co-training**, **late fusion** – multi-view clusterings; every cluster is
  a candidate.  Quality = ½ · (agreement of the two single-view clusterings
  inside the cluster minus outside) + ½ · min(q_X, q_Y).
* **Similarity excess** – no clustering.  With the similarity matrices W_X and
  W_Y, the pairwise excess is B = W_X ∘ (H·W_Y·H) (Y kernel centred); a
  subgroup maximises Q(G) = Σ B over pairs in G / |G|^a (leading eigenvector,
  best prefix, single-instance moves; further subgroups by deflation).
* **Dual dendrogram** – hierarchical clustering of X and of Y; a candidate is
  the intersection of a node of each tree, scored by the adjusted residual of
  that cell of the 2 × 2 table; only local maxima over parent / child moves
  are kept.

For the last two methods the X variables can be weighted by their relevance
for Y (kernel alignment), so that unrelated variables do not dilute the X
similarity, and their scores are not calibrated against selection – judge
them with the permutation test or the common measures of *Subgroup
Consolidation*.  On the bundled data with planted subgroups these two
methods recover the planted subgroups only partly.""",
        tabs="""\
* **Dendrograms** – the dendrograms of X and of Y side by side.  The rows of
  the selected subgroup are marked under the leaves of both trees.  Click a
  merge point in one tree to select a node: its rows are marked under the
  leaves of the other tree, showing where the same instances lie in the other
  space.  With a node selected in each tree the intersection can be sent on
  (*Send selection as subgroup*) and kept.
* **Classifier** – membership classifiers in X and in Y.
""", classifiers=": *Learner for X* and *Learner for Y* choose which connected learner is used in which space"),
    "mi": dict(
        title="MRSD-MI", file="mrsd-mi", name="Mutual Information",
        lead="Rules in X under which X and Y are more dependent than expected.",
        how="""\
Candidates are rules in X (beam search) or clusters.  Inside a candidate the
mutual information between X and Y is estimated and standardized against
shuffles of the Y rows within the candidate (z-score), which removes the
dependence of the raw estimate on the subgroup size.  The default estimator
(Gao et al., 2017) is consistent for numeric, categorical and mixed variables;
the KSG estimator (numeric) and a binning estimator are also available.  The
method looks for a changed *dependence*, not for a shift of the mean.""",
        tabs="""\
* **Mutual information** – the joint and every pairwise X–Y mutual
  information inside the subgroup and in the complement.
""", classifiers=None),
    "rules": dict(
        title="MRSD-Rules", file="mrsd-rules", name="Rules (CN2-SD)",
        lead="Rules in X with an exceptional distribution of the dependent variables.",
        how="""\
A CN2-SD style beam search over conjunctive rules in X.  The quality is a
multi-response weighted relative accuracy: the (Mahalanobis, dispersion or
directed) shift of Y under the rule times a coverage factor.  Weighted
covering lowers the weight of covered instances so that successive rules
describe different parts of the data.""",
        tabs="""\
* **Rule** – the conditions of the rule with the coverage of each condition
  alone and of the conjunction up to it.
""", classifiers=None),
    "ensemble": dict(
        title="MRSD-Ensemble", file="mrsd-ensemble", name="Consensus",
        lead="Runs several of the other algorithms and fuses their subgroups.",
        how="""\
The chosen member algorithms are run with reduced ("fast") settings, which can
be overridden per member (*Member overrides*); their subgroups are
fused by co-association (how often two instances fall in the same subgroup)
into consensus subgroups supported by several algorithms.  For combining the
results of algorithm widgets you have configured yourself, use *Subgroup
Consolidation* instead.""",
        tabs="""\
* **Consensus** – the member algorithms supporting the subgroup, their ranks
  and Jaccard similarity with the consensus.
""", classifiers=None),
}

OTHER = {
    "subgroup-datasets": ("Subgroup Datasets", """\
Loads the example datasets bundled with the add-on.

**Outputs:** *Data* – the chosen dataset with the dependent variables as class
variables.

Every dataset has several dependent variables and a meta column `planted`
with the ground truth of the planted subgroup(s); the algorithms ignore meta
columns, so the column can be used to check the results (e.g. as colour in a
Scatter Plot of the *Annotated Data*).  The files are also found by name by
Orange's *File* widget, which sets the roles automatically.

{datasets}
"""),
    "variable-roles": ("Variable Roles", """\
Sets which variables are independent (X), dependent (Y) or ignored.

**Inputs:** *Data*.  **Outputs:** *Data* with X as attributes, Y as class
variables and the ignored variables as metas – the form the algorithm widgets
expect.

Move variables between the three lists with the buttons or by dragging;
*X ⇄ Y* swaps the two roles and *Reset* restores the roles of the input.
Orange's *Select Columns* widget can be used instead.
"""),
    "subgroup-browser": ("Subgroup Browser", """\
Inspect, filter, rename, remove and select subgroups from one or several
algorithm widgets.

**Inputs:** *Subgroups* (several).
**Outputs:** *Subgroups* (the included ones that pass the filters),
*Selected Subgroups*, *Selected Data* (rows covered by the selected
subgroups), *Annotated Data*, *Subgroups Table*.

* Filters: adjusted p below α, size limits, text contained in the rule.
* Double-click a name to rename a subgroup; untick *Include* to leave it out
  of the outputs; *Remove selected* / *Keep only selected* / *Restore all*.
* The lower pane shows the details of the selected subgroup, including the
  statistics of the dependent variables and all recorded measures.
* *Annotated data*: one Yes/No column per subgroup, or a single column with
  the first covering subgroup.
"""),
    "subgroup-consolidation": ("Subgroup Consolidation", """\
Chooses a non-redundant set of subgroups from many candidates, typically the
outputs of several algorithms.

**Inputs:** *Subgroups* (several).
**Outputs:** *Selected Subgroups*, *Clustered Subgroups*, *Clusters Table*,
*Similarity Matrix*.

## Common measures

The algorithms' own quality scores are on different scales.  With *Compute
common measures* (default) three measures are computed for every candidate
from its rows alone, in the same way whatever algorithm found it; all are in
[0, 1], larger is better, and 1 need not be attainable:

* **Y distinctness** – 2·AUC − 1 of predicting membership from Y
  (out-of-fold, decision tree).
* **X describability** – the same from X.
* **Relation exceptionality** – the share of the subgroup's own variation of
  Y that is explained only when the model of Y from X may differ inside the
  subgroup (cross-validated regression).  Captures a shifted level as well as
  a changed slope, which Y distinctness does not see.  Its values are small
  even for real subgroups; use it for ordering rather than with a threshold.

They are computed on the data the subgroups were found on, so they are
optimistic; they serve to compare subgroups.

## Methods

* **Cleaning** (default): repeatedly take the best candidate by the quality q,
  add it to the result and remove every candidate at least s_min similar to
  it.  q can be one of the common measures (default: the better of Y
  distinctness and relation exceptionality), the algorithm's own score, a
  p-value, the rank within the algorithm or the size.
* **Clustering**: hierarchical clustering of the candidates on their
  similarity, cut at a threshold or into a number of clusters, with one
  representative per cluster.  *Support* is the number of different
  algorithms in a cluster.

Similarity is the Jaccard, overlap or Dice coefficient of the row sets.
"""),
    "subgroup-classifier": ("Subgroup Classifier", """\
Describes a subgroup (membership yes / no) with any Orange classifier.

**Inputs:** *Subgroups*, *Learner* (several).
**Outputs:** *Model*, *Membership Data*, *Evaluation*, *Evaluation Results*.

AUC and CA are estimated with stratified cross-validation; the training AUC is
shown separately and is optimistic.  The subgroup itself was found on the same
data, so even the cross-validated AUC measures how describable the subgroup is
in X, not whether it would be found again.  The algorithm widgets offer the
same through their *Learner* input and *Model* outputs; this widget is for
subgroups coming from the Browser or Consolidation.
"""),
    "subgroup-profile": ("Subgroup Profile", """\
Bar chart of the dependent variables inside the chosen subgroups.

**Inputs:** *Subgroups*.  **Outputs:** *Selected Subgroups*.

Display: effect size against all data, standardized difference against the
complement, or the means in the subgroup and in the complement.  Categorical
dependent variables are shown as differences of level shares.
"""),
    "subgroup-overlap": ("Subgroup Overlap", """\
Heat map of the pairwise overlap of the subgroups.

**Inputs:** *Subgroups*.
**Outputs:** *Intersection*, *First Only*, *Second Only*, *Overlap Matrix*.

The measure is the Jaccard, overlap or Dice coefficient.  Click a cell (row A,
column B) to send the rows in both subgroups (*Intersection*), in A but not in
B (*First Only*) and in B but not in A (*Second Only*).
"""),
    "save-subgroups": ("Save Subgroups", """\
Writes the subgroups to a file.

**Inputs:** *Subgroups*.

Formats: CSV (one row per subgroup; optionally a second file with the
membership of every row), Excel (sheets for subgroups, effects, membership and
configuration; needs `openpyxl`), JSON (everything including the covered
rows), pickle (for *Load Subgroups*) and a text report.
"""),
    "load-subgroups": ("Load Subgroups", """\
Loads subgroups saved as a pickle by *Save Subgroups*.

**Inputs:** *Data* (optional; the table the subgroups were computed on, to
re-attach it).  **Outputs:** *Subgroups*, *Annotated Data*.

Only open pickle files you trust: unpickling can execute code.
"""),
}

WIDGET_ALG = {"pt": ["pt"], "hc": ["hc"], "c2": ["swapping", "semisupervised", "ranking"], "cca": ["cca"],
              "emm": ["emm"], "mv": ["mv"], "mi": ["mi"], "rules": ["rules"], "ensemble": ["ensemble"]}
METHOD_NAMES = {"swapping": "swapping", "semisupervised": "semi-supervised", "ranking": "ranking"}


def default_text(p):
    d = p.setting_defaults()
    if isinstance(p, (params.OptionalIntParam, params.OptionalFloatParam)):
        return str(d[p.name]) if d.get(f"{p.name}_enabled") else p.none_label
    v = d[p.name]
    if isinstance(p, params.ChoiceParam):
        return dict(p.choices).get(v, str(v))
    if isinstance(p, params.MultiChoiceParam):
        return ", ".join(dict(p.choices).get(x, str(x)) for x in v)
    if isinstance(p, params.BoolParam):
        return "yes" if v else "no"
    return str(v) if v != "" else "—"


def param_table(algs):
    rows, seen = [], {}
    for alg in algs:
        for p in params.PARAMS[alg]:
            if p.name in seen:
                seen[p.name]["methods"].append(alg)
                continue
            entry = {"p": p, "methods": [alg]}
            seen[p.name] = entry
            rows.append(entry)
    out = ["| group | parameter | default | notes |", "|---|---|---|---|"]
    for e in rows:
        p = e["p"]
        notes = p.tip or ""
        if isinstance(p, (params.ChoiceParam, params.MultiChoiceParam)):
            opts = "; ".join(lbl for _, lbl in p.choices)
            notes = (notes + " " if notes else "") + f"Options: {opts}."
        if len(algs) > 1 and len(e["methods"]) < len(algs):
            notes = (notes + " " if notes else "") + "Method: " + ", ".join(METHOD_NAMES[m] for m in e["methods"]) + "."
        out.append(f"| {p.group} | {p.label} | {default_text(p)} | {notes.replace('|', '/')} |")
    return "\n".join(out)


def pictures(file):
    """Markdown for the screenshots of a widget that exist in doc/widgets/images (main picture first)."""
    folder = os.path.join(WID, "images")
    if not os.path.isdir(folder):
        return ""
    names = sorted(n for n in os.listdir(folder) if n == file + ".png" or
                   (n.startswith(file + "-") and n.endswith(".png")))
    names.sort(key=lambda n: (n != file + ".png", n))
    out = []
    for n in names:
        label = n[len(file):-4].strip("-").replace("-", " ") or "widget"
        out.append(f"![{label}](images/{n})")
        if n != file + ".png":
            out.append(f"*Tab: {label}*")
        out.append("")
    return "\n".join(out)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.rstrip() + "\n")


def main():
    index = ["# Subgroups – an Orange add-on for Multiple Responses Subgroup Discovery", "",
             "Subgroup discovery with several dependent variables: find groups of instances, described by the",
             "independent variables X, in which the dependent variables Y behave differently.", "",
             "* [Getting started](getting-started.md)", "* [Concepts, measures and inference](concepts.md)", "",
             "## Widgets", "", "### Data", "",
             "* [Subgroup Datasets](widgets/subgroup-datasets.md)", "* [Variable Roles](widgets/variable-roles.md)", "",
             "### Algorithms", ""]
    for key, a in ALGORITHMS.items():
        index.append(f"* [{a['title']}](widgets/{a['file']}.md) – {a['name']}: {a['lead']}")
        tabs = a["tabs"]
        pics = pictures(a["file"]).split("\n\n", 1)
        body = [f"# {a['title']}: {a['name']}", "", a["lead"], "", pics[0], "", "## Inputs", "", ALG_INPUTS,
                "## Outputs", "",
                ALG_OUTPUTS, "## How it works", "", a["how"], ""]
        if a["classifiers"] is not None:
            body += [CLASSIFIERS.format(extra=a["classifiers"]), ""]
        body += [ALG_COMMON.format(tabs=tabs, pt_note=" – except in this widget, where the permutation test is the "
                                                        "core of the method (1000 permutations)" if a.get("pt") else ""),
                 "", "## Parameters", "", param_table(WIDGET_ALG[key]), ""]
        if len(pics) > 1 and pics[1].strip():
            body += ["## Screenshots of the tabs", "", pics[1], ""]
        write(os.path.join(WID, a["file"] + ".md"), "\n".join(body))
    index += ["", "### Working with the results", ""]
    ds_table = ["| name | description |", "|---|---|"] + [f"| `{n}` | {d} |" for n, (_, d) in datasets.DATASETS.items()]
    for file, (title, text) in OTHER.items():
        if file in ("subgroup-datasets", "variable-roles"):
            pass
        else:
            index.append(f"* [{title}](widgets/{file}.md)")
        pic = pictures(file)
        write(os.path.join(WID, file + ".md"), f"# {title}\n\n" + (pic + "\n" if pic else "")
              + text.format(datasets="\n".join(ds_table)))
    write(os.path.join(DOC, "index.md"), "\n".join(index))
    print("wrote", len(ALGORITHMS) + len(OTHER) + 1, "pages to", DOC)


if __name__ == "__main__":
    main()
