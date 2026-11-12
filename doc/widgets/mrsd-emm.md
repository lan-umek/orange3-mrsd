# MRSD-EMM: Exceptional Model Mining

Rules in X under which the relation between X and Y differs from the complement.

![widget](images/mrsd-emm.png)

## Inputs

| input | type | meaning |
|---|---|---|
| **Data** | Table | independent variables X = attributes, dependent variables Y = class variables; meta attributes are ignored |
| **Subgroups** | Subgroups | optional. If no Data is connected, the data of these subgroups is used. In MRSD-C2 the subgroups are also the starting subgroups of the search |
| **Learner** | Learner (several) | optional Orange learners (Tree, Naive Bayes, Logistic Regression …). In widgets with a *Classifier* tab they describe the selected subgroup and provide the *Model* outputs; in the classification-based algorithms they also replace the built-in classifiers of the search |

## Outputs

| output | type | meaning |
|---|---|---|
| **Subgroups** | Subgroups | the subgroups shown in the table (after the filters of the *Results* box) |
| **Data** | Table | the input data, unchanged – for chaining algorithm widgets |
| **Annotated Data** | Table | the data with one Yes/No meta column per shown subgroup |
| **Subgroups Table** | Table | one row per subgroup: rule, prototype in Y, size, quality, p-values, statistics of the dependent variables |
| **Selected Subgroup Data** | Table | the rows of the subgroup selected in the table (or chosen in an inspector) |
| **Membership Data (X)** / **(Y)** | Table | all rows with a class `in <subgroup>` (no / yes) and X, respectively Y, as features |
| **Model (X)** / **(Y)** | Model | the membership model of the selected subgroup fitted with a connected Orange learner (for Tree Viewer, Nomogram …) |

## How it works

Candidates are conjunctive rules in X found by beam search (or clusters, or
random subsets).  A candidate is scored by how exceptional its X–Y model is
compared with the complement: differences of correlations (Fisher z),
of regression slopes or of R².  Scores are normalized by their sampling error
so that small subgroups are not favoured.

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

Permutation tests are **off by default** (Permutations = 0).  When
switched on, the raw p-value, the family-wise adjusted p-value (Westfall–Young
max-T, or a re-run of the whole search on permuted data) and the
Benjamini–Hochberg value are added.  Search-level permutations repeat the
whole search and can be slow.

## Tabs of the main area

* **Subgroups** – the table: rule / description in X, prototype in Y (means
  and modal levels), size, quality, p-values and the difference of every
  dependent variable from the whole data.
* **Correlations** – correlations of every numeric X–Y pair inside the
  subgroup and in the complement with the z of their difference; the selected
  pair is plotted with both regression lines.
* **Prototype** – descriptive statistics of every X and Y variable in the
  subgroup and in its complement (mean, SD, median, IQR or level shares),
  effect sizes, a parametric and a non-parametric test per variable
  (Welch t / Mann–Whitney U; χ² / Fisher exact or Monte-Carlo χ²) and a bar
  chart with several display variants.  The p-values are descriptive: the
  subgroup was selected on the same data.
* **Measures** – everything the algorithm recorded about the subgroup.


## Parameters

| group | parameter | default | notes |
|---|---|---|---|
| Model | Exceptionality | correlations (Fisher z) | Options: correlations (Fisher z); regression slopes; R² difference. |
| Model | Correlation aggregation | max |z| | Options: max /z/; mean z². |
| Model | Combination over targets (regression) | Fisher | Options: Fisher; maximum. |
| Model | Also use Y as descriptive (selection on outcome!) | no |  |
| Search | Search | beam search | Options: beam search; clustering; random. |
| Search | Beam width | 20 |  |
| Search | Maximum conditions per rule | 3 |  |
| Search | Bins for numeric attributes | 4 |  |
| Search | Clustering (clustering search) | k-means | Options: k-means; k-medoids; agglomerative; spectral. |
| Search | Numbers of clusters (clustering search) | — |  |
| Search | Single number of clusters | use list |  |
| Search | Random candidates (random search) | 100 |  |
| Subgroup size | Minimum size | 10 | Smallest subgroup reported |
| Subgroup size | Maximum size | automatic (n / 2) | Largest subgroup (automatic = n / 2) |
| Post-processing | Minimum exceptionality (−log10 p) | 0.0 |  |
| Post-processing | Report at most | 50 |  |
| Post-processing | Redundancy threshold (Jaccard) | 0.7 | Subgroups more similar than this to a better one are pruned |
| Inference | Permutations (0 = no test) | 0 |  |
| Inference | Family-wise correction over | all evaluated candidates (search) | Options: all evaluated candidates (search); returned candidates only. |
| Other | Random seed | 0 | Fixed seed for reproducible results |

## Screenshots of the tabs

![correlations](images/mrsd-emm-correlations.png)
*Tab: correlations*

![measures](images/mrsd-emm-measures.png)
*Tab: measures*

![prototype](images/mrsd-emm-prototype.png)
*Tab: prototype*
