# MRSD-CCA: Canonical Correlation Analysis

Subgroups in which the canonical correlation between X and Y differs from the rest.

![widget](images/mrsd-cca.png)

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

A regularized canonical correlation analysis of X and Y is computed on all
rows and the instances are clustered in the canonical space.  A cluster is
scored by Fisher's z of the difference between the first canonical correlation
inside the cluster and in its complement.  The cluster is described post hoc
by a rule in X (a decision tree).  The classification qualities are reported
only; they are not part of the score.

## Classifiers

Without a connected learner the built-in classifiers chosen in the widget are
used.  When at least one Orange learner is connected to the *Learner* input,
*Use the connected Orange learner(s)* appears (ticked) and the built-in
choice is hidden: *Learner for X* and *Learner for Y* choose which connected learner is used in which space.  Slow learners (random forest, SVM) make the search
slow, because every candidate is scored by cross-validation.


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
* **Canonical space** – scatter of the first canonical variates of X and Y
  with the subgroup and the complement, their regression lines and the table
  of canonical weights.
* **Classifier** – membership classifiers in X and in Y.
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
| CCA | Canonical components | 2 |  |
| CCA | Ridge regularization λ | 0.0 |  |
| CCA | Clustering space | X scores | Options: X scores; Y scores; combined. |
| Clustering | Clustering | k-means | Options: k-means; agglomerative. |
| Clustering | Numbers of clusters | 2, 3, 4 |  |
| Clustering | Single number of clusters | use list |  |
| Classification | Report out-of-fold classification quality | no |  |
| Classification | Classifier in X | Random forest | Options: Random forest; Decision tree; Logistic regression; SVM (RBF); SVM (linear); Naive Bayes; k-nearest neighbours; Gradient boosting; AdaBoost; Neural network (MLP). |
| Classification | Classifier in Y | Random forest | Options: Random forest; Decision tree; Logistic regression; SVM (RBF); SVM (linear); Naive Bayes; k-nearest neighbours; Gradient boosting; AdaBoost; Neural network (MLP). |
| Classification | Folds | 5 |  |
| Subgroup size | Minimum size | 10 | Smallest subgroup reported |
| Subgroup size | Maximum size | automatic (n / 2) | Largest subgroup (automatic = n / 2) |
| Post-processing | Redundancy threshold (Jaccard) | 0.7 | Subgroups more similar than this to a better one are pruned |
| Inference | Permutations (0 = no test) | 0 |  |
| Inference | Family-wise correction over | all evaluated candidates (search) | Options: all evaluated candidates (search); returned candidates only. |
| Other | Random seed | 0 | Fixed seed for reproducible results |

## Screenshots of the tabs

![canonical space](images/mrsd-cca-canonical-space.png)
*Tab: canonical space*

![classifier](images/mrsd-cca-classifier.png)
*Tab: classifier*

![measures](images/mrsd-cca-measures.png)
*Tab: measures*

![prototype](images/mrsd-cca-prototype.png)
*Tab: prototype*
