# MRSD-HC: Hierarchical Clustering and Classification

Subgroups are nodes of the dendrogram of Y that a classifier can tell apart from the rest using X.

![widget](images/mrsd-hc.png)

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

The instances are clustered hierarchically on the dependent variables Y.
Every node of the dendrogram (within the size limits) is a candidate: a group
of instances with similar responses.  Its quality is the out-of-fold AUC of
predicting membership of the node from X – a group that is homogeneous in Y
and recognisable from X.  With several classifiers the quality is the best of
them.  The optional test permutes the rows of X (the nodes come from Y only)
and compares the AUC; the adjustment runs over all examined nodes.

## Classifiers

Without a connected learner the built-in classifiers chosen in the widget are
used.  When at least one Orange learner is connected to the *Learner* input,
*Use the connected Orange learner(s)* appears (ticked) and the built-in
choice is hidden; MRSD-HC scores every node with all connected learners and takes the best. *Also use the built-in classifiers ticked below* adds the built-in ones.  Slow learners (random forest, SVM) make the search
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
* **Dendrogram** – the dendrogram of Y with the node of the selected subgroup
  highlighted.  Click a merge point to choose another node; a node that is not
  a reported subgroup is sent on as an ad-hoc subgroup and can be kept.
* **Classifier** – the membership classifier in X: out-of-fold AUC, training
  AUC (optimistic) and a readable model (tree rules, coefficients,
  importances).
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
| Clustering of Y | Distance on Y | Manhattan | Options: Manhattan; Euclidean; cosine; Hamming; Chebyshev; Minkowski. |
| Clustering of Y | Linkage | Ward | Options: Ward; complete; average; single; weighted; centroid; median. |
| Separability from X | Classifiers (quality = best) | Decision tree | Options: Random forest; Decision tree; Logistic regression; SVM (RBF); SVM (linear); Naive Bayes; k-nearest neighbours; Gradient boosting; AdaBoost; Neural network (MLP). |
| Separability from X | Fast classifier configuration | yes |  |
| Separability from X | Evaluation | out-of-fold (cross-validation) | Options: out-of-fold (cross-validation); train on train (diagnostic only). |
| Separability from X | Folds | 3 |  |
| Separability from X | Require lower dispersion than the whole data | no |  |
| Separability from X | Minimum AUC (after inference) | off |  |
| Inference | Permutations of X (0 = no test) | 0 |  |
| Subgroup size | Minimum size | 5 | Smallest subgroup reported |
| Subgroup size | Maximum size | automatic (n / 2) | Largest subgroup (automatic = n / 2) |
| Post-processing | Redundancy threshold (Jaccard) | 0.7 | Subgroups more similar than this to a better one are pruned |
| Other | Random seed | 0 | Fixed seed for reproducible results |

## Screenshots of the tabs

![classifier](images/mrsd-hc-classifier.png)
*Tab: classifier*

![dendrogram](images/mrsd-hc-dendrogram.png)
*Tab: dendrogram*

![measures](images/mrsd-hc-measures.png)
*Tab: measures*

![prototype](images/mrsd-hc-prototype.png)
*Tab: prototype*
