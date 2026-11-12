# Concepts, measures and inference

## The task

The data has independent (descriptive) variables **X** and several dependent
variables (responses) **Y**, numeric or categorical.  A **subgroup** is a set
of instances that is (a) describable in X and (b) interesting in Y: the
dependent variables, or their relation to X, behave differently inside the
subgroup than in the rest of the data.  This is **Multiple Responses Subgroup
Discovery (MRSD)**.

## What the algorithms look for

| algorithm | name | a subgroup is … |
|---|---|---|
| MRSD-PT | partitioning and tables | a cell of a contingency table of a partition of X against a partition of Y that holds more instances than expected |
| MRSD-HC | hierarchical clustering and classification | a cluster of Y that a classifier recognises from X |
| MRSD-C2 | classification in both spaces | a set whose membership is predictable from X and from Y |
| MRSD-Rules | rules (CN2-SD) | a rule in X with a shifted distribution of Y |
| MRSD-MV | multi-view | a group coherent in both views X and Y |
| MRSD-EMM | exceptional model mining | a rule in X under which the X–Y model differs |
| MRSD-CCA | canonical correlation | a cluster with a different canonical correlation |
| MRSD-MI | mutual information | a rule in X under which X and Y are more dependent |
| MRSD-Ensemble | consensus | a group supported by several of the above |

The first five look for a different *distribution* of Y; EMM, CCA and MI look
for a different *relation* between X and Y.  A subgroup of the second kind
need not differ in the means of Y at all.

## Description of a subgroup

* **Rule / description (X)** – for rule-based algorithms the rule that defines
  the subgroup; for the others a decision-tree approximation, with its
  precision and recall with respect to the subgroup in the *Measures* tab.
* **Prototype (Y)** – the means of the numeric and the modal levels (with
  their shares) of the categorical dependent variables in the subgroup.
* **Classifier** – any Orange learner can describe the membership instead of
  a rule (*Learner* input, *Model* outputs).

## Quality

Each algorithm has its own quality score, comparable only within that
algorithm: an AUC for HC and C2, test-statistic-like scores for PT, Rules,
EMM, CCA, MI, bounded mixtures for MV.  To compare subgroups of different
algorithms use the **common measures** of *Subgroup Consolidation* (Y
distinctness, X describability, relation exceptionality), which are computed
from the rows of a subgroup in the same way for all algorithms.

## Inference

The null hypothesis is that X and Y are independent.  Rows of the view that
was not used to define the candidates are permuted.  When candidates are
defined from both views, a test of a fixed row set is anti-conservative, so
the whole search is re-run on permuted data.  Reported are the raw p-value,
the family-wise adjusted p-value (covering every candidate the search
examined) and the Benjamini–Hochberg value.

Permutation inference is optional and off by default except in MRSD-PT.

## What the numbers do not tell

All measures, including cross-validated AUCs and the tests in the *Prototype*
tab, are computed on the data in which the subgroup was found.  They compare
subgroups and guard against some chance findings, but they do not estimate
how well a subgroup would replicate in new data.  For that, set data aside
before the analysis and examine the found rules or membership models on it.

## Redundancy

Algorithms return many overlapping subgroups.  Within an algorithm they are
pruned by Jaccard similarity (*Redundancy threshold*).  Across algorithms
*Subgroup Consolidation* applies the cleaning procedure: take the best
subgroup by a quality q, remove all subgroups at least s_min similar to it,
repeat.
