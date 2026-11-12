# Subgroups – an Orange add-on for Multiple Responses Subgroup Discovery

Subgroup discovery with several dependent variables: find groups of instances, described by the
independent variables X, in which the dependent variables Y behave differently.

* [Getting started](getting-started.md)
* [Concepts, measures and inference](concepts.md)

## Widgets

### Data

* [Subgroup Datasets](widgets/subgroup-datasets.md)
* [Variable Roles](widgets/variable-roles.md)

### Algorithms

* [MRSD-PT](widgets/mrsd-pt.md) – Partitioning and Tables: Subgroups are cells of the contingency tables of pairs of partitions of X and of Y.
* [MRSD-HC](widgets/mrsd-hc.md) – Hierarchical Clustering and Classification: Subgroups are nodes of the dendrogram of Y that a classifier can tell apart from the rest using X.
* [MRSD-C2](widgets/mrsd-c2.md) – Classification in Both Spaces: Subgroups whose membership can be predicted both from X and from Y.
* [MRSD-CCA](widgets/mrsd-cca.md) – Canonical Correlation Analysis: Subgroups in which the canonical correlation between X and Y differs from the rest.
* [MRSD-EMM](widgets/mrsd-emm.md) – Exceptional Model Mining: Rules in X under which the relation between X and Y differs from the complement.
* [MRSD-MV](widgets/mrsd-mv.md) – Multi-View: X and Y are two views of the same instances; subgroups are coherent in both.
* [MRSD-MI](widgets/mrsd-mi.md) – Mutual Information: Rules in X under which X and Y are more dependent than expected.
* [MRSD-Rules](widgets/mrsd-rules.md) – Rules (CN2-SD): Rules in X with an exceptional distribution of the dependent variables.
* [MRSD-Ensemble](widgets/mrsd-ensemble.md) – Consensus: Runs several of the other algorithms and fuses their subgroups.

### Working with the results

* [Subgroup Browser](widgets/subgroup-browser.md)
* [Subgroup Consolidation](widgets/subgroup-consolidation.md)
* [Subgroup Classifier](widgets/subgroup-classifier.md)
* [Subgroup Profile](widgets/subgroup-profile.md)
* [Subgroup Overlap](widgets/subgroup-overlap.md)
* [Save Subgroups](widgets/save-subgroups.md)
* [Load Subgroups](widgets/load-subgroups.md)
