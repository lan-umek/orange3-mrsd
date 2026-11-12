# MRSD — Multiple Responses Subgroup Discovery

Algorithms for discovering interesting subgroups in data with several response
variables **Y** described by explanatory variables **X**.  Version 2.0.0.

| id | module | idea | candidates come from | inference |
|----|--------|------|----------------------|-----------|
| `pt` | `mrsdpt.py` | partitions of X × partitions of Y, cell-level f-divergence | both (contingency cells) | cell permutation test, max-T over all cells |
| `hc` | `mrsdhc.py` | dendrogram of Y, out-of-fold separability from X | Y | X permuted, max-T over all nodes |
| `swapping` | `mrsdc2_swapping.py` | local search moving instances, OOF AUC in X and Y | both | search re-run on permuted Y |
| `semisupervised` | `mrsdc2_semisupervised.py` | self-training on OOF membership probabilities | both | search re-run on permuted Y |
| `ranking` | `mrsdc2_ranking.py` | top-s of OOF probability rankings | both | search re-run on permuted Y |
| `cca` | `mrsdcca.py` | clusters in regularized canonical space, Fisher-z of ρ₁ in G vs rest | both | Y permuted, family-wise over search |
| `emm` | `mrsdemm.py` | exceptional model mining: Fisher-z / interaction F-tests | X (rules) | Y permuted, family-wise over search |
| `mv` | `mrsd_mv.py` | multi-view: co-regularized spectral clustering (default), averaged affinities, co-training, late fusion, similarity excess, dual dendrogram | both | Y permuted on the X-rule of the cluster |
| `mi` | `mrsd_mi.py` | null-calibrated mutual information in a rule (estimator for mixed variable types by default; KSG, binning) | X (rules) | Y permuted, family-wise over search |
| `rules` | `mrsd_rules.py` | CN2-SD beam search with multi-response WRAcc | X (rules) | search re-run on permuted Y |
| `ensemble` | `mrsd_ensemble.py` | co-association consensus of the others | all | whole pipeline re-run on permuted Y |

## Names

**MRSD** = Multiple Responses Subgroup Discovery.  **PT** = partitioning and
tables (partitions of X and of Y, contingency tables of their pairs),
**HC** = hierarchical clustering (of Y) and classification (from X),
**C2** = classification in both spaces (X and Y; methods swapping,
semi-supervised, ranking), **CCA** = canonical correlation analysis,
**EMM** = exceptional model mining, **MV** = multi-view, **MI** = mutual
information.

## Installation

```
pip install mrsd-subgroups
```

## Quick start

```python
import pandas as pd
import mrsd_subgroups as mrsd
# or:  from mrsd_subgroups.mrsd_rules import run_mrsd_rules

df = pd.read_csv("data.csv")
result = mrsd.run_mrsd_rules(df, X_columns=["age", "edu", "income"], Y_columns=["y1", "y2"],
                             random_state=0)
print(result.summary())
for sg in result:
    print(sg)                        # Subgroup('edu = high AND age >= 50', n=41, q=0.71, p=0.004, p_adj=0.004)
    print(sg.target_statistics)      # per-target means / effect sizes inside vs. all
```

Every algorithm has `run_<name>(df, X_columns, Y_columns, **config)`, a
`<Name>Config` dataclass and a class with `fit(dataset)`; `mrsd.ALGORITHMS`
maps short ids to the run functions.  `examples/compare_algorithms.py` runs
all of them on a planted and a pure-noise dataset.

Classifier parameters (`classification_model` of MRSD-HC, `classifier_X` /
`classifier_Y` of the two-space algorithms) accept a name from
`classifiers.ClassificationModel` or any scikit-learn compatible estimator
object.  A detailed description of every algorithm is in
[docs/algorithms.md](docs/algorithms.md).

## What every result contains

* `covered_instances` – row indices; `rule` – description in X in original
  units (exact for rule-based methods, a decision-tree approximation with
  `description_precision` / `description_recall` for cluster-based ones);
* `quality_measures['quality']` – the algorithm's own score (scale is
  algorithm-specific; compare within an algorithm only);
* `p_value` – raw permutation p-value; `p_value_corrected` – family-wise
  adjusted (Westfall–Young max-T or search-level swap randomization,
  see `quality_measures['inference']`); `quality_measures['p_bh']` – BH FDR;
* `target_statistics` – per-response means, SDs and effect sizes inside the
  subgroup vs. the whole data.

`p_value` is **only** ever written by `inference.py`; an algorithm that did not
test leaves it `None`.

## Statistical conventions (see `inference.py`)

* Null hypothesis: X and Y independent; rows of the view that was *not* used
  to define the candidates are permuted.
* Candidates defined from both views (C2 family, CCA, MV, ensemble) cannot be
  tested honestly with a fixed index set; their default inference re-runs the
  whole search on permuted data (`inference_statistic="search_permutation"`,
  `inference_mode="search"`).  The cheaper conditional test is available as an
  option and is anti-conservative on noise.
* Exceptionality scores are variance-normalized (Fisher-z, F-tests,
  Hotelling T², null-shuffled MI) so that they do not favour small subgroups.
* Nominal variables are one-hot encoded for distances/models and used as
  labels in rules; they are never treated as ordinal codes.

## Tests

```
pip install -e .[test]               # in a clone of the repository
python -m pytest tests -q           # ~8 min; planted-subgroup recovery, pure-noise FWER, determinism, categorical data
```

## Layout

The package is `mrsd_subgroups/`: `common.py` (typed feature spaces, out-of-fold classification), `rules.py`
(conditions, rules, beam search, tree descriptions), `inference.py`
(permutation framework), `classifiers.py` (two-space evaluator),
`core.py` (`Subgroup`, `SetOfSubgroups`; `subgroups.py` is an alias),
`clustering.py`, `divergence.py`, `data_loader.py`, the eleven algorithm
modules, `tests/`, `examples/`, `DEV_NOTES.md` (conventions for contributors).
