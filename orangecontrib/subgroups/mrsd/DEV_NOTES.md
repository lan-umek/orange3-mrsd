# MRSD refactoring – conventions for algorithm modules

MRSD = **Multiple Responses Subgroup Discovery**. Use this expansion in all docstrings.

## Shared modules (do NOT edit; report needed changes instead)

- `common.py` – `FeatureSpace.from_dataframe(df)`: typed view of X or Y.
  - `.codes` (n,p): numeric as-is, categorical as integer codes (for rules / tree models).
  - `.numeric` (n,d): standardized numeric + one-hot categorical (for distances, CCA, kernels, classifiers, Hotelling).
  - `.columns`, `.is_categorical`, `.categories[j]`, `.decode(j, code)`, `.numeric_names`.
  - `make_rng(seed)` -> `np.random.Generator`; `derive_seed(rng)` -> int for sklearn. **Never call `np.random.seed`.**
  - `OutOfFoldClassifier(estimator, n_folds, random_state).predict_proba(X, y)` / `.auc(X, y)`: cross-fitted membership probabilities. **Never predict on training rows.**
  - `mask_from_indices(indices, n)`, `indices_from_mask(mask)`.
- `data_loader.py` – `Dataset.X_space`, `Dataset.Y_space` (cached FeatureSpaces), `Dataset.categorical_X_columns()`.
- `rules.py` – `Condition`, `Rule` (original units, `str(rule)` readable, `rule.evaluate(fs)` -> mask, hashable), `ConditionGenerator(fs, n_bins, min_support, local_cutpoints, allow_negation)`, `BeamSearch(generator, quality_fn(mask)->float, beam_width, max_depth, n_min, n_max).search()` -> list of `SearchResult(rule, mask, quality, depth)` sorted best first, `bs.n_evaluated`; `describe_mask_with_tree(fs, mask, max_depth=3)` -> `Description(rule, precision, recall, jaccard)` for index-set methods.
- `inference.py` – `validate_subgroups(subgroups, Y_space=..., X_space=..., statistic='y_shift'|'x_auc', n_permutations, random_state, selection_note=...)` fills `p_value` (raw permutation), `p_value_corrected` (Westfall–Young max-T, FWER), `quality_measures['p_bh']`, `['test_statistic']`, `['t2']`, `['p_asymptotic']`, `['inference']`. Also `hotelling_t2(Y, mask)`, `y_shift_statistic`, `y_shift_batch`, `permutation_test_candidates(masks, stat_fn, data, B, seed, batch_fn)`.
  - Rule of thumb: candidates selected from X → `statistic='y_shift'` (Y rows permuted). Candidates selected from Y → `statistic='x_auc'` (X rows permuted). Candidates selected from both (C2 family, CCA, MV, ensemble) → a fixed-index-set test is anti-conservative on noise (verified); the default must be a search-level swap randomization (`inference.search_permutation_pvalues`: re-run the whole search on permuted Y, including the generation of starting sets, and compare with the best quality found). The conditional `'y_shift'` variant may be offered as a fast option with a selection note.
  - Family-wise adjustment must cover every candidate the search *evaluated*, not only the returned ones (`BeamSearch.search()` returns all visited candidates for this reason).
- `classifiers.py` – `make_classifier(name, random_state, fast)`, `ClassifierFactory` (facade), `DualSpaceEvaluator(X, Y, clf_X, clf_Y, n_folds, aggregation, random_state, fast).evaluate(mask)` -> `DualEvaluation(q_X, q_Y, quality, proba_X, proba_Y)` (OOF, cached, paired folds), `.fresh_copy(seed)` for honest re-scoring, `aggregate()`, `initial_subgroups(n, k, n_min, n_max, rng, X, Y, method)`, `resubstitution_auc` (diagnostic only).
- `subgroups.py` – `Subgroup` now has `algorithm`, `rule`, `description_precision`, `description_recall`, `.quality` (= `quality_measures['quality']`), `.mask(n)`, `.fill_target_statistics(Y_df)`, `.indices_key`. `SetOfSubgroups.deduplicate()`, `.prune(quality_func, s_min)` (dedup + greedy Jaccard), `.summary(alpha)`. **`p_value` is set only by `inference.validate_subgroups`** – never write `1 - quality` into it.
- `clustering.py` – labels are contiguous and `k` is the actual number of clusters; `prepare_data_for_clustering(df, 'onehot'|'codes')` returns `(matrix, FeatureSpace)`.
- `divergence.py` – vectorized; `cell_f_divergence`, `cell_standardized_residuals`, `tables_for_permutations`, `cell_statistics_batch`, `FDivergenceRegistry.canonical/is_unbounded_at_zero/distinct_divergences`.

## Every algorithm module must

1. Build `fs_X = dataset.X_space`, `fs_Y = dataset.Y_space`; use `.numeric` for distance/model/CCA/kernel work and `.codes` only for tree models and rule conditions. No label-encoding of categoricals as ordinal numbers anywhere.
2. Use `rng = make_rng(config.random_state)` and derive all seeds from it. No `np.random.seed`, no hard-coded `random_state=42`, no module-level `warnings.filterwarnings('ignore')`.
3. Never swallow exceptions silently. Use `logger = logging.getLogger("mrsd")` and `logger.warning(...)` with the exception for data-dependent failures; let configuration errors (`TypeError`, bad enum) propagate.
4. Produce `Subgroup`s with: `algorithm="<name>"`, `description` (short human text), `rule` (a `rules.Rule` in original units — from the search itself, or from `describe_mask_with_tree` with `description_precision/recall` filled), `quality_measures['quality']` (the search score), algorithm-specific measures under their own keys, and `fill_target_statistics(dataset.Y)` called.
5. At the end of `fit()`: `result = SetOfSubgroups(...).prune(QualityMeasures.quality, s_min=config.redundancy_threshold)` (add `redundancy_threshold: float = 0.7` to the config), then, if `config.n_permutations > 0` (add this config field, default 200, plus `inference_statistic` where the choice is not obvious), call `validate_subgroups(...)` with the appropriate statistic. Return the pruned, validated set. Default `n_max` must be `n // 2`, not `n`.
6. Keep the public API (`run_<alg>(data, X_columns, Y_columns, **config)`, `<Config>` dataclass, main class with `fit(dataset)`) working. Remove config fields / enum members that do nothing, or implement them. Keep `verbose` printing modest.
7. Reference statistics must compare the subgroup with its **complement** (D \ G), not with the whole dataset.
8. Any exceptionality/difference score must be normalized by its sampling error (z-score / -log10 p) or calibrated by permutation so that it does not trivially favour small subgroups.
9. Update the module docstring to describe what is actually implemented. Keep an `if __name__ == "__main__":` demo that plants a subgroup and prints whether it was recovered.
10. Tests: in `tests/test_<module>.py` (pytest) add (a) a planted-subgroup recovery test with a fixed seed (Jaccard of the best returned subgroup with the planted one ≥ 0.5, and adjusted p < 0.05), (b) a pure-noise test (no subgroup with adjusted p < 0.05, or at most what α allows across a few seeds), (c) determinism with fixed seed, (d) categorical X and Y columns run without error. Keep each test under ~30 s; use small n (150–300) and few permutations (100–200).

## Imports
All modules live in the package `mrsd_subgroups` and use relative imports (`from .common import ...`); tests add the repository root to `sys.path` via `tests/conftest.py`.
