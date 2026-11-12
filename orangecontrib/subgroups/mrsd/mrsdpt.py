"""
MRSD-PT: Multiple Responses Subgroup Discovery with Partitioning and Tables
(PT = partitioning and tables: partitions of X and of Y, contingency tables of their pairs).

What is implemented
-------------------
1. Several partitions of X and of Y are built: clustering (k-means, k-medoids,
   agglomerative, spectral) on the numeric view (`FeatureSpace.numeric`:
   standardized numeric + one-hot categorical columns) for every requested
   number of clusters, plus the *natural* partition (unique value combinations
   of the categorical columns, in original labels) when there are categorical
   columns and the number of combinations is small enough.
2. For every pair (X-partition, Y-partition) the contingency table O is built.
   The candidate subgroups are the cells with n_min <= O_ij <= n_max; the
   subgroup is the set of instances in the cell.
3. Per-cell statistic: the cell's contribution E_ij f(O_ij/E_ij) to the
   f-divergence of the table (`cell_statistic="f_divergence"`), or the squared
   adjusted standardized residual (`cell_statistic="residual"`).
4. Inference: ONE set of B row permutations (independence of X and Y) is shared
   by all tables and all cells.  Every cell statistic is standardized by its
   permutation mean and sd (so that cells of tables with different k, l are on
   a common scale); the standardized value is the subgroup's `quality`.  From
   the observed (m,) and permuted (B, m) statistics over ALL cells examined,
   `inference.pvalues_from_permutations` gives the raw permutation p-value
   (`p_value`), the Westfall-Young max-T adjusted p-value (`p_value_corrected`,
   FWER over every cell examined) and Benjamini-Hochberg
   (`quality_measures['p_bh']`).  The table-level permutation p-value of the
   f-divergence statistic is kept in `quality_measures['p_table']`.
5. Only after the p-values are computed over the full candidate set, the set is
   deduplicated and pruned (greedy Jaccard, `redundancy_threshold`); the
   unpruned set stays available through `get_all_candidates()`.
6. Every subgroup gets a `rule` in X space: the exact rule for cells of a
   natural X partition, otherwise a decision-tree approximation
   (`rules.describe_mask_with_tree`) with its precision / recall.

Cluster-count precedence: `k_values_X` / `k_values_Y` > `k_values` > `max_k`
(= 2..max_k) > default [2, 3, 4, 5].
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .clustering import ClusteringFactory, PartitionResult
from .common import FeatureSpace, derive_seed, make_rng
from .data_loader import DataLoader, Dataset
from .divergence import (
    ContingencyTable,
    FDivergenceRegistry,
    FDivergenceType,
    cell_f_divergence,
    cell_standardized_residuals,
    cell_statistics_batch,
    compute_f_divergence_statistic,
    tables_for_permutations,
)
from .inference import pvalues_from_permutations
from .rules import Condition, Rule, describe_mask_with_tree
from .core import QualityMeasures, SetOfSubgroups, Subgroup

logger = logging.getLogger("mrsd")

_CELL_STATISTICS = ("f_divergence", "residual")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MRSDPTConfig:
    """
    Configuration of MRSD-PT.

    Cluster counts (precedence: k_values_X / k_values_Y > k_values > max_k > default):
        k_values_X, k_values_Y: numbers of clusters for X / Y partitions
        k_values: used for both sides when the specific list is not given
        max_k: shorthand for k_values = [2, ..., max_k]
    Partitions:
        clustering_algorithms_X/Y: names from clustering.ClusteringFactory
        use_natural_partition_X/Y: add the partition by unique combinations of
            the categorical columns (only if there are categorical columns and
            at most `max_natural_partitions` combinations)
    Statistics:
        f_divergence: generator name (divergence.FDivergenceType)
        cell_statistic: "f_divergence" (cell contribution E f(O/E)) or
            "residual" (squared adjusted standardized residual)
        n_permutations: B shared row permutations (0 disables inference)
        alpha: significance level used by get_significant_subgroups
    Subgroups:
        n_min, n_max: cell size limits (n_max None -> n // 2)
        redundancy_threshold: Jaccard threshold of the final pruning
    """
    k_values_X: Optional[List[int]] = None
    k_values_Y: Optional[List[int]] = None
    k_values: Optional[List[int]] = None
    max_k: Optional[int] = None

    clustering_algorithms_X: List[str] = field(default_factory=lambda: ["kmeans", "kmedoids"])
    clustering_algorithms_Y: List[str] = field(default_factory=lambda: ["kmeans", "kmedoids"])

    f_divergence: str = "chi_squared"
    cell_statistic: str = "f_divergence"
    n_permutations: int = 1000
    alpha: float = 0.05

    n_min: int = 5
    n_max: Optional[int] = None
    redundancy_threshold: float = 0.7

    use_natural_partition_X: bool = True
    use_natural_partition_Y: bool = True
    max_natural_partitions: int = 20

    random_state: Optional[int] = None
    verbose: bool = True

    def __post_init__(self):
        if self.max_k is not None:
            default_k = list(range(2, int(self.max_k) + 1))
        else:
            default_k = [2, 3, 4, 5]
        if self.k_values is not None:
            default_k = list(self.k_values)
        if self.k_values_X is None:
            self.k_values_X = list(default_k)
        if self.k_values_Y is None:
            self.k_values_Y = list(default_k)
        if self.cell_statistic not in _CELL_STATISTICS:
            raise ValueError(f"cell_statistic must be one of {_CELL_STATISTICS}, got {self.cell_statistic!r}")
        FDivergenceType(self.f_divergence)  # raises ValueError for unknown names
        if self.n_min < 1:
            raise ValueError("n_min must be >= 1")
        if self.n_permutations < 0:
            raise ValueError("n_permutations must be >= 0")

    def get_k_values_X(self) -> List[int]:
        return list(self.k_values_X)

    def get_k_values_Y(self) -> List[int]:
        return list(self.k_values_Y)


@dataclass
class PartitionConfiguration:
    """A partition of one side: algorithm name, requested k and the fitted result."""
    algorithm: str
    k: int
    result: Optional[PartitionResult] = None

    @property
    def actual_k(self) -> int:
        return self.result.k if self.result is not None else self.k

    def __repr__(self):
        return f"PartitionConfig({self.algorithm}, k={self.k})"


# =============================================================================
# HELPERS
# =============================================================================

def _residuals_batch(tables: np.ndarray) -> np.ndarray:
    """Squared adjusted standardized residuals for a (B, k, l) batch of tables."""
    O = tables.astype(float)
    N = O.sum(axis=(1, 2), keepdims=True)
    r = O.sum(axis=2, keepdims=True)
    c = O.sum(axis=1, keepdims=True)
    E = r * c / np.maximum(N, 1)
    denom = np.sqrt(np.maximum(E, 1e-12) * (1 - r / np.maximum(N, 1)) * (1 - c / np.maximum(N, 1)))
    return ((O - E) / np.maximum(denom, 1e-12)) ** 2


def _natural_partition(fs: FeatureSpace, max_partitions: int) -> Optional[PartitionResult]:
    """Natural partition on the categorical columns of `fs` (original labels)."""
    cat_cols = [c for c, is_cat in zip(fs.columns, fs.is_categorical) if is_cat]
    if not cat_cols:
        return None
    df = fs.raw_dataframe()[cat_cols]
    n_comb = df.drop_duplicates().shape[0]
    if n_comb > max_partitions:
        logger.info("natural partition skipped: %d combinations > max_natural_partitions=%d",
                    n_comb, max_partitions)
        return None
    return ClusteringFactory.get_clusterer("natural").fit(df)


def _natural_rule(fs: FeatureSpace, values: Dict[str, object]) -> Rule:
    conds = [Condition(col, fs.column_index(col), "=", val) for col, val in values.items()]
    return Rule(conds)


# =============================================================================
# ALGORITHM
# =============================================================================

class MRSDPT:
    """
    MRSD-PT: Multiple Responses Subgroup Discovery with Partitioning and Tables
(PT = partitioning and tables: partitions of X and of Y, contingency tables of their pairs).

    `fit(dataset)` returns the pruned, validated SetOfSubgroups; the full
    candidate set (every cell examined) is available via `get_all_candidates()`.
    """

    def __init__(self, config: Optional[MRSDPTConfig] = None):
        self.config = config or MRSDPTConfig()
        self._partition_results_X: Dict[str, PartitionResult] = {}
        self._partition_results_Y: Dict[str, PartitionResult] = {}
        self._all_subgroups: List[Subgroup] = []
        self._result: Optional[SetOfSubgroups] = None
        self.n_cells_examined: int = 0
        self.timings: Dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, dataset: Dataset) -> SetOfSubgroups:
        cfg = self.config
        rng = make_rng(cfg.random_state)
        t0 = time.perf_counter()

        fs_X = dataset.X_space
        fs_Y = dataset.Y_space
        n = fs_X.n
        n_max = cfg.n_max if cfg.n_max is not None else n // 2

        if cfg.verbose:
            print(f"MRSD-PT: n={n}, |X|={fs_X.p}, |Y|={fs_Y.p}, f={cfg.f_divergence}, "
                  f"cell_statistic={cfg.cell_statistic}, B={cfg.n_permutations}")

        # ---- partitions -------------------------------------------------
        configs_X = self._build_partitions(fs_X, cfg.clustering_algorithms_X, cfg.get_k_values_X(),
                                           cfg.use_natural_partition_X, rng, "X")
        configs_Y = self._build_partitions(fs_Y, cfg.clustering_algorithms_Y, cfg.get_k_values_Y(),
                                           cfg.use_natural_partition_Y, rng, "Y")
        self.timings["partitions"] = time.perf_counter() - t0
        if cfg.verbose:
            print(f"  partitions: X={len(configs_X)}, Y={len(configs_Y)} "
                  f"({self.timings['partitions']:.2f}s)")
        if not configs_X or not configs_Y:
            logger.warning("MRSD-PT: no usable partition on one side; returning empty set")
            self._all_subgroups = []
            self._result = SetOfSubgroups()
            return self._result

        # ---- shared permutations ----------------------------------------
        B = cfg.n_permutations
        perms = self._make_permutations(rng, n, B)

        f_type = FDivergenceType(cfg.f_divergence)
        f_func = FDivergenceRegistry.get_function(f_type)
        unbounded = FDivergenceRegistry.is_unbounded_at_zero(f_type)

        # ---- tables and cell statistics ---------------------------------
        t1 = time.perf_counter()
        records: List[dict] = []
        obs_blocks: List[np.ndarray] = []
        perm_blocks: List[np.ndarray] = []
        tables_with_empty_cells = 0

        for cx, cy in product(configs_X, configs_Y):
            lx, ly = cx.result.labels, cy.result.labels
            table = ContingencyTable.from_labels(lx, ly)
            O = table.table
            E = table.get_expected_under_independence()
            if unbounded and (O == 0).any():
                tables_with_empty_cells += 1
            f_table = float(compute_f_divergence_statistic(table, f_func))

            if cfg.cell_statistic == "f_divergence":
                cell_obs = cell_f_divergence(table, f_func)
            else:
                cell_obs = cell_standardized_residuals(table) ** 2

            cand = np.argwhere((O >= cfg.n_min) & (O <= n_max))
            if len(cand) == 0:
                continue

            if B > 0:
                perm_tables = np.concatenate([tables_for_permutations(lx, ly, p) for p in perms], axis=0)
                if cfg.cell_statistic == "f_divergence":
                    cell_perm = cell_statistics_batch(perm_tables, f_func)       # (B, k, l)
                else:
                    cell_perm = _residuals_batch(perm_tables)
                table_perm = cell_statistics_batch(perm_tables, f_func).sum(axis=(1, 2))
                p_table = float(((table_perm >= f_table).sum() + 1) / (B + 1))
                ci, cj = cand[:, 0], cand[:, 1]
                T_obs = cell_obs[ci, cj]                                          # (m_t,)
                T_perm = cell_perm[:, ci, cj]                                     # (B, m_t)
                mu = T_perm.mean(axis=0)
                sd = T_perm.std(axis=0)
                ok = sd > 1e-12
                z_obs = np.where(ok, (T_obs - mu) / np.where(ok, sd, 1.0), 0.0)
                z_perm = np.where(ok[None, :], (T_perm - mu[None, :]) / np.where(ok, sd, 1.0)[None, :], 0.0)
                obs_blocks.append(z_obs)
                perm_blocks.append(z_perm)
            else:
                p_table = None
                z_obs = cell_obs[cand[:, 0], cand[:, 1]]

            for idx, (i, j) in enumerate(cand):
                records.append(dict(cx=cx, cy=cy, i=int(i), j=int(j), observed=int(O[i, j]),
                                    expected=float(E[i, j]), stat=float(cell_obs[i, j]),
                                    z=float(z_obs[idx]), f_table=f_table, p_table=p_table))

        self.n_cells_examined = len(records)
        self.timings["statistics"] = time.perf_counter() - t1
        if unbounded and tables_with_empty_cells:
            logger.warning("MRSD-PT: f-divergence '%s' is unbounded at 0 and %d table(s) have empty cells; "
                           "their statistics are dominated by the clipping constant. Prefer 'chi_squared', "
                           "'hellinger', 'kl_divergence' or 'total_variation'.",
                           cfg.f_divergence, tables_with_empty_cells)
        if cfg.verbose:
            print(f"  {len(records)} candidate cells from {len(configs_X) * len(configs_Y)} tables "
                  f"({self.timings['statistics']:.2f}s)")

        # ---- inference over ALL cells examined --------------------------
        perm_res = None
        if B > 0 and records:
            observed = np.concatenate(obs_blocks)
            perm = np.concatenate(perm_blocks, axis=1)
            perm_res = pvalues_from_permutations(observed, perm)

        # ---- subgroups ----------------------------------------------------
        t2 = time.perf_counter()
        subgroups: List[Subgroup] = []
        for idx, rec in enumerate(records):
            sg = self._create_subgroup(rec, fs_X, dataset, f_type, n)
            if perm_res is not None:
                sg.p_value = float(perm_res.p_raw[idx])
                sg.p_value_corrected = float(perm_res.p_adj[idx])
                sg.correction_method = "westfall_young_maxt"
                sg.quality_measures["p_bh"] = float(perm_res.p_bh[idx])
                sg.quality_measures["inference"] = (
                    f"cell {cfg.cell_statistic} standardized by permutation; rows permuted; B={B}; "
                    f"max-T over {len(records)} cells")
            subgroups.append(sg)
        self.timings["subgroups"] = time.perf_counter() - t2

        self._all_subgroups = subgroups
        result = SetOfSubgroups(subgroups).prune(QualityMeasures.quality, s_min=cfg.redundancy_threshold)
        self._result = result
        self.timings["total"] = time.perf_counter() - t0
        if cfg.verbose:
            n_sig = sum(1 for sg in subgroups if sg.p_value_corrected is not None and sg.p_value_corrected < cfg.alpha)
            print(f"  {len(result)} subgroups after pruning; {n_sig} of {len(subgroups)} cells significant "
                  f"at FWER {cfg.alpha} (total {self.timings['total']:.2f}s)")
        return result

    # ------------------------------------------------------------ partitions
    @staticmethod
    def _make_permutations(rng: np.random.Generator, n: int, B: int) -> List[np.ndarray]:
        """B row permutations, in chunks of at most ~2M entries."""
        if B <= 0:
            return []
        chunk = max(1, min(B, 2_000_000 // max(n, 1)))
        out = []
        done = 0
        while done < B:
            b = min(chunk, B - done)
            out.append(np.argsort(rng.random((b, n)), axis=1))
            done += b
        return out

    def _build_partitions(self, fs: FeatureSpace, algorithms: List[str], k_values: List[int],
                          use_natural: bool, rng: np.random.Generator, side: str) -> List[PartitionConfiguration]:
        configs: List[PartitionConfiguration] = []
        store = self._partition_results_X if side == "X" else self._partition_results_Y
        store.clear()
        n = fs.n
        for alg in algorithms:
            clusterer = ClusteringFactory.get_clusterer(alg)   # ValueError for unknown names propagates
            for k in k_values:
                if k < 2 or k >= n:
                    logger.warning("MRSD-PT: skipping %s k=%d on %s (need 2 <= k < n=%d)", alg, k, side, n)
                    continue
                seed = derive_seed(rng)
                try:
                    res = clusterer.fit(fs.numeric, k=k, random_state=seed)
                except ValueError as e:
                    logger.warning("MRSD-PT: %s k=%d on %s failed: %s", alg, k, side, e)
                    continue
                if res.k < 2:
                    logger.warning("MRSD-PT: %s k=%d on %s produced a single cluster; skipped", alg, k, side)
                    continue
                if res.k != k:
                    logger.info("MRSD-PT: %s k=%d on %s produced %d clusters", alg, k, side, res.k)
                configs.append(PartitionConfiguration(alg, k, res))
                store[f"{alg}_{k}"] = res
        if use_natural:
            nat = _natural_partition(fs, self.config.max_natural_partitions)
            if nat is not None and nat.k >= 2:
                configs.append(PartitionConfiguration("natural", nat.k, nat))
                store["natural"] = nat
        return configs

    # ------------------------------------------------------------- subgroup
    def _create_subgroup(self, rec: dict, fs_X: FeatureSpace, dataset: Dataset,
                         f_type: FDivergenceType, n: int) -> Subgroup:
        cx: PartitionConfiguration = rec["cx"]
        cy: PartitionConfiguration = rec["cy"]
        i, j = rec["i"], rec["j"]
        rx, ry = cx.result, cy.result
        mask = (rx.labels == i) & (ry.labels == j)
        indices = set(int(t) for t in np.flatnonzero(mask))

        # readable description
        desc_x = self._side_description("X", cx, i, "k")
        desc_y = self._side_description("Y", cy, j, "l")
        description = f"{desc_x} | {desc_y}"

        # rule in X space
        if cx.algorithm == "natural":
            values = rx.metadata["partition_values"][i]
            rule = _natural_rule(fs_X, values)
            covered = rule.evaluate(fs_X)
            inter = int((covered & mask).sum())
            precision = inter / max(int(covered.sum()), 1)
            recall = inter / max(int(mask.sum()), 1)
        else:
            d = describe_mask_with_tree(fs_X, mask)
            rule, precision, recall = d.rule, d.precision, d.recall

        def _pick(meta, key, idx):
            arr = meta.get(key)
            return arr[idx] if arr is not None and idx < len(arr) else None

        sg = Subgroup(
            description=description,
            covered_instances=indices,
            algorithm="mrsd_pt",
            rule=rule,
            description_precision=float(precision),
            description_recall=float(recall),
            clustering_X=cx.algorithm, k=rx.k, cell_i=i,
            clustering_Y=cy.algorithm, l=ry.k, cell_j=j,
            f_divergence_type=f_type.value,
            f_divergence_value=rec["f_table"],
            observed_count=rec["observed"],
            expected_count=rec["expected"],
            medoid_X=_pick(rx.metadata, "medoids", i),
            medoid_Y=_pick(ry.metadata, "medoids", j),
            medoid_index_X=(int(rx.metadata["medoid_indices"][i]) if "medoid_indices" in rx.metadata else None),
            medoid_index_Y=(int(ry.metadata["medoid_indices"][j]) if "medoid_indices" in ry.metadata else None),
            centroid_X=_pick(rx.metadata, "centroids", i),
            centroid_Y=_pick(ry.metadata, "centroids", j),
            partition_result_X=rx,
            partition_result_Y=ry,
            clustering_metadata_X={"algorithm": cx.algorithm, "k_requested": cx.k, "k_actual": rx.k},
            clustering_metadata_Y={"algorithm": cy.algorithm, "k_requested": cy.k, "k_actual": ry.k},
        )
        sg.quality_measures.update({
            "quality": rec["z"],
            "cell_statistic": rec["stat"],
            "cell_statistic_type": self.config.cell_statistic,
            "observed": rec["observed"],
            "expected": rec["expected"],
            "f_divergence_table": rec["f_table"],
        })
        if rec["p_table"] is not None:
            sg.quality_measures["p_table"] = rec["p_table"]
        sg.fill_target_statistics(dataset.Y)
        return sg

    @staticmethod
    def _side_description(side: str, cfg: PartitionConfiguration, cell: int, kname: str) -> str:
        if cfg.algorithm == "natural":
            vals = cfg.result.metadata["partition_values"][cell]
            return f"{side}: natural " + ", ".join(f"{c}={v}" for c, v in vals.items())
        return f"{side}: {cfg.algorithm} {kname}={cfg.result.k} cluster {cell}"

    # ---------------------------------------------------------------- access
    def get_all_candidates(self) -> SetOfSubgroups:
        """Every cell examined (unpruned), with p-values."""
        return SetOfSubgroups(self._all_subgroups)

    def get_significant_subgroups(self, alpha: Optional[float] = None,
                                  all_candidates: bool = False) -> SetOfSubgroups:
        """Subgroups with max-T adjusted p-value < alpha (from the pruned result,
        or from all candidates when `all_candidates=True`)."""
        if alpha is None:
            alpha = self.config.alpha
        src = self.get_all_candidates() if all_candidates else (self._result or SetOfSubgroups())
        return SetOfSubgroups([sg for sg in src if sg.p_value_corrected is not None and sg.p_value_corrected < alpha])

    def get_partition_results(self) -> Tuple[Dict[str, PartitionResult], Dict[str, PartitionResult]]:
        return dict(self._partition_results_X), dict(self._partition_results_Y)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_mrsdpt(data: Union[pd.DataFrame, str], X_columns: List[str], Y_columns: List[str],
               **config_kwargs) -> SetOfSubgroups:
    """Run MRSD-PT on a DataFrame or a file path; returns the pruned, validated set."""
    if isinstance(data, str):
        dataset = DataLoader.load(data, X_columns=X_columns, Y_columns=Y_columns)
    else:
        dataset = DataLoader.from_dataframe(data, X_columns=X_columns, Y_columns=Y_columns)
    config = MRSDPTConfig(**config_kwargs)
    return MRSDPT(config).fit(dataset)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.choice(["a", "b", "c"], n)
    planted = x1 > 1.0
    y1 = rng.normal(size=n) + 2.0 * planted
    y2 = rng.normal(size=n) - 1.0 * planted
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y1": y1, "y2": y2})

    algo = MRSDPT(MRSDPTConfig(k_values=[2, 3, 4], clustering_algorithms_X=["kmeans"],
                               clustering_algorithms_Y=["kmeans"], n_permutations=1000,
                               n_min=10, random_state=0, verbose=True))
    result = algo.fit(DataLoader.from_dataframe(df, ["x1", "x2", "x3"], ["y1", "y2"]))
    best = min(result, key=lambda s: (s.p_value_corrected, -s.quality))
    planted_idx = set(np.flatnonzero(planted))
    jac = len(best.covered_instances & planted_idx) / len(best.covered_instances | planted_idx)
    print(f"\nBest cell: {best}\n  rule: {best.rule} (prec={best.description_precision:.2f}, "
          f"rec={best.description_recall:.2f})\n  Jaccard with planted subgroup: {jac:.2f} -> "
          f"{'recovered' if jac >= 0.5 and best.p_value_corrected < 0.05 else 'NOT recovered'}")
