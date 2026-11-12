"""
Choosing an "optimal" non-redundant set of subgroups from many candidates
(typically the union of the outputs of several algorithms) on the basis of
their pairwise similarity.

Two procedures are offered.

A. Cleaning (Umek, doctoral dissertation; `clean`): repeatedly take the best
   candidate by a quality measure q, add it to the result and remove every
   candidate at least s_min similar to it.  Equivalent to
   `SetOfSubgroups.filter_redundant` of the MRSD package, here with a choice
   of q that is meaningful across algorithms (adjusted p-value, rank within
   the algorithm, ...) because the algorithms' own quality scales differ.

B. Clustering with representatives (`consolidate`):
1. Similarity of two subgroups = Jaccard (default), overlap coefficient or
   Dice coefficient of their instance sets.
2. Agglomerative clustering of the candidates on distance 1 - similarity
   (average / complete / single linkage).  Clusters are cut either at a
   similarity threshold (members of a cluster are at least `threshold`
   similar, on average for average linkage) or into a fixed number k.
3. Every cluster gets a *support* = number of distinct algorithms whose
   candidates fall in it (a subgroup found independently by several methods
   is more credible), and a *representative* chosen by one of
       'best'     – best ranked member (see 4.),
       'medoid'   – member with the largest mean similarity to the others,
       'largest'  – largest member,
       'smallest' – smallest member;
4. Ranking of members / clusters uses, in this order of priority, the
   criteria listed in `criteria`:
       'support'  – more supporting algorithms first,
       'p_adj'    – smaller adjusted p-value first (comparable across algorithms),
       'p'        – smaller raw p-value first,
       'rank'     – better within-algorithm rank first (quality scales differ
                    between algorithms, ranks do not),
       'size'     – larger first.
5. Clusters with fewer than `min_support` algorithms are dropped; the
   representatives of the remaining clusters, in rank order, are the result.

Nothing here needs Orange, so it is unit-testable on its own.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from orangecontrib.subgroups.backend import SubgroupResults

SIMILARITIES = ("jaccard", "overlap", "dice")
LINKAGES = ("average", "complete", "single")
REPRESENTATIVES = ("best", "medoid", "largest", "smallest")
CRITERIA = ("support", "p_adj", "p", "rank", "size")


def similarity_matrix(results: SubgroupResults, measure: str = "jaccard") -> np.ndarray:
    sets = [set(sg.covered_instances) for sg in results.subgroups]
    n = len(sets)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            inter = len(a & b)
            if measure == "jaccard":
                d = len(a | b)
            elif measure == "overlap":
                d = min(len(a), len(b))
            elif measure == "dice":
                d = (len(a) + len(b)) / 2.0
            else:
                raise ValueError(f"unknown similarity {measure!r}")
            S[i, j] = S[j, i] = inter / d if d else 0.0
    return S


def within_algorithm_ranks(results: SubgroupResults) -> np.ndarray:
    """Rank (1 = best) of every subgroup within the subgroups of the same
    algorithm, by quality (descending); ties keep input order."""
    algs = [sg.algorithm or results.algorithm for sg in results.subgroups]
    ranks = np.zeros(len(results), dtype=int)
    for alg in set(algs):
        idx = [i for i, a in enumerate(algs) if a == alg]
        qual = [results[i].quality if results[i].quality is not None else -math.inf for i in idx]
        order = sorted(range(len(idx)), key=lambda k: -qual[k])
        for r, k in enumerate(order):
            ranks[idx[k]] = r + 1
    return ranks


@dataclasses.dataclass
class Cluster:
    label: int
    members: List[int]                 # indices into the results
    algorithms: List[str]              # distinct supporting algorithms
    representative: int
    mean_similarity: float             # mean pairwise similarity of the members (1 for singletons)
    best_p_adj: Optional[float]

    @property
    def support(self) -> int:
        return len(self.algorithms)


@dataclasses.dataclass
class ConsolidationResult:
    clusters: List[Cluster]            # in rank order (best first), after the support filter
    dropped: List[Cluster]             # clusters below min_support
    labels: np.ndarray                 # cluster label of every candidate
    similarity: np.ndarray
    ranks: np.ndarray                  # within-algorithm ranks of every candidate

    @property
    def representatives(self) -> List[int]:
        return [c.representative for c in self.clusters]


def _sort_key(results: SubgroupResults, ranks: np.ndarray, support: Dict[int, int], criteria: Sequence[str]):
    def key(i: int):
        sg = results[i]
        parts = []
        for c in criteria:
            if c == "support":
                parts.append(-support.get(i, 1))
            elif c == "p_adj":
                parts.append(sg.p_value_corrected if sg.p_value_corrected is not None else 2.0)
            elif c == "p":
                parts.append(sg.p_value if sg.p_value is not None else 2.0)
            elif c == "rank":
                parts.append(int(ranks[i]))
            elif c == "size":
                parts.append(-sg.coverage)
            else:
                raise ValueError(f"unknown criterion {c!r}")
        parts.append(i)
        return tuple(parts)
    return key


def consolidate(results: SubgroupResults, measure: str = "jaccard", linkage_method: str = "average",
                threshold: Optional[float] = 0.5, n_clusters: Optional[int] = None,
                representative: str = "best", criteria: Sequence[str] = ("support", "p_adj", "rank", "size"),
                min_support: int = 1) -> ConsolidationResult:
    """Cluster the candidates by similarity and pick one representative per cluster.

    Exactly one of `threshold` (similarity in [0, 1]; a cluster is cut where
    the linkage distance exceeds 1 - threshold) and `n_clusters` is used;
    `n_clusters` wins when both are given."""
    n = len(results)
    if n == 0:
        return ConsolidationResult([], [], np.zeros(0, dtype=int), np.zeros((0, 0)), np.zeros(0, dtype=int))
    S = similarity_matrix(results, measure)
    ranks = within_algorithm_ranks(results)
    if n == 1:
        labels = np.ones(1, dtype=int)
    else:
        D = np.clip(1.0 - S, 0.0, 1.0)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method=linkage_method)
        if n_clusters is not None:
            labels = fcluster(Z, t=max(1, min(int(n_clusters), n)), criterion="maxclust")
        else:
            t = 1.0 - float(threshold if threshold is not None else 0.5)
            labels = fcluster(Z, t=t + 1e-12, criterion="distance")
    algs = [sg.algorithm or results.algorithm for sg in results.subgroups]

    clusters: List[Cluster] = []
    for lab in sorted(set(labels.tolist())):
        members = [i for i in range(n) if labels[i] == lab]
        sup_algs = sorted({algs[i] for i in members})
        support = {i: len(sup_algs) for i in members}
        key = _sort_key(results, ranks, support, criteria)
        if representative == "best":
            rep = min(members, key=key)
        elif representative == "medoid":
            sums = {i: float(S[i, members].sum()) for i in members}
            best = max(sums.values())
            rep = min((i for i in members if abs(sums[i] - best) < 1e-12), key=key)   # ties -> best ranked
        elif representative == "largest":
            rep = max(members, key=lambda i: (results[i].coverage, -members.index(i)))
        elif representative == "smallest":
            rep = min(members, key=lambda i: (results[i].coverage, members.index(i)))
        else:
            raise ValueError(f"unknown representative rule {representative!r}")
        if len(members) > 1:
            sub = S[np.ix_(members, members)]
            mean_sim = float((sub.sum() - len(members)) / (len(members) * (len(members) - 1)))
        else:
            mean_sim = 1.0
        p_adjs = [results[i].p_value_corrected for i in members if results[i].p_value_corrected is not None]
        clusters.append(Cluster(int(lab), members, sup_algs, rep, mean_sim, min(p_adjs) if p_adjs else None))

    support_by_rep = {c.representative: c.support for c in clusters}
    key = _sort_key(results, ranks, support_by_rep, criteria)
    clusters.sort(key=lambda c: key(c.representative))
    kept = [c for c in clusters if c.support >= min_support]
    dropped = [c for c in clusters if c.support < min_support]
    return ConsolidationResult(kept, dropped, np.asarray(labels, dtype=int), S, ranks)


QUALITIES = ("quality", "p_adj", "p", "size", "rank", "criteria", "common_distinct_Y", "common_describe_X",
             "common_relation", "common_best")


def quality_function(results: SubgroupResults, quality: str = "quality", criteria: Sequence[str] = CRITERIA):
    """q(G) for the cleaning procedure (higher is better).

    'quality'  – the algorithm's own search score (comparable only within one algorithm);
    'p_adj'    – 1 - adjusted p-value;  'p' – 1 - raw p-value;
    'size'     – coverage;
    'rank'     – minus the rank within the algorithm (1 = best; comparable across algorithms);
    'criteria' – lexicographic combination of `criteria` (support is ignored here).
    Ties are broken by the input order (earlier first)."""
    ranks = within_algorithm_ranks(results)
    if quality == "quality":
        return lambda i: (results[i].quality if results[i].quality is not None else -math.inf, -i)
    if quality == "p_adj":
        return lambda i: (-(results[i].p_value_corrected if results[i].p_value_corrected is not None else 2.0), -i)
    if quality == "p":
        return lambda i: (-(results[i].p_value if results[i].p_value is not None else 2.0), -i)
    if quality == "size":
        return lambda i: (results[i].coverage, -i)
    if quality == "rank":
        return lambda i: (-int(ranks[i]), -i)
    if quality == "common_best":       # the better of Y distinctness and relation exceptionality
        def best(i):
            qm = results[i].quality_measures or {}
            vals = [qm.get("common_distinct_Y"), qm.get("common_relation")]
            vals = [v for v in vals if v is not None and not math.isnan(v)]
            return (max(vals) if vals else -math.inf, -i)
        return best
    if quality.startswith("common_"):  # a common measure stored in quality_measures (see common_measures.py)
        def common(i):
            v = (results[i].quality_measures or {}).get(quality)
            return (v if v is not None and not math.isnan(v) else -math.inf, -i)
        return common
    if quality == "criteria":
        key = _sort_key(results, ranks, {}, [c for c in criteria if c != "support"])
        return lambda i: tuple(-v for v in key(i))
    raise ValueError(f"unknown quality {quality!r}")


def clean(results: SubgroupResults, measure: str = "jaccard", s_min: float = 0.5, quality: str = "quality",
          criteria: Sequence[str] = ("p_adj", "rank", "size"), max_subgroups: Optional[int] = None
          ) -> List[int]:
    r"""Cleaning of a set of interesting subgroups (Umek, doctoral dissertation):

        G' <- {}
        while |G| > 0:
            G_i <- argmax_{G in G} q(G)
            G' <- G' + {G_i}
            for G_j in G: if s(G_i, G_j) >= s_min: G <- G \ {G_j}
        return G'

    Input: the set G (the candidates), similarity s, threshold s_min and quality q;
    output: the indices of the cleaned set G' in the order they were chosen.
    Because s(G_i, G_i) = 1 >= s_min, the chosen subgroup itself leaves G as well."""
    n = len(results)
    if n == 0:
        return []
    S = similarity_matrix(results, measure)
    q = quality_function(results, quality, criteria)
    remaining = set(range(n))
    chosen: List[int] = []
    while remaining:
        best = max(remaining, key=q)
        chosen.append(best)
        remaining = {j for j in remaining if S[best, j] < s_min}
        if max_subgroups and len(chosen) >= max_subgroups:
            break
    return chosen


def greedy_selection(results: SubgroupResults, measure: str = "jaccard", threshold: float = 0.5,
                     criteria: Sequence[str] = ("p_adj", "rank", "size"), max_subgroups: Optional[int] = None
                     ) -> List[int]:
    """`clean` with q = the lexicographic ranking criteria (kept for compatibility)."""
    return clean(results, measure, threshold, "criteria", criteria, max_subgroups)
