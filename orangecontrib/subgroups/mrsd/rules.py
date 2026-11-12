"""
Rule representation, condition generation, beam search and post-hoc rule
extraction shared by all MRSD algorithms.

Every subgroup returned by the package should carry a `Rule` in the original
units of the X variables (labels for categorical attributes, raw values for
numeric ones) so that the result is interpretable.  Algorithms that search the
rule space directly (MRSD-Rules, MRSD-EMM, MRSD-MI) use `BeamSearch`;
algorithms that produce index sets (clusters, dendrogram nodes, local search)
use `describe_mask_with_tree` to obtain an approximate description and report
how well it matches the set (precision / recall / Jaccard).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .common import FeatureSpace


# =============================================================================
# CONDITIONS AND RULES
# =============================================================================

def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


@dataclass(frozen=True)
class Condition:
    """
    Single condition on one attribute.

    operator: '=' , '!=' (categorical, `value` is a label),
              'in' (categorical set membership, `value` is a tuple of labels),
              '<', '>=', 'range' (numeric; 'range' is value <= x < value_upper,
              closed on the right when `closed_upper` is True)
    Values are always in ORIGINAL units.
    """
    attribute: str
    attribute_idx: int
    operator: str
    value: object
    value_upper: Optional[float] = None
    closed_upper: bool = False
    closed_lower: bool = True

    def __str__(self) -> str:
        a = self.attribute
        if self.operator == "range":
            rb = "]" if self.closed_upper else ")"
            lb = "[" if self.closed_lower else "("
            return f"{a} ∈ {lb}{_fmt(self.value)}, {_fmt(self.value_upper)}{rb}"
        if self.operator == "in":
            return f"{a} ∈ {{{', '.join(str(v) for v in self.value)}}}"
        if self.operator in ("<", ">=", "<=", ">"):
            return f"{a} {self.operator} {_fmt(self.value)}"
        return f"{a} {self.operator} {self.value}"

    def evaluate(self, fs: FeatureSpace) -> np.ndarray:
        """Boolean mask over the rows of `fs`."""
        j = self.attribute_idx
        col = fs.codes[:, j]
        if fs.is_categorical[j]:
            labels = fs.categories[j]
            if self.operator == "=":
                code = labels.index(self.value)
                return col == code
            if self.operator == "!=":
                code = labels.index(self.value)
                return col != code
            if self.operator == "in":
                codes = [labels.index(v) for v in self.value]
                return np.isin(col, codes)
            raise ValueError(f"Operator {self.operator} not valid for categorical attribute")
        if self.operator == "<":
            return col < self.value
        if self.operator == "<=":
            return col <= self.value
        if self.operator == ">":
            return col > self.value
        if self.operator == ">=":
            return col >= self.value
        if self.operator == "range":
            upper = (col <= self.value_upper) if self.closed_upper else (col < self.value_upper)
            lower = (col >= self.value) if self.closed_lower else (col > self.value)
            return lower & upper
        raise ValueError(f"Unknown operator {self.operator}")


@dataclass
class Rule:
    """Conjunction of conditions."""
    conditions: List[Condition] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.conditions:
            return "TRUE"
        return " AND ".join(str(c) for c in sorted(self.conditions, key=lambda c: c.attribute_idx))

    def __len__(self) -> int:
        return len(self.conditions)

    def key(self) -> Tuple:
        return tuple(sorted((c.attribute_idx, c.operator, str(c.value), str(c.value_upper)) for c in self.conditions))

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        return isinstance(other, Rule) and self.key() == other.key()

    def evaluate(self, fs: FeatureSpace) -> np.ndarray:
        mask = np.ones(fs.n, dtype=bool)
        for c in self.conditions:
            mask &= c.evaluate(fs)
        return mask

    def uses_attribute(self, j: int) -> bool:
        return any(c.attribute_idx == j for c in self.conditions)

    def extended(self, cond: Condition) -> "Rule":
        """New rule with one more condition. Two numeric conditions on the same
        attribute are merged into their intersection (a range) when possible."""
        conds = list(self.conditions)
        for i, c in enumerate(conds):
            if c.attribute_idx != cond.attribute_idx:
                continue
            if c.operator in _NUM_OPS and cond.operator in _NUM_OPS:
                merged = _merge_numeric(c, cond)
            elif c.operator in _CAT_OPS and cond.operator in _CAT_OPS:
                merged = _merge_categorical(c, cond)
            else:
                merged = None
            if merged is not None:
                conds[i] = merged
                return Rule(conds)
        conds.append(cond)
        return Rule(conds)


_NUM_OPS = ("<", "<=", ">", ">=", "range")
_CAT_OPS = ("=", "!=", "in")


def _merge_categorical(a: Condition, b: Condition) -> Optional[Condition]:
    """Intersection of the allowed label sets. '!=' needs the full label set, so it
    is only merged with '=' / 'in' (where the result is explicit)."""
    def allowed(c):
        if c.operator == "=":
            return {c.value}
        if c.operator == "in":
            return set(c.value)
        return None
    sa, sb = allowed(a), allowed(b)
    if sa is None and sb is None:
        return None
    if sa is None:
        sa, sb = sb, sa
        a, b = b, a
    if sb is None:  # b is '!='
        inter = sa - {b.value}
    else:
        inter = sa & sb
    if not inter:
        return None
    if len(inter) == 1:
        return Condition(a.attribute, a.attribute_idx, "=", next(iter(inter)))
    return Condition(a.attribute, a.attribute_idx, "in", tuple(sorted(inter, key=str)))


def _bounds(c: Condition) -> Tuple[float, bool, float, bool]:
    """(lower, closed_lower, upper, closed_upper)"""
    if c.operator == ">=":
        return float(c.value), True, np.inf, False
    if c.operator == ">":
        return float(c.value), False, np.inf, False
    if c.operator == "<":
        return -np.inf, False, float(c.value), False
    if c.operator == "<=":
        return -np.inf, False, float(c.value), True
    return float(c.value), bool(c.closed_lower), float(c.value_upper), bool(c.closed_upper)


def _merge_numeric(a: Condition, b: Condition) -> Optional[Condition]:
    lo_a, cla, hi_a, cha = _bounds(a)
    lo_b, clb, hi_b, chb = _bounds(b)
    if lo_a > lo_b:
        lo, cl = lo_a, cla
    elif lo_b > lo_a:
        lo, cl = lo_b, clb
    else:
        lo, cl = lo_a, cla and clb
    if hi_a < hi_b:
        hi, ch = hi_a, cha
    elif hi_b < hi_a:
        hi, ch = hi_b, chb
    else:
        hi, ch = hi_a, cha and chb
    if hi < lo or (hi == lo and not (cl and ch)):
        return None
    if np.isinf(lo) and np.isinf(hi):
        return None
    if np.isinf(hi):
        return Condition(a.attribute, a.attribute_idx, ">=" if cl else ">", lo)
    if np.isinf(lo):
        return Condition(a.attribute, a.attribute_idx, "<=" if ch else "<", hi)
    return Condition(a.attribute, a.attribute_idx, "range", lo, hi, ch, cl)


# =============================================================================
# CONDITION GENERATOR
# =============================================================================

class ConditionGenerator:
    """
    Generates candidate conditions for one attribute given the current coverage.

    Numeric attributes: `n_bins` quantile cut points (computed on the whole
    column once, or re-computed inside the current coverage when
    `local_cutpoints=True`), yielding '<', '>=' and 'range' conditions; the last
    range is closed so that the column maximum is covered.
    Categorical attributes: '=' for every level, and '!=' when `allow_negation`.
    """

    def __init__(self, fs: FeatureSpace, n_bins: int = 5, min_support: int = 10,
                 local_cutpoints: bool = False, allow_negation: bool = False,
                 use_ranges: bool = True):
        self.fs = fs
        self.n_bins = n_bins
        self.min_support = min_support
        self.local_cutpoints = local_cutpoints
        self.allow_negation = allow_negation
        self.use_ranges = use_ranges
        self._global_edges: Dict[int, np.ndarray] = {}
        for j in range(fs.p):
            if not fs.is_categorical[j]:
                self._global_edges[j] = self._edges(fs.codes[:, j])

    def _edges(self, col: np.ndarray) -> np.ndarray:
        if len(col) == 0:
            return np.array([])
        uniq = np.unique(col)
        if len(uniq) <= self.n_bins + 1:
            # few distinct values (e.g. integer scores): cut at every value
            return uniq.astype(float)
        qs = np.linspace(0, 100, self.n_bins + 1)
        return np.unique(np.percentile(col, qs))

    def conditions_for(self, j: int, current_mask: np.ndarray) -> List[Condition]:
        fs = self.fs
        name = fs.columns[j]
        col = fs.codes[:, j]
        out: List[Condition] = []
        if fs.is_categorical[j]:
            labels = fs.categories[j]
            for code, lab in enumerate(labels):
                m = (col == code) & current_mask
                if m.sum() >= self.min_support:
                    out.append(Condition(name, j, "=", lab))
                if self.allow_negation:
                    m2 = (col != code) & current_mask
                    if m2.sum() >= self.min_support and m2.sum() < current_mask.sum():
                        out.append(Condition(name, j, "!=", lab))
            return out
        edges = self._edges(col[current_mask]) if self.local_cutpoints else self._global_edges[j]
        if len(edges) < 2:
            return out
        for e in edges[1:-1]:
            m = (col >= e) & current_mask
            if self.min_support <= m.sum() < current_mask.sum():
                out.append(Condition(name, j, ">=", float(e)))
            m = (col < e) & current_mask
            if self.min_support <= m.sum() < current_mask.sum():
                out.append(Condition(name, j, "<", float(e)))
        if self.use_ranges and len(edges) > 2:
            for i in range(len(edges) - 1):
                lo, hi = float(edges[i]), float(edges[i + 1])
                closed = i == len(edges) - 2
                m = (col >= lo) & ((col <= hi) if closed else (col < hi)) & current_mask
                if self.min_support <= m.sum() < current_mask.sum():
                    out.append(Condition(name, j, "range", lo, hi, closed))
        return out


# =============================================================================
# BEAM SEARCH
# =============================================================================

@dataclass
class SearchResult:
    rule: Rule
    mask: np.ndarray
    quality: float
    depth: int

    @property
    def size(self) -> int:
        return int(self.mask.sum())


class BeamSearch:
    """
    Generic beam search over conjunctive rules.

    Args:
        generator: ConditionGenerator
        quality_fn: mask -> float (higher is better)
        beam_width, max_depth: search parameters
        n_min, n_max: coverage limits for a rule to be a candidate
        refine_same_attribute: allow a second numeric condition on an attribute
            already used (ranges are intersected)
    Covering weights, if any, are the caller's business: close over them in
    `quality_fn`.
    """

    def __init__(self, generator: ConditionGenerator, quality_fn: Callable,
                 beam_width: int = 10, max_depth: int = 3,
                 n_min: int = 10, n_max: Optional[int] = None,
                 refine_same_attribute: bool = True):
        self.gen = generator
        self.quality_fn = quality_fn
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.n_min = n_min
        self.n_max = n_max
        self.refine_same_attribute = refine_same_attribute
        self.n_evaluated = 0

    def search(self, start_mask: Optional[np.ndarray] = None,
               include_root: bool = False) -> List[SearchResult]:
        """
        Returns all distinct candidates (by covered set) visited in the beam,
        sorted by quality (best first).  `n_evaluated` counts quality
        evaluations (useful for multiple-testing accounting).
        """
        fs = self.gen.fs
        n = fs.n
        n_max = self.n_max if self.n_max is not None else n
        root_mask = np.ones(n, dtype=bool) if start_mask is None else start_mask.copy()
        root = SearchResult(Rule([]), root_mask, -np.inf, 0)
        seen: Dict[bytes, SearchResult] = {}
        self.n_evaluated = 0
        if include_root and self.n_min <= root_mask.sum() <= n_max:
            root.quality = float(self.quality_fn(root_mask))
            self.n_evaluated += 1
            seen[np.packbits(root_mask).tobytes()] = root
        beam = [root]
        for depth in range(1, self.max_depth + 1):
            candidates: Dict[bytes, SearchResult] = {}
            for parent in beam:
                for j in range(fs.p):
                    if parent.rule.uses_attribute(j) and not (self.refine_same_attribute and not fs.is_categorical[j]):
                        continue
                    for cond in self.gen.conditions_for(j, parent.mask):
                        rule = parent.rule.extended(cond)
                        if len(rule) == len(parent.rule) and rule == parent.rule:
                            continue
                        mask = parent.mask & cond.evaluate(fs)
                        size = int(mask.sum())
                        if size < self.n_min or size > n_max or size == parent.size:
                            continue
                        key = np.packbits(mask).tobytes()
                        if key in seen or key in candidates:
                            continue
                        q = float(self.quality_fn(mask))
                        self.n_evaluated += 1
                        candidates[key] = SearchResult(rule, mask, q, depth)
            if not candidates:
                break
            ranked = sorted(candidates.values(), key=lambda r: r.quality, reverse=True)
            beam = ranked[: self.beam_width]
            seen.update(candidates)
        return sorted(seen.values(), key=lambda r: r.quality, reverse=True)


# =============================================================================
# POST-HOC DESCRIPTION OF AN INDEX SET
# =============================================================================

@dataclass
class Description:
    rule: Rule
    precision: float   # |rule ∩ G| / |rule|
    recall: float      # |rule ∩ G| / |G|
    jaccard: float

    def __str__(self):
        return f"{self.rule}  (precision={self.precision:.2f}, recall={self.recall:.2f})"


def describe_mask_with_tree(fs: FeatureSpace, mask: np.ndarray, max_depth: int = 3,
                            min_samples_leaf: int = 5, random_state: Optional[int] = 0) -> Description:
    """
    Fit a shallow decision tree (membership ~ X) and return the path to the
    positive leaf with the best F1 against the mask as a Rule in original
    units, together with how well that rule reproduces the set.
    """
    y = mask.astype(int)
    n = fs.n
    if y.sum() == 0 or y.sum() == n:
        return Description(Rule([]), float(y.mean()), 1.0, float(y.mean()))
    X = fs.codes
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  class_weight="balanced", random_state=random_state)
    tree.fit(X, y)
    t = tree.tree_
    best: Optional[Description] = None

    def walk(node: int, conds: List[Condition]):
        nonlocal best
        if t.children_left[node] == -1:
            rule = Rule(list(conds))
            m = rule.evaluate(fs) if conds else np.ones(n, dtype=bool)
            inter = int((m & mask).sum())
            if inter == 0:
                return
            prec = inter / m.sum()
            rec = inter / mask.sum()
            jac = inter / (m | mask).sum()
            f1 = 2 * prec * rec / (prec + rec)
            if best is None or f1 > best._f1:
                d = Description(rule, prec, rec, jac)
                d._f1 = f1
                best = d
            return
        j = int(t.feature[node])
        thr = float(t.threshold[node])
        name = fs.columns[j]
        if fs.is_categorical[j]:
            labels = fs.categories[j]
            left_codes = [c for c in range(len(labels)) if c <= thr]
            right_codes = [c for c in range(len(labels)) if c > thr]
            left_c = Condition(name, j, "in", tuple(labels[c] for c in left_codes)) if len(left_codes) > 1 else Condition(name, j, "=", labels[left_codes[0]])
            right_c = Condition(name, j, "in", tuple(labels[c] for c in right_codes)) if len(right_codes) > 1 else Condition(name, j, "=", labels[right_codes[0]])
            walk(int(t.children_left[node]), conds + [left_c])
            walk(int(t.children_right[node]), conds + [right_c])
        else:
            walk(int(t.children_left[node]), conds + [Condition(name, j, "<=", thr)])
            walk(int(t.children_right[node]), conds + [Condition(name, j, ">", thr)])

    walk(0, [])
    if best is None:
        return Description(Rule([]), float(y.mean()), 1.0, float(y.mean()))
    # merge numeric conditions on the same attribute into ranges
    merged = Rule([])
    for c in best.rule.conditions:
        merged = merged.extended(c)
    m = merged.evaluate(fs)
    inter = int((m & mask).sum())
    return Description(merged, inter / max(m.sum(), 1), inter / mask.sum(), inter / max((m | mask).sum(), 1))
