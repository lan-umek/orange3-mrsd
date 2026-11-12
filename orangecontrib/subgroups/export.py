"""Export of SubgroupResults to files (no Orange / Qt needed)."""
from __future__ import annotations

import json
import os
import pickle

import pandas as pd

from orangecontrib.subgroups.backend import SubgroupResults

FORMATS = [
    ("csv", "CSV – one row per subgroup (*.csv)", ".csv"),
    ("xlsx", "Excel – summary, target statistics, membership (*.xlsx)", ".xlsx"),
    ("json", "JSON – everything incl. covered rows (*.json)", ".json"),
    ("pkl", "Pickle – SubgroupResults object (*.pkl)", ".pkl"),
    ("txt", "Text report (*.txt)", ".txt"),
]


def save_results(results: SubgroupResults, path: str, fmt: str, membership: bool = True) -> None:
    summary = results.summary_frame()
    if fmt == "csv":
        summary.to_csv(path, index=False)
        if membership:
            root, _ = os.path.splitext(path)
            results.membership_frame().to_csv(root + "_membership.csv", index_label="row")
    elif fmt == "xlsx":
        with pd.ExcelWriter(path) as writer:
            summary.to_excel(writer, sheet_name="subgroups", index=False)
            results.target_effects().to_excel(writer, sheet_name="effects")
            if membership:
                results.membership_frame().to_excel(writer, sheet_name="membership", index_label="row")
            cfg = pd.DataFrame({"parameter": list(results.config.keys()),
                                "value": [str(v) for v in results.config.values()]})
            cfg.to_excel(writer, sheet_name="configuration", index=False)
    elif fmt == "json":
        d = results.to_dict()
        if not membership:
            for sg in d["subgroups"]:
                sg.pop("rows", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    elif fmt == "pkl":
        r = results.copy()
        r.data = None                      # the Orange table is not needed to reload the subgroups
        with open(path, "wb") as f:
            pickle.dump(r, f)
    elif fmt == "txt":
        with open(path, "w", encoding="utf-8") as f:
            f.write(text_report(results))
    else:
        raise ValueError(f"unknown format {fmt}")


def text_report(results: SubgroupResults) -> str:
    lines = [f"Algorithm: {results.algorithm_name}",
             f"Rows: {len(results.frame)}   X: {', '.join(results.X_columns)}   Y: {', '.join(results.Y_columns)}",
             f"Configuration: {results.config}", "", results.subgroups.summary(), ""]
    for i, sg in enumerate(results.subgroups):
        lines.append(f"[{results.name(i)}] {results.rule_text(i)}")
        lines.append(f"    n = {sg.coverage}, quality = {sg.quality}, p = {sg.p_value}, "
                     f"adjusted p = {sg.p_value_corrected}")
        for t, st in (sg.target_statistics or {}).items():
            lines.append(f"    {t}: " + ", ".join(f"{k} = {v:.3f}" for k, v in st.items()))
    return "\n".join(lines) + "\n"




_OLD_MODULES = {"core", "common", "rules", "inference", "classifiers", "clustering", "divergence", "data_loader",
                "subgroups", "mrsd", "mrsdpt", "mrsdhc", "mrsdc2_swapping", "mrsdc2_semisupervised",
                "mrsdc2_ranking", "mrsdcca", "mrsdemm", "mrsd_mv", "mrsd_mi", "mrsd_rules", "mrsd_ensemble"}


class _Unpickler(pickle.Unpickler):
    """Reads files saved before the MRSD modules became a proper package
    (their classes were pickled as ``core.Subgroup`` etc.)."""

    def find_class(self, module, name):
        if module in _OLD_MODULES:
            module = "orangecontrib.subgroups.mrsd" + ("" if module == "mrsd" else "." + module)
        return super().find_class(module, name)


def load_pickle(path: str):
    """Load an object saved by ``save_results(..., fmt='pickle')``."""
    with open(path, "rb") as f:
        return _Unpickler(f).load()
