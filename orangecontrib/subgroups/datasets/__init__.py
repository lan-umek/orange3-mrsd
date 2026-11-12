"""
Example datasets for subgroup discovery with several dependent variables.

The directory is registered as an Orange dataset search path (entry point
``orange.data.io.search_paths``), so ``Orange.data.Table("customers")`` and
the File widget find the files by name.  Each ``.tab`` file flags the
dependent variables as class variables and carries a meta column ``planted``
with the ground truth of the planted subgroup(s).
"""
import os
from typing import Dict, List, Tuple

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))

# name -> (title, description)
DATASETS: Dict[str, Tuple[str, str]] = {
    "planted_shift": (
        "Planted shift (synthetic)",
        "300 rows. X: x1, x2 (numeric), x3 (a/b/c). Y: y1, y2 (numeric), y3 (u/v). "
        "Planted subgroup: x1 > 1 (about 16 % of rows) has y1 shifted by +1.5 SD and y2 by −0.75 SD."),
    "planted_rule": (
        "Planted rule (synthetic)",
        "300 rows. X: age, edu (low/mid/high), inc. Y: y1, y2. "
        "Planted subgroup: edu = high AND age ≥ 50 has y1 shifted by +1.5 SD and y2 by +0.75 SD."),
    "pure_noise": (
        "Pure noise (synthetic)",
        "300 rows. X: x1–x4, x_cat. Y: y1–y3. X and Y are independent – nothing should be significant "
        "(a check of the false-positive control of the permutation tests)."),
    "customers": (
        "Customers (synthetic)",
        "600 rows. X: age, income, tenure_years, region, channel, education. "
        "Y: satisfaction, monthly_spend, visits, churn_risk. Two planted subgroups: "
        "region = North AND age < 35 (higher spend and visits, lower satisfaction) and "
        "education = tertiary AND high income (higher satisfaction, lower churn risk)."),
    "students": (
        "Students (synthetic)",
        "400 rows. X: study_hours, attendance, sleep_hours, parental_education, program. "
        "Y: math, reading, science, wellbeing. Planted: sleep_hours < 6.2 AND study_hours > 5 "
        "(lower wellbeing, higher math and science) and program = arts (higher reading, lower science)."),
    "clinical": (
        "Clinical (synthetic)",
        "350 rows. X: age, bmi, smoker, activity, sex. Y: systolic, cholesterol, glucose, hdl. "
        "Planted: smoker = yes AND bmi > 28 (higher systolic pressure and glucose, lower HDL)."),
}


def dataset_path(name: str, ext: str = ".tab") -> str:
    """Absolute path of a bundled dataset (name without extension)."""
    return os.path.join(DATASETS_DIR, name + ext)


def available() -> List[str]:
    return [n for n in DATASETS if os.path.exists(dataset_path(n))]
