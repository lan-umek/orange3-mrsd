"""
Generate the example datasets shipped with the add-on
(orangecontrib/subgroups/datasets/*.tab and *.csv).

Every dataset has several independent variables (X), several dependent
variables (Y, flagged as class variables in the .tab header so that Orange's
File widget assigns the roles automatically) and a meta column `planted`
with the ground truth of the planted subgroup(s), which the algorithms ignore.

    python tools/make_datasets.py
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "orangecontrib", "subgroups", "datasets")


def write_tab(df: pd.DataFrame, X, Y, metas, path):
    cols = list(X) + list(Y) + list(metas)
    df = df[cols]
    types, flags = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            types.append("continuous")
        else:
            types.append("discrete" if c not in metas or df[c].nunique() < 20 else "string")
        flags.append("class" if c in Y else "meta" if c in metas else "")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(cols) + "\n")
        f.write("\t".join(types) + "\n")
        f.write("\t".join(flags) + "\n")
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append("" if np.isnan(v) else f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")
    df.to_csv(path[:-4] + ".csv", index=False, float_format="%.4f")


def planted_shift(n=300, seed=0, effect=1.5):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    x3 = rng.choice(["a", "b", "c"], n)
    y1, y2 = rng.normal(size=n), rng.normal(size=n)
    y3 = rng.choice(["u", "v"], n)
    mask = x1 > 1.0
    y1[mask] += effect
    y2[mask] -= effect / 2
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y1": y1, "y2": y2, "y3": y3,
                       "planted": np.where(mask, "yes", "no")})
    return df, ["x1", "x2", "x3"], ["y1", "y2", "y3"], ["planted"]


def planted_rule(n=300, seed=1, effect=1.5):
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 70, n)
    edu = rng.choice(["low", "mid", "high"], n)
    inc = rng.normal(size=n)
    y1, y2 = rng.normal(size=n), rng.normal(size=n)
    mask = (edu == "high") & (age >= 50)
    y1[mask] += effect
    y2[mask] += effect / 2
    df = pd.DataFrame({"age": age, "edu": edu, "inc": inc, "y1": y1, "y2": y2,
                       "planted": np.where(mask, "yes", "no")})
    return df, ["age", "edu", "inc"], ["y1", "y2"], ["planted"]


def pure_noise(n=300, seed=2, p_x=4, p_y=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"x{i + 1}": rng.normal(size=n) for i in range(p_x)})
    df["x_cat"] = rng.choice(["a", "b", "c"], n)
    for j in range(p_y):
        df[f"y{j + 1}"] = rng.normal(size=n)
    df["planted"] = "no"
    return df, [f"x{i + 1}" for i in range(p_x)] + ["x_cat"], [f"y{j + 1}" for j in range(p_y)], ["planted"]


def customers(n=600, seed=3):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 75, n).round(0)
    income = np.exp(rng.normal(10.5, 0.4, n)).round(0)
    tenure = rng.exponential(4, n).round(1)
    region = rng.choice(["North", "South", "East", "West"], n)
    channel = rng.choice(["store", "online", "mixed"], n, p=[0.4, 0.35, 0.25])
    education = rng.choice(["primary", "secondary", "tertiary"], n, p=[0.2, 0.5, 0.3])
    satisfaction = rng.normal(6.5, 1.5, n)
    spend = np.exp(rng.normal(4.5, 0.5, n))
    visits = rng.poisson(6, n).astype(float)
    churn = rng.beta(2, 5, n)
    g1 = (region == "North") & (age < 35)
    spend[g1] *= 1.8
    visits[g1] += 4
    satisfaction[g1] -= 1.5
    g2 = (education == "tertiary") & (income > np.quantile(income, 0.7))
    satisfaction[g2] += 1.5
    churn[g2] *= 0.4
    planted = np.where(g1, "young North", np.where(g2, "rich tertiary", "no"))
    df = pd.DataFrame({"age": age, "income": income, "tenure_years": tenure, "region": region,
                       "channel": channel, "education": education,
                       "satisfaction": satisfaction.clip(0, 10).round(2), "monthly_spend": spend.round(2),
                       "visits": visits, "churn_risk": churn.clip(0, 1).round(3), "planted": planted})
    return df, ["age", "income", "tenure_years", "region", "channel", "education"], \
        ["satisfaction", "monthly_spend", "visits", "churn_risk"], ["planted"]


def students(n=400, seed=4):
    rng = np.random.default_rng(seed)
    study = rng.gamma(3, 1.5, n).round(1)
    attendance = rng.beta(6, 2, n).round(3) * 100
    sleep = rng.normal(7, 1.2, n).round(1)
    parental = rng.choice(["primary", "secondary", "tertiary"], n, p=[0.25, 0.45, 0.3])
    program = rng.choice(["science", "arts", "business"], n)
    base = 0.3 * study + 0.1 * attendance / 10 + rng.normal(0, 1, n)
    math = 60 + 5 * base + rng.normal(0, 6, n)
    reading = 60 + 4 * base + rng.normal(0, 6, n)
    science = 60 + 5 * base + rng.normal(0, 6, n)
    wellbeing = rng.normal(6.5, 1.3, n)
    g1 = (sleep < 6.2) & (study > 5)
    wellbeing[g1] -= 1.8
    math[g1] += 6
    science[g1] += 5
    g2 = program == "arts"
    reading[g2] += 7
    science[g2] -= 6
    df = pd.DataFrame({"study_hours": study, "attendance": attendance.round(1), "sleep_hours": sleep,
                       "parental_education": parental, "program": program,
                       "math": math.clip(0, 100).round(1), "reading": reading.clip(0, 100).round(1),
                       "science": science.clip(0, 100).round(1), "wellbeing": wellbeing.clip(0, 10).round(2),
                       "planted": np.where(g1, "tired hard-workers", np.where(g2, "arts", "no"))})
    return df, ["study_hours", "attendance", "sleep_hours", "parental_education", "program"], \
        ["math", "reading", "science", "wellbeing"], ["planted"]


def clinical(n=350, seed=5):
    rng = np.random.default_rng(seed)
    age = rng.uniform(25, 80, n).round(0)
    bmi = rng.normal(26, 4, n).round(1)
    smoker = rng.choice(["yes", "no"], n, p=[0.3, 0.7])
    activity = rng.choice(["low", "moderate", "high"], n, p=[0.4, 0.4, 0.2])
    sex = rng.choice(["F", "M"], n)
    systolic = 110 + 0.4 * age + 0.8 * (bmi - 26) + rng.normal(0, 10, n)
    cholesterol = 4.5 + 0.02 * age + rng.normal(0, 0.8, n)
    glucose = 5.0 + 0.03 * (bmi - 26) + rng.normal(0, 0.6, n)
    hdl = 1.4 - 0.02 * (bmi - 26) + rng.normal(0, 0.25, n)
    g1 = (smoker == "yes") & (bmi > 28)
    systolic[g1] += 14
    hdl[g1] -= 0.3
    glucose[g1] += 0.8
    df = pd.DataFrame({"age": age, "bmi": bmi, "smoker": smoker, "activity": activity, "sex": sex,
                       "systolic": systolic.round(1), "cholesterol": cholesterol.round(2),
                       "glucose": glucose.round(2), "hdl": hdl.round(2),
                       "planted": np.where(g1, "yes", "no")})
    return df, ["age", "bmi", "smoker", "activity", "sex"], ["systolic", "cholesterol", "glucose", "hdl"], ["planted"]


GENERATORS = {
    "planted_shift": planted_shift,
    "planted_rule": planted_rule,
    "pure_noise": pure_noise,
    "customers": customers,
    "students": students,
    "clinical": clinical,
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, gen in GENERATORS.items():
        df, X, Y, M = gen()
        write_tab(df, X, Y, M, os.path.join(OUT, name + ".tab"))
        print(f"{name}: {len(df)} rows, X={X}, Y={Y}")
