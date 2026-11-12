"""Synthetic data generators shared by the tests."""
import numpy as np
import pandas as pd


def planted_shift(n=300, seed=0, effect=1.5, categorical=False):
    """X: x1,x2 numeric (+ x3 categorical); Y: y1,y2. Planted: x1 > 1 -> y1 += effect, y2 -= effect/2.
    Returns (df, X_cols, Y_cols, mask)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    y1 = rng.normal(size=n); y2 = rng.normal(size=n)
    mask = x1 > 1.0
    y1[mask] += effect; y2[mask] -= effect / 2
    df = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
    X = ["x1", "x2"]
    if categorical:
        df["x3"] = rng.choice(["a", "b", "c"], n)
        df["y3"] = rng.choice(["u", "v"], n)
        X.append("x3")
        return df, X, ["y1", "y2", "y3"], mask
    return df, X, ["y1", "y2"], mask


def planted_categorical_rule(n=300, seed=0, effect=1.5):
    """Planted: edu == 'high' AND age >= 50 -> y1 += effect."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 70, n)
    edu = rng.choice(["low", "mid", "high"], n)
    inc = rng.normal(size=n)
    y1 = rng.normal(size=n); y2 = rng.normal(size=n)
    mask = (edu == "high") & (age >= 50)
    y1[mask] += effect
    df = pd.DataFrame({"age": age, "edu": edu, "inc": inc, "y1": y1, "y2": y2})
    return df, ["age", "edu", "inc"], ["y1", "y2"], mask


def pure_noise(n=300, seed=0, p_x=3, p_y=2):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(p_x)})
    for j in range(p_y):
        df[f"y{j}"] = rng.normal(size=n)
    return df, [f"x{i}" for i in range(p_x)], [f"y{j}" for j in range(p_y)]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)
