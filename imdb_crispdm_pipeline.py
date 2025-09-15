#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMDb Top 250 Movies — End‑to‑End CRISP‑DM Pipeline

What this script does
---------------------
1) Loads the Kaggle "IMDB Top 250 Movies.csv" file.
2) Normalizes column names and parses key fields:
   - runtime -> minutes
   - budget/box_office -> numeric USD (supports suffixes M/B)
3) Data Understanding:
   - head(), info(), describe(), missingness
   - extended numeric summary (skew, kurtosis)
4) Data Preparation:
   - cap runtime to [40, 240]
   - log transform budget and box office
   - encode genres (multi-hot), certificates (one-hot), directors (frequency)
   - standardize numeric features; build X and y
   - emit warnings for anomalies
5) Modeling:
   - Train/test split (80/20)
   - Baselines: Linear Regression, Decision Tree
   - Handle NaNs via SimpleImputer (median) OR drop-NaN path
   - Stronger models: Ridge, Gradient Boosting, Random Forest
   - Hyperparameter tuning with RandomizedSearchCV for RF & GB (fast grid)
6) Evaluation:
   - RMSE, MAE, R2, Explained Variance
   - Feature Importances (Decision Tree + Tuned RF)
   - Comparison plots
7) Saves plots to ./outputs/
   - comparison_metrics.png
   - dt_feature_importance.png
   - rf_feature_importance.png

Usage
-----
$ python imdb_crispdm_pipeline.py --data "/path/to/IMDB Top 250 Movies.csv"

Notes
-----
- Designed for teaching/assignment use; small dataset (n=250).
- Uses only pandas, numpy, scikit-learn, matplotlib.
"""

import os
import re
import argparse
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)

# -----------------------------
# Utilities
# -----------------------------

def normalize_col(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace("%", "pct").replace("#", "num")
    for ch in [" ", "-", "/", "(", ")", ".", ":"]:
        c = c.replace(ch, "_")
    while "__" in c:
        c = c.replace("__", "_")
    return c.strip("_")

def parse_runtime(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    hours = re.search(r"(\\d+)h", s)
    mins = re.search(r"(\\d+)m", s)
    h = int(hours.group(1)) if hours else 0
    m = int(mins.group(1)) if mins else 0
    return h * 60 + m

def parse_money(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "").replace("$", "")
    mult = 1
    if s.endswith("M"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("B"):
        mult, s = 1_000_000_000, s[:-1]
    try:
        return float(s) * mult
    except Exception:
        return np.nan

def print_section(title):
    print("\\n" + "="*len(title))
    print(title)
    print("="*len(title))

def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# -----------------------------
# Pipeline
# -----------------------------

def load_and_understand(data_path: str) -> pd.DataFrame:
    print_section("Load & Initial Checks")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    print(f"File: {data_path}  |  Exists: {os.path.exists(data_path)}")

    df_raw = pd.read_csv(data_path, encoding="utf-8", engine="python")
    print(f"Raw shape: {df_raw.shape}")
    print(f"Raw columns: {list(df_raw.columns)}")

    df = df_raw.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    print(f"Normalized columns: {list(df.columns)}")

    print_section("Head (5)")
    print(df.head(5).to_string(index=False))

    print_section("DataFrame.info()")
    buf = StringIO(); df.info(buf=buf)
    print(buf.getvalue())

    print_section("Missing Values per Column")
    print(df.isna().sum())

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print_section("Numeric Describe (quick)")
        print(df[num_cols].describe().T)
    return df

def data_preparation(df: pd.DataFrame):
    print_section("Data Preparation: Parsing & Transformations")

    # Parse
    df["runtime_min"] = df["run_time"].apply(parse_runtime)
    df["budget_num"] = df["budget"].apply(parse_money)
    df["box_office_num"] = df["box_office"].apply(parse_money)

    # Log transforms
    df["log_budget"] = np.log1p(df["budget_num"])
    df["log_box_office"] = np.log1p(df["box_office_num"])

    # Cap runtime
    before = df["runtime_min"].copy()
    df.loc[df["runtime_min"] < 40, "runtime_min"] = 40
    df.loc[df["runtime_min"] > 240, "runtime_min"] = 240

    capped = int((before != df["runtime_min"]).sum())
    if capped > 0:
        warnings.warn(f"{capped} runtime values capped to [40, 240].")

    if (df["budget_num"] > 1e9).any():
        warnings.warn("Budgets > $1B found — potential outliers.")
    if (df["box_office_num"] > 3e9).any():
        warnings.warn("Box office > $3B found — potential anomalies.")

    # Encodings
    df["genre_list"] = df["genre"].apply(lambda x: [g.strip() for g in str(x).split(",")] if pd.notna(x) else [])
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(mlb.fit_transform(df["genre_list"]),
                                 columns=[f"genre_{g}" for g in mlb.classes_],
                                 index=df.index)

    top_certs = df["certificate"].value_counts().index[:5]
    df["certificate_clean"] = df["certificate"].where(df["certificate"].isin(top_certs), "Other")
    cert_encoded = pd.get_dummies(df["certificate_clean"], prefix="cert")

    # Director frequency
    director_counts = Counter()
    for d in df["directors"].dropna():
        for x in str(d).split(","):
            director_counts[x.strip()] += 1
    df["director_score"] = df["directors"].apply(
        lambda x: max([director_counts.get(n.strip(), 0) for n in str(x).split(",")]) if pd.notna(x) else 0
    )

    # Scaling
    num_features = ["year", "runtime_min", "log_budget", "log_box_office", "director_score"]
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(scaler.fit_transform(df[num_features]),
                                columns=[f"{c}_scaled" for c in num_features],
                                index=df.index)

    # Final feature matrix
    X_full = pd.concat([X_num_scaled, genre_encoded, cert_encoded], axis=1)
    y = df["rating"]

    print_section("Prepared Feature Matrix Summary")
    print(f"X_full shape: {X_full.shape}")
    print("Preview:")
    print(X_full.head())

    print_section("Key Numeric Summaries")
    for col in ["runtime_min", "budget_num", "box_office_num", "log_budget", "log_box_office"]:
        print(f"\n-- {col} --")
        print(df[col].describe())

    return df, X_full, y

def model_and_evaluate(X: pd.DataFrame, y: pd.Series, outdir: str):
    print_section("Modeling: Train/Test Split")
    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Baselines
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=5),
        "Ridge Regression": Ridge(alpha=1.0),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=3),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10, min_samples_split=2),
    }

    metrics = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        metrics[name] = {
            "RMSE": mean_squared_error(y_test, pred, squared=False),
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred),
            "Explained Variance": explained_variance_score(y_test, pred)
        }
        print(f"{name:>18} -> RMSE: {metrics[name]['RMSE']:.4f} | MAE: {metrics[name]['MAE']:.4f} | "
              f"R2: {metrics[name]['R2']:.3f} | EV: {metrics[name]['Explained Variance']:.3f}")

    # Quick DT importances
    if hasattr(models["Decision Tree"], "feature_importances_"):
        dt_importances = pd.Series(models["Decision Tree"].feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        plt.figure(figsize=(8,5))
        dt_importances.plot(kind="barh")
        plt.title("Top 15 Features (Decision Tree)")
        plt.gca().invert_yaxis()
        save_plot(os.path.join(outdir, "dt_feature_importance.png"))

    # Tune RF & GB (fast randomized search)
    print_section("Tuning: RandomizedSearchCV (RF & GB)")
    rf = RandomForestRegressor(random_state=42)
    rf_params = {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]}
    rf_rand = RandomizedSearchCV(rf, rf_params, scoring="neg_root_mean_squared_error",
                                 cv=3, n_iter=5, n_jobs=-1, random_state=42)
    rf_rand.fit(X_train, y_train)
    best_rf = rf_rand.best_estimator_
    pred_rf = best_rf.predict(X_test)
    rf_metrics = {
        "RMSE": mean_squared_error(y_test, pred_rf, squared=False),
        "MAE": mean_absolute_error(y_test, pred_rf),
        "R2": r2_score(y_test, pred_rf),
        "Explained Variance": explained_variance_score(y_test, pred_rf)
    }
    print(f"Best RF Params: {rf_rand.best_params_}")
    print(f"Best RF -> RMSE: {rf_metrics['RMSE']:.4f} | MAE: {rf_metrics['MAE']:.4f} | "
          f"R2: {rf_metrics['R2']:.3f} | EV: {rf_metrics['Explained Variance']:.3f}")

    # RF importances
    rf_importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(8,5))
    rf_importances.plot(kind="barh", color="skyblue")
    plt.title("Top 15 Features (Random Forest - Tuned)")
    plt.gca().invert_yaxis()
    save_plot(os.path.join(outdir, "rf_feature_importance.png"))

    # Comparison plots
    metrics_full = metrics.copy()
    metrics_full["Random Forest (Tuned)"] = rf_metrics
    mdf = pd.DataFrame(metrics_full).T

    fig, axes = plt.subplots(2,2, figsize=(12,10))
    mdf["RMSE"].plot(kind="bar", ax=axes[0,0], title="RMSE (Lower is Better)", color="skyblue")
    mdf["MAE"].plot(kind="bar", ax=axes[0,1], title="MAE (Lower is Better)", color="salmon")
    mdf["R2"].plot(kind="bar", ax=axes[1,0], title="R² (Higher is Better)", color="seagreen")
    mdf["Explained Variance"].plot(kind="bar", ax=axes[1,1], title="Explained Variance (Higher is Better)", color="orange")
    save_plot(os.path.join(outdir, "comparison_metrics.png"))

    print_section("Model Comparison Table")
    print(mdf.sort_values("RMSE"))

    return best_rf, mdf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="IMDB Top 250 Movies.csv",
                        help="Path to the 'IMDB Top 250 Movies.csv' file")
    parser.add_argument("--outputs", type=str, default="outputs", help="Directory to save plots")
    args = parser.parse_args()

    df = load_and_understand(args.data)
    df, X, y = data_preparation(df)
    best_model, metrics_df = model_and_evaluate(X, y, args.outputs)

    print_section("Done")
    print("Outputs saved to:", os.path.abspath(args.outputs))
    print("Key files:")
    print(" - outputs/comparison_metrics.png")
    print(" - outputs/dt_feature_importance.png")
    print(" - outputs/rf_feature_importance.png")

if __name__ == "__main__":
    main()
