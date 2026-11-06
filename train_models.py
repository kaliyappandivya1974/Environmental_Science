"""Validate the synthetic sensor dataset and train reproducible demo models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "environmental_monitoring_synthetic_dataset.csv"
MODEL_DIR = ROOT / "models"
SEED = 42
VERSION = "synthetic-demo-v1"

NUMERIC = ["rainfall_mm_1h", "temperature_c", "humidity_pct", "wind_speed_kmh",
           "vegetation_dryness_index", "water_level_m", "water_level_rise_m_per_h",
           "pm25_ug_m3", "pm10_ug_m3", "water_ph", "water_turbidity_ntu",
           "dissolved_oxygen_mg_l"]
LABELS = ["flood_risk_label", "fire_risk_label", "water_quality_anomaly_label"]
RANGES = {"rainfall_mm_1h": (0, 500), "temperature_c": (-60, 65),
          "humidity_pct": (0, 100), "wind_speed_kmh": (0, 400),
          "vegetation_dryness_index": (0, 100), "water_level_m": (-10, 30),
          "water_level_rise_m_per_h": (-20, 20), "pm25_ug_m3": (0, 2000),
          "pm10_ug_m3": (0, 5000), "water_ph": (0, 14),
          "water_turbidity_ntu": (0, 10000), "dissolved_oxygen_mg_l": (0, 30)}


def classification_metrics(y_true, y_pred):
    labels = [0, 1]
    return {"class_balance": {str(k): int(v) for k, v in pd.Series(y_true).value_counts().sort_index().items()},
            "classes_missing_from_test": [label for label in labels if label not in set(pd.Series(y_true).unique())],
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)}


def classification_split(X, y):
    """Stratify whenever each class can occur in both partitions."""
    counts = pd.Series(y).value_counts()
    can_stratify = len(counts) > 1 and int(counts.min()) >= 2
    return (*train_test_split(X, y, test_size=0.2, random_state=SEED,
                              stratify=y if can_stratify else None), can_stratify)


def make_preprocessor(numeric, categorical=True, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    transformers = [("numeric", Pipeline(steps), numeric)]
    if categorical:
        transformers.append(("zone", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["zone_type"]))
    return ColumnTransformer(transformers, remainder="drop")


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    required = {"timestamp", "zone_type", *NUMERIC, *LABELS}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in NUMERIC + LABELS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.timestamp.isna().any():
        raise ValueError(f"{int(df.timestamp.isna().sum())} timestamps could not be parsed")
    if df[LABELS].isna().any().any():
        raise ValueError("Target labels contain missing or non-numeric values")
    if not all(set(df[c].unique()).issubset({0, 1}) for c in LABELS):
        raise ValueError("Expected binary 0/1 target labels")
    invalid = {c: int((~df[c].between(*bounds) & df[c].notna()).sum()) for c, bounds in RANGES.items()}
    invalid = {k: v for k, v in invalid.items() if v}
    if invalid:
        raise ValueError(f"Sensor values outside configured plausible ranges: {invalid}")
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    flood_features = ["rainfall_mm_1h", "water_level_m", "water_level_rise_m_per_h",
                      "humidity_pct", "temperature_c"]
    fire_features = ["temperature_c", "humidity_pct", "wind_speed_kmh", "vegetation_dryness_index"]
    water_features = ["water_ph", "water_turbidity_ntu", "dissolved_oxygen_mg_l"]
    pm_features = [c for c in NUMERIC if c != "pm25_ug_m3"]
    results = {}
    bundles = {}
    for name, features, target in [("flood", flood_features + ["zone_type"], "flood_risk_label"),
                                   ("fire", fire_features + ["zone_type"], "fire_risk_label")]:
        X = df[features]
        y = df[target]
        xtr, xte, ytr, yte, stratified = classification_split(X, y)
        pipeline = Pipeline([("preprocess", make_preprocessor([c for c in features if c != "zone_type"])),
                             ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                              class_weight="balanced_subsample", random_state=SEED,
                                                              n_jobs=-1))])
        pipeline.fit(xtr, ytr)
        pred = pipeline.predict(xte)
        results[name] = {"features": features, "test_size": len(yte), "stratified_split": stratified,
                         **classification_metrics(yte, pred)}
        bundles[name] = {"pipeline": pipeline, "features": features, "version": VERSION}

    # Fit only on the training partition. Synthetic anomaly labels are held out for evaluation only.
    Xw = df[water_features]
    yw = df["water_quality_anomaly_label"]
    xtr, xte, ytr, yte, stratified = classification_split(Xw, yw)
    anomaly = Pipeline([("preprocess", make_preprocessor(water_features, categorical=False, scale=True)),
                        ("model", IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=-1))])
    anomaly.fit(xtr)
    # sklearn maps -1 to anomalous; map anomaly=1 for comparison.
    pred = (anomaly.predict(xte) == -1).astype(int)
    results["water_anomaly"] = {"features": water_features, "test_size": len(yte), "stratified_split": stratified,
                                "evaluation_only_label": "water_quality_anomaly_label",
                                **classification_metrics(yte, pred)}
    bundles["water_anomaly"] = {"pipeline": anomaly, "features": water_features, "version": VERSION}

    X = df[pm_features + ["zone_type"]]
    y = df["pm25_ug_m3"]
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    reg = Pipeline([("preprocess", make_preprocessor(pm_features)),
                    ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                    random_state=SEED, n_jobs=-1))])
    reg.fit(xtr, ytr)
    pred = reg.predict(xte)
    results["pm25"] = {"features": pm_features + ["zone_type"], "test_size": len(yte),
                        "mae": float(mean_absolute_error(yte, pred)),
                        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                        "r2": float(r2_score(yte, pred))}
    bundles["pm25"] = {"pipeline": reg, "features": pm_features + ["zone_type"], "version": VERSION}

    MODEL_DIR.mkdir(exist_ok=True)
    for name, bundle in bundles.items():
        joblib.dump(bundle, MODEL_DIR / f"{name}.joblib")
    report = {"dataset": DATA.name, "rows": len(df), "duplicates_removed": duplicates,
              "dtypes": {k: str(v) for k, v in pd.read_csv(DATA).dtypes.items()},
              "missing_values": {k: int(v) for k, v in pd.read_csv(DATA).isna().sum().items()},
              "invalid_range_counts": invalid,
              "label_distributions": {k: {str(label): int(n) for label, n in df[k].value_counts().sort_index().items()} for k in LABELS},
              "preserved_fields": ["timestamp", "zone_type"],
              "split_note": "Stratification is used when every class has at least two rows; otherwise a seeded unstratified holdout is used and reported per model.",
              "excluded_from_features": LABELS,
              "synthetic_data_disclaimer": "Synthetic data can make evaluation results unrealistically optimistic; these metrics do not establish real-world performance.",
              "random_seed": SEED, "model_version": VERSION, "metrics": results}
    (MODEL_DIR / "training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["rows", "duplicates_removed", "missing_values", "invalid_range_counts", "label_distributions", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
