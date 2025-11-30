#!/usr/bin/env python3


from __future__ import annotations

import os
import re
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Raw CSV input paths (unchanged so the script still works with your files)
CSV_A_COST = "Crop/datafile (1).csv"
CSV_B_PROD_BY_YEARS = "Crop/datafile (2).csv"
CSV_C_VARIETY = "Crop/datafile (3).csv"
CSV_D_INDEX = "Crop/produce.csv"
CSV_E_PRODUCTION_TIME = "Crop/produce.csv"

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# HELPER UTILITIES
# ---------------------------------------------------------------------------

def _clean_numeric_string(value: object) -> str | float | np.nan:

    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    if not s:
        return np.nan

    # Early return if there is no comma at all.
    if "," not in s:
        s = s.replace(" ", "")
        s = re.sub(r"[^0-9eE\.\-]", "", s)
        return s if s else np.nan

    # There is at least one comma.
    parts = s.split(",")

    if len(parts) == 2:
        # Either a thousands separator (e.g. 1,234) or a decimal comma.
        left, right = parts
        if len(right) == 3:
            # Likely thousands separator.
            merged = left + right
            merged = re.sub(r"[^0-9eE\.\-]", "", merged)
            return merged
        # Otherwise treat comma as decimal point.
        merged = f"{left}.{right}"
        merged = re.sub(r"[^0-9eE\.\-]", "", merged)
        return merged

    # More than one comma -> drop all commas and clean.
    merged = "".join(parts)
    merged = re.sub(r"[^0-9eE\.\-]", "", merged)
    return merged


def safe_float(value: object) -> float:

    if pd.isna(value):
        return np.nan

    try:
        return float(value)
    except Exception:
        try:
            cleaned = _clean_numeric_string(value)
            if cleaned in (None, ""):
                return np.nan
            return float(cleaned)
        except Exception:
            return np.nan


# ---------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# ---------------------------------------------------------------------------

def load_cost_file(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    for col in df.columns:
        if col.lower() in {"crop", "state"}:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = df[col].apply(safe_float)

    return df


def _melt_year_block(df: pd.DataFrame, prefixes: List[str], value_name: str) -> pd.DataFrame:

    value_cols = [c for c in df.columns if any(p in c for p in prefixes)]
    id_cols = [c for c in df.columns if c not in value_cols]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="_variable",
        value_name=value_name,
    )

    def parse_year(label: str) -> float:
        label = str(label)
        # Try patterns like 2006-07 or 2006-2007
        match = re.search(r"(20\d{2})\D*(\d{2,4})", label)
        if match:
            first = int(match.group(1))
            second = match.group(2)
            if len(second) == 2:
                # example: 2006-07 -> 2007
                end_year = int(str(first)[:2] + second)
                return end_year
            return int(second)

        # Fallback: any single 4-digit year starting with 20
        match2 = re.search(r"(20\d{2})", label)
        if match2:
            return int(match2.group(1))
        return np.nan

    long_df["Year"] = long_df["_variable"].apply(parse_year)
    long_df.drop(columns=["_variable"], inplace=True)
    long_df[value_name] = long_df[value_name].apply(safe_float)

    return long_df


def load_prod_by_years(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    prod = _melt_year_block(df, ["Production"], "Production")
    area = _melt_year_block(df, ["Area"], "Area")
    yield_ = _melt_year_block(df, ["Yield"], "Yield")

    # Identify non-measurement columns (Crop, State, etc.)
    measurement_cols = [c for c in df.columns if any(k in c for k in ["Production", "Area", "Yield"])]
    id_cols = [c for c in df.columns if c not in measurement_cols]

    # Prefer Crop as key when available
    key_cols = ["Crop"] if "Crop" in df.columns else id_cols

    merged = prod.merge(area, on=key_cols + ["Year"], how="outer")
    merged = merged.merge(yield_, on=key_cols + ["Year"], how="outer")

    if "Crop" in merged.columns:
        merged["Crop"] = merged["Crop"].astype(str).str.strip()

    return merged


def load_variety_file(path: str) -> pd.DataFrame:
    """Load variety/season/zone table (file 3) with basic string cleaning."""
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col].replace({"": None}, inplace=True)

    return df


def load_index_file(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    id_cols = [c for c in df.columns if not re.search(r"20\d{2}", c)]
    year_cols = [c for c in df.columns if c not in id_cols]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="YearLabel",
        value_name="IndexVal",
    )

    def parse_index_year(label: str) -> float:
        label = str(label)
        m = re.search(r"(20\d{2})\D*(\d{2})", label)
        if m:
            start = int(m.group(1))
            final_two = int(m.group(2))
            end_year = int(str(start)[:2] + f"{final_two:02d}")
            return end_year if end_year >= start else start

        m2 = re.search(r"(20\d{2})", label)
        if m2:
            return int(m2.group(1))
        return np.nan

    long_df["Year"] = long_df["YearLabel"].apply(parse_index_year)
    long_df["IndexVal"] = long_df["IndexVal"].apply(safe_float)

    # Assume the first non-year column refers to the category, rename for clarity.
    if id_cols:
        first_id = id_cols[0]
        if first_id.lower() != "year":
            long_df.rename(columns={first_id: "Category"}, inplace=True)

    return long_df.drop(columns=["YearLabel"])


def load_production_timeseries(path: str) -> pd.DataFrame:
    """Load macro time-series table (file 5) and melt to Year/Value."""
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    base_id_cols = ["Particulars", "Frequency", "Unit"]
    id_cols = [c for c in base_id_cols if c in df.columns]

    # identify columns that look like years or contain years ("3-1993", etc.)
    year_cols = [
        c
        for c in df.columns
        if re.search(r"(19|20)\d{2}", c) or re.search(r"\d{4}", c)
    ]
    if not year_cols:
        year_cols = [c for c in df.columns if re.search(r"\d{4}", c)]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="YearLabel",
        value_name="Value",
    )

    def parse_year(label: str) -> float:
        label = str(label)
        m = re.search(r"(19|20)\d{2}", label)
        if m:
            return int(m.group(0))
        return np.nan

    long_df["Year"] = long_df["YearLabel"].apply(parse_year)
    long_df["Value"] = long_df["Value"].apply(safe_float)

    return long_df.drop(columns=["YearLabel"])


# ---------------------------------------------------------------------------
# MERGING LOGIC
# ---------------------------------------------------------------------------

def build_master_table(
    cost_df: pd.DataFrame,
    prod_year_df: pd.DataFrame,
    variety_df: pd.DataFrame,
    index_df: pd.DataFrame,
    ts_df: pd.DataFrame,
) -> pd.DataFrame:


    base = prod_year_df.copy()

    if "Crop" in base.columns:
        base["Crop"] = base["Crop"].astype(str).str.strip()

    # --- merge cost information ---
    cost = cost_df.copy()
    if "Crop" in cost.columns:
        cost["Crop"] = cost["Crop"].astype(str).str.strip()

    if {"State", "Crop"}.issubset(cost.columns) and "State" in base.columns:
        if "Year" in cost.columns:
            base = base.merge(cost, on=["Crop", "State", "Year"], how="left")
        else:
            base = base.merge(cost, on=["Crop", "State"], how="left")
    else:
        base = base.merge(cost, on=["Crop"], how="left")

    # --- merge variety / season / zone (no year dimension) ---
    var = variety_df.copy()
    if "Crop" in var.columns:
        var["Crop"] = var["Crop"].astype(str).str.strip()
        var = var.drop_duplicates(subset=["Crop"], keep="first")
        base = base.merge(var, on="Crop", how="left")

    # --- merge index data (Category assumed to align with Crop) ---
    idx = index_df.copy()
    if "Category" in idx.columns:
        idx["Category"] = idx["Category"].astype(str).str.strip()
        idx_renamed = idx.rename(columns={"Category": "Crop"})
        base = base.merge(idx_renamed, on=["Crop", "Year"], how="left")

    # --- merge macro timeseries by fuzzy matching on Particulars ---
    ts = ts_df.copy()

    if "Particulars" in ts.columns:

        def lookup_ts_value(row) -> float:
            crop_name = str(row.get("Crop", ""))
            year_val = row.get("Year")
            if pd.isna(year_val) or not crop_name:
                return np.nan

            mask = ts["Particulars"].str.contains(crop_name, case=False, na=False)
            mask &= ts["Year"] == year_val

            matches = ts.loc[mask, "Value"]
            if matches.empty:
                return np.nan
            return matches.iloc[0]

        base["TimeseriesValue"] = base.apply(lookup_ts_value, axis=1)
    else:
        base["TimeseriesValue"] = np.nan

    return base


# ---------------------------------------------------------------------------
# EXPLORATORY ANALYSIS
# ---------------------------------------------------------------------------

def run_basic_eda(df: pd.DataFrame) -> None:
    """Print simple dataset diagnostics and a histogram for Production."""
    print("=== BASIC EDA ===")
    print("Shape (rows, columns):", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nMissing values (top 20 columns):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nSample rows:")
    print(df.head(8))

    if "Production" in df.columns:
        print("\nProduction statistics:")
        print(df["Production"].describe())

        try:
            df["Production"].dropna().astype(float).hist(bins=40)
            plt.title("Production distribution")
            plt.xlabel("Production")
            plt.ylabel("Count")
            plt.show()
        except Exception:
            # plotting is non-critical; ignore if backend is not available
            pass


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING & MODELING
# ---------------------------------------------------------------------------

def prepare_features_for_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare feature matrix X and target vector y for modeling.

    Target: Production
    Features: Area, Yield, cost columns, IndexVal, TimeseriesValue, Year,
              encoded Crop and State.
    """

    data = df.copy()
    # ensure we only keep rows where Production is available
    data = data.dropna(subset=["Production"])

    numeric_candidates = [
        "Production",
        "Area",
        "Yield",
        "Cost of Cultivation (`/Hectare) A2+FL",
        "Cost of Cultivation (`/Hectare) C2",
        "Cost of Production (`/Quintal) C2",
        "Yield (Quintal/ Hectare)",
        "IndexVal",
        "TimeseriesValue",
    ]

    for col in numeric_candidates:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Fill missing Production using Area * Yield when possible
    if {"Production", "Area", "Yield"}.issubset(data.columns):
        mask_missing_prod = (
            data["Production"].isna()
            & data["Area"].notna()
            & data["Yield"].notna()
        )
        data.loc[mask_missing_prod, "Production"] = (
            data.loc[mask_missing_prod, "Area"]
            * data.loc[mask_missing_prod, "Yield"]
        )

    feature_cols: List[str] = []

    for col in [
        "Area",
        "Yield",
        "Cost of Cultivation (`/Hectare) A2+FL",
        "Cost of Cultivation (`/Hectare) C2",
        "Cost of Production (`/Quintal) C2",
        "Yield (Quintal/ Hectare)",
        "IndexVal",
        "TimeseriesValue",
    ]:
        if col in data.columns:
            feature_cols.append(col)

    if "Year" in data.columns:
        data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
        feature_cols.append("Year")

    # Encode categorical identifiers as integer codes.
    for cat_col in ["Crop", "State"]:
        if cat_col in data.columns:
            data[cat_col] = data[cat_col].astype(str).str.strip()
            code_col = f"{cat_col}_code"
            data[code_col] = data[cat_col].astype("category").cat.codes
            feature_cols.append(code_col)

    # Drop rows with any missing feature or target values.
    data = data.dropna(subset=feature_cols + ["Production"], how="any")

    X = data[feature_cols].fillna(0)
    y = data["Production"].astype(float)

    return X, y, data


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """Train a RandomForest regressor and report evaluation metrics."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(
        f"Test rows: {len(X_test)} | "
        f"MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}"
    )

    return model, X_test, y_test, preds


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading input files...")

    cost_df = load_cost_file(CSV_A_COST)
    print("Loaded cost file:", cost_df.shape)

    prod_year_df = load_prod_by_years(CSV_B_PROD_BY_YEARS)
    print("Loaded production-by-years file:", prod_year_df.shape)

    variety_df = load_variety_file(CSV_C_VARIETY)
    print("Loaded variety file:", variety_df.shape)

    index_df = load_index_file(CSV_D_INDEX)
    print("Loaded index file:", index_df.shape)

    ts_df = load_production_timeseries(CSV_E_PRODUCTION_TIME)
    print("Loaded timeseries file:", ts_df.shape)

    print("\nMerging sources to build master table...")
    master = build_master_table(cost_df, prod_year_df, variety_df, index_df, ts_df)
    print("Master table shape:", master.shape)

    master_path = os.path.join(OUTPUT_DIR, "master_pre_merge.csv")
    master.to_csv(master_path, index=False)
    print(f"Saved master table to {master_path}")

    run_basic_eda(master)

    print("\nPreparing features and target variable...")
    X, y, df_model = prepare_features_for_model(master)
    print("Feature matrix shape:", X.shape)

    if X.shape[0] < 20:
        print(
            "Not enough rows for modeling after cleaning and merging. "
            "Please inspect the merged data and join keys."
        )
        return

    print("\nTraining baseline RandomForest model...")
    model, X_test, y_test, preds = train_and_evaluate(X, y)

    # Save a sample of predictions
    results_df = df_model.loc[X_test.index].copy()
    results_df["y_true"] = y_test
    results_df["y_pred"] = preds

    results_path = os.path.join(OUTPUT_DIR, "model_results_sample.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved prediction sample to {results_path}")

    # Persist trained model
    model_path = os.path.join(OUTPUT_DIR, "rf_production_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    # Show top feature importances
    print("\nTop feature importances:")
    try:
        importances = model.feature_importances_
        feature_names = list(X.columns)
        ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        for name, importance in ranked[:20]:
            print(f"{name}: {importance:.4f}")
    except Exception as exc:  # pragma: no cover - very unlikely branch
        print("Could not compute feature importances:", exc)


if __name__ == "__main__":
    main()