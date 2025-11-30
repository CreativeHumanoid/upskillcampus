
#!/usr/bin/env python3
"""
silica_forecast.py

End-to-end, single-file workflow that:

1. Loads the mining flotation plant CSV.
2. Cleans messy numeric fields (mix of comma/point formats).
3. Resamples measurements onto a uniform time grid.
4. Builds lagged and rolling-window features.
5. Trains a RandomForest model for multiple forecast horizons.
6. Compares the model against a simple persistence baseline.

Designed to keep behavior equivalent to the previous script while
changing structure, naming, and comments.
"""

import warnings
from typing import Dict, List, Tuples

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit  # kept for compatibility, even if not used

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Path to the raw industrial CSV (same as before; adjust to your local path)
CSV_PATH = "/MiningProcess_Flotation_Plant_Database.csv"  # Download the CSV as described in the README

# Column names and forecasting config
DATETIME_COL = "date"
TARGET_COL = "% Silica Concentrate"

# Time grid frequency (1 minute by default)
TIME_FREQ = "1T"

# Lag/rolling windows in minutes (using the same values as original script)
LAG_MINUTES: List[int] = [1, 5, 10, 30, 60]
ROLL_WINDOWS: List[int] = [5, 15, 60]

# Evaluation setup
TEST_DAYS = 7  # last N days held out for test
RANDOM_SEED = 42
USE_XGBOOST = False  # kept for future use; not used in this file


# ---------------------------------------------------------------------------
# NUMERIC CLEANING HELPERS
# ---------------------------------------------------------------------------

def _normalize_numeric_token(raw_val) -> str:
    """
    Take a raw string that may contain commas, spaces, or grouping symbols
    and return a 'float-ready' string.

    Heuristic:
      - If there's exactly one comma and the right side has 3 digits,
        interpret it as a thousands separator: '1,234' -> '1234'.
      - If there's exactly one comma and the right side has != 3 digits,
        treat it as a decimal: '1,2' -> '1.2'.
      - If there are multiple commas, strip all of them.
      - Always trim whitespace.
    """
    if pd.isna(raw_val):
        return raw_val

    s = str(raw_val).strip()
    if s == "":
        return s

    if "," not in s:
        # nothing to normalize except trimming
        return s

    parts = s.split(",")
    if len(parts) == 2:
        left, right = parts
        if len(right) == 3:
            # 'thousands' grouping
            return f"{left}{right}"
        # likely decimal comma
        return f"{left}.{right}"

    # many commas: assume all are grouping
    return "".join(parts)


def safe_to_float(value) -> float:
    """Convert any numeric-looking object to float with a couple of fallbacks."""
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        try:
            normalized = _normalize_numeric_token(value)
            if normalized in ("", None):
                return np.nan
            return float(normalized)
        except Exception:
            return np.nan


# ---------------------------------------------------------------------------
# LOADING & CLEANING
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Read the raw CSV and perform:
    - whitespace stripping on column names
    - numeric coercion via safe_to_float for all non-datetime columns
    - datetime parsing and indexing
    """
    # Try comma-separated file (dataset commonly provided this way)
    raw = pd.read_csv(path, sep=",", dtype=str)

    # Normalize column headers
    raw.columns = [c.strip() for c in raw.columns]

    if DATETIME_COL not in raw.columns:
        raise ValueError(
            f"Expected datetime column '{DATETIME_COL}' not found. "
            f"Available columns: {raw.columns.tolist()}"
        )

    df = raw.copy()
    df[DATETIME_COL] = df[DATETIME_COL].astype(str).str.strip()

    # Clean all non-datetime columns
    for col in df.columns:
        if col == DATETIME_COL:
            continue

        series = df[col].astype(str).str.strip()
        series = series.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        series = series.map(lambda v: _normalize_numeric_token(v) if pd.notna(v) else v)
        df[col] = series.map(safe_to_float)

    # Parse the datetime column and set as index
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    df = df.dropna(subset=[DATETIME_COL])
    df = df.set_index(DATETIME_COL).sort_index()

    return df


# ---------------------------------------------------------------------------
# RESAMPLING
# ---------------------------------------------------------------------------

def resample_to_freq(df: pd.DataFrame, freq: str = TIME_FREQ, agg: str = "mean") -> pd.DataFrame:
    """
    Resample the time series to a fixed frequency.

    - Data at the new frequency is aggregated with `agg` (mean by default).
    - Missing timestamps are filled by forward fill, which is a typical
      choice for industrial process signals.
    """
    # Aggregate values at the chosen frequency
    resampled = df.resample(freq).agg(agg)
    # Maintain continuity of 'slow' signals
    resampled = resampled.ffill()
    return resampled


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: List[int] = LAG_MINUTES,
    windows: List[int] = ROLL_WINDOWS,
) -> pd.DataFrame:
    """
    Construct a feature matrix with:

    - raw signals (including target),
    - lagged versions of all numeric columns (for each lag in minutes),
    - rolling mean and std windows,
    - simple calendar/time features.

    Assumes the index is a regular time index with frequency compatible
    with the `lags` values (minutes in this project).
    """
    X = df.copy()

    # Lagged versions for each numeric column (mirrors previous behavior)
    for lag in lags:
        lagged = X.shift(lag)
        lagged.columns = [f"{col}_lag{lag}m" for col in lagged.columns]
        X = pd.concat([X, lagged], axis=1)

    # Rolling windows built from the *original* signals
    for win in windows:
        roll_mean = df.rolling(window=win, min_periods=1).mean().add_suffix(f"_roll_mean{win}m")
        roll_std = df.rolling(window=win, min_periods=1).std().add_suffix(f"_roll_std{win}m")
        X = pd.concat([X, roll_mean, roll_std], axis=1)

    # Time-of-day / calendar features
    X["hour"] = X.index.hour
    X["minute"] = X.index.minute
    X["dayofweek"] = X.index.dayofweek

    # Remove the initial rows that don't have the largest lag available
    if lags:
        max_lag = max(lags)
        required_col = f"{target_col}_lag{max_lag}m"
        X = X.dropna(subset=[required_col])

    return X


# ---------------------------------------------------------------------------
# TRAIN / TEST SPLITTING
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, test_days: int = TEST_DAYS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time: the last `test_days` worth of data becomes the test set.
    """
    last_ts = df.index.max()
    split_point = last_ts - pd.Timedelta(days=test_days)
    train = df[df.index <= split_point]
    test = df[df.index > split_point]
    return train, test


# ---------------------------------------------------------------------------
# BASELINES AND METRICS
# ---------------------------------------------------------------------------

def persistence_baseline(values: pd.Series, horizon_minutes: int) -> np.ndarray:
    """
    Simple persistence forecaster: 'tomorrow equals today'.

    For this industrial series, we implement it as using the value at t
    as the prediction for t + horizon.
    """
    # Shift *backwards* by horizon to get y_{t} for time index t+h
    return values.shift(horizon_minutes).values


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics, skipping NaNs in predictions."""
    mask = ~np.isnan(y_pred)
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]

    if y_true.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------

def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    drop_iron: bool = False,
) -> Tuple[np.ndarray, RandomForestRegressor]:
    """
    Train a RandomForestRegressor for one particular forecasting horizon.

    If `drop_iron` is True, all features derived from '% Iron Concentrate'
    are removed from the input matrix.
    """
    if drop_iron:
        cols_to_remove = [c for c in X_train.columns if "% Iron Concentrate" in c]
        X_train = X_train.drop(columns=cols_to_remove, errors="ignore")
        X_test = X_test.drop(columns=cols_to_remove, errors="ignore")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return predictions, model


# ---------------------------------------------------------------------------
# MULTI-HORIZON EXPERIMENT
# ---------------------------------------------------------------------------

def evaluate_horizons(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    horizons_hours: List[int] = None,
    drop_iron: bool = False,
):
    """
    Evaluate multiple forecast horizons (e.g., 1h, 2h, 4h, ...) using:

    - a RandomForest model per horizon (one-step-direct strategy)
    - a persistence baseline that simply reuses recent values
    """
    if horizons_hours is None:
        horizons_hours = [1, 2, 4, 8, 12, 24]

    results = []

    # Build feature matrix once; keep target column separately
    X_full = build_feature_matrix(df, target_col=target_col, lags=LAG_MINUTES, windows=ROLL_WINDOWS)
    y_full = X_full[target_col]
    X_full = X_full.drop(columns=[target_col])

    # Split into train/test respecting time order
    joined = X_full.join(y_full)
    train_df, test_df = temporal_split(joined, test_days=TEST_DAYS)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    for h in horizons_hours:
        horizon_minutes = h * 60

        # Align targets so that features at time t predict y at t + h
        y_train_h = y_train.shift(-horizon_minutes).dropna()
        X_train_h = X_train.loc[y_train_h.index]

        y_test_h = y_test.shift(-horizon_minutes).dropna()
        X_test_h = X_test.loc[y_test_h.index]

        # Persistence baseline: reuse the smallest lag already created
        if LAG_MINUTES:
            lag_col = f"{target_col}_lag{LAG_MINUTES[0]}m"
            persistence_pred = X_test_h[lag_col].values
        else:
            persistence_pred = np.full(len(y_test_h), y_test_h.iloc[0])

        baseline_metrics = score_predictions(y_test_h.values, persistence_pred)

        # Train RandomForest for this horizon
        model_pred, model = fit_random_forest(
            X_train_h,
            y_train_h.values,
            X_test_h,
            y_test_h.values,
            drop_iron=drop_iron,
        )
        model_metrics = score_predictions(y_test_h.values, model_pred)

        results.append(
            {
                "h_hours": h,
                "persistence": baseline_metrics,
                "model": model_metrics,
                "n_train": len(X_train_h),
                "n_test": len(X_test_h),
            }
        )

        print(
            f"H={h}h | n_train={len(X_train_h):6d} | n_test={len(X_test_h):6d} | "
            f"pers MAE={baseline_metrics['MAE']:.4f} | model MAE={model_metrics['MAE']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    print("Step 1/5: Loading and cleaning raw data...")
    df_raw = load_and_clean(CSV_PATH)
    print(f"Loaded {len(df_raw)} rows from {df_raw.index.min()} to {df_raw.index.max()}")
    print("Available columns:", df_raw.columns.tolist())

    print(f"\nStep 2/5: Resampling to uniform {TIME_FREQ} grid...")
    df_resampled = resample_to_freq(df_raw, freq=TIME_FREQ, agg="mean")
    print(f"After resampling: {len(df_resampled)} rows")

    if TARGET_COL not in df_resampled.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found after cleaning. "
            f"Columns present: {df_resampled.columns.tolist()}"
        )

    # Drop any rows with missing target
    df_resampled = df_resampled.dropna(subset=[TARGET_COL])
    print(f"Rows after dropping NaNs in target: {len(df_resampled)}")

    print("\nStep 3/5: Quick visualization of the target...")
    try:
        df_resampled[TARGET_COL].plot(title="% Silica Concentrate (cleaned)", figsize=(12, 3))
        plt.tight_layout()
        plt.show()
    except Exception:
        # plotting is nice-to-have, not critical
        pass

    horizons = [1, 2, 4, 8, 12, 24]

    print("\nStep 4/5: Multi-horizon evaluation WITH % Iron Concentrate in features")
    results_including_iron = evaluate_horizons(
        df_resampled,
        target_col=TARGET_COL,
        horizons_hours=horizons,
        drop_iron=False,
    )

    print("\nStep 5/5: Multi-horizon evaluation WITHOUT % Iron Concentrate in features")
    results_excluding_iron = evaluate_horizons(
        df_resampled,
        target_col=TARGET_COL,
        horizons_hours=horizons,
        drop_iron=True,
    )

    print("\n=== MAE Summary by Horizon (model incl. vs excl. % Iron, plus persistence) ===")
    for res_inc, res_exc in zip(results_including_iron, results_excluding_iron):
        h = res_inc["h_hours"]
        mae_incl = res_inc["model"]["MAE"]
        mae_excl = res_exc["model"]["MAE"]
        mae_pers = res_inc["persistence"]["MAE"]
        print(
            f"{h:2d}h -> "
            f"model(incl) MAE={mae_incl:.4f} | "
            f"model(excl) MAE={mae_excl:.4f} | "
            f"persistence MAE={mae_pers:.4f}"
        )


if __name__ == "__main__":
    main()
