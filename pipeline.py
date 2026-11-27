from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "open", "high", "low", "close", "volume",
    "return_1", "return_5", "return_20", "range_pct", "body_pct",
    "sma_ratio_5", "sma_ratio_20", "sma_ratio_50",
    "volatility_10", "volatility_20", "rsi_14", "macd", "macd_signal",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def _norm_col(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _parse_time(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        maximum = numeric.dropna().abs().max()
        unit = "ms" if pd.notna(maximum) and maximum > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True).dt.tz_localize(None)
    return pd.to_datetime(
        series.astype(str).str.strip(),
        errors="coerce",
        utc=True,
        format="mixed",
    ).dt.tz_localize(None)


def load_gold_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=None, engine="python")
    raw.columns = [_norm_col(c) for c in raw.columns]
    aliases = {
        "datetime": ["datetime", "date_time", "timestamp", "time", "date", "gmt_time"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "price", "c"],
        "volume": ["volume", "tick_volume", "tickvol", "vol"],
    }

    def pick(names: Iterable[str], required: bool = True) -> str | None:
        for name in names:
            if name in raw.columns:
                return name
        if required:
            raise ValueError(f"Thiếu cột dữ liệu. Cần một trong: {', '.join(names)}")
        return None

    date_col = pick(aliases["datetime"])
    date = _parse_time(raw[date_col])
    if "date" in raw and "time" in raw and date_col == "date":
        date = _parse_time(raw["date"].astype(str) + " " + raw["time"].astype(str))
    result = pd.DataFrame({
        "date": date,
        "open": pd.to_numeric(raw[pick(aliases["open"])], errors="coerce"),
        "high": pd.to_numeric(raw[pick(aliases["high"])], errors="coerce"),
        "low": pd.to_numeric(raw[pick(aliases["low"])], errors="coerce"),
        "close": pd.to_numeric(raw[pick(aliases["close"])], errors="coerce"),
    })
    volume = pick(aliases["volume"], required=False)
    result["volume"] = pd.to_numeric(raw[volume], errors="coerce") if volume else 0.0
    return result


def infer_gap_threshold(dates: pd.Series) -> pd.Timedelta:
    """Return a conservative threshold for separating unusually large time gaps."""
    parsed = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    differences = parsed.diff().dropna()
    differences = differences[differences > pd.Timedelta(0)]
    if differences.empty:
        return pd.Timedelta(days=3)
    cadence = differences.value_counts().index[0]
    return max(cadence * 3, pd.Timedelta(days=3))


def temporal_split_indices(
    row_count: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[int, int]:
    """Return chronological train and validation end positions."""
    if not 0 < train_fraction < 1:
        raise ValueError("Tỷ lệ train phải nằm trong khoảng (0, 1).")
    if not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Tỷ lệ validation phải dương và tổng train + validation nhỏ hơn 1.")
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError("Không đủ dữ liệu để chia train/validation/test theo thời gian.")
    return train_end, validation_end


def clean_data(raw: pd.DataFrame, max_rows: int = 100_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = raw.copy()
    report = [{"step": "raw", "rows": len(d), "details": ""}]
    d["date"] = _parse_time(d["date"])
    missing_count = int(d[["date", "open", "high", "low", "close"]].isna().any(axis=1).sum())
    d = d.dropna(subset=["date", "open", "high", "low", "close"])
    report.append({
        "step": "drop_missing",
        "rows": len(d),
        "details": f"Removed {missing_count} rows with missing/invalid time or OHLC.",
    })
    d = d.sort_values("date", kind="stable")
    duplicate_count = int(d["date"].duplicated(keep="last").sum())
    d = d.drop_duplicates("date", keep="last")
    report.append({
        "step": "sort_deduplicate",
        "rows": len(d),
        "details": f"Removed {duplicate_count} duplicate timestamps; kept the last source row.",
    })
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    d = d[(d["high"] >= d["low"]) & d["open"].between(d["low"], d["high"]) & d["close"].between(d["low"], d["high"])]
    report.append({"step": "validate_ohlc", "rows": len(d), "details": ""})
    d["volume"] = d["volume"].fillna(0).clip(lower=0)
    report.append({
        "step": "preserve_prices",
        "rows": len(d),
        "details": "No full-series return outlier filter; valid flat candles are retained.",
    })
    gap_threshold = infer_gap_threshold(d["date"])
    gap_count = int(d["date"].diff().gt(gap_threshold).sum())
    report.append({
        "step": "detect_time_gaps",
        "rows": len(d),
        "details": f"{gap_count} gaps longer than {gap_threshold}. No rows were imputed.",
    })
    if len(d) > max_rows:
        d = d.tail(max_rows)
    d = d.reset_index(drop=True)
    report.append({"step": f"keep_latest_{max_rows}", "rows": len(d), "details": ""})
    return d, pd.DataFrame(report)


def make_features(clean: pd.DataFrame) -> pd.DataFrame:
    d = clean.copy()
    close = d["close"]
    for period in (1, 5, 20):
        d[f"return_{period}"] = close.pct_change(period)
    d["range_pct"] = (d["high"] - d["low"]) / close
    d["body_pct"] = (d["close"] - d["open"]) / d["open"]
    for period in (5, 20, 50):
        d[f"sma_ratio_{period}"] = close / close.rolling(period).mean() - 1
    for period in (10, 20):
        d[f"volatility_{period}"] = d["return_1"].rolling(period).std()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["hour_sin"] = np.sin(2 * np.pi * d["date"].dt.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["date"].dt.hour / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["date"].dt.dayofweek / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["date"].dt.dayofweek / 7)
    d["target_return"] = close.shift(-1) / close - 1
    d["target_date"] = d["date"].shift(-1)
    return d.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["target_return"]).reset_index(drop=True)


def _target_dates(data: pd.DataFrame) -> pd.Series:
    if "target_date" in data:
        return pd.to_datetime(data["target_date"], errors="coerce")
    return pd.to_datetime(data["date"], errors="coerce").shift(-1)


def _prediction_output(
    data: pd.DataFrame,
    test_indices: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, np.ndarray]:
    target_dates = _target_dates(data)
    actual = data["close"].to_numpy() * (1 + data["target_return"].to_numpy())
    naive = data["close"].to_numpy()
    predictions = np.full(len(data), np.nan)
    predictions[test_indices] = predicted
    naive_output = np.full(len(data), np.nan)
    naive_output[test_indices] = naive[test_indices]
    return {
        "actual": actual,
        "predicted": predictions,
        "naive_predicted": naive_output,
        "dates": target_dates.astype(str).to_numpy(),
    }


def _split_labels(length: int, train_end: int, test_start: int) -> np.ndarray:
    return np.where(
        np.arange(length) < train_end,
        "train",
        np.where(np.arange(length) < test_start, "validation", "test"),
    )


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MAPE": float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100),
    }


def train_linear(
    data: pd.DataFrame,
    split: int,
    pca_components: int = 0,
    validation_split: int | None = None,
) -> tuple[LinearRegression, dict, dict]:
    train_end = validation_split if validation_split is not None else split
    scaler = StandardScaler()
    x_train = scaler.fit_transform(data.iloc[:train_end][FEATURES])
    test_indices = np.arange(split, len(data))
    x_test = scaler.transform(data.iloc[split:][FEATURES])
    model = LinearRegression().fit(x_train, data.iloc[:train_end]["target_return"])
    test_rows = data.iloc[split:]
    actual_test = test_rows["close"].to_numpy() * (1 + test_rows["target_return"].to_numpy())
    predicted_test = test_rows["close"].to_numpy() * (1 + model.predict(x_test))
    artifacts = {"scaler": scaler, "pca": None, "model": model}
    output = _prediction_output(data, test_indices, predicted_test)
    output["split"] = _split_labels(len(data), train_end, split)
    baseline_metrics = metrics(actual_test, test_rows["close"].to_numpy())
    output["baseline_metrics"] = baseline_metrics
    output["artifacts"] = artifacts
    return model, metrics(actual_test, predicted_test), output


def train_knn(
    data: pd.DataFrame,
    split: int,
    validation_split: int | None = None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    train_end = validation_split if validation_split is not None else split
    y = (data["target_return"] > 0).astype(int)
    x_train_full = data.iloc[:train_end][FEATURES]
    x_test_full = data.iloc[split:][FEATURES]
    y_train = y.iloc[:train_end]
    y_test = y.iloc[split:]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_full)
    x_test = scaler.transform(x_test_full)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    key = "K-NN (22 features)"
    return (
        {
            key: {
                "Accuracy": float(accuracy_score(y_test, predicted)),
                "Precision": float(precision_score(y_test, predicted, zero_division=0)),
                "Recall": float(recall_score(y_test, predicted, zero_division=0)),
                "F1": float(f1_score(y_test, predicted, zero_division=0)),
            },
        },
        {key: {"scaler": scaler, "pca": None, "model": model}},
    )


def build_lstm_sequences(
    data: pd.DataFrame,
    window: int,
    split: int,
    pca_components: int | None = None,
    validation_split: int | None = None,
):
    if window < 1:
        raise ValueError("window phải lớn hơn 0.")
    if split <= 0 or split >= len(data):
        raise ValueError("Mốc test phải nằm giữa các dòng dữ liệu.")
    train_end = validation_split if validation_split is not None else int(split * 0.9)
    if train_end < window or train_end >= split:
        raise ValueError("Cần đủ dữ liệu theo thứ tự thời gian cho train, validation và test.")
    data = data.reset_index(drop=True)
    scaler = StandardScaler()
    scaler.fit(data.iloc[:train_end][FEATURES])
    scaled = scaler.transform(data[FEATURES])
    reducer = None
    if pca_components is not None:
        if pca_components <= 0 or pca_components > len(FEATURES):
            raise ValueError(
                f"PCA phải nằm trong khoảng 1 đến {len(FEATURES)} thành phần."
            )
        reducer = PCA(n_components=pca_components, random_state=42)
        reducer.fit(scaled[:train_end])
        scaled = reducer.transform(scaled)

    xs, ys, bases, dates, indices = [], [], [], [], []
    for end in range(window - 1, len(data)):
        xs.append(scaled[end - window + 1:end + 1])
        ys.append(data.iloc[end]["target_return"])
        bases.append(data.iloc[end]["close"])
        dates.append(str(_target_dates(data).iloc[end]))
        indices.append(end)
    indices = np.asarray(indices)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    train = indices < train_end
    validation = (indices >= train_end) & (indices < split)
    test = indices >= split
    return (
        xs[train], ys[train],
        xs[validation], ys[validation],
        xs[test], ys[test],
        np.asarray(bases)[test], np.asarray(dates)[test],
        indices[test], scaler, reducer,
    )


def train_lstm(
    data: pd.DataFrame,
    split: int,
    window: int = 60,
    epochs: int = 40,
    batch_size: int = 256,
    pca_components: int | None = None,
    validation_split: int | None = None,
):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    (
        x_train, y_train, x_val, y_val, x_test, y_test,
        bases, dates, test_indices, scaler, reducer,
    ) = build_lstm_sequences(
        data, window, split, pca_components, validation_split
    )
    model = Sequential([
        Input(shape=(window, x_train.shape[2])),
        LSTM(96, return_sequences=True),
        Dropout(0.20),
        LSTM(48),
        Dropout(0.20),
        Dense(24, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss="mse", metrics=["mae"])
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size, shuffle=False,
        callbacks=[EarlyStopping(patience=7, restore_best_weights=True), ReduceLROnPlateau(patience=3, factor=0.5)],
        verbose=1,
    )
    predicted_return = model.predict(x_test, batch_size=batch_size, verbose=0).ravel()
    actual_test = bases * (1 + y_test)
    predicted_test = bases * (1 + predicted_return)
    output = _prediction_output(data, test_indices, predicted_test)
    output["dates"][test_indices] = dates
    train_end = validation_split if validation_split is not None else int(split * 0.9)
    output["split"] = _split_labels(len(data), train_end, split)
    output["baseline_metrics"] = metrics(actual_test, bases)
    output["history"] = history.history
    output["artifacts"] = {
        "scaler": scaler,
        "pca": reducer,
        "window": window,
        "pca_components": pca_components,
    }
    return model, metrics(actual_test, predicted_test), {
        **output,
        "history": history.history,
    }


def save_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float), encoding="utf-8")