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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        unit = "ms" if numeric.dropna().abs().max() > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True).dt.tz_localize(None)
    return pd.to_datetime(series.astype(str).str.strip(), errors="coerce", utc=True).dt.tz_localize(None)


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


def clean_data(raw: pd.DataFrame, max_rows: int = 100_000, max_return_z: float = 8.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = raw.copy()
    report = [{"step": "raw", "rows": len(d)}]
    d = d.dropna(subset=["date", "open", "high", "low", "close"])
    report.append({"step": "drop_missing", "rows": len(d)})
    d = d.sort_values("date").drop_duplicates("date", keep="last")
    report.append({"step": "sort_deduplicate", "rows": len(d)})
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    d = d[(d["high"] >= d["low"]) & d["open"].between(d["low"], d["high"]) & d["close"].between(d["low"], d["high"])]
    report.append({"step": "validate_ohlc", "rows": len(d)})
    d["volume"] = d["volume"].fillna(0).clip(lower=0)
    log_return = np.log(d["close"]).diff()
    median, mad = log_return.median(), (log_return - log_return.median()).abs().median()
    if pd.notna(mad) and mad > 0:
        z = (log_return - median).abs() / (1.4826 * mad)
        d = d[(z <= max_return_z) | z.isna()]
    report.append({"step": "remove_return_outliers", "rows": len(d)})
    d = d[(d["open"] != d["high"]) | (d["high"] != d["low"]) | (d["low"] != d["close"])]
    if len(d) > max_rows:
        d = d.tail(max_rows)
    d = d.reset_index(drop=True)
    report.append({"step": f"keep_latest_{max_rows}", "rows": len(d)})
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
    return d.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["target_return"]).reset_index(drop=True)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MAPE": float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100),
    }


def train_linear(data: pd.DataFrame, split: int, pca_components: int = 0) -> tuple[LinearRegression, dict, dict]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(data.loc[:split - 1, FEATURES])
    x_test = scaler.transform(data.loc[split:, FEATURES])
    reducer = PCA(n_components=pca_components, random_state=42) if pca_components else None
    if reducer:
        x_train, x_test = reducer.fit_transform(x_train), reducer.transform(x_test)
    model = LinearRegression().fit(x_train, data.loc[:split - 1, "target_return"])
    base = data.loc[split:, "close"].to_numpy()
    actual = data.loc[split:, "close"].to_numpy() * (1 + data.loc[split:, "target_return"].to_numpy())
    predicted = base * (1 + model.predict(x_test))
    artifacts = {"scaler": scaler, "pca": reducer, "model": model}
    return model, metrics(actual, predicted), {"actual": actual, "predicted": predicted, "dates": data.loc[split:, "date"].astype(str).to_numpy(), "artifacts": artifacts}


def build_lstm_sequences(data: pd.DataFrame, window: int, split: int):
    scaler = StandardScaler()
    scaler.fit(data.loc[:split - 1, FEATURES])
    scaled = scaler.transform(data[FEATURES])
    xs, ys, bases, dates, indices = [], [], [], [], []
    for end in range(window - 1, len(data)):
        xs.append(scaled[end - window + 1:end + 1])
        ys.append(data.loc[end, "target_return"])
        bases.append(data.loc[end, "close"])
        dates.append(str(data.loc[end, "date"]))
        indices.append(end)
    indices = np.asarray(indices)
    train = indices < split
    test = ~train
    return (
        np.asarray(xs)[train], np.asarray(ys)[train],
        np.asarray(xs)[test], np.asarray(ys)[test],
        np.asarray(bases)[test], np.asarray(dates)[test],
        scaler,
    )


def train_lstm(data: pd.DataFrame, split: int, window: int = 60, epochs: int = 40, batch_size: int = 256):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    x_train, y_train, x_test, y_test, bases, dates, scaler = build_lstm_sequences(data, window, split)
    model = Sequential([
        Input(shape=(window, len(FEATURES))),
        LSTM(96, return_sequences=True),
        Dropout(0.20),
        LSTM(48),
        Dropout(0.20),
        Dense(24, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss="mse", metrics=["mae"])
    history = model.fit(
        x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size,
        callbacks=[EarlyStopping(patience=7, restore_best_weights=True), ReduceLROnPlateau(patience=3, factor=0.5)],
        verbose=1,
    )
    predicted_return = model.predict(x_test, batch_size=batch_size, verbose=0).ravel()
    actual = bases * (1 + y_test)
    predicted = bases * (1 + predicted_return)
    return model, metrics(actual, predicted), {
        "actual": actual, "predicted": predicted, "dates": dates,
        "history": history.history, "artifacts": {"scaler": scaler, "window": window},
    }


def save_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
