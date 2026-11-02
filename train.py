import argparse
from pathlib import Path

import joblib
import pandas as pd

from pipeline import clean_data, load_gold_csv, make_features, save_json, train_linear, train_lstm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--pca-components", type=int, default=12)
    args = parser.parse_args()

    root = Path(__file__).parent
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    raw = load_gold_csv(args.csv)
    clean, report = clean_data(raw, max_rows=args.max_rows)
    data = make_features(clean)
    split = int(len(data) * 0.85)
    if split <= args.window or len(data) - split < 10:
        raise ValueError("Không đủ dữ liệu sau làm sạch để chia train/test.")

    _, linear_metrics, linear_output = train_linear(data, split, args.pca_components)
    lstm, lstm_metrics, lstm_output = train_lstm(data, split, args.window, args.epochs)
    joblib.dump(linear_output["artifacts"], model_dir / "linear.joblib")
    joblib.dump(lstm_output["artifacts"], model_dir / "lstm_preprocess.joblib")
    lstm.save(model_dir / "lstm.keras")
    pd.DataFrame({"date": lstm_output["dates"], "actual": lstm_output["actual"], "predicted": lstm_output["predicted"]}).to_csv(model_dir / "predictions.csv", index=False)
    report.to_csv(model_dir / "cleaning_report.csv", index=False)
    data.to_csv(model_dir / "features.csv", index=False)
    save_json(model_dir / "metrics.json", {"Linear Regression": linear_metrics, "LSTM": lstm_metrics, "rows": len(data), "split": split, "window": args.window})
    save_json(model_dir / "history.json", lstm_output["history"])
    print(f"Đã train {len(data):,} dòng. LSTM: {lstm_metrics}")


if __name__ == "__main__":
    main()
