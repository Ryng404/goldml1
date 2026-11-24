import argparse
from pathlib import Path

import joblib
import pandas as pd

from pipeline import (
    clean_data,
    load_gold_csv,
    make_features,
    save_json,
    train_knn,
    train_linear,
    train_lstm,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--pca-components",
        nargs="+",
        type=int,
        default=[12, 6, 3, 1],
        help="Các số chiều PCA cần thử cho K-NN.",
    )
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

    _, linear_metrics, linear_output = train_linear(data, split)
    knn_metrics, knn_artifacts, pca_report = train_knn(
        data,
        split,
        args.pca_components,
    )

    joblib.dump(
        {"default": linear_output["artifacts"]},
        model_dir / "linear.joblib",
    )
    joblib.dump(knn_artifacts, model_dir / "knn.joblib")
    report.to_csv(model_dir / "cleaning_report.csv", index=False)
    data.to_csv(model_dir / "features.csv", index=False)

    all_metrics = {
        "Linear Regression": linear_metrics,
        **knn_metrics,
        "rows": len(data),
        "split": split,
        "window": args.window,
        "pca_components": args.pca_components,
        "pca_report": pca_report,
    }

    prediction_output = linear_output
    prediction_source = "Linear Regression"
    lstm_error = None
    try:
        lstm, lstm_metrics, lstm_output = train_lstm(data, split, args.window, args.epochs)
        joblib.dump(lstm_output["artifacts"], model_dir / "lstm_preprocess.joblib")
        lstm.save(model_dir / "lstm.keras")
        prediction_output = lstm_output
        prediction_source = "LSTM"
        all_metrics["LSTM"] = lstm_metrics
        all_metrics["history"] = lstm_output["history"]
        save_json(model_dir / "history.json", lstm_output["history"])
    except ImportError as exc:
        lstm_error = str(exc)
        print("Cảnh báo: Không thể huấn luyện LSTM vì TensorFlow bị Application Control chặn.")
        print("Đang lưu dự đoán tạm bằng Linear Regression.")

    pd.DataFrame(
        {
            "date": prediction_output["dates"],
            "actual": prediction_output["actual"],
            "predicted": prediction_output["predicted"],
        }
    ).to_csv(model_dir / "predictions.csv", index=False)

    all_metrics["prediction_source"] = prediction_source
    if lstm_error:
        all_metrics["lstm_error"] = lstm_error
    save_json(model_dir / "metrics.json", all_metrics)
    print(f"Đã train {len(data):,} dòng. Dự đoán đang dùng: {prediction_source}")


if __name__ == "__main__":
    main()