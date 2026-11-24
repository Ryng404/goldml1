import argparse
from pathlib import Path

import joblib
import pandas as pd

from pipeline import (
    clean_data,
    load_gold_csv,
    make_features,
    save_json,
    train_direction_classifiers,
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
        help="Các số chiều PCA cần thử cho K-NN và Logistic Regression.",
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

    linear_pca_components = args.pca_components[0] if args.pca_components else 0
    _, linear_metrics, linear_output = train_linear(data, split, linear_pca_components)
    linear_experiment_metrics = {}
    linear_experiment_artifacts = {}
    for components in args.pca_components:
        _, variant_metrics, variant_output = train_linear(
            data,
            split,
            int(components),
        )
        linear_experiment_metrics[f"Linear Regression (PCA={components})"] = variant_metrics
        linear_experiment_artifacts[f"PCA={components}"] = variant_output["artifacts"]
    classifier_metrics, classifier_artifacts = train_direction_classifiers(
        data,
        split,
        args.pca_components,
    )

    joblib.dump(
        {"default": linear_output["artifacts"], "variants": linear_experiment_artifacts},
        model_dir / "linear.joblib",
    )
    joblib.dump(classifier_artifacts, model_dir / "classifiers.joblib")
    report.to_csv(model_dir / "cleaning_report.csv", index=False)
    data.to_csv(model_dir / "features.csv", index=False)

    all_metrics = {
        "Linear Regression": linear_metrics,
        **linear_experiment_metrics,
        **classifier_metrics,
        "rows": len(data),
        "split": split,
        "window": args.window,
        "pca_components": args.pca_components,
    }

    prediction_output = linear_output
    prediction_source = f"Linear Regression (PCA={linear_pca_components})"
    lstm_error = None
    try:
        lstm, lstm_metrics, lstm_output = train_lstm(data, split, args.window, args.epochs)
        lstm.save(model_dir / "lstm.keras")
        joblib.dump(lstm_output["artifacts"], model_dir / "lstm_preprocess.joblib")
        prediction_output = lstm_output
        prediction_source = "LSTM"
        all_metrics["LSTM"] = lstm_metrics
        all_metrics["history"] = lstm_output["history"]
        save_json(model_dir / "history.json", lstm_output["history"])
    except ImportError as exc:
        lstm_error = str(exc)
        print("Cảnh báo: Không thể huấn luyện LSTM do thư viện TensorFlow bị policy chặn.")
        print("Dự đoán tạm thời trong web app sẽ dùng Linear Regression; bảng K-NN/Logistic vẫn được lưu.")

    pd.DataFrame({
        "date": prediction_output["dates"],
        "actual": prediction_output["actual"],
        "predicted": prediction_output["predicted"],
    }).to_csv(model_dir / "predictions.csv", index=False)

    all_metrics["prediction_source"] = prediction_source
    if lstm_error:
        all_metrics["lstm_error"] = lstm_error

    save_json(model_dir / "metrics.json", all_metrics)
    print(f"Đã train {len(data):,} dòng. Dự đoán đang dùng: {prediction_source}")


if __name__ == "__main__":
    main()
