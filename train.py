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
    lstm, lstm_metrics, lstm_output = train_lstm(data, split, args.window, args.epochs)
    joblib.dump(
        {
            "default": linear_output["artifacts"],
            "variants": linear_experiment_artifacts,
        },
        model_dir / "linear.joblib",
    )
    joblib.dump(classifier_artifacts, model_dir / "classifiers.joblib")
    joblib.dump(lstm_output["artifacts"], model_dir / "lstm_preprocess.joblib")
    lstm.save(model_dir / "lstm.keras")
    pd.DataFrame({"date": lstm_output["dates"], "actual": lstm_output["actual"], "predicted": lstm_output["predicted"]}).to_csv(model_dir / "predictions.csv", index=False)
    report.to_csv(model_dir / "cleaning_report.csv", index=False)
    data.to_csv(model_dir / "features.csv", index=False)
    save_json(
        model_dir / "metrics.json",
        {
            "Linear Regression": linear_metrics,
            **linear_experiment_metrics,
            **classifier_metrics,
            "LSTM": lstm_metrics,
            "rows": len(data),
            "split": split,
            "window": args.window,
            "pca_components": args.pca_components,
        },
    )
    save_json(model_dir / "history.json", lstm_output["history"])
    print(f"Đã train {len(data):,} dòng. LSTM: {lstm_metrics}")


if __name__ == "__main__":
    main()