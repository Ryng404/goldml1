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
        help="Các số thành phần PCA cần thử cho LSTM.",
    )
    parser.add_argument(
        "--allow-linear-fallback",
        action="store_true",
        help="Chỉ chạy Linear Regression nếu TensorFlow không khả dụng.",
    )
    args = parser.parse_args()

    lstm_error = None
    try:
        import tensorflow  # noqa: F401
    except (ImportError, OSError) as exc:
        if "Application Control" in str(exc):
            lstm_error = (
                "Windows Application Control đang chặn TensorFlow DLL. "
                "Hãy chạy trong môi trường Linux/WSL2 được phép hoặc nhờ "
                "quản trị viên cho phép thư viện TensorFlow."
            )
        else:
            lstm_error = (
                "TensorFlow không tải được. Kiểm tra phiên bản Python, "
                "TensorFlow và môi trường cài đặt."
            )
        if not args.allow_linear_fallback:
            parser.error(
                f"{lstm_error}\n"
                "Nếu chỉ muốn chạy baseline Linear Regression, thêm "
                "--allow-linear-fallback."
            )
        print(
            "Cảnh báo: TensorFlow không khả dụng; "
            "đang chạy baseline theo yêu cầu --allow-linear-fallback."
        )

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
    knn_metrics, knn_artifacts = train_knn(data, split)

    joblib.dump(
        {"default": linear_output["artifacts"]},
        model_dir / "linear.joblib",
    )
    pd.DataFrame(
        {
            "date": linear_output["dates"],
            "actual": linear_output["actual"],
            "predicted": linear_output["predicted"],
        }
    ).to_csv(model_dir / "linear_predictions.csv", index=False)
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
    }

    prediction_output = linear_output
    prediction_source = "Linear Regression" if lstm_error else "LSTM (22 features)"
    if not lstm_error:
        variants = [(None, "LSTM (22 features)", "lstm")]
        variants.extend(
            (components, f"LSTM + PCA {components} components", f"lstm_pca{components}")
            for components in dict.fromkeys(args.pca_components)
        )
        for components, label, model_name in variants:
            lstm, lstm_metrics, lstm_output = train_lstm(
                data,
                split,
                args.window,
                args.epochs,
                pca_components=components,
            )
            joblib.dump(
                lstm_output["artifacts"],
                model_dir / f"{model_name}_preprocess.joblib",
            )
            lstm.save(model_dir / f"{model_name}.keras")
            pd.DataFrame(
                {
                    "date": lstm_output["dates"],
                    "actual": lstm_output["actual"],
                    "predicted": lstm_output["predicted"],
                }
            ).to_csv(model_dir / f"{model_name}_predictions.csv", index=False)
            all_metrics[label] = lstm_metrics
            all_metrics[f"{model_name}_history"] = lstm_output["history"]
            if components is None:
                prediction_output = lstm_output
                save_json(model_dir / "history.json", lstm_output["history"])

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