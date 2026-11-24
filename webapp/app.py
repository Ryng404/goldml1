from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="GoldML", page_icon="📈", layout="wide")
st.title("📈 GoldML - Dự đoán giá vàng")
st.caption("Chọn một thời điểm trong tập kiểm tra để xem dự đoán.")

metrics_path = MODEL_DIR / "metrics.json"
pred_path = MODEL_DIR / "predictions.csv"
if not metrics_path.exists() or not pred_path.exists():
    st.warning("Chưa có model. Hãy chạy: `python train.py --csv <file-kaggle.csv> --max-rows 100000`")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
prediction_source = metrics.get("prediction_source", "LSTM")
pred = pd.read_csv(pred_path, parse_dates=["date"])
pred = pred.dropna(subset=["date", "actual", "predicted"]).sort_values("date").reset_index(drop=True)
if pred.empty:
    st.error("File predictions.csv không có dữ liệu hợp lệ.")
    st.stop()

left, right = st.columns(2)
with left:
    st.metric("Số dòng đã xử lý", f"{metrics['rows']:,}")
with right:
    st.metric("Nguồn dự đoán", prediction_source)

st.subheader("Dự đoán tại thời điểm chọn")
st.info(
    "Danh sách thời điểm lấy từ tập test lịch sử đã lưu trong `predictions.csv`. "
    "Đây là chức năng xem lại dự đoán, chưa phải dữ liệu giá trực tiếp theo thời gian thực."
)

if metrics.get("lstm_error"):
    st.warning("LSTM chưa thể chạy vì Windows/Application Control chặn TensorFlow. Hiện đang dùng Linear Regression.")

date_min = pred["date"].min().date()
date_max = pred["date"].max().date()
selected_date = st.date_input(
    "Chọn ngày cần xem dự đoán",
    value=date_max,
    min_value=date_min,
    max_value=date_max,
)
same_day = pred[pred["date"].dt.date == selected_date]

if same_day.empty:
    nearest_index = pred["date"].dt.date.map(lambda value: abs(value - selected_date)).idxmin()
    selected_row = pred.loc[[nearest_index]]
    st.warning(f"Không có phiên đúng ngày {selected_date:%d/%m/%Y}; đang hiển thị phiên gần nhất.")
else:
    selected_row = same_day.tail(1)

row = selected_row.iloc[0]
error = float(row["predicted"] - row["actual"])
error_pct = abs(error) / max(abs(float(row["actual"])), 1e-8) * 100
st.caption(f"Phiên được chọn: {row['date']:%d/%m/%Y %H:%M}")
metric_cols = st.columns(4)
metric_cols[0].metric("Giá thật", f"{row['actual']:,.2f}")
metric_cols[1].metric(f"Giá {prediction_source} dự đoán", f"{row['predicted']:,.2f}")
metric_cols[2].metric("Chênh lệch", f"{error:+,.2f}")
metric_cols[3].metric("Sai số tương đối", f"{error_pct:.3f}%")

selected_fig = go.Figure()
selected_fig.add_trace(go.Scatter(
    x=pred["date"], y=pred["actual"], name="Giá thật",
    line={"color": "#111827", "width": 1.5},
))
selected_fig.add_trace(go.Scatter(
    x=pred["date"], y=pred["predicted"], name=prediction_source,
    line={"color": "#b45309", "width": 1.5},
))
selected_fig.add_trace(go.Scatter(
    x=[row["date"]], y=[row["predicted"]], name="Thời điểm chọn",
    mode="markers", marker={"color": "#dc2626", "size": 12},
))
selected_fig.update_layout(
    height=480, xaxis_title="Thời gian", yaxis_title="Giá đóng cửa",
    hovermode="x unified",
)
st.plotly_chart(selected_fig, use_container_width=True)
st.download_button(
    "Tải kết quả thời điểm đã chọn",
    data=selected_row.to_csv(index=False).encode("utf-8"),
    file_name=f"gold_prediction_{row['date']:%Y%m%d_%H%M}.csv",
    mime="text/csv",
)

st.subheader("So sánh mô hình")
ignored_keys = {
    "rows", "split", "window", "pca_components", "pca_report",
    "prediction_source", "lstm_error", "history",
}
model_metrics = pd.DataFrame.from_dict(
    {name: values for name, values in metrics.items() if name not in ignored_keys},
    orient="index",
)

price_names = [
    name for name in model_metrics.index
    if name.startswith("Linear Regression") or name == "LSTM"
]
price_metrics = model_metrics.loc[
    [name for name in price_names if {"MAE", "RMSE", "R2", "MAPE"}.intersection(model_metrics.loc[name].dropna().index)],
    [column for column in ["MAE", "RMSE", "R2", "MAPE"] if column in model_metrics.columns],
]
if not price_metrics.empty:
    st.dataframe(price_metrics.style.format("{:.6f}", na_rep="-"), use_container_width=True)
else:
    st.info("Chưa có kết quả mô hình dự đoán giá.")

rf_names = [name for name in model_metrics.index if name.startswith("Random Forest")]
if rf_names:
    rf_columns = [column for column in ["Accuracy", "Precision", "Recall", "F1"] if column in model_metrics.columns]
    st.markdown("**Random Forest theo số chiều PCA**")
    st.dataframe(
        model_metrics.loc[rf_names, rf_columns].style.format("{:.6f}", na_rep="-"),
        use_container_width=True,
    )

st.subheader("Thay đổi PCA")
st.caption("PCA giảm số chiều đặc trưng kỹ thuật từ 22 xuống 12, 6, 3 và 1 chiều.")
pca_report = metrics.get("pca_report", {})
if pca_report:
    pca_table = pd.DataFrame(pca_report).T
    st.dataframe(pca_table.style.format({"Tỷ lệ phương sai PCA": "{:.2f}%"}), use_container_width=True)

    pca_order = [
        "22 đặc trưng gốc", "PCA 12", "PCA 6", "PCA 3", "PCA 1"
    ]
    pca_order = [label for label in pca_order if label in pca_report]
    pca_fig = go.Figure()
    pca_fig.add_trace(go.Scatter(
        x=[pca_report[label]["Số đặc trưng"] for label in pca_order],
        y=[pca_report[label]["Tỷ lệ phương sai PCA"] for label in pca_order],
        mode="lines+markers+text",
        text=pca_order,
        textposition="top center",
        name="Tỷ lệ phương sai",
    ))
    pca_fig.update_layout(
        height=360,
        xaxis={"title": "Số chiều", "autorange": "reversed"},
        yaxis={"title": "Phương sai giữ lại (%)", "range": [0, 105]},
    )
    st.plotly_chart(pca_fig, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pred["date"], y=pred["actual"], name="Giá thật",
    line={"color": "#111827"},
))
fig.add_trace(go.Scatter(
    x=pred["date"], y=pred["predicted"], name=prediction_source,
    line={"color": "#b45309"},
))
fig.update_layout(
    height=480, xaxis_title="Thời gian", yaxis_title="Giá đóng cửa",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Xem 100 kết quả gần nhất"):
    st.dataframe(
        pred.tail(100).sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
