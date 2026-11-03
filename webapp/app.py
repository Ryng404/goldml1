from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="GoldML", page_icon="📈", layout="wide")
st.title("📈 GoldML - Dự đoán giá vàng")
st.caption("LSTM là mô hình chính. Chọn một thời điểm trong tập kiểm tra để xem dự đoán.")

metrics_path = MODEL_DIR / "metrics.json"
pred_path = MODEL_DIR / "predictions.csv"
if not metrics_path.exists() or not pred_path.exists():
    st.warning("Chưa có model. Hãy chạy: `python train.py --csv <file-kaggle.csv> --max-rows 100000`")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
pred = pd.read_csv(pred_path, parse_dates=["date"])
pred = pred.dropna(subset=["date", "actual", "predicted"]).sort_values("date").reset_index(drop=True)
if pred.empty:
    st.error("File predictions.csv không có dữ liệu hợp lệ.")
    st.stop()

left, right = st.columns(2)
with left:
    st.metric("Số dòng đã xử lý", f"{metrics['rows']:,}")
with right:
    st.metric("Cửa sổ LSTM", f"{metrics['window']} phiên")

st.subheader("Dự đoán tại thời điểm chọn")
st.info(
    "Danh sách thời điểm lấy từ tập test lịch sử đã lưu trong `predictions.csv`. "
    "Đây là chức năng xem lại dự đoán, chưa phải dữ liệu giá trực tiếp theo thời gian thực."
)

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
    nearest_index = (pred["date"].dt.date.map(lambda value: abs(value - selected_date))).idxmin()
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
metric_cols[1].metric("Giá LSTM dự đoán", f"{row['predicted']:,.2f}")
metric_cols[2].metric("Chênh lệch", f"{error:+,.2f}")
metric_cols[3].metric("Sai số tương đối", f"{error_pct:.3f}%")

selected_fig = go.Figure()
selected_fig.add_trace(go.Scatter(
    x=pred["date"], y=pred["actual"], name="Giá thật",
    line={"color": "#111827", "width": 1.5},
))
selected_fig.add_trace(go.Scatter(
    x=pred["date"], y=pred["predicted"], name="LSTM",
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
score = pd.DataFrame(metrics).T.drop(index=["rows", "split", "window"], errors="ignore")
st.dataframe(score.style.format("{:.6f}"), use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=pred["date"], y=pred["actual"], name="Giá thật", line={"color": "#111827"}))
fig.add_trace(go.Scatter(x=pred["date"], y=pred["predicted"], name="LSTM", line={"color": "#b45309"}))
fig.update_layout(height=480, xaxis_title="Thời gian", yaxis_title="Giá đóng cửa", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Xem 100 kết quả gần nhất"):
    st.dataframe(pred.tail(100).sort_values("date", ascending=False), use_container_width=True, hide_index=True)
