from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="GoldML", page_icon="📈", layout="wide")
st.title("📈 GoldML - Dự đoán giá vàng")
st.caption("LSTM là mô hình chính; Hồi quy tuyến tính và Naive được dùng để đối chiếu.")

metrics_path = MODEL_DIR / "metrics.json"
pred_path = MODEL_DIR / "predictions.csv"
if not metrics_path.exists() or not pred_path.exists():
    st.warning("Chưa có model. Hãy chạy: `python train.py --csv <file-kaggle.csv> --max-rows 100000`")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
pred = pd.read_csv(pred_path, parse_dates=["date"])
left, right = st.columns(2)
with left:
    st.metric("Số dòng đã xử lý", f"{metrics['rows']:,}")
with right:
    st.metric("Cửa sổ LSTM", f"{metrics['window']} phiên")

st.subheader("So sánh mô hình")
score = pd.DataFrame(metrics).T.drop(index=["rows", "split", "window"], errors="ignore")
st.dataframe(score.style.format("{:.6f}"), use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=pred["date"], y=pred["actual"], name="Giá thật", line={"color": "#111827"}))
fig.add_trace(go.Scatter(x=pred["date"], y=pred["predicted"], name="LSTM", line={"color": "#b45309"}))
fig.update_layout(height=480, xaxis_title="Thời gian", yaxis_title="Giá đóng cửa", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Kết quả gần nhất")
st.dataframe(pred.tail(100).sort_values("date", ascending=False), use_container_width=True, hide_index=True)
