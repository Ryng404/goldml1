from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import infer_gap_threshold

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_VARIANTS = [
    ("Linear Regression", "linear"),
    ("LSTM (22 features)", "lstm"),
    ("LSTM + PCA 12 components", "lstm_pca12"),
    ("LSTM + PCA 6 components", "lstm_pca6"),
    ("LSTM + PCA 3 components", "lstm_pca3"),
    ("LSTM + PCA 1 component", "lstm_pca1"),
]

st.set_page_config(
    page_title="GoldML | Gold Market",
    page_icon="Au",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #080d16;
        --panel: #101927;
        --line: #202c3b;
        --muted: #91a0b5;
        --text: #e7edf5;
        --gold: #f0b90b;
        --green: #0ecb81;
        --red: #f6465d;
    }
    .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #0c1420;
        border-right: 1px solid var(--line);
    }
    .block-container { max-width: 1500px; padding-top: 1.1rem; padding-bottom: 3rem; }
    .market-header {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid var(--line); padding: 4px 0 18px; margin-bottom: 22px;
    }
    .brand { color: var(--gold); font-weight: 800; font-size: 1.1rem; letter-spacing: .04em; }
    .market-tag { color: var(--muted); font-size: .82rem; }
    .symbol { font-size: 1.6rem; font-weight: 750; margin: 0; color: var(--text); }
    .sub-symbol { color: var(--muted); font-size: .9rem; }
    .quote {
        border: 1px solid var(--line); background: linear-gradient(120deg, #121d2b, #0d1521);
        border-radius: 12px; padding: 17px 20px; margin: 12px 0 18px;
    }
    .quote-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
    .quote-value { font-size: 2rem; font-weight: 750; line-height: 1.25; color: var(--text); }
    .quote-note { color: var(--muted); font-size: .82rem; }
    [data-testid="stMetric"] {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 10px; padding: 13px 15px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--text); font-size: 1.25rem; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 9px; }
    h1, h2, h3 { color: var(--text); }
    div[data-baseweb="select"] > div { background: var(--panel); border-color: var(--line); }
    .section-label {
        color: var(--text); font-size: 1.05rem; font-weight: 650;
        margin: 1.1rem 0 .4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

metrics_path = MODEL_DIR / "metrics.json"
if not metrics_path.exists():
    st.warning(
        "Chưa có kết quả huấn luyện. Chạy `python train.py --csv <file-kaggle.csv>` "
        "để tạo các cấu hình LSTM và dữ liệu hiển thị."
    )
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
available_models = []
for label, model_name in MODEL_VARIANTS:
    pred_path = MODEL_DIR / f"{model_name}_predictions.csv"
    if label in metrics and pred_path.exists():
        available_models.append((label, model_name, pred_path))

# Support older training output that only saved the selected model predictions.
legacy_lstm_path = MODEL_DIR / "predictions.csv"
if not any(model_name == "lstm" for _, model_name, _ in available_models):
    if "LSTM (22 features)" in metrics and legacy_lstm_path.exists():
        available_models.insert(0, ("LSTM (22 features)", "lstm", legacy_lstm_path))
    elif metrics.get("prediction_source") in ("LSTM", "LSTM (22 features)") and legacy_lstm_path.exists():
        available_models.insert(0, ("LSTM (22 features)", "lstm", legacy_lstm_path))
if (
    "Linear Regression" in metrics
    and not any(model_name == "linear" for _, model_name, _ in available_models)
    and metrics.get("prediction_source") == "Linear Regression"
    and legacy_lstm_path.exists()
):
    available_models.insert(0, ("Linear Regression", "linear", legacy_lstm_path))

if not available_models:
    if metrics.get("lstm_error"):
        st.error(f"Không có mô hình LSTM để hiển thị. Chi tiết: {metrics['lstm_error']}")
    else:
        st.warning("Chưa tìm thấy kết quả LSTM. Hãy huấn luyện lại để tạo đủ các biến thể PCA.")
    st.stop()

with st.sidebar:
    st.markdown("## GoldML")
    st.caption("GOLD MARKET · MODEL TERMINAL")
    st.divider()
    st.markdown("**Chọn mô hình dự đoán**")
    selected_label = st.selectbox(
        "Cấu hình",
        [model[0] for model in available_models],
        label_visibility="collapsed",
    )
    selected_model = next(model for model in available_models if model[0] == selected_label)
    st.divider()
    st.metric("Bản ghi đã xử lý", f"{metrics.get('rows', 0):,}")
    st.metric("Cửa sổ quan sát", f"{metrics.get('window', 60)} phiên")
    st.caption("Các chỉ số được tính trên tập kiểm tra theo thứ tự thời gian.")

selected_label, selected_model_name, selected_path = selected_model
pred = pd.read_csv(selected_path, parse_dates=["date"])
pred["actual"] = pd.to_numeric(pred["actual"], errors="coerce")
pred["predicted"] = pd.to_numeric(pred["predicted"], errors="coerce")
pred = (
    pred.dropna(subset=["date", "actual"])
    .sort_values("date")
    .reset_index(drop=True)
)
if pred.empty:
    st.error(f"Không có dự đoán hợp lệ trong {selected_path.name}.")
    st.stop()

forecast_rows = pred.dropna(subset=["predicted"])
if forecast_rows.empty:
    st.error(f"Không có giá dự đoán hợp lệ trong {selected_path.name}. Hãy train lại mô hình.")
    st.stop()

last = pred.iloc[-1]
last_forecast = forecast_rows.iloc[-1]
previous = pred.iloc[-2] if len(pred) > 1 else last
change = float(last["actual"] - previous["actual"])
change_pct = change / max(abs(float(previous["actual"])), 1e-8) * 100
selected_metrics = metrics.get(selected_label, metrics.get("LSTM", {}))
error = float(last_forecast["predicted"] - last_forecast["actual"])
error_pct = abs(error) / max(abs(float(last_forecast["actual"])), 1e-8) * 100

st.markdown(
    """
    <div class="market-header">
      <div class="brand">GOLDML <span class="market-tag">/ MARKET ANALYTICS</span></div>
      <div class="market-tag">MACHINE LEARNING · TIME SERIES</div>
    </div>
    """,
    unsafe_allow_html=True,
)

head_left, head_right = st.columns([3, 1])
with head_left:
    st.markdown('<p class="symbol">XAU · GOLD</p>', unsafe_allow_html=True)
    st.markdown(
        '<span class="sub-symbol">Giá đóng cửa kiểm tra · '
        f'Cấu hình đang xem: {selected_label}</span>',
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown(
        '<div style="text-align:right;color:#91a0b5;padding-top:10px">'
        f'TẬP KIỂM TRA<br><b style="color:#e7edf5">{len(pred):,} phiên</b></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="quote"><div class="quote-label">Giá thực gần nhất trong tập kiểm tra</div>'
    f'<div class="quote-value">{last["actual"]:,.2f}</div>'
    f'<div class="quote-note">{last["date"]:%d/%m/%Y} · '
    f'<span style="color:{"#0ecb81" if change >= 0 else "#f6465d"}">'
    f'{change:+,.2f} ({change_pct:+.3f}%)</span> so với phiên kiểm tra trước</div></div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Giá dự đoán", f'{last_forecast["predicted"]:,.2f}')
k2.metric("Sai lệch gần nhất", f"{error:+,.2f}")
k3.metric("Sai số tương đối", f"{error_pct:.3f}%")
k4.metric("RMSE mô hình", f'{selected_metrics.get("RMSE", float("nan")):,.4f}')
k5.metric(
    "RMSE Naive",
    f'{metrics.get("Naive Baseline", {}).get("RMSE", float("nan")):,.4f}',
)

chart_controls, _ = st.columns([1, 3])
with chart_controls:
    chart_range = st.selectbox(
        "Khoảng thời gian",
        ["Toàn bộ", "7 ngày", "1 tháng", "3 tháng", "6 tháng", "1 năm"],
        label_visibility="collapsed",
    )
range_days = {"7 ngày": 7, "1 tháng": 30, "3 tháng": 90, "6 tháng": 180, "1 năm": 365}
chart_data = pred
if chart_range in range_days:
    cutoff = pred["date"].max() - pd.Timedelta(days=range_days[chart_range])
    chart_data = pred[pred["date"] >= cutoff].reset_index(drop=True)

gap_threshold = infer_gap_threshold(pred["date"])
gap_intervals = []
if len(chart_data) > 1:
    time_deltas = chart_data["date"].diff()
    gap_intervals = [
        (chart_data.loc[index - 1, "date"], chart_data.loc[index, "date"])
        for index in time_deltas[time_deltas > gap_threshold].index
    ]

plot_x, plot_actual, plot_predicted, plot_naive = [], [], [], []
for index, row in chart_data.iterrows():
    if index and row["date"] - chart_data.loc[index - 1, "date"] > gap_threshold:
        midpoint = chart_data.loc[index - 1, "date"] + (
            row["date"] - chart_data.loc[index - 1, "date"]
        ) / 2
        plot_x.append(midpoint)
        plot_actual.append(None)
        plot_predicted.append(None)
        plot_naive.append(None)
    plot_x.append(row["date"])
    plot_actual.append(row["actual"])
    plot_predicted.append(row["predicted"])
    plot_naive.append(row.get("naive_predicted"))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=plot_x,
        y=plot_actual,
        name="Giá thực tế",
        mode="lines",
        line={"color": "#91a0b5", "width": 1.8},
        connectgaps=False,
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_x,
        y=plot_predicted,
        name="Giá dự đoán",
        mode="lines",
        line={"color": "#f0b90b", "width": 2},
        connectgaps=False,
    )
)
if "naive_predicted" in chart_data:
    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=plot_naive,
            name="Naive (giá phiên trước)",
            mode="lines",
            line={"color": "#f6465d", "width": 1, "dash": "dot"},
            connectgaps=False,
        )
    )
for gap_start, gap_end in gap_intervals:
    fig.add_vrect(
        x0=gap_start,
        x1=gap_end,
        fillcolor="rgba(246,70,93,.12)",
        line_width=0,
        layer="below",
    )
if "split" in chart_data:
    transitions = chart_data.loc[
        chart_data["split"].ne(chart_data["split"].shift())
    ].iloc[1:]
    for _, boundary in transitions.iterrows():
        fig.add_vline(
            x=boundary["date"],
            line_dash="dash",
            line_color="#7c8da5",
            annotation_text=f"{boundary['split'].capitalize()} start",
            annotation_position="top",
        )
fig.update_layout(
    template="plotly_dark",
    height=470,
    paper_bgcolor="#080d16",
    plot_bgcolor="#080d16",
    margin={"l": 8, "r": 16, "t": 24, "b": 8},
    hovermode="x unified",
    legend={"orientation": "h", "y": 1.08, "x": 0},
    xaxis={"title": "", "showgrid": False, "rangeslider": {"visible": False}},
    yaxis={"title": "Giá đóng cửa", "gridcolor": "#202c3b"},
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Vùng đỏ biểu thị {len(gap_intervals)} khoảng trống lớn hơn "
    f"{gap_threshold}; đường dự đoán và Naive chỉ có ở tập Test. "
    "Chọn 7 ngày để kiểm tra độ trễ pha ở cận cảnh."
)

st.markdown('<div class="section-label">Hiệu suất các mô hình dự đoán</div>', unsafe_allow_html=True)
comparison_rows = []
for label, model_name, _ in available_models:
    row_metrics = metrics.get(label, {})
    comparison_rows.append(
        {
            "Mô hình": label,
            "Số chiều đầu vào": (
                22 if model_name in ("linear", "lstm")
                else int(model_name.removeprefix("lstm_pca"))
            ),
            "MAE": row_metrics.get("MAE"),
            "RMSE": row_metrics.get("RMSE"),
            "R²": row_metrics.get("R2"),
            "MAPE (%)": row_metrics.get("MAPE"),
            "Đang xem": "✓" if label == selected_label else "",
        }
    )
baseline_metrics = metrics.get("Naive Baseline")
if baseline_metrics:
    comparison_rows.append(
        {
            "Mô hình": "Naive Baseline (giá phiên trước)",
            "Số chiều đầu vào": 1,
            "MAE": baseline_metrics.get("MAE"),
            "RMSE": baseline_metrics.get("RMSE"),
            "R²": baseline_metrics.get("R2"),
            "MAPE (%)": baseline_metrics.get("MAPE"),
            "Đang xem": "",
        }
    )
comparison = pd.DataFrame(comparison_rows)
st.dataframe(
    comparison.style.format(
        {"MAE": "{:.5f}", "RMSE": "{:.5f}", "R²": "{:.5f}", "MAPE (%)": "{:.3f}"},
        na_rep="—",
    ),
    use_container_width=True,
    hide_index=True,
)
st.caption("Giá thực và giá dự đoán được lưu theo từng phiên trong tập kiểm tra; đây không phải dữ liệu giá trực tiếp.")

with st.expander("Xem dữ liệu dự đoán gần nhất"):
    recent = pred.tail(100).sort_values("date", ascending=False).copy()
    recent["Sai lệch"] = recent["predicted"] - recent["actual"]
    st.dataframe(recent, use_container_width=True, hide_index=True)
    st.download_button(
        "Tải CSV cấu hình đang xem",
        data=pred.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model_name}_predictions.csv",
        mime="text/csv",
    )

knn_metrics = {
    name: values for name, values in metrics.items()
    if name.startswith("K-NN")
}
if knn_metrics:
    with st.expander("K-NN phân loại xu hướng (22 đặc trưng, không PCA)"):
        st.dataframe(
            pd.DataFrame.from_dict(knn_metrics, orient="index"),
            use_container_width=True,
        )

linear_metrics = metrics.get("Linear Regression")
if linear_metrics:
    with st.expander("Linear Regression hồi quy giá (22 đặc trưng)"):
        st.dataframe(
            pd.DataFrame([linear_metrics], index=["Linear Regression"]),
            use_container_width=True,
        )
        if not (MODEL_DIR / "linear_predictions.csv").exists():
            st.info(
                "Bộ train này chưa lưu chuỗi dự đoán Linear Regression để xem trên biểu đồ. "
                "Train lại bằng phiên bản mới để tạo kết quả này."
            )
