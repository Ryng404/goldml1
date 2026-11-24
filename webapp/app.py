from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="GoldML Analytics",
    page_icon="Au",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #172554 0, #0f172a 38%, #020617 100%);
        color: #e5e7eb;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1f2937; }
    [data-testid="stMetric"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #f8fafc;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30,41,59,.96), rgba(15,23,42,.96));
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 12px 32px rgba(0,0,0,.18);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,.12);
        border-radius: 14px;
        overflow: hidden;
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1380px; }
    .hero {
        border: 1px solid rgba(250,204,21,.18);
        background: linear-gradient(135deg, rgba(250,204,21,.10), rgba(30,41,59,.82));
        border-radius: 22px;
        padding: 24px 28px;
        margin-bottom: 20px;
    }
    .hero h1 { color: #f8fafc; font-size: 2rem; margin: 0 0 6px 0; }
    .hero p { color: #94a3b8; margin: 0; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 1px solid rgba(148,163,184,.12);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px 10px 0 0;
        color: #94a3b8;
        padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        color: #facc15;
        border-bottom: 2px solid #facc15;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

metrics_path = MODEL_DIR / "metrics.json"
pred_path = MODEL_DIR / "predictions.csv"

st.markdown(
    '<div class="hero"><h1>GoldML Analytics</h1>'
    '<p>Phân tích dự đoán giá vàng bằng Linear Regression, LSTM và PCA.</p></div>',
    unsafe_allow_html=True,
)

if not metrics_path.exists() or not pred_path.exists():
    st.warning(
        "Chưa có model. Chạy `python train.py --csv <file-kaggle.csv> --max-rows 100000` "
        "rồi khởi động lại giao diện."
    )
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
prediction_source = metrics.get("prediction_source", "LSTM")
pred = pd.read_csv(pred_path, parse_dates=["date"])
pred = pred.dropna(subset=["date", "actual", "predicted"]).sort_values("date").reset_index(drop=True)

if pred.empty:
    st.error("File predictions.csv không có dữ liệu hợp lệ.")
    st.stop()

with st.sidebar:
    st.markdown("### GoldML")
    st.caption("Dự đoán giá vàng")
    st.divider()
    st.metric("Dòng dữ liệu", f"{metrics.get('rows', 0):,}")
    st.metric("Cửa sổ LSTM", f"{metrics.get('window', '-')}")
    st.metric("Mô hình dự đoán", prediction_source)
    if metrics.get("lstm_error"):
        st.warning("TensorFlow đang bị Windows chặn.")

overview_tab, compare_tab, pca_tab = st.tabs(
    ["Tổng quan", "So sánh mô hình", "Phân tích PCA"]
)

with overview_tab:
    date_min = pred["date"].min().date()
    date_max = pred["date"].max().date()
    selected_date = st.date_input(
        "Thời điểm kiểm tra",
        value=date_max,
        min_value=date_min,
        max_value=date_max,
    )
    same_day = pred[pred["date"].dt.date == selected_date]

    if same_day.empty:
        nearest_index = pred["date"].dt.date.map(
            lambda value: abs(value - selected_date)
        ).idxmin()
        selected_row = pred.loc[[nearest_index]]
        st.caption(f"Hiển thị phiên gần {selected_date:%d/%m/%Y} nhất.")
    else:
        selected_row = same_day.tail(1)

    row = selected_row.iloc[0]
    error = float(row["predicted"] - row["actual"])
    error_pct = abs(error) / max(abs(float(row["actual"])), 1e-8) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Giá thực", f"{row['actual']:,.2f}")
    m2.metric("Giá dự đoán", f"{row['predicted']:,.2f}")
    m3.metric("Chênh lệch", f"{error:+,.2f}")
    m4.metric("Sai số tương đối", f"{error_pct:.3f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred["date"],
        y=pred["actual"],
        name="Giá thực",
        line={"color": "#94a3b8", "width": 1.8},
    ))
    fig.add_trace(go.Scatter(
        x=pred["date"],
        y=pred["predicted"],
        name=prediction_source,
        line={"color": "#facc15", "width": 2.2},
    ))
    fig.add_trace(go.Scatter(
        x=[row["date"]],
        y=[row["predicted"]],
        name="Thời điểm chọn",
        mode="markers",
        marker={"color": "#fb7185", "size": 12, "line": {"color": "#fff", "width": 1}},
    ))
    fig.update_layout(
        template="plotly_dark",
        height=470,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02},
        xaxis={"title": "Thời gian", "rangeslider": {"visible": False}},
        yaxis={"title": "Giá đóng cửa"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Tải kết quả thời điểm đã chọn",
        data=selected_row.to_csv(index=False).encode("utf-8"),
        file_name=f"gold_prediction_{row['date']:%Y%m%d_%H%M}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with compare_tab:
    ignored_keys = {
        "rows", "split", "window", "pca_components", "pca_report",
        "prediction_source", "lstm_error", "history",
    }
    model_metrics = pd.DataFrame.from_dict(
        {
            name: values
            for name, values in metrics.items()
            if name not in ignored_keys
            and (
                name.startswith("Linear Regression")
                or name == "LSTM"
            )
        },
        orient="index",
    )
    metric_columns = [
        column
        for column in ["MAE", "RMSE", "R2", "MAPE"]
        if column in model_metrics.columns
    ]
    if not model_metrics.empty and metric_columns:
        comparison = model_metrics.loc[
            model_metrics[metric_columns].notna().any(axis=1),
            metric_columns,
        ]
        st.markdown("#### Hiệu suất mô hình")
        st.caption("MAE, RMSE và MAPE càng thấp càng tốt; R2 càng gần 1 càng tốt.")
        st.dataframe(
            comparison.sort_values("RMSE").style.format("{:.6f}", na_rep="-"),
            use_container_width=True,
        )

        chart = go.Figure()
        chart.add_trace(go.Bar(
            x=comparison.index,
            y=comparison["RMSE"],
            marker={"color": ["#facc15" if "PCA=12" in name else "#38bdf8" for name in comparison.index]},
            text=comparison["RMSE"].round(4),
            textposition="outside",
        ))
        chart.update_layout(
            template="plotly_dark",
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            xaxis={"title": ""},
            yaxis={"title": "RMSE"},
        )
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("Chưa có metric mô hình.")

with pca_tab:
    st.markdown("#### Giảm chiều đặc trưng")
    st.caption("PCA biến 22 đặc trưng kỹ thuật thành 12, 6, 3 hoặc 1 thành phần.")
    pca_report = metrics.get("pca_report", {})
    if pca_report:
        pca_order = [
            "22 đặc trưng gốc",
            "PCA 12",
            "PCA 6",
            "PCA 3",
            "PCA 1",
        ]
        pca_order = [label for label in pca_order if label in pca_report]
        pca_table = pd.DataFrame(pca_report).T
        st.dataframe(
            pca_table.style.format({"Tỷ lệ phương sai PCA": "{:.2f}%"}),
            use_container_width=True,
        )

        dimensions = [pca_report[label]["Số đặc trưng"] for label in pca_order]
        variance = [pca_report[label]["Tỷ lệ phương sai PCA"] for label in pca_order]
        pca_fig = go.Figure()
        pca_fig.add_trace(go.Scatter(
            x=dimensions,
            y=variance,
            mode="lines+markers+text",
            text=pca_order,
            textposition="top center",
            line={"color": "#facc15", "width": 3},
            marker={"color": "#38bdf8", "size": 10},
        ))
        pca_fig.update_layout(
            template="plotly_dark",
            height=410,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
            xaxis={"title": "Số chiều", "autorange": "reversed"},
            yaxis={"title": "Phương sai giữ lại (%)", "range": [0, 105]},
        )
        st.plotly_chart(pca_fig, use_container_width=True)
    else:
        st.info("Chưa có báo cáo PCA. Hãy chạy lại training.")

with st.expander("Xem 100 kết quả gần nhất"):
    st.dataframe(
        pred.tail(100).sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
