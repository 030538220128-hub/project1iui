from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Thị trường & Mô hình — Nhà Tốt AI",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .page-banner {
        background: linear-gradient(135deg, #1B365D 0%, #1E293B 70%, #F37021 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .page-banner h1 {
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        color: #FFFFFF;
    }
    .page-banner p {
        margin: 0;
        opacity: 0.92;
        font-size: 0.95rem;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #F37021;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "house_prices_clean.csv"


@st.cache_data
def load_clean_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


try:
    df = load_clean_data(DATA_PATH)
except Exception as e:
    st.error(f"Không thể đọc dữ liệu: {e}")
    st.stop()


with st.sidebar:
    st.markdown("### 🎛️ Bộ lọc thị trường")
    st.caption("Tùy chỉnh phạm vi thống kê EDA và phân tích thị trường")
    st.divider()
    filter_quan = st.selectbox("Lọc theo Quận", ["Tất cả"] + sorted(df["quan"].dropna().unique().tolist()))
    filter_loai = st.selectbox("Lọc theo Loại hình", ["Tất cả"] + sorted(df["loai_hinh"].dropna().unique().tolist()))
    st.divider()
    st.markdown(
        """
        **Nội dung chính:**
        - **1. Thị trường 3 quận:** KPIs, đơn giá trung vị theo Quận/Phường & Loại hình.
        - **2. Benchmark 8 Models:** So sánh 4 mô hình Scikit-Learn & 4 mô hình PySpark ML.
        - **3. Thống kê Anomaly:** Tỷ lệ phân bố Quá đắt / Quá rẻ / Cấu trúc.
        """
    )

st.markdown(
    """
    <div class="page-banner">
        <h1>📊 Tổng quan Thị trường 3 Quận &amp; Đánh giá 8 Mô hình ML</h1>
        <p>Phân tích 7,939 tin đăng bất động sản sạch tại Bình Thạnh, Gò Vấp, Phú Nhuận và đối chiếu hiệu năng giữa Traditional ML &amp; PySpark ML.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Áp dụng bộ lọc từ sidebar
df_view = df.copy()
if filter_quan != "Tất cả":
    df_view = df_view[df_view["quan"].eq(filter_quan)]
if filter_loai != "Tất cả":
    df_view = df_view[df_view["loai_hinh"].eq(filter_loai)]

# =========================================================
# 1. KPI THỊ TRƯỜNG 3 QUẬN
# =========================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng số tin sạch", f"{len(df_view):,}")
c2.metric("Giá bán trung vị", f"{df_view['gia_ty'].median():.2f} tỷ VNĐ" if not df_view.empty else "N/A")
c3.metric("Đơn giá trung vị", f"{df_view['gia_m2_calc'].median():.1f} triệu/m²" if not df_view.empty else "N/A")
c4.metric("Số tin bất thường (Top 5%)", f"{int(df_view['is_anomaly'].sum()):,}" if not df_view.empty else "0")


st.write("")
with st.container(border=True):
    st.subheader("1. Phân tích mặt bằng giá theo Quận & Loại hình bất động sản")

    district_stats = (
        df.groupby("quan", as_index=False)
        .agg(
            So_tin=("row_id", "count"),
            Gia_trung_vi_ty=("gia_ty", "median"),
            Gia_trung_binh_ty=("gia_ty", "mean"),
            Don_gia_trung_vi_tr_m2=("gia_m2_calc", "median"),
            Dien_tich_trung_vi_m2=("dien_tich_m2", "median"),
            Ty_le_anomaly_pct=("is_anomaly", lambda s: round(s.mean() * 100, 2)),
        )
        .round(2)
    )

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.write("**Bảng thống kê tổng hợp theo từng Quận:**")
        st.dataframe(district_stats, width="stretch", hide_index=True)
    with col_right:
        st.write("**Đơn giá trung vị (triệu VNĐ/m²) theo Quận:**")
        st.bar_chart(
            district_stats.set_index("quan")[["Don_gia_trung_vi_tr_m2"]],
            horizontal=True,
        )

    type_stats = (
        df.groupby("loai_hinh", as_index=False)
        .agg(
            So_tin=("row_id", "count"),
            Gia_trung_vi_ty=("gia_ty", "median"),
            Don_gia_trung_vi_tr_m2=("gia_m2_calc", "median"),
        )
        .sort_values("Don_gia_trung_vi_tr_m2", ascending=False)
        .round(2)
    )
    t_left, t_right = st.columns([2, 3])
    with t_left:
        st.write("**Thống kê theo Loại hình nhà ở:**")
        st.dataframe(type_stats, width="stretch", hide_index=True)
    with t_right:
        st.write("**Đơn giá trung vị (triệu VNĐ/m²) theo Loại hình:**")
        st.bar_chart(
            type_stats.set_index("loai_hinh")[["Don_gia_trung_vi_tr_m2"]],
            horizontal=True,
        )


# =========================================================
# 2. KẾT QUẢ SO SÁNH 8 MÔ HÌNH (TRADITIONAL ML & PYSPARK ML)
# =========================================================

with st.container(border=True):
    st.subheader("2. Kết quả đánh giá Bài toán 1: Dự đoán giá nhà (8 Models — Scikit-Learn vs PySpark ML)")

    model_summary_df = pd.DataFrame(
        [
            {"Environment": "Traditional ML", "Model": "XGBoost (CV Winner - Deployed)", "Target": "log1p", "CV_RMSE (tỷ)": 7.135, "Test_MAE (tỷ)": 1.316, "Test_RMSE (tỷ)": 3.055, "Test_R²": 0.760, "MAPE (%)": 14.92},
            {"Environment": "Traditional ML", "Model": "LightGBM (Best Test Observed)", "Target": "log1p", "CV_RMSE (tỷ)": 7.184, "Test_MAE (tỷ)": 1.322, "Test_RMSE (tỷ)": 2.617, "Test_R²": 0.824, "MAPE (%)": 15.24},
            {"Environment": "Traditional ML", "Model": "Random Forest", "Target": "log1p", "CV_RMSE (tỷ)": 7.394, "Test_MAE (tỷ)": 1.281, "Test_RMSE (tỷ)": 2.870, "Test_R²": 0.788, "MAPE (%)": 14.49},
            {"Environment": "Traditional ML", "Model": "Linear Regression", "Target": "raw", "CV_RMSE (tỷ)": 31.891, "Test_MAE (tỷ)": 2.382, "Test_RMSE (tỷ)": 4.160, "Test_R²": 0.554, "MAPE (%)": 32.32},
            {"Environment": "Traditional ML", "Model": "Dummy Median (Baseline)", "Target": "raw", "CV_RMSE (tỷ)": None, "Test_MAE (tỷ)": 3.300, "Test_RMSE (tỷ)": 6.365, "Test_R²": -0.044, "MAPE (%)": 43.46},
            {"Environment": "PySpark ML", "Model": "PySpark Random Forest", "Target": "log1p", "CV_RMSE (tỷ)": 7.798, "Test_MAE (tỷ)": 1.366, "Test_RMSE (tỷ)": 3.262, "Test_R²": 0.726, "MAPE (%)": 15.44},
            {"Environment": "PySpark ML", "Model": "PySpark GBT", "Target": "log1p", "CV_RMSE (tỷ)": 9.190, "Test_MAE (tỷ)": 1.516, "Test_RMSE (tỷ)": 3.297, "Test_R²": 0.720, "MAPE (%)": 17.43},
            {"Environment": "PySpark ML", "Model": "PySpark Decision Tree", "Target": "log1p", "CV_RMSE (tỷ)": 9.951, "Test_MAE (tỷ)": 1.649, "Test_RMSE (tỷ)": 4.011, "Test_R²": 0.586, "MAPE (%)": 18.62},
            {"Environment": "PySpark ML", "Model": "PySpark Linear Regression", "Target": "raw", "CV_RMSE (tỷ)": 7.525, "Test_MAE (tỷ)": 2.386, "Test_RMSE (tỷ)": 4.180, "Test_R²": 0.550, "MAPE (%)": 32.31},
        ]
    )

    st.dataframe(model_summary_df, width="stretch", hide_index=True)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.write("**So sánh R² trên tập Test giữa các mô hình:**")
        chart_r2 = model_summary_df[model_summary_df["Test_R²"] > 0].set_index("Model")[["Test_R²"]]
        st.bar_chart(chart_r2, horizontal=True)

    with m_col2:
        st.write("**Top 10 Đặc trưng quan trọng nhất (XGBoost Feature Importance):**")
        fi_df = pd.DataFrame(
            [
                {"Feature": "loai_hinh = Nhà ngõ, hẻm", "Importance": 0.1230},
                {"Feature": "dien_tich_m2", "Importance": 0.1095},
                {"Feature": "so_phong_ngu", "Importance": 0.0903},
                {"Feature": "quan = Phú Nhuận", "Importance": 0.0619},
                {"Feature": "loai_hinh = Nhà mặt phố, mặt tiền", "Importance": 0.0526},
                {"Feature": "quan = Gò Vấp", "Importance": 0.0416},
                {"Feature": "chieu_dai_m", "Importance": 0.0276},
                {"Feature": "giay_to = Sổ chung / vi bằng", "Importance": 0.0250},
                {"Feature": "phuong = Phường 22", "Importance": 0.0217},
                {"Feature": "so_phong_ve_sinh", "Importance": 0.0204},
            ]
        ).set_index("Feature")
        st.bar_chart(fi_df, horizontal=True)


# =========================================================
# 3. TỔNG HỢP PHÁT HIỆN BẤT THƯỜNG (BÀI TOÁN 2)
# =========================================================

with st.container(border=True):
    st.subheader("3. Phân bố kết quả Phát hiện bất thường (Bài toán 2 — Ngưỡng Top 5% = 39.56 điểm)")

    anom_counts = (
        df["anomaly_type"]
        .value_counts()
        .rename_axis("Phân loại")
        .reset_index(name="Số lượng tin")
    )
    anom_counts["Tỷ lệ (%)"] = (anom_counts["Số lượng tin"] / len(df) * 100).round(2)

    a_col1, a_col2 = st.columns([2, 3])
    with a_col1:
        st.dataframe(anom_counts, width="stretch", hide_index=True)
        st.caption("Tỷ lệ đồng thuận đa tín hiệu (≥ 2 tín hiệu cùng cảnh báo): **85.06% (296 / 348 tin)**.")
    with a_col2:
        only_flagged = anom_counts[anom_counts["Phân loại"] != "Bình thường"].set_index("Phân loại")[["Số lượng tin"]]
        st.bar_chart(only_flagged, horizontal=True)
