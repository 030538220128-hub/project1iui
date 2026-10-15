from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Market & Model Overview",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "house_prices_clean.csv"


@st.cache_data
def load_clean_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


st.title("📊 Tổng quan Thị trường & Đánh giá Mô hình (Project 1)")
st.write(
    """
    Tổng hợp phân bố thị trường bất động sản tại **Bình Thạnh, Gò Vấp, Phú Nhuận**
    và kết quả thực nghiệm **8 mô hình Machine Learning** (4 mô hình Traditional ML & 4 mô hình PySpark ML).
    """
)

st.divider()

try:
    df = load_clean_data(DATA_PATH)
except Exception as e:
    st.error(f"Không thể đọc dữ liệu: {e}")
    st.stop()


# =========================================================
# 1. KPI THỊ TRƯỜNG 3 QUẬN
# =========================================================

st.subheader("1. Thống kê tổng quan 3 quận (7,939 tin đăng sạch)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng số tin sạch", f"{len(df):,}")
c2.metric("Giá bán trung vị", f"{df['gia_ty'].median():.2f} tỷ VNĐ")
c3.metric("Đơn giá trung vị", f"{df['gia_m2_calc'].median():.1f} triệu/m²")
c4.metric("Số tin bất thường (Top 5%)", f"{int(df['is_anomaly'].sum()):,}")

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
    st.write("**Bảng thống kê theo từng Quận:**")
    st.dataframe(district_stats, width="stretch", hide_index=True)
with col_right:
    st.write("**Đơn giá trung vị (triệu VNĐ/m²) theo Quận:**")
    st.bar_chart(
        district_stats.set_index("quan")[["Don_gia_trung_vi_tr_m2"]],
        horizontal=True,
    )

st.divider()


# =========================================================
# 2. KẾT QUẢ SO SÁNH 8 MÔ HÌNH (TRADITIONAL ML & PYSPARK ML)
# =========================================================

st.subheader("2. Kết quả đánh giá Bài toán 1: Dự đoán giá nhà (8 Models)")

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

st.divider()


# =========================================================
# 3. TỔNG HỢP PHÁT HIỆN BẤT THƯỜNG (BÀI TOÁN 2)
# =========================================================

st.subheader("3. Phân bố kết quả Phát hiện bất thường (Bài toán 2)")

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
with a_col2:
    only_flagged = anom_counts[anom_counts["Phân loại"] != "Bình thường"].set_index("Phân loại")[["Số lượng tin"]]
    st.bar_chart(only_flagged, horizontal=True)
