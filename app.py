from pathlib import Path

import joblib
import streamlit as st


st.set_page_config(
    page_title="House Price Intelligence",
    page_icon="🏠",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent

PRICE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "project1_price_model.pkl"
)

ANOMALY_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "project1_anomaly_model.pkl"
)


@st.cache_resource
def load_price_artifact():
    return joblib.load(PRICE_MODEL_PATH)


@st.cache_resource
def load_anomaly_artifact():
    return joblib.load(ANOMALY_MODEL_PATH)


st.title("🏠 House Price Intelligence")

st.write(
    """
    Hệ thống phân tích giá nhà tại **Bình Thạnh,
    Gò Vấp và Phú Nhuận**.

    Project gồm hai bài toán Machine Learning:
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 House Price Prediction")

    st.write(
        """
        Dự đoán giá bán nhà (1 căn hoặc upload CSV hàng loạt) dựa trên 19 đặc trưng:
        vị trí, diện tích, kết cấu, pháp lý và mặt bằng giá khu vực.
        """
    )

    st.page_link(
        "pages/1_House_Price_Prediction.py",
        label="Mở Price Prediction →",
        icon="💰",
    )

with col2:
    st.subheader("⚠️ Anomaly Detection")

    st.write(
        """
        Phát hiện các tin đăng có dấu hiệu quá rẻ, quá đắt
        hoặc có tổ hợp đặc trưng bất thường bằng Composite Score 4 tín hiệu.
        """
    )

    st.page_link(
        "pages/2_Anomaly_Detection.py",
        label="Mở Anomaly Detection →",
        icon="⚠️",
    )

with col3:
    st.subheader("📊 Market & Model Overview")

    st.write(
        """
        Xem thống kê thị trường 3 quận, bảng đối chiếu 8 mô hình
        (Traditional ML vs PySpark ML) và Feature Importance.
        """
    )

    st.page_link(
        "pages/3_Market_Overview.py",
        label="Mở Market & Model Overview →",
        icon="📊",
    )

st.divider()

try:
    price_artifact = load_price_artifact()
    anomaly_artifact = load_anomaly_artifact()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Price Prediction Model",
            price_artifact.get(
                "model_name",
                "N/A",
            ),
        )

    with c2:
        st.metric(
            "Price Training Rows",
            f"{price_artifact.get('training_rows', 0):,}",
        )

    with c3:
        st.metric(
            "Anomaly Top-k",
            (
                f"{anomaly_artifact.get('top_k', 0) * 100:.0f}%"
            ),
        )

except Exception as e:
    st.warning(
        f"Chưa load được model: {e}"
    )