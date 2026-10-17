from pathlib import Path

import joblib
import streamlit as st


st.set_page_config(
    page_title="Nhà Tốt — House Price Intelligence",
    page_icon="🏠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
PRICE_MODEL_PATH = BASE_DIR / "models" / "project1_price_model.pkl"
ANOMALY_MODEL_PATH = BASE_DIR / "models" / "project1_anomaly_model.pkl"
BANNER_PATH = BASE_DIR / "nhatot.jpg"

st.markdown(
    """
    <style>
    .hero-box {
        background: linear-gradient(135deg, #1B365D 0%, #0F172A 65%, #F37021 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px rgba(27, 54, 93, 0.14);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(243, 112, 33, 0.25);
        border: 1px solid #F37021;
        color: #FFD8BE;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        color: #FFFFFF;
    }
    .hero-sub {
        font-size: 1rem;
        opacity: 0.92;
        margin: 0;
        line-height: 1.5;
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


@st.cache_resource
def load_price_artifact():
    return joblib.load(PRICE_MODEL_PATH)


@st.cache_resource
def load_anomaly_artifact():
    return joblib.load(ANOMALY_MODEL_PATH)


with st.sidebar:
    st.markdown("### 🏠 Nhà Tốt AI Intelligence")
    st.caption("Đồ án Tốt nghiệp Data Science (DL07) — Trung tâm Tin học ĐH KHTN TP.HCM")
    st.divider()
    st.markdown("#### 🧭 Điều hướng nhanh")
    st.page_link("app.py", label="Trang chủ", icon="🏠")
    st.page_link("pages/1_House_Price_Prediction.py", label="1. Dự đoán giá nhà", icon="💰")
    st.page_link("pages/2_Anomaly_Detection.py", label="2. Phát hiện bất thường", icon="⚠️")
    st.page_link("pages/3_Market_Overview.py", label="3. Thị trường & Mô hình", icon="📊")
    st.divider()
    st.markdown(
        """
        **Phạm vi dữ liệu:**
        - 📍 Quận Bình Thạnh
        - 📍 Quận Gò Vấp
        - 📍 Quận Phú Nhuận
        - 📊 **7,939** tin đăng đã kiểm toán
        - 🧩 **19** đặc trưng chuẩn hóa
        """
    )
    st.divider()
    st.markdown("#### 👥 Thành viên thực hiện")
    st.markdown(
        """
        - 👤 **Lê Văn Toàn**  
          📧 `toanlevan2808@gmail.com`
        - 👤 **Chung Tiến Quang**  
          📧 `Dongtachungtienquang@gmail.com`
        """
    )


st.markdown(
    """
    <div class="hero-box">
        <div class="hero-badge">ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE · PROJECT 1</div>
        <div class="hero-title">🏠 Nhà Tốt — House Price & Anomaly Intelligence</div>
        <p class="hero-sub">
            Hệ thống định giá bất động sản tự động và phát hiện tin đăng bất thường đa tín hiệu
            tại <b>Bình Thạnh, Gò Vấp và Phú Nhuận</b> dựa trên Machine Learning &amp; PySpark.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    price_artifact = load_price_artifact()
    anomaly_artifact = load_anomaly_artifact()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Price Prediction Model",
            price_artifact.get("model_name", "XGBoost"),
        )
    with k2:
        st.metric(
            "Price Training Rows",
            f"{price_artifact.get('training_rows', 7939):,}",
        )
    with k3:
        st.metric(
            "Số đặc trưng (Features)",
            f"{len(price_artifact.get('features', []))} biến",
        )
    with k4:
        st.metric(
            "Anomaly Top-k",
            f"{anomaly_artifact.get('top_k', 0.05) * 100:.0f}%",
        )
except Exception as e:
    st.warning(f"Chưa load được model: {e}")

st.write("")
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("💰 1. Dự đoán giá nhà")
        st.write(
            """
            Ước lượng **giá bán (tỷ VNĐ)** và **đơn giá (triệu/m²)** cho 1 căn nhà
            hoặc **upload file CSV không có giá** để định giá hàng loạt.
            Hỗ trợ lọc động Phường theo Quận và đối chiếu biên độ giá khu vực.
            """
        )
        st.page_link(
            "pages/1_House_Price_Prediction.py",
            label="Mở House Price Prediction →",
            icon="💰",
        )

with col2:
    with st.container(border=True):
        st.subheader("⚠️ 2. Phát hiện bất thường")
        st.write(
            """
            Chấm điểm **Composite Score (0–100)** từ 4 tín hiệu (`Residual-Z`,
            `P10–P90`, `Historical Min/Max`, `Isolation Forest`) kèm giải thích XAI.
            Hỗ trợ duyệt tin có sẵn, kiểm tra tin mới và **upload CSV có giá**.
            """
        )
        st.page_link(
            "pages/2_Anomaly_Detection.py",
            label="Mở Anomaly Detection →",
            icon="⚠️",
        )

with col3:
    with st.container(border=True):
        st.subheader("📊 3. Thị trường & Mô hình")
        st.write(
            """
            Dashboard thống kê mặt bằng giá 3 quận, bảng đối chiếu thực nghiệm
            **8 mô hình (Scikit-Learn vs PySpark ML)**, biểu đồ Feature Importance
            và phân bố các nhóm tin bất thường.
            """
        )
        st.page_link(
            "pages/3_Market_Overview.py",
            label="Mở Market & Model Overview →",
            icon="📊",
        )

if BANNER_PATH.exists():
    st.write("")
    with st.container(border=True):
        _, img_col, _ = st.columns([1, 6, 1])
        with img_col:
            st.image(
                str(BANNER_PATH),
                caption="Mô hình nghiệp vụ Dự đoán giá & Phát hiện bất thường trên Nhà Tốt (nhatot.com)",
                width="stretch",
            )