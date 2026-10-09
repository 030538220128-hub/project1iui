from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="💰",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "project1_price_model.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_artifact(model_path):
    # Include the model path in the cache key to isolate page artifacts.
    return joblib.load(model_path)


try:
    artifact = load_artifact(MODEL_PATH)

except Exception as e:
    st.error(
        f"Không thể load model: {e}"
    )
    st.stop()


model = artifact["model"]

FEATURES = artifact["features"]
NUMERIC_FEATURES = artifact["numeric_features"]
CATEGORICAL_FEATURES = artifact["categorical_features"]

numeric_defaults = artifact["numeric_defaults"]
category_options = artifact["category_options"]


# =========================================================
# HELPERS
# =========================================================

def default_num(name, fallback=0.0):
    value = numeric_defaults.get(
        name,
        fallback,
    )

    if value is None or pd.isna(value):
        return float(fallback)

    return float(value)


def options(name):
    values = category_options.get(
        name,
        [],
    )

    values = [
        str(v)
        for v in values
        if pd.notna(v)
    ]

    return ["Không rõ"] + values


def nullable_category(value):
    if value == "Không rõ":
        return np.nan

    return value


# =========================================================
# HEADER
# =========================================================

st.title("💰 Dự đoán giá nhà")

st.write(
    """
    Nhập thông tin bất động sản để model
    ước lượng **giá bán theo đơn vị tỷ VNĐ**.
    """
)

st.caption(
    f"Model: {artifact.get('model_name', 'N/A')} "
    f"• Target mode: {artifact.get('target_mode', 'N/A')}"
)

st.divider()


# =========================================================
# FORM
# =========================================================

with st.form(
    "price_prediction_form"
):

    st.subheader(
        "📍 Vị trí và loại hình"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        quan = st.selectbox(
            "Quận",
            options("quan"),
        )

        phuong = st.selectbox(
            "Phường",
            options("phuong"),
        )

    with col2:

        duong = st.selectbox(
            "Đường",
            options("duong"),
        )

        loai_hinh = st.selectbox(
            "Loại hình",
            options("loai_hinh"),
        )

    with col3:

        giay_to = st.selectbox(
            "Giấy tờ pháp lý",
            options(
                "giay_to_phap_ly"
            ),
        )

        dac_diem = st.selectbox(
            "Đặc điểm",
            options("dac_diem"),
        )


    st.subheader(
        "📐 Diện tích và kết cấu"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        dien_tich = st.number_input(
            "Diện tích (m²)",
            min_value=1.0,
            value=max(
                1.0,
                default_num(
                    "dien_tich_num",
                    50,
                ),
            ),
            step=1.0,
        )

        dien_tich_dat = (
            st.number_input(
                "Diện tích đất (m²)",
                min_value=1.0,
                value=max(
                    1.0,
                    default_num(
                        "dien_tich_dat_num",
                        50,
                    ),
                ),
                step=1.0,
            )
        )

        dien_tich_su_dung = (
            st.number_input(
                "Diện tích sử dụng (m²)",
                min_value=0.0,
                value=max(
                    0.0,
                    default_num(
                        "dien_tich_su_dung_num",
                        50,
                    ),
                ),
                step=1.0,
            )
        )

    with col2:

        phong_ngu = st.number_input(
            "Số phòng ngủ",
            min_value=0.0,
            value=max(
                0.0,
                default_num(
                    "so_phong_ngu_num",
                    3,
                ),
            ),
            step=1.0,
        )

        phong_wc = st.number_input(
            "Số phòng vệ sinh",
            min_value=0.0,
            value=max(
                0.0,
                default_num(
                    "so_phong_ve_sinh_num",
                    2,
                ),
            ),
            step=1.0,
        )

        so_tang = st.number_input(
            "Tổng số tầng",
            min_value=0.0,
            value=max(
                0.0,
                default_num(
                    "tong_so_tang_num",
                    2,
                ),
            ),
            step=1.0,
        )

    with col3:

        chieu_ngang = st.number_input(
            "Chiều ngang (m)",
            min_value=0.0,
            value=max(
                0.0,
                default_num(
                    "chieu_ngang_num",
                    4,
                ),
            ),
            step=0.1,
        )

        chieu_dai = st.number_input(
            "Chiều dài (m)",
            min_value=0.0,
            value=max(
                0.0,
                default_num(
                    "chieu_dai_num",
                    12,
                ),
            ),
            step=0.1,
        )


    st.subheader(
        "🏠 Thông tin bổ sung"
    )

    col1, col2 = st.columns(2)

    with col1:

        noi_that = st.selectbox(
            "Tình trạng nội thất",
            options(
                "tinh_trang_noi_that"
            ),
        )

    with col2:

        huong = st.selectbox(
            "Hướng cửa chính",
            options(
                "huong_cua_chinh"
            ),
        )


    submitted = st.form_submit_button(
        "💰 Dự đoán giá",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# PREDICT
# =========================================================

if submitted:

    input_data = {
        "dien_tich_num":
            dien_tich,

        "dien_tich_dat_num":
            dien_tich_dat,

        "dien_tich_su_dung_num":
            dien_tich_su_dung,

        "so_phong_ngu_num":
            phong_ngu,

        "so_phong_ve_sinh_num":
            phong_wc,

        "tong_so_tang_num":
            so_tang,

        "chieu_ngang_num":
            chieu_ngang,

        "chieu_dai_num":
            chieu_dai,

        "quan":
            nullable_category(quan),

        "phuong":
            nullable_category(phuong),

        "duong":
            nullable_category(duong),

        "loai_hinh":
            nullable_category(loai_hinh),

        "giay_to_phap_ly":
            nullable_category(giay_to),

        "tinh_trang_noi_that":
            nullable_category(noi_that),

        "huong_cua_chinh":
            nullable_category(huong),

        "dac_diem":
            nullable_category(dac_diem),
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES,
    )

    try:

        prediction = float(
            model.predict(
                input_df
            )[0]
        )

        prediction = max(
            0.0,
            prediction,
        )

    except Exception as e:

        st.error(
            f"Lỗi khi dự đoán: {e}"
        )
        st.stop()


    st.divider()

    st.subheader(
        "Kết quả dự đoán"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Giá dự đoán",
            f"{prediction:,.2f} tỷ VNĐ",
        )

    with c2:

        price_m2 = (
            prediction
            * 1000
            / dien_tich
        )

        st.metric(
            "Giá / m²",
            f"{price_m2:,.1f} triệu",
        )

    with c3:

        st.metric(
            "Diện tích",
            f"{dien_tich:,.1f} m²",
        )


    if prediction < 1:

        st.info(
            "Model dự đoán mức giá dưới 1 tỷ VNĐ."
        )

    elif prediction < 5:

        st.success(
            "Phân khúc giá thấp – trung bình."
        )

    elif prediction < 15:

        st.info(
            "Phân khúc giá trung bình."
        )

    else:

        st.warning(
            "Đây là bất động sản thuộc "
            "phân khúc giá cao."
        )


    with st.expander(
        "Xem dữ liệu đưa vào model"
    ):

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True,
        )


st.divider()

st.caption(
    """
    Model được huấn luyện từ dữ liệu tại
    Bình Thạnh, Gò Vấp và Phú Nhuận.
    Kết quả là mức giá ước lượng từ dữ liệu
    đăng bán, không phải giá giao dịch chính thức.
    """
)
