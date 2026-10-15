from pathlib import Path
import re

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
MODEL_PATH = BASE_DIR / "models" / "project1_price_model.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_artifact(model_path: Path):
    return joblib.load(model_path)


try:
    artifact = load_artifact(MODEL_PATH)
except Exception as e:
    st.error(f"Không thể load model: {e}")
    st.stop()


model = artifact["model"]
FEATURES = artifact["features"]
MARKET_FEATURES = artifact.get(
    "market_features",
    [
        "gia_khuvuc_hientai",
        "gia_khuvuc_mean",
        "gia_khuvuc_min",
        "gia_khuvuc_max",
        "gia_khuvuc_std",
        "gia_khuvuc_trend_pct",
    ],
)
numeric_defaults = artifact["numeric_defaults"]
category_options = artifact["category_options"]
phuong_by_quan = artifact.get("phuong_by_quan", {})
market_group_defaults = artifact.get("market_group_defaults", pd.DataFrame())


# =========================================================
# HELPERS
# =========================================================

def default_num(name: str, fallback: float = 0.0) -> float:
    value = numeric_defaults.get(name, fallback)
    if value is None or pd.isna(value):
        return float(fallback)
    return float(value)


def options(name: str) -> list[str]:
    values = [
        str(v)
        for v in category_options.get(name, [])
        if pd.notna(v) and str(v) != "Không rõ"
    ]
    return ["Không rõ"] + values


def phuong_options(selected_quan: str) -> list[str]:
    if selected_quan and selected_quan != "Không rõ" and selected_quan in phuong_by_quan:
        return ["Không rõ"] + list(phuong_by_quan[selected_quan])
    return options("phuong")


def nullable_category(value: str):
    if value == "Không rõ" or pd.isna(value):
        return np.nan
    return value


def get_market_defaults(quan_val, loai_hinh_val) -> dict[str, float]:
    result = {}
    matched = pd.DataFrame()
    if not market_group_defaults.empty:
        mask = (
            market_group_defaults["quan"].astype(str).eq(str(quan_val))
            & market_group_defaults["loai_hinh"].astype(str).eq(str(loai_hinh_val))
        )
        matched = market_group_defaults.loc[mask]

    for feat in MARKET_FEATURES:
        val = np.nan
        if not matched.empty:
            val = matched.iloc[0].get(feat, np.nan)
        if pd.isna(val):
            val = default_num(feat, 0.0)
        result[feat] = float(val)
    return result


def parse_first_float(val, default=np.nan):
    if pd.isna(val):
        return default
    if isinstance(val, (int, float, np.number)):
        return float(val)
    text = str(val).strip().replace(",", ".")
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group(0)) if m else default


def prepare_batch_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    col_aliases = {
        "dien_tich_m2": ["dien_tich_m2", "dien_tich_num", "dien_tich"],
        "dien_tich_su_dung_m2": ["dien_tich_su_dung_m2", "dien_tich_su_dung_num", "dien_tich_su_dung"],
        "so_phong_ngu": ["so_phong_ngu", "so_phong_ngu_num"],
        "so_phong_ve_sinh": ["so_phong_ve_sinh", "so_phong_ve_sinh_num"],
        "tong_so_tang": ["tong_so_tang", "tong_so_tang_num"],
        "chieu_ngang_m": ["chieu_ngang_m", "chieu_ngang_num", "chieu_ngang"],
        "chieu_dai_m": ["chieu_dai_m", "chieu_dai_num", "chieu_dai"],
    }
    for target_col, candidates in col_aliases.items():
        if target_col not in df.columns:
            found = next((c for c in candidates if c in df.columns), None)
            if found is not None:
                df[target_col] = df[found].apply(parse_first_float)
            else:
                df[target_col] = default_num(target_col, 0.0)
        else:
            df[target_col] = df[target_col].apply(parse_first_float)

    for cat_col in ["quan", "phuong", "loai_hinh", "giay_to_phap_ly", "tinh_trang_noi_that", "dac_diem"]:
        if cat_col not in df.columns:
            df[cat_col] = np.nan

    for feat in MARKET_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan

    for idx, row in df.iterrows():
        mkt = get_market_defaults(row.get("quan"), row.get("loai_hinh"))
        for feat, val in mkt.items():
            if pd.isna(df.at[idx, feat]):
                df.at[idx, feat] = val

    return df[FEATURES].copy()


# =========================================================
# HEADER
# =========================================================

st.title("💰 Dự đoán giá nhà")

st.write(
    """
    Ước lượng **giá bán bất động sản (tỷ VNĐ)** và **đơn giá (triệu VNĐ/m²)**
    dựa trên bộ **19 đặc trưng** (kích thước, kết cấu, vị trí, pháp lý và mặt bằng giá khu vực).
    """
)

st.caption(
    f"Model: {artifact.get('model_name', 'XGBoost')} "
    f"• Target mode: {artifact.get('target_mode', 'log1p')} "
    f"• Số đặc trưng: {len(FEATURES)}"
)

st.divider()

mode = st.radio(
    "Chế độ dự đoán",
    [
        "✍️ Nhập thông tin 1 căn nhà",
        "📤 Upload CSV không có giá → Dự đoán hàng loạt",
    ],
    horizontal=True,
)


# =========================================================
# MODE 1: SINGLE HOUSE PREDICTION
# =========================================================

if mode == "✍️ Nhập thông tin 1 căn nhà":
    st.subheader("📍 Vị trí và loại hình")

    loc_c1, loc_c2, loc_c3 = st.columns(3)
    with loc_c1:
        quan_opts = options("quan")
        default_quan_idx = 1 if len(quan_opts) > 1 else 0
        quan = st.selectbox("Quận", quan_opts, index=default_quan_idx)
    with loc_c2:
        phuong_opts = phuong_options(quan)
        phuong = st.selectbox("Phường", phuong_opts)
    with loc_c3:
        loai_opts = options("loai_hinh")
        default_loai_idx = 1 if len(loai_opts) > 1 else 0
        loai_hinh = st.selectbox("Loại hình", loai_opts, index=default_loai_idx)

    with st.form("price_prediction_form"):
        st.subheader("📐 Diện tích và kết cấu")

        col1, col2, col3 = st.columns(3)

        with col1:
            dien_tich = st.number_input(
                "Diện tích đất / công nhận (m²)",
                min_value=1.0,
                value=max(1.0, default_num("dien_tich_m2", 52.0)),
                step=1.0,
            )
            dien_tich_su_dung = st.number_input(
                "Diện tích sử dụng (m²)",
                min_value=0.0,
                value=max(0.0, default_num("dien_tich_su_dung_m2", 120.0)),
                step=1.0,
            )
            so_tang = st.number_input(
                "Tổng số tầng",
                min_value=1.0,
                value=max(1.0, default_num("tong_so_tang", 3.0)),
                step=1.0,
            )

        with col2:
            phong_ngu = st.number_input(
                "Số phòng ngủ",
                min_value=0.0,
                value=max(0.0, default_num("so_phong_ngu", 3.0)),
                step=1.0,
            )
            phong_wc = st.number_input(
                "Số phòng vệ sinh",
                min_value=0.0,
                value=max(0.0, default_num("so_phong_ve_sinh", 3.0)),
                step=1.0,
            )

        with col3:
            chieu_ngang = st.number_input(
                "Chiều ngang (m)",
                min_value=0.0,
                value=max(0.0, default_num("chieu_ngang_m", 4.0)),
                step=0.1,
            )
            chieu_dai = st.number_input(
                "Chiều dài (m)",
                min_value=0.0,
                value=max(0.0, default_num("chieu_dai_m", 13.0)),
                step=0.1,
            )

        st.subheader("🏠 Pháp lý và đặc điểm")

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            giay_to = st.selectbox("Giấy tờ pháp lý", options("giay_to_phap_ly"))
        with f_col2:
            noi_that = st.selectbox("Tình trạng nội thất", options("tinh_trang_noi_that"))
        with f_col3:
            dac_diem = st.selectbox("Đặc điểm", options("dac_diem"))

        submitted = st.form_submit_button(
            "💰 Dự đoán giá",
            type="primary",
            width="stretch",
        )

    if submitted:
        market_vals = get_market_defaults(
            nullable_category(quan),
            nullable_category(loai_hinh),
        )

        input_data = {
            "dien_tich_m2": float(dien_tich),
            "dien_tich_su_dung_m2": float(dien_tich_su_dung),
            "so_phong_ngu": float(phong_ngu),
            "so_phong_ve_sinh": float(phong_wc),
            "tong_so_tang": float(so_tang),
            "chieu_ngang_m": float(chieu_ngang),
            "chieu_dai_m": float(chieu_dai),
            **market_vals,
            "quan": nullable_category(quan),
            "phuong": nullable_category(phuong),
            "loai_hinh": nullable_category(loai_hinh),
            "giay_to_phap_ly": nullable_category(giay_to),
            "tinh_trang_noi_that": nullable_category(noi_that),
            "dac_diem": nullable_category(dac_diem),
        }

        input_df = pd.DataFrame([input_data], columns=FEATURES)

        try:
            prediction = float(model.predict(input_df)[0])
            prediction = max(0.0, prediction)
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")
            st.stop()

        st.divider()
        st.subheader("Kết quả dự đoán")

        safe_area = max(float(dien_tich), 0.001)
        price_m2 = prediction * 1000.0 / safe_area

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Giá dự đoán", f"{prediction:,.2f} tỷ VNĐ")
        with c2:
            st.metric("Giá / m²", f"{price_m2:,.1f} triệu")
        with c3:
            st.metric("Diện tích", f"{dien_tich:,.1f} m²")
        with c4:
            st.metric(
                "Biên độ lịch sử khu vực",
                f"{market_vals['gia_khuvuc_min']:.0f} – {market_vals['gia_khuvuc_max']:.0f} tr/m²",
            )

        if prediction < 1:
            st.info("Model dự đoán mức giá dưới 1 tỷ VNĐ.")
        elif prediction < 5:
            st.success("Phân khúc giá thấp – trung bình (< 5 tỷ VNĐ).")
        elif prediction < 15:
            st.info("Phân khúc giá trung bình – khá (5 – 15 tỷ VNĐ).")
        else:
            st.warning("Đây là bất động sản thuộc phân khúc giá cao (>= 15 tỷ VNĐ).")

        with st.expander("Xem dữ liệu 19 đặc trưng đưa vào model"):
            st.dataframe(input_df, width="stretch", hide_index=True)


# =========================================================
# MODE 2: BATCH CSV PREDICTION
# =========================================================

else:
    st.subheader("📤 Upload CSV không có giá → Dự đoán giá hàng loạt")
    st.write(
        """
        Upload file CSV chứa danh sách bất động sản cần định giá.
        Hệ thống tự động chuẩn hóa cả định dạng CSV gốc (`dien_tich`, `so_phong_ngu`, ...)
        lẫn định dạng đã làm sạch (`dien_tich_m2`, ...).
        """
    )

    csv_file = st.file_uploader(
        "Chọn file CSV cần dự đoán giá",
        type=["csv"],
        key="batch_price_csv",
    )

    if csv_file is not None:
        try:
            raw_df = pd.read_csv(csv_file)
            st.write(f"Đã tải lên **{len(raw_df):,}** dòng | **{len(raw_df.columns)}** cột.")
            st.dataframe(raw_df.head(10), width="stretch")

            price_cols = [c for c in ["gia_ban", "gia_ty", "gia_m2", "don_gia"] if c in raw_df.columns]
            if price_cols:
                st.info(
                    "Lưu ý: File đang có sẵn cột giá ("
                    + ", ".join(price_cols)
                    + "). Hệ thống sẽ bỏ qua các cột này khi dự đoán để chống rò rỉ dữ liệu (data leakage)."
                )

            if st.button("💰 Dự đoán giá cho toàn bộ file CSV", type="primary", width="stretch"):
                batch_X = prepare_batch_dataframe(raw_df)
                preds = np.maximum(0.0, model.predict(batch_X))

                out_df = raw_df.copy()
                out_df["gia_du_doan_ty"] = np.round(preds, 3)
                safe_areas = np.maximum(batch_X["dien_tich_m2"].fillna(50.0).to_numpy(dtype=float), 0.001)
                out_df["don_gia_du_doan_trieu_m2"] = np.round(preds * 1000.0 / safe_areas, 2)

                m1, m2, m3 = st.columns(3)
                m1.metric("Số căn đã dự đoán", f"{len(out_df):,}")
                m2.metric("Giá dự đoán trung vị", f"{float(np.median(preds)):.2f} tỷ VNĐ")
                m3.metric("Đơn giá dự đoán trung vị", f"{float(np.median(out_df['don_gia_du_doan_trieu_m2'])):.1f} tr/m²")

                st.dataframe(out_df.head(50), width="stretch")

                st.download_button(
                    "📥 Tải kết quả dự đoán (.csv)",
                    data=out_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="ket_qua_du_doan_gia.csv",
                    mime="text/csv",
                    width="stretch",
                )
        except Exception as e:
            st.error(f"Không thể xử lý file CSV: {e}")


st.divider()
st.caption(
    """
    Model được huấn luyện từ 7,939 tin đăng sạch tại Bình Thạnh, Gò Vấp và Phú Nhuận.
    Kết quả là mức giá ước lượng tham khảo từ dữ liệu rao bán trên Nhà Tốt.
    """
)
