from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# CONFIG & STYLING
# =========================================================

st.set_page_config(
    page_title="Dự đoán giá nhà — Nhà Tốt AI",
    page_icon="💰",
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


def infer_quan_phuong_from_dia_chi(df: pd.DataFrame) -> pd.DataFrame:
    if "dia_chi" not in df.columns:
        return df
    addr = df["dia_chi"].fillna("").astype(str)
    if "quan" not in df.columns or df["quan"].isna().all():
        df["quan"] = np.select(
            [
                addr.str.contains("Bình Thạnh|Binh Thanh", case=False, regex=True),
                addr.str.contains("Gò Vấp|Go Vap", case=False, regex=True),
                addr.str.contains("Phú Nhuận|Phu Nhuan", case=False, regex=True),
            ],
            ["Binh Thanh", "Go Vap", "Phu Nhuan"],
            default=np.nan,
        )
    if "phuong" not in df.columns or df["phuong"].isna().all():
        df["phuong"] = addr.str.extract(r"(Phường\s+[^,()]+)", expand=False).str.strip()
    return df


def prepare_batch_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = infer_quan_phuong_from_dia_chi(raw_df.copy())
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

    pairs = df[["quan", "loai_hinh"]].drop_duplicates()
    for _, p_row in pairs.iterrows():
        q_val, l_val = p_row["quan"], p_row["loai_hinh"]
        mkt = get_market_defaults(q_val, l_val)
        mask = df["quan"].astype(str).eq(str(q_val)) & df["loai_hinh"].astype(str).eq(str(l_val))
        for feat, val in mkt.items():
            df.loc[mask & df[feat].isna(), feat] = val

    return df[FEATURES].copy()


# =========================================================
# SIDEBAR & HEADER
# =========================================================

with st.sidebar:
    st.markdown("### 💰 Module 1: Định giá BĐS")
    st.caption(
        f"Model: **{artifact.get('model_name', 'XGBoost')}** "
        f"(`{artifact.get('target_mode', 'log1p')}`)"
    )
    st.markdown(
        """
        **Bộ 19 đặc trưng đầu vào:**
        - **7 biến vật lý:** Diện tích đất, diện tích sử dụng, số tầng, phòng ngủ, WC, ngang, dài.
        - **6 biến phân loại:** Quận, Phường, Loại hình, Pháp lý, Nội thất, Đặc điểm.
        - **6 biến thị trường:** Tự động tra cứu mặt bằng giá theo `(Quận, Loại hình)`.
        """
    )

st.markdown(
    """
    <div class="page-banner">
        <h1>💰 Bài toán 1: Dự đoán giá bán bất động sản</h1>
        <p>Ước lượng giá trị hợp lý (tỷ VNĐ) và đơn giá (triệu VNĐ/m²) từ bộ 19 đặc trưng đã kiểm toán trên dữ liệu Nhà Tốt.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    with st.container(border=True):
        st.subheader("📍 Bước 1: Chọn vị trí & loại hình bất động sản")
        st.caption("Danh sách Phường và mặt bằng giá tham chiếu sẽ tự động cập nhật ngay khi bạn chọn Quận.")

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

        preview_mkt = get_market_defaults(nullable_category(quan), nullable_category(loai_hinh))
        st.info(
            f"📊 **Mặt bằng giá tham chiếu ({quan} — {loai_hinh}):** "
            f"Hiện tại **{preview_mkt['gia_khuvuc_hientai']:.1f} tr/m²** | "
            f"Trung bình **{preview_mkt['gia_khuvuc_mean']:.1f} tr/m²** | "
            f"Biên độ lịch sử **{preview_mkt['gia_khuvuc_min']:.1f} – {preview_mkt['gia_khuvuc_max']:.1f} tr/m²** "
            f"(Xu hướng: `{preview_mkt['gia_khuvuc_trend_pct']:+.1f}%`)"
        )

    with st.form("price_prediction_form", border=True):
        st.subheader("📐 Bước 2: Thông số diện tích, kết cấu & pháp lý")

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
            giay_to = st.selectbox("Giấy tờ pháp lý", options("giay_to_phap_ly"))

        with col3:
            chieu_ngang = st.number_input(
                "Chiều ngang mặt tiền (m)",
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
            noi_that = st.selectbox("Tình trạng nội thất", options("tinh_trang_noi_that"))

        dac_diem = st.selectbox("Đặc điểm vị trí / kết cấu (Hẻm xe hơi, Nở hậu, Mặt tiền...)", options("dac_diem"))

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

        safe_area = max(float(dien_tich), 0.001)
        price_m2 = prediction * 1000.0 / safe_area

        with st.container(border=True):
            st.subheader("🎯 Kết quả định giá từ mô hình AI")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Giá dự đoán", f"{prediction:,.2f} tỷ VNĐ")
            with c2:
                diff_m2 = price_m2 - market_vals["gia_khuvuc_hientai"]
                st.metric(
                    "Giá / m²",
                    f"{price_m2:,.1f} triệu",
                    delta=f"{diff_m2:+.1f} tr/m² so với TB khu vực",
                    delta_color="off",
                )
            with c3:
                st.metric("Diện tích", f"{dien_tich:,.1f} m²")
            with c4:
                st.metric(
                    "Biên độ lịch sử khu vực",
                    f"{market_vals['gia_khuvuc_min']:.0f} – {market_vals['gia_khuvuc_max']:.0f} tr/m²",
                )

            if prediction < 1:
                st.info("🏷️ **Phân khúc:** Mức giá rất thấp (< 1 tỷ VNĐ).")
            elif prediction < 5:
                st.success("🏷️ **Phân khúc:** Giá thấp – trung bình (1 – 5 tỷ VNĐ), thanh khoản cao.")
            elif prediction < 15:
                st.info("🏷️ **Phân khúc:** Giá trung bình – khá (5 – 15 tỷ VNĐ), phân khúc chủ đạo tại 3 quận.")
            else:
                st.warning("🏷️ **Phân khúc:** Bất động sản giá trị cao / mặt tiền / biệt thự (>= 15 tỷ VNĐ).")

            cmp_df = pd.DataFrame(
                {
                    "Mốc so sánh": [
                        "1. Giá sàn lịch sử (Min)",
                        "2. Trung bình khu vực (Mean)",
                        "3. Mặt bằng hiện tại",
                        "4. Căn nhà đang định giá",
                        "5. Giá trần lịch sử (Max)",
                    ],
                    "Đơn giá (triệu VNĐ/m²)": [
                        round(market_vals["gia_khuvuc_min"], 1),
                        round(market_vals["gia_khuvuc_mean"], 1),
                        round(market_vals["gia_khuvuc_hientai"], 1),
                        round(price_m2, 1),
                        round(market_vals["gia_khuvuc_max"], 1),
                    ],
                }
            ).set_index("Mốc so sánh")

            st.write("**Đối chiếu đơn giá dự đoán (triệu VNĐ/m²) với mặt bằng khu vực:**")
            st.bar_chart(cmp_df, horizontal=True)

            with st.expander("🔍 Xem chi tiết vector 19 đặc trưng đưa vào Pipeline"):
                st.dataframe(input_df, width="stretch", hide_index=True)


# =========================================================
# MODE 2: BATCH CSV PREDICTION
# =========================================================

else:
    with st.container(border=True):
        st.subheader("📤 Upload CSV không có giá → Dự đoán giá hàng loạt")
        st.write(
            """
            Tải lên file CSV chứa danh sách các căn nhà cần định giá (không cần cột `gia_ban` hay `gia_ty`).
            Hệ thống tự động nhận diện cả định dạng CSV thô từ Nhà Tốt (`dien_tich`, `dia_chi`, ...)
            lẫn định dạng đã làm sạch (`dien_tich_m2`, `quan`, `phuong`, ...).
            """
        )

        sample_csv_df = pd.DataFrame(
            [
                {
                    "tieu_de": "Nhà hẻm xe hơi Lê Quang Định, P7, Bình Thạnh",
                    "quan": "Binh Thanh",
                    "phuong": "Phường 7",
                    "loai_hinh": "Nhà ngõ, hẻm",
                    "dien_tich_m2": 48.0,
                    "dien_tich_su_dung_m2": 140.0,
                    "so_phong_ngu": 3,
                    "so_phong_ve_sinh": 3,
                    "tong_so_tang": 3,
                    "chieu_ngang_m": 4.0,
                    "chieu_dai_m": 12.0,
                    "giay_to_phap_ly": "Đã có sổ",
                    "tinh_trang_noi_that": "Nội thất đầy đủ",
                    "dac_diem": "Hẻm xe hơi",
                },
                {
                    "tieu_de": "Nhà mặt tiền Phan Xích Long, Phú Nhuận",
                    "quan": "Phu Nhuan",
                    "phuong": "Phường 2",
                    "loai_hinh": "Nhà mặt phố, mặt tiền",
                    "dien_tich_m2": 64.0,
                    "dien_tich_su_dung_m2": 240.0,
                    "so_phong_ngu": 5,
                    "so_phong_ve_sinh": 5,
                    "tong_so_tang": 4,
                    "chieu_ngang_m": 4.0,
                    "chieu_dai_m": 16.0,
                    "giay_to_phap_ly": "Đã có sổ",
                    "tinh_trang_noi_that": "Nội thất đầy đủ",
                    "dac_diem": "Mặt tiền",
                },
                {
                    "tieu_de": "Nhà phố Quang Trung, P10, Gò Vấp",
                    "quan": "Go Vap",
                    "phuong": "Phường 10",
                    "loai_hinh": "Nhà ngõ, hẻm",
                    "dien_tich_m2": 52.0,
                    "dien_tich_su_dung_m2": 110.0,
                    "so_phong_ngu": 3,
                    "so_phong_ve_sinh": 2,
                    "tong_so_tang": 2,
                    "chieu_ngang_m": 4.0,
                    "chieu_dai_m": 13.0,
                    "giay_to_phap_ly": "Đã có sổ",
                    "tinh_trang_noi_that": "Hoàn thiện cơ bản",
                    "dac_diem": "Nhà nở hậu",
                },
            ]
        )

        st.download_button(
            "📄 Tải file CSV mẫu (3 căn nhà chưa có giá) để thử nghiệm ngay",
            data=sample_csv_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="mau_nha_chua_co_gia.csv",
            mime="text/csv",
        )

        csv_file = st.file_uploader(
            "Chọn file CSV cần dự đoán giá",
            type=["csv"],
            key="batch_price_csv",
        )

    if csv_file is not None:
        try:
            raw_df = pd.read_csv(csv_file)
            with st.container(border=True):
                st.write(f"Đã tải lên **{len(raw_df):,}** căn nhà | **{len(raw_df.columns)}** cột thuộc tính.")
                st.dataframe(raw_df.head(10), width="stretch")

                price_cols = [c for c in ["gia_ban", "gia_ty", "gia_m2", "don_gia"] if c in raw_df.columns]
                if price_cols:
                    st.info(
                        "Lưu ý: File đang có sẵn cột giá ("
                        + ", ".join(price_cols)
                        + "). Hệ thống sẽ tự động bỏ qua các cột này khi dự đoán để chống rò rỉ dữ liệu (data leakage)."
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
