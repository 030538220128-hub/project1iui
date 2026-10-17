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
    page_title="Phát hiện bất thường — Nhà Tốt AI",
    page_icon="⚠️",
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
MODEL_PATH = BASE_DIR / "models" / "project1_anomaly_model.pkl"


# =========================================================
# LOAD
# =========================================================

@st.cache_resource
def load_artifact(model_path: Path):
    return joblib.load(model_path)


try:
    artifact = load_artifact(MODEL_PATH)
except Exception as e:
    st.error(f"Không thể load model: {e}")
    st.stop()


reference_rows = artifact["reference_rows"].copy()
category_options = artifact["category_options"]
numeric_defaults = artifact["numeric_defaults"]
phuong_by_quan = artifact.get("phuong_by_quan", {})


# =========================================================
# HELPERS
# =========================================================

def options(name: str) -> list[str]:
    values = [
        str(x)
        for x in category_options.get(name, [])
        if pd.notna(x) and str(x) != "Không rõ"
    ]
    return ["Không rõ"] + values


def phuong_options(selected_quan: str) -> list[str]:
    if selected_quan and selected_quan != "Không rõ" and selected_quan in phuong_by_quan:
        return ["Không rõ"] + list(phuong_by_quan[selected_quan])
    return options("phuong")


def nullable(value):
    if value == "Không rõ" or pd.isna(value):
        return np.nan
    return value


def num_default(name: str, fallback: float = 0.0) -> float:
    value = numeric_defaults.get(name, fallback)
    if value is None or pd.isna(value):
        return float(fallback)
    return float(value)


def get_market_defaults(quan, loai_hinh) -> dict[str, float]:
    df = artifact["market_group_defaults"]
    mask = (
        df["quan"].astype(str).eq(str(quan))
        & df["loai_hinh"].astype(str).eq(str(loai_hinh))
    )
    matched = df.loc[mask]
    result = {}
    for feature in artifact["market_features"]:
        value = np.nan
        if not matched.empty:
            value = matched.iloc[0].get(feature, np.nan)
        if pd.isna(value):
            value = num_default(feature, 0.0)
        result[feature] = float(value)
    return result


def get_p1090_reference(quan, phuong, loai_hinh):
    tables = artifact["p1090_reference_tables"]

    l1 = tables["level_1_quan_phuong_loai_hinh"]
    mask = (
        l1["quan"].astype(str).eq(str(quan))
        & l1["phuong"].astype(str).eq(str(phuong))
        & l1["loai_hinh"].astype(str).eq(str(loai_hinh))
    )
    row = l1.loc[mask]
    if not row.empty:
        r = row.iloc[0]
        return float(r["ref_p10"]), float(r["ref_p90"]), "phường + loại hình"

    l2 = tables["level_2_quan_loai_hinh"]
    mask = (
        l2["quan"].astype(str).eq(str(quan))
        & l2["loai_hinh"].astype(str).eq(str(loai_hinh))
    )
    row = l2.loc[mask]
    if not row.empty:
        r = row.iloc[0]
        return float(r["ref_p10"]), float(r["ref_p90"]), "quận + loại hình"

    l3 = tables["level_3_quan"]
    mask = l3["quan"].astype(str).eq(str(quan))
    row = l3.loc[mask]
    if not row.empty:
        r = row.iloc[0]
        return float(r["ref_p10"]), float(r["ref_p90"]), "quận"

    return np.nan, np.nan, "Không có"


def percentile_iforest(raw_score: float) -> float:
    scores = np.asarray(artifact["if_reference_scores"], dtype=float)
    if len(scores) == 0:
        return 0.0
    rank = np.searchsorted(scores, raw_score, side="right")
    return float(rank / len(scores))


# =========================================================
# NEW LISTING ANOMALY
# =========================================================

def analyze_listing(data: dict, custom_weights: dict | None = None, custom_threshold: float | None = None):
    gia_ty = float(data["gia_ty"])
    dien_tich = float(data["dien_tich_m2"])
    safe_dien_tich = max(dien_tich, 0.001)
    gia_m2 = gia_ty * 1000.0 / safe_dien_tich

    market = get_market_defaults(data["quan"], data["loai_hinh"])
    row = {
        **data,
        **market,
        "gia_m2_calc": gia_m2,
    }

    reg_df = pd.DataFrame([{c: row.get(c, np.nan) for c in artifact["reg_features"]}])
    pred_log = artifact["price_model"].predict(reg_df)[0]
    predicted_price = max(float(np.expm1(pred_log)), 0.0)

    # 1. RESIDUAL-Z
    residual = gia_ty - predicted_price
    resid_std = artifact["residual_std"]
    z = (residual - artifact["residual_mean"]) / resid_std if resid_std > 0 else 0.0
    score_resid = float(np.clip(abs(z) / artifact["residual_score_max_z"], 0.0, 1.0))

    # 2. HISTORICAL MIN/MAX
    hist_min = market["gia_khuvuc_min"]
    hist_max = market["gia_khuvuc_max"]
    hist_direction = 0.0
    hist_distance = 0.0
    if gia_m2 < hist_min:
        hist_direction = -1.0
        hist_distance = hist_min - gia_m2
    elif gia_m2 > hist_max:
        hist_direction = 1.0
        hist_distance = gia_m2 - hist_max

    hist_width = max(hist_max - hist_min, artifact["historical_width_floor"])
    score_hist = float(np.clip(hist_distance / hist_width, 0.0, 1.0))

    # 3. P10 - P90
    p10, p90, ref_level = get_p1090_reference(data["quan"], data["phuong"], data["loai_hinh"])
    p_direction = 0.0
    p_distance = 0.0
    score_p1090 = 0.0
    if pd.notna(p10) and pd.notna(p90):
        if gia_m2 < p10:
            p_direction = -1.0
            p_distance = p10 - gia_m2
        elif gia_m2 > p90:
            p_direction = 1.0
            p_distance = gia_m2 - p90
        width = max(p90 - p10, artifact["p1090_width_floor"])
        score_p1090 = float(np.clip(p_distance / width, 0.0, 1.0))

    # 4. ISOLATION FOREST
    if_cols = artifact["if_numeric_features"] + artifact["if_categorical_features"]
    if_row = pd.DataFrame([{c: row.get(c, np.nan) for c in if_cols}])
    X_if = artifact["if_preprocessor"].transform(if_row)
    raw_if = -float(artifact["isolation_forest"].decision_function(X_if)[0])
    score_iforest = percentile_iforest(raw_if)

    # COMPOSITE
    weights = custom_weights or artifact["weights"]
    composite = 100.0 * (
        weights["resid"] * score_resid
        + weights["hist"] * score_hist
        + weights["p1090"] * score_p1090
        + weights["iforest"] * score_iforest
    )

    threshold = float(custom_threshold if custom_threshold is not None else artifact["anomaly_threshold"])
    is_anomaly = composite >= threshold

    # DIRECTION
    dw = artifact["direction_weights"]
    signed_score = (
        dw["resid"] * np.sign(residual) * score_resid
        + dw["hist"] * hist_direction * score_hist
        + dw["p1090"] * p_direction * score_p1090
    )

    if not is_anomaly:
        anomaly_type = "Bình thường"
    elif signed_score > dw["cutoff"]:
        anomaly_type = "Quá đắt"
    elif signed_score < -dw["cutoff"]:
        anomaly_type = "Quá rẻ"
    else:
        anomaly_type = "Bất thường cấu trúc / khác"

    reasons = []
    if abs(z) >= 2:
        if residual > 0:
            reasons.append(f"Giá thực cao hơn dự đoán {abs(residual):.2f} tỷ")
        else:
            reasons.append(f"Giá thực thấp hơn dự đoán {abs(residual):.2f} tỷ")
    if score_hist > 0:
        reasons.append(
            "Đơn giá cao hơn historical max" if hist_direction > 0 else "Đơn giá thấp hơn historical min"
        )
    if score_p1090 > 0:
        reasons.append(
            "Đơn giá cao hơn P90 nhóm tham chiếu" if p_direction > 0 else "Đơn giá thấp hơn P10 nhóm tham chiếu"
        )
    if score_iforest >= 0.95:
        reasons.append("Isolation Forest: tổ hợp feature thuộc top 5% bất thường")
    if not reasons:
        reasons.append("Composite Score được tạo từ nhiều tín hiệu mức vừa.")

    return {
        "predicted_price": predicted_price,
        "gia_m2": gia_m2,
        "residual": residual,
        "residual_z": z,
        "score_resid": score_resid,
        "score_hist": score_hist,
        "score_p1090": score_p1090,
        "score_iforest": score_iforest,
        "composite": composite,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "reason": "; ".join(reasons),
        "p10": p10,
        "p90": p90,
        "ref_level": ref_level,
        "hist_min": hist_min,
        "hist_max": hist_max,
    }


def show_result(result: dict):
    with st.container(border=True):
        st.subheader("🔎 Kết quả thẩm định bất thường")

        if not result["is_anomaly"]:
            st.success("✅ BÌNH THƯỜNG — Tin đăng nằm trong biên độ an toàn của thị trường.")
        elif result["anomaly_type"] == "Quá rẻ":
            st.warning("⚠️ PHÁT HIỆN: QUÁ RẺ — Cảnh báo tin mồi câu khách, sổ chung/vi bằng hoặc nhập sai đơn vị.")
        elif result["anomaly_type"] == "Quá đắt":
            st.error("🚨 PHÁT HIỆN: QUÁ ĐẮT — Cảnh báo giá rao vượt xa mặt bằng khu vực và cấu trúc nhà.")
        else:
            st.warning("⚠️ BẤT THƯỜNG CẤU TRÚC — Tổ hợp diện tích / số tầng / phòng ngủ dị biệt.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Composite Score", f"{result['composite']:.2f}")
        with c2:
            st.metric("Ngưỡng anomaly", f"{result['threshold']:.2f}")
        with c3:
            st.metric("Giá model kỳ vọng", f"{result['predicted_price']:.2f} tỷ")
        with c4:
            st.metric("Giá / m²", f"{result['gia_m2']:.1f} triệu")

        st.markdown("##### 📊 Phân rã 4 tín hiệu bất thường (Thang chuẩn hóa 0 – 1)")
        signal_df = pd.DataFrame(
            {
                "Tín hiệu": [
                    "1. Residual-Z (40%)",
                    "2. Historical Min/Max (20%)",
                    "3. P10–P90 (25%)",
                    "4. Isolation Forest (15%)",
                ],
                "Score": [
                    result["score_resid"],
                    result["score_hist"],
                    result["score_p1090"],
                    result["score_iforest"],
                ],
            }
        )
        st.bar_chart(signal_df.set_index("Tín hiệu"), horizontal=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Sai số Residual", f"{result['residual']:+.2f} tỷ")
            st.caption(f"Z-score = {result['residual_z']:.2f}")
        with c2:
            st.metric("Biên độ lịch sử (Min – Max)", f"{result['hist_min']:.1f} – {result['hist_max']:.1f}")
            st.caption("triệu VNĐ/m²")
        with c3:
            if pd.notna(result["p10"]) and pd.notna(result["p90"]):
                st.metric("Khoảng P10 – P90", f"{result['p10']:.1f} – {result['p90']:.1f}")
                st.caption(f"Cấp tham chiếu: {result['ref_level']}")
            else:
                st.metric("Khoảng P10 – P90", "N/A")

        st.info("💡 **Giải thích XAI (Lý do gắn cờ):** " + result["reason"])


# =========================================================
# SIDEBAR & HEADER
# =========================================================

st.markdown(
    """
    <div class="page-banner">
        <h1>⚠️ Bài toán 2: Phát hiện tin đăng giá bất thường</h1>
        <p>Hệ thống chấm điểm đa tín hiệu (Composite Score 0–100) giúp sàng lọc tin đăng Quá rẻ, Quá đắt hoặc Bất thường cấu trúc kèm giải thích XAI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "Nguồn dữ liệu",
    [
        "📚 Chọn tin có sẵn",
        "✍️ Kiểm tra tin mới",
        "📤 Upload CSV có giá → Phát hiện bất thường hàng loạt",
    ],
    horizontal=True,
)

with st.sidebar:
    st.markdown("### 🎛️ Bảng điều khiển Anomaly")
    if mode == "📚 Chọn tin có sẵn":
        st.markdown("#### 🔍 Bộ lọc tin đăng")
        only_anomaly = st.checkbox("Chỉ hiển thị tin đã được gắn cờ anomaly (Top 5%)", value=True)
        df = reference_rows.copy()
        if only_anomaly:
            df = df[df["is_anomaly"].fillna(False).astype(bool)]

        districts = sorted(df["quan"].dropna().astype(str).unique())
        selected_district = st.selectbox("Quận", districts)
        df = df[df["quan"].astype(str).eq(selected_district)]

        anom_types = ["Tất cả"] + sorted(df["anomaly_type"].dropna().astype(str).unique().tolist())
        selected_type = st.selectbox("Loại phân loại", anom_types)
        if selected_type != "Tất cả":
            df = df[df["anomaly_type"].astype(str).eq(selected_type)]

        indices = df.index.tolist()

        def label(idx):
            r = df.loc[idx]
            title = str(r.get("tieu_de", "Không có tiêu đề"))
            if len(title) > 60:
                title = title[:57] + "..."
            score = r.get("composite_score", np.nan)
            return f"[{r.get('anomaly_type', '')} | {score:.1f}đ] {title}"

        if not df.empty:
            selected_idx = st.selectbox("Chọn tin đăng cần xem chi tiết", indices, format_func=label)
            row = df.loc[selected_idx]
        else:
            row = None

    elif mode == "✍️ Kiểm tra tin mới":
        st.markdown("#### 📍 1. Vị trí & Loại hình")
        quan_opts = options("quan")
        default_quan_idx = 1 if len(quan_opts) > 1 else 0
        quan = st.selectbox("Quận", quan_opts, index=default_quan_idx)
        phuong = st.selectbox("Phường", phuong_options(quan))
        loai_opts = options("loai_hinh")
        default_loai_idx = 1 if len(loai_opts) > 1 else 0
        loai_hinh = st.selectbox("Loại hình", loai_opts, index=default_loai_idx)

        with st.form("new_listing_form", border=True):
            st.markdown("#### 📐 2. Giá rao & Kích thước")
            gia_ty = st.number_input("Giá bán rao trên tin (tỷ VNĐ)", min_value=0.1, value=7.0, step=0.1)
            dien_tich = st.number_input(
                "Diện tích đất / công nhận (m²)",
                min_value=1.0,
                value=max(1.0, num_default("dien_tich_m2", 52.0)),
            )
            dien_tich_sd = st.number_input(
                "Diện tích sử dụng (m²)",
                min_value=0.0,
                value=max(0.0, num_default("dien_tich_su_dung_m2", 120.0)),
            )
            phong_ngu = st.number_input(
                "Phòng ngủ",
                min_value=0.0,
                value=max(0.0, num_default("so_phong_ngu", 3.0)),
            )
            phong_wc = st.number_input(
                "Phòng vệ sinh",
                min_value=0.0,
                value=max(0.0, num_default("so_phong_ve_sinh", 3.0)),
            )
            so_tang = st.number_input(
                "Số tầng",
                min_value=0.0,
                value=max(0.0, num_default("tong_so_tang", 3.0)),
            )
            ngang = st.number_input(
                "Chiều ngang (m)",
                min_value=0.0,
                value=max(0.0, num_default("chieu_ngang_m", 4.0)),
            )
            dai = st.number_input(
                "Chiều dài (m)",
                min_value=0.0,
                value=max(0.0, num_default("chieu_dai_m", 13.0)),
            )
            giay_to = st.selectbox("Giấy tờ pháp lý", options("giay_to_phap_ly"))
            noi_that = st.selectbox("Nội thất", options("tinh_trang_noi_that"))
            dac_diem = st.selectbox("Đặc điểm", options("dac_diem"))

            submitted = st.form_submit_button(
                "⚠️ Phân tích bất thường",
                type="primary",
                width="stretch",
            )

    else:
        st.markdown("#### ⚙️ Cấu hình trọng số w₁..w₄")
        method_choice = st.selectbox(
            "Phương pháp phát hiện",
            [
                "Composite Score (Kết hợp 4 tín hiệu - Khuyên dùng)",
                "Residual-z (Sai số dự đoán XGBoost)",
                "Vi phạm Min/Max (Biên độ đơn giá)",
                "Khoảng P10 - P90 (Bách phân vị đơn giá)",
                "Isolation Forest (Bất thường đa chiều)",
            ],
            key="batch_anom_method",
        )
        w_resid = st.slider("w1 (Residual-Z)", 0.0, 1.0, float(artifact["weights"]["resid"]), 0.05)
        w_p1090 = st.slider("w2 (P10–P90)", 0.0, 1.0, float(artifact["weights"]["p1090"]), 0.05)
        w_hist = st.slider("w3 (Historical)", 0.0, 1.0, float(artifact["weights"]["hist"]), 0.05)
        w_if = st.slider("w4 (Isolation Forest)", 0.0, 1.0, float(artifact["weights"]["iforest"]), 0.05)
        top_k_pct = st.slider("Top-k% gắn cờ", 1, 50, int(artifact["top_k"] * 100), 1)


# =========================================================
# MODE 1 — EXISTING
# =========================================================

if mode == "📚 Chọn tin có sẵn":
    if row is None or df.empty:
        st.warning("Không có tin đăng nào khớp với bộ lọc hiện tại ở thanh bên trái.")
    else:
        with st.container(border=True):
            st.subheader(str(row.get("tieu_de", "")))
            st.caption(f"📍 Địa chỉ: {row.get('dia_chi', '')} | Loại hình: {row.get('loai_hinh', 'N/A')} | Diện tích: {row.get('dien_tich_m2', 0):.1f} m²")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Giá đăng", f"{row['gia_ty']:.2f} tỷ")
            c2.metric("Giá dự đoán OOF", f"{row['gia_du_doan_oof']:.2f} tỷ")
            c3.metric("Composite Score", f"{row['composite_score']:.2f}")
            c4.metric("Phân loại", row["anomaly_type"])

            score_df = pd.DataFrame(
                {
                    "Tín hiệu": [
                        "1. Residual-Z (40%)",
                        "2. Historical Min/Max (20%)",
                        "3. P10–P90 (25%)",
                        "4. Isolation Forest (15%)",
                    ],
                    "Score": [
                        row["score_resid"],
                        row["score_hist"],
                        row["score_p1090"],
                        row["score_iforest"],
                    ],
                }
            )
            st.bar_chart(score_df.set_index("Tín hiệu"), horizontal=True)

            if bool(row["is_anomaly"]):
                st.warning(f"⚠️ Cảnh báo kiểm duyệt: **{row['anomaly_type']}**")
            else:
                st.success("✅ Tin đăng Bình thường")

            st.info("💡 **Giải thích XAI:** " + str(row["ly_do_bat_thuong"]))


# =========================================================
# MODE 2 — NEW LISTING
# =========================================================

elif mode == "✍️ Kiểm tra tin mới":
    if not submitted:
        with st.container(border=True):
            st.markdown("### 👈 Bảng nhập liệu đã sẵn sàng ở Thanh bên trái (Sidebar)")
            st.write(
                """
                Hãy nhập mức giá rao bán cùng thông số diện tích, vị trí và kết cấu của căn nhà
                ở **Thanh bên trái (Sidebar)**, sau đó nhấn nút **⚠️ Phân tích bất thường**
                để hệ thống chấm điểm Composite Score và đối chiếu 4 tín hiệu thị trường.
                """
            )
            inf1, inf2 = st.columns(2)
            with inf1:
                st.metric("Vị trí kiểm tra", f"{phuong}, {quan}")
            with inf2:
                st.metric("Giá rao dự kiến", f"{gia_ty:.2f} tỷ VNĐ")

    if submitted:
        if quan == "Không rõ" or loai_hinh == "Không rõ":
            st.warning("Nên chọn Quận và Loại hình để hệ thống có nhóm tham chiếu phù hợp.")

        data = {
            "gia_ty": gia_ty,
            "dien_tich_m2": dien_tich,
            "dien_tich_su_dung_m2": dien_tich_sd,
            "so_phong_ngu": phong_ngu,
            "so_phong_ve_sinh": phong_wc,
            "tong_so_tang": so_tang,
            "chieu_ngang_m": ngang,
            "chieu_dai_m": dai,
            "quan": nullable(quan),
            "phuong": nullable(phuong),
            "loai_hinh": nullable(loai_hinh),
            "giay_to_phap_ly": nullable(giay_to),
            "tinh_trang_noi_that": nullable(noi_that),
            "dac_diem": nullable(dac_diem),
        }

        try:
            result = analyze_listing(data)
            show_result(result)
        except Exception as e:
            st.error(f"Lỗi khi phân tích: {e}")


# =========================================================
# MODE 3 — BATCH CSV ANOMALY DETECTION
# =========================================================

else:
    with st.container(border=True):
        st.subheader("📤 Upload CSV có giá → Phát hiện bất thường hàng loạt")
        st.write(
            """
            Tải lên file CSV chứa danh sách nhà **kèm cột giá bán thực tế** (`gia_ty`, `gia_ban`, `gia_m2` hoặc `don_gia`).
            Các thanh trượt điều chỉnh trọng số w₁..w₄ và ngưỡng Top-k% đã được chuyển sang **Thanh bên trái (Sidebar)**.
            """
        )

        sample_anom_df = pd.DataFrame(
            [
                {
                    "tieu_de": "[Mẫu 1 - Bình thường] Nhà hẻm xe hơi P7 Bình Thạnh",
                    "quan": "Binh Thanh",
                    "phuong": "Phường 7",
                    "loai_hinh": "Nhà ngõ, hẻm",
                    "gia_ty": 6.8,
                    "dien_tich_m2": 50.0,
                    "dien_tich_su_dung_m2": 140.0,
                    "so_phong_ngu": 3,
                    "so_phong_ve_sinh": 3,
                    "tong_so_tang": 3,
                    "chieu_ngang_m": 4.0,
                    "chieu_dai_m": 12.5,
                    "giay_to_phap_ly": "Đã có sổ",
                },
                {
                    "tieu_de": "[Mẫu 2 - Quá rẻ / Nghi vấn tin ảo] Nhà 100m² Phú Nhuận chỉ 0.85 tỷ",
                    "quan": "Phu Nhuan",
                    "phuong": "Phường 14",
                    "loai_hinh": "Nhà ngõ, hẻm",
                    "gia_ty": 0.85,
                    "dien_tich_m2": 100.0,
                    "dien_tich_su_dung_m2": 200.0,
                    "so_phong_ngu": 4,
                    "so_phong_ve_sinh": 3,
                    "tong_so_tang": 3,
                    "chieu_ngang_m": 5.0,
                    "chieu_dai_m": 20.0,
                    "giay_to_phap_ly": "Sổ chung / vi bằng",
                },
                {
                    "tieu_de": "[Mẫu 3 - Quá đắt / Thổi giá] Nhà hẻm 30m² Gò Vấp rao 28 tỷ",
                    "quan": "Go Vap",
                    "phuong": "Phường 10",
                    "loai_hinh": "Nhà ngõ, hẻm",
                    "gia_ty": 28.0,
                    "dien_tich_m2": 30.0,
                    "dien_tich_su_dung_m2": 60.0,
                    "so_phong_ngu": 2,
                    "so_phong_ve_sinh": 2,
                    "tong_so_tang": 2,
                    "chieu_ngang_m": 3.0,
                    "chieu_dai_m": 10.0,
                    "giay_to_phap_ly": "Đã có sổ",
                },
            ]
        )

        st.download_button(
            "📄 Tải file CSV mẫu (gồm tin Bình thường, Quá rẻ, Quá đắt) để thử nghiệm ngay",
            data=sample_anom_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="mau_kiem_tra_bat_thuong.csv",
            mime="text/csv",
        )

        csv_anom = st.file_uploader(
            "Chọn file CSV có cột giá để kiểm tra bất thường",
            type=["csv"],
            key="batch_anom_csv",
        )

    if csv_anom is not None:
        try:
            df_up = pd.read_csv(csv_anom)
            with st.container(border=True):
                st.write(f"Đã tải lên **{len(df_up):,}** dòng | **{len(df_up.columns)}** cột.")
                st.dataframe(df_up.head(10), width="stretch")

                if st.button("⚠️ Chạy phát hiện bất thường cho file CSV", type="primary", width="stretch"):
                    if method_choice.startswith("Residual-z"):
                        norm_w = {"resid": 1.0, "p1090": 0.0, "hist": 0.0, "iforest": 0.0}
                    elif method_choice.startswith("Vi phạm Min/Max"):
                        norm_w = {"resid": 0.0, "p1090": 0.0, "hist": 1.0, "iforest": 0.0}
                    elif method_choice.startswith("Khoảng P10 - P90"):
                        norm_w = {"resid": 0.0, "p1090": 1.0, "hist": 0.0, "iforest": 0.0}
                    elif method_choice.startswith("Isolation Forest"):
                        norm_w = {"resid": 0.0, "p1090": 0.0, "hist": 0.0, "iforest": 1.0}
                    else:
                        w_sum = max(w_resid + w_p1090 + w_hist + w_if, 1e-6)
                        norm_w = {
                            "resid": w_resid / w_sum,
                            "p1090": w_p1090 / w_sum,
                            "hist": w_hist / w_sum,
                            "iforest": w_if / w_sum,
                        }

                    if {"score_resid", "score_hist", "score_p1090", "score_iforest"}.issubset(df_up.columns):
                        out_df = df_up.copy()
                        out_df["composite_score"] = 100.0 * (
                            norm_w["resid"] * out_df["score_resid"]
                            + norm_w["hist"] * out_df["score_hist"]
                            + norm_w["p1090"] * out_df["score_p1090"]
                            + norm_w["iforest"] * out_df["score_iforest"]
                        )
                        cutoff_score = float(np.quantile(out_df["composite_score"].dropna(), 1.0 - top_k_pct / 100.0))
                        out_df["is_anomaly"] = out_df["composite_score"] >= cutoff_score
                    else:
                        def _parse_num(val, fallback: float) -> float:
                            if pd.isna(val):
                                return fallback
                            if isinstance(val, (int, float, np.number)):
                                return float(val)
                            m = re.search(r"[-+]?\d*\.?\d+", str(val).replace(",", "."))
                            return float(m.group(0)) if m else fallback

                        def _parse_price_ty(val, fallback: float = 5.0) -> float:
                            if pd.isna(val):
                                return fallback
                            if isinstance(val, (int, float, np.number)):
                                return float(val)
                            s = str(val).lower().replace(",", ".")
                            num = _parse_num(s, fallback)
                            if "triệu" in s or "tr" in s:
                                return num / 1000.0
                            return num

                        work_df = df_up.copy()
                        if "dia_chi" in work_df.columns:
                            addr = work_df["dia_chi"].fillna("").astype(str)
                            if "quan" not in work_df.columns or work_df["quan"].isna().all():
                                work_df["quan"] = np.select(
                                    [
                                        addr.str.contains("Bình Thạnh|Binh Thanh", case=False, regex=True),
                                        addr.str.contains("Gò Vấp|Go Vap", case=False, regex=True),
                                        addr.str.contains("Phú Nhuận|Phu Nhuan", case=False, regex=True),
                                    ],
                                    ["Binh Thanh", "Go Vap", "Phu Nhuan"],
                                    default="Go Vap",
                                )
                            if "phuong" not in work_df.columns or work_df["phuong"].isna().all():
                                work_df["phuong"] = addr.str.extract(r"(Phường\s+[^,()]+)", expand=False).str.strip()

                        gia_series = (
                            work_df["gia_ty"].map(lambda v: _parse_num(v, 5.0))
                            if "gia_ty" in work_df.columns
                            else work_df.get("gia_ban", pd.Series(5.0, index=work_df.index)).map(_parse_price_ty)
                        )
                        dt_col = "dien_tich_m2" if "dien_tich_m2" in work_df.columns else "dien_tich"
                        dt_series = (
                            work_df[dt_col].map(lambda v: max(1.0, _parse_num(v, 52.0)))
                            if dt_col in work_df.columns
                            else pd.Series(52.0, index=work_df.index)
                        )

                        batch_df = pd.DataFrame(index=work_df.index)
                        batch_df["gia_ty"] = gia_series.astype(float)
                        batch_df["dien_tich_m2"] = dt_series.astype(float)
                        batch_df["gia_m2_calc"] = batch_df["gia_ty"] * 1000.0 / np.maximum(batch_df["dien_tich_m2"], 0.001)

                        num_map = {
                            "dien_tich_su_dung_m2": ("dien_tich_su_dung_m2", "dien_tich_su_dung", 120.0),
                            "so_phong_ngu": ("so_phong_ngu", "so_phong_ngu_num", 3.0),
                            "so_phong_ve_sinh": ("so_phong_ve_sinh", "so_phong_ve_sinh_num", 3.0),
                            "tong_so_tang": ("tong_so_tang", "tong_so_tang_num", 3.0),
                            "chieu_ngang_m": ("chieu_ngang_m", "chieu_ngang", 4.0),
                            "chieu_dai_m": ("chieu_dai_m", "chieu_dai", 12.0),
                        }
                        for target_c, (c1_name, c2_name, fb) in num_map.items():
                            src_c = c1_name if c1_name in work_df.columns else (c2_name if c2_name in work_df.columns else None)
                            batch_df[target_c] = (
                                work_df[src_c].map(lambda v, f=fb: _parse_num(v, f)) if src_c else fb
                            )

                        for cat_c, fb_cat in [
                            ("quan", "Go Vap"),
                            ("phuong", np.nan),
                            ("loai_hinh", "Nhà ngõ, hẻm"),
                            ("giay_to_phap_ly", np.nan),
                            ("tinh_trang_noi_that", np.nan),
                            ("dac_diem", np.nan),
                        ]:
                            batch_df[cat_c] = work_df[cat_c] if cat_c in work_df.columns else fb_cat

                        for feat in artifact["market_features"]:
                            batch_df[feat] = work_df[feat] if feat in work_df.columns else np.nan

                        for (q_val, l_val), idx_grp in batch_df.groupby(["quan", "loai_hinh"], dropna=False).groups.items():
                            mkt = get_market_defaults(q_val, l_val)
                            for feat, val in mkt.items():
                                batch_df.loc[idx_grp, feat] = batch_df.loc[idx_grp, feat].fillna(val)

                        reg_df = batch_df[artifact["reg_features"]]
                        pred_prices = np.maximum(np.expm1(artifact["price_model"].predict(reg_df)), 0.0)
                        residuals = batch_df["gia_ty"].to_numpy() - pred_prices
                        resid_std = artifact["residual_std"]
                        z_vals = (residuals - artifact["residual_mean"]) / resid_std if resid_std > 0 else np.zeros_like(residuals)
                        score_resid = np.clip(np.abs(z_vals) / artifact["residual_score_max_z"], 0.0, 1.0)

                        gia_m2_arr = batch_df["gia_m2_calc"].to_numpy()
                        hist_min_arr = batch_df["gia_khuvuc_min"].to_numpy()
                        hist_max_arr = batch_df["gia_khuvuc_max"].to_numpy()
                        hist_dir = np.where(gia_m2_arr < hist_min_arr, -1.0, np.where(gia_m2_arr > hist_max_arr, 1.0, 0.0))
                        hist_dist = np.where(gia_m2_arr < hist_min_arr, hist_min_arr - gia_m2_arr, np.where(gia_m2_arr > hist_max_arr, gia_m2_arr - hist_max_arr, 0.0))
                        hist_width = np.maximum(hist_max_arr - hist_min_arr, artifact["historical_width_floor"])
                        score_hist = np.clip(hist_dist / hist_width, 0.0, 1.0)

                        p1090_cache = {}
                        p10_arr = np.empty(len(batch_df), dtype=float)
                        p90_arr = np.empty(len(batch_df), dtype=float)
                        for i, (q_v, p_v, l_v) in enumerate(zip(batch_df["quan"], batch_df["phuong"], batch_df["loai_hinh"])):
                            key = (str(q_v), str(p_v), str(l_v))
                            if key not in p1090_cache:
                                p1090_cache[key] = get_p1090_reference(q_v, p_v, l_v)[:2]
                            p10_arr[i], p90_arr[i] = p1090_cache[key]

                        valid_p = ~np.isnan(p10_arr) & ~np.isnan(p90_arr)
                        p_dir = np.where(valid_p & (gia_m2_arr < p10_arr), -1.0, np.where(valid_p & (gia_m2_arr > p90_arr), 1.0, 0.0))
                        p_dist = np.where(valid_p & (gia_m2_arr < p10_arr), p10_arr - gia_m2_arr, np.where(valid_p & (gia_m2_arr > p90_arr), gia_m2_arr - p90_arr, 0.0))
                        p_width = np.maximum(np.where(valid_p, p90_arr - p10_arr, 0.0), artifact["p1090_width_floor"])
                        score_p1090 = np.where(valid_p, np.clip(p_dist / p_width, 0.0, 1.0), 0.0)

                        if_cols = artifact["if_numeric_features"] + artifact["if_categorical_features"]
                        X_if = artifact["if_preprocessor"].transform(batch_df[if_cols])
                        raw_if = -artifact["isolation_forest"].decision_function(X_if)
                        ref_scores = np.asarray(artifact["if_reference_scores"], dtype=float)
                        score_iforest = (
                            np.searchsorted(ref_scores, raw_if, side="right") / len(ref_scores)
                            if len(ref_scores) > 0
                            else np.zeros_like(raw_if)
                        )

                        comp_scores = 100.0 * (
                            norm_w["resid"] * score_resid
                            + norm_w["hist"] * score_hist
                            + norm_w["p1090"] * score_p1090
                            + norm_w["iforest"] * score_iforest
                        )
                        cutoff_score = float(np.quantile(comp_scores, 1.0 - top_k_pct / 100.0))
                        is_anom = comp_scores >= cutoff_score

                        dw = artifact["direction_weights"]
                        signed = (
                            dw["resid"] * np.sign(residuals) * score_resid
                            + dw["hist"] * hist_dir * score_hist
                            + dw["p1090"] * p_dir * score_p1090
                        )
                        anom_types = np.where(
                            ~is_anom,
                            "Bình thường",
                            np.where(
                                signed > dw["cutoff"],
                                "Quá đắt",
                                np.where(signed < -dw["cutoff"], "Quá rẻ", "Bất thường cấu trúc / khác"),
                            ),
                        )

                        out_df = work_df.copy()
                        out_df["gia_du_doan_ty"] = np.round(pred_prices, 2)
                        out_df["composite_score"] = np.round(comp_scores, 2)
                        out_df["is_anomaly"] = is_anom
                        out_df["anomaly_type"] = anom_types

                    flagged = out_df[out_df["is_anomaly"]].sort_values("composite_score", ascending=False)
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Tổng số tin kiểm tra", f"{len(out_df):,}")
                    k2.metric("Số tin bất thường", f"{len(flagged):,}")
                    k3.metric("Ngưỡng Composite Score", f"{cutoff_score:.2f}")

                    st.write("### 🚨 Danh sách tin bị gắn cờ bất thường")
                    st.dataframe(flagged.head(100), width="stretch")

                    st.download_button(
                        "📥 Tải danh sách kết quả phát hiện bất thường (.csv)",
                        data=out_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="ket_qua_phat_hien_bat_thuong.csv",
                        mime="text/csv",
                        width="stretch",
                    )
        except Exception as e:
            st.error(f"Không thể xử lý file CSV: {e}")
