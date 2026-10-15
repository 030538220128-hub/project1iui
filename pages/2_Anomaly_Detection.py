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
    page_title="Anomaly Detection",
    page_icon="⚠️",
    layout="wide",
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
    st.divider()

    if not result["is_anomaly"]:
        st.success("✅ BÌNH THƯỜNG")
    elif result["anomaly_type"] == "Quá rẻ":
        st.warning("⚠️ PHÁT HIỆN: QUÁ RẺ")
    elif result["anomaly_type"] == "Quá đắt":
        st.error("🚨 PHÁT HIỆN: QUÁ ĐẮT")
    else:
        st.warning("⚠️ BẤT THƯỜNG CẤU TRÚC")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Composite Score", f"{result['composite']:.2f}")
    with c2:
        st.metric("Ngưỡng anomaly", f"{result['threshold']:.2f}")
    with c3:
        st.metric("Giá model kỳ vọng", f"{result['predicted_price']:.2f} tỷ")
    with c4:
        st.metric("Giá / m²", f"{result['gia_m2']:.1f} triệu")

    st.subheader("4 tín hiệu")
    signal_df = pd.DataFrame(
        {
            "Tín hiệu": [
                "Residual-Z",
                "Historical Min/Max",
                "P10–P90",
                "Isolation Forest",
            ],
            "Score": [
                result["score_resid"],
                result["score_hist"],
                result["score_p1090"],
                result["score_iforest"],
            ],
        }
    )
    st.bar_chart(signal_df.set_index("Tín hiệu"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Residual", f"{result['residual']:+.2f} tỷ")
        st.caption(f"Z = {result['residual_z']:.2f}")
    with c2:
        st.metric("Historical range", f"{result['hist_min']:.1f} – {result['hist_max']:.1f}")
        st.caption("triệu VNĐ/m²")
    with c3:
        if pd.notna(result["p10"]) and pd.notna(result["p90"]):
            st.metric("P10 – P90", f"{result['p10']:.1f} – {result['p90']:.1f}")
            st.caption(result["ref_level"])
        else:
            st.metric("P10 – P90", "N/A")

    st.info("**Lý do:** " + result["reason"])


# =========================================================
# HEADER
# =========================================================

st.title("⚠️ Phát hiện bất thường giá nhà")

st.write(
    """
    Hệ thống kết hợp **4 tín hiệu** để đánh giá một tin đăng:
    Residual-Z, Historical Min/Max, P10–P90 và Isolation Forest.
    """
)
st.caption("Composite Score càng cao → tin càng đáng ưu tiên kiểm tra.")
st.divider()


# =========================================================
# MODE
# =========================================================

mode = st.radio(
    "Nguồn dữ liệu",
    [
        "📚 Chọn tin có sẵn",
        "✍️ Kiểm tra tin mới",
        "📤 Upload CSV có giá → Phát hiện bất thường hàng loạt",
    ],
    horizontal=True,
)


# =========================================================
# MODE 1 — EXISTING
# =========================================================

if mode == "📚 Chọn tin có sẵn":
    st.subheader("Chọn tin từ dataset")
    only_anomaly = st.checkbox("Chỉ hiển thị tin đã được gắn cờ anomaly", value=True)

    df = reference_rows.copy()
    if only_anomaly:
        df = df[df["is_anomaly"].fillna(False).astype(bool)]

    districts = sorted(df["quan"].dropna().astype(str).unique())
    selected_district = st.selectbox("Quận", districts)
    df = df[df["quan"].astype(str).eq(selected_district)]

    indices = df.index.tolist()

    def label(idx):
        row = df.loc[idx]
        title = str(row.get("tieu_de", "Không có tiêu đề"))
        if len(title) > 90:
            title = title[:87] + "..."
        score = row.get("composite_score", np.nan)
        return f"{title} [Score: {score:.1f}]"

    selected_idx = st.selectbox("Tin đăng", indices, format_func=label)
    row = df.loc[selected_idx]

    st.divider()
    st.subheader(str(row.get("tieu_de", "")))
    st.caption(str(row.get("dia_chi", "")))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá đăng", f"{row['gia_ty']:.2f} tỷ")
    c2.metric("Giá dự đoán OOF", f"{row['gia_du_doan_oof']:.2f} tỷ")
    c3.metric("Composite Score", f"{row['composite_score']:.2f}")
    c4.metric("Phân loại", row["anomaly_type"])

    score_df = pd.DataFrame(
        {
            "Tín hiệu": ["Residual-Z", "Historical", "P10–P90", "Isolation Forest"],
            "Score": [
                row["score_resid"],
                row["score_hist"],
                row["score_p1090"],
                row["score_iforest"],
            ],
        }
    )
    st.bar_chart(score_df.set_index("Tín hiệu"))

    if bool(row["is_anomaly"]):
        st.warning(f"⚠️ {row['anomaly_type']}")
    else:
        st.success("✅ Bình thường")

    st.info("**Lý do:** " + str(row["ly_do_bat_thuong"]))


# =========================================================
# MODE 2 — NEW LISTING
# =========================================================

elif mode == "✍️ Kiểm tra tin mới":
    st.subheader("Nhập một tin đăng mới")

    with st.form("new_listing_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gia_ty = st.number_input("Giá bán (tỷ VNĐ)", min_value=0.1, value=7.0, step=0.1)
            dien_tich = st.number_input(
                "Diện tích (m²)",
                min_value=1.0,
                value=max(1.0, num_default("dien_tich_m2", 52.0)),
            )
            dien_tich_sd = st.number_input(
                "Diện tích sử dụng (m²)",
                min_value=0.0,
                value=max(0.0, num_default("dien_tich_su_dung_m2", 120.0)),
            )

        with c2:
            quan = st.selectbox("Quận", options("quan"))
            phuong = st.selectbox("Phường", phuong_options(quan))
            loai_hinh = st.selectbox("Loại hình", options("loai_hinh"))

        with c3:
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

        with st.expander("Thông tin bổ sung"):
            c1, c2 = st.columns(2)
            with c1:
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
            with c2:
                noi_that = st.selectbox("Nội thất", options("tinh_trang_noi_that"))
                dac_diem = st.selectbox("Đặc điểm", options("dac_diem"))

        submitted = st.form_submit_button(
            "⚠️ Phân tích bất thường",
            type="primary",
            width="stretch",
        )

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
    st.subheader("📤 Upload CSV có giá → Phát hiện bất thường hàng loạt")
    st.write(
        """
        Upload file CSV chứa danh sách nhà **kèm cột giá bán** (`gia_ty` hoặc `gia_ban`)
        để hệ thống chấm điểm 4 tín hiệu và lọc danh sách bất thường.
        """
    )

    with st.expander("⚙️ Tùy chỉnh trọng số Composite Score & Ngưỡng Top-k%", expanded=False):
        wc1, wc2, wc3, wc4, wc5 = st.columns(5)
        w_resid = wc1.slider("w1 (Residual-Z)", 0.0, 1.0, float(artifact["weights"]["resid"]), 0.05)
        w_p1090 = wc2.slider("w2 (P10–P90)", 0.0, 1.0, float(artifact["weights"]["p1090"]), 0.05)
        w_hist = wc3.slider("w3 (Historical)", 0.0, 1.0, float(artifact["weights"]["hist"]), 0.05)
        w_if = wc4.slider("w4 (Isolation Forest)", 0.0, 1.0, float(artifact["weights"]["iforest"]), 0.05)
        top_k_pct = wc5.slider("Top-k% gắn cờ", 1, 20, int(artifact["top_k"] * 100), 1)

    csv_anom = st.file_uploader(
        "Chọn file CSV có cột giá để kiểm tra bất thường",
        type=["csv"],
        key="batch_anom_csv",
    )

    if csv_anom is not None:
        try:
            df_up = pd.read_csv(csv_anom)
            st.write(f"Đã tải lên **{len(df_up):,}** dòng | **{len(df_up.columns)}** cột.")
            st.dataframe(df_up.head(10), width="stretch")

            if st.button("⚠️ Chạy phát hiện bất thường cho file CSV", type="primary", width="stretch"):
                w_sum = max(w_resid + w_p1090 + w_hist + w_if, 1e-6)
                norm_w = {
                    "resid": w_resid / w_sum,
                    "p1090": w_p1090 / w_sum,
                    "hist": w_hist / w_sum,
                    "iforest": w_if / w_sum,
                }

                # Nếu file đã có sẵn các cột score (như house_prices_clean.csv), tính cực nhanh
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
                    # Phân tích từng dòng (tối đa 500 dòng đầu để phản hồi tức thì)
                    work_df = df_up.head(500).copy()
                    records = []
                    for _, r in work_df.iterrows():
                        gia_val = r.get("gia_ty", np.nan)
                        if pd.isna(gia_val) and "gia_ban" in r:
                            m = re.search(r"[-+]?\d*\.?\d+", str(r["gia_ban"]).replace(",", "."))
                            gia_val = float(m.group(0)) if m else 5.0
                        dt_val = r.get("dien_tich_m2", r.get("dien_tich", 50.0))
                        m_dt = re.search(r"[-+]?\d*\.?\d+", str(dt_val).replace(",", "."))
                        dt_val = float(m_dt.group(0)) if m_dt else 50.0

                        row_input = {
                            "gia_ty": float(gia_val if pd.notna(gia_val) else 5.0),
                            "dien_tich_m2": max(1.0, float(dt_val)),
                            "dien_tich_su_dung_m2": float(r.get("dien_tich_su_dung_m2", dt_val)),
                            "so_phong_ngu": float(r.get("so_phong_ngu", 3.0)),
                            "so_phong_ve_sinh": float(r.get("so_phong_ve_sinh", 2.0)),
                            "tong_so_tang": float(r.get("tong_so_tang", 2.0)),
                            "chieu_ngang_m": float(r.get("chieu_ngang_m", 4.0)),
                            "chieu_dai_m": float(r.get("chieu_dai_m", 12.0)),
                            "quan": r.get("quan", "Go Vap"),
                            "phuong": r.get("phuong", np.nan),
                            "loai_hinh": r.get("loai_hinh", "Nhà ngõ, hẻm"),
                            "giay_to_phap_ly": r.get("giay_to_phap_ly", np.nan),
                            "tinh_trang_noi_that": r.get("tinh_trang_noi_that", np.nan),
                            "dac_diem": r.get("dac_diem", np.nan),
                        }
                        res = analyze_listing(row_input, custom_weights=norm_w)
                        records.append(res)

                    res_df = pd.DataFrame(records)
                    out_df = work_df.copy()
                    out_df["gia_du_doan_ty"] = np.round(res_df["predicted_price"].values, 2)
                    out_df["composite_score"] = np.round(res_df["composite"].values, 2)
                    cutoff_score = float(np.quantile(out_df["composite_score"], 1.0 - top_k_pct / 100.0))
                    out_df["is_anomaly"] = out_df["composite_score"] >= cutoff_score
                    out_df["anomaly_type"] = np.where(
                        out_df["is_anomaly"], res_df["anomaly_type"].values, "Bình thường"
                    )
                    out_df["ly_do_bat_thuong"] = res_df["reason"].values

                flagged = out_df[out_df["is_anomaly"]].sort_values("composite_score", ascending=False)
                k1, k2, k3 = st.columns(3)
                k1.metric("Tổng số tin kiểm tra", f"{len(out_df):,}")
                k2.metric("Số tin bất thường", f"{len(flagged):,}")
                k3.metric("Ngưỡng Composite Score", f"{cutoff_score:.2f}")

                st.write("### Danh sách tin bị gắn cờ bất thường")
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


st.divider()
st.caption(
    """
    Đây là hệ thống ưu tiên các tin cần kiểm tra, không phải kết luận chắc chắn một tin đăng là sai.
    Dataset không có nhãn anomaly ground-truth.
    """
)
