# House Price Intelligence (Project 1 — Nhà Tốt)

Ứng dụng Streamlit định giá bất động sản và phát hiện tin đăng bất thường tại 3 quận TP.HCM (**Bình Thạnh, Gò Vấp, Phú Nhuận**).

- 🌐 **Live Demo:** [https://project1iui-74ixm6xxkjzeutpe7qbvt5.streamlit.app/](https://project1iui-74ixm6xxkjzeutpe7qbvt5.streamlit.app/)
- 🎓 **Khóa học:** Đồ án Tốt nghiệp Data Science (DL07) — Trung tâm Tin học ĐH Khoa học Tự nhiên TP.HCM
- 👥 **Nhóm học viên thực hiện:**
  - **Lê Văn Toàn** — `toanlevan2808@gmail.com`
  - **Chung Tiến Quang** — `Dongtachungtienquang@gmail.com`

## Giới thiệu tính năng

1. **House Price Prediction (`1_House_Price_Prediction.py`)**: Dự đoán giá bán bất động sản (tỷ VNĐ) và đơn giá (triệu VNĐ/m²) dựa trên **19 đặc trưng** (vị trí, diện tích, kết cấu, pháp lý và 6 đặc trưng tham chiếu giá thị trường `gia_khuvuc_*`) sử dụng mô hình Pipeline XGBoost Regressor. Hỗ trợ lọc động Phường theo Quận và dự đoán hàng loạt qua file CSV.
2. **Anomaly Detection (`2_Anomaly_Detection.py`)**: Phát hiện tin đăng bất thường (quá rẻ, quá đắt, bất thường cấu trúc) dựa trên mô hình lai 4 tín hiệu (`Residual-Z`, `P10-P90`, `Historical Min/Max`, `Isolation Forest`) kèm nhãn giải thích XAI minh bạch. Hỗ trợ tải file CSV phát hiện bất thường hàng loạt với thanh trượt tinh chỉnh trọng số $w_1..w_4$ và ngưỡng `Top-k%`.
3. **Market Overview (`3_Market_Overview.py`)**: Bảng điều khiển tổng quan thị trường BĐS tại 3 quận (KPIs, biểu đồ đơn giá trung vị theo Quận/Phường, tỷ lệ pháp lý và trình duyệt dữ liệu).

## Cấu trúc thư mục

```text
project1iui/
├── data/
│   ├── quan-binh-thanh.csv
│   ├── quan-go-vap.csv
│   ├── quan-phu-nhuan.csv
│   └── house_prices_clean.csv
├── models/
│   ├── project1_price_model.pkl
│   └── project1_anomaly_model.pkl
├── pages/
│   ├── 1_House_Price_Prediction.py
│   ├── 2_Anomaly_Detection.py
│   └── 3_Market_Overview.py
├── tests/
│   └── test_page_navigation.py
├── app.py
├── README.md
└── requirements.txt
```

## Chạy Local

Sử dụng **Python 3.11** hoặc **Python 3.12** (phiên bản tương thích với các mô hình đã huấn luyện):

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
source .venv/bin/activate       # Trên Linux/macOS
# Hoặc trên Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

# 2. Cài đặt các thư viện phụ thuộc
python -m pip install -r requirements.txt

# 3. Khởi chạy ứng dụng Streamlit
python -m streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại: `http://localhost:8501`.

## Triển khai lên Streamlit Community Cloud

- Chọn repository, branch và main file: `app.py`.
- Trong mục **Advanced settings** khi deploy, chọn **Python 3.11**.
- Giữ nguyên `requirements.txt`, thư mục `data/` và `models/` trong repository.
- **Lưu ý quan trọng về mô hình**:
  - Mô hình `project1_price_model.pkl` và `project1_anomaly_model.pkl` yêu cầu phiên bản `scikit-learn==1.6.1` và `xgboost==3.0.5`.
  - Không tự ý nâng cấp phiên bản `xgboost` lên 3.4+ mà chưa tái huấn luyện và xuất lại mô hình, để tránh lỗi `XGBoostError: input stream corrupted`.

## Kiểm thử tự động (Unit Tests)

```bash
python -m unittest discover -s tests -v
```
