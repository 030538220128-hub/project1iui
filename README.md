# House Price Intelligence (Project 1)

Ứng dụng Streamlit định giá bất động sản và phát hiện tin đăng bất thường tại 3 quận TP.HCM (**Bình Thạnh, Gò Vấp, Phú Nhuận**).

## Giới thiệu tính năng

1. **House Price Prediction**: Dự đoán giá bán bất động sản (tỷ VNĐ) và đơn giá (triệu VNĐ/m²) dựa trên 16 đặc trưng vị trí, kích thước, kết cấu và pháp lý sử dụng mô hình Pipeline XGBoost Regressor.
2. **Anomaly Detection**: Phát hiện tin đăng bất thường (quá rẻ, quá đắt, bất thường cấu trúc) dựa trên mô hình lai 4 tín hiệu (Residual-Z, P10-P90, Historical Min/Max, Isolation Forest) kèm nhãn giải thích XAI minh bạch.

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
│   └── 2_Anomaly_Detection.py
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
