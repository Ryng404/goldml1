# GoldML - Dự đoán giá vàng với Linear Regression và LSTM

Dự án gồm pipeline huấn luyện và sản phẩm web Streamlit. Pipeline chỉ dùng hai thuật toán theo yêu cầu:

- **Hồi quy tuyến tính**: baseline dễ giải thích, có thể bật PCA để giảm chiều dữ liệu.
- **LSTM**: mô hình chính, học chuỗi 60 phiên gần nhất và dự đoán phần trăm thay đổi của phiên kế tiếp.

Pipeline mặc định dùng tối đa **100.000 dòng mới nhất** từ CSV Kaggle. Cách xử lý dữ liệu gồm chuẩn hóa tên cột, parse thời gian, loại dòng lỗi/trùng, kiểm tra OHLC, loại outlier bằng ngưỡng IQR trên log-return và tạo đặc trưng kỹ thuật không nhìn trước tương lai.

## Chạy nhanh

```bash
cd GoldML
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --csv /path/to/gold.csv --max-rows 100000
streamlit run webapp/app.py
```

Sau khi train, các file trong `models/` được web app tự động đọc. Có thể dùng `--pca-components 0` để tắt PCA hoặc đặt số thành phần, ví dụ `12`.

## Dữ liệu Kaggle

CSV cần có các cột tương đương `date/time`, `open`, `high`, `low`, `close`; cột volume là tùy chọn. Bộ đọc tự nhận dạng dấu phẩy, chấm phẩy và tab. Nếu file có hai cột `date` và `time`, chúng sẽ được ghép lại.

## Chạy notebook

Mở `GoldML_Improved.ipynb`. Notebook gọi cùng các hàm trong `pipeline.py`, vì vậy kết quả trong notebook và web app không bị lệch logic.
