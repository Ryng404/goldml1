# GoldML - Dự đoán giá vàng với LSTM và mô hình so sánh

Dự án gồm pipeline huấn luyện và dashboard Streamlit theo dõi kết quả dự đoán. Pipeline gồm:

- **Hồi quy tuyến tính**: baseline hồi quy trên 22 đặc trưng.
- **K-NN**: phân loại xu hướng tăng/giảm trên 22 đặc trưng, không dùng PCA.
- **LSTM**: mô hình chuỗi chính, học 60 phiên gần nhất và dự đoán phần trăm thay đổi của phiên kế tiếp. Huấn luyện 5 cấu hình để so sánh: 22 đặc trưng gốc và PCA 12, 6, 3 hoặc 1 thành phần.

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

Sau khi train, dashboard tự động đọc các kết quả trong `models/`. Ở thanh bên, chọn một trong các cấu hình LSTM để xem biểu đồ và chỉ số tương ứng. Mặc định chạy LSTM với PCA 12, 6, 3 và 1 thành phần:

```bash
python train.py --csv /path/to/gold.csv --pca-components 12 6 3 1
```

### TensorFlow trên Windows

Huấn luyện LSTM cần TensorFlow nạp được thư viện native runtime. Nếu Windows Application Control chặn DLL của TensorFlow, lệnh train mặc định sẽ dừng và báo lỗi rõ ràng thay vì giả vờ chạy LSTM. Hãy huấn luyện trong môi trường Linux/WSL2 được phép hoặc nhờ quản trị viên cho phép TensorFlow. Nếu chỉ muốn chạy baseline Linear Regression, hãy bật rõ chế độ dự phòng:

```bash
python train.py --csv /path/to/gold.csv --allow-linear-fallback
```

## Dữ liệu Kaggle

CSV cần có các cột tương đương `date/time`, `open`, `high`, `low`, `close`; cột volume là tùy chọn. Bộ đọc tự nhận dạng dấu phẩy, chấm phẩy và tab. Nếu file có hai cột `date` và `time`, chúng sẽ được ghép lại.

## Chạy notebook

Mở `GoldML_Improved.ipynb`. Notebook gọi cùng các hàm trong `pipeline.py`, vì vậy kết quả trong notebook và web app không bị lệch logic.
