# GoldML - Dự đoán giá vàng với LSTM và mô hình so sánh

Dự án gồm pipeline huấn luyện và dashboard Streamlit theo dõi kết quả dự đoán. Pipeline gồm:

- **Hồi quy tuyến tính**: baseline hồi quy trên 22 đặc trưng.
- **K-NN**: phân loại xu hướng tăng/giảm trên 22 đặc trưng, không dùng PCA.
- **LSTM**: mô hình chuỗi chính, học 60 phiên gần nhất và dự đoán phần trăm thay đổi của phiên kế tiếp. Huấn luyện 5 cấu hình để so sánh: 22 đặc trưng gốc và PCA 12, 6, 3 hoặc 1 thành phần.

Pipeline mặc định dùng tối đa **100.000 dòng mới nhất** từ CSV Kaggle. Cách xử lý dữ liệu gồm chuẩn hóa tên cột, ép thời gian về datetime, sắp xếp tăng dần, loại dòng lỗi/trùng, kiểm tra OHLC và báo cáo các khoảng trống thời gian lớn. Không nội suy giá, không loại nến phẳng hợp lệ và không lọc outlier bằng thống kê toàn chuỗi; nhờ vậy không tạo khoảng trống nhân tạo hoặc dùng thông tin từ tập kiểm tra để làm sạch dữ liệu.

Dữ liệu được chia theo thời gian thành **70% Train / 15% Validation / 15% Test**. Scaler và PCA chỉ fit trên Train; LSTM dùng Validation theo thời gian để early stopping và không shuffle. Chỉ số trên Test gồm RMSE, MAE, MAPE và so sánh với **Naive Baseline** (dùng giá đóng cửa phiên trước). Các file dự đoán lưu cả giá thực theo ngày đích, nhãn phân đoạn và dự đoán Test để dashboard hiển thị đúng ranh giới tập.

## Chạy nhanh

```bash
cd GoldML
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --csv /path/to/gold.csv --max-rows 100000
streamlit run webapp/app.py
```

Sau khi train, dashboard tự động đọc các kết quả trong `models/`. Ở thanh bên, chọn Linear Regression hoặc một cấu hình LSTM để xem biểu đồ giá thực tế/dự đoán, Naive Baseline và chỉ số tương ứng. Biểu đồ đánh dấu mốc Validation/Test, tô vùng khoảng trống lớn (không nối đường qua khoảng trống) và có chế độ zoom 7 ngày. Mặc định chạy LSTM với PCA 12, 6, 3 và 1 thành phần:

```bash
python train.py --csv /path/to/gold.csv --pca-components 12 6 3 1
```

Để thay bộ dữ liệu, truyền đường dẫn tới CSV mới khi chạy `train.py`; không cần đổi tên hay đưa CSV vào Git. Ví dụ:

```bash
python train.py --csv /path/to/new_gold.csv --max-rows 100000
```

Lệnh này làm sạch và dùng tối đa 100.000 dòng hợp lệ mới nhất, sau đó tạo lại các kết quả trong `models/`. CSV cần có cột ngày/giờ và OHLC (`open`, `high`, `low`, `close`); volume là tùy chọn.

### TensorFlow trên Windows

Huấn luyện LSTM cần TensorFlow nạp được thư viện native runtime. Nếu Windows Application Control chặn DLL của TensorFlow, lệnh train mặc định sẽ dừng và báo lỗi rõ ràng thay vì giả vờ chạy LSTM. Hãy huấn luyện trong môi trường Linux/WSL2 được phép hoặc nhờ quản trị viên cho phép TensorFlow. Nếu chỉ muốn chạy baseline Linear Regression, hãy bật rõ chế độ dự phòng:

```bash
python train.py --csv /path/to/gold.csv --allow-linear-fallback
```

## Dữ liệu Kaggle

CSV cần có các cột tương đương `date/time`, `open`, `high`, `low`, `close`; cột volume là tùy chọn. Bộ đọc tự nhận dạng dấu phẩy, chấm phẩy và tab. Nếu file có hai cột `date` và `time`, chúng sẽ được ghép lại.

## Chạy notebook

Mở `GoldML_Improved.ipynb`. Notebook gọi cùng các hàm trong `pipeline.py`, vì vậy kết quả trong notebook và web app không bị lệch logic.
