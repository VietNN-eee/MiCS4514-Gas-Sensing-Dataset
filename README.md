# MiCS-4514 cross-sensor calibration transfer – code and figures

Thư mục này chứa mã Python và hình ảnh dùng trong bản thảo *Cross-Sensor Calibration Transfer for CO, NO₂, and H₂S Identification and Quantification Using Heater-Modulated Dual-Channel MiCS-4514 Sensors*.

## Cấu trúc
```
Data10.11.26/
├── Base_Air_45points_Vo.xlsx, CO_45points_Vo.xlsx, NO2_45points_Vo.xlsx, H2S_45points_Vo.xlsx   (dữ liệu)
├── code/        mã Python
├── results/     kết quả tính toán (JSON/NPZ) dùng để vẽ Hình 7–9, S4 và các bảng
└── figures/     hình PNG (300 dpi)
```

## Chạy lại
```
pip install -r code/requirements.txt
cd code
python run_all.py            # chạy lại toàn bộ phân tích rồi vẽ hình (khoảng 1–2 giờ)
python run_all.py figures    # chỉ vẽ lại hình từ kết quả có sẵn trong results/
```
Đường dẫn mặc định: dữ liệu ở thư mục cha của `code/`. Có thể đổi bằng biến môi trường `ENOSE_DATA_DIR`, `ENOSE_RES_DIR`, `ENOSE_FIG_DIR`.

## Các file mã (đánh số theo bản thảo v8)
| File | Nội dung |
|---|---|
| `core.py` | Đọc dữ liệu (S1–S3, 7 mức nồng độ/khí, 523 chu kỳ), đặc trưng, CV theo khối liên tiếp trong từng điều kiện, giao thức huấn luyện trên một cảm biến – kiểm tra trên cảm biến khác |
| `run_cls.py` | Phân loại chuyển giao: Bảng II; Bảng S6, S8–S12; ma trận nhầm lẫn cho Hình 5(a), S6 |
| `run_fig7b.py` | Độ chính xác theo cảm biến nguồn (Hình 5(b), graphical abstract) |
| `run_r2.py` | Thống kê chỉ từ tập huấn luyện so với transductive trên cùng folds (Bảng S7) |
| `run_reg.py`, `run_samermse.py`, `run_r2c.py` | Định lượng log₁₀ nồng độ: Bảng IV, Hình 7, Bảng S11 (đường chéo), S16–S18 |
| `run_misc.py` | Số điểm lấy mẫu (Hình 6, Bảng S13), thiết kế tập tham chiếu và CORAL/DS (Bảng III, S14, S15), độ nhạy và độ phân giải (Bảng S4, S5) |
| `run_r2b.py` | Nhiễu khí sạch trong từng đoạn, tự tương quan bậc 1 (Bảng S5) |
| `make_figures.py` | Vẽ Hình 1–7, S1–S6 và graphical abstract (300 dpi; đổi bằng biến `ENOSE_DPI`) |
| `run_all.py` | Chạy tất cả |

## Hình (300 dpi)
| File | Hình trong bản thảo v8 |
|---|---|
| `Fig1.png` | Hình 1 – Hệ đo, mạch gia nhiệt/đọc, sơ đồ chia dữ liệu |
| `Fig2.png` | Hình 2 – Quỹ đạo đáp ứng của S1 |
| `Fig3.png` | Hình 3 – Đường chuẩn ΔV theo nồng độ |
| `Fig4.png` | Hình 4 – Biến thiên giữa các cảm biến (khí sạch) |
| `Fig5.png` | Hình 5 – Ma trận nhầm lẫn S1→S2 và độ chính xác theo cảm biến nguồn |
| `Fig6.png` | Hình 6 – Số điểm lấy mẫu, vùng chu kỳ, độ quan trọng RF |
| `Fig7.png` | Hình 7 – Định lượng chuyển giao giữa cảm biến |
| `FigS1.png` | Hình S1 – Chu kỳ khí sạch theo ramp PWM |
| `FigS2.png` | Hình S2 – Đường nền khí sạch và ranh giới các đoạn |
| `FigS3.png`, `FigS4.png` | Hình S3, S4 – Quỹ đạo đáp ứng của S2, S3 |
| `FigS5.png` | Hình S5 – PCA khớp trên S1 |
| `FigS6.png` | Hình S6 – Ma trận nhầm lẫn raw + SVM |
| `Graphical_Abstract.png` | Graphical abstract |

Môi trường đã dùng: Python 3.13, scikit-learn 1.9.1.
