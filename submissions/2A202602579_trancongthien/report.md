# Báo cáo kết quả Lab Day 2: Phân loại cỏ dại (DeepWeeds)

## 1. Tóm tắt
Bài thí nghiệm tập trung xây dựng mô hình học sâu để phân loại 9 loại cỏ dại trên tập dữ liệu DeepWeeds. Thông qua việc thử nghiệm nhiều kiến trúc mạng (ResNet50, ConvNeXt, EfficientNet...), tinh chỉnh công thức huấn luyện (Augmentation, Focal Loss) và áp dụng các kỹ thuật suy luận (TTA, Temperature Scaling), cấu hình tốt nhất đạt được là **[Tên mô hình]** kết hợp với **[Công thức tốt nhất]**. Mô hình đạt độ chính xác Macro-F1 trên tập test là **[X.XXXX] ± [Y.YYYY]** với 3 seeds. Dựa trên sự cân bằng giữa độ chính xác và độ trễ, cấu hình **[Cấu hình Y]** được đề xuất để triển khai thực tế trên robot.

## 2. Dữ liệu và thiết lập
- **Dataset:** Tập dữ liệu ảnh DeepWeeds với 17.509 ảnh độ phân giải 256x256. Dữ liệu mất cân bằng nghiêm trọng (lớp Negatives chiếm hơn phân nửa).
- **Phân chia:** Sử dụng 1 fold duy nhất (train_subset0, val_subset0, test_subset0) tương ứng với 60/20/20.
- **Công thức nền:** Optimizer AdamW, LR backbone 1e-4, LR head 1e-3, Weight decay 0.05, Lịch Cosine Annealing, Batch size 64, AMP = True, Số epochs = 12.
- **Môi trường:** Python 3.x, PyTorch [Version], Timm [Version], [Kaggle/Colab GPU].
- **Seed khởi tạo:** Sử dụng random seed = 0 cho các thử nghiệm sàng lọc. Chung kết chạy 3 seeds (0, 1, 2).

## 3. Kết quả so sánh backbone (Bước 1)
Với công thức nền, kết quả so sánh 5 kiến trúc backbone như sau (chi tiết xem tại `results.xlsx` sheet `Backbones`):
* [Backbone 1]: Macro-F1 = ..., Params = ..., GMACs = ...
* [Backbone 2]: Macro-F1 = ..., Params = ..., GMACs = ...
* [Backbone 3]: Macro-F1 = ..., Params = ..., GMACs = ...
* [Backbone 4]: Macro-F1 = ..., Params = ..., GMACs = ...
* [Backbone 5]: Macro-F1 = ..., Params = ..., GMACs = ...

**Nhận xét:** Kiến trúc mạng **[Backbone X]** mang lại kết quả hội tụ F1 tốt nhất (hoặc tốt nhì nhưng có độ trễ cực thấp). Thứ hạng này có/không tương đồng với tập ImageNet do...

## 4. Kết quả công thức huấn luyện (Bước 2)
Từ backbone tốt nhất, các trục tham số được điều chỉnh:
- **Trục A (Augmentation):** TrivialAugment/RandAugment làm tăng/giảm Macro-F1 lên **[X]** điểm so với cắt lật cơ bản.
- **Trục B (Mixup/CutMix):** Sử dụng [Mixup/Cutmix] giúp các lớp thiểu số... 
- **Trục C (Loss Function):** Focal Loss (gamma=2) giúp mô hình tập trung vào các lớp bị sai nhiều (Chinee Apple, Snake Weed), làm tăng mạnh điểm F1 thay vì chỉ tập trung vào lớp Negatives. Cải thiện được **[Y]** điểm F1.
- **Trục D (EMA/Optimizer):** Kết hợp EMA Weight hỗ trợ mô hình trơn tru hơn...

**Nhận xét:** Việc áp dụng **[Loss / Aug]** có tác dụng rõ rệt nhất. Khi kết hợp tất cả các yếu tố tốt nhất lại, hiệu ứng có xu hướng cộng dồn / triệt tiêu lẫn nhau vì...

## 5. Kết quả suy luận (Bước 3)
Chỉ thực hiện trên tập Validation với các kỹ thuật (không huấn luyện lại):
1. **TTA (Lật ngang / Multi-crop):** Cải thiện **[Z]** điểm F1 nhưng đánh đổi thời gian suy luận tăng gấp **[K]** lần.
2. **Temperature Scaling:** Không làm đổi F1 nhưng giúp ECE giảm từ **[E1]** xuống **[E2]**, dự đoán có xác suất tự tin chuẩn xác hơn.
3. **Độ trễ (Latency):** Việc gộp BatchNorm vào Conv (fusion) giúp p95 latency giảm được **[L]** ms. Mô hình [Model] FP16 chỉ mất khoảng **[M]** ms để xử lý một ảnh.

## 6. Cấu hình tốt nhất & Phân tích lỗi (Bước 4)
Cấu hình lọt vào vòng chung kết (`exp_id = F01`):
- Backbone: **[Backbone Z]**
- Training: **[Focal Loss + RandAugment + EMA...]**
- Inference: **[Fusion + Temperature Scaling + TTA 2-views]**

Kết quả qua 3 seeds (0, 1, 2) trên tập Test:
- **Macro-F1 (Test):** `[Mean_F1] ± [Std_F1]`
- **Top-1 Acc (Test):** `[Mean_Acc] ± [Std_Acc]`

**Phân tích lỗi (Confusion Matrix):** Dựa vào ma trận nhầm lẫn, mô hình vẫn dễ dự đoán nhầm lớp **[A]** sang **[B]**. Nguyên nhân có thể do môi trường ánh sáng giống nhau và lá nhỏ bị nhoè khi resize về 256x256.

## 7. Kết luận và khuyến nghị
- Cấu hình tốt nhất cải thiện **[X]** điểm so với mốc nền, vượt qua hoàn toàn ngưỡng nhiễu của hệ thống (std).
- Yếu tố đóng góp nhiều nhất vào hiệu suất phân loại cỏ dại là **[Kiến trúc backbone / Loss function / TTA]**.
- **Khuyến nghị triển khai:** Nếu chạy trên hệ thống đám mây (Cloud) không giới hạn tài nguyên, sử dụng cấu hình chung kết (có TTA) để tối đa hoá chính xác. Nếu gắn trực tiếp lên máy kéo/robot với ngân sách độ trễ < 50ms, nên bỏ TTA và đổi backbone thành **[EfficientNet/ConvNeXt-T]** để đạt tốc độ FPS cao nhưng vẫn giữ được macro-F1 trên 90%.

## 8. Hạn chế
- Số hạt giống (seeds) mới là 3, và chỉ dùng 1 fold thay vì 5-folds Cross-Validation.
- Chưa thử nghiệm Distillation hay thích ứng miền (Domain Adaptation) trên điều kiện ánh sáng thực tế.

## 9. Phụ lục
- Link Notebook: [Chèn link Kaggle/Colab]
- Lệnh chạy chính: `python code/train.py --set exp_id=F01 ...`
