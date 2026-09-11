# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/9/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.10.14, PyTorch 2.6.0, Ultralytics 8.4.179

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không


## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 468, cab, 1, 0.510915, ImageNet-1K
- **Record này mô tả toàn ảnh như thế nào?** Nó đại diện cho đối tượng chính, chiếm diện tích lớn nhất hoặc nổi bật nhất mang tính tổng quát cho toàn bộ khung hình.
- **Ai định nghĩa class list mà checkpoint có thể dự đoán?** Danh sách này được định nghĩa bởi người xây dựng và huấn luyện mô hình, thường dựa trên các tập dữ liệu lớn và phổ biến như ImageNet, COCO hoặc Pascal VOC, phản ánh các lớp đối tượng mà mô hình đã được huấn luyện để nhận biết.
- **Vì sao cần giữ cả ID, tên lớp và tên taxonomy?** Cần giữ cả ID, tên lớp và tên taxonomy để đảm bảo tính đầy đủ và rõ ràng của thông tin phân loại. ID là mã định danh duy nhất, tên lớp cung cấp ngữ nghĩa con người có thể đọc hiểu, và tên taxonomy giúp phân loại hierarchical và liên kết với các bộ dữ liệu/tiêu chuẩn đã biết, từ đó facilitates việc kiểm tra và sử dụng mô hình hiệu quả hơn.
- **Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?** Nếu ảnh có nhiều chủ thể, guideline cần quy định rõ cách xử lý, ví dụ như chỉ gắn nhãn cho chủ thể chính chiếm diện tích lớn nhất, hoặc gắn nhãn cho tất cả các chủ thể có kích thước tối thiểu nhất định.
- **Vì sao model score không phải ground truth?** Model score chỉ là xác suất dự đoán của mô hình dựa trên dữ liệu huấn luyện và không phản ánh thực tế khách quan.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- **Một record** (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): bus, 0.912557, [93.17, 187.95, 223.01, 320.91], 129.84, 132.96
- **Diễn giải vị trí box bằng lời:** Tọa độ bbox_xyxy xác định điểm góc trên-bên trái $(x_{min}, y_{min})$ và góc dưới-bên phải $(x_{max}, y_{max})$ của một hình chữ nhật bao quanh vật thể. Trong ví dụ này, box bao phủ một khu vực từ pixel $(93.17, 187.95)$ đến $(223.01, 320.91)$ với chiều rộng $129.84$ và chiều cao $132.96$.
- **So sánh số prediction ở hai threshold:** Với threshold=0.5, có 15 prediction. Với threshold=0.3, có 32 prediction.
- **Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?** Với threshold=0.5, có 15 prediction, độ bao phủ thấp nhưng khối lượng reviewer ít. Với threshold=0.3, có 32 prediction, độ bao phủ cao nhưng khối lượng reviewer nhiều hơn.
- **Đề xuất một quy tắc box chặt:** Box phải ôm sát nhất có thể phần hiển thị của vật thể (các biên viền ngoài cùng), không tạo ra khoảng trống thừa quá lớn và không cắt lẹm vào vật thể.
- **Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?** Nếu một object bị che khuất hoặc cắt mép nghiêm trọng (ví dụ: dưới 50% diện tích hiển thị), cần có guideline cụ thể hoặc sự quyết định từ reviewer để xác định có nên giữ lại prediction đó hay không.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- **Một record** (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): traffic-001, bus, 0.925745, 120 điểm, [148, 189, ...]
- **Polygon bổ sung chi tiết gì so với box?** Polygon bổ sung chi tiết về hình dạng thực tế của vật thể, cho phép phân loại chính xác hơn các vật thể có hình dạng không đều.
- **`instance_id` dùng để làm gì và không phải loại ID nào?** Instance_id dùng để phân biệt các thể hiện khác nhau của cùng một lớp vật thể trong một bức ảnh. Không phải loại ID nào là image-level ID, class-level ID, image-level ID là mã định danh duy nhất của một bức ảnh trong toàn bộ tập dữ liệu, class-level ID là mã định danh duy nhất của một lớp trong toàn bộ tập dữ liệu.
- **Đề xuất một quy tắc biên mask:** Mask nên bao phủ toàn bộ vật thể, không cắt lẹm vào vật thể và không tạo ra khoảng trống thừa quá lớn.
- **Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?** Nếu một object bị che khuất hoặc cắt mép nghiêm trọng (ví dụ: dưới 50% diện tích hiển thị), cần có guideline cụ thể hoặc sự quyết định từ reviewer để xác định có nên giữ lại prediction đó hay không.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | 1 Nhãn text/ID cấp toàn ảnh | Ảnh có nhiều chủ thể bằng nhau không rõ chính/phụ | Áp dụng quy tắc ưu tiên chủ thể trung tâm/lớn nhất | Đánh giá nhãn được chọn có phản ánh đúng đối tượng quan trọng nhất không |
| **Phát hiện vật thể** | Bounding box (x,y,w,h) + Nhãn | Box bao quá rộng, lẹm vật thể hoặc vẽ chồng lấn | Kéo đỉnh tọa độ ôm sát mép nhìn thấy thực tế của vật thể | Box có sát viền chưa, có bỏ sót vật nào không, nhãn dán đúng không |
| **Instance segmentation** | Polygon mask (mảng tọa độ) + Nhãn | Đường bao dính nền, bị gãy khúc, bị nhầm vùng bóng đổ | Chỉnh sửa từng điểm (points) chạy dọc theo biên giới vật thể | Đường biên mask có mịn không, có bị lẹm/dư so với mép pixel không |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không lưu trữ ảnh/dữ liệu cá nhân ra bên ngoài môi trường làm việc đã được cấp phép, sử dụng các biện pháp mã hóa cần thiết để bảo vệ thông tin.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Reviewer hoặc người quản lý khóa học.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
