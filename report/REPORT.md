# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** Không ghi nhận trong output hiện có

**Runtime Colab:** Không ghi nhận trong output hiện có

**Python / PyTorch / Ultralytics:** Python và PyTorch không ghi nhận; Ultralytics `8.4.145` (theo JSON)

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `classification_predictions.json` và `classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915; `taxonomy_name`= "ImageNet-1K"

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

Record này nói về cả bức ảnh vì tác vụ là `image_classification`: model xếp hạng các lớp cho toàn ảnh, không cung cấp box hoặc mask để chỉ ra một vật thể riêng. Nhãn `cab` không cho biết vị trí hoặc số lượng taxi trong ảnh.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

Danh sách tên lớp đi kèm taxonomy/checkpoint. Model `yolo11n-cls.pt` dự đoán trong hệ lớp ImageNet-1K đã có sẵn, không tự nghĩ ra tên lớp mới khi chạy dự đoán.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

`class_id` dùng để ánh xạ nhãn trong hệ thống, `class_name` giúp người đọc hiểu nhãn, còn `taxonomy_name` xác định hệ lớp đang dùng. Một ID ở hai taxonomy có thể mang nghĩa khác nhau; ghi đủ ba trường giúp tránh nhầm lẫn và truy vết chính xác nhãn.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

Guideline cần quy định đối tượng/chủ đề mục tiêu, tiêu chí chọn lớp chính, thứ tự ưu tiên khi có nhiều lớp phù hợp và cách xử lý trường hợp không rõ hoặc ngoài taxonomy. Người gán nhãn phải chọn một lớp theo cùng quy tắc; nếu không thể quyết định thì chuyển reviewer/Lab Coach, không tự chọn theo score cao nhất.

6. Vì sao không được lấy `score` của model làm ground truth?

`score` là điểm model gán cho dự đoán, không phải bằng chứng rằng nhãn đúng. Model vẫn có thể dự đoán sai với score cao. Ví dụ `cab` đạt `0.510915`, còn `minibus` hạng 2 đạt `0.164284`; các điểm này chỉ thể hiện đánh giá của model. Ground truth phải được xác định bằng cách đối chiếu ảnh với guideline và kiểm tra chất lượng.

## 2. Phát hiện vật thể

**Mở file:** `detection_predictions.json` và `detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= "bowl" ; `score`= 0.719063 ; `bbox_xyxy`=[32.65, 342.12, 100.16, 384.93] ; `bbox_width`= 67.51 ; `bbox_height`= 42.81

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

Box có góc trên-trái `(32.65, 342.12)` và góc dưới-phải `(100.16, 384.93)`, nằm ở vùng dưới bên trái ảnh. Chiều rộng là `100.16 − 32.65 = 67.51` pixel; chiều cao là `384.93 − 342.12 = 42.81` pixel.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

Ở ngưỡng `0.35` có 11 prediction; ở ngưỡng `0.60` có 6 prediction.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

Khi hạ ngưỡng, reviewer phải kiểm tra thêm các box được giữ lại: xác nhận vật có thật không, lớp có đúng không, box có trùng hoặc lệch không, rồi sửa hoặc loại prediction sai. Reviewer vẫn phải tìm vật bị bỏ sót vì nhiều box hơn không bảo đảm nhãn đầy đủ hoặc chính xác hơn.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Quy tắc box đề xuất: Với mỗi vật thể thuộc phạm vi và nhận diện được, vẽ một hình chữ nhật nhỏ nhất bao kín phần nhìn thấy của vật, giữ tọa độ trong giới hạn ảnh và không suy đoán phần bị che hoặc nằm ngoài khung hình.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Ở góc dưới bên trái ảnh, phần cánh tay/bàn tay của một người bị khung hình cắt ở mép trái. JSON ghi nhận prediction `person` với score `0.610931` và box `[0.12, 263.18, 61.55, 310.78]`. Tôi chọn áp dụng guideline vẽ box theo phần nhìn thấy và không kéo box ra ngoài ảnh; nếu không xác định được vật thể thuộc lớp nào thì escalation cho Lab Coach.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `segmentation_predictions.json` và `segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= "kitchen-001" ; `class_name`= "person" ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`= [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]] , đơn vị pixel.

2. So với box, polygon cho thêm thông tin gì về hình dáng vật thể?

Box chỉ xác định vùng chữ nhật bao quanh vật thể, có thể chứa cả nền và vật khác. Polygon mô tả đường bao, giúp thể hiện hình dạng và vùng thuộc vật thể chi tiết hơn. Số điểm polygon lớn không tự chứng minh mask chính xác.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

`instance_id` dùng để phân biệt từng đối tượng trong kết quả xuất. Ví dụ `kitchen-001` và `kitchen-009` đều có lớp `person` nhưng là hai instance dự đoán khác nhau. Class ID biểu thị loại đối tượng, còn instance ID biểu thị đối tượng cụ thể. Instance ID không phải tracking ID vì không bảo đảm liên kết cùng một đối tượng qua các khung hình hoặc các lần chạy khác nhau.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

Quy tắc vẽ biên mask đề xuất: Vẽ sát đường biên phần nhìn thấy của từng vật thể, loại nền và phần thuộc vật khác, không suy đoán phần bị che; nếu không phân biệt được biên thì chuyển reviewer/Lab Coach để thống nhất cách xử lý.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Ở vùng các chiếc bát trên mặt bàn phía dưới bên trái, hai prediction `kitchen-006` và `kitchen-002` có box giao nhau: `[0.11, 302.85, 85.7, 370.15]` và `[31.44, 343.77, 100.89, 385.55]`. Vùng này cần kiểm tra đường biên mask giữa các vật gần nhau; box giao nhau tự nó không chứng minh hai vật chạm hoặc che nhau. Nếu không nhìn rõ biên trên ảnh, tôi chọn escalation cho Lab Coach; nếu biên rõ và guideline đã quy định, tách mask theo từng instance và chỉ vẽ phần nhìn thấy.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)              | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                              | Annotator làm gì?                                                                   | Reviewer xem gì?                                                                           |
| --------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Phân loại ảnh         | Một nhãn lớp cho toàn ảnh theo taxonomy và guideline.      | `traffic`: `cab` hạng 1, score `0.510915`; `minibus` hạng 2, score `0.164284`. Chưa thể xác nhận lớp đúng chỉ từ thứ hạng và score.             | Xem toàn ảnh, chọn lớp chính theo guideline, chuyển ca mơ hồ.                       | Kiểm tra lớp chính, ánh xạ ID–tên–taxonomy và sự nhất quán giữa người gán.                 |
| Phát hiện vật thể     | Một class ID và box cho mỗi vật thể đủ điều kiện gán nhãn. | `kitchen`: phần tay/cánh tay người bị cắt ở mép trái; box `person` `[0.12, 263.18, 61.55, 310.78]` cần tuân theo quy tắc chỉ vẽ phần nhìn thấy. | Xác định từng vật, vẽ/sửa box theo guideline, bổ sung vật bị bỏ sót.                | Kiểm tra lớp, độ khít box, box trùng, vật thiếu và cách xử lý vật bị che/cắt mép.          |
| Instance segmentation | Một lớp và mask/polygon riêng cho mỗi instance.            | Box của `kitchen-006` và `kitchen-002` giao nhau, cần kiểm tra biên và khả năng che khuất; chưa đủ dữ liệu để kết luận mask sai.                | Tách từng instance, vẽ/sửa biên theo phần nhìn thấy, chuyển vùng không rõ để xử lý. | Kiểm tra tách/gộp instance, lấn nền, vùng bị thiếu và sự nhất quán của biên với guideline. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Chỉ sử dụng dữ liệu được phép trong phạm vi lab, giữ thông tin nguồn và giấy phép ảnh theo `IMAGE_ATTRIBUTION.md`, không đưa họ tên, MSSV, email, số điện thoại hoặc dữ liệu nhạy cảm vào báo cáo/output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi, tôi sẽ dừng xử lý và chia sẻ dữ liệu đó, báo Lab Coach hoặc giảng viên phụ trách qua kênh được chỉ định, rồi chờ hướng dẫn trước khi tiếp tục.

## 6. Checklist trước khi nộp link repo

Các file output hiện nằm trực tiếp trong `report/`. Sau khi upload, kiểm tra chúng trong thư mục đó trên GitHub:

- [x] `REPORT.md`
- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `classification_top5.png`
- [x] `detection_predictions.png`
- [x] `segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
