# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Pyhthon 3.13.16 / pytorch 2.11.0 + Cu 130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
-> Record này nói về CẢ BỨC ẢNH vì không có bounding box để định vị vật thể riêng lẻ và mô hình đang dự đoán để phân loại ảnh cho toàn bộ khung hình
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
-> Danh sách tên lớp đi kèm taxonomy / checkpoint - do được định nghĩa trước khi huấn luyện
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần ghi đủ cả 3 thông tin để đảm bảo tính chuẩn hóa, khả năng truy vết và tái lập kết quả:   
-> class_id: Máy tính và thuật toán lưu trữ, tính toán ma trận qua số nguyên nhằm tối ưu hiệu năng và tránh sai sót do lỗi chính tả hoặc định dạng chuỗi.   
-> class_name: Thể hiện bằng ngôn ngữ tự nhiên để con người đọc hiểu trực tiếp mà không cần tra cứu bảng mã ngược.
-> taxonomy_name: Cùng một từ hoặc cùng một ID số có thể mang ý nghĩa khác nhau giữa các bộ dữ liệu. Ghi rõ taxonomy giúp xác định chuẩn xác hệ phân loại đang sử dụng.   
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần đưa ra các tiêu chí ưu tiên nhất quán để người gán nhãn không chọn theo cảm tính. Cụ thể thì tài liệu hướng dẫn phải quy định rõ việc ưu tiên vật thể chiếm diện tích lớn nhất trong khung hình, nằm ở tiền cảnh hoặc trung tâm ảnh và là tiêu điểm chính của bức ảnh. Ngoài ra, cần thiết lập thứ tự phân cấp ưu tiên cụ thể giữa các nhóm đối tượng khi chúng xuất hiện ngang hàng.
6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ là xác suất dự đoán (mô hình có thể tự tin nhưng vẫn đoán sai) còn ground truth phải là sự thật khách quan do con người kiểm định để làm chuẩn đánh giá mô hình.


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`= 0.912624 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
-> Định dạng xyxy biểu diễn tọa độ theo thứ tự [x_{min}, y_{min}, x_{max}, y_{max}]. Như vậy góc trên-bên trái của hộp nằm tại tọa độ x = 385.33 pixel và y = 69.24 pixel. Góc dưới-bên phải của hộp nằm tại tọa độ x = 498.92 pixel và y = 348.92 pixel. Hộp này có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
-> Tại ngưỡng 0.20 có 17 prediction (vật thể), còn tại ngưỡng 0.60 có 6 prediction (vật thể). 
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
-> Khi hạ ngưỡng tin cậy xuống thấp, mô hình sẽ giữ lại rất nhiều dự đoán thừa thãi và không chắc chắn. Người reviewer sẽ phải tốn thêm nhiều thời gian và công sức để rà soát thủ công.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
-> Bounding box phải được vẽ vừa khít tuyệt đối với các biên ngoài cùng của vật thể, không để thừa khoảng trống của hậu cảnh quá 2-3 và không được cắt lẹm vào bất kỳ phần nào thuộc cơ thể hay chi tiết của đối tượng.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
-> Trên ảnh kitchen, đối tượng bị cắt mép và che khuất rất rõ là bàn tay người góc trái gắn nhãn person 0.61. Ngoài ra, chiếc lò bên trái (oven 0.69) cũng bị mép ảnh bên trái cắt mất một nửa thân. Đối với trường hợp này, mình nên hỏi Lab Coach thay vì tự ý viết thành quy tắc trong guideline. Lý do là vì tiêu chí xử lý vật thể bị cắt mép hay che khuất phụ thuộc chặt chẽ vào yêu cầu bài toán của dự án. Chẳng hạn như quy định tỷ lệ nhìn thấy tối thiểu. Đây là quyết định mang tính chính sách dữ liệu nên cần được phê duyệt thống nhất trước khi phổ biến thành quy tắc chung.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-001" ; `class_name`="person" ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`= [446.0,70.0],[445.0,71.0],[444.0,71.0],[443.0,72.0],[442.0,72.0].
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
-> Box là dạng hình hộp chỉ giới hạn phạm vi không gian ngoài cùng nên chứa rất nhiều điểm ảnh nền thừa thãi, đồng thời làm mất đi đường nét thực tế của đối tượng. Trong khi đó, polygon bao gồm chuỗi tọa độ viền khép kín chạy sát theo từng đường cong, nếp gấp quần áo và tư thế cơ thể, từ đó cung cấp chính xác hình dạng thực tế (geometry), diện tích bề mặt thực của vật thể đến từng điểm ảnh và tách biệt hoàn toàn vật thể khỏi hậu cảnh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
-> Chỉ số instance_id dùng để phân biệt và định danh duy nhất từng cá thể vật thể riêng biệt trong cùng một bức ảnh tĩnh, chẳng hạn như gắn mã person-001 cho người này và person-002 cho người khác. Còn class ID là dùng để định danh nhóm vật thể cùng loại như tất cả person trong ảnh. Tracking ID là mã định danh xuyên suốt dùng trong video để theo dõi cùng một đối tượng di chuyển qua nhiều khung hình liên tiếp.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
-> Đường biên polygon phải bám khít chính xác vào viền ngoài thực tế của vật thể với sai số không vượt quá 1-2 pixel, tuyệt đối không bao gồm pixel của nền phía sau hoặc các vật thể lân cận và các đoạn tiếp giáp gấp khúc phức tạp phải đặt đủ mật độ điểm nút để tái tạo trung thực hình dáng tự nhiên của đối tượng
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
-> Đối với trường hợp này, mình sẽ dùng guideline nếu tài liệu đã có sẵn quy tắc xử lý biên mờ hoặc ranh giới phân tách chuẩn. Tuy nhiên, nếu hai vật thể bị mất hẳn tương phản màu sắc đến mức không thể xác định điểm chia ranh giới bằng mắt thường, việc escalation là cần thiết để xin quy chuẩn gán nhãn thống nhất, tránh việc mỗi người gán nhãn phán đoán theo cảm tính dẫn đến sai lệch dữ liệu huấn luyện.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn lớp duy nhất gán cho toàn bộ ảnh | Ảnh có nhiều đối tượng cùng xuất hiện, khó chọn xem nhãn nào đại diện chính cho toàn cảnh |Đọc kỹ guideline ưu tiên, quan sát toàn ảnh và chọn duy nhất một nhãn phân loại phù hợp nhất  | Kiểm tra xem nhãn được gán có đúng với quy tắc ưu tiên trong guideline hay không |
| Phát hiện vật thể | Danh sách các box kèm nhãn cho từng đối tượng trong ảnh | Vật thể bị cắt cụt ở rìa ảnh như hộp person 0.61, hay hộp oven 0.69|Dùng công cụ vẽ các bounding box ôm sát từng vật thể, gán đúng tên lớp cho từng hộp theo quy chuẩn| Rà soát độ khít của các hộp, kiểm tra vật thể bị bỏ sót hoặc hộp vẽ sai/thừa|
| Instance segmentation | Mặt nạ pixel khép kín kèm instance_id và class_id riêng| Ranh giới mờ và tiệp màu ở các vùng tiếp giáp khó tách biệt | Dùng công cụ polygon chấm nhiều điểm nút nối tiếp nhau ôm sát vật thể | Phóng to kiểm tra độ chính xác của đường viền mask ở cấp độ pixel |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
-> Không huấn luyện hoặc tinh chỉnh mô hình, Không tự đổi checkpoint, threshold hoặc mã nguồn trừ khi GV cho phép (nếu đổi phải ghi trong REPORT.md) và không sử dụng ảnh tải lên từ máy tính.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
-> Khi phát hiện hình ảnh hoặc dữ liệu bất thường, mình sẽ lập tức dừng thao tác gán nhãn trên mẫu dữ liệu đó, ghi nhận lại mã định danh và báo cáo ngay cho Lab Coach.


## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
