# Kế hoạch nghiên cứu MVP cho bài toán phát hiện `negative / non-negative` trên review nhân sự tiếng Việt

## 1. Mục tiêu của bản kế hoạch này

Bản kế hoạch này chỉ giữ phần lõi đủ để:

- hình thành một đề tài nghiên cứu có luận điểm rõ;
- trả lời trực tiếp các research question chính;
- xây được một pipeline có thể triển khai và đánh giá trong repo hiện tại;
- giữ tính khách quan bằng thiết kế đánh giá chặt, thay vì mở rộng quá nhiều hướng cùng lúc.

Trung tâm của bài vẫn là review nhân sự tiếng Việt. Tuy nhiên, để tăng tính khách quan, bài được phép bổ sung đúng một nguồn employee-review tiếng Anh cùng domain dưới dạng public mirror.

## 2. Cập nhật về nguồn dữ liệu cùng domain

Kết luận trước đây rằng “không có public dataset cùng domain để dùng” là không còn chính xác.

Kết luận đúng hơn ở thời điểm hiện tại là:

- có public mirrored dataset cùng domain employee reviews;
- nhưng chưa phải benchmark license-clean theo nghĩa học thuật an toàn;
- vì vậy nên dùng nó như `external mirrored dataset` hoặc `auxiliary same-domain source`, không nên trình bày như benchmark chuẩn được cấp phép rộng rãi.

Nguồn đã xác định được:

- `lallantop/glassdoor` trên Hugging Face
- quy mô tham chiếu hiện xem được: khoảng `838,566 rows`, khoảng `88 MB`
- các trường quan sát được trên trang dataset gồm kiểu `overall_rating`, `pros`, `cons`, `headline`, `recommend`, `ceo_approv`, `firm`, `job_title`
- link: `https://huggingface.co/datasets/lallantop/glassdoor`

## 3. Loại hình nghiên cứu nên chọn

Đề tài này nên được định nghĩa là:

- **nghiên cứu định lượng là chính**
- **có phần định tính nhúng vào để giải thích lỗi và hiện tượng ngôn ngữ**

Nói chính xác hơn, đây là thiết kế:

- **quantitative-dominant mixed design**

Phần định lượng dùng để trả lời:

- hệ thống phát hiện bất mãn nên được thiết kế thế nào;
- mô hình nào hiệu quả hơn;
- mô hình mới có cải thiện được chỉ số quan trọng hay không.

Phần định tính dùng để trả lời:

- vì sao mô hình sai;
- hiện tượng tiếng Việt nào gây lỗi;
- đầu ra aspect-level có thực sự hữu ích cho doanh nghiệp không.

## 4. Tuyên bố bài toán nghiên cứu

Đề tài không còn được phát biểu như một bài toán sentiment 3 lớp tổng quát.

Phát biểu đúng là:

> Xây dựng một hệ thống phát hiện sự không hài lòng trong review nhân sự tiếng Việt ở mức văn bản, đồng thời chỉ ra các khía cạnh tiêu cực chính để hỗ trợ doanh nghiệp hành động.

Từ đó, bài toán lõi gồm hai tầng:

1. **Document-level negative detection**
   - nhãn: `negative` (có bất kỳ yếu tố tiêu cực — gồm mixed/neutral) / `non-negative` (chỉ tích cực thuần)
   - mục tiêu nghiệp vụ: nhận diện review **có dấu hiệu tiêu cực**, kể cả review 4–5★ còn complaint

2. **Aspect-level negative signal identification**
   - nhiều khía cạnh có thể cùng xuất hiện trong một review

Đây là cách phát biểu đúng với mục tiêu nghiệp vụ hơn so với:

- `negative / neutral / positive`

vì doanh nghiệp cần:

- biết review có thực sự tích cực hay không;
- biết phần không tích cực nằm ở khía cạnh nào.

> **Lý do đổi framing (2026-06):** chỉ cần một phần review có negative thì tính là có yếu tố negative; review 4 sao vẫn có thể chứa complaint; neutral/mixed gộp vào `negative`. `non-negative` chỉ khi hoàn toàn tích cực, không có cue tiêu cực.

## 5. Research questions đã rút gọn

Giữ đúng **3 research questions chính** và **1 research question phụ**.

### RQ1

Với review nhân sự tiếng Việt, làm thế nào để xây dựng một hệ thống phát hiện review `non-positive / positive` có độ nhạy cao đối với các review thật sự không tích cực?

### RQ2

Trong bài toán phát hiện bất mãn trên review nhân sự tiếng Việt, non-transformer hay transformer là lựa chọn phù hợp hơn khi xét đồng thời hiệu năng, độ ổn định và khả năng giải thích?

### RQ3

Việc bổ sung tầng `aspect-level negative detection` có giúp tăng giá trị hành động của hệ thống mà không làm suy giảm đáng kể hiệu năng phát hiện bất mãn ở mức văn bản hay không?

### RQ4 (phụ)

Những hiện tượng ngôn ngữ nào trong review nhân sự tiếng Việt là nguyên nhân chính gây lỗi cho hệ thống phát hiện bất mãn?

## 6. Giả thuyết nghiên cứu

### H1

Một hệ thống được thiết kế trực tiếp cho bài toán `non-positive / positive` sẽ đạt hiệu quả đủ mạnh trên các chỉ số phát hiện không tích cực, đặc biệt là `non-positive recall`, `non-positive F1`, `F2-non-positive`, và `PR-AUC-non-positive`.

### H2

Transformer tiếng Việt sẽ vượt non-transformer ở các review có ngữ cảnh phức tạp, phủ định và nhiều khía cạnh chồng lấn; nhưng non-transformer vẫn là baseline mạnh và có giá trị giải thích cao.

### H3

Một mô hình có tầng `aspect-level negative detection` sẽ tạo đầu ra hữu ích hơn cho doanh nghiệp mà không làm giảm đáng kể hiệu năng của document-level binary classification.

## 7. Phạm vi MVP

### Dữ liệu chính

- tập `1900.com.vn` hiện có trong repo;
- đây là nguồn chính để trả lời trực tiếp các RQ về tiếng Việt.

### Dữ liệu ngoài được phép bổ sung

- `lallantop/glassdoor` trên Hugging Face;
- chỉ dùng như `external mirrored dataset`;
- vai trò phù hợp:
- auxiliary transfer cho binary dissatisfaction detection;
- external robustness check ở cùng domain employee reviews;
- bổ sung tính khách quan khi tranh luận rằng framing `non-positive / positive` không chỉ đúng với đúng một tập tiếng Việt.

### Bài toán

- `non-positive / positive` (3-class weak labels giữ nguyên nội bộ; train binary gộp 0+1 vs 2)

### Baseline

- TF-IDF word-char + Logistic Regression
- TF-IDF word-char + LinearSVC

### Transformer baseline

- PhoBERT binary

### Lớp giải thích

- `aspect-level negative detection` ở mức silver labels

### Không đưa vào MVP

- nhiều dataset ngoài cùng lúc;
- nhiều transformer ngoài PhoBERT;
- nhiều mô hình sâu phức tạp không phục vụ trực tiếp RQ;
- nhiều task phụ ngoài dissatisfaction detection và aspect-negative detection.

## 8. Xây dựng mô hình nghiên cứu

### 8.1 Baseline định lượng

#### Baseline 1: TF-IDF word-char + Logistic Regression

Vai trò:

- baseline mạnh;
- dễ tái lập;
- dễ giải thích.

#### Baseline 2: TF-IDF word-char + LinearSVC

Vai trò:

- baseline cạnh tranh tốt cho text classification;
- thường mạnh ở dữ liệu sparse text;
- phù hợp làm chuẩn non-transformer.

#### Baseline 3: PhoBERT binary

Vai trò:

- đại diện cho transformer tiếng Việt;
- là đối thủ trực tiếp của các baseline tuyến tính.

### 8.2 Mô hình nghiên cứu đề xuất

Tên tạm đề xuất:

- **VADAN**
- viết tắt của **Vietnamese Aspect-aware Dissatisfaction Attention Network**

### 8.3 Ý tưởng của VADAN

VADAN không phải một transformer mới từ đầu, mà là kiến trúc gắn đúng với bài toán:

#### Tầng 1: shared encoder

Dùng PhoBERT để mã hóa toàn bộ review thành biểu diễn ngữ nghĩa.

#### Tầng 2: dissatisfaction head

Head binary để dự đoán:

- `negative`
- `non-negative`

#### Tầng 3: aspect-negative head

Head multi-label để dự đoán các khía cạnh tiêu cực, ví dụ:

- salary
- benefits
- workload
- management
- environment
- growth
- process
- culture

#### Tầng 4: aspect-guided attention hoặc aspect-guided pooling

Tín hiệu aspect được dùng để điều hướng phần biểu diễn phục vụ binary prediction.

Nói ngắn:

- nếu chỉ dùng PhoBERT binary, ta có classifier;
- nếu dùng VADAN, ta có classifier + explanation layer gắn cấu trúc hơn với bài toán.

### 8.4 Loss function của VADAN

Tổng loss:

`L = lambda1 * L_doc + lambda2 * L_aspect + lambda3 * L_consistency`

Trong đó:

#### `L_doc`

Loss cho document-level binary classification.

MVP nên bắt đầu bằng:

- weighted BCE

#### `L_aspect`

Loss cho multi-label aspect detection.

MVP nên dùng:

- `BCEWithLogitsLoss` có trọng số

#### `L_consistency`

Regularization đơn giản:

- nếu xác suất `negative` cao;
- thì ít nhất một aspect-negative cũng nên có xác suất cao.

## 9. Thiết kế nhãn cho MVP

### 9.1 Nhãn document-level

Thiết kế tối thiểu:

- `negative`
- `non-negative`
- `ambiguous` chỉ dùng làm cờ, không là lớp huấn luyện chính

### 9.2 Gold label và weak label

- nhãn hiện tại trong pipeline chủ yếu là **weak labels**
- nếu muốn có kết luận học thuật mạnh hơn, cần tạo thêm một tập **gold labels**

Gold label ở đây nghĩa là:

- nhãn được con người gán trực tiếp theo guideline rõ ràng;
- có kiểm tra đồng thuận;
- không phụ thuộc hoàn toàn vào rule hoặc rating prior.

## 10. Kế hoạch dùng dữ liệu tiếng Anh cho tính khách quan

Nguồn dùng:

- `lallantop/glassdoor`

Mục tiêu dùng:

1. kiểm tra framing `negative / non-negative` trên cùng domain employee reviews ở ngôn ngữ khác;
2. làm nguồn transfer hoặc pre-screen cho complaint cues cùng domain;
3. làm external robustness check, không thay thế tập tiếng Việt.

Những điều không được claim:

- không nói đây là benchmark license-clean;
- không nói đây là bằng chứng trực tiếp cho đặc thù tiếng Việt;
- không để phần tiếng Anh lấn át đóng góp chính của bài.

## 11. Thiết kế đánh giá

### 11.1 Đánh giá định lượng cho RQ1

Chỉ số chính:

- Negative Precision
- Negative Recall
- Negative F1
- F2-negative
- PR-AUC-negative

Ưu tiên nghiên cứu:

- không bỏ sót review tiêu cực;
- vì vậy `negative recall` và `F2-negative` là hai chỉ số trọng tâm.

### 11.2 Đánh giá cho RQ2

So sánh:

- Logistic Regression
- LinearSVC
- PhoBERT binary
- VADAN

Tiêu chí:

- hiệu năng;
- độ ổn định qua split hoặc seed;
- khả năng giải thích;
- chi phí triển khai.

### 11.3 Đánh giá cho RQ3

So sánh:

- mô hình chỉ document-level;
- mô hình document-level + aspect-negative branch

Kết luận cần rút ra:

- đầu ra aspect-level có làm hệ thống hữu ích hơn cho doanh nghiệp hay không;
- có đánh đổi nhiều hiệu năng document-level hay không.

### 11.4 Đánh giá định tính cho RQ4

Phân tích các nhóm lỗi:

- phủ định;
- mỉa mai hoặc chê nhẹ;
- câu khen mở đầu nhưng than phiền ở phần sau;
- review rating cao nhưng text tiêu cực;
- từ lóng, viết tắt, cách nói gián tiếp trong review nhân sự tiếng Việt.

## 12. Kết luận hành động

Kế hoạch chốt cho giai đoạn này là:

- giữ trung tâm nghiên cứu ở review nhân sự tiếng Việt;
- chuyển hẳn sang bài toán phát hiện bất mãn `negative / non-negative`;
- dùng aspect-negative detection để tăng giá trị hành động;
- dùng `lallantop/glassdoor` như một `external mirrored dataset` cùng domain nhằm tăng tính khách quan, nhưng không trình bày nó như benchmark license-clean;
- ưu tiên đào sâu chất lượng nhãn, đánh giá lỗi, và thiết kế mô hình phù hợp hơn với bài toán, thay vì mở rộng quá nhiều hướng.
