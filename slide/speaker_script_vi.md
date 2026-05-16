# Speaker Script - Employee Review Intelligence Dashboard

## Slide 1 - Title
Chào thầy cô và các bạn. Nhóm 16 trình bày đề tài Employee Review Intelligence Dashboard. Dự án phân tích review công ty tiếng Việt từ 1900.com.vn, sau đó chuyển dữ liệu review thành dự đoán sentiment và business insight theo từng aspect để hỗ trợ ra quyết định.

## Slide 2 - Context & Problem
Review nhân viên là nguồn phản hồi công khai về môi trường làm việc, lương thưởng, workload và quản lý. Tuy nhiên rating sao không đủ, vì một review có thể 4 sao nhưng vẫn chứa complaint nghiêm trọng. Vì vậy nhóm cần phân tích text và sentiment theo aspect.

## Slide 3 - Industry Demand
Bài toán này có nhu cầu thực tế vì doanh nghiệp ngày càng cần people analytics và business analytics. Thay vì đọc thủ công từng review, hệ thống giúp biến phản hồi không cấu trúc thành tín hiệu có thể theo dõi theo thời gian, ngành, công ty và nhóm nhân viên.

## Slide 4 - Pain Points to Solution
Có bốn khó khăn chính: dữ liệu không cấu trúc, nhãn yếu có nhiễu, lớp neutral mơ hồ, và đọc thủ công không scale. Giải pháp của nhóm là pipeline end-to-end: crawl dữ liệu, tiền xử lý tiếng Việt, train mô hình sentiment, trích xuất aspect, và trực quan hóa bằng dashboard.

## Slide 5 - Research Objectives
Mục tiêu thứ nhất là so sánh nhiều setting mô hình: original 3-class, cleaned 3-class, binary no-neutral, mixed-conflict 4-class, các baseline FastText/MLP/LSTM và TF-IDF. Mục tiêu thứ hai là rút insight kinh doanh theo aspect, ngành, nhóm công ty và thời gian. Mục tiêu thứ ba là xây dựng dashboard proof of concept.

## Slide 6 - Dataset Collection
Dataset gồm 10,000 review tiếng Việt được crawl từ 1900.com.vn. Pipeline có scraper, Bloom filter để chống trùng URL, Kafka/PostgreSQL cho ingestion và lưu trữ, sau đó dữ liệu được đưa vào NLP preprocessing.

## Slide 7 - Dataset Schema & Distribution
Mỗi review có company, industry, rating, pros, cons, advice, job title, employee status, location và date. Phân phối nhãn bị lệch: positive chiếm nhiều nhất, neutral và negative ít hơn. Vì vậy nhóm dùng Macro F1 làm metric chính thay vì chỉ nhìn accuracy.

## Slide 8 - Preprocessing Pipeline
Tiền xử lý tiếng Việt gồm chuẩn hóa Unicode, lowercase, bỏ URL/HTML, tokenize tiếng Việt và xử lý stopword. Bước này giảm nhiễu trước khi train model truyền thống và trước khi chạy aspect extraction.

## Slide 9 - Stopword Categories
Stopword gồm từ chức năng tiếng Việt, đại từ, liên từ, giới từ, lượng từ và một số stopword tiếng Anh. Tuy nhiên cần giữ ý nghĩa phủ định như "không" hoặc "chưa", vì bỏ nhầm sẽ đảo sentiment.

## Slide 10 - Modeling Challenge
Thách thức lớn nhất là lớp neutral. Rating 1-2 sao là negative, 3 sao là neutral, 4-5 sao là positive, nhưng neutral có thể là trung bình, mixed opinion, hoặc cảm xúc yếu. Đây là lý do original 3-class khó đạt điểm cao.

## Slide 11 - Experiment Design
Nhóm thử nhiều hướng: original 3-class, cleaned 3-class bằng confident learning, binary no-neutral, và mixed-conflict 4-class. Ngoài TF-IDF, nhóm vẫn so sánh với FastText + MLP, FastText + LSTM và RandomForest để xem vấn đề nằm ở model hay ở chất lượng nhãn.

## Slide 12 - Confident Learning
Confident learning dùng prediction xác suất out-of-fold để phát hiện nhãn có khả năng sai. Kết quả flag 1,570 label issues. Điều này cho thấy nhiễu nhãn, đặc biệt ở neutral và mixed review, là bottleneck chính.

## Slide 13 - Experiment Results
Kết quả tốt nhất tổng thể là TF-IDF Word+Char + LinearSVC cho binary sentiment, accuracy 93.25% và Macro F1 0.918. Với setting 3-class, cleaned 3-class bằng TF-IDF + Logistic đạt accuracy 91.78% và Macro F1 0.877. Original 3-class chỉ đạt Macro F1 khoảng 0.720.

## Slide 14 - Model Comparison
Chart selected model comparison không chỉ có TF-IDF. Nó chọn các model đại diện: TF-IDF variants ở nhóm trên, FastText + MLP đạt Macro F1 0.728, FastText + LSTM đạt 0.700, và FastText + RandomForest đạt 0.648. MLP vẫn có trong so sánh, nhưng thấp hơn cleaned TF-IDF vì vấn đề chính của dataset là label quality hơn là độ phức tạp kiến trúc.

## Slide 15 - Neutral Handling
Neutral không nên xóa nếu mục tiêu là business interpretation, vì nó giúp giữ lại các review không chắc chắn hoặc mixed. Nhưng nếu cần alert positive/negative rõ ràng, binary no-neutral là setting mạnh nhất. Vì vậy recommendation là dùng binary cho alerting và cleaned 3-class cho phân tích nghiệp vụ.

## Slide 16 - Result Benchmarking
Kết quả này hợp lý so với nghiên cứu sentiment trước đây: neutral thường khó hơn positive/negative, và 3-class sentiment trên dữ liệu noisy thường có Macro F1 thấp hơn binary. Vì vậy việc cleaned 3-class tăng mạnh là tín hiệu quan trọng.

## Slide 17 - Business Insights
Sau phần model, nhóm chuyển sang ABSA để phân tích 10 workplace aspects như Salary & Benefits, Work Environment, Management, Work-Life Balance và Facilities & Tools. Mục tiêu là tìm pain point có thể hành động.

## Slide 18 - ABSA Overview
Từ 10,000 review, hệ thống trích xuất 32,789 aspect mentions. Salary & Benefits là pain point rõ nhất với 5,626 negative mentions và khoảng 60% negative. Work Environment có volume cao nhất và bị phân cực mạnh.

## Slide 19 - Trend Analysis
Trend analysis cho thấy vấn đề nào đang tăng theo thời gian. Facilities & Tools và Work Environment có xu hướng tăng negative ở giai đoạn gần đây, nên đây là tín hiệu rủi ro mới nổi cho doanh nghiệp.

## Slide 20 - Segment Hotspots
Segment hotspot giúp biết negative tập trung ở ngành hoặc nhóm công ty nào. Điều này biến sentiment analysis từ báo cáo tổng quan thành công cụ ưu tiên hành động.

## Slide 21 - Representative Companies
Representative companies làm insight cụ thể hơn. Cùng một aspect nhưng mỗi công ty có thể cần hành động khác nhau, ví dụ có nơi cần benchmark lương, có nơi cần giảm workload.

## Slide 22 - Key Findings
Bốn finding chính là: Salary & Benefits là pain point lớn nhất, Work Environment bị phân cực, một số issue tăng mạnh ở giai đoạn gần đây, và drilldown theo company/industry giúp action cụ thể hơn.

## Slide 23 - Business Priority Roadmap
Roadmap ưu tiên: P1 là Salary & Benefits và Work Environment; P2 là Facilities & Tools và Work-Life Balance vì có dấu hiệu emerging risk; Career Growth và Management nên được monitor liên tục.

## Slide 24 - End-to-End Project Flow
Kiến trúc hệ thống gồm crawl dữ liệu, dedup bằng Bloom filter, Kafka/PostgreSQL, preprocessing, sentiment model, ABSA và Streamlit dashboard. Pipeline có thể chạy batch để cập nhật insight định kỳ.

## Slide 25 - Summary
Tóm lại, nhóm xây dựng pipeline từ data collection đến modeling và dashboard. Kết quả mạnh nhất cho binary là Macro F1 0.918; kết quả 3-class tốt nhất là Macro F1 0.877 sau label cleaning. Về business, Salary & Benefits là vấn đề ưu tiên cao nhất.

## Slide 26 - References
Các reference hỗ trợ cho phần sentiment benchmark, neutral handling và confident learning, giúp giải thích vì sao neutral khó và vì sao label quality là yếu tố quyết định trong bài toán này.
