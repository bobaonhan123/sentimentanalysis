# Speaker Script - Sentiment Analysis System for Employee Reviews

## Slide 1 - Title
Chào thầy cô và các bạn. Nhóm 16 trình bày đề tài Sentiment Analysis System for Employee Reviews. Project chuyển review tiếng Việt về nơi làm việc thành tín hiệu sentiment và các ưu tiên business.

## Slide 2 - Context & Problem
Phần này giới thiệu vì sao phân tích employee review có giá trị và bài toán dashboard cần giải quyết.

## Slide 3 - Why Employee Review Analytics Matters
Slide này đưa ra ba tín hiệu business: tỷ lệ employee engagement thấp, chi phí năng suất bị mất rất lớn, và people analytics là thị trường đang tăng trưởng. Vì vậy việc đọc và tổng hợp review tự động là cần thiết.

## Slide 4 - Job - Pain - Gain
Nhiệm vụ là hiểu nhân viên phàn nàn về điều gì, ở đâu, và mức độ khẩn cấp ra sao. Điểm khó là review rất nhiễu và không có cấu trúc, còn giá trị đạt được là dashboard biến review thành trend, aspect quan trọng, và action priority.

## Slide 5 - Flow Architecture
Đây là flow trừu tượng dùng chung cho project. Review được thu thập, chuẩn bị, chuyển thành biểu diễn văn bản, dùng để train và so sánh model, rồi tạo sentiment prediction, aspect insight, trend, hotspot và dashboard output.

## Slide 6 - Dataset Collection
Dataset được tự thu thập từ 1900.com.vn. Flow gồm scrape company và review page, deduplicate bằng Bloom filter, đưa qua Kafka, lưu PostgreSQL, rồi chuẩn bị text tiếng Việt.

## Slide 7 - Dataset Snapshot
Mỗi review có nội dung, rating, company context, thời gian và profile metadata. Chart so sánh label từ rating với weak label sau khi dùng keyword và ABSA evidence; 1,469 review, tương đương 14.7%, đã được gán lại label.

## Slide 8 - Business Insights
Phần này chuyển từ dataset và sentiment output sang phân tích aspect cho business.

## Slide 9 - Aspect Sentiment Overview
Chart tổng hợp sentiment theo từng workplace aspect. Salary & Benefits là điểm negative rõ nhất, còn Work Environment có volume thảo luận lớn nhất.

## Slide 10 - Critical Issues by Business Priority
Salary & Benefits và Work Environment được đưa vào P1 vì có negative volume và negative ratio cao. Facilities & Tools là P2 vì có mức tăng negative ratio lớn nhất.

## Slide 11 - Which Issues Are Getting Worse?
Trend chart tập trung vào các aspect đang xấu đi. Watchlist gồm Facilities & Tools, Work Environment, và Work-Life Balance.

## Slide 12 - Where Negative Signals Concentrate
Heatmap cho thấy negative signal tập trung theo industry và company-volume group. Cách nhìn này cụ thể hơn so với chỉ xem tổng quan toàn bộ dataset.

## Slide 13 - What Words Drive the Pain Points?
Keyword chart cho biết các từ và matched terms đứng sau sentiment của từng aspect. Điều này giúp chuyển kết quả phân tích thành action cụ thể hơn.

## Slide 14 - Key Business Findings
Slide này có bốn kết luận chính: Salary & Benefits là pain point lớn nhất, Work Environment gây phân cực, Facilities & Tools là rủi ro đang tăng, và dashboard hỗ trợ ưu tiên hành động.

## Slide 15 - Preprocessing Pipeline
Phần này giới thiệu pipeline preprocessing tiếng Việt trước khi modeling và aspect analysis.

## Slide 16 - Text Preprocessing: 8 Steps
Pipeline chuẩn hóa text qua Unicode normalization, lowercase, loại noise, lọc ký tự, chuẩn hóa whitespace, tokenization bằng underthesea, và stopword removal. Các bước này giảm nhiễu do web scraping và giúp model nhận input ổn định hơn.

## Slide 17 - Stopword Categories
Stopword được tạo từ frequency toàn corpus: đếm token trên toàn dataset, review các từ xuất hiện nhiều nhưng ít giá trị sentiment, rồi chọn lọc stopword tiếng Việt. Bảng bên dưới nhóm các stopword cuối cùng thành function words, conjunctions, prepositions, pronouns, quantifiers và compound fragments.

## Slide 18 - Modeling Strategy
Phần này tập trung vào weak label, label quality, neutral handling, và các sentiment model có thể triển khai.

## Slide 19 - The Core Modeling Challenge
Weak label ban đầu đến từ rating: 1-2 sao là negative, 3 sao là neutral, 4-5 sao là positive, sau đó keyword và ABSA conflict có thể điều chỉnh label. Neutral là class khó vì nó có thể là trung bình, mixed, không chắc chắn, hoặc cảm xúc yếu.

## Slide 20 - Experiment Design
Project test bốn phiên bản: original 3-class, cleaned 3-class, binary no-neutral, và mixed-conflict 4-class. Thiết kế này giúp tách ảnh hưởng của label quality, neutral handling và model family.

## Slide 21 - Confident Learning
Confident learning được dùng để audit label quality. Phương pháp train out-of-fold probabilistic models, ước lượng label không hợp lý, và xuất các dòng đáng nghi; trong dataset này có 1,570 likely label issues.

## Slide 22 - Experiment Results: 7 Target Models
Bảng này so sánh bảy model mục tiêu bằng Accuracy, Macro Recall và Macro F1. Với neutral label, FastText+MLP là neural baseline mạnh nhất, nhưng no-neutral Linear SVC cho F1 cao nhất là 0.918.

## Slide 23 - Consistent Model Comparison
Chart giữ cùng bảy model mục tiêu và so sánh Macro F1 theo neutral setting. Recommendation thực tế là dùng binary no-neutral cho polarity alert có độ tin cậy cao, và dùng cleaned 3-class khi cần diễn giải neutral cho business analysis.

## Slide 24 - Best Neural Model Convergence
Convergence chart dùng neural run tốt nhất có lưu history theo epoch. Classical ML như Logistic Regression và Linear SVC không có epoch curve, nên chart này chỉ áp dụng cho neural model.

## Slide 25 - Practical Value of the Topic
Giá trị thực tế là hệ thống chuyển review không cấu trúc thành sentiment, aspect distribution và time trend có thể đo được. Nó cũng hỗ trợ market listening và decision support mà không cần đọc thủ công hàng nghìn review.

## Slide 26 - Summary
Summary nhắc lại những gì project đã xây dựng và kết quả chính. Binary polarity F1 cao nhất là 0.918, cleaned 3-class F1 là 0.877, label audit tìm được 1,570 likely issues, và Salary là pain point lớn nhất với 60% negative sentiment.

## Slide 27 - Selected References
Slide cuối liệt kê các tài liệu tham khảo chính, gồm báo cáo workforce engagement, benchmark sentiment analysis, và confident learning cho label quality.
