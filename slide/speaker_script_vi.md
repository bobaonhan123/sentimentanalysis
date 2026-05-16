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

## Slide 10 - Where Negative Signals Concentrate
Heatmap cho thấy negative signal tập trung theo industry và company-volume group. Cách nhìn này cụ thể hơn so với chỉ xem tổng quan toàn bộ dataset.

## Slide 11 - Industry Deep Dive: Software IT
Deep dive chọn Software IT vì đây là segment gần đây có sample lớn nhất, gồm 2,306 reviews và 8,941 aspect mentions từ 2023 đến 2026 YTD. Mixed chart cho thấy negative ratio tăng từ 22.0% năm 2023 lên 55.3% năm 2025; riêng 2026 YTD chỉ nên xem là tín hiệu sớm vì chưa đủ cả năm.

## Slide 12 - Year-over-Year Aspect Movement
Slide này so sánh 2024 và 2025 trong Software IT. Salary & Benefits tăng 19.3 điểm phần trăm và Work Environment tăng 18.1 điểm, nghĩa là rủi ro năm 2025 đến từ compensation và team climate chứ không chỉ từ process hay tools.

## Slide 13 - Representative Company Drilldown
Bubble chart so sánh các công ty đại diện trong Software IT theo aspect. Cách đọc là màu và số trong bong bóng thể hiện negative ratio, còn kích thước bong bóng thể hiện mention volume. Vì vậy bong bóng to màu xanh nghĩa là nhiều người nhắc đến aspect đó, nhưng tỷ lệ negative thấp. FPT Software là case quy mô lớn nhưng tích cực, LGEDV có watch point ở Technology & Product, còn BMBSOFT và ALTEK có rủi ro tập trung ở Salary và Work Environment.

## Slide 14 - Business Recommendations from Aspect Analysis
Recommendation chuyển kết quả aspect thành action. Thứ nhất là benchmark compensation và benefit vì Salary & Benefits là pain point mạnh nhất. Thứ hai là review team climate vì Work Environment cũng xấu đi rõ. Thứ ba là dùng watchlist theo segment và company thay vì một action HR chung cho toàn bộ thị trường. Cuối cùng là đặt rule cảnh báo sớm để bắt các spike theo năm trước khi nó thành vấn đề cấu trúc.

## Slide 15 - Company Case: BMBSOFT 2025
Slide này biến recommendation thành một case cụ thể. Trong Software IT, BMBSOFT Vietnam năm 2025 có 39 reviews, 212 aspect mentions, và 61.8% aspect mentions là negative. Chart cho thấy vấn đề tập trung ở Salary & Benefits và Work Environment, cả hai đều khoảng 85% negative. Vì vậy recommendation không phải là một action HR chung: đầu tiên reset pay và benefits communication, sau đó chạy team-climate review ngắn, đồng thời giữ Career Growth như một điểm mạnh để retention.

## Slide 16 - What Words Drive the Pain Points?
Keyword chart cho biết các từ và matched terms đứng sau sentiment của từng aspect. Điều này giúp chuyển kết quả phân tích thành action cụ thể hơn.

## Slide 17 - Preprocessing Pipeline
Phần này giới thiệu pipeline preprocessing tiếng Việt trước khi modeling và aspect analysis.

## Slide 18 - Text Preprocessing: 8 Steps
Pipeline chuẩn hóa text qua Unicode normalization, lowercase, loại noise, lọc ký tự, chuẩn hóa whitespace, tokenization bằng underthesea, và stopword removal. Các bước này giảm nhiễu do web scraping và giúp model nhận input ổn định hơn.

## Slide 19 - Stopword Categories
Stopword được tạo từ frequency toàn corpus: đếm token trên toàn dataset, review các từ xuất hiện nhiều nhưng ít giá trị sentiment, rồi chọn lọc stopword tiếng Việt. Bảng bên dưới nhóm các stopword cuối cùng thành function words, conjunctions, prepositions, pronouns và quantifiers.

## Slide 20 - Modeling Strategy
Phần này tập trung vào weak label, label quality, neutral handling, và các sentiment model có thể triển khai.

## Slide 21 - The Core Modeling Challenge
Weak label ban đầu đến từ rating: 1-2 sao là negative, 3 sao là neutral, 4-5 sao là positive, sau đó keyword và ABSA conflict có thể điều chỉnh label. Neutral là class khó vì nó có thể là trung bình, mixed, không chắc chắn, hoặc cảm xúc yếu.

## Slide 22 - Experiment Results: 7 Target Models
Bảng này so sánh bảy model mục tiêu bằng Accuracy, Macro Recall và Macro F1. Với neutral label, PhoBERT NeutralBoost 0.9 là dòng PhoBERT tốt nhất; ở no-neutral, PhoBERT threshold 0.68 đạt Macro F1 cao nhất là 0.960.

## Slide 23 - Consistent Model Comparison
Chart giữ cùng bảy model mục tiêu và so sánh Macro F1 theo neutral setting. Recommendation thực tế là dùng binary no-neutral cho polarity alert có độ tin cậy cao, và dùng cleaned 3-class khi cần diễn giải neutral cho business analysis.

## Slide 24 - Best Neural Model Convergence
Convergence chart dùng neural run tốt nhất có lưu history theo epoch. Classical ML như Logistic Regression và Linear SVC không có epoch curve, nên chart này chỉ áp dụng cho neural model.

## Slide 25 - Practical Value of the Topic
Giá trị thực tế là hệ thống chuyển review không cấu trúc thành sentiment, aspect distribution và time trend có thể đo được. Nó cũng hỗ trợ market listening và decision support mà không cần đọc thủ công hàng nghìn review.

## Slide 26 - Summary
Summary nhắc lại những gì project đã xây dựng và kết quả chính. Binary polarity F1 cao nhất là 0.960 với PhoBERT threshold 0.68, cleaned 3-class F1 là 0.877, label audit tìm được 1,570 likely issues, và Salary là pain point lớn nhất với 60% negative sentiment.

## Slide 27 - Selected References
