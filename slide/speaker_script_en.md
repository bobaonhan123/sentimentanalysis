# Speaker Script - Employee Review Intelligence Dashboard

## Slide 1 - Title
Good morning everyone. We are Group 16, and our topic for Business Sentiment Analysis is Employee Review Intelligence Dashboard. In this project, we analyze Vietnamese company reviews from 1900.com.vn and turn them into sentiment predictions and aspect-level HR insights. Our main goal is not only to build a classifier, but to create a decision-support tool for understanding employee pain points.

## Slide 2 - Context & Problem
First, let us start with the context. Employee reviews are a public source of workplace feedback. They often reveal issues about salary, culture, workload, and management that may not appear clearly in internal surveys. However, ratings alone are not enough. A review can have a high star rating but still contain serious complaints. That is why we need text analytics and aspect-level sentiment analysis.

## Slide 3 - Industry Demand
This topic is also relevant from an industry perspective. Gallup reports that global employee engagement fell to 21% in 2024, and engagement decline creates a major productivity cost. At the same time, the HR analytics market is projected to grow strongly. So the demand is clear: companies need better listening tools that turn employee feedback into timely signals.

## Slide 4 - Pain Points to Solution
There are four main pain points. The data is unstructured, the labels are noisy, neutral opinions are ambiguous, and manual reading does not scale. Therefore, our solution is an end-to-end pipeline: collect reviews, preprocess Vietnamese text, train sentiment models, extract workplace aspects, and visualize the results in a dashboard for HR and management.

## Slide 5 - Research Objectives
Our research has three objectives. From a modeling perspective, we compare different sentiment settings, including original three-class, cleaned three-class, binary, mixed-conflict, and several ML and deep learning baselines. From a business perspective, we identify critical issues by aspect, industry, company group, and time. From an application perspective, we build a usable dashboard as a proof of concept.

## Slide 6 - Dataset Collection
The dataset was self-collected from 1900.com.vn. We crawled 10,000 Vietnamese workplace reviews because there was no ready-made labeled dataset for this specific domain. The pipeline includes scraping, URL deduplication with a Bloom filter, Kafka-based ingestion, NLP preprocessing, and PostgreSQL storage.

## Slide 7 - Dataset Schema & Distribution
Each review contains rating, pros, cons, advice, job title, employee status, company, industry, location, and date. The label distribution is imbalanced: positive reviews account for about 57%, negative about 24%, and neutral about 19%. This imbalance is important because accuracy alone can be misleading, so we use macro F1 as the main metric.

## Slide 8 - Preprocessing Pipeline
Vietnamese text needs specific preprocessing. We normalize Unicode, lowercase text, remove URLs and HTML, tokenize Vietnamese words, and remove stopwords where appropriate. This step reduces noise before training traditional machine learning models and before aspect extraction.

## Slide 9 - Stopword Categories
The stopword list includes Vietnamese function words, pronouns, conjunctions, prepositions, quantifiers, and some English stopwords because workplace reviews may mix languages. However, for sentiment, we must be careful with negation words such as "khong" or "chua", because removing them can change the meaning.

## Slide 10 - Modeling Challenge
The core challenge is neutral sentiment. In our weak labeling strategy, ratings generate base labels: one to two stars are negative, three stars are neutral, and four to five stars are positive. But neutral can mean many things: average experience, mixed opinion, uncertain sentiment, or low emotional intensity. This is why the original three-class model struggled.

## Slide 11 - Experiment Design
To address this, we designed several experiments. We tested the original three-class model, a cleaned three-class model using confident learning, a binary model without neutral, and a four-class mixed-conflict model. We also kept model diversity by comparing TF-IDF models, FastText plus MLP, classic machine learning baselines, and LSTM. This lets us understand whether the main issue is model architecture or label quality.

## Slide 12 - Confident Learning
Confident learning was used to audit label quality. The method uses out-of-fold probabilistic predictions to find labels that are unlikely given the model's confidence. It flagged 1,570 likely label issues. This is important because it shows that noisy weak labels, especially around neutral and mixed reviews, are a major bottleneck.

## Slide 13 - Experiment Results
The best overall model is TF-IDF Word+Char plus LinearSVC for binary sentiment, with 93.25% accuracy and 0.918 macro F1. However, for a three-class research setting, the best model is TF-IDF Word+Char plus Logistic Regression after label cleaning, with 91.78% accuracy and 0.877 macro F1. The original three-class model only reached 0.720 macro F1, so label cleaning created the biggest improvement.

## Slide 14 - Neutral Handling
The key takeaway is that neutral should not simply be deleted. If we need a high-confidence alerting model, binary classification works best. But if we need business interpretation, neutral is still valuable because it prevents the system from forcing uncertain reviews into positive or negative. Therefore, our practical recommendation is to use binary for alerts and cleaned three-class for interpretation.

## Slide 15 - Result Benchmarking
Our results are reasonable compared with established sentiment research. SemEval-2013 reported about 69% F1 for message-level Twitter sentiment. SentiBench found that best three-class tools often achieved macro F1 around 0.6, and neutral was consistently harder than positive and negative. This supports our interpretation that the original 79-80% accuracy was not unusually low for a noisy neutral sentiment task.

## Slide 16 - Business Insights
Now we move from model performance to business insights. We use aspect-based sentiment analysis to extract ten workplace aspects, such as Salary & Benefits, Work Environment, Management, Work-Life Balance, and Facilities & Tools. The goal is to identify which issues are most critical and actionable.

## Slide 17 - ABSA Overview
Across 10,000 reviews, we extracted 32,789 aspect mentions. Salary & Benefits is the clearest negative outlier, with 5,626 negative mentions and about 60% negative sentiment. Work Environment has the highest total discussion volume and is both praised and criticized, which suggests strong differences across companies or teams.

## Slide 18 - Trend Analysis
The trend analysis shows whether issues are static or becoming worse. Facilities & Tools and Work Environment show rising negativity in recent windows. This matters because HR should not only look at the largest issue today, but also detect emerging risks early.

## Slide 19 - Segment Hotspots
The segment view shows where negative signals concentrate. By industry and company group, we can identify which sectors or company types have stronger salary, workload, or environment issues. This turns sentiment analysis into a prioritization tool rather than a general summary.

## Slide 20 - Representative Companies
Representative companies make the insights more actionable. For example, one company may need workload intervention, while another may need compensation benchmarking. The same aspect can lead to different actions depending on the company and industry context.

## Slide 21 - Key Findings
There are four key findings. First, Salary & Benefits is the main pain point. Second, Work Environment is highly polarizing. Third, trend analysis shows that some issues worsened around 2025 and cooled down in 2026 year-to-date. Finally, company-level drill-downs make the analysis actionable.

## Slide 22 - HR Priority Roadmap
Based on the analysis, we propose a priority roadmap. Salary & Benefits and Work Environment are P1 issues. Facilities & Tools and Work-Life Balance are P2 because they show emerging risk. Career Growth and Management should be monitored continuously.

## Slide 23 - End-to-End Project Flow
This slide shows the full system architecture. The pipeline can be run as a manual batch process. The scraper and Bloom filter collect and deduplicate URLs. Kafka supports ingestion, PostgreSQL stores structured data, and NLP preprocessing prepares text. Then sentiment models and ABSA generate insights, which are delivered through a Streamlit dashboard.

## Slide 24 - Summary
To conclude, we built a full pipeline from data collection to model training and business dashboard. The strongest model result is 0.918 macro F1 for binary polarity. The strongest three-class result is 0.877 macro F1 after label cleaning. From the business side, Salary & Benefits is the most critical pain point, and the dashboard shows how this system can support HR decision-making.

## Slide 25 - References
Finally, the project is supported by prior work on sentiment benchmarks, neutral handling, and confident learning. These references help us justify why neutral is difficult, why binary scores are usually higher, and why label quality is essential for this type of task.
