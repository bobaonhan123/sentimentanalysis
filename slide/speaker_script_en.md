# Speaker Script - Sentiment Analysis System for Employee Reviews

## Slide 1 - Title
Good morning everyone. We are Group 16, and our topic is Sentiment Analysis System for Employee Reviews. The project turns Vietnamese workplace reviews into sentiment signals and business priorities.

## Slide 2 - Context & Problem
This section explains why employee review analytics matters and what problem the dashboard is designed to solve.

## Slide 3 - Why Employee Review Analytics Matters
This slide shows three business signals: only 21% of global employees are engaged, engagement decline is linked to a 438-billion-dollar productivity loss, and people analytics is a growing market. This motivates faster ways to detect workforce problems from public feedback.

## Slide 4 - Job - Pain - Gain
The job is to understand what employees complain about, where it happens, and how urgent it is. The pain is that reviews are unstructured and noisy, while the gain is a dashboard that turns raw reviews into sentiment trends, critical aspects, and action priorities.

## Slide 5 - Flow Architecture
This is the abstract flow used across the whole project. Reviews are collected and prepared, converted into text representations, used to train and compare models, and then turned into sentiment predictions, aspect insights, trends, hotspots, and dashboard outputs.

## Slide 6 - Dataset Collection
The dataset was self-collected from 1900.com.vn. The collection flow goes from scraping company and review pages, to deduplication with a Bloom filter, ingestion through Kafka, storage in PostgreSQL, and Vietnamese text preparation.

## Slide 7 - Dataset Snapshot
Each review contains content fields, a star rating, company context, and time or profile metadata. The chart compares rating-only labels with weak labels after keyword and ABSA evidence; 1,469 reviews, or 14.7%, were re-labeled before model training.

## Slide 8 - Business Insights
This section moves from dataset and sentiment output to aspect-based business analysis.

## Slide 9 - Aspect Sentiment Overview
The chart summarizes sentiment by workplace aspect. Salary & Benefits is the clearest negative outlier, while Work Environment has the largest discussion volume.

## Slide 10 - Where Negative Signals Concentrate
The heatmaps show where negative signals concentrate by industry and by company-volume group. This makes the analysis more targeted than an overall summary.

## Slide 11 - Industry Deep Dive: Software IT
The deep dive picks Software IT because it has the largest recent sample, with 2,306 reviews and 8,941 aspect mentions from 2023 to 2026 YTD. The mixed chart shows that negative aspect ratio rose from 22.0% in 2023 to 55.3% in 2025, while 2026 YTD should be treated as an early partial-year signal.

## Slide 12 - Year-over-Year Aspect Movement
This slide compares 2024 and 2025 inside Software IT. Salary & Benefits increased by 19.3 percentage points and Work Environment by 18.1 points, so the 2025 deterioration is driven by compensation and team climate, not only by process or tools.

## Slide 13 - Representative Company Drilldown
The bubble chart compares representative Software IT companies by aspect. Read it as: color and the number inside each bubble show negative ratio, while bubble size shows mention volume. A large green bubble means many employees discussed that aspect, but the negative ratio is low. FPT Software is the scaled positive case, LGEDV has a specific Technology & Product watch point, while BMBSOFT and ALTEK show concentrated Salary and Work Environment risk.

## Slide 14 - Business Recommendations from Aspect Analysis
The recommendations convert aspect findings into actions. First, benchmark compensation and benefits because Salary & Benefits is the strongest pain point. Second, review team climate because Work Environment also worsened sharply. Third, use segment and company watchlists instead of one generic HR action. Finally, use early-warning rules so year-specific spikes are caught before they become structural.

## Slide 15 - Company Case: BMBSOFT 2025
This slide turns the recommendation into one concrete business case. In Software IT, BMBSOFT Vietnam in 2025 has 39 reviews, 212 aspect mentions, and 61.8% overall negative aspect mentions. The chart shows that the problem is concentrated in Salary & Benefits and Work Environment, both around 85% negative. The recommendation is therefore not a generic HR action: first reset pay and benefits communication, then run a short team-climate review, while preserving Career Growth as a positive anchor.

## Slide 16 - What Words Drive the Pain Points?
The keyword chart shows the words and matched terms behind the aspect sentiment. This helps translate aspect-level results into more concrete business actions.

## Slide 17 - Preprocessing Pipeline
This section introduces the Vietnamese-specific preprocessing pipeline used before modeling and aspect analysis.

## Slide 18 - Text Preprocessing: 8 Steps
The preprocessing pipeline standardizes Vietnamese text through Unicode normalization, lowercasing, noise removal, character filtering, whitespace cleanup, tokenization with underthesea, and stopword removal. These steps reduce web-scraping noise and make the text more consistent for the models.

## Slide 19 - Stopword Categories
The stopword list is built from corpus frequency: first we count tokens across the full dataset, then review high-frequency words with low sentiment value, and finally keep a curated Vietnamese stopword list. The table groups the final stopwords into function words, conjunctions, prepositions, pronouns, and quantifiers.

## Slide 20 - Modeling Strategy
This section focuses on weak labels, label quality, neutral handling, and deployable sentiment models.

## Slide 21 - The Core Modeling Challenge
The initial weak labels come from star ratings: 1-2 stars are negative, 3 stars are neutral, and 4-5 stars are positive, with keyword and ABSA conflicts used to adjust labels. The hard part is neutral, because it can mean average, mixed, uncertain, or low-intensity sentiment.

## Slide 22 - Experiment Results: 7 Target Models
This table compares the seven target model families using Accuracy, Macro Recall, and Macro F1. With neutral labels, PhoBERT NeutralBoost 0.9 is the best PhoBERT row; without neutral labels, PhoBERT threshold 0.68 gives the best Macro F1 at 0.960.

## Slide 23 - Consistent Model Comparison
The chart keeps the same seven model targets and compares Macro F1 across neutral settings. The practical recommendation is to keep a binary no-neutral model for high-confidence polarity alerts, and keep the cleaned 3-class setting when neutral interpretation is needed for business analysis.

## Slide 24 - Best Neural Model Convergence
This convergence chart uses the best neural run that has saved epoch history. It is included because neural models have train and validation curves, while classical ML models such as Logistic Regression and Linear SVC do not have epoch-based convergence curves.

## Slide 25 - Practical Value of the Topic
The practical value is that the system converts unstructured employee reviews into measurable sentiment, aspect distribution, and time trends. It also supports market listening and decision support without reading thousands of reviews manually.

## Slide 26 - Summary
The summary slide lists what we built and the key results. The strongest binary polarity F1 is 0.960 from PhoBERT threshold 0.68, the cleaned 3-class F1 is 0.877, the label audit found 1,570 likely issues, and Salary is the biggest pain point with 60% negative sentiment.

## Slide 27 - Selected References
