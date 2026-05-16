# Speaker Script - Employee Review Intelligence Dashboard

## Slide 1 - Title
Good morning everyone. We are Group 16, and our topic is Employee Review Intelligence Dashboard. The project turns Vietnamese workplace reviews into sentiment signals and business priorities.

## Slide 2 - Context & Problem
This section explains why employee review analytics matters and what problem the dashboard is designed to solve.

## Slide 3 - Why Employee Review Analytics Matters
The slide shows three business signals: only 21% of global employees are engaged, engagement decline is linked to a 438-billion-dollar productivity loss, and people analytics is a growing market. This motivates faster ways to detect workforce problems from public feedback.

## Slide 4 - Job - Pain - Gain
The job is to understand what employees complain about, where it happens, and how urgent it is. The pain is that reviews are unstructured and noisy, while the gain is a dashboard that turns raw reviews into sentiment trends, critical aspects, and action priorities.

## Slide 5 - Project Goals
The project has four goals: build the data pipeline, train sentiment models, extract business insights, and deliver a dashboard. These goals cover both the technical pipeline and the final decision-support output.

## Slide 6 - Dataset Collection
The dataset was self-collected from 1900.com.vn. The collection flow goes from scraping company and review pages, to deduplication with a Bloom filter, ingestion through Kafka, storage in PostgreSQL, and Vietnamese text preparation.

## Slide 7 - Dataset Snapshot
Each review contains content fields, a star rating, company context, and time or profile metadata. Because the labels are imbalanced, with positive reviews as the largest class, Macro F1 is more informative than accuracy alone.

## Slide 8 - Preprocessing Pipeline
This section introduces the Vietnamese-specific preprocessing pipeline used before modeling and aspect analysis.

## Slide 9 - Text Preprocessing: 8 Steps
The preprocessing pipeline standardizes Vietnamese text through Unicode normalization, lowercasing, noise removal, character filtering, whitespace cleanup, tokenization with underthesea, and stopword removal. These steps reduce web-scraping noise and make the text more consistent for the models.

## Slide 10 - Stopword Categories
The stopword list covers Vietnamese function words, conjunctions, prepositions, pronouns, quantifiers, English stopwords, and compound fragment tokens. This is needed because the reviews contain both Vietnamese language patterns and mixed-language noise.

## Slide 11 - Modeling Strategy
This section focuses on weak labels, label quality, neutral handling, and deployable sentiment models.

## Slide 12 - The Core Modeling Challenge
The initial weak labels come from star ratings: 1-2 stars are negative, 3 stars are neutral, and 4-5 stars are positive, with keyword and ABSA conflicts used to adjust labels. The hard part is neutral, because it can mean average, mixed, uncertain, or low-intensity sentiment.

## Slide 13 - Experiment Design
We tested four versions: original 3-class, cleaned 3-class, binary no-neutral, and mixed-conflict 4-class. The model families include TF-IDF Word+Char models and FastText plus MLP, so the comparison tests both label strategy and model family.

## Slide 14 - Confident Learning
Confident learning was used to audit label quality. It trains out-of-fold probabilistic models, estimates unlikely labels, and exports suspicious rows; in this dataset, it flagged 1,570 likely label issues.

## Slide 15 - Result: Label Quality Was the Turning Point
This slide compares the main label settings under the same 70/15/15 split design. Binary no-neutral reaches the highest Macro F1 at 0.918, while cleaned 3-class reaches 0.877, much higher than the original 3-class result of 0.720.

## Slide 16 - Model Comparison
This chart compares selected representative models, not only TF-IDF. FastText + MLP is included with Macro F1 0.728, while FastText + LSTM reaches 0.700 and FastText + RandomForest reaches 0.648; the stronger TF-IDF results show that label quality was more important than adding model complexity.

## Slide 17 - Neutral Handling
The slide answers whether neutral should be kept. Binary is best for high-confidence polarity alerting, but cleaned 3-class is better when neutral interpretation is needed for business analysis.

## Slide 18 - Result Benchmarking
The benchmark slide explains why the results are reasonable. Prior work shows that 3-class sentiment is difficult and neutral is often the hardest class, so the original 3-class F1 of 0.720 is not unusually low for noisy weak labels.

## Slide 19 - Business Insights
This section moves from model performance to business insight generation using aspect-based sentiment analysis.

## Slide 20 - Aspect Sentiment Overview
The chart summarizes sentiment by workplace aspect. Salary & Benefits is the clearest negative outlier, while Work Environment has the largest discussion volume.

## Slide 21 - Critical Issues by Business Priority
The priority slide highlights Salary & Benefits and Work Environment as P1 issues because they have high negative volume and high negative ratios. Facilities & Tools is P2 because it has the largest negative-ratio increase.

## Slide 22 - Which Issues Are Getting Worse?
The trend chart focuses on issues that are moving upward in negativity. The watchlist includes Facilities & Tools, Work Environment, and Work-Life Balance.

## Slide 23 - Where Negative Signals Concentrate
The heatmaps show where negative signals concentrate by industry and by company-volume group. This makes the analysis more targeted than an overall summary.

## Slide 24 - What Words Drive the Pain Points?
The keyword chart shows the words and matched terms behind the aspect sentiment. This helps translate aspect-level results into more concrete business actions.

## Slide 25 - Key Business Findings
There are four findings on this slide: Salary & Benefits is the main pain point, Work Environment is polarizing, Facilities & Tools is an emerging risk, and the dashboard supports prioritization from raw reviews to follow-up actions.

## Slide 26 - Practical Value of the Topic
The practical value is that the system converts unstructured employee reviews into measurable sentiment, aspect distribution, and time trends. It also supports market listening and decision support without reading thousands of reviews manually.

## Slide 27 - End-to-End Project Flow
The architecture starts from manual batch execution, then scraper and Bloom filter, Kafka queue, PostgreSQL storage, Vietnamese preprocessing, sentiment models, ABSA, and finally the Streamlit dashboard. This shows how the project connects data collection to business sentiment analysis.

## Slide 28 - Summary
The summary slide lists what we built and the key results. The strongest binary polarity F1 is 0.918, the cleaned 3-class F1 is 0.877, the label audit found 1,570 likely issues, and Salary is the biggest pain point with 60% negative sentiment.

## Slide 29 - Selected References
The final slide lists the main references used to support the project, including workforce engagement reports, sentiment benchmark studies, and confident learning for label quality.
