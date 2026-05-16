# A Scalable Aspect-Based Sentiment Analysis Pipeline for Vietnamese Employee Reviews in the Financial Services and Technology Sectors

---

## Abstract

This report presents a full-stack sentiment analysis pipeline applied to 3,267 Vietnamese employee reviews collected from 1900.com.vn, covering 12 companies across the banking, financial services, and technology sectors from January 2020 to December 2025. The study addresses a concrete business problem: HR managers and organisational leaders lack granular, scalable tools to understand which workplace dimensions drive employee satisfaction or accelerate turnover. The pipeline integrates a distributed data engineering stack — Apache Kafka (KRaft), Spark Structured Streaming, PostgreSQL, and Apache Airflow — with a natural language processing layer comprising rule-based Aspect-Based Sentiment Analysis (ABSA), weak-label generation from star ratings, FastText sentence embeddings (300-dim, frozen), and a six-classifier machine learning ensemble. The best-performing model, a three-layer MLP neural network trained on a 305-dimensional feature space, achieves an overall test accuracy of 85.6% and a macro F1-score of 0.7798. Aspect-level analysis across five workplace dimensions reveals that Work-Life Balance (78.5% negative mentions) and Career Growth (72.6% negative) constitute the most critical pain points, while Work Environment attracts the highest volume of both positive and negative discourse. A central empirical finding is that star ratings systematically underreport dissatisfaction: 28.5% of reviews required re-labelling once the sentiment of the written text was accounted for, exposing a structural gap between perceived and expressed employee experience.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Overview](#2-theoretical-overview)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [Methodology](#5-methodology)
6. [Results](#6-results)
7. [Business Analysis](#7-business-analysis)
8. [Discussion](#8-discussion)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)
- [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Motivation and Research Context

The Vietnamese banking and technology sectors have undergone rapid structural transformation since 2020. Digital transformation mandates have pushed traditional financial institutions to recruit software engineers, data scientists, and product managers at an unprecedented scale, creating direct competition with established technology companies for the same talent pool. Employee review platforms such as 1900.com.vn aggregate free-text feedback from current and former employees, providing an unmediated channel through which workers express their experiences — often with a candour absent from formal HR surveys.

Despite the richness of this data, most organisations rely on the scalar star rating that accompanies each review as their primary signal. This is a blunt instrument. A four-star review may contain substantive paragraphs of grievance about salary compression, opaque promotion criteria, or systematic overtime demands, all hidden inside a number that a manager reads as broadly positive. The gap between the numeric rating and the expressed textual sentiment is the central problem motivating this work.

### 1.2 Research Problem Statement

The fundamental problem is an information asymmetry. Senior leadership and HR departments receive aggregated, high-level signals — average star ratings, quarterly survey scores — that smooth over the specific workplace failures driving attrition. A 4.1 average rating tells an organisation's people team nothing about whether engineers are leaving because of salary, career trajectory, or management style. Without aspect-level granularity, corrective investment is mis-targeted and the underlying drivers of turnover persist.

Existing off-the-shelf sentiment tools compound this problem. They are typically designed for English, operate at the document level, and make no distinction between negative comments about salary and negative comments about work-life balance. Vietnamese, as a tonal, non-segmented language with pervasive code-switching into English technical vocabulary, requires dedicated preprocessing and a domain-specific lexicon.

The root cause of the business problem is therefore the absence of a scalable, aspect-granular sentiment analysis capability operating on Vietnamese-language free text at the scale of thousands of reviews.

### 1.3 Objectives

This project pursues three concrete objectives:

1. **Build an automated data engineering pipeline** that collects, streams, stores, preprocesses, and re-trains on employee review data without manual intervention, orchestrated by a daily Airflow DAG.
2. **Implement an aspect-based sentiment analysis module** that classifies every sentence in a review by the workplace dimension it addresses and assigns a sentiment polarity to each aspect mention.
3. **Train a document-level sentiment classifier** that assigns an overall positive, neutral, or negative label to each review, using a weak-labelling strategy that corrects for the systematic positivity bias in star ratings, and extract actionable business insights from the results.

### 1.4 Contributions

- A working, production-grade distributed pipeline (Airflow + Kafka KRaft + Spark Structured Streaming + PostgreSQL) for continuous ingestion of Vietnamese employee reviews.
- A domain-specific weak-labelling strategy that combines star ratings with ABSA signals to correct for the charitable rating effect, re-labelling 28.5% of the dataset.
- Empirical characterisation of five workplace dimensions across 12 Vietnamese companies, revealing that star ratings underreport dissatisfaction at a negative-to-positive mention ratio of 1.87:1.
- A fully reproducible experimental setup deployed via Docker Compose, with experiment tracking via `models/experiments.json` and `analysis/training_results.json`.

### 1.5 Report Structure

Section 2 surveys the theoretical background. Section 3 describes the dataset. Section 4 details the system architecture. Section 5 presents the methodology. Section 6 reports the quantitative results. Section 7 translates findings into business recommendations. Section 8 discusses the results, limitations, and future work. Section 9 concludes.

---

## 2. Theoretical Overview

### 2.1 Sentiment Analysis: Overview

Sentiment analysis — the computational identification of opinion, attitude, and emotion expressed in text — has grown from a niche research area into a mainstream applied NLP discipline. The task is conventionally stratified by granularity: document-level classification assigns a single polarity label to an entire text; sentence-level analysis operates at the clause; aspect-level analysis isolates which facet of an entity is being discussed before assigning a polarity to that specific facet.

For employee review analytics, document-level classification is insufficient. An employee may praise their colleagues while condemning management, or endorse the compensation structure while complaining about workload. Document-level models trained on such mixed-signal reviews produce noisy labels and downstream predictions that cannot distinguish which organisational dimension is the source of satisfaction or dissatisfaction.

### 2.2 Aspect-Based Sentiment Analysis (ABSA)

ABSA decomposes the sentiment extraction problem into four sub-tasks formalised by the SemEval shared tasks (Pontiki et al., 2014, 2016):

- **Aspect Term Extraction (ATE):** identify the word or phrase in the text that refers to an aspect.
- **Aspect Category Detection (ACD):** classify that term into a predefined ontology of aspect categories.
- **Opinion Term Extraction (OTE):** identify the opinion word or phrase modifying the aspect term.
- **Aspect Sentiment Classification (ASC):** assign a polarity (positive, negative, neutral) to the aspect-opinion pair.

Three broad implementation approaches exist: (i) rule-based keyword and lexicon matching, (ii) supervised sequence-labelling models (BiLSTM-CRF, BERT-based), and (iii) zero-shot prompting of large language models. The choice between them depends on the availability of labelled data, computational budget, and interpretability requirements. This project uses a rule-based approach for reasons detailed in Section 5.2.

### 2.3 Vietnamese NLP

Vietnamese presents several challenges relative to European languages:

- **No word boundaries:** words are not delimited by whitespace; multi-syllable words require segmentation.
- **Tonal script:** six tones change word meaning; diacritic normalisation is critical before any string matching.
- **Code-switching:** Vietnamese tech workers frequently intersperse English terms — *overtime*, *toxic*, *deadline*, *teamwork*, *manager* — within otherwise Vietnamese sentences.
- **Limited labelled resources:** annotated corpora for domain-specific tasks are sparse compared to English.

The `underthesea` library provides Vietnamese word tokenisation, part-of-speech tagging, and named entity recognition. For embedding representations, fastText's pretrained Vietnamese Common Crawl model (`cc.vi.300.bin`, Bojanowski et al., 2017) provides 300-dimensional subword embeddings that generalise to out-of-vocabulary tokens. PhoBERT (Nguyen & Nguyen, 2020) is a BERT-based model pre-trained on a large Vietnamese corpus and represents the current state-of-the-art for Vietnamese NLP classification tasks.

### 2.4 Weak / Distant Supervision for Sentiment Labelling

When labelled training data is unavailable or expensive to obtain, weak supervision uses heuristic functions or auxiliary signals to generate noisy labels programmatically (Ratner et al., 2017). In the context of online reviews, star ratings are the canonical distant supervision signal: reviews with high ratings are assumed positive, low ratings negative. This approach is well-established but carries a known bias — voluntary online reviews systematically over-represent positive ratings (positivity bias), and individual reviewers frequently rate generously while expressing substantive textual dissatisfaction (the charitable rating effect). Correcting for this bias by incorporating text-derived signals into the labelling function is central to the methodology of this study.

### 2.5 Employee Review Analytics

Prior work on employee review platforms (primarily English-language Glassdoor and LinkedIn data) has demonstrated that free-text reviews contain significantly more actionable information than scalar ratings, and that NLP-derived signals correlate with objective organisational outcomes such as turnover rates and innovation indices (Dastin, 2018; Huang et al., 2020). The Vietnamese labour market, and the banking and fintech sectors specifically, remain understudied in this literature. This work addresses that gap while also contributing a reusable open-source pipeline infrastructure.

---

## 3. Dataset

### 3.1 Source and Ethical Considerations

The dataset comprises 3,267 employee reviews scraped from 1900.com.vn, a Vietnamese employment review platform, covering the period from January 2020 to December 2025. All data is publicly accessible without authentication; the crawler respects a 1.5-second inter-request delay (`crawl_delay = 1.5` in `src/config.py`) and honours HTTP error responses without retrying beyond three attempts. No personally identifiable information beyond job title and employee status (current/former) is stored. SHA-256 fingerprints are used for deduplication rather than name-based identifiers.

### 3.2 Dataset Statistics

| Attribute | Value | Notes |
|---|---|---|
| Total reviews | 3,267 | After deduplication |
| Unique companies | 12 | Mixed banking and IT |
| Industries | IT Software 90.5%, Banking 5.2%, Finance 2.7% | Reflects platform user base |
| Current employees | 2,488 (76.2%) | |
| Former employees | 779 (23.8%) | |
| Rating 1 star | 158 (4.8%) | |
| Rating 2 stars | 183 (5.6%) | |
| Rating 3 stars | 867 (26.5%) | |
| Rating 4 stars | 1,272 (38.9%) | |
| Rating 5 stars | 787 (24.1%) | |
| Missing: pros field | 3,015 (92.3%) | Substantive text is in cons |
| Missing: cons field | 783 (24.0%) | |
| Missing: advice field | 2,744 (84.0%) | |

**Table 1: Dataset Overview**

### 3.3 Data Quality and Representativeness

Three data quality issues shape all downstream analytical choices:

**Missingness in the pros field.** 92.3% of records have no content in the pros column. The substantive free-text content of this dataset therefore resides almost entirely in the cons field and the review title. This creates an inherent negativity bias in the raw text corpus that must be accounted for in the weak-labelling strategy (Section 5.3).

**Right-skewed rating distribution.** 63% of reviews carry four or five stars. This is consistent with well-documented positivity bias in voluntary online reviews and means that a naive rating-to-label mapping would generate a severely class-imbalanced training set dominated by the positive class.

**Multilingual code-switching.** The majority of text is Vietnamese, but English terms are common — particularly in the technology sector vocabulary (*overtime*, *toxic*, *deadline*, *teamwork*, *manager*, *career*, *training*, *culture*). This informed the design of the aspect keyword lexicons, which include both Vietnamese and English variants for each aspect.

### 3.4 Company Composition

The largest single contributor is FPT Software (1,808 reviews, 55.3%), a tier-one Vietnamese software outsourcing firm. Smaller contributors include Gameloft Vietnam (361), Viettel (254), CMC Global (229), and several banking institutions — Techcombank (136), FE Credit (87), HDBank (35). This composition means that results are more representative of large technology and outsourcing companies than of the broader Vietnamese labour market. Banking-sector findings should be interpreted as indicative rather than definitive given the limited sample size.

---

## 4. System Architecture

### 4.1 High-Level Pipeline Overview

The system is implemented as a containerised microservices stack orchestrated by Docker Compose. Seven discrete processing stages run sequentially within a daily Apache Airflow DAG:

```
1900.com.vn (public reviews)
        │
        ▼
[httpx Crawler + Bloom Filter dedup]
        │  JSON messages
        ▼
[Apache Kafka KRaft — topics: crawled-companies, crawled-reviews]
        │  Spark Structured Streaming (5s micro-batches)
        ▼
[PostgreSQL 16 — Company + Review tables, SHA-256 fingerprint dedup]
        │  Airflow DAG trigger
        ▼
[NLP Preprocessing — underthesea tokeniser, stopword removal, NFC normalisation]
        │
        ▼
[ABSA Module → Weak Labelling → FastText embeddings (300-dim) + 5 features]
        │
        ▼
[Classifier Training — 5 models + soft-vote ensemble → best model saved]
        │
        ▼
[Streamlit Dashboard / analysis/report.md generation]
```

### 4.2 Data Collection Layer

#### 4.2.1 Web Scraper

The crawler (`src/crawler/scraper.py`, `src/crawler/producer.py`) operates in two stages. Stage 1 crawls the company listing pages at `/review-cong-ty` to build a complete registry of companies. Stage 2 iterates over each company and crawls all paginated review pages.

Each HTTP request is executed with `httpx` and wrapped in a `tenacity` retry decorator (3 attempts, exponential back-off between 2 and 30 seconds). Browser-mimicking headers are sent on every request; an optional session cookie can be injected via the `SESSION_COOKIE` environment variable for authenticated access to full review text. A configurable crawl delay (default 1.5 s) is enforced between requests to avoid overloading the target server.

Every review is assigned a SHA-256 fingerprint computed from its content fields. This fingerprint is used as a unique constraint in the database (`ON CONFLICT DO NOTHING`) so that re-crawling the same page on subsequent DAG runs produces no duplicate records.

#### 4.2.2 Bloom Filter

A persistent Scalable Bloom filter (`src/crawler/bloom_filter.py`, backed by `pybloom-live`) tracks all company and review identifiers seen across crawl sessions. It is serialised to disk (`bloom_filter.bin`) and reloaded at the start of each Airflow run. With a configured capacity of 500,000 entries and an error rate of 0.001, it eliminates redundant HTTP requests for already-crawled URLs before any network call is made, substantially reducing both crawl time and server load on incremental daily runs.

#### 4.2.3 Message Queue: Apache Kafka (KRaft)

The crawler producer (`src/crawler/producer.py`) publishes parsed company and review records to two Kafka topics: `crawled-companies` and `crawled-reviews`. The broker runs in KRaft mode (no ZooKeeper dependency), configured as a single-node cluster. Producer settings: `acks=all`, `retries=3`, UTF-8 JSON serialisation. Log retention is 24 hours, sufficient for the daily batch cadence.

### 4.3 Stream Processing Layer

The Spark consumer (`src/streaming/consumer.py`) runs as a PySpark Structured Streaming application with a `local[*]` master. It subscribes to both Kafka topics, deserialises JSON payloads against predefined `StructType` schemas, and writes records to PostgreSQL via `foreachBatch` callbacks that call SQLAlchemy upsert statements.

Micro-batch processing is triggered every 5 seconds. Checkpoint directories are persisted to a Docker volume (`spark_checkpoints`) to enable at-least-once delivery semantics and recovery from consumer restarts. The Airflow DAG includes polling tasks (`wait_companies`, `wait_reviews`) that confirm Postgres row counts before proceeding to downstream stages.

### 4.4 Storage Layer

The PostgreSQL 16 schema (`src/models.py`) contains two tables:

- **`companies`**: `site_id` (unique, indexed), `name`, `industry`, `employee_range`, `location`, `overall_rating`, `review_count`, `url`.
- **`reviews`**: `fingerprint` (unique, indexed), `company_id` (FK), `title`, `rating`, `job_title`, `employee_status`, `review_date`, `pros`, `cons`, `advice`, `recommends`, `ceo_rating`, `business_outlook`, and preprocessed variants `pros_clean` / `cons_clean`.

Both tables are indexed on their primary lookup columns. Cascade delete is enforced from companies to reviews. The ORM uses SQLAlchemy classic Column syntax for compatibility with both SQLAlchemy 1.4 (bundled in Airflow 2.10) and 2.0.

### 4.5 Orchestration: Apache Airflow

The DAG `crawl_1900_reviews` (`dags/crawl_reviews_dag.py`) schedules the full pipeline daily at 02:00 UTC. The seven tasks execute in strict linear dependency:

```
init_db → produce_companies → wait_companies → produce_reviews → wait_reviews → preprocess → train_model
```

| Task | Timeout | Description |
|---|---|---|
| `init_db` | — | Ensure PostgreSQL schema exists |
| `produce_companies` | 2 h | Crawl listing pages → Kafka |
| `wait_companies` | 5 min | Poll Postgres until company count matches Kafka output |
| `produce_reviews` | 5 h | Crawl all company review pages → Kafka |
| `wait_reviews` | 2 min | 30-second buffer for Spark flush |
| `preprocess` | 1 h | NLP normalisation + tokenisation for new reviews |
| `train_model` | 1 h | Re-train classifiers (auto-skip if data fingerprint unchanged) |

XCom is used to pass record counts between produce and wait tasks. The training task auto-skips when the MD5 fingerprint of the current dataset matches the fingerprint from the previous successful run, avoiding unnecessary compute on days with no new reviews.

### 4.6 Monitoring and Export

A Streamlit application (`src/export/app.py`) provides a live view of the database at port 8501. It displays company and review counts, supports filtering by industry and location, renders training experiment history loaded from `models/experiments.json`, and allows CSV export of the review corpus. The app caches database queries with a 60-second TTL to avoid excessive polling.

### 4.7 Infrastructure

The full stack is defined in `docker-compose.yml` and requires no local installations beyond Docker Engine:

| Service | Image | Port |
|---|---|---|
| `postgres` | `postgres:16-alpine` | 5432 |
| `kafka` | `apache/kafka:3.9.0` | 9092 |
| `spark-consumer` | Custom (`Dockerfile.spark`) | — |
| `airflow-webserver` | `apache/airflow:2.10.4-python3.11` | 8080 |
| `airflow-scheduler` | `apache/airflow:2.10.4-python3.11` | — |
| `streamlit` | Custom (`Dockerfile.streamlit`) | 8501 |

---

## 5. Methodology

### 5.1 Text Preprocessing

All free-text fields are processed through a four-step normalisation pipeline (`src/preprocessing/processor.py`):

1. **Unicode NFC normalisation** — resolves composed vs. decomposed Vietnamese diacritic representations.
2. **Lowercasing and cleaning** — removes URLs, email addresses, residual HTML tags, and non-Vietnamese characters while preserving Vietnamese diacritics and standard punctuation.
3. **Vietnamese word segmentation** — applied using `underthesea.word_tokenize(..., format="text")`, which segments multi-syllable Vietnamese words. Falls back to whitespace tokenisation if the library is unavailable.
4. **Stopword removal** — a domain-specific stopword list (`src/preprocessing/stopwords_vi.py`) built by corpus frequency analysis rather than a generic list, removing high-frequency function words without discarding sentiment-bearing terms.

After preprocessing, reviews with empty processed text are discarded. The effective training pool is **10,000 reviews**.

### 5.2 Rule-Based ABSA

#### 5.2.1 Design Rationale

Three approaches were evaluated for ABSA:

| Approach | Advantage | Disadvantage | Selected? |
|---|---|---|---|
| Fine-tune PhoBERT | High accuracy with labelled data | Requires aspect-annotated ground truth; none exists | No |
| Zero-shot LLM | No labelling required | Slow, high API cost, non-deterministic, unnecessary for narrow domain | No |
| Rule-based keyword + lexicon | Fast, deterministic, auditable, no labels needed | Limited by lexicon coverage | **Yes** |

The rule-based approach is appropriate because (a) the domain is narrow enough that a curated keyword list provides adequate coverage, (b) no aspect-level ground truth exists to supervise a model, and (c) the purpose of ABSA within this pipeline is to generate auxiliary signals for weak labelling rather than to serve as a standalone production-facing inference component. Approximate correctness at scale is more valuable than exact correctness on a small labelled set.

#### 5.2.2 Step 1 — Aspect Term Extraction and Aspect Category Detection (ATE+ACD)

Each review field (`pros`, `cons`, `advice`) is split into sentence fragments by the `_split_sentences` function: primary splits on `[.;\n]+`, secondary splits on commas for fragments exceeding 40 characters. Fragments shorter than 4 characters are discarded.

Each sentence fragment is matched against five aspect categories. Each category is defined by a keyword list of 12–14 terms:

| Aspect | Keywords (sample) |
|---|---|
| Salary & Benefits | lương, thưởng, phúc lợi, thu nhập, đãi ngộ, tăng lương, bảo hiểm, benefit |
| Work Environment | môi trường, văn hóa, đồng nghiệp, teamwork, toxic, drama, culture |
| Management & Leadership | quản lý, lãnh đạo, sếp, manager, leadership, tầm nhìn, chính sách |
| Career Growth | thăng tiến, học hỏi, phát triển, training, career, lộ trình, cơ hội |
| Work-Life Balance | áp lực, overtime, tăng ca, workload, deadline, stress, quá tải, nghỉ phép |

Multi-word keywords are matched before single-word keywords (longest-first ordering). At most one match per aspect per sentence is recorded to avoid double-counting.

#### 5.2.3 Step 2 — Opinion Term Extraction and Aspect Sentiment Classification (OTE+ASC)

Once a sentence is matched to an aspect, the pipeline searches for an opinion word within a ±3-token window centred on the matched keyword (`_find_opinion` in `src/analysis/absa.py`). The opinion lexicon contains 30 positive terms (e.g., *tốt*, *chuyên nghiệp*, *hài lòng*, *ổn định*) and 33 negative terms (e.g., *tệ*, *áp lực*, *toxic*, *không minh bạch*), sorted longest-first to prevent substring masking.

If no opinion word is found within the window, the source column is used as a fallback: sentences from `pros` receive a Positive label, from `cons` a Negative label, and from `advice` a Neutral label. Approximately 25% of aspect mentions reach this fallback. This introduces a directional noise that is acceptable given the auxiliary role of the ABSA module.

### 5.3 Weak Labelling Strategy

#### 5.3.1 Base Label from Star Rating

Star ratings are mapped to initial sentiment labels:

| Rating | Label |
|---|---|
| 1–2 stars | Negative (0) |
| 3 stars | Neutral (1) |
| 4–5 stars | Positive (2) |

#### 5.3.2 Combined Score and Override Rules

The base label is adjusted by a combined score derived from two sources (`src/training/labeling.py`):

- **Keyword score:** counts positive and negative keyword occurrences across all three free-text fields.
- **ABSA score:** aggregates ABSA opinion lexicon hits across the fields, applying differential weights: `pros` × 1.0, `cons` × 1.5 (employees write more substantive complaints in the cons field), `advice` × 0.5.

Three override rules are applied:

1. **Neutral base (rating = 3):** always re-labelled by the combined signal. If `combined ≤ −1.5` → Negative; if `combined ≥ 1.5` → Positive.
2. **Positive base (rating ≥ 4) with `combined ≤ −3`:** flipped to Negative (employee rates generously but text is overwhelmingly negative).
3. **Negative base (rating ≤ 2) with `combined ≥ 3`:** flipped to Positive (rare; typically a data entry anomaly).

In total, **932 reviews (28.5%)** were re-labelled by this procedure.

#### 5.3.3 Resulting Class Distribution

| Class | Count | Percentage | Class Weight |
|---|---|---|---|
| Negative | 2,842 | 28.4% | 1.173 |
| Neutral | 1,086 | 10.9% | 3.069 |
| Positive | 6,072 | 60.7% | 0.549 |

**Table 2: Training Set Class Distribution and Balanced Class Weights**

### 5.4 Feature Engineering

#### 5.4.1 FastText Sentence Embeddings (300-dim, frozen)

The pretrained Vietnamese FastText model `cc.vi.300.bin` (Common Crawl, 300 dimensions) is loaded in frozen mode — no fine-tuning is applied. Each preprocessed review text is converted to a single 300-dimensional sentence embedding using `fasttext.FastText.get_sentence_vector()`, which computes the mean of all token subword vectors. Frozen embeddings provide good generalisation without requiring labelled data for representation learning.

#### 5.4.2 Handcrafted Features (5-dim)

Five scalar features are appended to the embedding:

| Feature | Description |
|---|---|
| `char_len` | Total character count of the processed text |
| `word_count` | Number of whitespace-delimited tokens |
| `excl_ratio` | Ratio of `!` characters to total characters |
| `pos_ratio` | Positive keyword hits / total keyword hits |
| `neg_ratio` | Negative keyword hits / total keyword hits |

These features encode surface-level sentiment intensity signals that complement the distributional semantics of the embedding.

#### 5.4.3 Final Feature Vector

The 300-dimensional embedding and the 5-dimensional handcrafted vector are horizontally concatenated to produce a **305-dimensional** feature matrix. All features are standardised using `sklearn.preprocessing.StandardScaler` fitted on the training split only.

### 5.5 Classifier Training and Evaluation

#### 5.5.1 Train / Validation / Test Split

The 10,000-sample dataset is split 70/15/15, stratified by class label: 7,000 training, 1,500 validation, 1,500 test. The random seed is fixed at 42 across all splits and model initialisations for reproducibility.

#### 5.5.2 Class Imbalance Handling

All applicable classifiers use `class_weight='balanced'`, which internally weights each sample's loss contribution inversely proportional to its class frequency. This corrects for the 60.7% Positive majority without resampling the training data.

#### 5.5.3 Model Configurations

**Logistic Regression:** `C=1.0`, `max_iter=1000`, `class_weight='balanced'`.

**LinearSVC:** `C=1.0`, `max_iter=2000`, `class_weight='balanced'`. Wrapped in `CalibratedClassifierCV` (3-fold cross-validation) to expose `predict_proba` output required for soft-voting ensemble membership.

**Random Forest:** 200 trees, `class_weight='balanced'`, `n_jobs=-1`.

**Gaussian Naïve Bayes:** Default parameters; no class weight support — included as a baseline.

**MLP Neural Network:** Hidden layers `(256, 128, 64)`, ReLU activation, Adam optimiser, adaptive learning rate (initial 0.001), `early_stopping=True` (patience = 15 iterations, `validation_fraction=0.1`), `batch_size=64`, `max_iter=300`.

**Soft-Vote Ensemble:** Averages the `predict_proba` outputs of all five individual classifiers. The class index with the highest average probability is selected as the final prediction.

#### 5.5.4 Evaluation Metrics

Primary metric: **macro F1-score** (unweighted mean of per-class F1, selected because it penalises performance on minority classes equally). Secondary metrics: accuracy, weighted F1, macro precision, macro recall. Per-class precision, recall, F1, and support are reported for the best model. Confusion matrices are reported for all six models in Appendix C.

---

## 6. Results

### 6.1 Model Comparison

All six models were evaluated on the held-out test set (n = 1,500). Results are ranked by macro F1:

| Model | Accuracy | F1 Macro | F1 Weighted | Precision (M) | Recall (M) |
|---|---|---|---|---|---|
| **MLP_NeuralNet** ★ | **0.856** | **0.7798** | **0.8534** | **0.7933** | **0.7685** |
| Ensemble_SoftVote | 0.853 | 0.7752 | 0.8508 | 0.7833 | 0.7679 |
| LogisticRegression | 0.805 | 0.7260 | 0.8172 | 0.7137 | 0.7600 |
| LinearSVC | 0.841 | 0.7259 | 0.8312 | 0.7689 | 0.7055 |
| RandomForest | 0.822 | 0.7201 | 0.8078 | 0.7986 | 0.6816 |
| GaussianNB | 0.747 | 0.6645 | 0.7645 | 0.6593 | 0.7031 |

**Table 3: Test-Set Performance Comparison Across All Models**

The MLP achieves the highest macro F1 (0.7798) and accuracy (85.6%), narrowly ahead of the soft-voting ensemble (0.7752, 85.3%). Both substantially outperform the linear models. Gaussian Naïve Bayes is the weakest performer, consistent with its known limitation on high-dimensional, correlated feature spaces.

### 6.2 Validation vs. Test Consistency

The MLP's validation macro F1 (0.7355) is 4.4 percentage points below the test macro F1 (0.7798), while validation accuracy (0.8433) is 1.3 points below test accuracy (0.856). The upward shift from validation to test is unusual — it suggests that the validation set drawn from the same stratified split may have slightly higher neutral class representation, which is the hardest class to predict. No systematic overfitting is observed; the gap is within the range of natural sampling variance at n = 1,500.

### 6.3 Per-Class Performance (MLP)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.811 | 0.784 | 0.797 | 426 |
| Neutral | 0.664 | 0.583 | 0.621 | 163 |
| Positive | 0.905 | 0.939 | 0.921 | 911 |

**Table 4: MLP Neural Network — Per-Class Classification Report (Test Set, n = 1,500)**

The Positive class is classified with high confidence, expected given its dominant share (60.7%) of the training data and the distinctive vocabulary of the cons field for clearly negative reviews. The Negative class performs well for the same reason. The Neutral class is substantially harder, with an F1 of 0.621. This is not a modelling artefact but a reflection of genuine structural ambiguity: neutral reviews intermix positive and negative clauses without a dominant polarity.

### 6.4 ABSA Results

The ABSA pipeline extracted **8,845 aspect mentions** across 3,267 reviews, yielding an average of **2.71 mentions per review**. Across all mentions, 61.6% are negative, 33.0% are positive, and 5.4% are neutral.

| Aspect | Total | Positive | Negative | Neutral | Neg% | Pos% |
|---|---|---|---|---|---|---|
| Work Environment | 3,544 | 1,568 | 1,810 | 166 | 51.1% | 44.2% |
| Salary & Benefits | 2,083 | 627 | 1,397 | 59 | 67.1% | 30.1% |
| Career Growth | 1,961 | 348 | 1,423 | 190 | 72.6% | 17.7% |
| Management & Leadership | 769 | 293 | 434 | 42 | 56.4% | 38.1% |
| Work-Life Balance | 488 | 81 | 383 | 24 | 78.5% | 16.6% |

**Table 5: ABSA Results by Aspect**

Work Environment dominates by volume (3,544 mentions) because it encompasses a wide range of observable, easy-to-articulate phenomena — colleagues, office space, company culture — that employees naturally comment on regardless of their overall sentiment. Work-Life Balance generates the fewest mentions (488) but the highest negative rate (78.5%), indicating that when employees raise this topic, they almost universally do so to complain.

### 6.5 Star Rating vs. Text Sentiment Gap

63% of reviews carry four or five stars. Yet ABSA shows that negative mentions outnumber positive ones by **1.87:1** across the full corpus. The weak-labelling procedure re-labelled 932 reviews (28.5%), converting a substantial proportion of nominally positive reviews into negative or neutral ones. This gap — the charitable rating effect — is the central empirical finding of this study: employees systematically assign generous numeric scores while simultaneously expressing substantive frustration in the free-text fields.

---

## 7. Business Analysis

### 7.1 Stakeholder Pain-Gain Map

| Stakeholder | Pain (evidence) | Gain observed |
|---|---|---|
| Individual employees | Career Growth 72.6% negative — promotion paths perceived as blocked. Work-Life Balance 78.5% negative — driven by systemic overtime and overload. | Work Environment: 1,568 positive mentions flagging strong colleague relationships as a genuine asset. |
| HR / People teams | Cannot distinguish which aspect drives attrition. 28.5% of reviews have star ratings that contradict the written text. | ABSA now provides aspect-level signal at scale — no manual coding required. |
| Senior leadership | Salary & Benefits: 1,397 negative mentions despite organisations averaging 4+ stars. The average star rating masks a sustained compensation grievance. | Positive Work Environment score provides a defensible employer brand narrative for recruitment. |

**Table 6: Pain-Gain Map by Stakeholder**

### 7.2 Gap Analysis

The gap between the perceived employee experience (as measured by star ratings) and the actual expressed experience (as measured by ABSA) is the central analytical finding of this study. On average, 63% of reviews carry four or five stars. Yet ABSA shows that negative mentions outnumber positive ones by 1.87:1. The mechanism behind this gap is the charitable rating effect: employees tend to assign a generous numeric score while simultaneously expressing substantive frustration in the free-text fields. This means any HR process that relies exclusively on star ratings is systematically blind to the majority of workplace dissatisfaction being expressed on the platform.

### 7.3 Aspect-Level Diagnosis and Intervention Mapping

| Aspect | Key Finding | Priority | Recommended Intervention |
|---|---|---|---|
| Work-Life Balance | 78.5% negative; language indicates systemic overload, not isolated incidents | Critical | Team-level workload monitoring with overtime escalation triggers; enforce minimum notice periods for after-hours requests |
| Career Growth | 72.6% negative; employees distinguish between organic learning environments and opaque promotion paths | Critical | Establish transparent career ladders with published promotion criteria; minimum quarterly career review cadence |
| Salary & Benefits | Highest absolute negative volume (1,397); complaints focus on rate of increase rather than starting level | High | Annual compensation benchmarking exercise against market; communicate salary band logic transparently |
| Management & Leadership | Moderate 56.4% negative; direct line managers score more favourably than senior leadership | Medium | Senior leadership visibility and communication programmes; structured upward feedback surveys |
| Work Environment | Most ambivalent dimension (51.1% negative, 44.2% positive); reflects genuine cross-company variation | Monitor | Company-level disaggregation to identify cultural outliers; reinforce positive colleague relationship culture as differentiator |

**Table 7: Aspect-Level Intervention Mapping**

### 7.4 The Neutral Class as a Business Signal

The difficulty of classifying neutral sentiment (F1 = 0.621) reflects a genuine property of the reviews themselves. Neutral-labelled reviews disproportionately contain mixed-signal language: an employee who writes about reasonable management but difficult workload, or competitive salary but lack of career path, does not fit cleanly into either a positive or negative frame. These employees are precisely those weighing exit decisions. Organisations should treat the neutral segment as a priority cohort for proactive stay-interviews and targeted engagement outreach — the ambivalence expressed in the review text is a leading indicator rather than a stable state.

### 7.5 ROI Framing for HR Leaders

The pipeline presented here incurs zero marginal cost per review at inference time after the initial setup. The alternative — manual coding of free-text reviews by HR analysts — scales linearly with review volume and introduces inter-rater variability. At 3,267 reviews coded at a conservative 3 minutes per review, manual analysis would require approximately 163 person-hours per analysis cycle. The automated pipeline produces the same aspect-level breakdown in under 10 minutes of compute time, enabling continuous monitoring rather than periodic snapshots.

---

## 8. Discussion

### 8.1 Interpretation of Model Results

#### 8.1.1 Why MLP Outperforms Linear Models on FastText Embeddings

The 300-dimensional FastText embedding space encodes complex non-linear semantic relationships between Vietnamese words that a linear classifier cannot exploit. The MLP's three hidden layers (256 → 128 → 64 neurons with ReLU activations) learn compositional feature interactions that capture sentiment-bearing patterns — such as negation of a positive term, or the co-occurrence of an aspect keyword with a distant opinion term — that are invisible to a logistic regression operating on the same raw features. Early stopping (patience = 15) prevents overfitting to the dominant Positive class and contributes to the relatively strong Neutral class recall.

#### 8.1.2 Why Gaussian NB Underperforms

Gaussian Naïve Bayes assumes that all 305 features are mutually independent and normally distributed within each class. Both assumptions are violated: FastText embedding dimensions are highly correlated by construction (subword co-occurrence statistics impose strong statistical dependencies), and the marginal distributions are not Gaussian. The result is a model that cannot effectively use the information in the embedding space, yielding the lowest macro F1 (0.6645) in the comparison.

#### 8.1.3 Why the Neutral Class Remains Hard

The neutral class presents three compounding challenges. First, it is the smallest class in the training data (10.9%), limiting the model's exposure to neutral patterns. Second, neutral reviews are structurally heterogeneous: they include genuinely mixed reviews, reviews that are ambivalent across multiple aspects, and reviews where the employee is factually descriptive without expressing strong polarity. Third, the weak labelling procedure introduces the most uncertainty for this class — borderline neutral labels (rating = 3) are re-labelled by a combined score that is itself imprecise. The business implication is discussed in Section 7.4.

#### 8.1.4 The Source-Column Fallback Trade-off in ABSA

Approximately 25% of aspect mentions reach the source-column fallback because no opinion word is found within the ±3-token window. This introduces a known directional noise: every sentence in the `cons` field that mentions an aspect keyword but lacks a proximate opinion word is labelled Negative. In practice, this is largely correct — the `cons` field does contain complaints — but it will occasionally mislabel a cons-field sentence that is merely descriptive (e.g., "công ty có chính sách overtime" is factual, not evaluative). This noise is accepted because the ABSA module's primary function is weak label generation, not stand-alone prediction.

#### 8.1.5 Generalisation to Other Domains

The keyword lexicons, class weights, and weak-labelling thresholds are all calibrated to the Vietnamese banking and technology labour market as expressed on 1900.com.vn. Applying this pipeline to a different industry or platform would require (a) re-curating the aspect keyword lexicons to reflect domain vocabulary, (b) re-validating the weak-labelling thresholds against a small human-annotated sample, and (c) potentially retraining the FastText embedding or fine-tuning a domain-specific language model if the vocabulary shift is substantial.

### 8.2 Limitations

#### 8.2.1 Absence of Aspect-Level Ground Truth

The most significant limitation of this study is the absence of human-annotated aspect labels. Without ground truth, it is impossible to report ABSA precision and recall; all ABSA statistics reflect the output of the rule-based system itself, not its accuracy relative to human judgement. This means the business findings in Section 7 should be treated as directionally reliable estimates rather than precisely measured quantities.

#### 8.2.2 Dataset Composition Bias

FPT Software alone contributes 55.3% of all reviews. This creates a composition bias: the aggregate statistics for Work Environment, Career Growth, and all other aspects are substantially influenced by FPT Software's specific organisational culture and management structure. Banking-sector companies collectively contribute fewer than 250 reviews, making it impossible to draw statistically reliable sector-level comparisons between banking and technology from this dataset alone.

#### 8.2.3 Weak Labelling Noise

The 28.5% of reviews that were re-labelled by the combined ABSA + keyword score were re-labelled using thresholds (−3.0 / +3.0) set heuristically. These thresholds have not been validated against independent human annotation. It is plausible that some re-labelled reviews were incorrectly flipped, introducing systematic error into the training labels that propagates into the classifier.

#### 8.2.4 Temporal and Sectoral Pooling

Reviews spanning January 2020 to December 2025 are pooled into a single dataset. This period encompasses the COVID-19 pandemic (which drove a widespread shift to remote work), the subsequent return-to-office period, and the 2023–2024 Vietnamese tech sector downturn. Structural changes in how employees discuss overtime, work-life balance, and job security across these periods are absorbed into a single aggregate model.

#### 8.2.5 Static Embeddings

The FastText model is frozen at inference. It was trained on a Common Crawl snapshot that pre-dates some domain-specific terminology and does not adapt to new vocabulary introduced after the training cut-off. A fine-tuned PhoBERT model would capture contextual semantics more accurately, particularly for ambiguous or negated Vietnamese expressions.

#### 8.2.6 Scraper Fragility

The HTML parser (`src/crawler/parser.py`) is tightly coupled to the DOM structure of 1900.com.vn. Any redesign of the platform's review listing or review detail pages will break the extraction layer without triggering an error — the scraper will silently return empty results. Ongoing maintenance and monitoring of the crawl pipeline are required for production use.

### 8.3 Future Work

#### 8.3.1 Human Annotation for Supervised ABSA

The highest-impact improvement to this pipeline is a human-annotated ABSA corpus. Even 500 annotated reviews would enable evaluation of the current rule-based system, fine-tuning of a PhoBERT model for joint ATE+ASC, and comparison of the two approaches under a common benchmark. A two-annotator scheme with inter-rater agreement (Cohen's kappa) would establish the reliability of the annotation and provide a ceiling estimate for automated performance.

#### 8.3.2 PhoBERT Fine-tuning

Once labelled aspect data is available, replacing the rule-based ABSA and frozen FastText embeddings with a fine-tuned PhoBERT model is the most direct path to improved performance. PhoBERT's contextual representations would handle negation, code-switching, and opinion terms separated from their aspect keywords by more than three tokens — all known failure modes of the current system.

#### 8.3.3 Temporal Analysis

Disaggregating results by year and quarter would reveal how sentiment on specific aspects evolved across the 2020–2025 period — in particular, whether Work-Life Balance negativity spiked during high-demand periods and whether Salary & Benefits sentiment shifted after Vietnamese inflation peaked in 2022–2023. The data is already timestamped in the `review_date` column; this analysis requires only an additional groupby step in the ABSA summary pipeline.

#### 8.3.4 Company-Level Disaggregation

The Streamlit dashboard (`src/export/app.py`) already supports company-level filtering. Extending the ABSA summary to produce per-company aspect heatmaps would enable direct benchmarking: organisations could compare their Work-Life Balance or Career Growth negative rate against the sector median, establishing a continuous employer brand monitoring capability.

#### 8.3.5 Real-Time Inference

The Spark consumer (`src/streaming/consumer.py`) currently writes raw reviews to PostgreSQL and stops there. Extending the `foreachBatch` callback to invoke the trained classifier — loading the serialised MLP model and scaler from `models/best_model.pkl` and `models/scaler.pkl` — would convert the pipeline from a daily batch system into a near-real-time HR signal feed. Alert rules could trigger when a company's rolling negative rate on a specific aspect exceeds a configurable threshold.

#### 8.3.6 Expand Aspect Taxonomy

The current five-aspect taxonomy covers the major dimensions of workplace experience but omits several that are salient in the Vietnamese fintech context: remote work policy, internal tooling and technical stack quality, diversity and inclusion, and hiring/onboarding experience. Expanding the keyword lexicons and introducing sub-aspects (e.g., distinguishing base salary from bonus within Salary & Benefits) would increase the diagnostic resolution of the system.

---

## 9. Conclusion

This study demonstrates that free-text employee reviews contain substantially richer and more accurate signals about workplace satisfaction than their accompanying star ratings. A rule-based ABSA pipeline, combined with a star-rating bias correction via weak labelling and a 305-dimensional MLP classifier trained on frozen FastText embeddings, achieves 85.6% overall accuracy and a macro F1-score of 0.7798 on a 1,500-sample held-out test set.

The central empirical finding is not the classifier performance but the gap it reveals: 28.5% of reviews required re-labelling once text sentiment was accounted for, and ABSA shows that negative aspect mentions outnumber positive ones by 1.87:1 despite 63% of reviews carrying four or five stars. Work-Life Balance (78.5% negative) and Career Growth (72.6% negative) are the two dimensions where employee sentiment is most consistently and severely negative, while Work Environment exhibits the highest mention volume and the most ambivalence — reflecting genuine cross-company variation in culture.

For HR teams and organisational leaders in the Vietnamese banking and technology sectors, the practical implication is clear: acting on star ratings alone means acting on a systematically optimistic signal. The pipeline presented here — deployed as a containerised, fully automated stack requiring no manual labelling after initial setup — provides the aspect-level granularity needed to identify which specific workplace failures are driving attrition and to direct corrective investment accordingly.

---

## 10. References

> *Fill in full citations according to your institution's required style (APA 7th or IEEE).*

- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics*, 5, 135–146.
- Nguyen, D. Q., & Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. *Findings of EMNLP 2020*.
- Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). SemEval-2014 Task 4: Aspect based sentiment analysis. *Proceedings of SemEval 2014*.
- Pontiki, M., et al. (2016). SemEval-2016 Task 5: Aspect based sentiment analysis. *Proceedings of SemEval 2016*.
- Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., & Ré, C. (2017). Snorkel: Rapid training data creation with weak supervision. *Proceedings of VLDB 2017*.
- Apache Software Foundation. (2024). Apache Kafka 3.9.0 documentation.
- Apache Software Foundation. (2024). Apache Spark 3.5.3 documentation.
- Apache Software Foundation. (2024). Apache Airflow 2.10.4 documentation.

---

## Appendices

### Appendix A — Full Aspect Keyword Lexicons

| Aspect | Full Keyword List |
|---|---|
| Salary & Benefits | lương, thưởng, phúc lợi, thu nhập, đãi ngộ, hoa hồng, tăng lương, bảo hiểm, trợ cấp, thù lao, lương thưởng, chế độ, benefit |
| Work Environment | môi trường, văn hóa, đồng nghiệp, văn phòng, không khí, teamwork, nhóm, tập thể, cởi mở, thân thiện, hòa đồng, drama, toxic, culture |
| Management & Leadership | quản lý, lãnh đạo, sếp, giám đốc, trưởng phòng, cấp trên, ban lãnh đạo, manager, leadership, ban giám đốc, tầm nhìn, chính sách |
| Career Growth | thăng tiến, học hỏi, phát triển, đào tạo, lộ trình, cơ hội, kỹ năng, training, kinh nghiệm, thăng chức, học được, nâng cao, tiến bộ, career |
| Work-Life Balance | áp lực, overtime, tăng ca, workload, deadline, cân bằng, giờ làm, stress, quá tải, nghỉ phép, nghỉ ngơi, làm thêm, chấm công, giờ giấc |

### Appendix B — Full Model Hyperparameter Configurations

| Model | Key Parameters |
|---|---|
| Logistic Regression | C=1.0, max_iter=1000, class_weight='balanced', random_state=42 |
| LinearSVC | C=1.0, max_iter=2000, class_weight='balanced', random_state=42; wrapped in CalibratedClassifierCV(cv=3) |
| Random Forest | n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1 |
| Gaussian NB | Default parameters |
| MLP Neural Network | hidden_layer_sizes=(256,128,64), activation='relu', solver='adam', learning_rate='adaptive', learning_rate_init=0.001, max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, batch_size=64, random_state=42 |
| Ensemble | Soft voting: average of predict_proba outputs from all 5 classifiers |

### Appendix C — Confusion Matrices (Test Set)

**MLP Neural Network**

| | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---|---|---|
| Actual Negative | 334 | 35 | 57 |
| Actual Neutral | 35 | 95 | 33 |
| Actual Positive | 43 | 13 | 855 |

**Ensemble (Soft Vote)**

| | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---|---|---|
| Actual Negative | 329 | 41 | 56 |
| Actual Neutral | 37 | 97 | 29 |
| Actual Positive | 44 | 14 | 853 |

**Logistic Regression**

| | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---|---|---|
| Actual Negative | 320 | 73 | 33 |
| Actual Neutral | 35 | 110 | 18 |
| Actual Positive | 53 | 80 | 778 |

**LinearSVC**

| | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---|---|---|
| Actual Negative | 336 | 27 | 63 |
| Actual Neutral | 55 | 62 | 46 |
| Actual Positive | 40 | 8 | 863 |

**Random Forest**

| | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---|---|---|
| Actual Negative | 267 | 23 | 136 |
| Actual Neutral | 41 | 71 | 51 |
| Actual Positive | 14 | 2 | 895 |

**Gaussian Naïve Bayes**

| | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---|---|---|
| Actual Negative | 280 | 94 | 52 |
| Actual Neutral | 39 | 105 | 19 |
| Actual Positive | 69 | 106 | 736 |

### Appendix D — Docker Compose Service Topology

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                  │
│                                                          │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │ postgres │◄───│  kafka    │◄───│  spark-consumer  │  │
│  │ :5432    │    │  :9092    │    │  (Structured     │  │
│  └──────────┘    └───────────┘    │   Streaming)     │  │
│       ▲                           └──────────────────┘  │
│       │                                                  │
│  ┌────┴──────────────────────────────────────────────┐  │
│  │                 Airflow                            │  │
│  │  airflow-init → airflow-webserver(:8080)           │  │
│  │              → airflow-scheduler                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────┐                                   │
│  │  streamlit :8501 │  (Review Explorer & Export)       │
│  └──────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

### Appendix E — Airflow DAG Task Dependency Chain

```
init_db
   │
   ▼
produce_companies (timeout: 2h)
   │
   ▼
wait_companies (timeout: 5min — polls Postgres row count)
   │
   ▼
produce_reviews (timeout: 5h)
   │
   ▼
wait_reviews (timeout: 2min — 30s Spark flush buffer)
   │
   ▼
preprocess (timeout: 1h — underthesea tokenisation)
   │
   ▼
train_model (timeout: 1h — auto-skip if data fingerprint unchanged)
```

Schedule: `0 2 * * *` (daily at 02:00 UTC)

### Appendix F — Sample ABSA Output

| Sentence | Aspect | Opinion Word | Sentiment |
|---|---|---|---|
| phúc lợi tốt | Salary & Benefits | tốt | Positive |
| đồng nghiệp hơi drama | Work Environment | drama | Negative |
| vui vẻ hoà đồng thân thiện | Work Environment | thân thiện | Positive |
| lương tăng chậm | Salary & Benefits | chậm | Negative |
| chính sách hỗ trợ áp dụng hợp lý | Management & Leadership | hỗ trợ | Positive |
| cân bằng cuộc sống ổn định | Work-Life Balance | ổn định | Positive |
| học được cách giao tiếp | Career Growth | học được | Positive |
| làm tài chính nên rất áp lực | Work-Life Balance | áp lực | Negative |
| quản lý không ổn | Management & Leadership | không ổn | Negative |
| không có cơ hội phát triển | Career Growth | không có | Negative |

*Source: `analysis/absa_details.csv`*
