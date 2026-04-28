# Slide Content: Vietnamese Company Review Sentiment Analysis

---

## Section 1: Problem Statement

### Slide 1.1 — Motivation

Star ratings alone are insufficient to guide HR decisions. A 4-star review may still contain serious complaints about salary or career growth. To extract nuanced, actionable insights, we need to classify review text into sentiment categories and identify which workplace aspects are driving dissatisfaction.

**Why this problem?**

- Vietnamese NLP is under-resourced — limited labeled corpora exist for domain-specific sentiment tasks
- Standard tools (VADER, TextBlob) are English-only; domain adaptation to Vietnamese is non-trivial
- Workplace reviews are naturally multi-aspect → motivates Aspect-Based Sentiment Analysis (ABSA)
- Banks and fintech companies competing for talent need granular employee feedback analysis, not just average star ratings
- Class imbalance (60.7% positive) and noisy user-assigned ratings are real-world ML challenges worth studying

**Research objectives:**

1. Build a three-class sentiment classifier (Positive / Neutral / Negative) for Vietnamese workplace reviews
2. Apply Aspect-Based Sentiment Analysis (ABSA) to identify which workplace dimensions drive satisfaction or dissatisfaction
3. Design a complete end-to-end pipeline from data collection to insight visualization

---

### Slide 1.2 — Dataset

**Self-collected dataset** — no pre-existing labeled corpus was available for this domain. We constructed the dataset by crawling 1900.com.vn, a public Vietnamese employer review platform.

> _"We constructed a novel dataset by crawling 10,000 Vietnamese workplace reviews from 1900.com.vn — no pre-existing labeled corpus was available for this domain."_

**Collection Pipeline:**

```
1900.com.vn (/review-cong-ty)
  → Scraper traverses 250 listing pages (~20 companies/page)
  → For each company: crawl all review pages (/danh-gia-dn/{slug}-{id}?page=N)
  → Bloom Filter: deduplicate already-crawled URLs
  → Kafka Producer → Kafka Consumer
  → NLP Preprocessing
  → Store in PostgreSQL
  → Export to CSV
```

Airflow DAG schedules the crawl daily at 02:00 AM, enabling incremental data collection.

**Dataset Statistics:**

| Property         | Value                                 |
| ---------------- | ------------------------------------- |
| Total reviews    | 10,000                                |
| Primary industry | Banking / Fintech (HDBANK focus)      |
| Language         | Vietnamese (with mixed English terms) |
| Export file      | `1900_export_reviews.csv`             |

> Note: "1900" in the filename refers to the website name (1900.com.vn), not the record count.

**Schema:**

| Field             | Description                               |
| ----------------- | ----------------------------------------- |
| `company`         | Company name                              |
| `industry`        | Industry sector (e.g., Ngân hàng)         |
| `rating`          | 1–5 star rating                           |
| `job_title`       | Reviewer's position/role                  |
| `employee_status` | Current employee / Former employee        |
| `location`        | Work location (Hà Nội, Hồ Chí Minh, etc.) |
| `date`            | Review date                               |
| `pros`            | Positive aspects (free text)              |
| `cons`            | Negative aspects (free text)              |
| `advice`          | Suggestions to management                 |

**Label Distribution (imbalanced):**

| Class           | Count | Percentage |
| --------------- | ----- | ---------- |
| Positive (4–5★) | 6,072 | **60.7%**  |
| Negative (1–2★) | 2,842 | **28.4%**  |
| Neutral (3★)    | 1,086 | **10.9%**  |

→ _Chart: `chart_class_distribution.png`_

---

## Section 2: Preprocessing Pipeline

### Slide 2.1 — Text Preprocessing for Vietnamese

Vietnamese text requires dedicated preprocessing steps that differ significantly from English — in particular, word segmentation is non-trivial due to multi-syllable compound words.

**8-Step Pipeline** (`src/preprocessing/processor.py`):

| Step | Operation                        | Purpose                                                    |
| ---- | -------------------------------- | ---------------------------------------------------------- |
| 1    | **Unicode NFC Normalization**    | Standardize Vietnamese diacritics (à, á, ả, ã, ạ, ă, ắ...) |
| 2    | **Lowercase**                    | Reduce vocabulary size, normalize case                     |
| 3    | **Remove URLs & Emails**         | Strip noise: `http(s)://...`, `@email`                     |
| 4    | **Strip HTML Tags**              | Remove leaked HTML from web scraping                       |
| 5    | **Character Whitelist**          | Keep Vietnamese characters + digits + `.!?;:-` only        |
| 6    | **Collapse Whitespace**          | Normalize excessive spacing                                |
| 7    | **Tokenization** (`underthesea`) | Vietnamese-specific word segmentation                      |
| 8    | **Stopword Removal**             | Remove function words, pronouns, prepositions              |

**Flow Diagram:**

```
Raw Review Text
      │
      ▼
  [1] Unicode NFC
      │
      ▼
  [2] Lowercase
      │
      ▼
  [3] Remove URLs / Emails
      │
      ▼
  [4] Strip HTML Tags
      │
      ▼
  [5] Character Whitelist
      │
      ▼
  [6] Collapse Whitespace
      │
      ▼
  [7] underthesea Tokenizer  ←── Vietnamese word segmentation
      │
      ▼
  [8] Stopword Removal
      │
      ▼
  Clean Token Sequence
```

**Stopword Categories** (`src/preprocessing/stopwords_vi.py`):

- **Vietnamese function words**: có, được, là, làm, để, cho, sẽ, đã, đang, bị, phải, nên, cần, muốn...
- **Conjunctions & particles**: và, với, nhưng, nếu, khi, mà, vì, hay, hoặc...
- **Prepositions**: trong, trên, dưới, sau, trước, giữa, qua, đến, vào, ra...
- **Pronouns**: tôi, bạn, anh, chị, em, mình, họ, ta, chúng, người, ai...
- **Quantifiers**: các, những, mỗi, mọi, tất, cả, một, nhiều, ít...
- **Compound split fragments**: ty, trường, nghiệp, viên, triển, thiện...
- **English stopwords**: the, and, is, of, to, in, for, a, an, it, at, be...

**Key tool**: `underthesea` — a Python library purpose-built for Vietnamese NLP that handles multi-syllable compound words (e.g., "Hồ_Chí_Minh", "phát_triển").

---

## Section 3: Model & Training

### Slide 3.1 — Feature Engineering

**Two feature sources are combined into a single 305-dimensional input vector:**

**1. FastText Embeddings (300-dim)**

- Pretrained model: `cc.vi.300.bin` (Facebook AI, trained on Vietnamese Common Crawl)
- Weights are **frozen** — used as a feature extractor, not fine-tuned
- Captures semantic similarity across Vietnamese vocabulary
- Generates one 300-dim sentence embedding per review (mean pooling over tokens)

**2. Handcrafted Features (5-dim)**

| Feature      | Description                         |
| ------------ | ----------------------------------- |
| `char_len`   | Character length (normalized)       |
| `word_count` | Number of tokens post-tokenization  |
| `excl_ratio` | Ratio of exclamation marks (!)      |
| `pos_ratio`  | Ratio of positive keywords detected |
| `neg_ratio`  | Ratio of negative keywords detected |

**Final input vector: 300 + 5 = 305 dimensions per review**

---

### Slide 3.2 — Labeling Strategy (Weak Supervision)

User-assigned star ratings are noisy — a reviewer may give 4 stars but write an overwhelmingly negative review. We apply **weak supervision** to correct label noise.

**Base labels:**

- 1–2★ → Negative
- 3★ → Neutral
- 4–5★ → Positive

**Override rules** (`src/training/labeling.py`):

1. **Keyword signal**: Count positive/negative keywords across full text (pros + cons + advice)
2. **ABSA signal**: Score aspect-related opinion terms; `cons` column weighted **1.5×** (employees detail complaints more thoroughly)
3. **Combined signal**:
   - Neutral (3★) + combined signal ≤ −1.5 → **flip to Negative**
   - Neutral (3★) + combined signal ≥ +1.5 → **flip to Positive**
   - Positive (4–5★) + ABSA score ≤ −3 → **flip to Negative**

---

### Slide 3.3 — Class Imbalance Handling

**Method**: `class_weight='balanced'` on all sklearn estimators (weighted loss function)

**Computed weights:**

| Class    | Weight                            |
| -------- | --------------------------------- |
| Neutral  | **3.07×** (smallest class, 10.9%) |
| Negative | 1.17×                             |
| Positive | 0.55× (majority class, 60.7%)     |

---

### Slide 3.4 — Experiment Results

**6 models trained and evaluated** on an 70/15/15 train/val/test split (10,000 samples):

| Model                 | Accuracy  | F1 Macro  | F1 Weighted | Precision | Recall    |
| --------------------- | --------- | --------- | ----------- | --------- | --------- |
| **MLP Neural Net** ⭐ | **85.6%** | **0.780** | **0.853**   | **0.793** | **0.769** |
| Ensemble (Soft Vote)  | 85.3%     | 0.775     | 0.851       | 0.783     | 0.768     |
| Linear SVC            | 84.1%     | 0.726     | 0.831       | 0.769     | 0.706     |
| Logistic Regression   | 80.5%     | 0.726     | 0.817       | 0.714     | 0.760     |
| Random Forest         | 82.2%     | 0.720     | 0.808       | 0.799     | 0.682     |
| Gaussian NB           | 74.7%     | 0.665     | 0.765       | 0.659     | 0.703     |

→ _Charts: `chart_model_comparison.png`, `chart_per_class_f1.png`_

**Best Model — MLP Neural Net** (3 hidden layers: 256 → 128 → 64, Adam optimizer, early stopping):

**Per-class performance on test set (n=1,500):**

| Class         | Precision | Recall    | F1-Score  | Support |
| ------------- | --------- | --------- | --------- | ------- |
| Positive      | 0.905     | 0.939     | **0.921** | 911     |
| Negative      | 0.811     | 0.784     | **0.797** | 426     |
| Neutral       | 0.664     | 0.583     | **0.621** | 163     |
| **Macro Avg** | **0.793** | **0.769** | **0.780** | 1,500   |

**Confusion Matrix (Best Model):**

```
                  Predicted
                  Neg   Neu   Pos
Actual  Negative  334    35    57
        Neutral    35    95    33
        Positive   43    13   855
```

**Key observations:**

- Positive class achieves highest F1 (0.921) — benefits from most training data
- Neutral class is the hardest (F1 = 0.621) — smallest support (163 samples, 10.9%)
- 57 negatives misclassified as positive (13.4% false negative rate) — worth monitoring for HR risk detection

→ _Chart: `chart_confusion_matrix.png`_

---

## Section 4: Business Insights

### Slide 4.1 — Aspect-Based Sentiment Analysis (ABSA)

**Approach**: Rule-based keyword matching with lexicon-based opinion extraction (`src/analysis/absa.py`)

**5 Workplace Aspects analyzed:**

| Aspect                  | Positive Mentions | Negative Mentions | Total     | Negative % |
| ----------------------- | :---------------: | :---------------: | --------- | :--------: |
| Work-Life Balance       |        81         |        383        | 464       | **83%** 🔴 |
| Career Growth           |        348        |       1,423       | 1,771     | **80%** 🔴 |
| Management & Leadership |        293        |        434        | 727       |   60% 🟡   |
| Work Environment        |       1,568       |       1,810       | 3,378     |   54% 🟡   |
| Salary & Benefits       |       1,617       |       1,397       | 3,014     |   46% 🟢   |
| **Total**               |     **3,907**     |     **5,447**     | **9,354** |            |

**Opinion Term Extraction (OTE):**

- Scan a ±3 token window around each detected aspect keyword
- Match against positive (24 terms) and negative (32 terms) opinion lexicons
- Fallback: if no opinion word found, infer polarity from source field (`pros` → positive, `cons` → negative)

→ _Chart: `chart_absa_summary.png`_

---

### Slide 4.2 — Key Findings & Business Implications

**Finding 1 — Work-Life Balance & Career Growth are the most critical pain points**

- Work-Life Balance: 83% negative (overtime, áp lực, quá tải, stress, deadline)
- Career Growth: 80% negative (limited promotion tracks, insufficient training)
- **Implication**: HR should prioritize structured career development programs and workload management policies

**Finding 2 — Salary receives mixed signals**

- Nearly equal positive (1,617) and negative (1,397) mentions (46% negative)
- Compensation is competitive but inconsistently applied
- **Implication**: Standardize salary banding and benchmark against fintech market rates

**Finding 3 — Work Environment is the most-discussed aspect**

- 3,378 total mentions — both the most praised (1,568) and one of the most complained about (1,810)
- **Implication**: Culture is highly visible and polarizing; team-level culture varies significantly

**Finding 4 — Management has majority negative sentiment**

- 60% negative (434 vs. 293)
- Keywords: thiếu tầm nhìn (lack of vision), không công bằng (unfair), micro-manage
- **Implication**: Leadership training and 360-degree feedback mechanisms recommended

**Recommended HR Priority Roadmap:**

| Priority   | Area              | Action                                               |
| ---------- | ----------------- | ---------------------------------------------------- |
| 🔴 High    | Work-Life Balance | Review overtime policies; flexible work arrangements |
| 🔴 High    | Career Growth     | Structured promotion tracks; L&D budget              |
| 🟡 Medium  | Management        | Leadership coaching; 360-degree feedback             |
| 🟡 Medium  | Work Environment  | Culture audit; team-level action plans               |
| 🟢 Monitor | Salary & Benefits | Compensation benchmarking                            |

---

### Slide 4.3 — System Architecture & Demo

**End-to-End Pipeline:**

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Airflow   │────▶│   Crawler    │────▶│ Preprocessor │────▶│  PostgreSQL  │
│ (Daily 2AM) │     │ + Bloom Filt │     │  underthesea  │     │   (ORM)      │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                               ┌──────▼───────┐
                    ┌─────────────┐                            │   FastText   │
                    │  Streamlit  │◀───────────────────────────│  + MLP Model │
                    │  Dashboard  │                            │  + ABSA      │
                    └─────────────┘                            └──────────────┘
```

**Streamlit Dashboard features** (`analysis/dashboard.py`):

- Filters: company, industry, sentiment, rating range, date range
- KPIs: total reviews, avg rating, sentiment breakdown, unique companies
- Charts: sentiment trends over time, rating by company, ABSA aspect breakdown, top keywords in pros/cons, review distribution by location

**Run locally:**

```bash
streamlit run analysis/dashboard.py --server.port 8502
```
