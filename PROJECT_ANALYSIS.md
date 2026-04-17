# 1900.com.vn Sentiment Analysis — Project Analysis

## OVERVIEW

Pipeline phân tích sentiment reviews công ty Việt Nam từ 1900.com.vn.
Data đã crawl sẵn trong `data_post_processing/1900_export_reviews.csv` (10,000 reviews).

## PIPELINE

```
CSV (10k reviews) → Preprocess (underthesea tokenize + stopwords)
    → FastText cc.vi.300.bin (FROZEN, 300-dim embeddings)
    → Extra features (char_len, word_count, excl_ratio, pos/neg keyword ratio)
    → 305-dim feature vector
    → Train 5 models + Ensemble → Evaluate → Save best
    → Visualize trên Streamlit tab Experiments
```

## LABELING

- Rating 1-2★ = negative, 3★ = neutral, 4-5★ = positive
- Weak label: keyword signals adjust borderline (neutral) reviews

## MODELS (6 entries)

| # | Model             | Type       | Balance       |
|---|-------------------|------------|---------------|
| 1 | LogisticRegression | Individual | weighted loss |
| 2 | LinearSVC          | Individual | weighted loss |
| 3 | RandomForest       | Individual | weighted loss |
| 4 | GaussianNB         | Individual | -             |
| 5 | MLP_NeuralNet      | Neural Net | early stopping|
| 6 | Ensemble_SoftVote  | Ensemble   | soft voting   |

## DATA

- Source: `data_post_processing/1900_export_reviews.csv`
- 10,000 reviews, columns: company, industry, rating, title, pros, cons, advice, recommends
- Rating distribution: 1★=2245, 2★=726, 3★=2108, 4★=2667, 5★=2254
- Text chủ yếu ở cột `cons` (9,528 rows null `pros`)

## COMMANDS

```bash
# Train (chạy trên server, tắt SSH vẫn chạy)
nohup python run.py train --force > train.log 2>&1 &

# Xem log
tail -f train.log

# Xem kết quả trên Streamlit
nohup streamlit run src/export/app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
# Mở browser: http://ssh.openinfra.space:8501 → tab Experiments
```

## FILE STRUCTURE

```
src/
├── training/
│   ├── labeling.py      — Rating → sentiment + weak labeling
│   ├── balancing.py     — Class weight computation
│   ├── trainer.py       — Full pipeline: embeddings + 5 models + ensemble
│   └── experiment.py    — Experiment logging (JSON)
├── preprocessing/
│   └── processor.py     — NLP: normalize, tokenize, stopwords
├── export/
│   └── app.py           — Streamlit UI (4 tabs: Companies, Reviews, Export, Experiments)
├── models.py            — SQLAlchemy ORM
├── config.py            — Pydantic settings
└── database.py          — DB connection

models/                  — Saved models + experiments.json
data_post_processing/    — CSV data (already crawled)
dags/                    — Airflow DAG (crawl pipeline)
```

## DEPENDENCIES

Python 3.11+, fasttext-wheel, scikit-learn, imbalanced-learn, underthesea,
pandas, streamlit, plotly, joblib, sqlalchemy, psycopg2-binary
