# Final Full-Dataset Results

## Dataset scope

- Vietnamese corpus: `72,100` reviews
- English corpus: `838,566` reviews
- Source table: [model_statistics_latest.csv](/D:/Project/business%20analysis/sentimentanalysis/models/statistics/model_statistics_latest.csv:1)
- Run manifest: [full_experiment_manifest.json](/D:/Project/business%20analysis/sentimentanalysis/models/statistics/full_experiment_manifest.json:1)

## How to read this file

This file is organized by **model family**, not by one long leaderboard.

- Each family keeps `1-2` representative best models.
- For Vietnamese, rows using `rating_metadata` are shown separately as engineered-system references, not as the fairest text-only comparison.
- Transformer numbers here come from the current statistics export, which uses thresholded outputs for `PhoBERT` / `DistilBERT`.

## Vietnamese results (72,100)

### TF-IDF

| Model | Test Macro F1 | Test Accuracy | Notes |
|---|---:|---:|---|
| `TFIDF_WordChar_LogisticRegression` | 0.8820 | 0.8820 | Strongest text-only TF-IDF baseline |
| `TFIDF_WordChar_LogisticRegression_C2_char36` | 0.8802 | 0.8802 | Very close variant |

### TF-IDF + Cues

| Model | Test Macro F1 | Test Accuracy | Notes |
|---|---:|---:|---|
| `TFIDF_WordCharCue_LinearSVC` | 0.8848 | 0.8849 | Best fair VI champion in current full benchmark |
| `TFIDF_WordCharCue_LogisticRegression` | 0.8833 | 0.8833 | Nearly tied with the best cue-based model |

### FastText

| Model | Test Macro F1 | Test Accuracy | Notes |
|---|---:|---:|---|
| `FastText_MLP_NeuralNet_Tuned` | 0.7844 | 0.7845 | Best FastText variant on VI |
| `FastText_MLP_NeuralNet` | 0.7838 | 0.7841 | Very close second |

### Transformer

| Model | Test Macro F1 | Test Accuracy | Notes |
|---|---:|---:|---|
| `PhoBERT` | 0.8271 | 0.8325 | Current statistics file uses thresholded transformer output |

### TF-IDF + Metadata

| Model | Test Macro F1 | Test Accuracy | Notes |
|---|---:|---:|---|
| `Custom_VNReviewFusion_LinearSVC` | 0.9901 | 0.9901 | Very strong, but includes metadata and is not a fair text-only benchmark row |
| `Custom_VNReviewFusion_LogReg` | 0.9854 | 0.9855 | Same caution: engineered feature stack |

## English results (838,566)

### TF-IDF

| Model | Test Macro F1 | Test Accuracy | Notes |
|---|---:|---:|---|
| `TFIDF_WordChar_LogisticRegression` | 0.8329 | 0.8342 | Best EN text-only model in current full benchmark |
| `TFIDF_WordChar_LogisticRegression_C2_char36` | 0.8322 | 0.8334 | Very close variant |

### TF-IDF + Cues

| Model | Test Macro F1 | Test Accuracy | Notes |
|---|---:|---:|---|
| `TFIDF_WordCharCue_LogisticRegression` | 0.8325 | 0.8338 | Best cue-based EN row |
| `TFIDF_WordCharCue_LinearSVC` | 0.8290 | 0.8311 | Slightly behind the cue-based LogReg |

### FastText

| Model | Test Macro F1 | Test Accuracy | Notes |
|---|---:|---:|---|
| `FastText_MLP_NeuralNet_Tuned` | 0.7929 | 0.7948 | Best FastText variant on EN |
| `FastText_MLP_NeuralNet` | 0.7923 | 0.7940 | Very close second |

### Transformer

| Model | Test Macro F1 | Test Accuracy | Notes |
|---|---:|---:|---|
| `DistilBERT` | 0.7666 | 0.7703 | Current statistics file uses thresholded transformer output |

## Family-level takeaway

- `TF-IDF` is currently the strongest family on both full datasets.
- `TF-IDF + Cues` gives the best fair Vietnamese result.
- Plain `TF-IDF` gives the best English result.
- `FastText` is clearly behind TF-IDF on both VI and EN, but still useful as a lighter baseline family.
- `PhoBERT` / `DistilBERT` are present, but their current exported statistics are threshold-oriented rather than the cleanest balanced macro-F1 comparison.

## Practical champion picks

- Vietnamese full dataset (`72,100`): `TFIDF_WordCharCue_LinearSVC`
- English full dataset (`838,566`): `TFIDF_WordChar_LogisticRegression`
- Best FastText reference:
  - VI: `FastText_MLP_NeuralNet_Tuned`
  - EN: `FastText_MLP_NeuralNet_Tuned`
- Best transformer reference in current statistics export:
  - VI: `PhoBERT`
  - EN: `DistilBERT`
