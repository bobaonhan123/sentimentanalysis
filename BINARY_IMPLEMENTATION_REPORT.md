# Binary Dissatisfaction MVP Implementation Report

## 1. Scope implemented

The codebase has been extended with a complaint-first binary pipeline for Vietnamese employee reviews:

- Task: `negative` vs `non_negative`
- Objective: prioritize dissatisfaction detection instead of the old 3-class sentiment framing
- Weak supervision: complaint-first rules using rating priors, negative cue patterns, contrastive discourse cues, and aspect-linked complaint signals
- Model family:
  - TF-IDF word+char Logistic Regression
  - TF-IDF word+char Linear SVC with calibration
  - TF-IDF word+char MLP
  - `VADAN_MVP`: binary classifier + aspect-aware complaint branch + validation-tuned fusion

## 2. Files added/updated

- Added [src/training/binary_labeling.py](/d:/Project/business%20analysis/sentimentanalysis/src/training/binary_labeling.py)
- Added [src/training/binary_trainer.py](/d:/Project/business%20analysis/sentimentanalysis/src/training/binary_trainer.py)
- Updated [run.py](/d:/Project/business%20analysis/sentimentanalysis/run.py)

Generated outputs:

- [analysis/binary_labeled_reviews.csv](/d:/Project/business%20analysis/sentimentanalysis/analysis/binary_labeled_reviews.csv)
- [analysis/binary_ambiguous_reviews.csv](/d:/Project/business%20analysis/sentimentanalysis/analysis/binary_ambiguous_reviews.csv)
- [analysis/binary_training_summary.csv](/d:/Project/business%20analysis/sentimentanalysis/analysis/binary_training_summary.csv)
- [analysis/training_results.json](/d:/Project/business%20analysis/sentimentanalysis/analysis/training_results.json)
- [models/binary/VADAN_MVP.pkl](/d:/Project/business%20analysis/sentimentanalysis/models/binary/VADAN_MVP.pkl)
- [models/binary/best_model_meta.json](/d:/Project/business%20analysis/sentimentanalysis/models/binary/best_model_meta.json)

## 3. Labeling outcome

On the current source file [dataset/1900_export_reviews (5).csv](/d:/Project/business%20analysis/sentimentanalysis/dataset/1900_export_reviews%20(5).csv):

- total reviews: `72,100`
- `negative`: `22,928`
- `non_negative`: `49,172`
- `ambiguous`: `11,142`

This is materially less skewed than the old positive-heavy framing and is more aligned with a complaint-detection objective.

## 4. Full-data training result

Full command execution was run successfully through `python run.py train-binary`.

Trainable split:

- train: `42,672`
- val: `9,144`
- test: `9,144`

Best model:

- `TFIDF_WordChar_LinearSVC`
- Negative Precision: `0.7988`
- Negative Recall: `0.9394`
- Negative F1: `0.8634`
- F2-negative: `0.9075`
- PR-AUC-negative: `0.9566`
- Accuracy: `0.8911`

Baseline comparison from [analysis/binary_training_summary.csv](/d:/Project/business%20analysis/sentimentanalysis/analysis/binary_training_summary.csv):

| Model | Neg Precision | Neg Recall | Neg F1 | F2-neg | PR-AUC-neg |
|---|---:|---:|---:|---:|---:|
| TFIDF_WordChar_LinearSVC | 0.7988 | 0.9394 | 0.8634 | 0.9075 | 0.9566 |
| TFIDF_WordChar_LogisticRegression | 0.7904 | 0.9421 | 0.8596 | 0.9073 | 0.9553 |
| VADAN_MVP | 0.7624 | 0.9499 | 0.8459 | 0.9053 | 0.9504 |
| TFIDF_WordChar_MLP | 0.8240 | 0.8523 | 0.8379 | 0.8465 | 0.9272 |

Interpretation for the MVP:

- `TFIDF_WordChar_LinearSVC` is currently the best complaint detector under the chosen selection criterion because it gives the highest F2-negative together with very high negative recall.
- `VADAN_MVP` remains competitive and is still useful as the research-oriented aspect-aware direction, but on the present full-data run it does not beat the calibrated LinearSVC.

## 5. Audit outputs

The full pipeline now exports held-out test predictions and error-analysis files:

- [analysis/binary_best_model_test_predictions.csv](/d:/Project/business%20analysis/sentimentanalysis/analysis/binary_best_model_test_predictions.csv)
- [analysis/binary_false_negatives.csv](/d:/Project/business%20analysis/sentimentanalysis/analysis/binary_false_negatives.csv)
- [analysis/binary_false_positives.csv](/d:/Project/business%20analysis/sentimentanalysis/analysis/binary_false_positives.csv)
- [analysis/binary_label_source_error_summary.csv](/d:/Project/business%20analysis/sentimentanalysis/analysis/binary_label_source_error_summary.csv)
- [analysis/binary_hard_negative_error_summary.csv](/d:/Project/business%20analysis/sentimentanalysis/analysis/binary_hard_negative_error_summary.csv)

Best-model audit summary:

- best model: `TFIDF_WordChar_LinearSVC`
- threshold: `0.22`
- false negatives: `203`
- false positives: `793`

Current audit signal:

- false positives are dominated by `high_rating_prior`
- true negatives are strongly supported by explicit complaint cues such as `drama`, `underpaid`, `toxic`, `late_pay`
- this means the next rule-refinement step should focus more on high-rating reviews whose text is actually mild or mixed, rather than on explicit complaint lexicons

## 6. Important implementation detail

One runtime bug was found and fixed:

- The original capped stratified sampling logic used `groupby(...).apply(...)`, which dropped the `binary_label` column and caused `train_test_split(..., stratify=y)` to fail with `Input y contains NaN`.
- This was replaced with a safer helper `_stratified_cap_sample(...)`.

## 7. Current limitation

`PhoBERT_Binary` was not run in this environment because `torch` is not installed.

Recorded status:

- `missing_dependency: No module named 'torch'`

So the current MVP is fully operational for classical ML and the lightweight aspect-aware binary model, but not yet for transformer fine-tuning.

## 8. Recommended immediate next step

The next focused step should be:

1. audit and refine complaint rules on a manually checked subset of false positives / false negatives
2. build a small gold-labeled evaluation set for dissatisfaction detection
3. only then add PhoBERT fine-tuning as the stronger neural baseline
