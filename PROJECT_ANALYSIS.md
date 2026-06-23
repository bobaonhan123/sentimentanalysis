# 1900.com.vn Workplace Review Analysis - Current Project State

## Overview

The current working dataset is:

- `dataset/1900_export_reviews (5).csv`
- `72,100` Vietnamese workplace reviews

The active research direction is no longer generic 3-class sentiment classification. It is now:

- binary dissatisfaction detection: `negative` vs `non_negative`
- aspect-level negative signal extraction for actionability

## Current Binary Pipeline

```text
72,100 reviews
  -> Vietnamese preprocessing
  -> complaint-first weak labeling
  -> ambiguous review flagging
  -> binary model training
  -> aspect-aware audit outputs
  -> report / dashboard / export artifacts
```

## Current Labeling Outcome

From the active binary weak-labeling pipeline:

- `negative`: `22,928`
- `non_negative`: `49,172`
- `ambiguous`: `11,142`

## Best Full-Run Result

Latest full run on `dataset/1900_export_reviews (5).csv`:

- best model: `TFIDF_WordChar_LinearSVC`
- negative precision: `0.7988`
- negative recall: `0.9394`
- negative F1: `0.8634`
- F2-negative: `0.9075`
- PR-AUC-negative: `0.9566`
- accuracy: `0.8911`

## Key Files

Core pipeline:

- `src/training/binary_labeling.py`
- `src/training/binary_trainer.py`
- `src/common/data_paths.py`

Generated artifacts:

- `analysis/binary_labeled_reviews.csv`
- `analysis/binary_training_summary.csv`
- `analysis/binary_best_model_test_predictions.csv`
- `analysis/binary_false_negatives.csv`
- `analysis/binary_false_positives.csv`
- `models/binary/TFIDF_WordChar_LinearSVC.pkl`

## Commands

```bash
# Full binary training on the active dataset
python run.py train-binary

# Smoke run
python run.py train-binary --max-examples 600

# Predict with the current best binary model
python run.py predict-binary --text "review text here"
```

## Notes

- `data_post_processing/1900_export_reviews.csv` is legacy and no longer the default source.
- `dataset/1900_export_reviews (5).csv` is now the canonical review CSV for the active Python pipeline.
- `PhoBERT_Binary` is still skipped in this environment because `torch` is not installed.
