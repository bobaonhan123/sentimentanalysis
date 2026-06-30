# Binary Dissatisfaction MVP Review Report

## 1. Current status

The repository currently contains a working binary dissatisfaction pipeline for Vietnamese review text:

- task: `negative` vs `non_negative`
- objective: complaint-first detection instead of the old 3-class framing
- current implemented families:
  - TF-IDF word+char classical models
  - FastText 305-d + sklearn / MLP models
  - transformer binary trainers for VI / EN

However, the latest cross-model benchmark artifacts were found to be mixed with non-comparable and partially contaminated result sources. Those result artifacts have been removed from the workspace and must not be used for model selection.

## 2. What remains valid

The following parts are still valid as implementation assets:

- [src/training/binary_labeling.py](/D:/Project/business%20analysis/sentimentanalysis/src/training/binary_labeling.py)
- [src/training/binary_trainer.py](/D:/Project/business%20analysis/sentimentanalysis/src/training/binary_trainer.py)
- [src/training/variant_trainer.py](/D:/Project/business%20analysis/sentimentanalysis/src/training/variant_trainer.py)
- [src/training/fasttext_binary_trainer.py](/D:/Project/business%20analysis/sentimentanalysis/src/training/fasttext_binary_trainer.py)
- [src/training/phobert_binary_trainer.py](/D:/Project/business%20analysis/sentimentanalysis/src/training/phobert_binary_trainer.py)
- [scripts/run_full_experiment_pipeline.py](/D:/Project/business%20analysis/sentimentanalysis/scripts/run_full_experiment_pipeline.py)
- [scripts/build_final_statistics.py](/D:/Project/business%20analysis/sentimentanalysis/scripts/build_final_statistics.py)

The following analysis artifacts are still usable for binary error analysis:

- [analysis/binary_labeled_reviews.csv](/D:/Project/business%20analysis/sentimentanalysis/analysis/binary_labeled_reviews.csv)
- [analysis/binary_ambiguous_reviews.csv](/D:/Project/business%20analysis/sentimentanalysis/analysis/binary_ambiguous_reviews.csv)
- [analysis/binary_best_model_test_predictions.csv](/D:/Project/business%20analysis/sentimentanalysis/analysis/binary_best_model_test_predictions.csv)
- [analysis/binary_false_negatives.csv](/D:/Project/business%20analysis/sentimentanalysis/analysis/binary_false_negatives.csv)
- [analysis/binary_false_positives.csv](/D:/Project/business%20analysis/sentimentanalysis/analysis/binary_false_positives.csv)
- [analysis/binary_training_summary.csv](/D:/Project/business%20analysis/sentimentanalysis/analysis/binary_training_summary.csv)

## 3. What was invalidated and removed

The following artifact groups were intentionally removed because they mixed incompatible runs or misleading summary logic:

- `models/statistics/*`
- `analysis/slide_*.csv`
- slide comparison charts generated from mixed sources
- mirrored slide assets under `slide/analysis/` and `slide/public/analysis/`
- temporary overview summary derived from those statistics

Reason:

1. some summary tables mixed production transformer outputs with other experiment settings
2. some slide tables were built from historical logs and fallback values instead of one clean benchmark pass
3. some top-scoring rows used richer feature sets or settings that were not directly comparable to text-only baselines

## 4. Current conclusion

No final best model should be claimed from the deleted benchmark outputs.

At this point the correct statement is:

- the binary training code exists and runs
- several model families exist in the codebase
- previously exported full-benchmark summaries are not trustworthy enough for final comparison
- model selection must be rerun from a clean benchmark pass on the intended full dataset

## 5. Safe interpretation of remaining numbers

The file [analysis/binary_training_summary.csv](/D:/Project/business%20analysis/sentimentanalysis/analysis/binary_training_summary.csv) can still be treated as a local binary baseline comparison artifact, but not as the final cross-family benchmark for the whole project.

Use it only for:

- quick inspection of classical binary baselines
- error-analysis context
- implementation sanity checks

Do not use it for:

- thesis final model selection
- VI vs EN benchmark claims
- slide-level "best overall model" conclusions

## 6. Required next step

Before any new report or slide is written, rerun one clean benchmark pass and only then rebuild summary artifacts.

Recommended command:

```powershell
.venv\Scripts\python run.py run-full-experiments
```

After that:

1. inspect the raw per-run outputs first
2. verify all compared rows use the same task definition and comparable feature assumptions
3. rebuild final statistics from the clean run only
4. regenerate report / slide tables from that clean source

## 7. Reporting rule going forward

From this point onward, any review report should separate three layers explicitly:

1. implementation status
2. trusted benchmark results from one clean run
3. optional exploratory or slide-only comparisons

These layers must not be merged into one summary table again.
