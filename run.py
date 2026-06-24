"""CLI entry point — run crawl pipeline manually."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="1900.com.vn Review Crawler")
    sub = parser.add_subparsers(dest="command")

    # crawl
    crawl_p = sub.add_parser("crawl", help="Run full crawl pipeline")
    crawl_p.add_argument("--max-pages", type=int, default=None, help="Limit listing pages")

    crawl_csv_p = sub.add_parser("crawl-csv", help="Crawl reviews directly to training CSV without Postgres")
    crawl_csv_p.add_argument("--out", default=None, help="Output CSV path")
    crawl_csv_p.add_argument("--max-listing-pages", type=int, default=5, help="Limit listing pages")
    crawl_csv_p.add_argument("--max-companies", type=int, default=60, help="Limit companies crawled")
    crawl_csv_p.add_argument("--max-review-pages", type=int, default=2, help="Limit review pages per company")
    crawl_csv_p.add_argument("--target-rows", type=int, default=None, help="Stop after collecting this many reviews")
    crawl_csv_p.add_argument("--min-reviews", type=int, default=80, help="Legacy minimum rows before early stop when target-rows is unset")

    # preprocess
    sub.add_parser("preprocess", help="Run NLP preprocessing on stored reviews")

    # train
    train_p = sub.add_parser("train", help="Train sentiment analysis models")
    train_p.add_argument("--force", action="store_true", help="Force retrain even if data unchanged")
    train_p.add_argument("--csv", default=None, help="Path to reviews CSV (default: data/vi/raw/1900_export_reviews.csv)")

    train_variants_p = sub.add_parser("train-variants", help="Train negative vs non-negative sentiment variants")
    train_variants_p.add_argument("--csv", default=None, help="Path to reviews CSV (default: data/vi/raw/1900_export_reviews.csv)")
    train_variants_p.add_argument(
        "--source",
        choices=["current", "glassdoor-en"],
        default="current",
        help="Training data source",
    )
    train_variants_p.add_argument(
        "--english-max-rows",
        type=int,
        default=None,
        help="Limit English Glassdoor rows; omit/0 for full dataset",
    )

    compare_p = sub.add_parser("compare-glassdoor-datasets", help="Train current Vietnamese data and English Glassdoor data for comparison")
    compare_p.add_argument("--csv", default=None, help="Path to reviews CSV (default: data/vi/raw/1900_export_reviews.csv)")
    compare_p.add_argument(
        "--english-max-rows",
        type=int,
        default=None,
        help="Limit English rows; default matches current labeled row count",
    )

    scan_vi_p = sub.add_parser("scan-vietnamese-text", help="Scan slang/abbreviations/noisy Vietnamese tokens in labeled data")
    scan_vi_p.add_argument("--csv", default=None, help="Path to reviews CSV (default: data/vi/raw/1900_export_reviews.csv)")
    scan_vi_p.add_argument("--out", default=None, help="Output directory (default: analysis/)")
    scan_vi_p.add_argument("--top-n", type=int, default=250, help="Number of unknown tokens to keep")

    phobert_binary_p = sub.add_parser("train-phobert-binary", help="Fine-tune PhoBERT for binary sentiment")
    phobert_binary_p.add_argument("--csv", default=None, help="Path to reviews CSV (default: data/vi/raw/1900_export_reviews.csv)")
    phobert_binary_p.add_argument("--language", default="vi", choices=["vi", "en"], help="Language folder under models/phobert/")
    phobert_binary_p.add_argument("--max-examples", type=int, default=512, help="Limit examples for CPU smoke tests; 0 = full data")
    phobert_binary_p.add_argument("--epochs", type=int, default=1, help="Training epochs")
    phobert_binary_p.add_argument("--batch-size", type=int, default=8, help="Batch size")
    phobert_binary_p.add_argument("--max-len", type=int, default=160, help="Max token length")
    phobert_binary_p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"], help="Force training device")

    fasttext_binary_p = sub.add_parser(
        "train-fasttext-binary",
        help="Train FastText 305-dim + sklearn/MLP models (binary negative vs non-negative)",
    )
    fasttext_binary_p.add_argument("--csv", default=None, help="Vietnamese reviews CSV path")
    fasttext_binary_p.add_argument("--source", default="current", help="Dataset source key (current or glassdoor-en)")

    full_exp_p = sub.add_parser("run-full-experiments", help="Run full-scale VI + EN experiments and build statistics")
    full_exp_p.add_argument("--vi-csv", default=None, help="Vietnamese reviews CSV path")
    full_exp_p.add_argument("--smoke", action="store_true", help="Quick run on small data samples")
    full_exp_p.add_argument("--skip-vi-tfidf", action="store_true")
    full_exp_p.add_argument("--skip-en-tfidf", action="store_true")
    full_exp_p.add_argument("--skip-vi-phobert", action="store_true")
    full_exp_p.add_argument("--skip-en-phobert", action="store_true")
    full_exp_p.add_argument("--skip-vi-fasttext", action="store_true")
    full_exp_p.add_argument("--skip-en-fasttext", action="store_true")
    full_exp_p.add_argument("--phobert-epochs", type=int, default=None)
    full_exp_p.add_argument("--phobert-batch-size", type=int, default=8)
    full_exp_p.add_argument("--no-deploy-vi-best", action="store_true")

    sub.add_parser("build-statistics", help="Build final cross-language model statistics CSV")

    # init-db
    sub.add_parser("init-db", help="Create/update database tables")

    # absa
    absa_p = sub.add_parser("absa", help="Run ABSA on review CSV (business analysis)")
    absa_p.add_argument("--csv", default=None, help="Path to reviews CSV (default: analysis/1900_export_reviews.csv)")
    absa_p.add_argument("--out", default=None, help="Output directory for CSV + PNG results (default: analysis/)")

    # streamlit
    sub.add_parser("ui", help="Launch Streamlit export UI")

    args = parser.parse_args()

    if args.command == "crawl":
        from src.crawler.scraper import crawl_all
        crawl_all(max_listing_pages=args.max_pages)

    elif args.command == "crawl-csv":
        from src.crawler.csv_exporter import crawl_reviews_to_csv
        kwargs = {
            "max_listing_pages": args.max_listing_pages,
            "max_companies": args.max_companies,
            "max_review_pages": args.max_review_pages,
            "min_reviews": args.min_reviews,
        }
        if getattr(args, "target_rows", None):
            kwargs["target_rows"] = args.target_rows
        if args.out:
            kwargs["output_path"] = args.out
        result = crawl_reviews_to_csv(**kwargs)
        if result["status"] == "success":
            print(f"CSV generated: {result['csv']} ({result['rows']} rows, {result['companies']} companies)")
        else:
            print("CSV crawl failed: no rows collected")

    elif args.command == "preprocess":
        from src.preprocessing.processor import preprocess_reviews
        preprocess_reviews()

    elif args.command == "train":
        from src.training.trainer import train_pipeline
        result = train_pipeline(force=args.force, csv_path=args.csv)
        status = result.get("status", "unknown")
        if status == "success":
            best = result.get("best_model", {})
            print(f"Training complete. Best model: {best['name']} (F1={best['f1_macro']}, Acc={best['accuracy']})")
        elif status == "skipped":
            print(f"Training skipped: {result.get('reason')}")
        else:
            print(f"Training failed: {result.get('reason', 'unknown')}")

    elif args.command == "train-variants":
        from src.training.variant_trainer import train_variants
        result = train_variants(
            csv_path=args.csv,
            source=args.source,
            english_max_rows=args.english_max_rows or None,
        )
        status = result.get("status", "unknown")
        if status == "success":
            best = result.get("best_model", {})
            print(f"Variant training complete. Best model: {best['name']} (F1={best['f1_macro']}, Acc={best['accuracy']})")
        else:
            print(f"Variant training failed: {result.get('reason', 'unknown')}")

    elif args.command == "compare-glassdoor-datasets":
        import pandas as pd
        from datetime import datetime

        from src.training.labeling import load_labeled_data
        from src.artifacts.paths import COMPARISONS_DIR
        from src.training.variant_trainer import train_variants

        english_max_rows = args.english_max_rows
        if not english_max_rows:
            english_max_rows = len(load_labeled_data(args.csv))

        current = train_variants(csv_path=args.csv, source="current", deploy_best=False)
        english = train_variants(source="glassdoor-en", english_max_rows=english_max_rows, deploy_best=False)

        rows = []
        for label, result in [("current_vi", current), ("glassdoor_en", english)]:
            best = result.get("best_model", {})
            rows.append({
                "dataset": label,
                "status": result.get("status"),
                "sample_count": result.get("sample_count"),
                "best_model": best.get("name"),
                "val_f1_macro": best.get("val_f1_macro"),
                "test_f1_macro": best.get("f1_macro"),
                "accuracy": best.get("accuracy"),
                "run_dir": result.get("run_dir"),
            })

        COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_csv = COMPARISONS_DIR / f"glassdoor_dataset_comparison_{stamp}.csv"
        comparison_json = COMPARISONS_DIR / f"glassdoor_dataset_comparison_{stamp}.json"
        pd.DataFrame(rows).to_csv(comparison_csv, index=False, encoding="utf-8-sig")
        comparison_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"Dataset comparison complete: {comparison_csv}")
        for row in rows:
            print(
                f"{row['dataset']}: n={row['sample_count']} "
                f"F1={row['test_f1_macro']} Acc={row['accuracy']} run={row['run_dir']}"
            )

    elif args.command == "scan-vietnamese-text":
        from src.preprocessing.vietnamese_terms import scan_vietnamese_terms
        from src.training.labeling import load_labeled_data
        result = scan_vietnamese_terms(load_labeled_data(args.csv), out_dir=args.out, top_n=args.top_n)
        print(f"Vietnamese text scan complete: {result['known_terms_csv']}")
        print(f"Unknown tokens: {result['unknown_ascii_csv']}")
        print(f"Repeated chars: {result['repeated_char_csv']}")

    elif args.command == "train-phobert-binary":
        from src.training.phobert_binary_trainer import train_phobert_binary
        result = train_phobert_binary(
            csv_path=args.csv,
            language=args.language,
            max_examples=args.max_examples or None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device_name=args.device,
        )
        status = result.get("status", "unknown")
        if status == "success":
            test = result.get("threshold_test") or result.get("test", {})
            print(
                "PhoBERT binary training complete. "
                f"F1={test.get('f1_macro')} Acc={test.get('accuracy')} "
                f"threshold={result.get('threshold')}"
            )
        else:
            print(f"PhoBERT binary training failed: {result.get('reason', 'unknown')}")

    elif args.command == "train-fasttext-binary":
        from src.training.fasttext_binary_trainer import train_fasttext_binary
        result = train_fasttext_binary(csv_path=args.csv, source=args.source)
        status = result.get("status", "unknown")
        if status == "success":
            best = result.get("best_result", {})
            print(
                "FastText binary training complete. "
                f"Best={result.get('best_name')} F1={best.get('f1_macro')} Acc={best.get('accuracy')}"
            )
        elif status == "skipped":
            print(f"FastText binary training skipped: {result.get('reason')}")
        else:
            print(f"FastText binary training failed: {result.get('reason', 'unknown')}")

    elif args.command == "run-full-experiments":
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location(
            "run_full_experiment_pipeline",
            Path(__file__).resolve().parent / "scripts" / "run_full_experiment_pipeline.py",
        )
        pipeline_mod = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(pipeline_mod)

        manifest = pipeline_mod.run_full_experiment_pipeline(
            vi_csv=args.vi_csv,
            smoke=args.smoke,
            skip_vi_tfidf=args.skip_vi_tfidf,
            skip_en_tfidf=args.skip_en_tfidf,
            skip_vi_phobert=args.skip_vi_phobert,
            skip_en_phobert=args.skip_en_phobert,
            skip_vi_fasttext=args.skip_vi_fasttext,
            skip_en_fasttext=args.skip_en_fasttext,
            phobert_epochs=args.phobert_epochs,
            phobert_batch_size=args.phobert_batch_size,
            deploy_vi_best=not args.no_deploy_vi_best if not args.smoke else False,
        )
        print(json.dumps(
            {
                "status": manifest.get("status"),
                "mode": manifest.get("mode"),
                "vi_rows": manifest.get("vi_rows"),
                "en_rows": manifest.get("en_rows"),
                "statistics_csv": (manifest.get("statistics") or {}).get("csv"),
                "statistics_rows": (manifest.get("statistics") or {}).get("row_count"),
            },
            indent=2,
            ensure_ascii=False,
        ))

    elif args.command == "build-statistics":
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location(
            "build_final_statistics",
            Path(__file__).resolve().parent / "scripts" / "build_final_statistics.py",
        )
        mod = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        payload = mod.build_final_statistics()
        print(json.dumps({"csv": payload["csv"], "row_count": payload["row_count"]}, indent=2))

    elif args.command == "absa":
        from src.analysis.absa import run_absa
        result = run_absa(csv_path=args.csv, out_dir=args.out)
        if result["status"] == "success":
            print(f"\nDone. {result['aspect_mentions']:,} aspect mentions across {result['total_reviews']:,} reviews.")
            print(f"Summary CSV : {result['summary_csv']}")
            print(f"Details CSV : {result['details_csv']}")
            for c in result["charts"]:
                print(f"Chart       : {c}")
        else:
            print(f"ABSA failed: {result.get('reason', 'unknown')}")

    elif args.command == "init-db":
        from src.database import engine
        from src.models import Base
        Base.metadata.create_all(engine)
        print("Tables created")

    elif args.command == "ui":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/export/app.py"])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
