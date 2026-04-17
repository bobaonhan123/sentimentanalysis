"""CLI entry point — run crawl pipeline manually."""
from __future__ import annotations

import argparse
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

    # preprocess
    sub.add_parser("preprocess", help="Run NLP preprocessing on stored reviews")

    # train
    train_p = sub.add_parser("train", help="Train sentiment analysis models")
    train_p.add_argument("--force", action="store_true", help="Force retrain even if data unchanged")

    # init-db
    sub.add_parser("init-db", help="Create/update database tables")

    # streamlit
    sub.add_parser("ui", help="Launch Streamlit export UI")

    args = parser.parse_args()

    if args.command == "crawl":
        from src.crawler.scraper import crawl_all
        crawl_all(max_listing_pages=args.max_pages)

    elif args.command == "preprocess":
        from src.preprocessing.processor import preprocess_reviews
        preprocess_reviews()

    elif args.command == "train":
        from src.training.trainer import train_pipeline
        result = train_pipeline(force=args.force)
        status = result.get("status", "unknown")
        if status == "success":
            best = result.get("best_model", {})
            print(f"✅ Training complete! Best model: {best['name']} (F1={best['f1_macro']}, Acc={best['accuracy']})")
        elif status == "skipped":
            print(f"⏭️ Training skipped: {result.get('reason')}")
        else:
            print(f"❌ Training failed: {result.get('reason', 'unknown')}")

    elif args.command == "init-db":
        from src.database import engine
        from src.models import Base
        Base.metadata.create_all(engine)
        print("✅ Tables created")

    elif args.command == "ui":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/export/app.py"])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
