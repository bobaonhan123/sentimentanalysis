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
