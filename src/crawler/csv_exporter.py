"""Crawl 1900.com.vn reviews directly to a training CSV.

This bypasses Postgres for quick model iteration when the local DB is not
running. It reuses the existing parser so the output schema matches the
Streamlit export and training loader.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
import pandas as pd

from src.config import settings
from src.crawler.parser import (
    CompanyCard,
    ReviewItem,
    parse_company_listing,
    parse_reviews_page,
    parse_total_listing_pages,
    parse_total_review_pages,
)
from src.crawler.scraper import BASE_URL, HEADERS, LISTING_URL

logger = logging.getLogger(__name__)

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "vi" / "raw" / "1900_export_reviews.csv"


def _build_client() -> httpx.Client:
    cookies = {}
    if settings.session_cookie:
        for part in settings.session_cookie.split(";"):
            part = part.strip()
            if "=" in part:
                key, value = part.split("=", 1)
                cookies[key.strip()] = value.strip()
    return httpx.Client(headers=HEADERS, cookies=cookies, timeout=30.0, follow_redirects=True, http2=False)


def _fetch(client: httpx.Client, url: str) -> str:
    logger.info("FETCH | %s", url)
    response = client.get(url)
    response.raise_for_status()
    return response.text


def _crawl_listing(client: httpx.Client, max_listing_pages: int) -> list[CompanyCard]:
    html = _fetch(client, LISTING_URL)
    total_pages = min(parse_total_listing_pages(html), max_listing_pages)
    logger.info("LISTING | total_pages=%s limited_to=%s", parse_total_listing_pages(html), total_pages)

    cards: list[CompanyCard] = []
    seen: set[int] = set()
    for page in range(1, total_pages + 1):
        if page > 1:
            time.sleep(settings.crawl_delay)
            html = _fetch(client, f"{LISTING_URL}?page={page}")
        page_cards = parse_company_listing(html)
        logger.info("LISTING_PAGE | page=%s companies=%s", page, len(page_cards))
        for card in page_cards:
            if card.site_id not in seen:
                cards.append(card)
                seen.add(card.site_id)
    return cards


def _crawl_reviews(
    client: httpx.Client,
    company: CompanyCard,
    *,
    max_review_pages: int,
) -> list[ReviewItem]:
    html = _fetch(client, company.url)
    total_pages = min(parse_total_review_pages(html), max_review_pages)
    reviews = parse_reviews_page(html)
    logger.info(
        "REVIEW_PAGE | company=%s page=1/%s reviews=%s",
        company.name,
        total_pages,
        len(reviews),
    )

    for page in range(2, total_pages + 1):
        time.sleep(settings.crawl_delay)
        html = _fetch(client, f"{company.url}?page={page}")
        page_reviews = parse_reviews_page(html)
        logger.info(
            "REVIEW_PAGE | company=%s page=%s/%s reviews=%s",
            company.name,
            page,
            total_pages,
            len(page_reviews),
        )
        reviews.extend(page_reviews)
    return reviews


def _row(company: CompanyCard, review: ReviewItem) -> dict:
    return {
        "company": company.name,
        "industry": company.industry,
        "rating": review.rating,
        "title": review.title,
        "job_title": review.job_title,
        "employee_status": review.employee_status,
        "location": review.review_location or company.location,
        "date": review.review_date,
        "pros": review.pros,
        "cons": review.cons,
        "advice": review.advice,
        "recommends": review.recommends,
    }


def crawl_reviews_to_csv(
    *,
    output_path: str | Path = DEFAULT_CSV,
    max_listing_pages: int = 5,
    max_companies: int = 60,
    max_review_pages: int = 2,
    min_reviews: int = 80,
    target_rows: int | None = None,
) -> dict:
    """Crawl reviews and save a training CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _build_client()
    rows: list[dict] = []
    seen_reviews: set[str] = set()
    try:
        cards = _crawl_listing(client, max_listing_pages=max_listing_pages)
        cards = sorted(cards, key=lambda c: c.review_count or 0, reverse=True)[:max_companies]
        logger.info("CRAWL_CSV | companies_selected=%s", len(cards))

        for index, company in enumerate(cards, 1):
            logger.info("COMPANY | %s/%s | %s", index, len(cards), company.name)
            try:
                reviews = _crawl_reviews(client, company, max_review_pages=max_review_pages)
            except Exception as exc:
                logger.warning("COMPANY_FAILED | %s | %s", company.name, exc)
                continue

            for review in reviews:
                if review.fingerprint in seen_reviews or review.rating is None:
                    continue
                rows.append(_row(company, review))
                seen_reviews.add(review.fingerprint)

            logger.info("CRAWL_PROGRESS | rows=%s target=%s", len(rows), target_rows or min_reviews)
            stop_at = target_rows if target_rows and target_rows > 0 else min_reviews
            if len(rows) >= stop_at:
                break
            time.sleep(settings.crawl_delay)
    finally:
        client.close()

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("CSV_SAVED | path=%s rows=%s", output_path, len(df))
    return {
        "status": "success" if len(df) else "failed",
        "csv": str(output_path),
        "rows": len(df),
        "companies": int(df["company"].nunique()) if not df.empty else 0,
    }
