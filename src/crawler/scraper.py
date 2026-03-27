"""Main scraper: crawls 1900.com.vn company reviews and stores to Postgres."""
from __future__ import annotations

import logging
import time
from hashlib import sha256

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.database import get_session
from src.models import Base, Company, Review
from src.crawler.bloom_filter import BloomFilter
from src.crawler.parser import (
    CompanyCard,
    ReviewItem,
    parse_company_listing,
    parse_reviews_page,
    parse_total_listing_pages,
    parse_total_review_pages,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://1900.com.vn"
LISTING_URL = f"{BASE_URL}/review-cong-ty"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    "Referer": BASE_URL,
}


def _build_client() -> httpx.Client:
    cookies = {}
    if settings.session_cookie:
        # Parse cookie string "key1=val1; key2=val2"
        for part in settings.session_cookie.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                cookies[k.strip()] = v.strip()
    return httpx.Client(
        headers=HEADERS,
        cookies=cookies,
        timeout=30.0,
        follow_redirects=True,
        http2=False,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def _fetch(client: httpx.Client, url: str) -> str:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.text


# ── Stage 1: Crawl company listing ───────────────────────────────

def crawl_companies(max_pages: int | None = None) -> list[CompanyCard]:
    """Crawl /review-cong-ty pages and return all CompanyCard items."""
    client = _build_client()
    bloom = BloomFilter()
    all_cards: list[CompanyCard] = []

    # First page to discover total pages
    html = _fetch(client, LISTING_URL)
    total_pages = parse_total_listing_pages(html)
    if max_pages:
        total_pages = min(total_pages, max_pages)

    logger.info(f"Total listing pages: {total_pages}")

    for page in range(1, total_pages + 1):
        url = f"{LISTING_URL}?page={page}"
        logger.info(f"Crawling listing page {page}/{total_pages}")

        if page > 1:
            time.sleep(settings.crawl_delay)
            html = _fetch(client, url)

        cards = parse_company_listing(html)
        for card in cards:
            key = f"company:{card.site_id}"
            if bloom.add(key):
                all_cards.append(card)

        if page % 50 == 0:
            bloom.save()

    bloom.save()
    client.close()
    logger.info(f"Discovered {len(all_cards)} new companies")
    return all_cards


def save_companies(cards: list[CompanyCard]) -> None:
    """Upsert companies into Postgres."""
    session = get_session()
    try:
        for card in cards:
            stmt = pg_insert(Company).values(
                site_id=card.site_id,
                slug=card.slug,
                name=card.name,
                industry=card.industry,
                employee_range=card.employee_range,
                location=card.location,
                overall_rating=card.overall_rating,
                review_count=card.review_count,
                url=card.url,
            ).on_conflict_do_update(
                index_elements=["site_id"],
                set_={
                    "name": card.name,
                    "industry": card.industry,
                    "employee_range": card.employee_range,
                    "location": card.location,
                    "overall_rating": card.overall_rating,
                    "review_count": card.review_count,
                    "url": card.url,
                },
            )
            session.execute(stmt)
        session.commit()
        logger.info(f"Upserted {len(cards)} companies")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Stage 2: Crawl reviews per company ───────────────────────────

def crawl_reviews_for_company(company: Company) -> list[ReviewItem]:
    """Crawl all review pages for a single company."""
    client = _build_client()
    bloom = BloomFilter()
    all_reviews: list[ReviewItem] = []

    url = company.url
    html = _fetch(client, url)
    total_pages = parse_total_review_pages(html)
    logger.info(f"Company {company.name}: {total_pages} review pages")

    for page in range(1, total_pages + 1):
        page_url = f"{url}?page={page}" if page > 1 else url
        if page > 1:
            time.sleep(settings.crawl_delay)
            html = _fetch(client, page_url)

        reviews = parse_reviews_page(html)
        for r in reviews:
            key = f"review:{r.fingerprint}"
            if bloom.add(key):
                all_reviews.append(r)

    bloom.save()
    client.close()
    return all_reviews


def save_reviews(company_id: int, reviews: list[ReviewItem]) -> int:
    """Upsert reviews into Postgres. Returns count of new reviews."""
    session = get_session()
    new_count = 0
    try:
        for r in reviews:
            stmt = pg_insert(Review).values(
                company_id=company_id,
                fingerprint=r.fingerprint,
                title=r.title,
                rating=r.rating,
                job_title=r.job_title,
                employee_status=r.employee_status,
                review_location=r.review_location,
                review_date=r.review_date,
                pros=r.pros,
                cons=r.cons,
                advice=r.advice,
                recommends=r.recommends,
                ceo_rating=r.ceo_rating,
                business_outlook=r.business_outlook,
            ).on_conflict_do_nothing(index_elements=["fingerprint"])
            result = session.execute(stmt)
            if result.rowcount > 0:
                new_count += 1
        session.commit()
        logger.info(f"Saved {new_count} new reviews for company_id={company_id}")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return new_count


# ── Full pipeline ─────────────────────────────────────────────────

def crawl_all(max_listing_pages: int | None = None):
    """Full crawl: listing → companies → reviews."""
    from src.database import engine
    Base.metadata.create_all(engine)

    # Stage 1: crawl company listing
    logger.info("=== Stage 1: Crawling company listing ===")
    cards = crawl_companies(max_pages=max_listing_pages)
    save_companies(cards)

    # Stage 2: crawl reviews for each company
    logger.info("=== Stage 2: Crawling reviews ===")
    session = get_session()
    companies = session.execute(select(Company).order_by(Company.site_id)).scalars().all()
    session.close()

    total_new = 0
    for i, company in enumerate(companies, 1):
        logger.info(f"[{i}/{len(companies)}] Crawling reviews for: {company.name}")
        try:
            reviews = crawl_reviews_for_company(company)
            if reviews:
                new = save_reviews(company.id, reviews)
                total_new += new
        except Exception as e:
            logger.error(f"Error crawling {company.name}: {e}")
            continue
        time.sleep(settings.crawl_delay)

    logger.info(f"=== Done. Total new reviews: {total_new} ===")
