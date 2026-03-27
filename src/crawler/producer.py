"""Kafka producer: crawls 1900.com.vn and sends parsed data to Kafka topics.

Architecture:
  Crawler → parse → Kafka topic → (Spark consumer reads and upserts to Postgres)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict

import httpx
from kafka import KafkaProducer
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
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

TOPIC_COMPANIES = "crawled-companies"
TOPIC_REVIEWS = "crawled-reviews"


def _build_client() -> httpx.Client:
    cookies = {}
    if settings.session_cookie:
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


def _build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def _fetch(client: httpx.Client, url: str) -> str:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.text


# ── Produce companies ────────────────────────────────────────────

def produce_companies(max_pages: int | None = None) -> int:
    """Crawl company listing pages → Kafka topic. Returns count sent."""
    client = _build_client()
    producer = _build_producer()
    bloom = BloomFilter()
    count = 0

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
                data = asdict(card)
                producer.send(TOPIC_COMPANIES, key=str(card.site_id), value=data)
                count += 1

        if page % 50 == 0:
            bloom.save()
            producer.flush()

    bloom.save()
    producer.flush()
    producer.close()
    client.close()
    logger.info(f"Sent {count} companies to Kafka")
    return count


# ── Produce reviews ──────────────────────────────────────────────

def produce_reviews_for_company(company_url: str, company_site_id: int,
                                client: httpx.Client, producer: KafkaProducer,
                                bloom: BloomFilter) -> int:
    """Crawl reviews for one company → Kafka topic. Returns count sent."""
    count = 0
    html = _fetch(client, company_url)
    total_pages = parse_total_review_pages(html)

    for page in range(1, total_pages + 1):
        page_url = f"{company_url}?page={page}" if page > 1 else company_url
        if page > 1:
            time.sleep(settings.crawl_delay)
            html = _fetch(client, page_url)

        reviews = parse_reviews_page(html)
        for r in reviews:
            fp = r.fingerprint
            key = f"review:{fp}"
            if bloom.add(key):
                data = asdict(r)
                data["fingerprint"] = fp
                data["company_site_id"] = company_site_id
                producer.send(TOPIC_REVIEWS, key=fp, value=data)
                count += 1

    return count


def produce_all_reviews(max_companies: int | None = None) -> int:
    """Crawl reviews for all companies in DB → Kafka topic."""
    from sqlalchemy import select
    from src.database import get_session
    from src.models import Company

    session = get_session()
    companies = session.execute(select(Company).order_by(Company.site_id)).scalars().all()
    session.close()

    if max_companies:
        companies = companies[:max_companies]

    client = _build_client()
    producer = _build_producer()
    bloom = BloomFilter()
    total = 0

    for i, company in enumerate(companies, 1):
        logger.info(f"[{i}/{len(companies)}] Producing reviews for {company.name}")
        try:
            count = produce_reviews_for_company(
                company.url, company.site_id, client, producer, bloom
            )
            total += count
            logger.info(f"  Sent {count} reviews")
        except Exception as e:
            logger.error(f"  Error: {e}")
            continue
        time.sleep(settings.crawl_delay)

        if i % 50 == 0:
            bloom.save()
            producer.flush()

    bloom.save()
    producer.flush()
    producer.close()
    client.close()
    logger.info(f"Total reviews sent to Kafka: {total}")
    return total
