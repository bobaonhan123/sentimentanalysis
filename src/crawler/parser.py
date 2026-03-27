"""HTML parser for 1900.com.vn company listing & review pages.

Uses CSS selectors on the DOM tree for robust extraction.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from selectolax.parser import HTMLParser, Node


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class CompanyCard:
    """Parsed from /review-cong-ty listing."""
    site_id: int
    slug: str
    name: str
    url: str
    industry: str | None = None
    employee_range: str | None = None
    location: str | None = None
    overall_rating: float | None = None
    review_count: int | None = None


@dataclass
class ReviewItem:
    """Parsed from /danh-gia-dn/{slug}-{id} detail page."""
    title: str | None = None
    rating: float | None = None
    job_title: str | None = None
    employee_status: str | None = None
    review_location: str | None = None
    review_date: str | None = None
    pros: str | None = None
    cons: str | None = None
    advice: str | None = None
    recommends: str | None = None
    ceo_rating: str | None = None
    business_outlook: str | None = None

    @property
    def fingerprint(self) -> str:
        """SHA-256 of key fields for deduplication."""
        raw = f"{self.job_title}|{self.title}|{self.pros}|{self.cons}|{self.review_date}"
        return hashlib.sha256(raw.encode()).hexdigest()


# ── Constants ─────────────────────────────────────────────────────

_SLUG_ID_RE = re.compile(r"/danh-gia-dn/([\w-]+)-(\d+)")

_CITIES = {
    "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Bình Dương", "Hải Phòng",
    "Cần Thơ", "Bắc Ninh", "Gia Lai", "Thanh Hoá", "Hưng Yên",
    "Vũng Tàu", "Bình Định", "Thái Nguyên", "Quảng Ninh", "Long An",
    "Nghệ An", "Khánh Hòa", "Đồng Nai", "Lâm Đồng", "Đắk Lắk",
    "Bà Rịa", "Tiền Giang", "Bình Thuận", "Phú Thọ", "An Giang",
}


# ── URL helpers ───────────────────────────────────────────────────

def extract_slug_and_id(url: str) -> tuple[str, int] | None:
    m = _SLUG_ID_RE.search(url)
    if m:
        return m.group(1), int(m.group(2))
    return None


# ── Listing page parser ──────────────────────────────────────────

def parse_company_listing(html: str) -> list[CompanyCard]:
    """Parse /review-cong-ty?page=N → list of CompanyCard.

    Uses CSS selectors on div.company-item containers.
    """
    tree = HTMLParser(html)
    cards: list[CompanyCard] = []

    for card_div in tree.css("div.company-item"):
        a_tag = card_div.css_first("a[href*='/danh-gia-dn/']")
        if not a_tag:
            continue
        href = a_tag.attributes.get("href", "")
        parsed = extract_slug_and_id(href)
        if not parsed:
            continue
        slug, site_id = parsed

        if any(c.site_id == site_id for c in cards):
            continue

        # Company name: <p class="font-weight-bold ..."> or <img alt="">
        name_p = card_div.css_first("p.font-weight-bold")
        if name_p:
            name = name_p.text(strip=True)
        else:
            img = card_div.css_first("img[alt]")
            name = img.attributes.get("alt", slug) if img else slug

        # Rating: <span class="text-gray ..."><b>4.0</b> sao</span>
        rating = None
        for span in card_div.css("span.text-gray"):
            span_text = span.text() or ""
            if "sao" in span_text:
                b_tag = span.css_first("b")
                if b_tag:
                    try:
                        rating = float(b_tag.text(strip=True))
                    except ValueError:
                        pass
                break

        # Review count: <span class="text-primary ..."><b>1 reviews</b></span>
        review_count = None
        rc_b = card_div.css_first("span.text-primary b")
        if rc_b:
            m = re.search(r'(\d+)', rc_b.text(strip=True))
            if m:
                review_count = int(m.group(1))

        # Industry, employee range, location from inner spans
        industry = None
        employee_range = None
        location_text = None

        for span in card_div.css("div.text-gray span, div.mb-0 span"):
            text = span.text(strip=True)
            if not text or len(text) < 2:
                continue
            if any(skip in text for skip in ("sao", "reviews", "giờ trước", "lượt xem", "việc làm")):
                continue
            if "nhân viên" in text:
                employee_range = text
            elif any(city in text for city in _CITIES):
                if not location_text:
                    location_text = text
            elif not industry and len(text) > 2:
                industry = text

        cards.append(CompanyCard(
            site_id=site_id,
            slug=slug,
            name=name,
            url=f"https://1900.com.vn/danh-gia-dn/{slug}-{site_id}",
            industry=industry,
            employee_range=employee_range,
            location=location_text,
            overall_rating=rating,
            review_count=review_count,
        ))

    return cards


# ── Review page parser ───────────────────────────────────────────

def parse_reviews_page(html: str) -> list[ReviewItem]:
    """Parse /danh-gia-dn/{slug}-{id}?page=N → list of ReviewItem.

    Each review lives in a div.ReviewItem with class ReviewIndex{id}.
    """
    tree = HTMLParser(html)
    reviews: list[ReviewItem] = []

    for item in tree.css("div.ReviewItem"):
        cls = item.attributes.get("class", "") or ""
        if "ReviewIndex" not in cls:
            continue
        review = _parse_review_item(item)
        if review:
            reviews.append(review)

    return reviews


def _parse_review_item(item: Node) -> ReviewItem | None:
    """Parse a single div.ReviewItem DOM node."""

    # Rating: <span class="ratingNumber ...">4.0</span>
    rating = None
    rating_span = item.css_first("span.ratingNumber")
    if rating_span:
        try:
            rating = float(rating_span.text(strip=True))
        except ValueError:
            pass

    # Date: <div class="text-black-50 ..."><span>27/03/2026</span></div>
    review_date = None
    date_div = item.css_first("div.text-black-50")
    if date_div:
        date_span = date_div.css_first("span")
        if date_span:
            review_date = date_span.text(strip=True)

    # Title: <h2 class="... ReviewTitle ...">
    title = None
    title_h2 = item.css_first("h2.ReviewTitle")
    if title_h2:
        title = title_h2.text(strip=True)

    # Job title: <div class="... ReviewCandidateSubtext"><a href="...?careerId=...">
    job_title = None
    job_link = item.css_first("div.ReviewCandidateSubtext a[href*='careerId']")
    if job_link:
        job_title = job_link.text(strip=True)
    else:
        job_div = item.css_first("div.ReviewCandidateSubtext")
        if job_div:
            job_a = job_div.css_first("a")
            if job_a:
                job_title = job_a.text(strip=True)

    # Employee status + Location from <div class="mb-3">
    employee_status = None
    review_location = None

    for mb3 in item.css("div.mb-3"):
        for span in mb3.css("span.ReviewCandidateSubtext"):
            text = span.text(strip=True)
            cls = span.attributes.get("class", "") or ""
            if "text-dark" in cls:
                # Location span (has fa-map-marker icon)
                review_location = text.strip()
            elif "nhân viên" in text.lower():
                employee_status = text
        if employee_status or review_location:
            break

    # Badges: <div class="... ReviewRating">
    recommends = None
    ceo_rating = None
    business_outlook = None
    for badge in item.css("div.ReviewRating"):
        bt = badge.text(strip=True).lower()
        if "đề xuất" in bt:
            recommends = "Có"
        elif "ceo" in bt:
            ceo_rating = "Có"
        elif "triển vọng" in bt:
            business_outlook = "Có"

    # Pros / Cons / Advice: pair <strong> labels with following div.expandable-content
    # by document order to avoid mixing up sections
    pros = None
    cons = None
    advice = None

    current_label = None
    for node in item.css("strong, div.expandable-content"):
        if node.tag == "strong":
            lbl = node.text(strip=True).lower()
            if any(kw in lbl for kw in ("ưu điểm", "nhược điểm", "lời khuyên")):
                current_label = lbl
        elif "expandable-content" in (node.attributes.get("class", "") or ""):
            if current_label:
                content = node.text(strip=True) or None
                if "ưu điểm" in current_label:
                    pros = content
                elif "nhược điểm" in current_label:
                    cons = content
                elif "lời khuyên" in current_label:
                    advice = content
                current_label = None

    return ReviewItem(
        title=title,
        rating=rating,
        job_title=job_title,
        employee_status=employee_status,
        review_location=review_location,
        review_date=review_date,
        pros=pros,
        cons=cons,
        advice=advice,
        recommends=recommends,
        ceo_rating=ceo_rating,
        business_outlook=business_outlook,
    )


# ── Pagination ────────────────────────────────────────────────────

def parse_total_review_pages(html: str) -> int:
    tree = HTMLParser(html)
    max_page = 1
    for a_tag in tree.css("a[href*='page=']"):
        href = a_tag.attributes.get("href", "")
        m = re.search(r'page=(\d+)', href)
        if m:
            max_page = max(max_page, int(m.group(1)))
    return max_page


def parse_total_listing_pages(html: str) -> int:
    return parse_total_review_pages(html)
