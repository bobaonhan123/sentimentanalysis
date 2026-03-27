# 1900.com.vn Company Review Crawler

Pipeline crawl, import, preprocess và export review công ty từ [1900.com.vn](https://1900.com.vn).

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Airflow    │────▶│  Crawl       │────▶│  Preprocess  │────▶│  Postgres    │
│  (Schedule)  │     │  Companies   │     │  NLP/Stopw.  │     │  (ORM)       │
│  Daily 2AM   │     │  + Reviews   │     │  Tokenize    │     │              │
└─────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                         │                                          │
                    Bloom Filter                              ┌─────▼───────┐
                    (dedup URLs)                              │  Streamlit  │
                                                              │  Export/UI  │
                                                              │  CSV Debug  │
                                                              └─────────────┘
```

## Cấu trúc website 1900.com.vn

| Page | URL Pattern | Pagination |
|------|-------------|------------|
| Company listing | `/review-cong-ty?page=N` | 250 pages, ~20 companies/page |
| Company reviews | `/danh-gia-dn/{slug}-{id}?page=N` | 20 reviews/page |
| Tabs | `?tab=hot\|featured\|excellent` | Trên listing page |

**Review data**: rating (1-5★), title, job title, employee status, location, ưu điểm, nhược điểm, lời khuyên cho quản lý.

## Quick Start

### 1. Setup với `uv`

```bash
# Install uv (nếu chưa có)
pip install uv

# Tạo venv + install dependencies
uv venv
uv pip install -e ".[dev]"
```

### 2. Chạy Postgres + services

```bash
# Copy config
cp .env.example .env
# Sửa .env nếu cần

# Start all services
docker compose up -d
```

### 3. CLI Commands

```bash
# Tạo tables
python run.py init-db

# Chạy crawl (giới hạn 5 pages để test)
python run.py crawl --max-pages 5

# Preprocess reviews
python run.py preprocess

# Mở Streamlit UI
python run.py ui
```

### 4. Airflow

- UI: http://localhost:8080 (admin / admin)
- DAG: `crawl_1900_reviews` — chạy daily 2 AM
- Pipeline: `init_db → crawl_companies → crawl_reviews → preprocess`

### 5. Streamlit Export

- UI: http://localhost:8501
- Features:
  - Filter theo industry, location, rating
  - Search company name / review text
  - Paginated tables
  - Export CSV (companies / reviews / preprocessed)
  - Stats dashboard

## Cookies / Auth

Một số review bị truncate nếu chưa login. Để đọc đầy đủ:
1. Login trên browser tại 1900.com.vn
2. Copy cookie string từ DevTools → Network → Request Headers
3. Paste vào `SESSION_COOKIE` trong `.env`

## Preprocessing Pipeline

1. **Unicode NFC** normalization
2. **Lowercase**
3. **Remove** URLs, emails, HTML tags
4. **Keep** Vietnamese diacritics + basic punctuation
5. **Tokenize** với [underthesea](https://github.com/undertheseanlp/underthesea) (word segmentation)
6. **Remove stopwords** (Vietnamese + English function words)

## DB Schema

```
companies
├── id (PK)
├── site_id (unique — ID trên 1900.com.vn)
├── slug, name, industry, employee_range, location
├── overall_rating, review_count, url
└── created_at, updated_at

reviews
├── id (PK)
├── company_id (FK → companies.id)
├── fingerprint (unique — SHA-256 dedup)
├── title, rating, job_title, employee_status, review_location, review_date
├── pros, cons, advice
├── recommends, ceo_rating, business_outlook
├── pros_clean, cons_clean (← preprocessed)
└── created_at
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ (uv) |
| HTTP Client | httpx |
| HTML Parser | selectolax |
| ORM | SQLAlchemy 2.0 |
| Database | PostgreSQL 16 |
| Dedup | Bloom Filter (pybloom-live) |
| Scheduler | Apache Airflow 2.8 |
| NLP | underthesea |
| Export UI | Streamlit |
| Container | Docker Compose |
