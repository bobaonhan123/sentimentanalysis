"""
Airflow DAG: 1900.com.vn Review Crawl Pipeline (Kafka + Spark)

Schedule: Daily at 2 AM
Stages:
  1. init_db           — Ensure tables exist
  2. produce_companies — Crawl listing pages → Kafka topic
  3. wait_companies    — Wait for Spark consumer to ingest companies into DB
  4. produce_reviews   — Crawl reviews per company → Kafka topic
  5. wait_reviews      — Wait for Spark consumer to ingest reviews into DB
  6. preprocess        — NLP preprocessing (normalize, stopwords, tokenize)
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add project root to path so imports work inside Airflow
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=6),
}


def _init_db():
    from src.database import engine
    from src.models import Base
    Base.metadata.create_all(engine)


def _produce_companies(**context):
    from src.crawler.producer import produce_companies
    _init_db()
    count = produce_companies(max_pages=None)
    context["ti"].xcom_push(key="company_count", value=count)
    logger.info(f"Sent {count} companies to Kafka")


def _wait_companies(**context):
    """Poll Postgres until Spark consumer has ingested companies."""
    from sqlalchemy import func, select
    from src.database import get_session
    from src.models import Company

    expected = context["ti"].xcom_pull(task_ids="produce_companies", key="company_count") or 1
    deadline = time.time() + 180  # 3 min max wait

    while time.time() < deadline:
        session = get_session()
        count = session.execute(select(func.count(Company.id))).scalar()
        session.close()
        if count >= expected:
            logger.info(f"Companies ingested: {count}")
            return
        logger.info(f"Waiting for companies... {count}/{expected}")
        time.sleep(10)

    logger.warning("Timeout waiting for companies, continuing anyway")


def _produce_reviews(**context):
    from src.crawler.producer import produce_all_reviews
    count = produce_all_reviews(max_companies=None)
    context["ti"].xcom_push(key="review_count", value=count)
    logger.info(f"Sent {count} reviews to Kafka")


def _wait_reviews(**context):
    """Wait for Spark consumer to flush reviews into Postgres."""
    # Simple time-based wait: Spark triggers every 5s, give 30s buffer
    time.sleep(30)
    logger.info("Review ingestion wait complete")


def _preprocess(**context):
    from src.preprocessing.processor import preprocess_reviews
    count = preprocess_reviews(batch_size=500)
    context["ti"].xcom_push(key="preprocessed_count", value=count)
    logger.info(f"Preprocessed {count} reviews")


with DAG(
    dag_id="crawl_1900_reviews",
    default_args=default_args,
    description="Crawl company reviews from 1900.com.vn → Kafka → Spark → Postgres + NLP",
    schedule="0 2 * * *",  # Daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["crawl", "reviews", "1900", "kafka"],
) as dag:

    init_db = PythonOperator(
        task_id="init_db",
        python_callable=_init_db,
    )

    produce_companies = PythonOperator(
        task_id="produce_companies",
        python_callable=_produce_companies,
        execution_timeout=timedelta(hours=2),
    )

    wait_companies = PythonOperator(
        task_id="wait_companies",
        python_callable=_wait_companies,
        execution_timeout=timedelta(minutes=5),
    )

    produce_reviews = PythonOperator(
        task_id="produce_reviews",
        python_callable=_produce_reviews,
        execution_timeout=timedelta(hours=5),
    )

    wait_reviews = PythonOperator(
        task_id="wait_reviews",
        python_callable=_wait_reviews,
        execution_timeout=timedelta(minutes=2),
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=_preprocess,
        execution_timeout=timedelta(hours=1),
    )

    init_db >> produce_companies >> wait_companies >> produce_reviews >> wait_reviews >> preprocess
