"""Spark Structured Streaming consumer: Kafka → PostgreSQL.

Reads parsed company/review JSON from Kafka topics and upserts into Postgres
using foreachBatch with SQLAlchemy.

Run: python src/streaming/consumer.py
"""
from __future__ import annotations

import logging
import os
import sys

# Ensure project root on path
sys.path.insert(0, os.environ.get("APP_ROOT", "/opt/spark/app"))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    FloatType, IntegerType, StringType, StructField, StructType,
)

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
POSTGRES_URL = os.environ.get("DATABASE_URL", "postgresql://review:changeme@postgres:5432/review_company")
CHECKPOINT_DIR = os.environ.get("SPARK_CHECKPOINT_DIR", "/opt/spark/checkpoints")

# ── Kafka value schemas ───────────────────────────────────────────

COMPANY_SCHEMA = StructType([
    StructField("site_id", IntegerType()),
    StructField("slug", StringType()),
    StructField("name", StringType()),
    StructField("url", StringType()),
    StructField("industry", StringType()),
    StructField("employee_range", StringType()),
    StructField("location", StringType()),
    StructField("overall_rating", FloatType()),
    StructField("review_count", IntegerType()),
])

REVIEW_SCHEMA = StructType([
    StructField("title", StringType()),
    StructField("rating", FloatType()),
    StructField("job_title", StringType()),
    StructField("employee_status", StringType()),
    StructField("review_location", StringType()),
    StructField("review_date", StringType()),
    StructField("pros", StringType()),
    StructField("cons", StringType()),
    StructField("advice", StringType()),
    StructField("recommends", StringType()),
    StructField("ceo_rating", StringType()),
    StructField("business_outlook", StringType()),
    StructField("fingerprint", StringType()),
    StructField("company_site_id", IntegerType()),
])


# ── DB helpers (singleton engine) ─────────────────────────────────

_engine = None
_SessionFactory = None


def _get_session():
    global _engine, _SessionFactory
    if _engine is None or _SessionFactory is None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        if _engine is None:
            _engine = create_engine(POSTGRES_URL, pool_pre_ping=True, pool_size=5)
        _SessionFactory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _SessionFactory()


def _ensure_tables():
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine
        _engine = create_engine(POSTGRES_URL, pool_pre_ping=True, pool_size=5)
    from src.models import Base
    Base.metadata.create_all(_engine)


# ── Batch processors ─────────────────────────────────────────────

def _upsert_companies(df: DataFrame, batch_id: int):
    """foreachBatch callback: upsert companies into Postgres."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from src.models import Company

    rows = df.collect()
    if not rows:
        return

    session = _get_session()
    try:
        for row in rows:
            stmt = pg_insert(Company).values(
                site_id=row["site_id"],
                slug=row["slug"],
                name=row["name"],
                industry=row["industry"],
                employee_range=row["employee_range"],
                location=row["location"],
                overall_rating=row["overall_rating"],
                review_count=row["review_count"],
                url=row["url"],
            ).on_conflict_do_update(
                index_elements=["site_id"],
                set_={
                    "name": row["name"],
                    "industry": row["industry"],
                    "employee_range": row["employee_range"],
                    "location": row["location"],
                    "overall_rating": row["overall_rating"],
                    "review_count": row["review_count"],
                    "url": row["url"],
                },
            )
            session.execute(stmt)
        session.commit()
        logger.info(f"Batch {batch_id}: upserted {len(rows)} companies")
    except Exception as e:
        session.rollback()
        logger.error(f"Company batch {batch_id} error: {e}")
        raise
    finally:
        session.close()


def _upsert_reviews(df: DataFrame, batch_id: int):
    """foreachBatch callback: upsert reviews into Postgres."""
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from src.models import Company, Review

    rows = df.collect()
    if not rows:
        return

    session = _get_session()
    try:
        # Build site_id → company.id mapping
        site_ids = list({row["company_site_id"] for row in rows if row["company_site_id"]})
        company_map = {}
        if site_ids:
            companies = session.execute(
                select(Company).where(Company.site_id.in_(site_ids))
            ).scalars().all()
            company_map = {c.site_id: c.id for c in companies}

        new_count = 0
        for row in rows:
            company_id = company_map.get(row["company_site_id"])
            if not company_id:
                continue

            stmt = pg_insert(Review).values(
                company_id=company_id,
                fingerprint=row["fingerprint"],
                title=row["title"],
                rating=row["rating"],
                job_title=row["job_title"],
                employee_status=row["employee_status"],
                review_location=row["review_location"],
                review_date=row["review_date"],
                pros=row["pros"],
                cons=row["cons"],
                advice=row["advice"],
                recommends=row["recommends"],
                ceo_rating=row["ceo_rating"],
                business_outlook=row["business_outlook"],
            ).on_conflict_do_nothing(index_elements=["fingerprint"])
            result = session.execute(stmt)
            if result.rowcount > 0:
                new_count += 1

        session.commit()
        logger.info(f"Review batch {batch_id}: inserted {new_count}/{len(rows)} new reviews")
    except Exception as e:
        session.rollback()
        logger.error(f"Review batch {batch_id} error: {e}")
        raise
    finally:
        session.close()


def _ensure_topics():
    """Create Kafka topics if they don't exist."""
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError

    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
    for topic_name in ("crawled-companies", "crawled-reviews"):
        try:
            admin.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
            logger.info(f"Created Kafka topic: {topic_name}")
        except TopicAlreadyExistsError:
            logger.info(f"Kafka topic already exists: {topic_name}")
        except Exception as e:
            logger.warning(f"Could not create topic {topic_name}: {e}")
    admin.close()


# ── Spark Streaming ───────────────────────────────────────────────

def start_consumer():
    """Start Spark Structured Streaming consumers for both Kafka topics."""
    _ensure_tables()
    _ensure_topics()

    spark = (
        SparkSession.builder
        .appName("ReviewConsumer")
        .master("local[*]")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3")
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR)
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Starting Spark Structured Streaming consumers...")

    # ── Company stream ──
    companies_raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", "crawled-companies")
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )
    companies_parsed = (
        companies_raw
        .select(from_json(col("value").cast("string"), COMPANY_SCHEMA).alias("d"))
        .select("d.*")
    )
    company_query = (
        companies_parsed.writeStream
        .foreachBatch(_upsert_companies)
        .outputMode("append")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/companies")
        .trigger(processingTime="5 seconds")
        .queryName("companies_ingestion")
        .start()
    )

    # ── Review stream ──
    reviews_raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", "crawled-reviews")
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )
    reviews_parsed = (
        reviews_raw
        .select(from_json(col("value").cast("string"), REVIEW_SCHEMA).alias("d"))
        .select("d.*")
    )
    review_query = (
        reviews_parsed.writeStream
        .foreachBatch(_upsert_reviews)
        .outputMode("append")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/reviews")
        .trigger(processingTime="5 seconds")
        .queryName("reviews_ingestion")
        .start()
    )

    logger.info("Consumers started. Waiting for Kafka data...")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    start_consumer()
