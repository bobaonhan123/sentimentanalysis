from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Postgres
    postgres_user: str = "review"
    postgres_password: str = "changeme"
    postgres_db: str = "review_company"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = "postgresql://review:changeme@localhost:5432/review_company"

    # Kafka
    kafka_bootstrap_servers: str = "kafka:9092"

    # Crawler
    crawl_delay: float = 1.5
    crawl_max_pages: int = 250
    crawl_concurrency: int = 3
    bloom_capacity: int = 500_000
    bloom_error_rate: float = 0.001

    # Session cookie (optional – paste from browser for full reviews)
    session_cookie: str = ""


settings = Settings()
