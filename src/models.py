"""SQLAlchemy ORM models for company reviews from 1900.com.vn.

Uses classic Column() syntax for compatibility with SQLAlchemy 1.4 (Airflow) and 2.0.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Company(Base):
    __tablename__ = "companies"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    site_id = Column(Integer, unique=True, nullable=False, comment="ID trên 1900.com.vn")
    slug = Column(String(500), nullable=False)
    name = Column(String(500), nullable=False)
    industry = Column(String(300))
    employee_range = Column(String(100))
    location = Column(String(300))
    overall_rating = Column(Float)
    review_count = Column(Integer, default=0)
    url = Column(String(1000), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    reviews = relationship("Review", back_populates="company", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_companies_site_id", "site_id"),)


class Review(Base):
    __tablename__ = "reviews"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    company_id = Column(BigInteger, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    fingerprint = Column(String(64), unique=True, nullable=False, comment="SHA-256 dedup fingerprint")

    title = Column(Text)
    rating = Column(Float)
    job_title = Column(String(300))
    employee_status = Column(String(50), comment="Nhân viên hiện tại / cũ")
    review_location = Column(String(300))
    review_date = Column(String(100))

    pros = Column(Text, comment="Ưu điểm")
    cons = Column(Text, comment="Nhược điểm")
    advice = Column(Text, comment="Lời khuyên cho quản lý")

    recommends = Column(String(50))
    ceo_rating = Column(String(50))
    business_outlook = Column(String(50))

    # Preprocessed fields
    pros_clean = Column(Text, comment="Ưu điểm sau xử lí NLP")
    cons_clean = Column(Text, comment="Nhược điểm sau xử lí NLP")

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    company = relationship("Company", back_populates="reviews")

    __table_args__ = (
        Index("ix_reviews_company_id", "company_id"),
        Index("ix_reviews_fingerprint", "fingerprint"),
    )
