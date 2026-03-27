"""Streamlit debug/export app for review data."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in path
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
from sqlalchemy import func, select, text

from src.database import get_session
from src.models import Company, Review

st.set_page_config(page_title="1900.com.vn Review Explorer", layout="wide")
st.title("🔍 1900.com.vn — Review Explorer & Export")


# ── Sidebar: stats & filters ─────────────────────────────────────
@st.cache_data(ttl=60)
def get_stats():
    session = get_session()
    company_count = session.execute(select(func.count(Company.id))).scalar() or 0
    review_count = session.execute(select(func.count(Review.id))).scalar() or 0
    preprocessed = session.execute(
        select(func.count(Review.id)).where(Review.pros_clean.isnot(None))
    ).scalar() or 0
    session.close()
    return company_count, review_count, preprocessed


company_count, review_count, preprocessed_count = get_stats()

st.sidebar.header("📊 Database Stats")
st.sidebar.metric("Companies", f"{company_count:,}")
st.sidebar.metric("Reviews", f"{review_count:,}")
st.sidebar.metric("Preprocessed", f"{preprocessed_count:,}")

st.sidebar.divider()
st.sidebar.header("🔧 Filters")

# ── Industry filter ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_industries():
    session = get_session()
    rows = session.execute(
        select(Company.industry).where(Company.industry.isnot(None)).distinct()
    ).scalars().all()
    session.close()
    return sorted(set(r for r in rows if r))


industries = get_industries()
selected_industry = st.sidebar.selectbox("Industry", ["All"] + industries)

# ── Location filter ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_locations():
    session = get_session()
    rows = session.execute(
        select(Company.location).where(Company.location.isnot(None)).distinct()
    ).scalars().all()
    session.close()
    return sorted(set(r for r in rows if r))


locations = get_locations()
selected_location = st.sidebar.selectbox("Location", ["All"] + locations)

# ── Rating filter ────────────────────────────────────────────────
min_rating = st.sidebar.slider("Min Rating", 0.0, 5.0, 0.0, 0.5)

# ── Search ───────────────────────────────────────────────────────
search_term = st.sidebar.text_input("Search (company name or review text)")


# ── Tabs ──────────────────────────────────────────────────────────
tab_companies, tab_reviews, tab_export = st.tabs(["🏢 Companies", "💬 Reviews", "📥 Export"])


# ── Tab 1: Companies ─────────────────────────────────────────────
with tab_companies:
    st.subheader("Companies")

    page_size_c = st.selectbox("Page size", [20, 50, 100, 500], key="ps_c")
    page_num_c = st.number_input("Page", min_value=1, value=1, key="pn_c")

    @st.cache_data(ttl=30)
    def load_companies(_industry, _location, _min_rating, _search, _page, _page_size):
        session = get_session()
        q = select(Company)

        if _industry != "All":
            q = q.where(Company.industry == _industry)
        if _location != "All":
            q = q.where(Company.location == _location)
        if _min_rating > 0:
            q = q.where(Company.overall_rating >= _min_rating)
        if _search:
            q = q.where(Company.name.ilike(f"%{_search}%"))

        count_q = select(func.count()).select_from(q.subquery())
        total = session.execute(count_q).scalar() or 0

        q = q.order_by(Company.review_count.desc().nullslast())
        q = q.offset((_page - 1) * _page_size).limit(_page_size)

        rows = session.execute(q).scalars().all()
        session.close()

        data = [{
            "ID": c.site_id,
            "Name": c.name,
            "Industry": c.industry,
            "Location": c.location,
            "Rating": c.overall_rating,
            "Reviews": c.review_count,
            "URL": c.url,
        } for c in rows]

        return pd.DataFrame(data), total

    df_c, total_c = load_companies(selected_industry, selected_location, min_rating, search_term, page_num_c, page_size_c)
    st.caption(f"Total: {total_c:,} companies | Page {page_num_c} of {max(1, (total_c + page_size_c - 1) // page_size_c)}")
    st.dataframe(df_c, use_container_width=True)


# ── Tab 2: Reviews ───────────────────────────────────────────────
with tab_reviews:
    st.subheader("Reviews")

    page_size_r = st.selectbox("Page size", [20, 50, 100, 500], key="ps_r")
    page_num_r = st.number_input("Page", min_value=1, value=1, key="pn_r")
    show_clean = st.checkbox("Show preprocessed text", value=False)

    @st.cache_data(ttl=30)
    def load_reviews(_industry, _location, _min_rating, _search, _page, _page_size, _show_clean):
        session = get_session()
        q = select(Review, Company.name.label("company_name")).join(Company, Review.company_id == Company.id)

        if _industry != "All":
            q = q.where(Company.industry == _industry)
        if _location != "All":
            q = q.where(Company.location == _location)
        if _min_rating > 0:
            q = q.where(Review.rating >= _min_rating)
        if _search:
            q = q.where(
                Company.name.ilike(f"%{_search}%")
                | Review.pros.ilike(f"%{_search}%")
                | Review.cons.ilike(f"%{_search}%")
                | Review.title.ilike(f"%{_search}%")
            )

        count_q = select(func.count()).select_from(q.subquery())
        total = session.execute(count_q).scalar() or 0

        q = q.order_by(Review.created_at.desc())
        q = q.offset((_page - 1) * _page_size).limit(_page_size)

        rows = session.execute(q).all()
        session.close()

        data = []
        for row in rows:
            review = row[0]
            company_name = row[1]
            entry = {
                "Company": company_name,
                "Rating": review.rating,
                "Title": review.title,
                "Job": review.job_title,
                "Status": review.employee_status,
                "Location": review.review_location,
                "Date": review.review_date,
            }
            if _show_clean:
                entry["Pros (clean)"] = review.pros_clean
                entry["Cons (clean)"] = review.cons_clean
            else:
                entry["Pros"] = review.pros
                entry["Cons"] = review.cons
            entry["Advice"] = review.advice
            data.append(entry)

        return pd.DataFrame(data), total

    df_r, total_r = load_reviews(selected_industry, selected_location, min_rating, search_term, page_num_r, page_size_r, show_clean)
    st.caption(f"Total: {total_r:,} reviews | Page {page_num_r} of {max(1, (total_r + page_size_r - 1) // page_size_r)}")
    st.dataframe(df_r, use_container_width=True)


# ── Tab 3: Export ─────────────────────────────────────────────────
with tab_export:
    st.subheader("Export to CSV")

    export_what = st.radio("Export", ["Companies", "Reviews", "Reviews (preprocessed)"])

    col1, col2 = st.columns(2)
    with col1:
        export_limit = st.number_input("Max rows (0 = all)", min_value=0, value=0, step=1000)
    with col2:
        include_all_fields = st.checkbox("Include all fields", value=True)

    if st.button("🚀 Generate CSV", type="primary"):
        session = get_session()

        if export_what == "Companies":
            q = select(Company)
            if selected_industry != "All":
                q = q.where(Company.industry == selected_industry)
            if selected_location != "All":
                q = q.where(Company.location == selected_location)
            if min_rating > 0:
                q = q.where(Company.overall_rating >= min_rating)
            if search_term:
                q = q.where(Company.name.ilike(f"%{search_term}%"))
            if export_limit > 0:
                q = q.limit(export_limit)

            rows = session.execute(q).scalars().all()
            df = pd.DataFrame([{
                "site_id": c.site_id,
                "name": c.name,
                "industry": c.industry,
                "employee_range": c.employee_range,
                "location": c.location,
                "overall_rating": c.overall_rating,
                "review_count": c.review_count,
                "url": c.url,
            } for c in rows])

        else:
            q = select(Review, Company.name.label("company_name"), Company.industry.label("company_industry"))
            q = q.join(Company, Review.company_id == Company.id)
            if selected_industry != "All":
                q = q.where(Company.industry == selected_industry)
            if selected_location != "All":
                q = q.where(Company.location == selected_location)
            if min_rating > 0:
                q = q.where(Review.rating >= min_rating)
            if search_term:
                q = q.where(
                    Company.name.ilike(f"%{search_term}%")
                    | Review.pros.ilike(f"%{search_term}%")
                    | Review.cons.ilike(f"%{search_term}%")
                )
            if export_limit > 0:
                q = q.limit(export_limit)

            rows = session.execute(q).all()
            use_clean = "preprocessed" in export_what.lower()

            df = pd.DataFrame([{
                "company": row[1],
                "industry": row[2],
                "rating": row[0].rating,
                "title": row[0].title,
                "job_title": row[0].job_title,
                "employee_status": row[0].employee_status,
                "location": row[0].review_location,
                "date": row[0].review_date,
                "pros": row[0].pros_clean if use_clean else row[0].pros,
                "cons": row[0].cons_clean if use_clean else row[0].cons,
                "advice": row[0].advice,
                "recommends": row[0].recommends,
            } for row in rows])

        session.close()

        st.success(f"Generated {len(df):,} rows")
        st.dataframe(df.head(100), use_container_width=True)

        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"1900_export_{export_what.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )
