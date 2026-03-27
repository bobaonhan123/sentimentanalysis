"""Streamlit EDA Dashboard – Vietnamese Company Reviews (1900.com.vn)."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Review EDA Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "1900_export_reviews.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["pros_len"] = df["pros"].str.len().fillna(0).astype(int)
    df["cons_len"] = df["cons"].str.len().fillna(0).astype(int)
    # Derive simple sentiment from rating
    df["sentiment"] = pd.cut(
        df["rating"], bins=[0, 2.5, 4.0, 5.0], labels=["negative", "neutral", "positive"]
    )
    return df


df = load_data()

# ─── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.header("🔍 Bộ lọc")

selected_companies = st.sidebar.multiselect(
    "Công ty",
    options=sorted(df["company"].unique()),
    default=[],
    placeholder="Tất cả",
)

selected_industries = st.sidebar.multiselect(
    "Ngành nghề",
    options=sorted(df["industry"].unique()),
    default=[],
    placeholder="Tất cả",
)

selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=["positive", "neutral", "negative"],
    default=[],
    placeholder="Tất cả",
)

rating_range = st.sidebar.slider("Khoảng rating", 1.0, 5.0, (1.0, 5.0), 0.5)

date_range = st.sidebar.date_input(
    "Khoảng thời gian",
    value=(df["date"].min(), df["date"].max()),
    min_value=df["date"].min(),
    max_value=df["date"].max(),
)

# Apply filters
mask = pd.Series(True, index=df.index)
if selected_companies:
    mask &= df["company"].isin(selected_companies)
if selected_industries:
    mask &= df["industry"].isin(selected_industries)
if selected_sentiments:
    mask &= df["sentiment"].isin(selected_sentiments)
mask &= df["rating"].between(rating_range[0], rating_range[1])
if len(date_range) == 2:
    mask &= df["date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))

filtered = df[mask]

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("📊 EDA Dashboard – Đánh giá công ty 1900.com.vn")
st.caption(f"Tổng số review sau lọc: **{len(filtered):,}** / {len(df):,}")

# ─── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Tổng reviews", f"{len(filtered):,}")
k2.metric("Rating TB", f"{filtered['rating'].mean():.2f}" if len(filtered) else "N/A")
k3.metric("Positive %", f"{(filtered['sentiment'] == 'positive').mean() * 100:.1f}%" if len(filtered) else "N/A")
k4.metric("Negative %", f"{(filtered['sentiment'] == 'negative').mean() * 100:.1f}%" if len(filtered) else "N/A")
k5.metric("Số công ty", f"{filtered['company'].nunique()}")

st.divider()

# ─── Row 1: Sentiment distribution & Rating distribution ─────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Phân bố Sentiment (Ước tính theo rating)")
    sentiment_counts = filtered["sentiment"].value_counts()
    st.bar_chart(sentiment_counts, color="#5dade2")

with col2:
    st.subheader("Phân bố Rating")
    rating_counts = filtered["rating"].value_counts().sort_index()
    st.bar_chart(rating_counts, color="#8e44ad")

# ─── Row 2: Reviews over time ────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Số lượng review theo tháng")
    monthly = filtered.groupby("month").size().reset_index(name="count")
    monthly = monthly.sort_values("month")
    st.line_chart(monthly.set_index("month")["count"])

with col4:
    st.subheader("Sentiment theo thời gian")
    monthly_sent = (
        filtered.groupby(["month", "sentiment"]).size().unstack(fill_value=0)
    )
    monthly_sent = monthly_sent.sort_index()
    st.area_chart(monthly_sent)

# ─── Row 3: By company & by industry ─────────────────────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("Rating trung bình theo công ty")
    company_rating = (
        filtered.groupby("company")["rating"]
        .mean()
        .sort_values(ascending=True)
    )
    st.bar_chart(company_rating, horizontal=True, color="#1abc9c")

with col6:
    st.subheader("Số review theo ngành nghề")
    industry_counts = filtered["industry"].value_counts()
    st.bar_chart(industry_counts, color="#e67e22")

# ─── Row 4: Employee status & Recommendation ─────────────────────────────────
col7, col8 = st.columns(2)

with col7:
    st.subheader("Nhân viên hiện tại vs cũ")
    status_counts = filtered["employee_status"].value_counts()
    st.bar_chart(status_counts, color="#2980b9")

with col8:
    st.subheader("Tỷ lệ đề xuất (Recommends)")
    rec = filtered["recommends"].fillna("Không rõ").value_counts()
    st.bar_chart(rec, color="#3498db")

# ─── Row 5: Text length analysis ─────────────────────────────────────────────
st.divider()
st.subheader("📝 Phân tích độ dài văn bản theo Rating")

col9, col10 = st.columns(2)

with col9:
    st.write("**Độ dài phần Ưu điểm (pros)**")
    pros_stats = filtered.groupby("sentiment")["pros_len"].describe()[["mean", "50%", "max"]]
    pros_stats.columns = ["Trung bình", "Trung vị", "Max"]
    st.dataframe(pros_stats.round(1), use_container_width=True)

with col10:
    st.write("**Độ dài phần Nhược điểm (cons)**")
    cons_stats = filtered.groupby("sentiment")["cons_len"].describe()[["mean", "50%", "max"]]
    cons_stats.columns = ["Trung bình", "Trung vị", "Max"]
    st.dataframe(cons_stats.round(1), use_container_width=True)

# ─── Row 6: Rating by job title ──────────────────────────────────────────────
st.divider()
st.subheader("Rating trung bình theo vị trí công việc")

job_stats = (
    filtered.groupby("job_title")
    .agg(rating_mean=("rating", "mean"), review_count=("rating", "size"))
    .sort_values("rating_mean", ascending=False)
)
job_stats.columns = ["Rating TB", "Số review"]
st.dataframe(job_stats.round(2), use_container_width=True)

# ─── Row 7: Top review locations ─────────────────────────────────────────────
st.subheader("Phân bố review theo địa điểm")
loc_counts = filtered["location"].value_counts()
st.bar_chart(loc_counts, color="#16a085")

# ─── Row 8: Word frequency ───────────────────────────────────────────────────
st.divider()
st.subheader("🔤 Từ khóa phổ biến trong Ưu điểm / Nhược điểm")


def top_words(texts: pd.Series, top_n: int = 20) -> pd.DataFrame:
    stop = {"và", "là", "của", "cho", "có", "được", "trong", "không", "với", "rất",
            "các", "nhưng", "một", "để", "khi", "từ", "làm", "tại", "về", "như",
            "hơn", "đã", "hay", "này", "theo", "nên", "vì", "lên", "đến", "ra",
            "thì", "cũng", "mà", "bị", "ở", "do", "nhiều", "quá", "phải", "sẽ"}
    words: list[str] = []
    for text in texts.dropna():
        tokens = re.findall(r"\w+", text.lower())
        words.extend(t for t in tokens if t not in stop and len(t) > 1)
    counts = Counter(words).most_common(top_n)
    return pd.DataFrame(counts, columns=["Từ", "Tần suất"]).set_index("Từ")


wc1, wc2 = st.columns(2)

with wc1:
    st.write("**Ưu điểm (Pros)**")
    st.bar_chart(top_words(filtered["pros"]), color="#27ae60")

with wc2:
    st.write("**Nhược điểm (Cons)**")
    st.bar_chart(top_words(filtered["cons"]), color="#c0392b")

# ─── Sample data ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Dữ liệu mẫu")

display_cols = [
    "company", "industry", "rating", "sentiment", "title",
    "job_title", "employee_status", "location", "date", "pros", "cons",
    "advice", "recommends",
]
st.dataframe(
    filtered[display_cols].sort_values("date", ascending=False).head(50),
    use_container_width=True,
    height=400,
)
