# Phân tích cảm xúc review nhân sự (VI + EN)

Pipeline nghiên cứu và huấn luyện mô hình phát hiện **yếu tố tiêu cực** ở dạng nhị phân: `negative` (có bất kỳ dấu hiệu tiêu cực) vs `non-negative` (chỉ tích cực thuần).

- **Tiếng Việt (chính):** ~72k review từ [1900.com.vn](https://1900.com.vn) — review nhân sự ngành tài chính / công nghệ.
- **Tiếng Anh (mirror cùng domain):** ~838k review từ dataset Hugging Face [`lallantop/glassdoor`](https://huggingface.co/datasets/lallantop/glassdoor) — dùng làm nguồn đối chiếu, không phải benchmark license-clean.

Các họ mô hình được so sánh trên **cả VI và EN**: **TF-IDF** (LogReg, LinearSVC, ComplementNB, MLP, WordCharCue, cleanlab, resampling, VNReviewFusion/ENReviewFusion), **FastText 300-d + sklearn/MLP** (VI: `cc.vi.300.bin`, EN: `cc.en.300.bin`), **transformer fine-tune** (VI: PhoBERT, EN: DistilBERT). Chi tiết nghiên cứu: `NEGATIVE_NONNEGATIVE_RESEARCH_PLAN.md`, báo cáo kỹ thuật: `REPORT.md`.

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|------------|-----------|
| Python | **3.11+** (`requires-python` trong `pyproject.toml`) |
| Môi trường ảo | `venv` hoặc `uv` |
| PhoBERT / DistilBERT (tùy chọn) | `torch`, `transformers` — cần khi chạy transformer fine-tune hoặc pipeline đầy đủ |
| Docker (tùy chọn) | Chỉ cần nếu dùng crawl Postgres + Airflow + Streamlit DB |

> Không có `requirements.txt`; cài dependency từ `pyproject.toml`.

---

## Tải xuống & cài đặt

```bash
# Clone repo
git clone <repo-url> sentimentanalysis
cd sentimentanalysis

# Tạo virtualenv
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Cài package và dependencies
pip install -e ".[dev]"

# (Tùy chọn) PhoBERT — nếu chạy fine-tune transformer
pip install torch transformers
```

**Cách khác với `uv`:**

```bash
uv venv
uv pip install -e ".[dev]"
```

---

## Chạy trên máy khác

Sau khi `git push` / `git pull`, repo **không** chứa dữ liệu lớn, embedding FastText, hay artifact mô hình đã train. Làm theo các bước sau.

### 1. Clone và cài môi trường

```bash
git clone <repo-url> sentimentanalysis
cd sentimentanalysis

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"

# Bắt buộc nếu chạy smoke/full pipeline (PhoBERT + DistilBERT)
pip install torch transformers
```

Yêu cầu: **Python 3.11+** (`requires-python` trong `pyproject.toml`). Không có `requirements.txt`.

### Windows (sau `git pull` / clone)

Sau khi clone, bạn sẽ thấy `data/vi/raw/` và `data/en/glassdoor/` (file `.gitkeep` + `data/README.md`). **CSV và parquet vẫn không có trong Git.**

1. Kích hoạt venv: `.venv\Scripts\activate`
2. **Tiếng Việt:** copy `1900_export_reviews.csv` vào `data\vi\raw\` (hoặc chạy crawl — xem mục CSV bên dưới). Nếu thiếu thư mục: `mkdir data\vi\raw`
3. **Tiếng Anh:** không cần copy; chạy pipeline hoặc `python scripts\cache_glassdoor.py` — parquet sẽ được tạo trong `data\en\glassdoor\`.

### 2. Những gì KHÔNG có trong Git — lấy ở đâu

> **Lưu ý:** Thư mục `data/` có trong repo (placeholder `.gitkeep`); chỉ **nội dung** CSV/parquet bị ignore (xem `.gitignore`).


| Thành phần | Trong Git? | Cách có trên máy mới |
|------------|------------|----------------------|
| CSV review VI (`data/vi/raw/1900_export_reviews.csv`) | Không (`*.csv` bị ignore) | Copy từ máy cũ, hoặc crawl (xem bên dưới), hoặc dùng `analysis/1900_export_reviews.csv` nếu file đó đã được commit |
| Parquet EN Glassdoor (`data/en/glassdoor/*.parquet`) | Không | Tự tải lần chạy đầu từ Hugging Face, hoặc `python scripts/cache_glassdoor.py` |
| FastText `.bin` (`models/fasttext/{vi,en}/cc.*.300.bin`) | Không | Tự tải khi train lần đầu (`fasttext.util.download_model`), hoặc pre-download (mục FastText) |
| Artifact mô hình (`.pkl`, checkpoint, `models/**/runs/`) | Không | Sinh ra khi chạy train / pipeline — không cần copy từ máy cũ |
| Bảng thống kê (`models/statistics/*.csv`) | Không | Sinh sau `run-full-experiments` hoặc `build-statistics` |
| `.venv/`, `.env` | Không | Tạo lại trên máy mới |

**CSV tiếng Việt** — pipeline tìm theo thứ tự (`VI_DATA_CANDIDATES`):

1. `data/vi/raw/1900_export_reviews.csv`
2. `data_post_processing/1900_export_reviews.csv`
3. `analysis/1900_export_reviews.csv`

Nếu không có file nào → lỗi `Vietnamese CSV not found`. Cách nhanh để thử:

```bash
mkdir -p data/vi/raw
.venv/bin/python run.py crawl-csv --max-listing-pages 5 --max-companies 60 --target-rows 5000
```

Hoặc copy file CSV từ máy đã train (~72k dòng).

**Dữ liệu EN** — cần mạng ổn định; dataset `lallantop/glassdoor` (~838k review) tải qua package `datasets`.

### 3. Chạy tối thiểu: smoke vs full

| | Smoke test | Pipeline đầy đủ |
|---|------------|-----------------|
| Lệnh | `python analysis/run_full_experiment_pipeline.py --smoke` | `python analysis/run_full_experiment_pipeline.py` |
| Dữ liệu VI | ≥600 dòng (mặc định lấy 600) | Toàn bộ (~72k) |
| Dữ liệu EN | ~1500 dòng (tải HF) | Toàn bộ (~838k, tải HF) |
| Transformer | 256 mẫu, 1 epoch | Full data, 3 epoch |
| Thời gian | Vài phút–vài chục phút (CPU) | Nhiều giờ |
| Phụ thuộc thêm | `torch`, `transformers` | Cùng trên + đủ RAM/ổ đĩa |

Smoke chạy **tất cả** họ mô hình (TF-IDF, FastText, PhoBERT/DistilBERT) trên cả VI và EN.

Bỏ qua bước chậm khi full:

```bash
.venv/bin/python analysis/run_full_experiment_pipeline.py \
  --skip-vi-phobert --skip-en-fasttext
```

### 4. Docker (tùy chọn)

Chỉ cần khi dùng crawl Postgres + Airflow + Kafka/Spark + Streamlit — **không** bắt buộc cho pipeline huấn luyện / smoke test:

```bash
cp .env.example .env
docker compose up -d
.venv/bin/python run.py init-db
```

Chi tiết: mục [Crawl & hạ tầng (tùy chọn)](#crawl--hạ-tầng-tùy-chọn).

### 5. Lỗi thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|-------------|-------------|------------|
| `Vietnamese CSV not found` | Chưa có CSV VI | Crawl, copy từ máy cũ, hoặc đặt `--vi-csv <path>` |
| Lỗi tải / timeout Hugging Face | Chưa có parquet EN, mạng yếu | Chạy `scripts/cache_glassdoor.py` khi mạng ổn |
| `ModuleNotFoundError: torch` | Thiếu PyTorch | `pip install torch transformers` |
| FastText tải chậm lần đầu | File `.bin` ~ vài trăm MB | Chờ auto-download hoặc pre-download (mục FastText) |
| `underthesea` / NLP lỗi trên macOS | Thiếu dependency hệ thống | Dùng Python 3.11 từ python.org hoặc pyenv |
| Không thấy `model_statistics_latest.csv` | Chưa chạy pipeline / file bị gitignore | Chạy smoke hoặc full; file nằm local sau khi train |

---

## Dữ liệu

### Tiếng Việt (~72k review)

| Vị trí | Mô tả |
|--------|--------|
| `data/vi/raw/1900_export_reviews.csv` | File CSV chính cho huấn luyện |
| `data_post_processing/1900_export_reviews.csv` | Fallback |
| `analysis/1900_export_reviews.csv` | Fallback cũ |

Pipeline tự tìm file theo thứ tự trên (`VI_DATA_CANDIDATES` trong `src/training/labeling.py`).

**Nếu chưa có dữ liệu**, crawl trực tiếp ra CSV (không cần Postgres):

```bash
# Thu thập mẫu nhỏ để thử
.venv/bin/python run.py crawl-csv --max-listing-pages 5 --max-companies 60 --target-rows 5000

# Thu thập quy mô lớn hơn (điều chỉnh tham số theo nhu cầu)
.venv/bin/python run.py crawl-csv --target-rows 72000
```

Mặc định ghi ra `data/vi/raw/1900_export_reviews.csv`. Có thể đặt `--out <path>`.

### Tiếng Anh Glassdoor (~838k review)

| Vị trí | Mô tả |
|--------|--------|
| `data/en/glassdoor/labeled_reviews.parquet` | Cache sau lần tải đầu |
| `data/en/glassdoor/labeled_reviews_processed.parquet` | Bản đã preprocess (ưu tiên khi có) |

Lần chạy đầu sẽ tải từ Hugging Face (`datasets` package). Để cache trước:

```bash
.venv/bin/python scripts/cache_glassdoor.py
```

Cần kết nối mạng ổn định; file parquet ~ vài trăm MB sau khi xử lý.

---

## FastText embedding

Mô hình frozen FastText dùng cho FastText + sklearn/MLP:

| Ngôn ngữ | Vị trí ưu tiên | Legacy |
|----------|----------------|--------|
| VI | `models/fasttext/vi/cc.vi.300.bin` | `models/cc.vi.300.bin` |
| EN | `models/fasttext/en/cc.en.300.bin` | — |

**Tự động tải** khi chạy huấn luyện lần đầu (`fasttext.util.download_model`). Có thể tải thủ công trước:

```bash
.venv/bin/python -c "from src.training.trainer import _ensure_fasttext_model; _ensure_fasttext_model('vi')"
.venv/bin/python -c "from src.training.trainer import _ensure_fasttext_model; _ensure_fasttext_model('en')"
```

---

## Chạy thử nghiệm

### Smoke test (nhanh, ~ vài phút)

~600 dòng VI + ~1500 dòng EN, transformer 1 epoch:

```bash
.venv/bin/python analysis/run_full_experiment_pipeline.py --smoke
```

Tương đương:

```bash
.venv/bin/python run.py run-full-experiments --smoke
```

Pipeline smoke chạy **tất cả** họ mô hình trên cả VI và EN (TF-IDF, FastText, PhoBERT/DistilBERT).

### Pipeline đầy đủ

Toàn bộ dữ liệu; deploy mô hình TF-IDF VI tốt nhất vào `models/tfidf/vi/production/`:

```bash
.venv/bin/python analysis/run_full_experiment_pipeline.py
```

Tùy chọn bỏ qua bước chậm:

```bash
.venv/bin/python analysis/run_full_experiment_pipeline.py \
  --skip-vi-phobert --skip-en-fasttext
```

Tham số hữu ích: `--vi-csv`, `--vi-max-rows`, `--en-max-rows`, `--phobert-epochs`, `--skip-en-phobert`, `--skip-en-fasttext`, `--no-deploy-vi-best`.

### Huấn luyện từng họ mô hình

```bash
# TF-IDF variants (VI hoặc EN Glassdoor) — gồm cues, cleanlab, resampling, ReviewFusion
.venv/bin/python run.py train-variants --source current
.venv/bin/python run.py train-variants --source glassdoor-en

# FastText 305-d + classifiers (VI và EN)
.venv/bin/python run.py train-fasttext-binary --source current
.venv/bin/python run.py train-fasttext-binary --source glassdoor-en

# Transformer binary (VI: PhoBERT, EN: DistilBERT — cần torch + transformers)
.venv/bin/python run.py train-phobert-binary --language vi --max-examples 512 --epochs 1
.venv/bin/python run.py train-phobert-binary --language en --max-examples 512 --epochs 1
.venv/bin/python run.py train-phobert-binary --language vi --max-examples 0 --epochs 3

# So sánh dataset VI vs EN
.venv/bin/python run.py compare-glassdoor-datasets

# Tổng hợp bảng thống kê sau khi đã train (mặc định: mọi thuật toán riêng biệt)
.venv/bin/python run.py build-statistics
```

### Các entry point khác

| Lệnh | Mục đích |
|------|----------|
| `run.py train` | Pipeline huấn luyện 3-class cổ điển (FastText + ensemble) |
| `run.py absa` | Phân tích aspect-level (ABSA) trên CSV review |
| `run.py scan-vietnamese-text` | Quét slang / token tiếng Việt nhiễu |
| `run.py crawl` / `preprocess` / `ui` | Crawl Postgres, preprocess DB, Streamlit export |
| `analysis/train_binary_models.py` | Script phân tích/huấn luyện bổ sung trong `analysis/` |

> Không có REST API trong repo; inference qua module Python (`src/training/trainer.py`, artifacts trong `models/`).

---

## Đầu ra (outputs)

| Artifact | Đường dẫn |
|----------|-----------|
| Bảng thống kê mới nhất | `models/statistics/model_statistics_latest.csv` (+ `.json`) |
| Bảng có timestamp | `models/statistics/model_statistics_YYYYMMDD_HHMMSS.csv` |
| Manifest pipeline | `models/statistics/full_experiment_manifest.json` |
| TF-IDF production (VI) | `models/tfidf/vi/production/best_model.pkl`, `meta.json` |
| PhoBERT checkpoint (VI) | `models/phobert/vi/production/best/` |
| DistilBERT checkpoint (EN) | `models/phobert/en/production/best/` |
| FastText runs | `models/fasttext/{vi,en}/runs/` |
| So sánh dataset | `models/comparisons/glassdoor_dataset_comparison_*.json` |
| Kết quả huấn luyện | `analysis/training_results.json`, `models/experiments.json` |

Log chạy nền thường nằm trong `models/statistics/*.log`.

---

## Cấu trúc thư mục (tóm tắt)

```
sentimentanalysis/
├── run.py                          # CLI chính
├── analysis/
│   └── run_full_experiment_pipeline.py   # Pipeline thống nhất VI + EN
├── scripts/
│   ├── build_final_statistics.py   # Tổng hợp CSV thống kê
│   └── cache_glassdoor.py        # Cache EN parquet
├── src/
│   ├── crawler/                    # Crawl 1900.com.vn
│   ├── preprocessing/              # NLP tiếng Việt
│   ├── training/                   # Trainers (TF-IDF, FastText, PhoBERT)
│   ├── analysis/                   # ABSA
│   └── artifacts/paths.py          # Layout models/
├── data/
│   ├── vi/raw/                     # CSV review tiếng Việt
│   └── en/glassdoor/               # Parquet Glassdoor
└── models/
    ├── tfidf/{vi,en}/
    ├── fasttext/{vi,en}/
    ├── phobert/{vi,en}/
    ├── statistics/
    └── comparisons/
```

---

## Crawl & hạ tầng (tùy chọn)

Nếu cần pipeline crawl đầy đủ (Postgres + Airflow + Kafka/Spark):

```bash
cp .env.example .env          # SESSION_COOKIE nếu cần đọc review đầy đủ
docker compose up -d
.venv/bin/python run.py init-db
.venv/bin/python run.py crawl --max-pages 5
.venv/bin/python run.py ui    # Streamlit http://localhost:8501
```

Chi tiết kiến trúc và schema DB: xem `REPORT.md` mục System Architecture.

---

## Tài liệu liên quan

- `NEGATIVE_NONNEGATIVE_RESEARCH_PLAN.md` — kế hoạch nghiên cứu, RQ, thiết kế đánh giá
- `REPORT.md` — báo cáo pipeline ABSA + huấn luyện (bản tiếng Anh)
- `pyproject.toml` — dependencies và optional extras (`dev`, `lstm`)
