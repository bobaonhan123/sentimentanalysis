# Thư mục `data/`

Repo giữ **cấu trúc thư mục** (file `.gitkeep`); file dữ liệu lớn **không** nằm trong Git.

| Đường dẫn | Nội dung | Cách có trên máy mới |
|-----------|----------|----------------------|
| `vi/raw/` | CSV review tiếng Việt (`1900_export_reviews.csv`) | Copy từ máy đã train, crawl (`run.py crawl-csv`), hoặc dùng fallback `analysis/1900_export_reviews.csv` |
| `en/glassdoor/` | Parquet Glassdoor (cache) | Tự tải lần chạy đầu từ Hugging Face (`datasets`), hoặc `python scripts/cache_glassdoor.py` |

**Tiếng Việt:** đặt file CSV tại `data/vi/raw/1900_export_reviews.csv`. Đây là đường dẫn chuẩn đang được pipeline ưu tiên.

**Tiếng Anh:** không cần copy thủ công — chỉ cần mạng; parquet sẽ xuất hiện trong `en/glassdoor/` sau lần tải/cache đầu tiên.
