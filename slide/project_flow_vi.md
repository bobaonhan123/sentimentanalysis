# Giai thich flow project va cau hoi co the bi hoi

## 1. Dinh vi de tai

De tai nen duoc trinh bay la **Employee Review Intelligence Dashboard** thay vi chi la mot bai sentiment classification. Ly do la output cuoi cung khong chi tra ve positive/negative/neutral, ma giup HR va management tra loi cac cau hoi:

- Nhan vien dang phan nan nhieu nhat ve van de gi?
- Van de nao dang xau di theo thoi gian?
- Van de tap trung o nganh, nhom cong ty, hay cong ty nao?
- Dau la uu tien can hanh dong truoc?

Do do, website/dashboard la mot diem cong vi chung minh model co the duoc dua vao ung dung thuc te.

## 2. Flow tong the

```text
1900.com.vn
  -> Scraper
  -> Bloom Filter
  -> Kafka Producer / Consumer
  -> PostgreSQL
  -> Vietnamese NLP Preprocessing
  -> Weak Labeling + Label Quality Audit
  -> Sentiment Models
  -> ABSA Rule Engine
  -> Streamlit Dashboard / Report
```

## 3. Giai thich tung thanh phan

### Scraper

Scraper dung de thu thap review cong ty tu 1900.com.vn. Vi khong co san dataset gan nhan cho mien du lieu workplace review tieng Viet, nhom tu crawl 10,000 review.

**Tai sao can scraper?**

- Tao dataset rieng cho dung domain.
- Chu dong cap nhat du lieu moi.
- Lay duoc metadata nhu company, industry, location, date.

### Bloom Filter

Bloom Filter dung de tranh crawl trung URL da gap.

**Tai sao khong dung list/set binh thuong?**

- Bloom Filter nhe hon ve bo nho khi so URL lon.
- Phu hop voi pipeline crawl lap lai theo lich.
- Doi lai co xac suat false positive nho, nhung chap nhan duoc trong bai toan crawl.

### Kafka

Kafka dong vai tro message queue giua producer va consumer.

**Tai sao can Kafka?**

- Tach qua trinh crawl va xu ly.
- Neu consumer cham, producer van co the day message vao queue.
- Tang tinh on dinh va mo rong cho pipeline.

Neu giao vien hoi "dataset chi 10,000 review co can Kafka khong?", co the tra loi:

> Voi quy mo hien tai, Kafka khong bat buoc. Tuy nhien, project duoc thiet ke theo huong production pipeline, co the mo rong khi crawl them cong ty hoac chay dinh ky. Kafka la diem cong ve kien truc, khong phai yeu cau toi thieu.

### PostgreSQL

PostgreSQL luu review va metadata co cau truc.

**Tai sao can database?**

- De truy van theo company, industry, date, employee_status.
- Phuc vu dashboard va export.
- Giu du lieu ben vung hon so voi file CSV don le.

### NLP Preprocessing

Preprocessing gom normalize Unicode, lowercase, remove URL/email/HTML, tokenize tieng Viet, va xu ly stopwords.

**Tai sao tieng Viet kho hon tieng Anh?**

- Tu tieng Viet co the gom nhieu tieng, can word segmentation.
- Dau tieng Viet va Unicode co the bi lech encoding.
- Review thuc te co nhieu loi viet, viet tat, tieng Anh chen lan.

### Weak Labeling

Vi khong co nhan human-labeled, nhom dung star rating lam label ban dau:

- 1-2 sao -> negative
- 3 sao -> neutral
- 4-5 sao -> positive

Sau do dung keyword va ABSA signal de override mot so truong hop mau thuan.

**Diem yeu cua cach nay la gi?**

- Rating khong phai luc nao cung khop voi noi dung text.
- Neutral 3 sao co the la mixed opinion, khong that su neutral.
- Vi vay can label quality audit.

### Confident Learning / Cleanlab

Cleanlab duoc dung de phat hien label co kha nang sai. No dung predicted probability tu cross-validation de xem mau nao co label goc rat khong phu hop voi du doan cua model.

Ket qua:

- Phat hien 1,570 label nghi ngo.
- Sau khi prune, 3-class macro F1 tang tu 0.720 len 0.877.

**Cach giai thich voi giao vien:**

> Ket qua nay cho thay van de chinh khong phai model qua yeu, ma la label noise, dac biet o class neutral. Khi xu ly chat luong label, hieu nang tang rat ro.

### Sentiment Models

Nhom so sanh nhieu setting:

- Original 3-class
- Cleaned 3-class
- Binary no-neutral
- Mixed/conflict 4-class
- Cac baseline ML/DL: TF-IDF, FastText + MLP, Random Forest, Logistic Regression, LinearSVC, LSTM

Ket qua tot nhat:

- Binary no-neutral: Macro F1 = 0.918
- Cleaned 3-class: Macro F1 = 0.877

Hai huong chinh nay da duoc implement va train trong code:

- `binary_no_neutral`: da luu lam best deploy model trong `models/best_model.pkl`.
- cleaned 3-class: da luu artifact rieng trong `models/variants/`.

**Tai sao binary cao hon?**

Vi bo neutral lam bai toan de hon. Model chi can phan biet hai cuc sentiment ro rang.

**Co nen bo neutral khong?**

Khong nen bo hoan toan. Binary phu hop cho alert positive/negative. Cleaned 3-class phu hop cho phan tich co neutral.

### ABSA Rule Engine

ABSA dung de phan tich aspect-level sentiment. Thay vi chi noi review nay positive hay negative, he thong tra loi review dang noi ve van de nao:

- Salary & Benefits
- Work Environment
- Management & Leadership
- Work-Life Balance
- Career Growth
- Facilities & Tools
- Process & Communication

**Tai sao ABSA quan trong?**

Vi business action phai dua theo aspect. Biet mot review negative la chua du; HR can biet no negative vi luong, quan ly, hay workload.

### Streamlit Dashboard

Dashboard hien thi model result, aspect summary, trend, va hotspot.

**Day co phai diem cong khong?**

Co. Vi no bien model thanh mot proof-of-concept co the dung duoc. Tuy nhien khi thuyet trinh, nen noi dashboard la cong cu ung dung, khong phai dong gop khoa hoc chinh.

## 4. Cau hoi de bi giao vien hoi

### Q1. Tai sao accuracy ban dau chi khoang 79-80%?

Vi day la bai toan 3-class co neutral, du lieu weak-label va review mixed opinion. Neutral la class kho nhat. Cac benchmark sentiment lon cung cho thay neutral thuong lam giam hieu nang.

### Q2. Ket qua 79-80% co thap khong?

Khong qua thap. SemEval-2013 message-level sentiment co best F1 khoang 69%, SentiBench bao cao 3-class macro F1 cua nhieu tool chi quanh 0.6. Sau khi clean label, model 3-class cua nhom dat 0.877 macro F1.

### Q3. Tai sao binary dat cao hon 3-class?

Vi binary bo nhung mau neutral/mixed kho phan loai. Day la so sanh khong hoan toan cong bang ve muc do kho. Binary phu hop cho polarity alert, con 3-class phu hop hon cho phan tich business.

### Q4. Tai sao khong trinh bay cac huong thu nghiem kem hieu qua?

Vi slide can de nguoi nghe nam y chinh nhanh. Cac huong khong tot hon cleaned 3-class se duoc luu trong qua trinh thu nghiem, nhung khong dua vao phan ket qua chinh. Ket luan quan trong hon la: lam sach label giup cai thien ro hon viec tang do phuc tap model.

### Q5. Tai sao khong dung PhoBERT?

PhoBERT co the la huong cai thien tiep theo. Tuy nhien ket qua hien tai cho thay label noise moi la nut that lon. Neu fine-tune PhoBERT tren label nhieu nhieu, model van co the hoc sai. Do do nhom xu ly label quality truoc.

### Q6. Tai sao dung TF-IDF Word+Char ma lai tot?

Review tieng Viet co nhieu tu lap lai, keyword sentiment, loi viet va bien the tu. Character n-gram giup bat loi chinh ta va cum tu con. TF-IDF Word+Char cung nhe, de train, de deploy, va ket qua tot trong dataset nay.

### Q7. ABSA rule-based co yeu khong?

Rule-based ABSA khong manh bang model deep learning ABSA, nhung phu hop voi project vi:

- De giai thich cho business user.
- Co the kiem soat aspect taxonomy.
- Du nhanh de chay tren dashboard.
- Phu hop khi chua co dataset aspect-level human-labeled.

### Q8. Website co y nghia gi?

Website giup nguoi dung khong ky thuat truy cap ket qua:

- Xem sentiment summary.
- Xem pain point theo aspect.
- Loc theo company, industry, date.
- Theo doi trend va hotspot.

Day la bang chung rang mo hinh co kha nang ung dung trong HR analytics.

## 5. Cach ket luan khi thuyet trinh

Nen ket luan theo 3 y:

1. Ve ky thuat: label quality la nut that quan trong nhat; Cleanlab giup 3-class F1 tang manh.
2. Ve nghiep vu: Salary & Benefits va Work Environment la hai pain point quan trong nhat.
3. Ve ung dung: dashboard bien model thanh cong cu employee review intelligence cho HR va management.
