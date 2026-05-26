# Phân Cụm Thị Trường Việc Làm Việt Nam

Đồ án phân cụm (unsupervised clustering) tin tuyển dụng tiếng Việt từ ~600k dòng dữ liệu
---

## 📁 Cấu trúc thư mục

```
CUOIKI_KHDL/
├── data/
│   ├── raw/              # data.parquet (input gốc, ~600k jobs × 14 cột)
│   └── clean/            # output sau 02 & 03: clean_data_*.csv, train/test_processed.parquet, X_text_pca_*.npy
├── models/               # output từ 03 & training: pca_text.joblib, meta_scaler.joblib, kmeans.joblib, best_params.joblib
├── notebook/
│   ├── 01_split_and_export.ipynb       # split 90/10 stratified by year
│   ├── 02_1_clean_data.ipynb             # parse salary, clean text, dedup, stopwords
│   ├── 02_2_clean_data_overview.ipynb    # overview data after cleaning
│   ├── 03_feature_engineering.ipynb    # categorical mapping, embedding, PCA
│   ├── training.ipynb                  # Optuna tune + train KMeans K=9
│   └── testing.ipynb                   # inference-only pipeline trên test set
├── README.md
└── .gitignore
```

---

## 🛠️ Cài đặt

### Yêu cầu

- **Python 3.10+** (đã test trên 3.11, 3.12)
- ~8 GB RAM (cho bước embedding 384d trên ~450k dòng)
- ~3 GB dung lượng đĩa (cho data + models intermediate)

### Bước 1: Tạo virtual environment

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### Bước 2: Cài thư viện

```bash
pip install -U pip
pip install pandas numpy scikit-learn matplotlib seaborn ^
            sentence-transformers optuna joblib ^
            umap-learn pyvi pyarrow ipykernel
```

Hoặc dùng file requirements:

```bash
pip install -r requirements.txt
```

`requirements.txt` (tạo nếu chưa có):

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
sentence-transformers>=2.5
optuna>=3.4
joblib>=1.3
umap-learn>=0.5
pyvi>=0.1.1
pyarrow>=14.0
ipykernel>=6.25
```

### Bước 3: Tải data gốc

Đặt file `data.parquet` vào folder `data/raw/`.

---

## 🚀 Cách chạy

Chạy **tuần tự** các notebook trong thư mục `notebook/`:

| # | Notebook | Input | Output | Thời gian |
|---|----------|-------|--------|-----------|
| 1 | `01_split_and_export.ipynb` | `data/raw/data.parquet` | `data/raw/raw_data_{train,test}.{csv,parquet}` | ~40s |
| 2 | `02_1_clean_data.ipynb` | `raw_data_*.parquet` | `data/clean/clean_data_{train,test}.{csv,parquet}` | ~5 phút |
| 2 | `02_2_clean_data_overview.ipynb` | `data/clean/clean_data_{train,test}.{csv,parquet}` | các biểu đồ | ~5 phút |
| 3 | `03_feature_engineering.ipynb` | `clean_data_*.csv` | `data/clean/{train,test}_processed.parquet`, `X_text_pca_*.npy`, `models/pca_text.joblib` | **~30 phút** (embedding) |
| 4 | `training.ipynb` | output của 03 | `models/{meta_scaler,kmeans,best_params}.joblib` + cluster labels cho train | ~15 phút (Optuna + final fit) |
| 5 | `testing.ipynb` | model `.joblib` + `*_processed.parquet` + `X_text_pca_*.npy` | `data/clean/Test_with_cluster.csv` + plots | ~3 phút |

### Skip-mode

Sau khi đã chạy lần đầu, có thể skip để chỉ làm phần inference:
- Chỉ cần chạy `testing.ipynb` (standalone) nếu đã có sẵn `models/*.joblib` và `data/clean/*_processed.parquet` + `X_text_pca_*.npy`.

---

## 📊 Kết quả

**Pipeline**: Embedding 384d → PCA 60d → L2-normalize → hstack với meta (4d × trọng số) → L2-normalize → KMeans K=9

**Best params từ Optuna** (40 trials, silhouette score):
- `text_weight = 0.1`, `meta_weights = [1.5, 1.0, 0.1, 0.5]` cho [salary_log, experience, job_position, education]
- Composite score ≈ 0.47 (silhouette − imbalance penalty)

**7-9 phân khúc** đặc trưng bởi tier kinh nghiệm (Junior / Mid / Senior) và mảng ngành (front-office / backoffice / kỹ thuật / quản lý).

---

## 🔧 Troubleshooting

**`ValueError: Unsupported input type: float` khi embedding** → text cột có NaN. Đã được fix bằng `fillna('').astype(str)` trước khi `encode()`.

**`FileNotFoundError: data/clean/clean_data_train.csv`** → chưa chạy `02_clean_data.ipynb`. Chạy theo đúng thứ tự 01 → 02 → 03 → training → testing.

**Embedding chậm** → mặc định chạy trên CPU. Nếu có GPU, sentence-transformers tự dùng CUDA — kiểm tra `torch.cuda.is_available()`.

**Hết RAM khi load X_text_train.npy** → giảm `SAMPLE_SIZE` trong tuning (mặc định 50,000) hoặc dùng MiniBatchKMeans thay KMeans cho bước final.

---

## 📝 Notes

- File `clean_data_*.csv` rất lớn (~1 GB train). Parquet tương đương chỉ ~500 MB và đọc nhanh hơn 5-10x — pipeline ưu tiên đọc parquet, CSV chỉ để nộp bài.
- Mô hình KMeans được lưu cùng `best_params.joblib` → reproducible: load lại biết model đã train với tham số nào.
- t-SNE và UMAP plot cùng dữ liệu nhưng dùng 2 thuật toán giảm chiều khác — t-SNE để xác nhận cụm tồn tại, UMAP để hiểu quan hệ giữa các cụm.
