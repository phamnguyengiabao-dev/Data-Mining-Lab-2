# Đồ Án 2: Khai Thác Tập Phổ Biến — Thuật Toán Apriori

**Môn học:** Khai thác dữ liệu và ứng dụng (CSC14004)  
**Học kỳ:** HK2 – 2025/2026  
**Trường:** Đại học Khoa học Tự nhiên – ĐHQG TP.HCM  

---

## 📋 Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Kế hoạch thực hiện](#kế-hoạch-thực-hiện)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Hướng dẫn cài đặt môi trường](#hướng-dẫn-cài-đặt-môi-trường)
5. [Cách chạy](#cách-chạy)
6. [Ví dụ sử dụng](#ví-dụ-sử-dụng)
7. [Chạy kiểm thử](#chạy-kiểm-thử)
8. [Dữ liệu Benchmark](#dữ-liệu-benchmark)

---

## Giới thiệu

Đồ án cài đặt **thuật toán Apriori** cho bài toán **Frequent Itemset Mining (FIM)** hoàn toàn from scratch bằng Python. Thuật toán Apriori sử dụng chiến lược **level-wise search** kết hợp **tính chất Apriori (Downward Closure)** để sinh candidate itemsets và tỉa bớt các itemset không phổ biến.

Kết quả được so sánh và kiểm chứng với thư viện tham chiếu **SPMF** (Java).

---

## Kế hoạch thực hiện

### Chương 1: Nền tảng lý thuyết (20%)

| Mục | Nội dung | Trạng thái |
|-----|----------|------------|
| 1.1 | Định nghĩa Transaction Database, Support, Frequent Itemset | 📝 |
| 1.2 | Định nghĩa Closed Itemset, Maximal Itemset và quan hệ | 📝 |
| 1.3 | Tính chất Apriori (Downward Closure) — phát biểu & chứng minh | 📝 |
| 1.4 | Phân tích thuật toán Apriori: ý tưởng, cấu trúc dữ liệu, pseudocode | 📝 |
| 1.5 | Phân tích độ phức tạp thời gian & không gian | 📝 |
| 1.6 | So sánh lịch sử Apriori với các thuật toán FIM khác | 📝 |

### Chương 2: Ví dụ minh họa tay (30%)

| Mục | Nội dung | Trạng thái |
|-----|----------|------------|
| 2.1 | Ví dụ cơ sở: 5-7 giao dịch, 5-6 item, step-by-step, cross-check | 📝 |
| 2.2 | Ví dụ tình huống đặc biệt (ví dụ: tất cả giao dịch giống nhau, minsup=1,...) | 📝 |

### Chương 3: Cài đặt (25-30%)

| Mục | Nội dung | Trạng thái |
|-----|----------|------------|
| 3.1 | Cài đặt cơ bản thuật toán Apriori (from scratch) | ✅ |
| 3.2 | Xuất frequent itemsets + support khớp SPMF | ✅ |
| 3.3 | Unit tests trên ≥5 CSDL (kể cả toy dataset) | ✅ |
| 3.4 | Tối ưu hóa: bitset/bitarray cho candidate counting | ✅ |
| 3.5 | I/O định dạng SPMF, tham số dòng lệnh (minsup, file path) | ✅ |

### Chương 4: Thực nghiệm và đánh giá (20%)

| Mục | Nội dung | Trạng thái |
|-----|----------|------------|
| 4.1 | Kiểm tra correctness trên 4+ benchmark datasets | 📝 |
| 4.2 | Đồ thị thời gian chạy theo minsup (5-7 điểm) | 📝 |
| 4.3 | Đồ thị số lượng frequent itemsets theo minsup | 📝 |
| 4.4 | Đo RAM tối đa (peak memory) | 📝 |
| 4.5 | Scalability test (10%, 25%, 50%, 75%, 100%) | 📝 |
| 4.6 | Ảnh hưởng của độ dài giao dịch trung bình | 📝 |

### Chương 5: Ứng dụng thực tế (Optional - 10%)

| Mục | Nội dung | Trạng thái |
|-----|----------|------------|
| 5.1 | Phân tích giỏ hàng / Phát hiện mẫu log / Sinh học | ❌ Chưa quyết định |

> **Ghi chú:** ✅ = Đã hoàn thành | 📝 = Đang thực hiện / Chưa bắt đầu | ❌ = Chưa quyết định

---

## Cấu trúc thư mục

```
Group_ID/
├── README.md                  # File hướng dẫn
├── requirements.txt           # Dependencies
├── src/
│   ├── __init__.py
│   ├── algorithm/
│   │   ├── __init__.py
│   │   └── apriori.py         # Cài đặt thuật toán Apriori
│   ├── structures.py          # Cấu trúc dữ liệu (HashTree cho candidate)
│   └── utils.py               # Tiện ích đọc/ghi file SPMF
├── tests/
│   ├── test_correctness.py    # Unit tests kiểm tra tính đúng đắn
│   └── test_benchmark.py      # Benchmark tests (thời gian, bộ nhớ)
├── data/
│   ├── toy/                   # CSDL nhỏ cho ví dụ tay
│   │   ├── toy1.txt
│   │   └── toy2_special.txt
│   └── benchmark/             # CSDL benchmark (Chess, Mushroom, Retail,...)
├── notebooks/
│   └── demo.ipynb             # Notebook minh họa
└── docs/
    └── Report.pdf             # Báo cáo PDF
```

---

## Hướng dẫn cài đặt môi trường

### Yêu cầu
- Python ≥ 3.9

### Cài đặt

```bash
# Clone hoặc giải nén dự án
cd Group_ID/

# Cài đặt dependencies
pip install -r requirements.txt
```

---

## Cách chạy

### Chạy thuật toán Apriori từ dòng lệnh

```bash
# Cú pháp
python -m src.algorithm.apriori --input <file_path> --minsup <value> [--output <file_path>]

# Ví dụ: chạy trên toy dataset với minsup = 0.4 (40%)
python -m src.algorithm.apriori --input data/toy/toy1.txt --minsup 0.4

# Ví dụ: chạy trên benchmark dataset, xuất ra file
python -m src.algorithm.apriori --input data/benchmark/mushroom.txt --minsup 0.3 --output output.txt

# Minsup tuyệt đối (absolute support count)
python -m src.algorithm.apriori --input data/toy/toy1.txt --minsup 2 --absolute
```

### Tham số dòng lệnh

| Tham số | Mô tả | Mặc định |
|---------|--------|----------|
| `--input` | Đường dẫn file dữ liệu đầu vào (SPMF format) | **Bắt buộc** |
| `--minsup` | Ngưỡng minimum support | **Bắt buộc** |
| `--absolute` | Nếu có, minsup là absolute count; ngược lại là relative (0–1) | `False` |
| `--output` | Đường dẫn file xuất kết quả | `None` (in ra stdout) |

---

## Ví dụ sử dụng

### Trong code Python

```python
from src.utils import load_transactions_spmf
from src.algorithm.apriori import apriori

# Đọc dữ liệu
transactions = load_transactions_spmf("data/toy/toy1.txt")

# Chạy Apriori với minsup tương đối = 0.4
frequent_itemsets = apriori(transactions, min_support=0.4)

# In kết quả
for itemset, support in sorted(frequent_itemsets.items()):
    print(f"{itemset} : {support}")
```

---

## Chạy kiểm thử

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy riêng test correctness
pytest tests/test_correctness.py -v

# Chạy riêng test benchmark
pytest tests/test_benchmark.py -v

# Chỉ chạy benchmark trên 4 dataset đã chuẩn bị
pytest tests/test_benchmark.py -v -s
```

---

## Dữ liệu Benchmark

| Tập dữ liệu   | #Trans.  | #Items | AvgLen | Đặc điểm              |
|----------------|----------|--------|--------|------------------------|
| Chess          | 3,196    | 75     | 37.0   | Dày đặc (dense)        |
| Mushroom       | 8,416    | 119    | 23.0   | Dày đặc, nhiều item    |
| Retail         | 88,162   | 16,470 | 10.3   | Thưa (sparse), thực tế |
| Accidents      | 340,183  | 468    | 33.8   | Rất lớn, dày đặc       |
| T10I4D100K     | 100,000  | 870    | 10.1   | Tổng hợp, thưa         |

Hiện repo đã kèm sẵn 4 benchmark để thực hiện mục này:
`chess.txt`, `mushroom.txt`, `retail.txt`, `T10I4D100K.txt` trong [data/benchmark/README.md](/D:/Coding/Data mining/lab2 repo/Data-Mining-Lab-2/data/benchmark/README.md).

Nếu cần tải lại hoặc bổ sung `accidents.txt`, chạy:

```bash
python download_benchmarks.py
```

Nguồn dữ liệu: [SPMF Datasets](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php) hoặc [FIMI Repository](http://fimi.uantwerpen.be/data/).

---

## Tham khảo

1. R. Agrawal and R. Srikant, "Fast algorithms for mining association rules," *Proc. 20th Int. Conf. Very Large Data Bases (VLDB)*, 1994.
2. SPMF: An Open-Source Data Mining Library — [https://www.philippe-fournier-viger.com/spmf/](https://www.philippe-fournier-viger.com/spmf/)
