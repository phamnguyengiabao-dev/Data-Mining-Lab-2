# Benchmark Datasets

Thu muc nay chua cac tap du lieu benchmark cho phan thuc nghiem Apriori.

## Da bo sung san

| File | Dataset | #Trans. | #Items | AvgLen | Nhan xet |
|------|---------|---------|--------|--------|----------|
| `chess.txt` | Chess | 3,196 | 75 | 37.00 | Dense, itemset dai |
| `mushroom.txt` | Mushroom | 8,416 | 119 | 23.00 | Dense, nhieu item |
| `retail.txt` | Retail | 88,162 | 16,470 | 10.31 | Sparse, du lieu thuc te |
| `T10I4D100K.txt` | T10I4D100K | 100,000 | 870 | 10.10 | Synthetic, sparse |

## Tuy chon bo sung

- `accidents.txt`: 340,183 giao dich, 468 items, AvgLen 33.8.
- Co the tai them bang script `python download_benchmarks.py`.

## Nguon

- SPMF public datasets: https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php
- FIMI repository: http://fimi.uantwerpen.be/data/

## Ghi chu

- File `mushroom.txt` duoc tai tu `mushrooms.txt` tren SPMF va doi ten de dong bo voi code benchmark trong repo.
- Tat ca file deu o dinh dang giao dich SPMF: moi dong la mot transaction, cac item cach nhau boi dau cach.
