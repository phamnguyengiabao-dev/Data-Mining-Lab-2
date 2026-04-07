"""
apriori.py — Cài đặt thuật toán Apriori cho Frequent Itemset Mining.

Thuật toán Apriori (Agrawal & Srikant, 1994) sử dụng chiến lược
level-wise search (tìm kiếm theo mức) để khai thác tất cả frequent itemsets.

Ý tưởng cốt lõi:
  1. Tìm tất cả 1-itemsets phổ biến (L1).
  2. Với mỗi mức k ≥ 2:
     a) Sinh candidate k-itemsets (Ck) từ Lk-1 bằng phép join + pruning.
     b) Quét CSDL để đếm support của mỗi candidate trong Ck.
     c) Giữ lại những candidate có support ≥ minsup → Lk.
  3. Lặp lại cho đến khi Lk = ∅.

Tính chất Apriori (Downward Closure):
  Nếu một itemset không phổ biến, tất cả superset của nó cũng không phổ biến.
  → Dùng để tỉa (prune) candidate, giảm đáng kể không gian tìm kiếm.

Tối ưu hóa đã áp dụng:
  - Sử dụng bitarray (vertical layout) để tính support qua phép AND bit.
  - Tỉa candidate bằng tính chất Apriori trước khi đếm support.
  - Sắp xếp items trong candidate để join hiệu quả.

Tham khảo:
  R. Agrawal and R. Srikant, "Fast algorithms for mining association rules,"
  Proc. 20th Int. Conf. Very Large Data Bases (VLDB), 1994.
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.structures import TransactionDB
from src.utils import load_transactions_spmf, save_results_spmf, print_results


# Hàm sinh candidate (Candidate Generation)

def apriori_gen(
    prev_frequent: List[frozenset],
    k: int,
) -> Set[frozenset]:
    """
    Sinh candidate k-itemsets từ (k-1)-itemsets phổ biến.

    Bước 1 — Join:
      Hai (k-1)-itemsets p, q có thể join nếu chúng chia sẻ (k-2) items đầu tiên
      (khi đã sắp xếp) và khác nhau ở item cuối cùng.

    Bước 2 — Prune:
      Loại bỏ candidate c nếu bất kỳ (k-1)-subset nào của c không thuộc Lk-1.
      (Áp dụng tính chất Apriori / Downward Closure)

    Parameters
    prev_frequent : list[frozenset[int]]
        Danh sách (k-1)-itemsets phổ biến (Lk-1).
    k : int
        Kích thước của candidate cần sinh.

    Returns
    set[frozenset[int]]
        Tập candidate k-itemsets (Ck).
    """
    prev_set = set(prev_frequent)  # Để lookup nhanh O(1) khi prune
    candidates: Set[frozenset] = set()

    # Chuyển mỗi itemset sang dạng tuple đã sort để join hiệu quả
    sorted_itemsets = [tuple(sorted(fs)) for fs in prev_frequent]
    sorted_itemsets.sort()

    n = len(sorted_itemsets)

    for i in range(n):
        for j in range(i + 1, n):
            p = sorted_itemsets[i]
            q = sorted_itemsets[j]

            # Điều kiện join: (k-2) phần tử đầu giống nhau
            if p[: k - 2] == q[: k - 2]:
                # Tạo candidate bằng union
                candidate = frozenset(p) | frozenset(q)

                # Prune: kiểm tra tất cả (k-1)-subsets phải thuộc Lk-1
                if _all_subsets_frequent(candidate, k, prev_set):
                    candidates.add(candidate)
            else:
                # Vì đã sort, nếu prefix khác thì các phần tử tiếp theo cũng khác
                break

    return candidates


def _all_subsets_frequent(
    candidate: frozenset,
    k: int,
    prev_frequent_set: Set[frozenset],
) -> bool:
    """
    Kiểm tra tất cả (k-1)-subsets của candidate có thuộc Lk-1 không.

    Đây là bước prune dựa trên tính chất Apriori:
    Nếu bất kỳ subset nào không frequent → candidate chắc chắn không frequent.

    Parameters
    candidate : frozenset[int]
        Candidate k-itemset cần kiểm tra.
    k : int
        Kích thước candidate.
    prev_frequent_set : set[frozenset[int]]
        Tập Lk-1.

    Returns
    bool
        True nếu tất cả (k-1)-subsets đều frequent.
    """
    items = sorted(candidate)
    for combo in combinations(items, k - 1):
        subset = frozenset(combo)
        if subset not in prev_frequent_set:
            return False
    return True


# Thuật toán Apriori chính

def apriori(
    transactions: List[frozenset],
    min_support: float,
    absolute: bool = False,
    use_bitarray: bool = True,
    verbose: bool = False,
) -> Dict[frozenset, int]:
    """
    Thuật toán Apriori — Khai thác tất cả frequent itemsets.

    Pseudocode (theo Agrawal & Srikant, 1994):
    Input:  D = CSDL giao dịch, minsup = ngưỡng support tối thiểu
    Output: L = tập tất cả frequent itemsets

    1.  L1 = {frequent 1-itemsets}
    2.  k = 2
    3.  WHILE Lk-1 ≠ ∅:
    4.      Ck = apriori_gen(Lk-1)        // Sinh candidate
    5.      FOR EACH transaction t ∈ D:
    6.          Ct = subset(Ck, t)         // Tìm candidate chứa trong t
    7.          FOR EACH c ∈ Ct:
    8.              c.count += 1
    9.      Lk = {c ∈ Ck | c.count ≥ minsup}
    10.     k = k + 1
    11. RETURN L = L1 ∪ L2 ∪ ... ∪ Lk-1

    Parameters
    transactions : list[frozenset[int]]
        Danh sách các giao dịch.
    min_support : float
        Ngưỡng support. Nếu absolute=False thì đây là tỉ lệ (0–1),
        nếu absolute=True thì đây là absolute count.
    absolute : bool
        True nếu min_support là absolute count.
    use_bitarray : bool
        True để sử dụng bitarray tối ưu hóa.
    verbose : bool
        True để in thông tin debug từng mức.

    Returns
    dict[frozenset[int], int]
        Tất cả frequent itemsets cùng absolute support count.
    """
    n_trans = len(transactions)
    if n_trans == 0:
        return {}

    # Tính ngưỡng support tuyệt đối
    if absolute:
        min_sup_count = int(min_support)
    else:
        min_sup_count = max(1, int(min_support * n_trans))

    if verbose:
        print(f"[Apriori] Total transactions: {n_trans}")
        print(f"[Apriori] Min support count: {min_sup_count}")
        print(f"[Apriori] Use bitarray: {use_bitarray}")

    # Xây dựng TransactionDB
    db = TransactionDB(transactions, use_bitarray=use_bitarray)

    # Kết quả: tất cả frequent itemsets
    all_frequent: Dict[frozenset, int] = {}

    # ---- Bước 1: Tìm L1 (frequent 1-itemsets) ----
    L_prev = db.get_frequent_1_itemsets(min_sup_count)
    all_frequent.update(L_prev)

    if verbose:
        print(f"[Apriori] L1: {len(L_prev)} frequent 1-itemsets")

    k = 2

    # ---- Bước 2: Lặp level-wise ----
    while L_prev:
        # Sinh candidate Ck
        candidates = apriori_gen(list(L_prev.keys()), k)

        if verbose:
            print(f"[Apriori] C{k}: {len(candidates)} candidates")

        if not candidates:
            break

        # Đếm support cho mỗi candidate
        L_current: Dict[frozenset, int] = {}
        for candidate in candidates:
            sup = db.get_support(candidate)
            if sup >= min_sup_count:
                L_current[candidate] = sup

        if verbose:
            print(f"[Apriori] L{k}: {len(L_current)} frequent {k}-itemsets")

        all_frequent.update(L_current)
        L_prev = L_current
        k += 1

    return all_frequent


# Phiên bản KHÔNG dùng bitarray (baseline) — dùng để so sánh hiệu năng

def apriori_baseline(
    transactions: List[frozenset],
    min_support: float,
    absolute: bool = False,
    verbose: bool = False,
) -> Dict[frozenset, int]:
    """
    Phiên bản Apriori cơ bản KHÔNG sử dụng bitarray.

    Dùng phép duyệt tuần tự (horizontal scanning) để đếm support.
    Sử dụng làm baseline để so sánh hiệu năng với bản tối ưu.

    Parameters và Returns giống hàm apriori().
    """
    return apriori(
        transactions,
        min_support,
        absolute=absolute,
        use_bitarray=False,
        verbose=verbose,
    )


# CLI — Chạy từ dòng lệnh

def main():
    """Entry point khi chạy từ dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Apriori Frequent Itemset Mining (from scratch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python -m src.algorithm.apriori --input data/toy/toy1.txt --minsup 0.4
  python -m src.algorithm.apriori --input data/toy/toy1.txt --minsup 2 --absolute
  python -m src.algorithm.apriori --input data/benchmark/mushroom.txt --minsup 0.3 --output result.txt
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Đường dẫn file dữ liệu đầu vào (SPMF format).",
    )
    parser.add_argument(
        "--minsup", "-s",
        required=True,
        type=float,
        help="Ngưỡng minimum support (relative 0-1 hoặc absolute nếu có --absolute).",
    )
    parser.add_argument(
        "--absolute", "-a",
        action="store_true",
        help="Nếu có, minsup là absolute count (số nguyên).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Đường dẫn file xuất kết quả (SPMF format). Mặc định: in ra stdout.",
    )
    parser.add_argument(
        "--no-bitarray",
        action="store_true",
        help="Không sử dụng bitarray tối ưu (baseline mode).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="In thông tin chi tiết từng mức.",
    )

    args = parser.parse_args()

    # Đọc dữ liệu
    print(f"Loading data from: {args.input}")
    transactions = load_transactions_spmf(args.input)
    print(f"Transactions: {len(transactions)}")

    # Chạy Apriori
    start_time = time.perf_counter()
    result = apriori(
        transactions,
        min_support=args.minsup,
        absolute=args.absolute,
        use_bitarray=not args.no_bitarray,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - start_time

    print(f"\nRuntime: {elapsed:.4f}s")
    print(f"Total frequent itemsets: {len(result)}")

    # Xuất kết quả
    if args.output:
        save_results_spmf(args.output, result)
        print(f"Results saved to: {args.output}")
    else:
        print_results(result, len(transactions))


if __name__ == "__main__":
    main()
