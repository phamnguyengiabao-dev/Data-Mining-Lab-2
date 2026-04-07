"""
utils.py — Tiện ích đọc/ghi dữ liệu giao dịch theo định dạng SPMF.

Định dạng SPMF:
  - Mỗi dòng là một giao dịch.
  - Các item là số nguyên, cách nhau bởi dấu cách.
  - Ví dụ:
      1 3 4
      2 3 5
      1 2 3 4 5

Hàm chính:
  - load_transactions_spmf(filepath)  -> list[frozenset[int]]
  - save_results_spmf(filepath, results)
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple


def load_transactions_spmf(filepath: str) -> List[frozenset]:
    """
    Đọc file giao dịch theo định dạng SPMF.

    Parameters
    filepath : str
        Đường dẫn đến file dữ liệu.

    Returns
    list[frozenset[int]]
        Danh sách các giao dịch, mỗi giao dịch là frozenset các item (int).
    """
    transactions: List[frozenset] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            items = frozenset(int(x) for x in line.split())
            if items:
                transactions.append(items)
    return transactions


def save_results_spmf(
    filepath: str,
    frequent_itemsets: Dict[frozenset, int],
) -> None:
    """
    Ghi kết quả frequent itemsets ra file theo định dạng SPMF.

    Định dạng đầu ra (SPMF output format):
        item1 item2 ... #SUP: support_count

    Parameters
    filepath : str
        Đường dẫn file xuất kết quả.
    frequent_itemsets : dict[frozenset[int], int]
        Mapping từ itemset -> absolute support count.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for itemset, sup in sorted(
            frequent_itemsets.items(), key=lambda x: (len(x[0]), sorted(x[0]))
        ):
            items_str = " ".join(str(item) for item in sorted(itemset))
            f.write(f"{items_str} #SUP: {sup}\n")


def load_spmf_output(filepath: str) -> Dict[frozenset, int]:
    """
    Đọc file kết quả SPMF (output) để dùng cho so sánh.

    Định dạng:
        item1 item2 ... #SUP: support_count

    Returns
    dict[frozenset[int], int]
        Mapping từ itemset -> absolute support count.
    """
    results: Dict[frozenset, int] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "#SUP:" in line:
                parts = line.split("#SUP:")
                items_part = parts[0].strip()
                sup_part = parts[1].strip()
                items = frozenset(int(x) for x in items_part.split())
                support = int(sup_part)
                results[items] = support
            else:
                # Fallback: chỉ có items, không có support
                items = frozenset(int(x) for x in line.split())
                results[items] = -1
    return results


def get_all_items(transactions: List[frozenset]) -> set:
    """Lấy tập hợp tất cả các item xuất hiện trong CSDL."""
    all_items = set()
    for t in transactions:
        all_items.update(t)
    return all_items


def print_results(frequent_itemsets: Dict[frozenset, int], n_transactions: int) -> None:
    """In kết quả ra stdout theo định dạng đẹp."""
    print("")
    print(f"  FREQUENT ITEMSET MINING RESULTS")
    print(f"  Total frequent itemsets: {len(frequent_itemsets)}")
    print(f"  Total transactions: {n_transactions}")
    print("")

    # Nhóm theo kích thước itemset
    by_size: Dict[int, List[Tuple[frozenset, int]]] = {}
    for itemset, sup in frequent_itemsets.items():
        k = len(itemset)
        if k not in by_size:
            by_size[k] = []
        by_size[k].append((itemset, sup))

    for k in sorted(by_size.keys()):
        items_list = sorted(by_size[k], key=lambda x: sorted(x[0]))
        print(f"\n--- {k}-itemsets ({len(items_list)} sets) ---")
        for itemset, sup in items_list:
            items_str = "{" + ", ".join(str(i) for i in sorted(itemset)) + "}"
            rel_sup = sup / n_transactions
            print(f"  {items_str:30s}  sup = {sup} ({rel_sup:.2%})")

