"""
structures.py — Cấu trúc dữ liệu hỗ trợ cho thuật toán Apriori.

Bao gồm:
  - TransactionDB: Biểu diễn CSDL giao dịch dạng ngang (horizontal) và dọc (vertical/bitset).
  - HashTree: Cấu trúc hash-tree để đếm candidate itemsets hiệu quả (tùy chọn).

Tối ưu hóa:
  - Sử dụng bitarray để biểu diễn tidset (vertical layout) cho việc
    tính support bằng phép AND bit thay vì duyệt tuần tự.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set

import numpy as np

try:
    from bitarray import bitarray
    HAS_BITARRAY = True
except ImportError:
    HAS_BITARRAY = False


class TransactionDB:
    """
    Cơ sở dữ liệu giao dịch hỗ trợ cả biểu diễn ngang (horizontal) và dọc (vertical/bitset).

    Biểu diễn ngang: Danh sách các giao dịch, mỗi giao dịch là frozenset các items.
    Biểu diễn dọc (tidset/bitset): Mỗi item ánh xạ tới một bitarray, bit thứ i = 1
      nếu item xuất hiện trong giao dịch thứ i.

    Attributes
    transactions : list[frozenset[int]]
        Danh sách các giao dịch.
    n_transactions : int
        Số lượng giao dịch.
    items : list[int]
        Danh sách tất cả item (sorted).
    item_tidsets : dict[int, bitarray] | None
        Biểu diễn dọc dạng bitarray (nếu use_bitarray=True).
    """

    def __init__(
        self,
        transactions: List[frozenset],
        use_bitarray: bool = True,
    ):
        self.transactions = transactions
        self.n_transactions = len(transactions)

        # Lấy tất cả items
        all_items: Set[int] = set()
        for t in transactions:
            all_items.update(t)
        self.items = sorted(all_items)

        # Xây dựng biểu diễn dọc (vertical/bitset)
        self.use_bitarray = use_bitarray and HAS_BITARRAY
        self.item_tidsets: Optional[Dict[int, bitarray]] = None

        if self.use_bitarray:
            self._build_bitset_representation()

    def _build_bitset_representation(self) -> None:
        """Xây dựng biểu diễn dọc dạng bitarray cho mỗi item."""
        self.item_tidsets = {}
        for item in self.items:
            ba = bitarray(self.n_transactions)
            ba.setall(0)
            for tid, transaction in enumerate(self.transactions):
                if item in transaction:
                    ba[tid] = 1
            self.item_tidsets[item] = ba

    def get_support_bitset(self, itemset: frozenset) -> int:
        """
        Tính support của một itemset bằng phép AND trên bitarray.

        Độ phức tạp: O(n_transactions / word_size) cho mỗi phép AND,
        nhanh hơn đáng kể so với duyệt tuần tự O(n_transactions).

        Parameters
        itemset : frozenset[int]
            Tập items cần tính support.

        Returns
        int
            Absolute support count.
        """
        if not self.use_bitarray or self.item_tidsets is None:
            raise RuntimeError("Bitarray representation chưa được xây dựng.")

        items_list = sorted(itemset)
        # Bắt đầu với tidset của item đầu tiên
        result = bitarray(self.item_tidsets[items_list[0]])
        # AND lần lượt với các item còn lại
        for item in items_list[1:]:
            result &= self.item_tidsets[item]
        return result.count()

    def get_support_horizontal(self, itemset: frozenset) -> int:
        """
        Tính support bằng duyệt tuần tự trên biểu diễn ngang.

        Đây là phương pháp đơn giản, dùng khi không có bitarray.

        Parameters
        itemset : frozenset[int]
            Tập items cần tính support.

        Returns
        int
            Absolute support count.
        """
        count = 0
        for transaction in self.transactions:
            if itemset.issubset(transaction):
                count += 1
        return count

    def get_support(self, itemset: frozenset) -> int:
        """
        Tính support của một itemset, tự động chọn phương pháp tối ưu.

        Nếu bitarray khả dụng → dùng bitset counting.
        Ngược lại → dùng horizontal scanning.
        """
        if self.use_bitarray and self.item_tidsets is not None:
            return self.get_support_bitset(itemset)
        return self.get_support_horizontal(itemset)

    def get_frequent_1_itemsets(self, min_support_count: int) -> Dict[frozenset, int]:
        """
        Tìm tất cả 1-itemsets phổ biến.

        Parameters
        min_support_count : int
            Ngưỡng support tuyệt đối tối thiểu.

        Returns
        dict[frozenset[int], int]
            Mapping từ 1-itemset -> absolute support count.
        """
        freq_1: Dict[frozenset, int] = {}
        for item in self.items:
            sup = self.get_support(frozenset([item]))
            if sup >= min_support_count:
                freq_1[frozenset([item])] = sup
        return freq_1

    def __len__(self) -> int:
        return self.n_transactions

    def __repr__(self) -> str:
        return (
            f"TransactionDB(n_transactions={self.n_transactions}, "
            f"n_items={len(self.items)}, "
            f"use_bitarray={self.use_bitarray})"
        )
