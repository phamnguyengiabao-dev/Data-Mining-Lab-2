"""
test_correctness.py — Kiểm tra tính đúng đắn của thuật toán Apriori.

Bao gồm:
  - Test trên 5 toy datasets với kết quả tính tay.
  - Test so khớp giữa bản bitarray và bản baseline.
  - Test edge cases (dataset rỗng, minsup rất cao/thấp, ...).
"""
import os
import sys
import pytest

# Thêm project root vào path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.algorithm.apriori import apriori, apriori_baseline
from src.utils import load_transactions_spmf


# Đường dẫn dữ liệu
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "toy")


def _data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


# Test 1: toy1.txt — Ví dụ cơ bản (kiểm tra chéo bằng tay)
class TestToy1:
    """
    Dữ liệu toy1.txt:
        T1: {1, 3, 4}
        T2: {2, 3, 5}
        T3: {1, 2, 3, 5}
        T4: {2, 5}
        T5: {1, 2, 3, 4, 5}

    5 giao dịch, 5 items. minsup = 2 (40%).

    Đếm support bằng tay:
      {1}: 3, {2}: 4, {3}: 4, {4}: 2, {5}: 4
      {1,2}: 2, {1,3}: 3, {1,4}: 2, {1,5}: 2,
      {2,3}: 3, {2,5}: 4, {3,4}: 2, {3,5}: 3,
      {1,2,3}: 2, {1,2,5}: 2, {1,3,4}: 2, {1,3,5}: 2,
      {2,3,5}: 3,
      {1,2,3,5}: 2
    """

    @pytest.fixture
    def transactions(self):
        return load_transactions_spmf(_data_path("toy1.txt"))

    def test_load_data(self, transactions):
        """Kiểm tra đọc dữ liệu đúng."""
        assert len(transactions) == 5
        assert frozenset([1, 3, 4]) in transactions
        assert frozenset([2, 5]) in transactions

    def test_frequent_1_itemsets(self, transactions):
        """Kiểm tra 1-itemsets phổ biến với minsup=2."""
        result = apriori(transactions, min_support=2, absolute=True)
        # Tất cả 5 items đều có support >= 2
        for item in [1, 2, 3, 4, 5]:
            assert frozenset([item]) in result

    def test_specific_supports(self, transactions):
        """Kiểm tra support cụ thể của các itemsets đã tính tay."""
        result = apriori(transactions, min_support=2, absolute=True)

        # 1-itemsets
        assert result[frozenset([1])] == 3
        assert result[frozenset([2])] == 4
        assert result[frozenset([3])] == 4
        assert result[frozenset([4])] == 2
        assert result[frozenset([5])] == 4

        # 2-itemsets
        assert result[frozenset([2, 5])] == 4
        assert result[frozenset([1, 3])] == 3
        assert result[frozenset([2, 3])] == 3
        assert result[frozenset([3, 5])] == 3

        # 3-itemsets
        assert result[frozenset([2, 3, 5])] == 3
        assert result[frozenset([1, 2, 3])] == 2

        # 4-itemset
        assert result[frozenset([1, 2, 3, 5])] == 2

    def test_total_count_minsup2(self, transactions):
        """Tổng số frequent itemsets phải đúng."""
        result = apriori(transactions, min_support=2, absolute=True)
        # Đếm tay:
        #   1-itemsets: 5 ({1},{2},{3},{4},{5})
        #   2-itemsets: {1,2}:2, {1,3}:3, {1,4}:2, {1,5}:2, {2,3}:3,
        #               {2,5}:4, {3,4}:2, {3,5}:3 = 8
        #   3-itemsets: {1,2,3}:2, {1,2,5}:2, {1,3,4}:2, {1,3,5}:2,
        #               {2,3,5}:3 = 5
        #   4-itemsets: {1,2,3,5}:2 = 1
        # Tổng: 5 + 8 + 5 + 1 = 19
        assert len(result) == 19

    def test_relative_support(self, transactions):
        """Kiểm tra minsup tương đối (40% = 2/5)."""
        result = apriori(transactions, min_support=0.4)
        result_abs = apriori(transactions, min_support=2, absolute=True)
        assert result == result_abs

    def test_high_minsup(self, transactions):
        """Với minsup cao (80% = 4/5), chỉ có ít itemsets."""
        result = apriori(transactions, min_support=0.8)
        # Items có support >= 4: {2}:4, {3}:4, {5}:4, {2,5}:4
        assert len(result) == 4
        assert frozenset([2]) in result
        assert frozenset([3]) in result
        assert frozenset([5]) in result
        assert frozenset([2, 5]) in result


# Test 2: toy2_special.txt — Tình huống đặc biệt (tất cả giao dịch giống nhau)
class TestToy2Special:
    """
    Dữ liệu: 5 giao dịch, tất cả là {1, 2, 3}.
    Mọi itemset con đều có support = 5.
    """

    @pytest.fixture
    def transactions(self):
        return load_transactions_spmf(_data_path("toy2_special.txt"))

    def test_all_subsets_frequent(self, transactions):
        """Với minsup=1, tất cả subsets không rỗng đều frequent."""
        result = apriori(transactions, min_support=1, absolute=True)
        # 1-itemsets: {1}, {2}, {3} = 3
        # 2-itemsets: {1,2}, {1,3}, {2,3} = 3
        # 3-itemsets: {1,2,3} = 1
        # Tổng: 7
        assert len(result) == 7
        # Tất cả đều có support = 5
        for sup in result.values():
            assert sup == 5


# Test 3: toy3.txt — Ví dụ phức tạp hơn
class TestToy3:
    """
    Dữ liệu toy3.txt (7 giao dịch, 4 items):
        T1: {1, 2, 4}
        T2: {1, 3}
        T3: {1, 2, 3, 4}
        T4: {2, 3}
        T5: {1, 2, 3}
        T6: {1, 2, 3, 4}
        T7: {2, 4}

    Support (đếm tay):
      {1}: 5, {2}: 6, {3}: 5, {4}: 4
      {1,2}: 4, {1,3}: 4, {1,4}: 3, {2,3}: 4, {2,4}: 4, {3,4}: 2
      {1,2,3}: 3, {1,2,4}: 3, {1,3,4}: 2, {2,3,4}: 2
      {1,2,3,4}: 2
    """

    @pytest.fixture
    def transactions(self):
        return load_transactions_spmf(_data_path("toy3.txt"))

    def test_supports_minsup3(self, transactions):
        """Kiểm tra với minsup=3."""
        result = apriori(transactions, min_support=3, absolute=True)

        # 1-itemsets (sup >= 3): {1}:5, {2}:6, {3}:5, {4}:4
        assert result[frozenset([1])] == 5
        assert result[frozenset([2])] == 6
        assert result[frozenset([3])] == 5
        assert result[frozenset([4])] == 4

        # 2-itemsets (sup >= 3): {1,2}:4, {1,3}:4, {1,4}:3, {2,3}:4, {2,4}:4
        assert result[frozenset([1, 2])] == 4
        assert result[frozenset([1, 3])] == 4
        assert result[frozenset([1, 4])] == 3
        assert result[frozenset([2, 3])] == 4
        assert result[frozenset([2, 4])] == 4
        assert frozenset([3, 4]) not in result  # support = 2 < 3

        # 3-itemsets (sup >= 3): {1,2,3}:3, {1,2,4}:3
        assert result[frozenset([1, 2, 3])] == 3
        assert result[frozenset([1, 2, 4])] == 3

    def test_total_count_minsup3(self, transactions):
        """Tổng số: 4 + 5 + 2 = 11."""
        result = apriori(transactions, min_support=3, absolute=True)
        assert len(result) == 11


# Test 4: toy4_single.txt — Mỗi giao dịch chỉ 1 item
class TestToy4Single:
    """Mỗi giao dịch chỉ có 1 item duy nhất → không có 2-itemset nào."""

    @pytest.fixture
    def transactions(self):
        return load_transactions_spmf(_data_path("toy4_single.txt"))

    def test_no_2_itemsets(self, transactions):
        result = apriori(transactions, min_support=1, absolute=True)
        # Chỉ có 5 frequent 1-itemsets
        assert len(result) == 5
        for itemset in result:
            assert len(itemset) == 1


# Test 5: toy5_dense.txt — Dataset dày đặc
class TestToy5Dense:
    """7 giao dịch giống nhau {1,2,3,4,5,6}, minsup=1 → tất cả subsets frequent."""

    @pytest.fixture
    def transactions(self):
        return load_transactions_spmf(_data_path("toy5_dense.txt"))

    def test_all_subsets(self, transactions):
        result = apriori(transactions, min_support=1, absolute=True)
        # Tổng subsets không rỗng của {1,2,3,4,5,6} = 2^6 - 1 = 63
        assert len(result) == 63

    def test_all_support_7(self, transactions):
        """Tất cả đều có support = 7."""
        result = apriori(transactions, min_support=1, absolute=True)
        for sup in result.values():
            assert sup == 7


# Test: So sánh kết quả bitarray vs baseline
class TestBitarrayVsBaseline:
    """Kết quả phải giống nhau dù dùng bitarray hay không."""

    @pytest.mark.parametrize("filename", [
        "toy1.txt", "toy2_special.txt", "toy3.txt",
        "toy4_single.txt", "toy5_dense.txt"
    ])
    def test_same_results(self, filename):
        transactions = load_transactions_spmf(_data_path(filename))
        result_opt = apriori(transactions, min_support=2, absolute=True, use_bitarray=True)
        result_base = apriori_baseline(transactions, min_support=2, absolute=True)
        assert result_opt == result_base


# Test: Edge cases
class TestEdgeCases:

    def test_empty_database(self):
        """CSDL rỗng → không có frequent itemsets."""
        result = apriori([], min_support=0.5)
        assert result == {}

    def test_minsup_100_percent(self):
        """minsup = 1.0 (100%)."""
        transactions = load_transactions_spmf(_data_path("toy1.txt"))
        result = apriori(transactions, min_support=1.0)
        # Chỉ có items xuất hiện trong tất cả 5 giao dịch
        # Không có item nào có support = 5
        # {1}:3, {2}:4, {3}:4, {4}:2, {5}:4 → không item nào support=5
        assert len(result) == 0

    def test_minsup_very_low(self):
        """minsup rất thấp → tìm hầu hết itemsets."""
        transactions = load_transactions_spmf(_data_path("toy1.txt"))
        result = apriori(transactions, min_support=0.2)  # 1/5
        # Mọi item đều xuất hiện ít nhất 1 lần → minsup_count = 1
        # nhưng 0.2*5 = 1, nên minsup_count = 1
        assert len(result) > 0

    def test_single_transaction(self):
        """Chỉ 1 giao dịch."""
        transactions = [frozenset([1, 2, 3])]
        result = apriori(transactions, min_support=1, absolute=True)
        # Tất cả subsets: {1},{2},{3},{1,2},{1,3},{2,3},{1,2,3} = 7
        assert len(result) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
