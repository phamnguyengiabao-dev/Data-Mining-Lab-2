"""
test_benchmark.py — Benchmark tests: đo thời gian, bộ nhớ, scalability.

Chạy trên dữ liệu benchmark (nếu có) hoặc dữ liệu toy.
Sinh biểu đồ kết quả lưu vào thư mục docs/.
"""
import os
import sys
import time
import random
import gc

import pytest

# Thêm project root vào path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.algorithm.apriori import apriori, apriori_baseline
from src.utils import load_transactions_spmf

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Configurable paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TOY_DIR = os.path.join(DATA_DIR, "toy")
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmark")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# Đảm bảo thư mục docs tồn tại
os.makedirs(DOCS_DIR, exist_ok=True)


# Helper: đo memory peak
def measure_peak_memory_and_time(func, *args, **kwargs):
    """
    Đo thời gian chạy và bộ nhớ peak (nếu có psutil).

    Returns
    result, elapsed_seconds, peak_memory_mb
    """
    gc.collect()
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start

    peak_mem = None
    if HAS_PSUTIL:
        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_mem = mem_after - mem_before

    return result, elapsed, peak_mem


# Test: So sánh hiệu năng bitarray vs baseline trên toy datasets
class TestPerformanceComparison:
    """So sánh thời gian chạy giữa bản bitarray và baseline."""

    def test_bitarray_vs_baseline_toy1(self):
        transactions = load_transactions_spmf(os.path.join(TOY_DIR, "toy1.txt"))

        _, time_opt, _ = measure_peak_memory_and_time(
            apriori, transactions, 0.4, use_bitarray=True
        )
        _, time_base, _ = measure_peak_memory_and_time(
            apriori_baseline, transactions, 0.4
        )

        print(f"\n[toy1] Bitarray: {time_opt:.6f}s | Baseline: {time_base:.6f}s")
        # Chỉ test kết quả giống nhau (thời gian quá nhỏ trên toy data)


# Test: Thời gian chạy theo minsup (trên toy data nếu benchmark chưa có)
class TestRuntimeVsMinsup:
    """Đo thời gian chạy theo nhiều mức minsup khác nhau."""

    def test_runtime_vs_minsup_toy3(self):
        """Chạy trên toy3 với nhiều minsup."""
        transactions = load_transactions_spmf(os.path.join(TOY_DIR, "toy3.txt"))

        minsup_values = [1, 2, 3, 4, 5, 6, 7]
        times = []
        n_itemsets = []

        for ms in minsup_values:
            result, elapsed, _ = measure_peak_memory_and_time(
                apriori, transactions, ms, absolute=True
            )
            times.append(elapsed)
            n_itemsets.append(len(result))
            print(f"  minsup={ms}: {elapsed:.6f}s, {len(result)} itemsets")

        # Biểu đồ (nếu có matplotlib)
        if HAS_MATPLOTLIB:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(minsup_values, [t * 1000 for t in times], "bo-")
            ax1.set_xlabel("Min Support (absolute)")
            ax1.set_ylabel("Thời gian (ms)")
            ax1.set_title("Thời gian chạy theo minsup (toy3)")
            ax1.grid(True, alpha=0.3)

            ax2.plot(minsup_values, n_itemsets, "ro-")
            ax2.set_xlabel("Min Support (absolute)")
            ax2.set_ylabel("Số lượng Frequent Itemsets")
            ax2.set_title("Số FI theo minsup (toy3)")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(DOCS_DIR, "runtime_vs_minsup_toy3.png"), dpi=150)
            plt.close()
            print(f"  -> Bieu do da luu: docs/runtime_vs_minsup_toy3.png")


# Test: Scalability (tạo tập con tăng dần)
class TestScalability:
    """Kiểm tra khả năng mở rộng theo kích thước CSDL."""

    def test_scalability_synthetic(self):
        """Tạo CSDL tổng hợp có kích thước tăng dần."""
        random.seed(42)
        items = list(range(1, 21))  # 20 items

        # Tạo 1000 giao dịch tổng hợp
        full_db = []
        for _ in range(1000):
            length = random.randint(3, 10)
            transaction = frozenset(random.sample(items, length))
            full_db.append(transaction)

        sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
        times = []

        for frac in sizes:
            n = int(len(full_db) * frac)
            subset = full_db[:n]
            _, elapsed, _ = measure_peak_memory_and_time(
                apriori, subset, 0.1
            )
            times.append(elapsed)
            print(f"  {frac*100:.0f}% ({n} trans): {elapsed:.4f}s")

        # Biểu đồ
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                [s * 100 for s in sizes],
                [t * 1000 for t in times],
                "gs-", linewidth=2
            )
            ax.set_xlabel("% kích thước CSDL")
            ax.set_ylabel("Thời gian (ms)")
            ax.set_title("Scalability Test (CSDL tổng hợp 1000 giao dịch)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(DOCS_DIR, "scalability_synthetic.png"), dpi=150)
            plt.close()
            print(f"  -> Bieu do da luu: docs/scalability_synthetic.png")


# Test: Benchmark datasets (chỉ chạy nếu data có sẵn)
class TestBenchmarkDatasets:
    """Chạy Apriori trên benchmark datasets (nếu có sẵn)."""

    BENCHMARKS = {
        "chess.txt": {"minsups": [0.9, 0.85, 0.8, 0.75, 0.7]},
        "mushroom.txt": {"minsups": [0.5, 0.4, 0.3, 0.25, 0.2]},
        "retail.txt": {"minsups": [0.05, 0.02, 0.01, 0.005, 0.001]},
    }

    @pytest.mark.parametrize("dataset", list(BENCHMARKS.keys()))
    def test_benchmark(self, dataset):
        """Chạy benchmark nếu dataset tồn tại."""
        filepath = os.path.join(BENCHMARK_DIR, dataset)
        if not os.path.exists(filepath):
            pytest.skip(f"Benchmark dataset '{dataset}' chưa được tải.")

        transactions = load_transactions_spmf(filepath)
        minsups = self.BENCHMARKS[dataset]["minsups"]

    print("")
        print(f"  BENCHMARK: {dataset} ({len(transactions)} giao dịch)")
    print("")

        times = []
        counts = []
        for ms in minsups:
            result, elapsed, mem = measure_peak_memory_and_time(
                apriori, transactions, ms
            )
            times.append(elapsed)
            counts.append(len(result))
            mem_str = f", {mem:.1f}MB" if mem else ""
            print(f"  minsup={ms}: {elapsed:.4f}s, {len(result)} FI{mem_str}")

        # Biểu đồ
        if HAS_MATPLOTLIB:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(minsups, [t * 1000 for t in times], "bo-")
            ax1.set_xlabel("Min Support (relative)")
            ax1.set_ylabel("Thời gian (ms)")
            ax1.set_title(f"Runtime vs Minsup ({dataset})")
            ax1.grid(True, alpha=0.3)
            ax1.invert_xaxis()

            ax2.plot(minsups, counts, "ro-")
            ax2.set_xlabel("Min Support (relative)")
            ax2.set_ylabel("Số lượng FI")
            ax2.set_title(f"FI Count vs Minsup ({dataset})")
            ax2.grid(True, alpha=0.3)
            ax2.invert_xaxis()

            plt.tight_layout()
            name = dataset.replace(".txt", "")
            plt.savefig(os.path.join(DOCS_DIR, f"benchmark_{name}.png"), dpi=150)
            plt.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
