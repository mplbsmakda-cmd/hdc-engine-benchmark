import unittest
import numpy as np

from hdc_engine.benchmark import benchmark_vs_blas


class TestBenchmark(unittest.TestCase):
    def test_benchmark_runs_and_reports_speedup(self):
        rng = np.random.default_rng(123)
        dim = 2048
        n_classes = 500
        n_queries = 40

        class_float = rng.standard_normal((n_classes, dim)).astype(np.float32)
        true_labels = rng.integers(0, n_classes, size=n_queries)
        class_bits = (class_float[true_labels] > 0).astype(np.uint8)
        flip = rng.random(class_bits.shape) < 0.2
        class_bits[flip] = 1 - class_bits[flip]
        query_float = class_bits.astype(np.float32) * 2 - 1

        result = benchmark_vs_blas(class_float, query_float, true_labels=true_labels,
                                    sketch_bits=256, top_k=30, repeats=2)

        self.assertIn("speedup_cascade_vs_blas", result)
        self.assertIn("accuracy_cascade", result)
        self.assertGreaterEqual(result["accuracy_cascade"], 0.9)
        self.assertGreater(result["memory_savings_x"], 5)


if __name__ == "__main__":
    unittest.main()
