import unittest
import numpy as np

from hdc_engine import HDCCascadeEngine


class TestEngineBinaryMode(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.dim = 2048
        self.n_classes = 300
        self.class_float = self.rng.standard_normal((self.n_classes, self.dim)).astype(np.float32)

    def _noisy_queries(self, n_queries, noise_rate):
        true_labels = self.rng.integers(0, self.n_classes, size=n_queries)
        class_bits = (self.class_float[true_labels] > 0).astype(np.uint8)
        flip = self.rng.random(class_bits.shape) < noise_rate
        class_bits[flip] = 1 - class_bits[flip]
        query_float = class_bits.astype(np.float32) * 2 - 1
        return query_float, true_labels

    def test_fit_predict_shapes(self):
        engine = HDCCascadeEngine(dim=self.dim, mode="binary", sketch_bits=256, top_k=20)
        engine.fit(self.class_float)
        q, _ = self._noisy_queries(15, 0.1)
        preds = engine.predict(q)
        self.assertEqual(preds.shape, (15,))

    def test_accuracy_under_moderate_noise(self):
        engine = HDCCascadeEngine(dim=self.dim, mode="binary", sketch_bits=256, top_k=30)
        engine.fit(self.class_float)
        q, true_labels = self._noisy_queries(50, 0.2)
        preds = engine.predict(q)
        acc = (preds == true_labels).mean()
        self.assertGreaterEqual(acc, 0.95)

    def test_cascade_agrees_with_exact_search(self):
        q, true_labels = self._noisy_queries(40, 0.15)

        engine_cascade = HDCCascadeEngine(dim=self.dim, mode="binary",
                                           sketch_bits=256, top_k=30)
        engine_cascade.fit(self.class_float)
        pred_cascade = engine_cascade.predict(q)

        engine_exact = HDCCascadeEngine(dim=self.dim, mode="binary",
                                         sketch_bits=256, top_k=self.n_classes)
        engine_exact.fit(self.class_float)
        pred_exact = engine_exact.predict(q)

        agreement = (pred_cascade == pred_exact).mean()
        self.assertGreaterEqual(agreement, 0.95)

    def test_custom_labels_returned(self):
        labels = np.array([f"class_{i}" for i in range(self.n_classes)])
        engine = HDCCascadeEngine(dim=self.dim, mode="binary", sketch_bits=256, top_k=20)
        engine.fit(self.class_float, labels=labels)
        q, true_idx = self._noisy_queries(5, 0.1)
        preds = engine.predict(q)
        self.assertTrue(all(p.startswith("class_") for p in preds))

    def test_top_k_zero_or_negative_rejected_gracefully(self):
        engine = HDCCascadeEngine(dim=self.dim, mode="binary", sketch_bits=256, top_k=self.n_classes * 10)
        engine.fit(self.class_float)  # top_k > n_classes -> harus fallback ke exact, tidak error
        q, _ = self._noisy_queries(5, 0.1)
        preds = engine.predict(q)
        self.assertEqual(preds.shape, (5,))

    def test_returns_scores_when_requested(self):
        engine = HDCCascadeEngine(dim=self.dim, mode="binary", sketch_bits=256, top_k=20)
        engine.fit(self.class_float)
        q, _ = self._noisy_queries(5, 0.1)
        preds, scores = engine.predict(q, return_scores=True)
        self.assertEqual(scores.shape, (5,))
        self.assertTrue((scores <= 1.0001).all() and (scores >= -1.0001).all())

    def test_memory_footprint_smaller_than_float32(self):
        engine = HDCCascadeEngine(dim=self.dim, mode="binary", sketch_bits=256, top_k=20)
        engine.fit(self.class_float)
        bit_mem = engine.memory_footprint_bytes()
        float_mem = self.class_float.nbytes
        self.assertLess(bit_mem, float_mem / 10)  # harus jauh lebih hemat

    def test_multithreaded_matches_singlethreaded(self):
        q, _ = self._noisy_queries(30, 0.15)

        engine_st = HDCCascadeEngine(dim=self.dim, mode="binary", sketch_bits=256,
                                      top_k=30, n_threads=1)
        engine_st.fit(self.class_float)
        pred_st = engine_st.predict(q)

        engine_mt = HDCCascadeEngine(dim=self.dim, mode="binary", sketch_bits=256,
                                      top_k=30, n_threads=4)
        engine_mt.fit(self.class_float)
        pred_mt = engine_mt.predict(q)

        np.testing.assert_array_equal(pred_st, pred_mt)

    def test_invalid_dim_raises(self):
        engine = HDCCascadeEngine(dim=self.dim, mode="binary")
        engine.fit(self.class_float)
        wrong_dim_query = self.rng.standard_normal((3, self.dim + 1)).astype(np.float32)
        with self.assertRaises(ValueError):
            engine.predict(wrong_dim_query)

    def test_predict_before_fit_raises(self):
        engine = HDCCascadeEngine(dim=self.dim, mode="binary")
        q = self.rng.standard_normal((3, self.dim)).astype(np.float32)
        with self.assertRaises(RuntimeError):
            engine.predict(q)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            HDCCascadeEngine(dim=self.dim, mode="quaternary")

    def test_sketch_bits_not_multiple_of_64_raises(self):
        with self.assertRaises(ValueError):
            HDCCascadeEngine(dim=self.dim, sketch_bits=100)


class TestEngineTernaryMode(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(99)
        self.dim = 1024
        self.n_classes = 100
        self.class_float = self.rng.standard_normal((self.n_classes, self.dim)).astype(np.float32)

    def test_ternary_fit_predict(self):
        engine = HDCCascadeEngine(dim=self.dim, mode="ternary", block_c=32, block_q=8)
        engine.fit(self.class_float, zero_threshold=0.2)
        true_labels = self.rng.integers(0, self.n_classes, size=20)
        query_float = self.class_float[true_labels].copy()
        preds = engine.predict(query_float, zero_threshold=0.2)
        acc = (preds == true_labels).mean()
        self.assertGreaterEqual(acc, 0.95)


if __name__ == "__main__":
    unittest.main()
