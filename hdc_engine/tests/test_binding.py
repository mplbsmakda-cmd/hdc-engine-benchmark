import unittest
import numpy as np

from hdc_engine.binding import bind, unbind, bundle, permute


class TestBind(unittest.TestCase):
    def test_bind_unbind_recovers_original(self):
        rng = np.random.default_rng(7)
        a = (rng.random(1000) > 0.5).astype(np.uint8)
        b = (rng.random(1000) > 0.5).astype(np.uint8)
        bound = bind(a, b)
        recovered = unbind(bound, b)
        np.testing.assert_array_equal(recovered, a)

    def test_bind_result_is_dissimilar_from_inputs(self):
        rng = np.random.default_rng(8)
        dim = 5000
        a = (rng.random(dim) > 0.5).astype(np.uint8)
        b = (rng.random(dim) > 0.5).astype(np.uint8)
        bound = bind(a, b)

        def bipolar_sim(x, y):
            xb = x.astype(np.float32) * 2 - 1
            yb = y.astype(np.float32) * 2 - 1
            return np.dot(xb, yb) / dim

        sim_to_a = bipolar_sim(bound, a)
        sim_to_b = bipolar_sim(bound, b)
        # hasil bind harus quasi-orthogonal (mendekati 0) terhadap kedua input
        self.assertLess(abs(sim_to_a), 0.1)
        self.assertLess(abs(sim_to_b), 0.1)


class TestBundle(unittest.TestCase):
    def test_bundle_similar_to_all_members(self):
        rng = np.random.default_rng(9)
        dim = 5000
        vecs = (rng.random((5, dim)) > 0.5).astype(np.uint8)
        bundled = bundle(vecs)

        def bipolar_sim(x, y):
            xb = x.astype(np.float32) * 2 - 1
            yb = y.astype(np.float32) * 2 - 1
            return np.dot(xb, yb) / dim

        for v in vecs:
            sim = bipolar_sim(bundled, v)
            self.assertGreater(sim, 0.2)  # harus cukup mirip ke tiap anggota

    def test_bundle_identical_vectors_returns_same(self):
        dim = 200
        v = np.ones(dim, dtype=np.uint8)
        stacked = np.tile(v, (4, 1))
        bundled = bundle(stacked)
        np.testing.assert_array_equal(bundled, v)


class TestPermute(unittest.TestCase):
    def test_permute_invertible(self):
        rng = np.random.default_rng(10)
        v = (rng.random(300) > 0.5).astype(np.uint8)
        shifted = permute(v, shift=7)
        restored = permute(shifted, shift=-7)
        np.testing.assert_array_equal(restored, v)

    def test_permute_changes_vector(self):
        rng = np.random.default_rng(11)
        v = (rng.random(300) > 0.5).astype(np.uint8)
        shifted = permute(v, shift=3)
        self.assertFalse(np.array_equal(v, shifted))


if __name__ == "__main__":
    unittest.main()
