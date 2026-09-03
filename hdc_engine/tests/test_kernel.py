import unittest
import numpy as np

from hdc_engine.encoding import pack_to_u64, encode_bipolar, encode_ternary
from hdc_engine.kernel import hamming_similarity_blocked, ternary_similarity_blocked


class TestHammingSimilarity(unittest.TestCase):
    def test_matches_bruteforce_bipolar_dot(self):
        rng = np.random.default_rng(3)
        dim = 256
        a_float = rng.standard_normal((7, dim)).astype(np.float32)
        b_float = rng.standard_normal((11, dim)).astype(np.float32)

        a_bits = encode_bipolar(a_float)
        b_bits = encode_bipolar(b_float)
        a_u64 = pack_to_u64(a_bits)
        b_u64 = pack_to_u64(b_bits)

        sim = hamming_similarity_blocked(a_u64, b_u64, dim, block_r=4, block_q=3)

        # brute-force: konversi bit ke bipolar {-1,+1} lalu dot product / dim
        a_bipolar = a_bits.astype(np.float32) * 2 - 1
        b_bipolar = b_bits.astype(np.float32) * 2 - 1
        expected = (a_bipolar @ b_bipolar.T) / dim

        np.testing.assert_allclose(sim, expected, atol=1e-5)

    def test_identical_vectors_similarity_one(self):
        rng = np.random.default_rng(4)
        dim = 128
        v_float = rng.standard_normal((3, dim)).astype(np.float32)
        v_bits = encode_bipolar(v_float)
        v_u64 = pack_to_u64(v_bits)
        sim = hamming_similarity_blocked(v_u64, v_u64, dim)
        diag = np.diag(sim)
        np.testing.assert_allclose(diag, np.ones(3), atol=1e-6)

    def test_different_block_sizes_give_same_result(self):
        rng = np.random.default_rng(5)
        dim = 320
        a_bits = (rng.random((13, dim)) > 0.5).astype(np.uint8)
        b_bits = (rng.random((17, dim)) > 0.5).astype(np.uint8)
        a_u64, b_u64 = pack_to_u64(a_bits), pack_to_u64(b_bits)

        sim_small_block = hamming_similarity_blocked(a_u64, b_u64, dim, block_r=2, block_q=2)
        sim_large_block = hamming_similarity_blocked(a_u64, b_u64, dim, block_r=100, block_q=100)
        np.testing.assert_allclose(sim_small_block, sim_large_block, atol=1e-6)


class TestTernarySimilarity(unittest.TestCase):
    def test_matches_bruteforce_ternary_dot(self):
        rng = np.random.default_rng(6)
        dim = 192
        a_float = rng.standard_normal((5, dim)).astype(np.float32)
        b_float = rng.standard_normal((6, dim)).astype(np.float32)

        threshold = 0.3
        a_nz, a_sg = encode_ternary(a_float, threshold)
        b_nz, b_sg = encode_ternary(b_float, threshold)

        # bentuk ternary eksplisit {-1,0,+1} utk brute-force
        def to_ternary(nz, sg):
            t = np.zeros_like(nz, dtype=np.float32)
            t[(nz == 1) & (sg == 1)] = 1.0
            t[(nz == 1) & (sg == 0)] = -1.0
            return t

        a_t = to_ternary(a_nz, a_sg)
        b_t = to_ternary(b_nz, b_sg)
        expected = a_t @ b_t.T

        a_nz_u64, a_sg_u64 = pack_to_u64(a_nz), pack_to_u64(a_sg)
        b_nz_u64, b_sg_u64 = pack_to_u64(b_nz), pack_to_u64(b_sg)

        got = ternary_similarity_blocked(a_nz_u64, a_sg_u64, b_nz_u64, b_sg_u64,
                                          dim, block_r=3, block_q=2)
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_all_zero_vector_gives_zero_dot(self):
        dim = 128
        a_nz = np.zeros((1, dim), dtype=np.uint8)
        a_sg = np.zeros((1, dim), dtype=np.uint8)
        b_nz = np.ones((1, dim), dtype=np.uint8)
        b_sg = np.ones((1, dim), dtype=np.uint8)
        a_nz_u64, a_sg_u64 = pack_to_u64(a_nz), pack_to_u64(a_sg)
        b_nz_u64, b_sg_u64 = pack_to_u64(b_nz), pack_to_u64(b_sg)
        got = ternary_similarity_blocked(a_nz_u64, a_sg_u64, b_nz_u64, b_sg_u64, dim)
        self.assertAlmostEqual(got[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
