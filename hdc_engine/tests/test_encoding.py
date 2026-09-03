import unittest
import numpy as np

from hdc_engine.encoding import pack_to_u64, encode_bipolar, encode_ternary


class TestPacking(unittest.TestCase):
    def test_pack_roundtrip_via_unpackbits(self):
        rng = np.random.default_rng(1)
        bits = (rng.random((5, 128)) > 0.5).astype(np.uint8)
        packed = pack_to_u64(bits)
        self.assertEqual(packed.shape, (5, 2))  # 128/64 = 2
        # unpack manual utk verifikasi isi benar
        unpacked = np.unpackbits(packed.view(np.uint8), axis=1)
        np.testing.assert_array_equal(unpacked, bits)

    def test_pack_with_padding(self):
        rng = np.random.default_rng(2)
        bits = (rng.random((3, 100)) > 0.5).astype(np.uint8)  # bukan kelipatan 64
        packed = pack_to_u64(bits)
        self.assertEqual(packed.shape, (3, 2))  # ceil(100/64)=2
        unpacked = np.unpackbits(packed.view(np.uint8), axis=1)[:, :100]
        np.testing.assert_array_equal(unpacked, bits)

    def test_pack_empty_safe(self):
        bits = np.zeros((0, 64), dtype=np.uint8)
        packed = pack_to_u64(bits)
        self.assertEqual(packed.shape, (0, 1))


class TestEncodeBipolar(unittest.TestCase):
    def test_sign_mapping(self):
        x = np.array([[1.0, -1.0, 0.5, -0.5, 0.0]])
        bits = encode_bipolar(x)
        # x>0 -> 1, x<=0 -> 0 (termasuk nol dianggap negatif di skema bipolar)
        np.testing.assert_array_equal(bits, [[1, 0, 1, 0, 0]])

    def test_output_dtype(self):
        x = np.random.randn(4, 10).astype(np.float32)
        bits = encode_bipolar(x)
        self.assertEqual(bits.dtype, np.uint8)
        self.assertTrue(((bits == 0) | (bits == 1)).all())


class TestEncodeTernary(unittest.TestCase):
    def test_zero_detection(self):
        x = np.array([[0.9, -0.9, 0.001, -0.001, 0.0]])
        nz, sg = encode_ternary(x, zero_threshold=0.1)
        np.testing.assert_array_equal(nz, [[1, 1, 0, 0, 0]])
        # sign_bits hanya bermakna di posisi nz=1 -- posisi nz=0 adalah "don't care",
        # jadi hanya verifikasi sign pada posisi yang benar-benar nonzero.
        np.testing.assert_array_equal(sg[nz == 1], [1, 0])

    def test_no_threshold_only_exact_zero_is_zero(self):
        x = np.array([[0.0001, -0.0001, 0.0]])
        nz, sg = encode_ternary(x, zero_threshold=0.0)
        np.testing.assert_array_equal(nz, [[1, 1, 0]])


if __name__ == "__main__":
    unittest.main()
