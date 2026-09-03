"""
encoding.py
===========
Konversi representasi float -> bit (binary/ternary) dan packing ke uint64
supaya bisa dipopcount secara native oleh CPU (via np.bitwise_count).

Dua mode representasi:
- BINARY (bipolar {-1,+1})   : 1 bitplane -> sign bit saja.
- TERNARY ({-1, 0, +1})      : 2 bitplane -> (nonzero_bit, sign_bit),
                                 sesuai skema BitNet b1.58.
"""

import numpy as np

__all__ = [
    "pack_to_u64",
    "encode_bipolar",
    "encode_ternary",
]


def pack_to_u64(bits_01: np.ndarray) -> np.ndarray:
    """
    Pack matriks bit 0/1 berbentuk (N, D) menjadi (N, ceil(D/64)) uint64.
    Setiap uint64 menyimpan 64 dimensi -> siap dipopcount native.
    """
    if bits_01.dtype != np.uint8:
        bits_01 = bits_01.astype(np.uint8)
    n, d = bits_01.shape
    padded_d = ((d + 63) // 64) * 64
    if padded_d != d:
        pad = np.zeros((n, padded_d - d), dtype=np.uint8)
        bits_01 = np.hstack([bits_01, pad])
    packed8 = np.packbits(bits_01, axis=1)
    return packed8.view(np.uint64).reshape(n, packed8.shape[1] // 8)


def encode_bipolar(float_vectors: np.ndarray) -> np.ndarray:
    """
    Encode vektor float (N, D) -> bit 0/1 (N, D) berdasarkan tanda (sign).
    Representasi ini setara bipolar {-1,+1}: bit=1 -> +1, bit=0 -> -1.
    """
    return (float_vectors > 0).astype(np.uint8)


def encode_ternary(float_vectors: np.ndarray, zero_threshold: float = 0.0):
    """
    Encode vektor float (N, D) -> representasi ternary {-1, 0, +1} dalam
    2 bitplane terpisah (nonzero_bits, sign_bits), masing-masing (N, D) uint8.

    zero_threshold : nilai absolut di bawah ini dianggap 0 (sparsity control).
                      Semakin besar threshold -> makin banyak elemen jadi 0
                      -> makin hemat komputasi di tahap similarity ternary
                      (elemen nol tidak berkontribusi ke dot product).

    Return
    ------
    nonzero_bits : 1 jika |x| > threshold, else 0
    sign_bits    : 1 jika x > 0, else 0  (hanya bermakna saat nonzero_bits=1)
    """
    nonzero_bits = (np.abs(float_vectors) > zero_threshold).astype(np.uint8)
    sign_bits = (float_vectors > 0).astype(np.uint8)
    return nonzero_bits, sign_bits
