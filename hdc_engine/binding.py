"""
binding.py
==========
Operasi aljabar hyperdimensional computing untuk MEMBANGUN representasi
terstruktur (bukan sekadar flat feature vector) -- ini yang membuat HDC
lebih dari "classifier bit-packed", tapi genuinely engine komputasi:

- BIND (mengikat dua konsep jadi satu, mis. "role x filler")
- BUNDLE (menggabungkan banyak konsep jadi satu superposisi)
- PERMUTE (mengkodekan urutan/posisi, mis. utk data sekuensial)

Semua operasi bekerja pada representasi bit 0/1 (belum di-pack), lalu
di-pack terpisah lewat encoding.pack_to_u64 saat siap dipakai similarity.
"""

import numpy as np

__all__ = ["bind", "bundle", "permute", "unbind"]


def bind(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    """
    BIND dua vektor bit (XOR). Sifat penting: hasilnya QUASI-ORTHOGONAL
    terhadap a maupun b (tidak mirip keduanya) -- cocok utk representasi
    "role bound to filler", mis. bind(warna, merah).
    """
    return np.bitwise_xor(a_bits, b_bits)


def unbind(bound_bits: np.ndarray, key_bits: np.ndarray) -> np.ndarray:
    """
    UNBIND -- karena XOR self-invertible, bind(bind(a,b), b) == a.
    Dipakai untuk "mengeluarkan kembali" filler dari representasi gabungan.
    """
    return np.bitwise_xor(bound_bits, key_bits)


def bundle(list_of_bits: np.ndarray) -> np.ndarray:
    """
    BUNDLE banyak vektor bit jadi satu vektor "superposisi" lewat majority
    vote per-bit. list_of_bits: (N_vectors, dim) uint8 0/1.
    Hasilnya MIRIP dengan semua anggota (similarity > 0 ke semuanya),
    beda dengan BIND yang membuat hasil tidak mirip siapa pun.
    """
    n = list_of_bits.shape[0]
    votes = list_of_bits.sum(axis=0)
    # tie-break acak-tapi-deterministik: >= half dianggap 1 saat n genap
    threshold = n / 2.0
    return (votes >= threshold).astype(np.uint8)


def permute(bits: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    PERMUTE (cyclic shift) -- mengkodekan posisi/urutan. permute(v, 1) berbeda
    dan quasi-orthogonal terhadap v, tapi operasinya invertible (shift balik).
    Berguna utk merepresentasikan sekuens: bundle([v1, permute(v2,1), permute(v3,2)]).
    """
    return np.roll(bits, shift=shift, axis=-1)
