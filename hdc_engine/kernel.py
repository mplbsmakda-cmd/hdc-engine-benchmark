"""
kernel.py
=========
Kernel similarity bitwise, cache-blocked, sebagai pengganti matmul float
untuk representasi binary/ternary yang sudah di-pack ke uint64.

Semua fungsi di sini melakukan TILING manual (block_r x block_q) supaya
array intermediate hasil XOR selalu muat di cache CPU (L1/L2) -- ini
adalah pelajaran terpenting dari eksperimen awal: broadcast penuh tanpa
blocking membuat operasi bitwise justru LEBIH LAMBAT dari BLAS float32
karena menjadi memory-bandwidth bound.
"""

import numpy as np

__all__ = [
    "hamming_similarity_blocked",
    "ternary_similarity_blocked",
]


def hamming_similarity_blocked(query_u64, ref_u64, dim, block_r=256, block_q=32):
    """
    Cosine-similarity bipolar (via Hamming distance) untuk representasi BINARY.

    sim(a,b) = 1 - 2*hamming(a,b)/dim   (identik dengan cosine sim bipolar {-1,+1})

    query_u64 : (Q, n64) uint64
    ref_u64   : (R, n64) uint64
    Return    : (Q, R) float32 similarity matrix
    """
    Q = query_u64.shape[0]
    R = ref_u64.shape[0]
    sims = np.empty((Q, R), dtype=np.float32)
    for qi in range(0, Q, block_q):
        q_blk = query_u64[qi:qi + block_q]
        for ri in range(0, R, block_r):
            r_blk = ref_u64[ri:ri + block_r]
            xor = np.bitwise_xor(q_blk[:, None, :], r_blk[None, :, :])
            pc = np.bitwise_count(xor).sum(axis=2)
            sims[qi:qi + block_q, ri:ri + block_r] = 1.0 - 2.0 * pc / dim
    return sims


def ternary_similarity_blocked(query_nz_u64, query_sign_u64,
                                ref_nz_u64, ref_sign_u64,
                                dim, block_r=256, block_q=32):
    """
    Dot product ternary {-1,0,+1} murni bitwise (skema mirip BitNet):

      both_nz    = nz_a & nz_b                 (elemen dimana KEDUANYA tidak nol)
      diff_sign  = sign_a XOR sign_b
      mismatches = both_nz & diff_sign         (kedua nonzero, tapi tanda beda -> kontribusi -1)
      dot        = popcount(both_nz) - 2*popcount(mismatches)

    Ini identik secara matematis dengan sum(a_i * b_i) untuk a,b ternary,
    TANPA operasi perkalian sama sekali -- hanya AND/XOR/popcount.

    Return : (Q, R) float32, dot product ternary (belum dinormalisasi ke [-1,1];
              bagi dengan dim jika ingin skala cosine-like).
    """
    Q = query_nz_u64.shape[0]
    R = ref_nz_u64.shape[0]
    dots = np.empty((Q, R), dtype=np.float32)
    for qi in range(0, Q, block_q):
        qnz = query_nz_u64[qi:qi + block_q]
        qsg = query_sign_u64[qi:qi + block_q]
        for ri in range(0, R, block_r):
            rnz = ref_nz_u64[ri:ri + block_r]
            rsg = ref_sign_u64[ri:ri + block_r]

            both_nz = np.bitwise_and(qnz[:, None, :], rnz[None, :, :])
            diff_sign = np.bitwise_xor(qsg[:, None, :], rsg[None, :, :])
            mismatches = np.bitwise_and(both_nz, diff_sign)

            # PENTING: bitwise_count mengembalikan tipe unsigned. Harus di-cast
            # ke signed SEBELUM pengurangan, atau (pc_both - 2*pc_mis) bisa
            # underflow/wraparound jadi angka raksasa saat hasilnya negatif.
            pc_both = np.bitwise_count(both_nz).sum(axis=2).astype(np.int64)
            pc_mis = np.bitwise_count(mismatches).sum(axis=2).astype(np.int64)
            dots[qi:qi + block_q, ri:ri + block_r] = (pc_both - 2 * pc_mis).astype(np.float32)
    return dots
