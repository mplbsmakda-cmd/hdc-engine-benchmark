"""
cascade.py
==========
Pencarian dua tahap yang mengubah kompleksitas dari O(Q*C*D) menjadi
O(Q*C*d_sketch) + O(Q*K*D), memanfaatkan redundansi statistik representasi
hyperdimensional (bukti empiris: subsample kecil sudah cukup representatif
untuk menyaring kandidat yang benar, karena informasi tersebar merata
di seluruh dimensi -- lihat README bagian "Kenapa Cascade Bekerja").
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .kernel import hamming_similarity_blocked

__all__ = ["cascade_search"]


def _verify_one(qi, query_u64_row, cand_idx_row, class_u64, dim):
    c_vecs = class_u64[cand_idx_row]
    xor = np.bitwise_xor(query_u64_row[None, :], c_vecs)
    pc = np.bitwise_count(xor).sum(axis=1)
    sims = 1.0 - 2.0 * pc / dim
    local_best = np.argmax(sims)
    return qi, cand_idx_row[local_best], sims[local_best]


def cascade_search(query_u64, class_u64, query_sketch, class_sketch,
                    dim, sketch_bits, top_k, block_c=256, block_q=32,
                    n_threads=1):
    """
    Return: (pred_idx, best_score) masing-masing (Q,) -- index kelas terbaik
    dan skor similaritynya, per query.

    n_threads > 1 : paralelkan tahap verifikasi (tahap 2) antar-query
                    memakai ThreadPoolExecutor -- NumPy melepas GIL saat
                    menjalankan operasi bitwise besar, jadi threading nyata
                    memberi manfaat di CPU multi-core.
    """
    Q = query_u64.shape[0]
    C = class_u64.shape[0]
    k = min(top_k, C)

    # --- TAHAP 1: sketch murah untuk semua kelas ---
    sketch_sim = hamming_similarity_blocked(
        query_sketch, class_sketch, sketch_bits, block_r=block_c, block_q=block_q
    )
    cand_idx = np.argpartition(-sketch_sim, k - 1, axis=1)[:, :k]  # (Q, k)

    # --- TAHAP 2: verifikasi presisi penuh, hanya utk top-k kandidat ---
    pred_idx = np.empty(Q, dtype=np.int64)
    best_score = np.empty(Q, dtype=np.float32)

    if n_threads <= 1:
        for qi in range(Q):
            _, best_c, score = _verify_one(qi, query_u64[qi], cand_idx[qi], class_u64, dim)
            pred_idx[qi] = best_c
            best_score[qi] = score
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [
                ex.submit(_verify_one, qi, query_u64[qi], cand_idx[qi], class_u64, dim)
                for qi in range(Q)
            ]
            for fut in futures:
                qi, best_c, score = fut.result()
                pred_idx[qi] = best_c
                best_score[qi] = score

    return pred_idx, best_score
