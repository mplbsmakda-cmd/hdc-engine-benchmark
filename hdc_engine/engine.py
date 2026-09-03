"""
engine.py
=========
API utama: HDCCascadeEngine. Menyatukan encoding, kernel cache-blocked,
dan cascade pruning menjadi satu objek siap pakai.

Penyempurnaan dari versi sebelumnya:
- Validasi input eksplisit (dimensi, tipe, NaN) -- gagal cepat dgn pesan jelas.
- Mode ganda: "binary" (cepat, 1 bitplane) dan "ternary" (lebih presisi, 2 bitplane,
  skema BitNet-style, cocok kalau distribusi bobot punya banyak elemen mendekati nol).
- Opsi n_threads untuk paralelisasi tahap verifikasi cascade di CPU multi-core.
- Fallback otomatis ke exact search kalau top_k >= n_classes (tidak ada gunanya
  cascade untuk kasus itu, jadi dilewati agar tidak ada overhead sia-sia).
"""

import numpy as np

from .encoding import encode_bipolar, encode_ternary, pack_to_u64
from .kernel import hamming_similarity_blocked, ternary_similarity_blocked
from .cascade import cascade_search

__all__ = ["HDCCascadeEngine"]


class HDCCascadeEngine:
    """
    Engine klasifikasi/retrieval berbasis hyperdimensional binary/ternary vector.

    Parameter
    ---------
    dim : int
        Dimensi vektor hiperdimensional (disarankan >= 2000).
    mode : "binary" | "ternary"
        "binary"  -> 1 bitplane, tercepat, cocok utk kebanyakan kasus.
        "ternary" -> 2 bitplane, mendukung elemen bernilai nol eksplisit
                     (skema BitNet b1.58), lebih presisi utk data sparse.
    sketch_bits : int
        Jumlah bit awal dipakai sebagai sketch murah utk pruning tahap 1.
    top_k : int
        Kandidat yang lolos ke verifikasi presisi penuh tahap 2.
        Set >= n_classes untuk menonaktifkan pruning (exact search).
    block_c, block_q : int
        Ukuran blok cache-blocked kernel (tuning sesuai ukuran cache CPU).
    n_threads : int
        Jumlah thread untuk paralelisasi tahap verifikasi cascade.
    """

    def __init__(self, dim: int, mode: str = "binary", sketch_bits: int = 512,
                 top_k: int = 50, block_c: int = 256, block_q: int = 32,
                 n_threads: int = 1):
        if mode not in ("binary", "ternary"):
            raise ValueError(f"mode harus 'binary' atau 'ternary', dapat: {mode!r}")
        if sketch_bits % 64 != 0:
            raise ValueError("sketch_bits harus kelipatan 64")
        if sketch_bits > dim:
            raise ValueError("sketch_bits tidak boleh lebih besar dari dim")

        self.dim = dim
        self.mode = mode
        self.sketch_bits = sketch_bits
        self.sketch_n64 = sketch_bits // 64
        self.top_k = top_k
        self.block_c = block_c
        self.block_q = block_q
        self.n_threads = n_threads

        self.n_classes = 0
        self.labels = None
        # binary mode
        self.class_u64 = None
        self.class_sketch = None
        # ternary mode
        self.class_nz_u64 = None
        self.class_sign_u64 = None

    # ------------------------------------------------------------------
    def _validate_float(self, x: np.ndarray, name: str):
        if x.ndim != 2:
            raise ValueError(f"{name} harus 2D (N, dim), dapat shape {x.shape}")
        if x.shape[1] != self.dim:
            raise ValueError(f"{name} punya dim={x.shape[1]}, engine dikonfigurasi dim={self.dim}")
        if not np.isfinite(x).all():
            raise ValueError(f"{name} mengandung NaN/Inf")

    # ------------------------------------------------------------------
    def fit(self, class_float_vectors: np.ndarray, labels=None, zero_threshold: float = 0.0):
        """
        class_float_vectors : (n_classes, dim) float -- akan di-encode otomatis
                               sesuai self.mode (binary: sign saja, ternary: sign+nonzero).
        labels : opsional, dikembalikan saat predict (default: index integer).
        zero_threshold : hanya dipakai mode ternary -- ambang |x| dianggap nol.
        """
        self._validate_float(class_float_vectors, "class_float_vectors")
        self.n_classes = class_float_vectors.shape[0]
        self.labels = np.asarray(labels) if labels is not None else np.arange(self.n_classes)

        if self.mode == "binary":
            bits = encode_bipolar(class_float_vectors)
            self.class_u64 = pack_to_u64(bits)
            self.class_sketch = self.class_u64[:, :self.sketch_n64]
        else:  # ternary
            nz, sg = encode_ternary(class_float_vectors, zero_threshold)
            self.class_nz_u64 = pack_to_u64(nz)
            self.class_sign_u64 = pack_to_u64(sg)
        return self

    # ------------------------------------------------------------------
    def predict(self, query_float_vectors: np.ndarray, zero_threshold: float = 0.0,
                return_scores: bool = False):
        self._validate_float(query_float_vectors, "query_float_vectors")
        if self.mode == "binary":
            return self._predict_binary(query_float_vectors, return_scores)
        else:
            return self._predict_ternary(query_float_vectors, zero_threshold, return_scores)

    # ------------------------------------------------------------------
    def _predict_binary(self, query_float_vectors, return_scores):
        if self.class_u64 is None:
            raise RuntimeError("Panggil .fit() dahulu")

        q_bits = encode_bipolar(query_float_vectors)
        q_u64 = pack_to_u64(q_bits)
        q_sketch = q_u64[:, :self.sketch_n64]

        if self.top_k >= self.n_classes:
            # exact search murni, cache-blocked, tanpa cascade
            sims = hamming_similarity_blocked(
                q_u64, self.class_u64, self.dim, block_r=self.block_c, block_q=self.block_q
            )
            pred_local = np.argmax(sims, axis=1)
            best_score = sims[np.arange(sims.shape[0]), pred_local]
        else:
            pred_local, best_score = cascade_search(
                q_u64, self.class_u64, q_sketch, self.class_sketch,
                self.dim, self.sketch_bits, self.top_k,
                block_c=self.block_c, block_q=self.block_q, n_threads=self.n_threads
            )

        preds = self.labels[pred_local]
        if return_scores:
            return preds, best_score
        return preds

    # ------------------------------------------------------------------
    def _predict_ternary(self, query_float_vectors, zero_threshold, return_scores):
        if self.class_nz_u64 is None:
            raise RuntimeError("Panggil .fit() dahulu")

        nz, sg = encode_ternary(query_float_vectors, zero_threshold)
        q_nz_u64 = pack_to_u64(nz)
        q_sg_u64 = pack_to_u64(sg)

        # Catatan: cascade pruning utk mode ternary disederhanakan jadi exact
        # search cache-blocked (mode ternary biasanya dipakai saat n_classes
        # tidak sebesar mode binary, jadi manfaat pruning lebih kecil).
        dots = ternary_similarity_blocked(
            q_nz_u64, q_sg_u64, self.class_nz_u64, self.class_sign_u64,
            self.dim, block_r=self.block_c, block_q=self.block_q
        )
        pred_local = np.argmax(dots, axis=1)
        best_score = dots[np.arange(dots.shape[0]), pred_local]

        preds = self.labels[pred_local]
        if return_scores:
            return preds, best_score
        return preds

    # ------------------------------------------------------------------
    def memory_footprint_bytes(self) -> int:
        if self.mode == "binary":
            return 0 if self.class_u64 is None else self.class_u64.nbytes
        total = 0
        if self.class_nz_u64 is not None:
            total += self.class_nz_u64.nbytes
        if self.class_sign_u64 is not None:
            total += self.class_sign_u64.nbytes
        return total
