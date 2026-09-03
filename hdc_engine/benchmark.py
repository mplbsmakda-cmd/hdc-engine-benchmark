"""
benchmark.py
============
Utilitas pembanding waktu & akurasi antara HDCCascadeEngine vs BLAS float32
(matmul standar). Dipakai di tests/ dan bisa dipakai langsung oleh pengguna
untuk mengukur performa pada data mereka sendiri.
"""

import time
import numpy as np

from .engine import HDCCascadeEngine

__all__ = ["benchmark_vs_blas"]


def _timeit(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def benchmark_vs_blas(class_float_vectors: np.ndarray, query_float_vectors: np.ndarray,
                       true_labels: np.ndarray = None,
                       sketch_bits: int = 512, top_k: int = 50,
                       block_c: int = 256, block_q: int = 32, repeats: int = 3) -> dict:
    """
    Jalankan tiga metode pada data yang sama: BLAS float32, exact bit-packed
    (cache-blocked), dan HDC-Cascade -- kembalikan dict berisi waktu & akurasi
    (akurasi hanya dihitung jika true_labels diberikan).
    """
    n_classes, dim = class_float_vectors.shape

    engine_cascade = HDCCascadeEngine(dim=dim, mode="binary", sketch_bits=sketch_bits,
                                       top_k=top_k, block_c=block_c, block_q=block_q)
    engine_cascade.fit(class_float_vectors)

    engine_exact = HDCCascadeEngine(dim=dim, mode="binary", sketch_bits=sketch_bits,
                                     top_k=n_classes, block_c=block_c, block_q=block_q)
    engine_exact.fit(class_float_vectors)

    t_cascade = _timeit(lambda: engine_cascade.predict(query_float_vectors), repeats)
    t_exact = _timeit(lambda: engine_exact.predict(query_float_vectors), repeats)
    t_blas = _timeit(lambda: np.argmax(query_float_vectors @ class_float_vectors.T, axis=1), repeats)

    result = {
        "n_classes": n_classes,
        "n_queries": query_float_vectors.shape[0],
        "dim": dim,
        "time_blas_ms": t_blas * 1000,
        "time_exact_bitpacked_ms": t_exact * 1000,
        "time_cascade_ms": t_cascade * 1000,
        "speedup_cascade_vs_blas": t_blas / t_cascade,
        "speedup_cascade_vs_exact": t_exact / t_cascade,
        "memory_bytes_bitpacked": engine_cascade.memory_footprint_bytes(),
        "memory_bytes_float32": class_float_vectors.nbytes,
        "memory_savings_x": class_float_vectors.nbytes / max(engine_cascade.memory_footprint_bytes(), 1),
    }

    if true_labels is not None:
        pred_cascade = engine_cascade.predict(query_float_vectors)
        pred_exact = engine_exact.predict(query_float_vectors)
        pred_blas = np.argmax(query_float_vectors @ class_float_vectors.T, axis=1)
        result["accuracy_cascade"] = float((pred_cascade == true_labels).mean())
        result["accuracy_exact_bitpacked"] = float((pred_exact == true_labels).mean())
        result["accuracy_blas"] = float((pred_blas == true_labels).mean())

    return result
