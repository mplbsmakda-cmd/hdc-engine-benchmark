"""
hdc_engine
==========
Engine core AI berbasis CPU: representasi bit-packed hyperdimensional
(binary/ternary) dengan cascade pruning di atas kernel cache-blocked.

Quick start
-----------
    from hdc_engine import HDCCascadeEngine

    engine = HDCCascadeEngine(dim=10000, mode="binary", sketch_bits=512, top_k=50)
    engine.fit(class_float_vectors, labels=my_labels)
    predictions = engine.predict(query_float_vectors)
"""

from .engine import HDCCascadeEngine
from .encoding import encode_bipolar, encode_ternary, pack_to_u64
from .kernel import hamming_similarity_blocked, ternary_similarity_blocked
from .cascade import cascade_search
from .binding import bind, unbind, bundle, permute
from .benchmark import benchmark_vs_blas

__all__ = [
    "HDCCascadeEngine",
    "encode_bipolar", "encode_ternary", "pack_to_u64",
    "hamming_similarity_blocked", "ternary_similarity_blocked",
    "cascade_search",
    "bind", "unbind", "bundle", "permute",
    "benchmark_vs_blas",
]

__version__ = "1.0.0"
