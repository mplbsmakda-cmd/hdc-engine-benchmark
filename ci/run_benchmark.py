"""Benchmark script untuk GitHub Actions CI/CD. Adaptif: auto-detect RAM."""

import json
import os
import sys
import time
import gc
import traceback
import platform

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hdc_engine import HDCCascadeEngine


def detect_max_scale(default_dim=20000, default_classes=20000, safety_factor=0.7):
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem_avail_kb = None
        for line in lines:
            if line.startswith("MemAvailable:"):
                mem_avail_kb = int(line.split()[1])
                break
        if mem_avail_kb is None:
            return default_dim, default_classes
        ram_bytes = mem_avail_kb * 1024
        max_cells = (ram_bytes * safety_factor) / 12.0
        max_dim = int(max_cells**0.5)
        max_dim = max(5000, (max_dim // 5000) * 5000)
        max_dim = min(max_dim, 50000)
        return max_dim, max_dim
    except Exception:
        return default_dim, default_classes


def make_data(dim, n_classes, n_queries, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    class_float = rng.standard_normal((n_classes, dim), dtype=np.float32)
    true_labels = rng.integers(0, n_classes, size=n_queries)
    class_bits = (class_float[true_labels] > 0).astype(np.uint8)
    flip = rng.random(class_bits.shape, dtype=np.float32) < noise
    class_bits[flip] = 1 - class_bits[flip]
    query_float = (class_bits.astype(np.float32) * 2.0) - 1.0
    return class_float, query_float, true_labels


def mem_mb(x):
    if x is None:
        return 0.0
    if isinstance(x, (int, np.integer)):
        return float(x) / (1024 * 1024)
    return float(x.nbytes) / (1024 * 1024)


def run_phase(name, dim, n_classes, n_queries, noise=0.25,
              mode="binary", top_k=50, sketch_bits=512,
              n_threads=1, repeats=2, run_blas=True):
    print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
    print(f"  dim={dim}  n_classes={n_classes}  n_queries={n_queries}  noise={noise:.0%}", flush=True)
    print(f"  mode={mode}  top_k={top_k}  sketch_bits={sketch_bits}  n_threads={n_threads}", flush=True)

    t0 = time.perf_counter()
    class_float, query_float, true_labels = make_data(dim, n_classes, n_queries, noise=noise, seed=42)
    t_data = time.perf_counter() - t0
    print(f"  data gen: {t_data*1000:.0f} ms", flush=True)

    result = {"phase": name, "dim": dim, "n_classes": n_classes, "n_queries": n_queries,
              "noise": noise, "mode": mode, "top_k": top_k,
              "sketch_bits": sketch_bits, "n_threads": n_threads, "repeats": repeats,
              "data_gen_ms": t_data * 1000}

    try:
        t0 = time.perf_counter()
        engine_cascade = HDCCascadeEngine(dim=dim, mode=mode, sketch_bits=sketch_bits,
                                          top_k=top_k, n_threads=n_threads)
        engine_cascade.fit(class_float)
        t_fit = time.perf_counter() - t0
        print(f"  cascade.fit: {t_fit*1000:.0f} ms  mem_bitpacked={mem_mb(engine_cascade.memory_footprint_bytes())}MB", flush=True)

        times_cascade = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            pred_cascade = engine_cascade.predict(query_float)
            times_cascade.append(time.perf_counter() - t0)
        t_cascade = min(times_cascade)
        acc_cascade = float((pred_cascade == true_labels).mean())
        del pred_cascade

        t0 = time.perf_counter()
        engine_exact = HDCCascadeEngine(dim=dim, mode=mode, sketch_bits=sketch_bits,
                                       top_k=n_classes, n_threads=n_threads)
        engine_exact.fit(class_float)
        times_exact = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            pred_exact = engine_exact.predict(query_float)
            times_exact.append(time.perf_counter() - t0)
        t_exact = min(times_exact)
        acc_exact = float((pred_exact == true_labels).mean())
        del pred_exact, engine_exact
        gc.collect()

        t_blas, acc_blas = None, None
        if run_blas:
            times_blas = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                pred_blas = np.argmax(query_float @ class_float.T, axis=1)
                times_blas.append(time.perf_counter() - t0)
            t_blas = min(times_blas)
            acc_blas = float((pred_blas == true_labels).mean())
            del pred_blas

        result.update({
            "time_blas_ms": (t_blas * 1000) if t_blas else None,
            "time_exact_bitpacked_ms": t_exact * 1000,
            "time_cascade_ms": t_cascade * 1000,
            "speedup_cascade_vs_blas": (t_blas / t_cascade) if t_blas else None,
            "speedup_cascade_vs_exact": t_exact / t_cascade,
            "memory_bytes_bitpacked": engine_cascade.memory_footprint_bytes(),
            "memory_bytes_float32": class_float.nbytes,
            "memory_savings_x": class_float.nbytes / max(engine_cascade.memory_footprint_bytes(), 1),
            "fit_ms": t_fit * 1000,
            "accuracy_cascade": acc_cascade,
            "accuracy_exact_bitpacked": acc_exact,
            "accuracy_blas": acc_blas,
            "ok": True,
        })
        print(f"  RESULTS: cascade={result['time_cascade_ms']:.1f}ms  exact={result['time_exact_bitpacked_ms']:.1f}ms  blas={result['time_blas_ms'] if t_blas else 'n/a'}ms", flush=True)
        if t_blas:
            print(f"           speedup cascade vs blas = {result['speedup_cascade_vs_blas']:.2f}x", flush=True)
        print(f"           speedup cascade vs exact = {result['speedup_cascade_vs_exact']:.2f}x", flush=True)
        if acc_blas is not None:
            print(f"           acc: cascade={acc_cascade:.4f}  exact={acc_exact:.4f}  blas={acc_blas:.4f}", flush=True)
        else:
            print(f"           acc: cascade={acc_cascade:.4f}  exact={acc_exact:.4f}", flush=True)
        print(f"           mem savings: {result['memory_savings_x']:.1f}x", flush=True)
    except Exception as e:
        result["ok"] = False
        result["error"] = repr(e)
        result["traceback"] = traceback.format_exc()
        print(f"  FAILED: {e!r}", flush=True)
    finally:
        try:
            del class_float, query_float, true_labels, engine_cascade
        except Exception:
            pass
        gc.collect()
    return result


def main():
    all_results = []
    print("#" * 70, flush=True)
    print("# hdc_engine BENCHMARK on GitHub Actions", flush=True)
    print(f"# Python: {platform.python_version()}", flush=True)
    print(f"# NumPy:  {np.__version__}", flush=True)
    print(f"# CPU:    {os.cpu_count()} cores", flush=True)
    print("#" * 70, flush=True)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "w") as f:
            f.write("# hdc_engine Benchmark Results\n\n")
            f.write(f"- **Python:** {platform.python_version()}  \n")
            f.write(f"- **NumPy:**  {np.__version__}  \n")
            f.write(f"- **CPU cores:** {os.cpu_count()}  \n")
            f.write(f"- **OS:** {platform.platform()}  \n\n")

    max_dim = int(os.environ.get("INPUT_MAX_DIM", "0"))
    max_classes = int(os.environ.get("INPUT_MAX_CLASSES", "0"))
    max_queries = int(os.environ.get("INPUT_MAX_QUERIES", "500"))
    if max_dim <= 0 or max_classes <= 0:
        max_dim, max_classes = detect_max_scale()
        print(f"  auto-detect max_scale: {max_dim}x{max_classes}", flush=True)
    else:
        print(f"  using input max_scale: {max_dim}x{max_classes}", flush=True)

    all_results.append(run_phase(
        "Phase 1: Baseline README (DIM=10000, C=5000, Q=200)",
        dim=10000, n_classes=5000, n_queries=200, noise=0.25,
        top_k=50, sketch_bits=512, repeats=3,
    ))
    all_results.append(run_phase(
        "Phase 2: Push dimensi",
        dim=max_dim, n_classes=5000, n_queries=200, noise=0.25,
        top_k=50, sketch_bits=512, repeats=3,
    ))
    all_results.append(run_phase(
        "Phase 3: Push kelas",
        dim=10000, n_classes=max_classes, n_queries=200, noise=0.25,
        top_k=50, sketch_bits=512, repeats=2,
    ))
    all_results.append(run_phase(
        f"Phase 4: MAX SCALE binary ({max_dim}x{max_classes}, Q={max_queries})",
        dim=max_dim, n_classes=max_classes, n_queries=max_queries, noise=0.25,
        top_k=100, sketch_bits=512, repeats=3,
    ))
    n_threads = min(os.cpu_count() or 2, 4)
    all_results.append(run_phase(
        f"Phase 5: MAX SCALE multi-thread (n_threads={n_threads})",
        dim=max_dim, n_classes=max_classes, n_queries=max_queries, noise=0.25,
        top_k=100, sketch_bits=512, n_threads=n_threads, repeats=2,
    ))
    all_results.append(run_phase(
        "Phase 6: MAX SCALE ternary mode",
        dim=max_dim, n_classes=max_classes, n_queries=min(max_queries, 200), noise=0.25,
        mode="ternary", top_k=100, sketch_bits=512, repeats=2,
    ))
    for noise in [0.05, 0.15, 0.25, 0.35, 0.45]:
        all_results.append(run_phase(
            f"Phase 7 noise={noise:.0%}: MAX SCALE",
            dim=max_dim, n_classes=max_classes, n_queries=200, noise=noise,
            top_k=100, sketch_bits=512, repeats=2,
        ))
    for top_k in [10, 50, 100, 500, 2000]:
        all_results.append(run_phase(
            f"Phase 8 top_k={top_k}: MAX SCALE",
            dim=max_dim, n_classes=max_classes, n_queries=200, noise=0.25,
            top_k=top_k, sketch_bits=512, repeats=2,
        ))
    for sb in [128, 256, 512, 1024, 2048]:
        all_results.append(run_phase(
            f"Phase 9 sketch_bits={sb}: MAX SCALE",
            dim=max_dim, n_classes=max_classes, n_queries=200, noise=0.25,
            top_k=100, sketch_bits=sb, repeats=2,
        ))

    output_dir = os.environ.get("GITHUB_WORKSPACE", ".")
    out_json = os.path.join(output_dir, "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON saved: {out_json}", flush=True)

    out_md = os.path.join(output_dir, "benchmark_results.md")
    with open(out_md, "w") as f:
        f.write("# hdc_engine Benchmark Results (CI)\n\n")
        f.write(f"- **Runner:** Python {platform.python_version()}, NumPy {np.__version__}, {os.cpu_count()} cores\n")
        f.write(f"- **Max scale tested:** {max_dim} dim x {max_classes} classes = {max_dim*max_classes:,} elements\n\n")
        f.write("## Main Results (Phase 1-6)\n\n")
        f.write("| Phase | dim | classes | queries | mode | t_cascade (ms) | t_exact (ms) | t_blas (ms) | speedup vs BLAS | acc cascade | acc exact | mem savings |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        main_phases = [r for r in all_results if not any(s in r["phase"] for s in ["noise=", "top_k=", "sketch_bits="])]
        for r in main_phases:
            t_c = f"{r.get('time_cascade_ms', 0):.1f}" if r.get("ok") else "FAIL"
            t_e = f"{r.get('time_exact_bitpacked_ms', 0):.1f}" if r.get("ok") else "FAIL"
            t_b = f"{r.get('time_blas_ms', 0):.1f}" if r.get("ok") and r.get("time_blas_ms") else "-"
            sp = f"{r.get('speedup_cascade_vs_blas', 0):.2f}x" if r.get("speedup_cascade_vs_blas") else "-"
            ac = f"{r.get('accuracy_cascade', 0)*100:.2f}%" if r.get("ok") else "FAIL"
            ae = f"{r.get('accuracy_exact_bitpacked', 0)*100:.2f}%" if r.get("ok") else "FAIL"
            ms = f"{r.get('memory_savings_x', 0):.1f}x" if r.get("ok") else "-"
            f.write(f"| {r['phase'].split(':')[0]} | {r['dim']} | {r['n_classes']:,} | {r['n_queries']} | {r['mode']} | {t_c} | {t_e} | {t_b} | {sp} | {ac} | {ae} | {ms} |\n")

        if summary_path:
            with open(summary_path, "a") as sf:
                sf.write(open(out_md).read())
                sf.write("\n## All Raw Results\n\n")
                sf.write("See `benchmark_results.json` artifact for full details.\n")

    print(f"Markdown saved: {out_md}", flush=True)
    print(f"\nTotal phases: {len(all_results)}  OK: {sum(1 for r in all_results if r.get('ok'))}  FAIL: {sum(1 for r in all_results if not r.get('ok'))}", flush=True)


if __name__ == "__main__":
    main()
