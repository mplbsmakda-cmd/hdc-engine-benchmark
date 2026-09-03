# hdc_engine — Engine Core AI Berbasis CPU (Bit-Packed + Cascade Pruning)

Engine inferensi/klasifikasi/retrieval berbasis representasi hyperdimensional
binary/ternary, dioptimasi untuk CPU lewat kombinasi:
1. **Representasi bit-packed** (uint64 + popcount native) — ~32x lebih hemat memori vs float32.
2. **Kernel cache-blocked** — mencegah ledakan memory-bandwidth pada broadcast penuh.
3. **Cascade pruning** — sketch murah menyaring kandidat sebelum verifikasi presisi penuh,
   mengubah kompleksitas dari O(Q·C·D) menjadi O(Q·C·d_sketch) + O(Q·K·D).

Hasil benchmark terukur (lihat `hdc_engine/tests/test_benchmark.py` dan bagian
"Cara Menjalankan Benchmark" di bawah, DIM=10000, 5000 kelas, noise 25%):
**2.8x–9.5x lebih cepat dari BLAS float32** (bervariasi antar run tergantung
beban CPU/cache saat itu — jalankan sendiri untuk angka pada hardware Anda),
**~32x lebih hemat memori**, akurasi identik dengan exact search selama ada
sinyal nyata pada data (bukan pencarian di antara vektor acak independen).

## Struktur Modul

```
hdc_engine/
├── __init__.py       # API publik: HDCCascadeEngine, encode_*, kernel, binding, benchmark
├── encoding.py        # float -> bit (binary/ternary), packing ke uint64
├── kernel.py           # similarity bitwise cache-blocked (binary & ternary)
├── cascade.py          # pruning dua tahap (sketch -> verifikasi), opsional multi-thread
├── binding.py           # operasi kompositori HDC: bind, unbind, bundle, permute
├── engine.py             # HDCCascadeEngine — API utama yang menyatukan semua modul
├── benchmark.py           # utilitas pembanding waktu/akurasi/memori vs BLAS
└── tests/
    ├── test_encoding.py
    ├── test_kernel.py
    ├── test_binding.py
    ├── test_engine.py
    └── test_benchmark.py
```

Prinsip modular: setiap file punya satu tanggung jawab. `engine.py` adalah satu-satunya
tempat yang menyatukan `encoding` + `kernel` + `cascade` menjadi API siap pakai —
modul lain bisa dipakai independen (mis. hanya butuh kernel similarity tanpa cascade,
atau hanya butuh operasi `bind`/`bundle` untuk membangun representasi terstruktur sendiri).

## Kenapa Cascade Bekerja (Ringkasan Riset)

Representasi hyperdimensional (dimensi tinggi, misal 10.000 bit) bersifat *redundan* —
informasi tersebar merata di semua dimensi, bukan terkonsentrasi di beberapa bit saja.
Konsekuensinya: subsample kecil (misal 512 bit pertama = 5% dari total) sudah cukup
representatif untuk *menyaring* kandidat yang jelas tidak relevan, TANPA menghitung
similarity penuh. Kandidat yang lolos penyaringan (top-K) baru dihitung presisi penuh.

Ini terbukti empiris pada data dengan sinyal nyata (query = versi noisy dari kelas asli):
akurasi cascade identik dengan exact search bahkan pada noise 35%. **Catatan penting**:
ini TIDAK berlaku untuk kasus tanpa sinyal (mis. mencari argmax di antara vektor-vektor
acak yang saling independen) — cascade mengandalkan adanya margin nyata antara jawaban
benar dan yang salah, bukan sekadar redundansi bit semata.

## Instalasi

Tidak ada dependency eksternal selain NumPy (>= 2.0, karena memakai `np.bitwise_count`
yang baru tersedia di NumPy 2.0+).

```bash
pip install numpy>=2.0
```

Lalu cukup taruh folder `hdc_engine/` di project Anda, atau install lokal:

```bash
cd hdc_engine_project/
pip install -e .   # jika memakai setup.py/pyproject (opsional, lihat catatan di bawah)
```

Tanpa setup.py, cukup pastikan direktori yang berisi folder `hdc_engine/` ada di `PYTHONPATH`.

## Cara Pakai Cepat

```python
import numpy as np
from hdc_engine import HDCCascadeEngine

# Data: 5000 kelas, dimensi 10000
class_vectors = np.random.randn(5000, 10000).astype(np.float32)
labels = np.array([f"kelas_{i}" for i in range(5000)])

engine = HDCCascadeEngine(dim=10000, mode="binary", sketch_bits=512, top_k=50)
engine.fit(class_vectors, labels=labels)

query = np.random.randn(10, 10000).astype(np.float32)
predictions = engine.predict(query)
print(predictions)
```

### Mode Ternary (BitNet-style, mendukung nilai nol eksplisit)

```python
engine = HDCCascadeEngine(dim=10000, mode="ternary", block_c=128, block_q=16)
engine.fit(class_vectors, zero_threshold=0.3)   # elemen |x|<0.3 dianggap 0
predictions = engine.predict(query, zero_threshold=0.3)
```

### Operasi Kompositori (bind/bundle/permute)

```python
from hdc_engine import bind, bundle, permute
import numpy as np

warna = np.random.randint(0, 2, 10000, dtype=np.uint8)
merah = np.random.randint(0, 2, 10000, dtype=np.uint8)

konsep = bind(warna, merah)          # "warna terikat pada merah"
gabungan = bundle(np.stack([konsep, warna]))  # superposisi banyak konsep
sekuens = bundle([permute(v, i) for i, v in enumerate([warna, merah])])  # encode urutan
```

## Cara Menjalankan Test

```bash
cd hdc_engine_project/
python3 -m unittest discover -s hdc_engine/tests -p "test_*.py" -v
```

32 test mencakup: korespondensi matematis kernel vs brute-force NumPy, invertibility
operasi binding, akurasi engine di bawah noise, konsistensi cascade vs exact search,
konsistensi single-thread vs multi-thread, dan validasi input (dimensi salah, fit belum
dipanggil, parameter tidak valid).

## Cara Menjalankan Benchmark

```bash
python3 -c "
import numpy as np
from hdc_engine import benchmark_vs_blas

rng = np.random.default_rng(0)
dim, n_classes, n_queries = 10000, 5000, 200
class_float = rng.standard_normal((n_classes, dim)).astype(np.float32)
true_labels = rng.integers(0, n_classes, size=n_queries)
class_bits = (class_float[true_labels] > 0).astype(np.uint8)
flip = rng.random(class_bits.shape) < 0.25
class_bits[flip] = 1 - class_bits[flip]
query_float = class_bits.astype(np.float32) * 2 - 1

result = benchmark_vs_blas(class_float, query_float, true_labels=true_labels)
for k, v in result.items():
    print(f'{k}: {v}')
"
```

## Batasan Jujur (Baca Sebelum Produksi)

- **Ini implementasi NumPy**, bukan kernel C++ native. Untuk performa maksimal di
  produksi, ganti kernel popcount dengan intrinsic SIMD (`POPCNT`/`VPOPCNTQ`) di C/C++
  seperti pendekatan `bitnet.cpp` — NumPy murni tidak bisa menutup gap sepenuhnya untuk
  *exact* dense search (BLAS float32 tetap lebih cepat ~2.5x untuk operasi identik;
  keunggulan cascade muncul karena mengubah bentuk masalahnya, bukan mengalahkan BLAS
  di operasi yang sama).
- **`sketch_bits`/`top_k` adalah trade-off akurasi-vs-kecepatan** — selalu validasi
  pada data Anda sendiri dengan membandingkan ke mode exact (`top_k >= n_classes`).
- **Mode ternary belum memakai cascade pruning** (masih exact search cache-blocked) —
  cocok untuk n_classes yang tidak terlalu besar; kombinasi ternary+cascade adalah
  pengembangan lanjutan yang belum diimplementasikan di versi ini.
- Cascade mengasumsikan **ada sinyal nyata** antara query dan jawaban benar. Untuk
  kasus nearest-neighbor pada data yang benar-benar acak/tanpa struktur, akurasi
  cascade bisa jauh lebih rendah dari exact search — selalu uji dengan data representatif.

## Versi

`hdc_engine.__version__ == "1.0.0"`


<!-- CI trigger test -->

<!-- retrigger -->
