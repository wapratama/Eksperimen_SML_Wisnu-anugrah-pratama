# Eksperimen_SML_Wisnu-anugrah-pratama

Repository ini berisi tahapan eksperimen dan otomatisasi preprocessing dataset
**Facies Classification dari Wireline Logs** (Hugoton & Panoma Fields, Kansas)
sebagai bagian dari **Proyek Akhir Kelas Membangun Sistem Machine
Learning (MSML) Dicoding** untuk Kriteria 1.

## Struktur Repository

```
Eksperimen_SML_Wisnu-anugrah-pratama/
├── .github/workflows/preprocessing.yml       # CI: otomatisasi preprocessing
├── facies_raw.csv                            # Dataset raw dari sumber asli (3,232 baris)
├── preprocessing/
│   └── Eksperimen_Wisnu-anugrah-pratama.ipynb    # Notebook eksperimen manual
│   └── automate_Wisnu-anugrah-pratama.py         # Script otomatisasi preprocessing
│   └── facies_preprocessing.csv                  # Output hasil preprocessing
└── README.md
```
*Catatan: Tersimpan pula beberapa gambar hasil EDA dari notebook di folder `preprocessing/`*

## Dataset

**Facies Classification dari Wireline Log** — 3.232 baris, 11 kolom, dari
9 sumur di Hugoton & Panoma Fields (Council Grove gas reservoir), Kansas.
Digunakan dalam SEG 2016 Machine Learning Contest.

Data diunduh otomatis saat runtime, bukan file statis:
`https://raw.githubusercontent.com/seg/2016-ml-contest/master/training_data.csv`

Sumber: [seg/2016-ml-contest](https://github.com/seg/2016-ml-contest)

Referensi: Dubois et al. (2007); Hall, B. (2016) — *Facies classification
using machine learning*, The Leading Edge.

**Target**: `Facies` (9 kelas lithofacies)

| Kode | Label | Deskripsi |
|---|---|---|
| 1 | SS | Nonmarine sandstone |
| 2 | CSiS | Nonmarine coarse siltstone |
| 3 | FSiS | Nonmarine fine siltstone |
| 4 | SiSh | Marine siltstone & shale |
| 5 | MS | Mudstone |
| 6 | WS | Wackestone |
| 7 | D | Dolomite |
| 8 | PS | Packstone-grainstone |
| 9 | BS | Phylloid-algal bafflestone |

**Fitur prediktor**: `GR` (gamma ray), `ILD_log10` (resistivity), `DeltaPHI`
(porosity difference), `PHIND` (average porosity), `PE` (photoelectric
effect), `NM_M` (indikator nonmarine/marine), `RELPOS` (posisi relatif),
`Depth`, `Formation`, `Well Name`.

## Tahapan Preprocessing

1. **Download otomatis**: dataset diunduh langsung dari sumber asli GitHub
   SEG 2016 ML Contest menggunakan `urllib.request` — bukan file CSV statis
   yang sudah tersimpan, memastikan data selalu terbaru dari sumbernya
2. **EDA**: distribusi kelas (class imbalance ~7.5x), crossplot GR vs
   resistivity per facies, boxplot log curve per facies
3. **Handling missing values**: imputasi defensif median (dataset asli bersih)
4. **Encoding**: `LabelEncoder` untuk `Formation` (14 kategori) dan
   `Well Name` (8 kategori)
5. **Outlier handling**: capping log curve numerik menggunakan metode IQR
6. **Feature scaling**: `StandardScaler` untuk `Depth`, `GR`, `ILD_log10`,
   `DeltaPHI`, `PHIND`, `PE`

## Catatan Domain

Class imbalance pada kelas Dolomite (98 sampel) vs Coarse Siltstone (738
sampel) menjadi pertimbangan penting di tahap modelling, dimana menambahkan metrik F1-score
per kelas dan macro-average lebih informatif dibanding akurasi semata.

## Cara Menjalankan

### Setup awal (menggunakan uv)

```bash
# Install uv jika belum ada
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependency sesuai pyproject.toml + uv.lock
uv sync
```

### Manual (notebook)
Buka `preprocessing/Eksperimen_Wisnu-anugrah-pratama.ipynb` di Jupyter/Colab dan jalankan seluruh cell
secara berurutan.

### Otomatis (script)

Dijalankan dari **root repository**:

```bash
uv run python preprocessing/automate_Wisnu-anugrah-pratama.py --input facies_raw.csv --output preprocessing/facies_preprocessing.csv
```

Tambahkan flag `--force-download` untuk selalu mengunduh ulang dataset dari
sumber asli meskipun file lokal sudah ada:

```bash
uv run python preprocessing/automate_Wisnu-anugrah-pratama.py --force-download
```

### CI/CD (GitHub Actions)
Workflow `preprocessing.yml` berjalan otomatis setiap kali ada push yang
menyentuh `facies_raw.csv`, `automate_NamaSiswa.py`, `pyproject.toml`, atau
`uv.lock` — dan dapat dipicu manual melalui tab **Actions** di GitHub.

## Output

File `preprocessing/facies_preprocessing.csv` berisi dataset yang sudah:
- Bebas missing value
- Ter-encode (numerik penuh)
- Ter-scale (StandardScaler)
- Siap digunakan langsung untuk training model multiclass (lihat repo
  `Membangun_model`)
