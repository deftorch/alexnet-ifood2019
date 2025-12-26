# 📚 Panduan Lengkap: AlexNet iFood2019 dengan Google Colab

## Daftar Isi
1. [Persiapan Awal](#1-persiapan-awal)
2. [Setup Google Drive](#2-setup-google-drive)
3. [Download Dataset](#3-download-dataset)
4. [Menjalankan Training](#4-menjalankan-training)
5. [Workflow Tim 4 Orang](#5-workflow-tim-4-orang)
6. [Menggabungkan Hasil](#6-menggabungkan-hasil)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Persiapan Awal

### 1.1 Prasyarat
- Akun Google (untuk Colab dan Drive)
- Akun GitHub (repository sudah ada di: https://github.com/deftorch/alexnet-ifood2019)

### 1.2 Struktur Repository
```
alexnet-ifood2019/
├── src/                          # Source code
│   ├── models/alexnet.py         # 4 varian model AlexNet
│   ├── data_loader.py            # Data loading
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation
│   └── analysis.py               # Analisis komparatif
│
├── notebooks/                    # Jupyter Notebooks
│   ├── 00_setup_and_verification.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_train_all_models.ipynb
│   ├── 04_analysis_and_visualization.ipynb
│   ├── 05_merge_team_results.ipynb
│   │
│   ├── train_member1_baseline.ipynb   # Untuk Member 1
│   ├── train_member2_mod1.ipynb       # Untuk Member 2
│   ├── train_member3_mod2.ipynb       # Untuk Member 3
│   └── train_member4_combined.ipynb   # Untuk Member 4
│
└── docs/                         # Dokumentasi
    ├── PAPER_SUMMARY.md
    └── FINAL_REPORT.md
```

---

## 2. Setup Google Drive

### 2.1 Buat Folder Struktur

Di Google Drive, buat folder dengan struktur berikut:

```
My Drive/
└── AlexNet_iFood2019/
    ├── dataset/               # ← Upload dataset di sini
    │   ├── train_images/
    │   ├── val_images/
    │   ├── test_images/
    │   ├── train_info.csv
    │   ├── val_info.csv
    │   ├── test_info.csv
    │   └── class_list.txt
    │
    ├── checkpoints/           # Model checkpoint (otomatis terisi)
    ├── evaluation_results/    # Hasil evaluasi (otomatis terisi)
    └── analysis_results/      # Grafik & analisis (otomatis terisi)
```

### 2.2 Cara Membuat Folder

1. Buka [Google Drive](https://drive.google.com)
2. Klik **+ New** → **New folder**
3. Beri nama `AlexNet_iFood2019`
4. Buka folder tersebut
5. Buat subfolder: `dataset`, `checkpoints`, `evaluation_results`, `analysis_results`

### 2.3 Share Folder dengan Tim (Penting!)

Jika bekerja dalam tim:
1. Klik kanan folder `AlexNet_iFood2019`
2. Pilih **Share**
3. Tambahkan email semua anggota tim
4. Berikan akses **Editor**

---

## 3. Download Dataset

### 3.1 Sumber Dataset

Dataset iFood 2019 dapat diunduh dari:
- **Official**: https://github.com/karansikka1/iFood_2019
- **Kaggle**: https://www.kaggle.com/c/ifood-2019-fgvc6/data

### 3.2 Upload ke Google Drive

1. Download dataset (~100GB)
2. Extract file-file berikut:
   - `train_images/` (folder berisi gambar training)
   - `val_images/` (folder berisi gambar validasi)
   - `test_images/` (folder berisi gambar test)
   - `train_info.csv`
   - `val_info.csv`
   - `test_info.csv`
   - `class_list.txt`
3. Upload ke `My Drive/AlexNet_iFood2019/dataset/`

### 3.3 Verifikasi Dataset

Pastikan struktur benar:
```
AlexNet_iFood2019/dataset/
├── train_images/     (berisi ~118.000 gambar)
├── val_images/       (berisi ~12.000 gambar)
├── test_images/      (berisi ~28.000 gambar)
├── train_info.csv
├── val_info.csv
├── test_info.csv
└── class_list.txt
```

---

## 4. Menjalankan Training

### 4.1 Buka Google Colab

1. Pergi ke [Google Colab](https://colab.research.google.com)
2. Klik **File** → **Open notebook**
3. Pilih tab **GitHub**
4. Masukkan repo: `deftorch/alexnet-ifood2019`
5. Pilih notebook yang ingin dibuka

### 4.2 Aktifkan GPU

**PENTING**: Sebelum run, pastikan GPU aktif!

1. Klik **Runtime** → **Change runtime type**
2. Pilih **GPU** di Hardware accelerator
3. Klik **Save**

### 4.3 Urutan Notebook (Individual)

Jika bekerja sendiri, jalankan dalam urutan:

| Step | Notebook | Fungsi |
|------|----------|--------|
| 1 | `00_setup_and_verification.ipynb` | Setup awal & verifikasi |
| 2 | `01_data_exploration.ipynb` | Eksplorasi dataset |
| 3 | `03_train_all_models.ipynb` | Training semua model sekaligus |
| 4 | `04_analysis_and_visualization.ipynb` | Analisis hasil |

### 4.4 Tips Menjalankan Notebook

1. **Jalankan cell per cell** dengan `Shift + Enter`
2. **Atau jalankan semua** dengan `Runtime → Run all`
3. **Jangan tutup browser** selama training
4. Kalau disconnect, cukup jalankan ulang dari awal (checkpoint tersimpan)

---

## 5. Workflow Tim 4 Orang

### 5.1 Pembagian Tugas

| Member | Notebook | Model | Deskripsi |
|--------|----------|-------|-----------|
| Member 1 | `train_member1_baseline.ipynb` | AlexNet Baseline | Arsitektur original |
| Member 2 | `train_member2_mod1.ipynb` | AlexNet Mod1 | + Batch Normalization |
| Member 3 | `train_member3_mod2.ipynb` | AlexNet Mod2 | + Enhanced Dropout |
| Member 4 | `train_member4_combined.ipynb` | AlexNet Combined | Gabungan semua |

### 5.2 Langkah untuk Setiap Member

1. **Buka Colab** dan login dengan akun Google yang sudah di-share folder
2. **Buka notebook** sesuai assignment dari GitHub
3. **Aktifkan GPU** (Runtime → Change runtime type → GPU)
4. **Jalankan cell pertama (SETUP)** - akan mount Google Drive
5. **Jalankan cell TRAINING** - proses ~2-3 jam per model
6. **Jalankan cell EVALUATION** - evaluasi model
7. **Jalankan cell LIHAT HASIL** - visualisasi

### 5.3 Timeline Recommended

```
Hari 1:
├── Semua member: Setup & verifikasi (30 menit)
├── Upload dataset ke Drive bersama (jika belum)
└── Mulai training masing-masing

Hari 2:
├── Lanjutkan training jika belum selesai
├── Evaluation masing-masing model
└── Member leader: Merge results

Hari 3:
└── Buat laporan bersama
```

### 5.4 Sinkronisasi Hasil

Karena semua member terkoneksi ke folder Drive yang sama:
- ✅ Checkpoint otomatis tersimpan ke folder bersama
- ✅ Hasil evaluasi juga tersimpan bersama
- ✅ Tidak perlu transfer file manual

---

## 6. Menggabungkan Hasil

### 6.1 Kapan Merge?

**SETELAH** semua 4 member selesai training dan evaluation.

### 6.2 Cara Merge

1. Buka `05_merge_team_results.ipynb`
2. Jalankan semua cell
3. Notebook akan otomatis:
   - Mengecek kelengkapan file
   - Membuat tabel perbandingan
   - Generate grafik perbandingan
   - Membuat summary untuk laporan

### 6.3 Output yang Dihasilkan

```
analysis_results/
├── team_results_comparison.csv       # Tabel perbandingan
├── team_results_summary.md           # Summary untuk laporan
├── all_models_training_curves.png    # Grafik training
└── all_models_metrics_comparison.png # Grafik metrics
```

---

## 7. Troubleshooting

### ❌ "Drive not mounted"
**Solusi**: Jalankan ulang cell mount drive, klik link authorize

### ❌ "CUDA out of memory"
**Solusi**: 
- Kurangi `BATCH_SIZE` dari 128 jadi 64 atau 32
- Restart runtime: Runtime → Restart runtime

### ❌ "Module not found"
**Solusi**: Pastikan sudah menjalankan cell SETUP yang install dependencies

### ❌ "File not found: checkpoints/..."
**Solusi**: Training belum selesai atau checkpoint belum tersimpan. Pastikan training berjalan sampai selesai.

### ❌ Colab Disconnect
**Solusi**:
1. Training akan resume dari checkpoint terakhir
2. Jalankan ulang dari cell SETUP
3. Training script akan otomatis load checkpoint

### ❌ Training Lambat
**Tips**:
- Pastikan GPU aktif (bukan CPU)
- Gunakan `BATCH_SIZE = 128` jika GPU cukup
- Jangan buka tab lain terlalu banyak

### ❌ "Permission denied" saat akses Drive
**Solusi**: 
- Pastikan folder sudah di-share dengan akun Google yang digunakan
- Coba mount ulang Drive

---

## 📞 Kontak

Repository: https://github.com/deftorch/alexnet-ifood2019

---

**Good luck dengan project kalian! 🚀**
