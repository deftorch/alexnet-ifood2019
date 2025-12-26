# Report Filling Guide

Panduan untuk mengisi `FINAL_REPORT.md` dengan hasil eksperimen.

## 1. Persiapan

### File yang Diperlukan

Pastikan Anda memiliki file-file berikut setelah training:

```
checkpoints/
├── alexnet_baseline_history.json
├── alexnet_mod1_history.json
├── alexnet_mod2_history.json
└── alexnet_combined_history.json

evaluation_results/
├── alexnet_baseline_val_metrics.json
├── alexnet_mod1_val_metrics.json
├── alexnet_mod2_val_metrics.json
└── alexnet_combined_val_metrics.json

analysis_results/
├── training_curves_comparison.png
├── metrics_comparison.png
├── confusion_matrices.png
└── model_comparison_summary.csv
```

## 2. Mengisi Bagian-Bagian Report

### 2.1 Dataset Statistics

Jalankan di notebook:
```python
import pandas as pd

train_df = pd.read_csv('data/train_info.csv', header=None)
val_df = pd.read_csv('data/val_info.csv', header=None)
test_df = pd.read_csv('data/test_info.csv', header=None)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
```

### 2.2 Performance Table

Buka file metrics JSON dan isi tabel:

```python
import json

models = ['alexnet_baseline', 'alexnet_mod1', 'alexnet_mod2', 'alexnet_combined']

for model in models:
    with open(f'evaluation_results/{model}_val_metrics.json') as f:
        m = json.load(f)
    print(f"{model}:")
    print(f"  Accuracy: {m['accuracy']:.4f}")
    print(f"  Top-5: {m['top5_accuracy']:.4f}")
    print(f"  Macro F1: {m['macro_f1']:.4f}")
```

### 2.3 Menyisipkan Gambar

#### Dari Google Drive:
```markdown
![Training Curves](../analysis_results/training_curves_comparison.png)
```

#### Jika Export ke PDF:
Gunakan path relatif atau embed gambar inline.

### 2.4 Menulis Analisis

**Template untuk analisis:**

```markdown
### Effect of Batch Normalization

Berdasarkan hasil eksperimen, model dengan Batch Normalization (alexnet_mod1) 
menunjukkan [peningkatan/penurunan] accuracy sebesar [X]% dibandingkan baseline.

Hal ini disebabkan oleh:
1. [Alasan 1]
2. [Alasan 2]

Dari training curves terlihat bahwa [observasi tentang konvergensi].
```

## 3. Tips Penulisan

### Do's ✓
- Gunakan data aktual dari eksperimen
- Sertakan grafik untuk mendukung klaim
- Jelaskan mengapa tidak hanya apa
- Bandingkan antar model secara objektif

### Don'ts ✗
- Jangan buat klaim tanpa bukti
- Jangan salin paste tanpa memahami
- Hindari kesimpulan berlebihan
- Hindari placeholder [ISI] yang tertinggal

## 4. Checklist Final

Sebelum submit, pastikan:

- [ ] Semua [ISI] sudah diganti dengan data aktual
- [ ] Gambar sudah terpasang dengan benar
- [ ] Tabel memiliki data lengkap
- [ ] Analisis konsisten dengan data
- [ ] Referensi lengkap
- [ ] Format markdown benar
- [ ] Spell check

## 5. Export ke PDF

### Opsi 1: Pandoc
```bash
pandoc FINAL_REPORT.md -o FINAL_REPORT.pdf
```

### Opsi 2: VS Code Extension
Install "Markdown PDF" extension, lalu Right Click > Markdown PDF: Export (pdf)

### Opsi 3: Google Docs
1. Copy markdown ke Google Docs
2. Format ulang jika perlu
3. Download as PDF

## 6. Contoh Pengisian

### Sebelum:
```markdown
| Baseline | [ISI] | [ISI] | [ISI] |
```

### Sesudah:
```markdown
| Baseline | 0.4523 | 0.7891 | 0.4312 |
```

---

**Good luck dengan report Anda! 🎓**
