# Model 3 — Prediksi Restock (RandomForest)

Model machine learning yang belajar pola penjualan harian (weekend, awal
bulan/gajian, tren per produk) untuk memprediksi permintaan 7 hari ke depan,
lalu menghitung: produk mana yang perlu direstock, berapa hari lagi stok
habis, dan saran jumlah belanja.

## Status

- Model: RandomForestRegressor (scikit-learn), di-train dari 0
- Evaluasi: MAE 15.6% lebih baik dari baseline rata-rata 7 hari
  (diuji time-based di 30 hari terakhir yang tidak dipakai training)
- **Catatan penting:** model saat ini dilatih dengan DATA DUMMY (sintetis,
  pola menyerupai toko sembako). Prediksi belum menggambarkan toko asli.
  Setelah sistem transaksi suara berjalan beberapa minggu dan data penjualan
  harian per produk terkumpul, train ulang dengan data asli (lihat bawah) —
  tidak ada kode yang perlu diubah.

## File

| File | Fungsi |
|---|---|
| generate_dummy_data.py | bikin data penjualan dummy 180 hari |
| features.py | feature engineering (dipakai training & prediksi) |
| train_model.py | training + evaluasi vs baseline -> restock_model.pkl |
| predict_restock.py | prediksi via CLI -> tabel + restock_alert.json |
| restock_predictor.py | versi FUNGSI untuk service: input/output JSON (kontrak di bawah) |
| restock_model.pkl | model terlatih (dummy) |
| penjualan_dummy.csv, produk_sample.csv | data contoh |

## Cara pakai (demo)

```bash
pip install -r requirements.txt
python predict_restock.py                 # pakai model & data dummy yang ada
```

Training ulang dari awal (mis. setelah ganti data):

```bash
python generate_dummy_data.py             # atau siapkan CSV asli
python train_model.py [penjualan.csv]
python predict_restock.py [penjualan.csv] [stok.csv]
```

Format CSV penjualan: `tanggal,nama_produk,qty_terjual` (harian per produk,
minimal 14 hari histori per produk). Format stok: minimal kolom
`nama_produk,stok_saat_ini` (sama dengan produk_master.csv).

## Kontrak untuk backend (via restock_predictor.py)

```python
from restock_predictor import predict_restock
hasil = predict_restock(penjualan, stok, lead_time_hari=2, buffer_hari=1)
```

- `penjualan`: `[{"tanggal": "2026-07-01", "nama_produk": "...", "qty_terjual": 4}, ...]`
- `stok`: `[{"nama_produk": "...", "stok_saat_ini": 20}, ...]`
- return: `{"prediksi_mulai", "alerts": [...], "dilewati": [...]}` — tiap alert
  berisi status PERLU RESTOCK/AMAN, perkiraan stok habis (hari), dan saran qty.

Rencana integrasi (flow tahap 4): scheduler backend tiap malam -> query DB
penjualan + stok -> panggil fungsi ini (atau bungkus jadi endpoint
`POST /restock/predict`) -> simpan ke tabel `restock_predictions` -> pagi
harinya pedagang lihat daftar belanja di aplikasi.
