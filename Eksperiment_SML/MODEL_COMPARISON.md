# Perbandingan Model Sembako UMKM

Dua model dibuat untuk kebutuhan berbeda dalam sistem inventaris UMKM sembako.

---

## Ringkasan

| | Model Inventaris | Model Restock |
|---|---|---|
| **Notebook** | `experiment_model_inventaris.ipynb` | `experiment_model_restock.ipynb` |
| **File model** | `models/model_inventaris.pkl` | `model_coxph_restock.pkl` |
| **Pertanyaan yang dijawab** | Berapa banyak barang akan laku? | Kapan pelanggan akan beli lagi? |
| **Level prediksi** | `toko × barang × tanggal` | `pelanggan × barang` |
| **Algoritma** | RandomForestRegressor | Cox Proportional Hazards |
| **Metrik evaluasi** | R² = 0.93, MAE = 6.34 | Concordance index |
| **Output utama** | `predicted_sales` (jumlah unit) | `Pred_Days_Left`, probabilitas beli ulang |

---

## Model Inventaris — Prediksi Demand Harian

### Kegunaan

Menjawab: **"Besok/lusa/minggu depan, barang ini kira-kira laku berapa?"**

Diputuskan untuk:

- **Keputusan stok toko** — berapa banyak yang harus distok ulang.
- **Perencanaan pembelian ke supplier** — estimasi kebutuhan mingguan/bulanan.
- **Optimasi stok gudang** — hindari overstock dan stockout.

### Cara Kerja

```text
Histori sales harian barang di toko
    → hitung fitur lag, rolling mean, EWMA
    → RandomForest memprediksi jumlah sales
    → backend bandingkan dengan stok sekarang
    → keputusan: perlu restock berapa
```

### Input API

```json
{
  "store": "TOKO-001",
  "item": "SKU-BERAS-5KG",
  "date": "2026-08-01",
  "sales_history": [50, 48, 52, ...]  // 90 hari terakhir
}
```

### Output API

```json
{
  "store": "TOKO-001",
  "item": "SKU-BERAS-5KG",
  "date": "2026-08-01",
  "predicted_sales": 42
}
```

Backend kemudian menghitung:

```text
recommended_restock = max(0, predicted_sales + safety_stock - current_stock)
```

### Fitur Utama

Model tidak melihat `store`/`item` sebagai fitur — hanya sebagai kunci pengelompokan histori. Fitur yang benar-benar masuk ke model: `month`, `day_of_week`, `day_of_month`, `quarter`, `sales_lag_1/2/3/7/14/30`, `sales_roll_mean_7/14/30`, `sales_ewma_7/30`.

### Syarat Data

| Kondisi | Status |
|---|---|
| Histori < 30 hari | Jangan andalkan model |
| Histori 30–89 hari | Bisa, tapi kurang stabil |
| Histori ≥ 90 hari | Ideal |

---

## Model Restock — Prediksi Pembelian Ulang Pelanggan

### Kegunaan

Menjawab: **"Pelanggan ini kapan kemungkinan beli barang ini lagi?"**

Diputuskan untuk:

- **Notifikasi/pengingat ke pelanggan** — "Stok beras Anda mungkin sudah habis."
- **Program loyalitas** — tawarkan diskon saat pelanggan diperkirakan akan beli ulang.
- **Prediksi churn** — pelanggan yang sudah lewat jauh dari perkiraan beli ulang.

### Cara Kerja

```text
Setiap kali pelanggan checkout
    → ambil riwayat pembelian pelanggan × produk
    → hitung PurchaseNumber, DaysSincePrev, Avg_Days_Between
    → CoxPH memprediksi survival function
    → dapatkan Pred_Days_Left dan probabilitas beli ulang
    → simpan ke tabel prediksi
Scheduler harian
    → cek tabel prediksi
    → jika Pred_Days_Left ≤ threshold, kirim notifikasi
```

### Input API

```json
{
  "customer_id": 17841,
  "stock_code": "Beras Premium 5 kg",
  "purchase_number": 4,
  "days_since_prev": 31,
  "avg_days_between": 35,
  "quantity": 2,
  "unit_price": 15000,
  "basket_size": 5,
  "basket_unique": 3,
  "basket_value": 120000,
  "month": 8,
  "day_of_week": 3
}
```

### Output API

```json
{
  "customer_id": 17841,
  "stock_code": "Beras Premium 5 kg",
  "predicted_restock_date": "2026-08-18",
  "pred_days_left": 32,
  "prob_buy_within_7d": 0.12,
  "prob_buy_within_14d": 0.45,
  "prob_buy_within_30d": 0.82
}
```

### Aturan Notifikasi

```text
Kirim notifikasi jika:
  (Pred_Days_Left ≤ 3 DAN Prob_Buy_Within_14D ≥ 0.50)
  ATAU
  (Pred_Days_Left ≤ 7 DAN Prob_Buy_Within_7D ≥ 0.60)
```

### Syarat Data

| Jumlah transaksi pelanggan-produk | Kualitas |
|---|---|
| 1 | Bisa, tapi fallback ke nilai global |
| 2 | Mulai terpersonalisasi |
| 3–5 | Cukup baik untuk notifikasi |
| ≥ 6 | Stabil dan ideal |

---

## Kapan Pakai yang Mana?

| Kebutuhan | Model |
|---|---|
| **Stok toko: berapa banyak restock?** | Inventaris |
| **Notifikasi pelanggan: kapan ingatkan beli ulang?** | Restock |
| **Laporan demand mingguan ke supplier** | Inventaris |
| **Promo/diskon ke pelanggan yang diperkirakan akan beli** | Restock |
| **Dashboard stok gudang** | Inventaris |
| **Prediksi churn / pelanggan hilang** | Restock |

---

## Arsitektur Sistem (Gabungan)

```text
┌──────────────────────────────────────────┐
│              BACKEND UTAMA               │
│  (Flask / FastAPI / Express)             │
└──────┬────────────────────┬──────────────┘
       │                    │
       ▼                    ▼
┌──────────────┐   ┌──────────────────┐
│  INVENTARIS  │   │     RESTOK       │
│  (demand)    │   │  (notifikasi)    │
│              │   │                  │
│ Prediksi     │   │ Prediksi kapan   │
│ jumlah       │   │ pelanggan beli   │
│ barang laku  │   │ ulang            │
│              │   │                  │
│ → stok toko  │   │ → notifikasi     │
│ → supplier   │   │ → promo loyalitas│
└──────────────┘   └──────────────────┘
```

Kedua model bisa jalan bersamaan dan saling melengkapi
