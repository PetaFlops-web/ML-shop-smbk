"""
Feature engineering buat model restock — dipakai bareng oleh train_model.py
dan predict_restock.py (biar fitur saat training & prediksi DIJAMIN sama).

Dari riwayat penjualan harian per produk, tiap hari dibikin fitur:
- pola kalender : hari ke berapa dalam minggu/bulan, weekend, awal bulan (gajian)
- pola historis : penjualan 1 & 7 hari lalu, rata-rata & naik-turunnya 7/14 hari
"""

import numpy as np
import pandas as pd

# kolom fitur kalender + historis (belum termasuk one-hot produk)
BASE_FEATURES = [
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "is_awal_bulan",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
]


def calendar_features(tanggal: pd.Timestamp) -> dict:
    return {
        "day_of_week": tanggal.dayofweek,           # 0=Senin ... 6=Minggu
        "is_weekend": int(tanggal.dayofweek >= 5),
        "day_of_month": tanggal.day,
        "is_awal_bulan": int(tanggal.day <= 5),      # masa gajian, biasanya rame
    }


def history_features(qty_sebelumnya: list[float]) -> dict:
    """qty_sebelumnya: penjualan harian SEBELUM hari yang mau diprediksi,
    urut dari paling lama ke paling baru. Minimal 14 hari."""
    arr = np.array(qty_sebelumnya, dtype=float)
    return {
        "lag_1": arr[-1],
        "lag_7": arr[-7],
        "rolling_mean_7": arr[-7:].mean(),
        "rolling_mean_14": arr[-14:].mean(),
        "rolling_std_7": arr[-7:].std(),
    }


def build_training_table(df_penjualan: pd.DataFrame) -> pd.DataFrame:
    """
    df_penjualan: kolom [tanggal, nama_produk, qty_terjual], harian, tanpa bolong.
    Return: tabel fitur + target (qty_terjual) + one-hot nama produk.
    """
    df = df_penjualan.copy()
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.sort_values(["nama_produk", "tanggal"])

    rows = []
    for produk, grup in df.groupby("nama_produk"):
        qty = grup["qty_terjual"].tolist()
        tgl = grup["tanggal"].tolist()
        # mulai dari hari ke-14 (butuh 14 hari histori buat fitur rolling)
        for i in range(14, len(qty)):
            row = {"nama_produk": produk, "target": qty[i], "tanggal": tgl[i]}
            row.update(calendar_features(tgl[i]))
            row.update(history_features(qty[:i]))
            rows.append(row)

    tabel = pd.DataFrame(rows)
    tabel = pd.get_dummies(tabel, columns=["nama_produk"], prefix="produk")
    return tabel
