"""
Prediksi restock: pakai model .pkl hasil training buat jawab pertanyaan
"barang apa yang harus di-restock, kapan, dan berapa banyak?"

Cara kerja per produk:
1. Model memprediksi penjualan 7 hari ke depan (hari demi hari; prediksi hari
   ini dipakai buat bantu memprediksi hari besoknya — recursive forecasting)
2. Stok sekarang dibagi laju prediksi -> perkiraan berapa hari lagi stok habis
3. Kalau stok bakal habis sebelum (lead time + buffer), statusnya PERLU RESTOCK,
   dan saran qty = kebutuhan (lead time + 7 hari ke depan) - stok sekarang

Input:
- restock_model.pkl        (hasil train_model.py)
- CSV riwayat penjualan    (yang dipakai saat training)
- CSV stok produk          (format sama dengan produk_sample.csv backend:
                            minimal kolom nama_produk, stok_saat_ini)

Jalankan:
    python predict_restock.py
    python predict_restock.py penjualan_dummy.csv produk_sample.csv
Output: tabel di terminal + restock_alert.json (siap dikonsumsi backend)
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd

from features import calendar_features, history_features

PENJUALAN_CSV = sys.argv[1] if len(sys.argv) > 1 else "penjualan_dummy.csv"
STOK_CSV = sys.argv[2] if len(sys.argv) > 2 else "produk_sample.csv"

HORIZON = 7          # prediksi berapa hari ke depan
LEAD_TIME_HARI = 2   # lama nunggu barang dateng setelah pesan ke supplier
BUFFER_HARI = 1      # cadangan aman


def prediksi_produk(bundle, produk: str, histori_qty: list[float], mulai: pd.Timestamp):
    """Prediksi penjualan `HORIZON` hari ke depan untuk satu produk."""
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    histori = list(histori_qty)
    hasil = []

    for h in range(HORIZON):
        tanggal = mulai + pd.Timedelta(days=h)
        fitur = {**calendar_features(tanggal), **history_features(histori)}
        # one-hot produk: semua 0 kecuali produk ini
        for col in feature_cols:
            if col.startswith("produk_"):
                fitur[col] = 1 if col == f"produk_{produk}" else 0

        X = pd.DataFrame([fitur])[feature_cols]
        pred = max(0.0, float(model.predict(X)[0]))
        hasil.append(pred)
        histori.append(pred)  # prediksi hari ini jadi "histori" buat hari besok

    return hasil


def main():
    bundle = joblib.load("restock_model.pkl")

    df = pd.read_csv(PENJUALAN_CSV)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    stok_df = pd.read_csv(STOK_CSV).set_index("nama_produk")

    besok = df["tanggal"].max() + pd.Timedelta(days=1)
    alerts = []

    for produk in bundle["produk_list"]:
        grup = df[df["nama_produk"] == produk].sort_values("tanggal")
        if len(grup) < 14:
            print(f"[skip] {produk}: histori < 14 hari")
            continue
        if produk not in stok_df.index:
            print(f"[skip] {produk}: gak ada di file stok")
            continue

        histori = grup["qty_terjual"].tolist()
        stok = float(stok_df.loc[produk, "stok_saat_ini"])

        pred_7hari = prediksi_produk(bundle, produk, histori, besok)
        total_7hari = sum(pred_7hari)
        rata_per_hari = total_7hari / HORIZON

        habis_dalam = stok / rata_per_hari if rata_per_hari > 0 else float("inf")
        # perlu restock kalau stok habis sebelum barang sempat dateng dari supplier
        perlu_restock = habis_dalam <= (LEAD_TIME_HARI + BUFFER_HARI)

        # saran qty: kebutuhan sampai (lead time + 7 hari) dikurangi stok sekarang
        kebutuhan = rata_per_hari * (LEAD_TIME_HARI + HORIZON)
        saran_qty = max(0, int(np.ceil(kebutuhan - stok)))

        # status juga PERLU RESTOCK kalau prediksi 7 hari ke depan melebihi stok
        if total_7hari >= stok:
            perlu_restock = True

        alerts.append({
            "nama_produk": produk,
            "stok_saat_ini": stok,
            "prediksi_terjual_7hari": round(total_7hari, 1),
            "rata_rata_per_hari": round(rata_per_hari, 2),
            "perkiraan_stok_habis_dalam_hari": round(habis_dalam, 1),
            "status": "PERLU RESTOCK" if perlu_restock else "AMAN",
            "saran_qty_restock": saran_qty if perlu_restock else 0,
        })

    hasil = sorted(alerts, key=lambda a: a["perkiraan_stok_habis_dalam_hari"])

    print(f"\nPrediksi mulai {besok.date()} ({HORIZON} hari ke depan), "
          f"lead time supplier {LEAD_TIME_HARI} hari:\n")
    for a in hasil:
        print(f"  [{a['status']:13s}] {a['nama_produk']:20s} "
              f"stok {a['stok_saat_ini']:6.1f} | "
              f"prediksi 7 hari: {a['prediksi_terjual_7hari']:6.1f} | "
              f"habis dalam ~{a['perkiraan_stok_habis_dalam_hari']:5.1f} hari | "
              f"saran restock: {a['saran_qty_restock']}")

    with open("restock_alert.json", "w") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)
    print("\nDisimpan ke restock_alert.json (siap dipakai backend)")


if __name__ == "__main__":
    main()
