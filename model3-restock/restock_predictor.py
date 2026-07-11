"""
Prediksi restock versi service: sama logikanya dengan predict_restock.py di
restock-model, tapi input/output-nya JSON (bukan file CSV) — karena data
penjualan aslinya ada di database backend, bukan di ml-service.

Kontrak:
    predict_restock(penjualan, stok, lead_time_hari=2, buffer_hari=1) -> list[dict]

    penjualan: [{"tanggal": "2026-07-01", "nama_produk": "...", "qty_terjual": 4}, ...]
               (harian per produk, minimal 14 hari per produk)
    stok     : [{"nama_produk": "...", "stok_saat_ini": 20}, ...]
"""

import joblib
import numpy as np
import pandas as pd

from features import calendar_features, history_features

HORIZON = 7  # prediksi 7 hari ke depan

_bundle = None


def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load("restock_model.pkl")
    return _bundle


def _prediksi_produk(bundle, produk: str, histori: list, mulai: pd.Timestamp) -> list:
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    histori = list(histori)
    hasil = []
    for h in range(HORIZON):
        tanggal = mulai + pd.Timedelta(days=h)
        fitur = {**calendar_features(tanggal), **history_features(histori)}
        for col in feature_cols:
            if col.startswith("produk_"):
                fitur[col] = 1 if col == f"produk_{produk}" else 0
        X = pd.DataFrame([fitur])[feature_cols]
        pred = max(0.0, float(model.predict(X)[0]))
        hasil.append(pred)
        histori.append(pred)
    return hasil


def predict_restock(penjualan: list, stok: list,
                    lead_time_hari: int = 2, buffer_hari: int = 1) -> dict:
    bundle = _get_bundle()

    df = pd.DataFrame(penjualan)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    stok_map = {s["nama_produk"]: float(s["stok_saat_ini"]) for s in stok}

    besok = df["tanggal"].max() + pd.Timedelta(days=1)
    alerts, dilewati = [], []

    for produk in bundle["produk_list"]:
        grup = df[df["nama_produk"] == produk].sort_values("tanggal")
        if len(grup) < 14:
            dilewati.append({"nama_produk": produk, "alasan": "histori < 14 hari"})
            continue
        if produk not in stok_map:
            dilewati.append({"nama_produk": produk, "alasan": "tidak ada di data stok"})
            continue

        histori = grup["qty_terjual"].tolist()
        stok_skrg = stok_map[produk]

        pred = _prediksi_produk(bundle, produk, histori, besok)
        total_7hari = sum(pred)
        rata = total_7hari / HORIZON

        habis_dalam = stok_skrg / rata if rata > 0 else float("inf")
        perlu = habis_dalam <= (lead_time_hari + buffer_hari) or total_7hari >= stok_skrg
        kebutuhan = rata * (lead_time_hari + HORIZON)
        saran = max(0, int(np.ceil(kebutuhan - stok_skrg)))

        alerts.append({
            "nama_produk": produk,
            "stok_saat_ini": stok_skrg,
            "prediksi_terjual_7hari": round(total_7hari, 1),
            "rata_rata_per_hari": round(rata, 2),
            "perkiraan_stok_habis_dalam_hari": round(habis_dalam, 1),
            "status": "PERLU RESTOCK" if perlu else "AMAN",
            "saran_qty_restock": saran if perlu else 0,
        })

    alerts.sort(key=lambda a: a["perkiraan_stok_habis_dalam_hari"])

    # produk yang gak dikenal model (belum ada pas training) juga dilaporkan
    produk_dikenal = set(bundle["produk_list"])
    for nama in stok_map:
        if nama not in produk_dikenal:
            dilewati.append({"nama_produk": nama,
                             "alasan": "belum dikenal model (training ulang dengan data terbaru)"})

    return {"prediksi_mulai": str(besok.date()), "alerts": alerts, "dilewati": dilewati}
