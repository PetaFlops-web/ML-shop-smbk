import joblib
import numpy as np
import pandas as pd
from pathlib import Path

CATALOG_RESTOCK_PRODUCT = {
    "beras": "Beras 5kg",
    "gula pasir": "Gula Pasir 1kg",
    "indomie": "Indomie Goreng",
    "kopi": "Kopi Sachet",
    "minyak goreng": "Minyak Goreng 1L",
    "telur": "Telur 1kg",
    "tepung terigu": "Tepung Terigu 1kg",
    "terigu": "Tepung Terigu 1kg",
}


def _normalize_produk(nama: str) -> str:
    nama_lower = nama.lower()
    for keyword, nama_restock in CATALOG_RESTOCK_PRODUCT.items():
        if keyword in nama_lower:
            return nama_restock
    return nama 

HORIZON = 7         
LEAD_TIME_HARI = 2  
BUFFER_HARI = 1      

_MODEL_PATH = Path(__file__).resolve().parents[1] / "resources" / "models" / "restock_model.pkl"
_bundle = None 


# Feature 

def calendar_features(tanggal: pd.Timestamp) -> dict:
    return {
        "day_of_week": tanggal.dayofweek,           
        "is_weekend": int(tanggal.dayofweek >= 5),
        "day_of_month": tanggal.day,
        "is_awal_bulan": int(tanggal.day <= 5),    
    }


def history_features(qty_sebelumnya: list[float]) -> dict:
    arr = np.array(qty_sebelumnya, dtype=float)
    return {
        "lag_1": arr[-1],                
        "lag_7": arr[-7],               
        "rolling_mean_7": arr[-7:].mean(),
        "rolling_mean_14": arr[-14:].mean(),
        "rolling_std_7": arr[-7:].std(), 
    }


# Model loading

def _load_bundle() -> dict:
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(str(_MODEL_PATH))
    return _bundle


# Prediksi per produk

def _prediksi_produk(bundle: dict, produk: str, histori: list,
                     mulai: pd.Timestamp) -> list:
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    histori = list(histori)  
    hasil = []

    for h in range(HORIZON):
        tanggal = mulai + pd.Timedelta(days=h)
        fitur = {**calendar_features(tanggal), **history_features(histori)}

        # One-hot encoding produk, semua kolom "produk_*" di buat 0
        # kecuali kolom milik produk ini.
        for col in feature_cols:
            if col.startswith("produk_"):
                fitur[col] = 1 if col == f"produk_{produk}" else 0

        X = pd.DataFrame([fitur])[feature_cols]
        pred = max(0.0, float(model.predict(X)[0]))
        hasil.append(pred)
        histori.append(pred) 

    return hasil


# Function Prediction Stock

def predict_restock(penjualan: list, stok: list,
                    lead_time_hari: int = LEAD_TIME_HARI,
                    buffer_hari: int = BUFFER_HARI) -> dict:
    
    bundle = _load_bundle()

    df = pd.DataFrame(penjualan)
    df["nama_produk"] = df["nama_produk"].apply(_normalize_produk)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["qty_terjual"] = df["qty_terjual"].astype(float)
    df = df.groupby(["nama_produk", "tanggal"], as_index=False)["qty_terjual"].sum()

    stok_map = {}
    for s in stok:
        nama = _normalize_produk(s["nama_produk"])
        stok_map[nama] = stok_map.get(nama, 0.0) + float(s["stok_saat_ini"])

    besok = df["tanggal"].max() + pd.Timedelta(days=1)

    alerts = []
    dilewati = []

    input_products = set(df["nama_produk"].unique()) & set(bundle["produk_list"])

    for produk in sorted(input_products):
        grup = df[df["nama_produk"] == produk].sort_values("tanggal")

        if len(grup) < 14:
            dilewati.append({
                "nama_produk": produk,
                "alasan": "histori < 14 hari (minimal butuh 14 hari data)"
            })
            continue

        if produk not in stok_map:
            dilewati.append({
                "nama_produk": produk,
                "alasan": "tidak ditemukan di data stok"
            })
            continue

        histori = grup["qty_terjual"].tolist()
        stok_skrg = stok_map[produk]

        pred = _prediksi_produk(bundle, produk, histori, besok)
        total_7hari = sum(pred)
        rata = total_7hari / HORIZON

        habis_dalam = stok_skrg / rata if rata > 0 else float("inf")

        perlu = (
            habis_dalam <= (lead_time_hari + buffer_hari)
            or total_7hari >= stok_skrg
        )

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

    produk_dikenal = set(bundle["produk_list"])
    for nama in stok_map:
        if nama not in produk_dikenal:
            dilewati.append({
                "nama_produk": nama,
                "alasan": "belum dikenal model (perlu training ulang)"
            })

    return {
        "prediksi_mulai": str(besok.date()),
        "alerts": alerts,
        "dilewati": dilewati,
    }
