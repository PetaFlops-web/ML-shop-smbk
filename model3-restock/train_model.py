"""
Training model restock: belajar pola penjualan harian -> prediksi permintaan.

Model : RandomForestRegressor (scikit-learn), di-train DARI 0 pakai data kalian.
Input : penjualan.csv dengan kolom [tanggal, nama_produk, qty_terjual]
Output: restock_model.pkl  (berisi model + daftar kolom fitur + daftar produk)

Evaluasi dilakukan secara time-based (30 hari terakhir jadi data test — model
gak boleh "ngintip masa depan") dan dibandingkan dengan baseline sederhana
(tebakan = rata-rata 7 hari terakhir). Kalau model gak lebih baik dari
baseline, angka ini bakal kelihatan — jadi kalian tahu jujur performanya.

Jalankan:
    python train_model.py                      # pakai penjualan_dummy.csv
    python train_model.py data_asli.csv        # nanti kalau data asli udah ada
"""

import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from features import build_training_table

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "penjualan_dummy.csv"
TEST_DAYS = 30


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Data: {CSV_PATH} ({len(df)} baris)")

    tabel = build_training_table(df)
    feature_cols = [c for c in tabel.columns if c not in ("target", "tanggal")]

    # split time-based BERDASARKAN TANGGAL: 30 hari terakhir jadi data test,
    # sisanya buat training (model gak boleh "ngintip masa depan")
    cutoff = tabel["tanggal"].max() - pd.Timedelta(days=TEST_DAYS)
    train = tabel[tabel["tanggal"] <= cutoff]
    test = tabel[tabel["tanggal"] > cutoff]
    print(f"Training: {len(train)} baris (s/d {cutoff.date()}), "
          f"Test: {len(test)} baris ({TEST_DAYS} hari terakhir)")

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(train[feature_cols], train["target"])

    # evaluasi: model vs baseline (tebak pakai rata-rata 7 hari terakhir)
    pred_model = model.predict(test[feature_cols])
    mae_model = mean_absolute_error(test["target"], pred_model)
    mae_baseline = mean_absolute_error(test["target"], test["rolling_mean_7"])

    print(f"\nEvaluasi di {TEST_DAYS} hari terakhir (data yang gak dipakai training):")
    print(f"  MAE baseline (rata-rata 7 hari) : {mae_baseline:.3f}")
    print(f"  MAE model RandomForest          : {mae_model:.3f}")
    selisih = (mae_baseline - mae_model) / mae_baseline * 100
    print(f"  -> model {'LEBIH BAIK' if selisih > 0 else 'lebih buruk'} "
          f"{abs(selisih):.1f}% dari baseline")

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "produk_list": sorted(df["nama_produk"].unique().tolist()),
        "trained_on": CSV_PATH,
    }
    joblib.dump(bundle, "restock_model.pkl")
    print("\nModel disimpan: restock_model.pkl")


if __name__ == "__main__":
    main()
