"""
Bikin data penjualan DUMMY 180 hari buat training model restock.

PENTING: ini data sintetis (buatan), dipakai sementara sampai sistem transaksi
kalian jalan & ngumpulin data asli. Polanya dibikin mirip toko sembako nyata:
- weekend lebih rame (+30%)
- awal bulan / tanggal muda lebih rame (+25%)  -> efek gajian
- naik-turun acak harian (Poisson)

Nanti kalau data asli udah ada, export dari database dengan kolom yang sama:
    tanggal,nama_produk,qty_terjual
lalu train ulang pakai train_model.py — gak perlu ubah kode apapun.

Jalankan: python generate_dummy_data.py  -> menghasilkan penjualan_dummy.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)  # seed biar hasilnya konsisten

# rata-rata terjual per hari tiap produk (disesuaikan kasarnya sama toko sembako)
BASE_DEMAND = {
    "Beras 5kg": 4,
    "Minyak Goreng 1L": 7,
    "Gula Pasir 1kg": 5,
    "Telur 1kg": 6,
    "Indomie Goreng": 25,
    "Tepung Terigu 1kg": 3,
    "Kopi Sachet": 15,
}

JUMLAH_HARI = 180


def generate() -> pd.DataFrame:
    akhir = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    tanggal_list = pd.date_range(end=akhir, periods=JUMLAH_HARI, freq="D")

    rows = []
    for produk, base in BASE_DEMAND.items():
        for tanggal in tanggal_list:
            demand = float(base)
            if tanggal.dayofweek >= 5:      # Sabtu/Minggu
                demand *= 1.30
            if tanggal.day <= 5:            # awal bulan (gajian)
                demand *= 1.25
            qty = int(RNG.poisson(demand))  # naik-turun acak harian
            rows.append({
                "tanggal": tanggal.date(),
                "nama_produk": produk,
                "qty_terjual": qty,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("penjualan_dummy.csv", index=False)
    print(f"penjualan_dummy.csv dibuat: {len(df)} baris "
          f"({df['nama_produk'].nunique()} produk x {JUMLAH_HARI} hari)")
    print(df.tail(3).to_string(index=False))
