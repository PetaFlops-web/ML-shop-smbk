"""
Generator dataset buat LoRA fine-tuning Qwen (tugas: ekstraksi transaksi).

Tiap contoh = kalimat ala pedagang ngomong  +  jawaban JSON yang benar.
Variasi yang sengaja dimasukin biar model terlatih menghadapi gaya nyata:
- qty: angka ("2"), kata ("dua"), atau gak disebut (berarti 1)
- harga: "17rb", "17 ribu", "tujuh belas ribu", "17000", "17.000", atau gak disebut (null)
- satuan: kilo/kg/bungkus/botol/karung/sachet/pcs/liter, atau tanpa satuan
- urutan kata dibolak-balik, pemisah item pakai koma / "sama" / "dan"
- nama barang pakai alias sehari-hari (telor, migor, mie goreng, dst)

Output: train.jsonl & val.jsonl — format "messages" (system/user/assistant),
langsung kompatibel sama chat template HuggingFace buat SFT.

Jalankan: python generate_dataset.py
"""

import json
import random

random.seed(42)

with open("product_dictionary.json", encoding="utf-8") as f:
    PRODUCT_DICTIONARY = json.load(f)

# semua alias jadi kandidat "cara nyebut barang"
ALIASES = [alias for aliases in PRODUCT_DICTIONARY.values() for alias in aliases]

UNITS = ["kilo", "kg", "bungkus", "botol", "karung", "sachet", "pcs", "liter", ""]
SEPARATORS = [", ", " sama ", " dan "]
PREFIXES = ["", "", "beli ", "tadi laku ", "catat "]

SATUAN_WORDS = {
    1: "satu", 2: "dua", 3: "tiga", 4: "empat", 5: "lima",
    6: "enam", 7: "tujuh", 8: "delapan", 9: "sembilan", 10: "sepuluh",
    11: "sebelas",
}


def angka_ke_kata(n: int) -> str:
    """1-99 jadi kata Bahasa Indonesia."""
    if n <= 11:
        return SATUAN_WORDS[n]
    if n < 20:
        return f"{SATUAN_WORDS[n - 10]} belas"
    puluhan, sisa = divmod(n, 10)
    hasil = f"{SATUAN_WORDS[puluhan]} puluh"
    return f"{hasil} {SATUAN_WORDS[sisa]}" if sisa else hasil


def render_qty(qty: int) -> str:
    return str(qty) if random.random() < 0.5 else angka_ke_kata(qty)


def render_harga(harga_ribu: int) -> str:
    """harga_ribu contoh 17 (artinya 17.000). Balikin salah satu format."""
    gaya = random.choice(["rb", "ribu_angka", "ribu_kata", "digit", "digit_titik"])
    if gaya == "rb":
        return f"{harga_ribu}rb"
    if gaya == "ribu_angka":
        return f"{harga_ribu} ribu"
    if gaya == "ribu_kata":
        return f"{angka_ke_kata(harga_ribu)} ribu"
    if gaya == "digit":
        return str(harga_ribu * 1000)
    return f"{harga_ribu}.000"


def bikin_satu_item():
    alias = random.choice(ALIASES)
    qty = random.choices([1, 2, 3, 4, 5, 10, 12, 15, 20, 25],
                         weights=[30, 20, 12, 8, 10, 6, 4, 4, 4, 2])[0]
    unit = random.choice(UNITS)
    harga_ribu = random.choice([2, 3, 5, 6, 9, 11, 12, 14, 15, 17, 18, 20, 25, 27, 30, 55, 62])
    ada_harga = random.random() < 0.75
    qty_disebut = random.random() < 0.8 or qty != 1

    # susun frasa
    bagian = [alias]
    if qty_disebut:
        frasa_qty = f"{render_qty(qty)} {unit}".strip()
        # kadang qty di depan nama barang ("dua kilo beras")
        if random.random() < 0.25:
            bagian = [frasa_qty, alias]
        else:
            bagian = [alias, frasa_qty]
    if ada_harga:
        bagian.append(render_harga(harga_ribu))

    teks = " ".join(bagian)
    label = {
        "item": alias,
        "qty": qty if qty_disebut else 1,
        "harga": harga_ribu * 1000 if ada_harga else None,
    }
    return teks, label


SYSTEM_PROMPT = (
    "Kamu adalah sistem ekstraksi data transaksi toko sembako. "
    "Dari teks user, ekstrak SETIAP barang yang disebutkan. "
    "Balas HANYA dengan JSON array, tanpa penjelasan. "
    'Format tiap item: {"item": "nama barang", "qty": angka, "harga": angka_atau_null}. '
    "Kalau harga tidak disebut, isi null. Kalau qty tidak disebut, anggap 1."
)


def bikin_satu_contoh():
    n_item = random.choices([1, 2, 3, 4], weights=[35, 35, 20, 10])[0]
    items = [bikin_satu_item() for _ in range(n_item)]
    kalimat = random.choice(PREFIXES) + random.choice(SEPARATORS).join(t for t, _ in items)
    label = [lbl for _, lbl in items]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": kalimat},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ]
    }


def main(n_train=800, n_val=100):
    semua = [bikin_satu_contoh() for _ in range(n_train + n_val)]

    with open("train.jsonl", "w", encoding="utf-8") as f:
        for contoh in semua[:n_train]:
            f.write(json.dumps(contoh, ensure_ascii=False) + "\n")
    with open("val.jsonl", "w", encoding="utf-8") as f:
        for contoh in semua[n_train:]:
            f.write(json.dumps(contoh, ensure_ascii=False) + "\n")

    print(f"train.jsonl: {n_train} contoh | val.jsonl: {n_val} contoh\n")
    print("Contoh isi:")
    for contoh in semua[:4]:
        print("  Kalimat :", contoh["messages"][1]["content"])
        print("  Label   :", contoh["messages"][2]["content"])
        print()


if __name__ == "__main__":
    main()
