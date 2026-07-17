import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "resources"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = BASE_DIR / "qwen-ekstraksi-lora"
PRODUCT_DICTIONARY = BASE_DIR / "product_dictionary.json"
PRODUCT_MASTER = BASE_DIR / "produk_master.csv"

SYSTEM_PROMPT = (
    "Kamu adalah sistem ekstraksi data transaksi toko sembako. "
    "Dari teks user, ekstrak SETIAP barang yang disebutkan. "
    "Balas HANYA dengan JSON array, tanpa penjelasan. "
    'Format tiap item: {"item": "nama barang", "qty": angka, "harga": angka_atau_null}. '
    "Kalau harga tidak disebut, isi null. Kalau qty tidak disebut, anggap 1."
)

_alias = None
_harga = None
_tokenizer = None
_model = None

def _get_alias() -> dict[str, str]:
    global _alias
    if _alias is None:
        _alias = {}
        if PRODUCT_DICTIONARY.exists():
            kamus = json.loads(PRODUCT_DICTIONARY.read_text(encoding="utf-8"))
            _alias = {a.lower(): resmi for resmi, aliases in kamus.items() for a in aliases}
    return _alias


def _similarity_score(a: str, b: str) -> float:
    r_full = SequenceMatcher(None, a, b).ratio()
    ta = a.split()
    tb = b.split()
    r_tok = sum(max(SequenceMatcher(None, x, y).ratio() for y in tb) for x in ta) / len(ta) if ta and tb else 0
    return max(r_full, r_tok)


def match_catalog(nama: str) -> dict:
    alias = _get_alias()
    nama = nama.lower().strip()
    if not alias:
        return {}
    if nama in alias:
        return {"produk_katalog": alias[nama], "skor_cocok": 1.0, "status_cocok": "yakin"}

    skor_per_produk = {}
    for alias_nama, resmi in alias.items():
        skor = _similarity_score(nama, alias_nama)
        if skor > skor_per_produk.get(resmi, 0):
            skor_per_produk[resmi] = skor

    urut = sorted(skor_per_produk.items(), key=lambda item: -item[1])
    produk_utama, skor_utama = urut[0]
    skor_kedua = urut[1][1] if len(urut) > 1 else 0

    if skor_utama < 0.6:
        return {"produk_katalog": None, "skor_cocok": round(skor_utama, 3), "status_cocok": "tidak_ketemu"}
    if skor_utama >= 0.75 and (skor_utama - skor_kedua) > 0.03:
        return {"produk_katalog": produk_utama, "skor_cocok": round(skor_utama, 3), "status_cocok": "yakin"}
    return {
        "produk_katalog": produk_utama,
        "skor_cocok": round(skor_utama, 3),
        "status_cocok": "perlu_konfirmasi",
        "kandidat_lain": urut[1][0] if len(urut) > 1 else None,
    }


def _get_harga() -> dict[str, float]:
    global _harga
    if _harga is None:
        _harga = {}
        if PRODUCT_MASTER.exists():
            with PRODUCT_MASTER.open(encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    try:
                        _harga[row["nama_produk"]] = float(row["harga_jual"])
                    except (KeyError, ValueError, TypeError):
                        continue
    return _harga


def _load_model():
    global _tokenizer, _model
    if _model is None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
        _model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        _model.eval()
    return _tokenizer, _model


def predict(teks: str) -> list:
    tokenizer, model = _load_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": teks},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    import torch

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    hasil = json.loads(raw)
    if not isinstance(hasil, list):
        raise ValueError("Output model bukan JSON array")
    return hasil


def enrich_items(items: list) -> list:
    if not _get_alias():
        return items
    harga = _get_harga()
    for item in items:
        if not isinstance(item, dict) or not item.get("item"):
            continue
        item.update(match_catalog(str(item["item"])))
        if item.get("harga") is not None:
            item["sumber_harga"] = "ucapan"
        elif item.get("produk_katalog") in harga:
            item["harga"] = harga[item["produk_katalog"]]
            item["sumber_harga"] = "katalog"
    return items


def extract_transaction(teks: str, nama_sumber: str = "manual") -> dict:
    teks = teks.strip()

    if not teks:
        raise ValueError("Teks transkrip kosong")
    
    items = enrich_items(predict(teks))

    hasil = {
        "sumber": nama_sumber,
        "transkrip": teks,
        "items": items,
    }

    return hasil