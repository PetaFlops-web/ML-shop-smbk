"""
PENYAMBUNG MODEL 1 -> MODEL 2:
    Model 1 (whisper API Akbar) nyimpen transkrip -> named volume Docker
    Script ini (Model 2) ngambil transkrip -> prediksi -> output_predictions/

Taruh di folder lora-training (sebelahan folder qwen-ekstraksi-lora).

Pakai (pilih salah satu):
    # MODE API (sekali proses, service Model 1 harus lagi jalan):
    python model2_dari_transkrip.py --api

    # MODE WATCH (OTOMATIS: mantau terus, transkrip baru langsung diproses):
    python model2_dari_transkrip.py --api --watch

    # MODE FILE/FOLDER (kalau transkripnya berupa file lokal):
    python model2_dari_transkrip.py transcript_xxx.json
    python model2_dari_transkrip.py path/ke/folder_transkrip/
"""

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "qwen-ekstraksi-lora"
OUTPUT_DIR = Path("output_predictions")
DEFAULT_API = "http://127.0.0.1:8000"

SYSTEM_PROMPT = (
    "Kamu adalah sistem ekstraksi data transaksi toko sembako. "
    "Dari teks user, ekstrak SETIAP barang yang disebutkan. "
    "Balas HANYA dengan JSON array, tanpa penjelasan. "
    'Format tiap item: {"item": "nama barang", "qty": angka, "harga": angka_atau_null}. '
    "Kalau harga tidak disebut, isi null. Kalau qty tidak disebut, anggap 1."
)

# ---------------------------------------------------------------------------
# Helper murni (bisa ditest tanpa model)
# ---------------------------------------------------------------------------

def teks_dari_data(data, label: str = "") -> str:
    """Ambil teks transkrip dari berbagai kemungkinan bentuk JSON."""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        for key in ("text", "raw_text", "transcription", "transcript"):
            if key in data and isinstance(data[key], str):
                return data[key].strip()
    raise ValueError(f"Gak nemu field teks di {label or 'data'} — kirim contoh isinya ke Claude")


def daftar_nama_transkrip(data) -> list:
    """Baca daftar transkrip dari respon GET /transcripts (bentuknya toleran)."""
    if isinstance(data, dict):
        for k in ("transcripts", "files", "data", "results"):
            if k in data:
                data = data[k]
                break
    hasil = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                hasil.append(Path(item).name)
            elif isinstance(item, dict):
                for k in ("filename", "file", "name", "json_file"):
                    if k in item and isinstance(item[k], str):
                        hasil.append(Path(item[k]).name)
                        break
    return hasil




# ---- pencocokan ke katalog: toleransi typo ucapan / salah-dengar whisper ----
from difflib import SequenceMatcher

_ALIAS = None

def _get_alias():
    """Load kamus alias sekali. Kalau product_dictionary.json gak ada di folder
    ini, matching dilewati (items tetap keluar tanpa info katalog)."""
    global _ALIAS
    if _ALIAS is None:
        p = Path("product_dictionary.json")
        _ALIAS = {}
        if p.exists():
            kamus = json.loads(p.read_text(encoding="utf-8"))
            _ALIAS = {a.lower(): resmi for resmi, aliases in kamus.items() for a in aliases}
    return _ALIAS


def _skor_mirip(a: str, b: str) -> float:
    r_full = SequenceMatcher(None, a, b).ratio()
    ta, tb = a.split(), b.split()
    r_tok = (sum(max(SequenceMatcher(None, x, y).ratio() for y in tb) for x in ta) / len(ta)) if ta and tb else 0
    return max(r_full, r_tok)


def cocokkan_katalog(nama: str) -> dict:
    """Cari produk resmi yang paling mirip. Status:
    yakin (skor tinggi & jauh dari kandidat kedua) / perlu_konfirmasi (ambigu)
    / tidak_ketemu (kemungkinan produk di luar katalog)."""
    alias = _get_alias()
    nama = nama.lower().strip()
    if not alias:
        return {}
    if nama in alias:
        return {"produk_katalog": alias[nama], "skor_cocok": 1.0, "status_cocok": "yakin"}

    skor_per_produk = {}
    for a, resmi in alias.items():
        s = _skor_mirip(nama, a)
        if s > skor_per_produk.get(resmi, 0):
            skor_per_produk[resmi] = s
    urut = sorted(skor_per_produk.items(), key=lambda x: -x[1])
    (p1, s1) = urut[0]
    s2 = urut[1][1] if len(urut) > 1 else 0

    if s1 < 0.6:
        return {"produk_katalog": None, "skor_cocok": round(s1, 3), "status_cocok": "tidak_ketemu"}
    if s1 >= 0.75 and (s1 - s2) > 0.03:
        return {"produk_katalog": p1, "skor_cocok": round(s1, 3), "status_cocok": "yakin"}
    return {"produk_katalog": p1, "skor_cocok": round(s1, 3), "status_cocok": "perlu_konfirmasi",
            "kandidat_lain": urut[1][0] if len(urut) > 1 else None}


_HARGA = None

def _get_harga():
    """Load harga jual/modal per produk dari produk_master.csv (kalau ada di
    folder ini). Dipakai buat ngisi harga item yang gak disebut di ucapan."""
    global _HARGA
    if _HARGA is None:
        _HARGA = {}
        p = Path("produk_master.csv")
        if p.exists():
            import csv
            with p.open(encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    try:
                        _HARGA[row["nama_produk"]] = float(row["harga_jual"])
                    except (KeyError, ValueError, TypeError):
                        continue
    return _HARGA


# ---------------------------------------------------------------------------
# Model 2
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"Load Qwen + adapter '{ADAPTER_DIR}'...")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
        _model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        _model.eval()
        print(f"Siap. Device: {_model.device}\n")
    return _tokenizer, _model


def prediksi(teks: str) -> list:
    tokenizer, model = _load_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": teks},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def proses(nama_sumber: str, teks: str):
    print(f"[{nama_sumber}]")
    print(f"  Teks   : \"{teks}\"")
    try:
        items = prediksi(teks)
    except json.JSONDecodeError as e:
        print(f"  GAGAL  : output model bukan JSON sah ({e})\n")
        return
    if _get_alias():
        for it in items:
            if isinstance(it, dict) and it.get("item"):
                it.update(cocokkan_katalog(str(it["item"])))
                if it.get("harga") is not None:
                    it["sumber_harga"] = "ucapan"
                elif it.get("produk_katalog") and it["produk_katalog"] in _get_harga():
                    it["harga"] = _get_harga()[it["produk_katalog"]]
                    it["sumber_harga"] = "katalog"
    print(f"  Items  : {json.dumps(items, ensure_ascii=False)}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    tujuan = OUTPUT_DIR / f"pred_{Path(nama_sumber).stem}.json"
    tujuan.write_text(
        json.dumps({"sumber_transkrip": nama_sumber, "raw_text": teks, "items": items},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Simpan : {tujuan}\n")


def _sudah_diproses() -> set:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return {p.name for p in OUTPUT_DIR.glob("pred_*.json")}


def mode_api(base_url: str, diam_kalau_kosong: bool = False):
    import requests
    r = requests.get(f"{base_url}/transcripts", timeout=30)
    r.raise_for_status()
    names = daftar_nama_transkrip(r.json())
    if not names:
        if diam_kalau_kosong:
            return
        sys.exit(f"Gak bisa baca daftar transkrip. Respon mentah: {str(r.json())[:300]}")

    sudah = _sudah_diproses()
    baru = [n for n in names if f"pred_{Path(n).stem}.json" not in sudah]
    if not baru and diam_kalau_kosong:
        return
    print(f"{len(names)} transkrip ditemukan, {len(baru)} belum diproses.\n")
    for nama in baru:
        d = requests.get(f"{base_url}/transcripts/{nama}", timeout=30)
        d.raise_for_status()
        proses(nama, teks_dari_data(d.json(), nama))


def mode_watch(base_url: str, interval: int = 5):
    import requests
    print(f"MODE OTOMATIS: mantau transkrip baru tiap {interval} detik. Ctrl+C buat berhenti.")
    _load_model()  # load sekali di awal biar respon cepat
    while True:
        try:
            mode_api(base_url, diam_kalau_kosong=True)
        except requests.exceptions.ConnectionError:
            print("(service Model 1 gak kejangkau — pastiin uvicorn/docker nyala; coba lagi...)")
        except Exception as e:
            print(f"(error: {e} — lanjut mantau)")
        time.sleep(interval)


def mode_file(target: Path):
    if target.is_file():
        proses(target.name, teks_dari_data(json.loads(target.read_text(encoding="utf-8")), target.name))
    elif target.is_dir():
        files = sorted(target.glob("*.json"))
        if not files:
            sys.exit(f"Gak ada file .json di {target}")
        sudah = _sudah_diproses()
        baru = [f for f in files if f"pred_{f.stem}.json" not in sudah]
        print(f"{len(files)} transkrip ditemukan, {len(baru)} belum diproses.\n")
        for f in baru:
            proses(f.name, teks_dari_data(json.loads(f.read_text(encoding="utf-8")), f.name))
    else:
        sys.exit(f"Path gak ketemu: {target}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    watch = "--watch" in argv
    argv = [a for a in argv if a != "--watch"]

    if not argv and not watch:
        sys.exit("Pakai: python model2_dari_transkrip.py --api [--watch]  ATAU  <file.json|folder/>")

    if watch or (argv and argv[0] == "--api"):
        sisa = [a for a in argv if a != "--api"]
        url = sisa[0].rstrip("/") if sisa else DEFAULT_API
        if watch:
            mode_watch(url)
        else:
            mode_api(url)
    else:
        mode_file(Path(argv[0]))
