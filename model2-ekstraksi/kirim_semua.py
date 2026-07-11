"""
Kirim SEMUA file audio di folder ini ke Model 1 (whisper service) sekaligus.
File yang sudah pernah terkirim dicatat di sudah_terkirim.txt dan otomatis
dilewati — jadi aman dijalanin berulang-ulang.

Pakai (dari folder lora-training, service Model 1 harus nyala):
    python kirim_semua.py           # kirim semua yang belum pernah terkirim
    python kirim_semua.py --ulang   # paksa kirim ulang semuanya
Setelah itu:
    python model2_dari_transkrip.py --api
"""

import sys
from pathlib import Path

import requests

API = "http://127.0.0.1:8000/transcribe"
EKSTENSI = {".mp3", ".m4a", ".wav", ".flac"}
CATATAN = Path("sudah_terkirim.txt")


def main():
    ulang = "--ulang" in sys.argv
    sudah = set()
    if CATATAN.exists() and not ulang:
        sudah = set(CATATAN.read_text(encoding="utf-8").split())

    files = sorted(p for p in Path(".").iterdir()
                   if p.is_file() and p.suffix.lower() in EKSTENSI)
    if not files:
        sys.exit("Gak ada file audio (.mp3/.m4a/.wav/.flac) di folder ini")

    baru = [f for f in files if f.name not in sudah]
    print(f"{len(files)} audio ditemukan, {len(baru)} akan dikirim.\n")

    for f in baru:
        print(f"Kirim: {f.name} ...", flush=True)
        try:
            with open(f, "rb") as fh:
                r = requests.post(API, files={"file": (f.name, fh)}, timeout=300)
            r.raise_for_status()
            data = r.json()
            print(f'  Teks: "{data.get("text", "?")}"\n')
            sudah.add(f.name)
        except requests.exceptions.ConnectionError:
            sys.exit("Service Model 1 gak nyala — jalankan dulu uvicorn di folder ml-service")
        except Exception as e:
            print(f"  GAGAL ({e}) — lanjut ke file berikutnya\n")

    CATATAN.write_text("\n".join(sorted(sudah)), encoding="utf-8")
    print("Selesai. Lanjut: python model2_dari_transkrip.py --api")


if __name__ == "__main__":
    main()
