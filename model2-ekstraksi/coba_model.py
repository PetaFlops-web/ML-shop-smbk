"""
Coba model hasil training (adapter LoRA) secara interaktif.

Taruh file ini di folder lora-training (sebelahan sama folder qwen-ekstraksi-lora).

Pakai:
    python coba_model.py
    -> ketik kalimat transaksi, Enter, model jawab JSON
    -> ketik 'exit' buat keluar

Atau sekali jalan:
    python coba_model.py "indomie kari ayam 5 bungkus 15 ribu"
"""

import json
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # harus sama dengan yang dipakai training
ADAPTER_DIR = "qwen-ekstraksi-lora"          # folder hasil training

SYSTEM_PROMPT = (
    "Kamu adalah sistem ekstraksi data transaksi toko sembako. "
    "Dari teks user, ekstrak SETIAP barang yang disebutkan. "
    "Balas HANYA dengan JSON array, tanpa penjelasan. "
    'Format tiap item: {"item": "nama barang", "qty": angka, "harga": angka_atau_null}. '
    "Kalau harga tidak disebut, isi null. Kalau qty tidak disebut, anggap 1."
)

print(f"Load model {BASE_MODEL} + adapter {ADAPTER_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
print(f"Siap! Device: {model.device}\n")


def ekstrak(kalimat: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": kalimat},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:  # rapikan tampilannya kalau JSON sah
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return f"(output bukan JSON sah) {raw}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(ekstrak(" ".join(sys.argv[1:])))
    else:
        print("Ketik kalimat transaksi (atau 'exit' buat keluar):")
        while True:
            kalimat = input("\n> ").strip()
            if kalimat.lower() in ("exit", "quit", "keluar"):
                break
            if kalimat:
                print(ekstrak(kalimat))
