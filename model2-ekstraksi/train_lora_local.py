"""
Training LoRA Qwen versi LOKAL (buat dijalanin di terminal VSCode).
Isinya sama dengan train_lora_colab.ipynb, tapi bentuk script biasa.

Taruh file ini di folder yang sama dengan train.jsonl & val.jsonl.

Install dulu (sekali):
    pip install torch transformers peft trl datasets accelerate

Jalanin:
    python train_lora_local.py --uji-cepat   # tes dulu 50 contoh, 1 epoch (pastiin jalan)
    python train_lora_local.py               # training beneran (CPU: bisa berjam-jam!)

Hasil: folder qwen-ekstraksi-lora/ (adapter hasil tuning)
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uji-cepat", action="store_true",
                        help="training mini (50 contoh, 1 epoch) cuma buat mastiin pipeline jalan")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if not (os.path.exists("train.jsonl") and os.path.exists("val.jsonl")):
        sys.exit("train.jsonl / val.jsonl gak ketemu — jalankan dari folder lora-training!")

    pakai_gpu = torch.cuda.is_available()
    print(f"Device: {'GPU - ' + torch.cuda.get_device_name(0) if pakai_gpu else 'CPU'}")
    if not pakai_gpu and not args.uji_cepat:
        print("PERINGATAN: training di CPU bisa makan waktu BERJAM-JAM.")
        print("Saran: jalankan dengan --uji-cepat dulu buat mastiin semuanya jalan.\n")

    ds = load_dataset("json", data_files={"train": "train.jsonl", "val": "val.jsonl"})
    if args.uji_cepat:
        ds["train"] = ds["train"].select(range(50))
        args.epochs = 1
        print("[MODE UJI CEPAT] 50 contoh, 1 epoch — hasilnya belum akurat, ini cuma tes pipeline\n")

    print(f"Load model {MODEL_NAME} (pertama kali download ±1GB)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    sft_args = SFTConfig(
        output_dir="hasil-training",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        peft_config=lora_config,
        args=sft_args,
        processing_class=tokenizer,
    )
    trainer.train()

    # --- evaluasi cepat di data validasi ---
    model_ft = trainer.model
    model_ft.eval()
    system_prompt = ds["val"][0]["messages"][0]["content"]

    def prediksi(kalimat: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": kalimat},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model_ft.device)
        with torch.no_grad():
            out = model_ft.generate(**inputs, max_new_tokens=256, do_sample=False)
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    n_eval = 10 if args.uji_cepat else 30
    benar = 0
    for i in range(n_eval):
        contoh = ds["val"][i]
        kalimat = contoh["messages"][1]["content"]
        label = json.loads(contoh["messages"][2]["content"])
        try:
            keluaran = prediksi(kalimat).replace("```json", "").replace("```", "").strip()
            if json.loads(keluaran) == label:
                benar += 1
        except Exception:
            pass
        if i < 3:
            print(f"\nKalimat : {kalimat}")
            print(f"Label   : {label}")
            print(f"Model   : {prediksi(kalimat)}")

    print(f"\nAkurasi exact-match: {benar}/{n_eval} = {benar / n_eval * 100:.0f}%")

    trainer.save_model("qwen-ekstraksi-lora")
    tokenizer.save_pretrained("qwen-ekstraksi-lora")
    print("\nAdapter tersimpan di folder: qwen-ekstraksi-lora/")
    print("Cara pakai di qwen_extractor.py ada di CARA_PAKAI.txt / cell terakhir notebook.")


if __name__ == "__main__":
    main()
