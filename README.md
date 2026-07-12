# Sembako UMKM ML Service

Pipeline: Audio → Whisper (transkripsi) → Qwen+LoRA (ekstraksi item transaksi)

## Requirements

- Docker & Docker Compose
- _(optional)_ Python 3.10+ & pip — for local microphone test

## Start

```bash
docker compose up --build
```

Base URL: `http://127.0.0.1:8000`.

Stop:

```bash
docker compose down
```

## API Endpoints

### Model 1 — Whisper Transcription

#### `POST /transcribe` — Audio → Transkrip

Upload audio file, dapat transkrip JSON.

**Supported formats:** `.wav`, `.mp3`, `.flac`, `.m4a`

```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@rekaman.wav"
```

**Response:**

```json
{
  "status": "success",
  "json_file": "./output_transcriptions/transcript_20260710_214328.json",
  "text": "beras 2 kilo harga 25.000",
  "confidence": -0.2341,
  "no_speech": 0.0123
}
```

| Field | Description |
|---|---|
| `status` | Status transkripsi |
| `json_file` | Path file JSON transkrip yang disimpan |
| `text` | Teks hasil transkripsi |
| `confidence` | Semakin mendekati 0 → model semakin yakin |
| `no_speech` | Probabilitas tidak ada suara (0–1), semakin rendah semakin baik |

#### `GET /transcripts` — Daftar Transkrip

```bash
curl http://127.0.0.1:8000/transcripts
```

#### `GET /transcripts/{filename}` — Detail Transkrip

```bash
curl http://127.0.0.1:8000/transcripts/transcript_20260710_214328.json
```

---

### Model 2 — Ekstraksi Transaksi (Qwen+LoRA)

#### `POST /extract` — Teks → Item Transaksi

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "gula pasir 2 kilo 28.000, telur ayam 1 kilo", "source": "manual"}'
```

**Response:**

```json
{
  "sumber_transkrip": "manual",
  "raw_text": "gula pasir 2 kilo 28.000, telur ayam 1 kilo",
  "items": [
    {
      "item": "gula pasir",
      "qty": 2,
      "harga": 28000,
      "sumber_harga": "ucapan",
      "produk_katalog": "Gula Pasir Gulaku 1kg",
      "skor_cocok": 0.95,
      "status_cocok": "yakin"
    }
  ],
  "json_file": "output_predictions/pred_manual.json"
}
```

#### `POST /extract/transcript/{filename}` — Ekstrak dari File Transkrip

```bash
curl -X POST http://127.0.0.1:8000/extract/transcript/transcript_20260710_214328.json
```

#### `GET /predictions` — Daftar Hasil Prediksi

```bash
curl http://127.0.0.1:8000/predictions
```

#### `GET /predictions/{filename}` — Detail Prediksi

```bash
curl http://127.0.0.1:8000/predictions/pred_manual.json
```


---

## Microphone Recording Test (Optional)

`test_mic.py` — rekam 10 detik → kirim ke `/transcribe`, lanjut `/extract/transcript/{filename}`, lalu tampilkan `/predictions/{filename}` dan `/predictions`.

```bash
pip install sounddevice scipy requests
python test_mic.py
```

## Project Structure

```
.
├── compose.yml
├── test_mic.py
├── ml-service/
│   ├── Dockerfile
│   ├── main.py                  # FastAPI endpoint definitions
│   ├── ekstraksi.py              # Ekstraksi transaksi (Qwen+LoRA)
│   ├── whisper_audio.py         # Whisper transcription logic
│   ├── qwen-ekstraksi-lora/     # LoRA adapter model
│   ├── product_dictionary.json  # Katalog produk + alias
│   ├── produk_master.csv        # Harga jual produk
│   ├── requirements.txt
│   ├── output_transcriptions/   # Hasil transkrip (Model 1)
│   └── output_predictions/      # Hasil prediksi (Model 2)
```