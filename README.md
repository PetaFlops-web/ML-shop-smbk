# Sembako UMKM ML Service

Pipeline: Audio → Whisper (transcription) → Qwen+LoRA (transaction item extraction)

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

### `POST /transcribe` — Audio → Transcription + Extraction

Upload an audio file. The backend transcribes it with Whisper then extracts transaction items with Qwen+LoRA, all in one call.

**Supported formats:** `.wav`, `.mp3`, `.flac`, `.m4a`

```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@recording.wav"
```

**Response:**

```json
{
  "sumber_transkrip": "recording.wav",
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
  "json_file": "/app/output_predictions/pred_recording.json"
}
```

| Field | Description |
|---|---|
| `sumber_transkrip` | Source audio/transcript name |
| `raw_text` | Whisper transcription text |
| `items` | Extracted transaction items |
| `json_file` | Path to saved extraction JSON file |

## Microphone Recording Test (Optional)

`test_mic.py` — records 10 seconds → sends to `/transcribe`.

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
│   ├── ekstraksi.py             # Transaction extraction (Qwen+LoRA)
│   ├── whisper_audio.py         # Whisper transcription logic
│   ├── qwen-ekstraksi-lora/     # LoRA adapter model
│   ├── product_dictionary.json  # Product catalog + aliases
│   ├── produk_master.csv        # Product selling prices
│   ├── requirements.txt
│   └── output_predictions/      # Extraction results
```