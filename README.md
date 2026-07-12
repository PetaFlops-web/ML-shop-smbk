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

Upload an audio file. Backend will:

1. transcribe the audio with Whisper,
2. save the transcript to `output_transcriptions/`,
3. extract transaction items with Qwen+LoRA,
4. save the extraction result to `output_predictions/`,
5. return the extraction result.

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

### `GET /transcripts` — List Transcripts

```bash
curl http://127.0.0.1:8000/transcripts
```

**Response:**

```json
{
  "total": 1,
  "files": [
    {
      "filename": "transcript_20260710_214328.json",
      "path": "./output_transcriptions/transcript_20260710_214328.json",
      "size_kb": 0.12
    }
  ]
}
```

### `GET /transcripts/{filename}` — Transcript Detail

```bash
curl http://127.0.0.1:8000/transcripts/transcript_20260710_214328.json
```

### `GET /predictions` — List Extractions

```bash
curl http://127.0.0.1:8000/predictions
```

**Response:**

```json
{
  "total": 1,
  "files": [
    {
      "filename": "pred_recording.json",
      "path": "/app/output_predictions/pred_recording.json",
      "size_kb": 0.42
    }
  ]
}
```

### `GET /predictions/{filename}` — Extraction Detail

```bash
curl http://127.0.0.1:8000/predictions/pred_recording.json
```

## Microphone Recording Test (Optional)

`test_mic.py` — records 10 seconds → sends to `/transcribe` → displays `/predictions/{filename}` and `/predictions`.

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
│   ├── output_transcriptions/   # Transcription results (Model 1)
│   └── output_predictions/      # Extraction results (Model 2)
```