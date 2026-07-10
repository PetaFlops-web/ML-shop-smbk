# Speech-to-Text API

## Requirements

- Docker & Docker Compose
- _(optional)_ Python 3.10+ & pip — for local microphone test

## Start

```bash
docker compose up --build
```

Base url : `http://127.0.0.1:8000`.

Stop Docker compose:

```bash
docker compose down
```

## API Endpoints

### `POST /transcribe` — Audio Transcription

Upload an audio file, get transcription as JSON.

**Supported formats:** `.wav`, `.mp3`, `.flac`, `.m4a`

**Example with curl:**

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
| `status` | Transcription status |
| `json_file` | Saved JSON transcript path |
| `text` | Transcribed text |
| `confidence` | Closer to 0 means better model accuracy |
| `no_speech` | Probability of no speech (0–1), lower is better |

### `GET /transcripts` — All Transcripts

```bash
curl http://127.0.0.1:8000/transcripts
```

### `GET /transcripts/{filename}` — Transcript Detail
```bash
curl http://127.0.0.1:8000/transcripts/transcript_20260710_214328.json
```

## Microphone Recording Test (Optional)

Script `test_mic.py` records 5 seconds of audio from local microphone and sends it to the API.

```bash
pip install sounddevice scipy requests
python test_mic.py
```

## Project Structure

```
.
├── compose.yml              # Docker Compose config
├── test_mic.py              # Microphone recording test script
├── ml-service/
│   ├── Dockerfile
│   ├── main.py              # FastAPI app endpoint definitions
│   ├── whisper_audio.py     # Whisper transcription logic
│   ├── requirements.txt
│   └── output_transcriptions/  # Saved transcript
└── output_transcriptions/      # Volume mount from container
```
