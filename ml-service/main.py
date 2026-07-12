
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
import tempfile
from whisper_audio import transcribe_audio
from ekstraksi import extract_transaction

app = FastAPI(title="Sembako UMKM ML Service")


# Whisper Audio Transcription

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    allowed = {".wav", ".mp3", ".flac", ".m4a"}
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid file format: {ext}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path)
        
        text = transcript["text"]

        result = extract_transaction(text, file.filename)
    finally:
        os.remove(tmp_path)

    return result 