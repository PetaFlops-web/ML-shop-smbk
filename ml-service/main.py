
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
import glob
import os
import shutil
import tempfile
from whisper_audio import transcribe_audio, output_dir
from ekstraksi import extract_transaction, read_prediction, list_predictions as get_all_predictions

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


@app.get("/transcripts")
def list_transcripts():
    files = sorted(glob.glob(f"{output_dir}/transcript_*.json"), reverse=True)
    result = []
    for f in files:
        size = os.path.getsize(f)
        result.append({
            "filename": os.path.basename(f),
            "path": f,
            "size_kb": round(size / 1024, 2),
        })
    return {"total": len(result), "files": result}


@app.get("/transcripts/{filename}", response_class=PlainTextResponse)
def get_transcript(filename: str):
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Ekstraksi Transaksi

@app.get("/predictions")
def predictions_list():
    return {"total": len(get_all_predictions()), "files": get_all_predictions()}


@app.get("/predictions/{filename}")
def get_prediction(filename: str):
    try:
        return read_prediction(filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Prediction tidak ditemukan")