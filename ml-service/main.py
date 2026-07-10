from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
import os, glob, shutil, tempfile
from whisper_audio import transcribe_audio, output_dir

app = FastAPI(title="Whisper Audio Transcription API")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    allowed = {".wav", ".mp3", ".flac", ".m4a"}
    ext = os.path.splitext(file.filename)[-1].lower()

    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid file format: {ext}")

    # Save Audioke temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = transcribe_audio(tmp_path)
    finally:
        os.remove(tmp_path)

    return {
        "status"    : "success",
        "json_file" : result["json_file"], 
        "text"      : result["text"],
        "confidence": result["confidence"],
        "no_speech" : result["no_speech"],
    }

@app.get("/transcripts")
def list_transcripts():
    files = sorted(glob.glob(f"{output_dir}/transcript_*.json"), reverse=True)
    result = []
    for f in files:
        size = os.path.getsize(f)
        result.append({
            "filename": os.path.basename(f),
            "path"    : f,
            "size_kb" : round(size / 1024, 2),
        })
    return {"total": len(result), "files": result}

@app.get("/transcripts/{filename}", response_class=PlainTextResponse)
def get_transcript(filename: str):
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()