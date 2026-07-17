
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
import tempfile
from app.services.whisper_audio import transcribe_audio
from app.services.ekstraksi import extract_transaction
from app.services.restock_prediction import predict_restock



class RestockRequest(BaseModel):
    penjualan: list[dict]
    stok: list[dict]
    lead_time_hari: int = 2
    buffer_hari: int = 1


app = FastAPI(title="Sembako UMKM ML Service")




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


@app.post("/predict-restock")
async def restock_prediction(request: RestockRequest):
    try:
        return predict_restock(
            request.penjualan,
            request.stok,
            request.lead_time_hari,
            request.buffer_hari,
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc