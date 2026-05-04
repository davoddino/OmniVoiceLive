import tempfile
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel

app = FastAPI(title="cavadatts-stt")

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="int8_float16",
)

@app.get("/health")
def health():
    return {"ok": True, "engine": "faster-whisper", "model": "small"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()

        segments, info = model.transcribe(
            tmp.name,
            language="it",
            vad_filter=True,
            beam_size=1,
            temperature=0.0,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()

    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
    }