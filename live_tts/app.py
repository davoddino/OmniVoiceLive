from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from live_tts.config import LiveTTSConfig
from live_tts.llm import LLMStreamer
from live_tts.session import RealtimeSession
from live_tts.stt import STTService
from live_tts.tts import BaseTTS, create_tts


STATIC_DIR = Path(__file__).resolve().parent / "static"

config = LiveTTSConfig.from_env()
stt_service = STTService(config)
llm_streamer = LLMStreamer(config)
tts_service: BaseTTS = create_tts(config)

app = FastAPI(title="OmniVoice Live TTS", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup() -> None:
    await tts_service.start()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "ok": True,
        "tts_backend": config.tts_backend,
        "tts_sample_rate": tts_service.sample_rate,
        "stt_backend": config.stt_backend,
        "llm_backend": config.llm_backend,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    session = RealtimeSession(websocket, config, stt_service, llm_streamer, tts_service)
    await session.run()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "live_tts.app:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
