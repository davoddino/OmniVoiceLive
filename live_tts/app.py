from __future__ import annotations

import logging
from importlib.util import find_spec
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
logger = logging.getLogger(__name__)

config = LiveTTSConfig.from_env()
stt_service = STTService(config)
llm_streamer = LLMStreamer(config)
tts_service: BaseTTS = create_tts(config)

app = FastAPI(title="OmniVoice Live TTS", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup() -> None:
    ensure_websocket_support()
    logger.info(
        "live_tts startup tts_backend=%s stt_backend=%s llm_backend=%s ssl=%s session_voice_anchor=%s startup_voice_anchor=%s client_barge_threshold=%.4f",
        config.tts_backend,
        config.stt_backend,
        config.llm_backend,
        bool(config.ssl_certfile and config.ssl_keyfile),
        config.tts_session_voice_anchor,
        config.tts_startup_voice_anchor,
        config.client_barge_threshold,
    )
    await tts_service.start()
    logger.info("live_tts ready tts_sample_rate=%s", tts_service.sample_rate)


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
        "websocket_support": has_websocket_support(),
        "client_barge_threshold": config.client_barge_threshold,
        "client_barge_stop_ms": config.client_barge_stop_ms,
        "client_barge_commit_ms": config.client_barge_commit_ms,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    session = RealtimeSession(websocket, config, stt_service, llm_streamer, tts_service)
    await session.run()


def main() -> None:
    import uvicorn

    level_name = config.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    if bool(config.ssl_certfile) != bool(config.ssl_keyfile):
        raise RuntimeError(
            "Set both LIVE_TTS_SSL_CERTFILE and LIVE_TTS_SSL_KEYFILE, or neither."
        )
    ensure_websocket_support()

    uvicorn_kwargs = {
        "host": config.host,
        "port": config.port,
        "reload": config.reload,
        "log_level": config.log_level.lower(),
    }
    if config.ssl_certfile and config.ssl_keyfile:
        uvicorn_kwargs["ssl_certfile"] = config.ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = config.ssl_keyfile

    uvicorn.run("live_tts.app:app", **uvicorn_kwargs)


def has_websocket_support() -> bool:
    return find_spec("websockets") is not None or find_spec("wsproto") is not None


def ensure_websocket_support() -> None:
    if has_websocket_support():
        return
    raise RuntimeError(
        "WebSocket support is not installed. Install project dependencies again "
        "so uvicorn can serve /ws: uv sync, or uv add websockets."
    )
