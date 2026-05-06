from __future__ import annotations

import logging
from importlib.util import find_spec
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from live_tts.config import LiveTTSConfig
from live_tts.llm import LLMStreamer
from live_tts.rag import RAGRetriever
from live_tts.session import RealtimeSession
from live_tts.stt import STTService
from live_tts.tts import BaseTTS, create_tts


STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger(__name__)

config = LiveTTSConfig.from_env()
stt_service = STTService(config)
llm_streamer = LLMStreamer(config)
tts_service: BaseTTS = create_tts(config)
rag_retriever = RAGRetriever(config)

app = FastAPI(title="OmniVoice Live TTS", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup() -> None:
    ensure_websocket_support()
    logger.info(
        "live_tts startup tts_backend=%s stt_backend=%s llm_backend=%s ssl=%s voice_mode=%s reference_audio=%r instruct=%r num_step=%s/%s speed=%.2f segments=%s-%s/%s-%s vad_threshold=%.4f vad_adaptive=%s rag_enabled=%s recording_enabled=%s client_barge_threshold=%.4f",
        config.tts_backend,
        config.stt_backend,
        config.llm_backend,
        bool(config.ssl_certfile and config.ssl_keyfile),
        config.tts_voice_mode,
        config.tts_reference_audio if config.tts_voice_mode == "fixed_reference" else "",
        config.tts_instruct,
        config.tts_num_step_first,
        config.tts_num_step_next,
        config.tts_speed,
        config.segment_min_first_chars,
        config.segment_max_first_chars,
        config.segment_min_next_chars,
        config.segment_max_next_chars,
        config.vad_speech_threshold,
        config.vad_adaptive,
        config.rag_enabled,
        config.recording_enabled,
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
        "tts_voice_mode": config.tts_voice_mode,
        "tts_voice_id": config.tts_voice_id,
        "tts_reference_audio": config.tts_reference_audio,
        "tts_instruct": config.tts_instruct,
        "tts_language": config.tts_language,
        "stt_language": config.stt_language,
        "tts_num_step_first": config.tts_num_step_first,
        "tts_num_step_next": config.tts_num_step_next,
        "tts_speed": config.tts_speed,
        "tts_position_temperature": config.tts_position_temperature,
        "tts_class_temperature": config.tts_class_temperature,
        "tts_loudness_target_lufs": config.tts_loudness_target_lufs,
        "tts_crossfade_ms": config.tts_crossfade_ms,
        "segment_min_first_chars": config.segment_min_first_chars,
        "segment_max_first_chars": config.segment_max_first_chars,
        "segment_min_next_chars": config.segment_min_next_chars,
        "segment_max_next_chars": config.segment_max_next_chars,
        "vad_speech_threshold": config.vad_speech_threshold,
        "vad_adaptive": config.vad_adaptive,
        "vad_noise_calibration_ms": config.vad_noise_calibration_ms,
        "vad_start_multiplier": config.vad_start_multiplier,
        "vad_continue_multiplier": config.vad_continue_multiplier,
        "vad_start_ms": config.vad_start_ms,
        "vad_end_ms": config.vad_end_ms,
        "websocket_support": has_websocket_support(),
        "client_barge_threshold": config.client_barge_threshold,
        "client_barge_stop_ms": config.client_barge_stop_ms,
        "client_barge_commit_ms": config.client_barge_commit_ms,
        "rag_enabled": config.rag_enabled,
        "rag_timeout_ms": config.rag_timeout_ms,
        "rag_max_chunks": config.rag_max_chunks,
        "recording_enabled": config.recording_enabled,
        "recording_dir": config.recording_dir,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    session = RealtimeSession(
        websocket,
        config,
        stt_service,
        llm_streamer,
        tts_service,
        rag_retriever,
    )
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
