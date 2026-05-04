from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_SYSTEM_PROMPT = """
Sei un assistente vocale call-center per CavadaLabs.
Rispondi sempre in italiano naturale e parlato.
Usa frasi brevi, concrete e facili da ascoltare.
Non usare markdown, elenchi, titoli, asterischi o codice.
Quando non hai abbastanza informazioni, fai una domanda breve.
Non superare 100 parole salvo necessita' reale.

CavadaLabs aiuta le aziende italiane a integrare intelligenza artificiale,
OCR, trascrizioni, knowledge management, server GPU, CRM, ERP e database.
""".strip()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LiveTTSConfig:
    host: str
    port: int
    reload: bool
    log_level: str
    ssl_certfile: str
    ssl_keyfile: str

    tts_backend: str
    tts_model: str
    tts_device_map: str
    tts_dtype: str
    tts_language: str
    tts_instruct: str
    tts_num_step_first: int
    tts_num_step_next: int
    tts_speed: float
    tts_frame_ms: int
    tts_warmup_enabled: bool
    tts_warmup_text: str
    tts_self_condition: bool
    tts_anchor_min_seconds: float

    stt_backend: str
    stt_url: str
    stt_language: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str

    llm_backend: str
    llm_url: str
    llm_model: str
    llm_temperature: float
    llm_top_p: float
    llm_timeout_s: float
    system_prompt: str

    vad_speech_threshold: float
    vad_start_ms: int
    vad_end_ms: int
    vad_min_turn_ms: int
    vad_preroll_ms: int
    vad_max_turn_s: float

    @classmethod
    def from_env(cls) -> "LiveTTSConfig":
        return cls(
            host=_env("LIVE_TTS_HOST", "0.0.0.0"),
            port=_env_int("LIVE_TTS_PORT", 8020),
            reload=_env_bool("LIVE_TTS_RELOAD", False),
            log_level=_env("LIVE_TTS_LOG_LEVEL", "info"),
            ssl_certfile=_env("LIVE_TTS_SSL_CERTFILE", ""),
            ssl_keyfile=_env("LIVE_TTS_SSL_KEYFILE", ""),
            tts_backend=_env("LIVE_TTS_TTS_BACKEND", "omnivoice"),
            tts_model=_env("LIVE_TTS_MODEL", "k2-fsa/OmniVoice"),
            tts_device_map=_env("LIVE_TTS_DEVICE_MAP", "cuda:0"),
            tts_dtype=_env("LIVE_TTS_DTYPE", "float16"),
            tts_language=_env("LIVE_TTS_LANGUAGE", "it"),
            tts_instruct=_env("LIVE_TTS_INSTRUCT", "female, low pitch"),
            tts_num_step_first=_env_int("LIVE_TTS_NUM_STEP_FIRST", 16),
            tts_num_step_next=_env_int("LIVE_TTS_NUM_STEP_NEXT", 24),
            tts_speed=_env_float("LIVE_TTS_SPEED", 1.05),
            tts_frame_ms=_env_int("LIVE_TTS_FRAME_MS", 40),
            tts_warmup_enabled=_env_bool("LIVE_TTS_WARMUP", True),
            tts_warmup_text=_env("LIVE_TTS_WARMUP_TEXT", "Ciao, sono pronta."),
            tts_self_condition=_env_bool("LIVE_TTS_SELF_CONDITION", True),
            tts_anchor_min_seconds=_env_float("LIVE_TTS_ANCHOR_MIN_SECONDS", 0.45),
            stt_backend=_env("LIVE_TTS_STT_BACKEND", "auto"),
            stt_url=_env("LIVE_TTS_STT_URL", ""),
            stt_language=_env("LIVE_TTS_STT_LANGUAGE", "it"),
            whisper_model=_env("LIVE_TTS_WHISPER_MODEL", "small"),
            whisper_device=_env("LIVE_TTS_WHISPER_DEVICE", "cuda"),
            whisper_compute_type=_env(
                "LIVE_TTS_WHISPER_COMPUTE_TYPE", "int8_float16"
            ),
            llm_backend=_env("LIVE_TTS_LLM_BACKEND", "openai"),
            llm_url=_env(
                "LIVE_TTS_LLM_URL",
                "http://192.168.0.20:8001/v1/chat/completions",
            ),
            llm_model=_env("LIVE_TTS_LLM_MODEL", "qwen3.6-35b"),
            llm_temperature=_env_float("LIVE_TTS_LLM_TEMPERATURE", 0.0),
            llm_top_p=_env_float("LIVE_TTS_LLM_TOP_P", 1.0),
            llm_timeout_s=_env_float("LIVE_TTS_LLM_TIMEOUT_S", 120.0),
            system_prompt=_env("LIVE_TTS_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            vad_speech_threshold=_env_float("LIVE_TTS_VAD_THRESHOLD", 0.014),
            vad_start_ms=_env_int("LIVE_TTS_VAD_START_MS", 140),
            vad_end_ms=_env_int("LIVE_TTS_VAD_END_MS", 650),
            vad_min_turn_ms=_env_int("LIVE_TTS_VAD_MIN_TURN_MS", 320),
            vad_preroll_ms=_env_int("LIVE_TTS_VAD_PREROLL_MS", 220),
            vad_max_turn_s=_env_float("LIVE_TTS_VAD_MAX_TURN_S", 18.0),
        )
