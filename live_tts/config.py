from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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


_ENV_FILES_LOADED = False


def load_env_files() -> None:
    global _ENV_FILES_LOADED
    if _ENV_FILES_LOADED:
        return
    _ENV_FILES_LOADED = True

    explicit = os.getenv("LIVE_TTS_ENV_FILE")
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path.cwd() / ".env",
            Path.cwd() / "live_tts.env",
            repo_root / ".env",
            repo_root / "live_tts.env",
        ]

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_env_file(resolved)


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        os.environ.setdefault(key, _parse_env_value(value))


def _parse_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


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
    tts_session_voice_anchor: bool
    tts_startup_voice_anchor: bool
    tts_startup_anchor_text: str

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
    client_barge_threshold: float
    client_barge_stop_ms: int
    client_barge_commit_ms: int
    client_barge_cooldown_ms: int

    @classmethod
    def from_env(cls) -> "LiveTTSConfig":
        load_env_files()
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
            tts_session_voice_anchor=_env_bool(
                "LIVE_TTS_SESSION_VOICE_ANCHOR", True
            ),
            tts_startup_voice_anchor=_env_bool(
                "LIVE_TTS_STARTUP_VOICE_ANCHOR", True
            ),
            tts_startup_anchor_text=_env(
                "LIVE_TTS_STARTUP_ANCHOR_TEXT",
                "Buongiorno, sono pronta ad aiutarti. Dimmi pure di cosa hai bisogno.",
            ),
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
            client_barge_threshold=_env_float(
                "LIVE_TTS_CLIENT_BARGE_THRESHOLD", 0.012
            ),
            client_barge_stop_ms=_env_int("LIVE_TTS_CLIENT_BARGE_STOP_MS", 20),
            client_barge_commit_ms=_env_int("LIVE_TTS_CLIENT_BARGE_COMMIT_MS", 45),
            client_barge_cooldown_ms=_env_int(
                "LIVE_TTS_CLIENT_BARGE_COOLDOWN_MS", 700
            ),
        )
