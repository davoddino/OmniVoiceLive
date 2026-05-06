from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "cavadalabs_voice.md"
)
DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


_ENV_FILES_LOADED = False


def load_env_files() -> None:
    global _ENV_FILES_LOADED
    if _ENV_FILES_LOADED:
        return
    _ENV_FILES_LOADED = True
    protected_keys = set(os.environ)

    explicit = os.getenv("LIVE_TTS_ENV_FILE")
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            repo_root / "live_tts.env",
            Path.cwd() / "live_tts.env",
            repo_root / ".env",
            Path.cwd() / ".env",
        ]

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_env_file(resolved, protected_keys)


def _load_env_file(path: Path, protected_keys: set[str]) -> None:
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
        if key not in protected_keys:
            os.environ[key] = _parse_env_value(value)


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


def _env_any(names: tuple[str, ...], default: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw != "":
            return raw
    return default


def _env_int_any(names: tuple[str, ...], default: int) -> int:
    raw = _env_any(names, "")
    return default if raw == "" else int(raw)


def _env_float_any(names: tuple[str, ...], default: float) -> float:
    raw = _env_any(names, "")
    return default if raw == "" else float(raw)


def _env_bool_any(names: tuple[str, ...], default: bool) -> bool:
    raw = _env_any(names, "")
    if raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _configured_system_prompt() -> str:
    inline_prompt = os.getenv("LIVE_TTS_SYSTEM_PROMPT")
    if inline_prompt:
        return inline_prompt

    prompt_file = os.getenv("LIVE_TTS_SYSTEM_PROMPT_FILE")
    if prompt_file:
        return Path(prompt_file).expanduser().read_text(encoding="utf-8").strip()

    return DEFAULT_SYSTEM_PROMPT


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
    tts_voice_id: str
    tts_device_map: str
    tts_dtype: str
    tts_language: str
    tts_instruct: str
    tts_voice_mode: str
    tts_num_step_first: int
    tts_num_step_next: int
    tts_speed: float
    tts_guidance_scale: float
    tts_stability: float
    tts_similarity_boost: float
    tts_style: float
    tts_temperature: float
    tts_seed: int | None
    tts_position_temperature: float
    tts_class_temperature: float
    tts_postprocess_output: bool
    tts_denoise: bool
    tts_frame_ms: int
    tts_output_format: str
    tts_loudness_target_lufs: float
    tts_loudness_enabled: bool
    tts_crossfade_ms: int
    tts_crossfade_enabled: bool
    tts_warmup_enabled: bool
    tts_warmup_text: str
    tts_self_condition: bool
    tts_anchor_min_seconds: float
    tts_anchor_max_seconds: float
    tts_session_voice_anchor: bool
    tts_startup_voice_anchor: bool
    tts_startup_anchor_text: str
    tts_reference_audio: str
    tts_reference_text: str

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

    segment_min_first_chars: int
    segment_max_first_chars: int
    segment_min_next_chars: int
    segment_max_next_chars: int

    vad_speech_threshold: float
    vad_adaptive: bool
    vad_noise_calibration_ms: int
    vad_start_multiplier: float
    vad_continue_multiplier: float
    vad_start_ms: int
    vad_end_ms: int
    vad_min_turn_ms: int
    vad_preroll_ms: int
    vad_max_turn_s: float
    client_barge_threshold: float
    client_barge_stop_ms: int
    client_barge_commit_ms: int
    client_barge_cooldown_ms: int

    rag_enabled: bool
    rag_docs_dir: str
    rag_timeout_ms: int
    rag_max_chunks: int
    rag_max_context_chars: int
    rag_fallback_to_llm: bool

    recording_enabled: bool
    recording_dir: str
    recording_queue_size: int
    recording_prebuffer_seconds: float

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
            tts_voice_id=_env("LIVE_TTS_VOICE_ID", ""),
            tts_device_map=_env("LIVE_TTS_DEVICE_MAP", "cuda:0"),
            tts_dtype=_env("LIVE_TTS_DTYPE", "float16"),
            tts_language=_env("LIVE_TTS_LANGUAGE", "it"),
            tts_instruct=_env("LIVE_TTS_INSTRUCT", "male, middle-aged, low pitch"),
            tts_voice_mode=_env("LIVE_TTS_VOICE_MODE", "session_anchor"),
            tts_num_step_first=_env_int("LIVE_TTS_NUM_STEP_FIRST", 40),
            tts_num_step_next=_env_int("LIVE_TTS_NUM_STEP_NEXT", 40),
            tts_speed=_env_float("LIVE_TTS_SPEED", 1.05),
            tts_guidance_scale=_env_float("LIVE_TTS_GUIDANCE_SCALE", 2.0),
            tts_stability=_env_float("LIVE_TTS_STABILITY", 0.90),
            tts_similarity_boost=_env_float("LIVE_TTS_SIMILARITY_BOOST", 0.80),
            tts_style=_env_float("LIVE_TTS_STYLE", 0.15),
            tts_temperature=_env_float("LIVE_TTS_TEMPERATURE", 0.0),
            tts_seed=_env_optional_int("LIVE_TTS_SEED"),
            tts_position_temperature=_env_float(
                "LIVE_TTS_POSITION_TEMPERATURE", 0.0
            ),
            tts_class_temperature=_env_float("LIVE_TTS_CLASS_TEMPERATURE", 0.0),
            tts_postprocess_output=_env_bool("LIVE_TTS_POSTPROCESS_OUTPUT", False),
            tts_denoise=_env_bool("LIVE_TTS_DENOISE", True),
            tts_frame_ms=_env_int("LIVE_TTS_FRAME_MS", 40),
            tts_output_format=_env("LIVE_TTS_OUTPUT_FORMAT", "pcm16"),
            tts_loudness_target_lufs=_env_float(
                "LIVE_TTS_LOUDNESS_TARGET_LUFS", -16.0
            ),
            tts_loudness_enabled=_env_bool("LIVE_TTS_LOUDNESS_NORMALIZATION", True),
            tts_crossfade_ms=_env_int("LIVE_TTS_CROSSFADE_MS", 20),
            tts_crossfade_enabled=_env_bool("LIVE_TTS_CROSSFADE", True),
            tts_warmup_enabled=_env_bool("LIVE_TTS_WARMUP", True),
            tts_warmup_text=_env("LIVE_TTS_WARMUP_TEXT", "Ciao, sono qui."),
            tts_self_condition=_env_bool("LIVE_TTS_SELF_CONDITION", True),
            tts_anchor_min_seconds=_env_float("LIVE_TTS_ANCHOR_MIN_SECONDS", 1.6),
            tts_anchor_max_seconds=_env_float("LIVE_TTS_ANCHOR_MAX_SECONDS", 6.0),
            tts_session_voice_anchor=_env_bool(
                "LIVE_TTS_SESSION_VOICE_ANCHOR", False
            ),
            tts_startup_voice_anchor=_env_bool(
                "LIVE_TTS_STARTUP_VOICE_ANCHOR", False
            ),
            tts_startup_anchor_text=_env(
                "LIVE_TTS_STARTUP_ANCHOR_TEXT",
                (
                    "Parlo in italiano con voce maschile, calma, chiara e "
                    "professionale."
                ),
            ),
            tts_reference_audio=_env(
                "LIVE_TTS_REFERENCE_AUDIO",
                "voice_candidates/14.wav",
            ),
            tts_reference_text=_env(
                "LIVE_TTS_REFERENCE_TEXT",
                (
                    "Ciao! Certo, ti aiuto volentieri. Con CavadaLabs possiamo "
                    "creare un chatbot per il tuo sito, collegarlo ai contenuti "
                    "aziendali e renderlo semplice da aggiornare. Partiamo dalle "
                    "tue esigenze e scegliamo insieme la soluzione piu adatta."
                ),
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
            llm_temperature=_env_float("LIVE_TTS_LLM_TEMPERATURE", 0.3),
            llm_top_p=_env_float("LIVE_TTS_LLM_TOP_P", 1.0),
            llm_timeout_s=_env_float("LIVE_TTS_LLM_TIMEOUT_S", 120.0),
            system_prompt=_configured_system_prompt(),
            segment_min_first_chars=_env_int("LIVE_TTS_SEGMENT_MIN_FIRST_CHARS", 90),
            segment_max_first_chars=_env_int("LIVE_TTS_SEGMENT_MAX_FIRST_CHARS", 280),
            segment_min_next_chars=_env_int("LIVE_TTS_SEGMENT_MIN_NEXT_CHARS", 120),
            segment_max_next_chars=_env_int("LIVE_TTS_SEGMENT_MAX_NEXT_CHARS", 380),
            vad_speech_threshold=_env_float("LIVE_TTS_VAD_THRESHOLD", 0.020),
            vad_adaptive=_env_bool_any(
                ("LIVE_TTS_VAD_ADAPTIVE", "VAD_ADAPTIVE"), True
            ),
            vad_noise_calibration_ms=_env_int_any(
                ("LIVE_TTS_VAD_NOISE_CALIBRATION_MS", "VAD_NOISE_CALIBRATION_MS"),
                1000,
            ),
            vad_start_multiplier=_env_float_any(
                ("LIVE_TTS_VAD_START_MULTIPLIER", "VAD_START_MULTIPLIER"), 2.2
            ),
            vad_continue_multiplier=_env_float_any(
                ("LIVE_TTS_VAD_CONTINUE_MULTIPLIER", "VAD_CONTINUE_MULTIPLIER"),
                1.4,
            ),
            vad_start_ms=_env_int_any(
                (
                    "LIVE_TTS_VAD_START_MS",
                    "LIVE_TTS_VAD_MIN_SPEECH_MS",
                    "VAD_MIN_SPEECH_MS",
                ),
                180,
            ),
            vad_end_ms=_env_int_any(
                (
                    "LIVE_TTS_VAD_END_MS",
                    "LIVE_TTS_VAD_END_SILENCE_MS",
                    "VAD_END_SILENCE_MS",
                ),
                500,
            ),
            vad_min_turn_ms=_env_int("LIVE_TTS_VAD_MIN_TURN_MS", 320),
            vad_preroll_ms=_env_int("LIVE_TTS_VAD_PREROLL_MS", 220),
            vad_max_turn_s=_env_float("LIVE_TTS_VAD_MAX_TURN_S", 18.0),
            client_barge_threshold=_env_float(
                "LIVE_TTS_CLIENT_BARGE_THRESHOLD", 0.022
            ),
            client_barge_stop_ms=_env_int("LIVE_TTS_CLIENT_BARGE_STOP_MS", 80),
            client_barge_commit_ms=_env_int("LIVE_TTS_CLIENT_BARGE_COMMIT_MS", 140),
            client_barge_cooldown_ms=_env_int(
                "LIVE_TTS_CLIENT_BARGE_COOLDOWN_MS", 900
            ),
            rag_enabled=_env_bool_any(("LIVE_TTS_RAG_ENABLED", "RAG_ENABLED"), False),
            rag_docs_dir=_env("LIVE_TTS_RAG_DOCS_DIR", "rag_docs"),
            rag_timeout_ms=_env_int_any(("LIVE_TTS_RAG_TIMEOUT_MS", "RAG_TIMEOUT_MS"), 300),
            rag_max_chunks=_env_int_any(("LIVE_TTS_RAG_MAX_CHUNKS", "RAG_MAX_CHUNKS"), 3),
            rag_max_context_chars=_env_int_any(
                ("LIVE_TTS_RAG_MAX_CONTEXT_CHARS", "RAG_MAX_CONTEXT_CHARS"), 2500
            ),
            rag_fallback_to_llm=_env_bool_any(
                ("LIVE_TTS_RAG_FALLBACK_TO_LLM", "RAG_FALLBACK_TO_LLM"), True
            ),
            recording_enabled=_env_bool("LIVE_TTS_RECORDING_ENABLED", True),
            recording_dir=_env("LIVE_TTS_RECORDING_DIR", "recordings"),
            recording_queue_size=_env_int("LIVE_TTS_RECORDING_QUEUE_SIZE", 256),
            recording_prebuffer_seconds=_env_float(
                "LIVE_TTS_RECORDING_PREBUFFER_SECONDS", 3.0
            ),
        )
