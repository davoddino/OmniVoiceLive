from __future__ import annotations

from dataclasses import asdict, dataclass

from live_tts.config import LiveTTSConfig


@dataclass(frozen=True)
class VoiceSessionConfig:
    provider: str
    voice_id: str
    model: str
    speed: float
    stability: float
    similarity_boost: float | None
    style: float | None
    temperature: float | None
    seed: int | None
    sample_rate: int
    output_format: str
    loudness_target_lufs: float
    loudness_enabled: bool
    crossfade_ms: int
    guidance_scale: float
    position_temperature: float
    class_temperature: float
    num_step_first: int
    num_step_next: int
    voice_mode: str
    instruct: str
    language: str

    @classmethod
    def from_config(
        cls,
        config: LiveTTSConfig,
        sample_rate: int,
        language: str | None = None,
    ) -> "VoiceSessionConfig":
        voice_mode = config.tts_voice_mode.strip().lower().replace("-", "_")
        instruct = config.tts_instruct.strip()
        voice_id = config.tts_voice_id.strip()
        if not voice_id:
            voice_id = f"{config.tts_backend}:{voice_mode}:{instruct or 'auto'}"
        return cls(
            provider=config.tts_backend,
            voice_id=voice_id,
            model=config.tts_model,
            speed=config.tts_speed,
            stability=config.tts_stability,
            similarity_boost=config.tts_similarity_boost,
            style=config.tts_style,
            temperature=config.tts_temperature,
            seed=config.tts_seed,
            sample_rate=sample_rate,
            output_format=config.tts_output_format,
            loudness_target_lufs=config.tts_loudness_target_lufs,
            loudness_enabled=config.tts_loudness_enabled,
            crossfade_ms=(
                config.tts_crossfade_ms if config.tts_crossfade_enabled else 0
            ),
            guidance_scale=config.tts_guidance_scale,
            position_temperature=config.tts_position_temperature,
            class_temperature=config.tts_class_temperature,
            num_step_first=config.tts_num_step_first,
            num_step_next=config.tts_num_step_next,
            voice_mode=voice_mode,
            instruct=instruct,
            language=language or config.tts_language,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
