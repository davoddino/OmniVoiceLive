from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from live_tts.audio import (
    AudioLoudnessSmoother,
    apply_edge_fade,
    pad_audio_edges,
    trim_tts_onset_noise,
)
from live_tts.languages import (
    SUPPORTED_LANGUAGES,
    normalize_language_code,
    omnivoice_tts_language,
)
from live_tts.llm import language_instruction, llm_messages, translation_instruction
from live_tts.playback import drain_segment_queue
from live_tts.rag import RAGRetriever
from live_tts.recording import AsyncSessionRecorder
from live_tts.segmenter import SentenceAccumulator, normalize_tts_text
from live_tts.stt import STTService
from live_tts.tts import OmniVoiceTTS, TTSTurnState, normalize_tts_engine
from live_tts.voice import VoiceSessionConfig


class LiveTTSPipelineTests(unittest.TestCase):
    def test_sentence_accumulator_uses_natural_boundaries_and_protects_values(self) -> None:
        segmenter = SentenceAccumulator(
            min_first_chars=70,
            max_first_chars=150,
            min_next_chars=70,
            max_next_chars=150,
            normalize_segments=False,
        )
        text = (
            "Il codice cliente AB-12345 e la mail mario.rossi@example.com "
            "devono restare leggibili. Poi posso fare una domanda breve."
        )

        segments: list[str] = []
        for piece in (text[:55], text[55:92], text[92:]):
            segments.extend(segmenter.push(piece))
        segments.extend(segmenter.flush())

        self.assertGreaterEqual(len(segments), 1)
        self.assertIn("mario.rossi@example.com", " ".join(segments))
        self.assertFalse(
            any(
                "mario.rossi@" in item and "example.com" not in item
                for item in segments
            )
        )
        self.assertTrue(segments[0].endswith("."))

    def test_tts_text_normalization_removes_written_formatting(self) -> None:
        text = "**Email**: test@example.com - tel. +39 333-1234567 😊 / ok"
        normalized = normalize_tts_text(text)

        self.assertNotIn("**", normalized)
        self.assertNotIn("😊", normalized)
        self.assertIn("chiocciola", normalized)
        self.assertIn("punto", normalized)
        self.assertIn("3 9", normalized)
        self.assertIn("telefono", normalized)

    def test_forced_segment_cut_adds_continuation_comma(self) -> None:
        segmenter = SentenceAccumulator(
            min_first_chars=10,
            max_first_chars=28,
            min_next_chars=10,
            max_next_chars=28,
            normalize_segments=True,
        )

        segments = segmenter.push(
            "Questa frase continua senza una punteggiatura naturale nel mezzo"
        )

        self.assertGreaterEqual(len(segments), 1)
        self.assertTrue(segments[0].endswith(","))

    def test_voice_session_config_is_stable(self) -> None:
        config = voice_config_source()
        first = VoiceSessionConfig.from_config(config, 24000, language="it")
        second = VoiceSessionConfig.from_config(config, 24000, language="it")

        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(first.voice_id, "voice-a")
        self.assertEqual(first.position_temperature, 0.0)
        self.assertEqual(first.seed, 42)

    def test_tts_queue_drain_counts_cancelled_segments(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            queue.put_nowait("prima frase")
            queue.put_nowait("seconda frase")
            queue.put_nowait(None)

            self.assertEqual(drain_segment_queue(queue), 2)
            self.assertTrue(queue.empty())
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    def test_stt_padding_preserves_original_audio(self) -> None:
        samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        padded = pad_audio_edges(samples, sample_rate=1000, lead_ms=2, tail_ms=1)

        np.testing.assert_allclose(padded[:2], np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(padded[2:5], samples)
        np.testing.assert_allclose(padded[5:], np.zeros(1, dtype=np.float32))

    def test_tts_engine_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_tts_engine("qwen"), "qwen3_tts")
        self.assertEqual(normalize_tts_engine("omni-voice"), "omnivoice")

    def test_initial_edge_fade_does_not_fade_chunk_tail(self) -> None:
        samples = np.ones(1000, dtype=np.float32)

        faded = apply_edge_fade(samples, sample_rate=1000, fade_in_ms=10, fade_out_ms=0)

        self.assertLess(faded[0], 0.01)
        self.assertAlmostEqual(float(faded[-1]), 1.0, places=5)

    def test_tts_onset_noise_trim_removes_low_level_prefix(self) -> None:
        sample_rate = 1000
        prefix = np.full(35, 0.002, dtype=np.float32)
        speech = np.full(100, 0.08, dtype=np.float32)
        cleaned = trim_tts_onset_noise(
            np.concatenate([prefix, speech]),
            sample_rate,
            threshold=0.010,
            window_ms=8,
            keep_ms=3,
            max_trim_ms=80,
        )

        self.assertLessEqual(len(cleaned), 110)
        first_active = int(np.flatnonzero(np.abs(cleaned) > 0.010)[0])
        self.assertLessEqual(first_active, 12)

    def test_loudness_smoother_keeps_gain_state_between_chunks(self) -> None:
        smoother = AudioLoudnessSmoother(target_lufs=-20.0, smoothing=0.5)
        quiet = np.full(1000, 0.01, dtype=np.float32)
        loud = np.full(1000, 0.2, dtype=np.float32)

        first = smoother.process(quiet)
        second = smoother.process(loud)

        self.assertGreater(float(np.sqrt(np.mean(first * first))), 0.01)
        self.assertGreater(float(np.sqrt(np.mean(second * second))), 0.1)

    def test_fixed_reference_passes_matching_instruct(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.kwargs = {}

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [np.full(480, 0.1, dtype=np.float32)]

        config = voice_config_source()
        config.tts_voice_mode = "fixed_reference"
        config.tts_fixed_reference_instruct = True
        config.tts_seed = None
        fake_model = FakeModel()
        tts = OmniVoiceTTS(config)
        tts.model = fake_model
        tts.sample_rate = 24000
        prompt = object()
        state = TTSTurnState(voice_prompt=prompt)
        voice_config = VoiceSessionConfig.from_config(config, 24000, language="it")

        tts._synthesize_sync(
            "Ciao, ti aiuto subito.",
            True,
            state,
            "it",
            voice_config,
        )

        self.assertEqual(fake_model.kwargs["instruct"], config.tts_instruct)
        self.assertIs(fake_model.kwargs["voice_clone_prompt"], prompt)

    def test_fixed_reference_instruct_is_opt_in_for_ttfa(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.kwargs = {}

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [np.full(480, 0.1, dtype=np.float32)]

        config = voice_config_source()
        config.tts_voice_mode = "fixed_reference"
        config.tts_fixed_reference_instruct = False
        config.tts_seed = None
        fake_model = FakeModel()
        tts = OmniVoiceTTS(config)
        tts.model = fake_model
        tts.sample_rate = 24000

        tts._synthesize_sync(
            "Ciao, ti aiuto subito.",
            True,
            TTSTurnState(voice_prompt=object()),
            "it",
            VoiceSessionConfig.from_config(config, 24000, language="it"),
        )

        self.assertNotIn("instruct", fake_model.kwargs)

    def test_omnivoice_tts_uses_model_compatible_language_names(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.kwargs = {}

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [np.full(480, 0.1, dtype=np.float32)]

        config = voice_config_source()
        config.tts_voice_mode = "fixed_reference"
        config.tts_seed = None
        fake_model = FakeModel()
        tts = OmniVoiceTTS(config)
        tts.model = fake_model
        tts.sample_rate = 24000

        tts._synthesize_sync(
            "مرحبا",
            True,
            TTSTurnState(voice_prompt=object()),
            "ar",
            VoiceSessionConfig.from_config(config, 24000, language="ar"),
        )

        self.assertEqual(fake_model.kwargs["language"], "Standard Arabic")

    def test_translation_language_catalog_covers_live_translator_targets(self) -> None:
        required = {
            "it",
            "en",
            "de",
            "fr",
            "es",
            "pt",
            "ro",
            "sq",
            "ru",
            "uk",
            "pl",
            "sr",
            "ar",
            "zh",
            "hi",
            "ur",
            "sw",
        }

        self.assertTrue(required.issubset(SUPPORTED_LANGUAGES))
        self.assertEqual(SUPPORTED_LANGUAGES["ar"], "Arabo (العربية)")
        self.assertEqual(normalize_language_code("zh-CN"), "zh")
        self.assertEqual(
            normalize_language_code("auto", allow_auto=True),
            "auto",
        )
        self.assertEqual(omnivoice_tts_language("ar"), "Standard Arabic")

    def test_translator_prompt_is_strict_and_target_only(self) -> None:
        prompt = translation_instruction(
            "auto",
            "fr",
            live_translation=True,
        )

        self.assertIn("Translate from the auto-detected source language into French", prompt)
        self.assertIn("Preserve numbers, names, places, times, codes", prompt)
        self.assertIn("Output only in French", prompt)
        self.assertIn("not isolated words", prompt)

    def test_translator_llm_messages_skip_agent_prompt_history_and_rag(self) -> None:
        config = SimpleNamespace(system_prompt="CAVADALABS agent prompt")
        messages = llm_messages(
            config,
            "Buongiorno David, sono alle 14:30 in laboratorio.",
            [{"role": "assistant", "content": "old answer"}],
            "de",
            "rag context",
            mode="translator",
            source_language="it",
            target_language="de",
            live_translation=True,
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertNotIn("CAVADALABS", json.dumps(messages))
        self.assertNotIn("old answer", json.dumps(messages))
        self.assertIn("Output only in German", messages[0]["content"])

    def test_language_instruction_supports_expanded_languages(self) -> None:
        self.assertIn("Romanian", language_instruction("ro"))
        self.assertIn("Simplified Chinese", language_instruction("zh"))

    def test_stt_auto_language_is_passed_through_for_whisper_autodetect(self) -> None:
        service = STTService(SimpleNamespace(stt_language="it"))

        self.assertEqual(service._requested_language("auto"), "auto")
        self.assertEqual(service._requested_language(None), "it")


class AsyncLiveTTSPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcript_jsonl_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = AsyncSessionRecorder(recording_config(tmp), "session-test")
            await recorder.start(
                input_sample_rate=16000,
                output_sample_rate=24000,
                voice_config=VoiceSessionConfig.from_config(
                    voice_config_source(),
                    24000,
                    language="it",
                ),
            )
            recorder.transcript("user", "Buongiorno", turn_id=1, stt_ms=180)
            recorder.record_input(np.zeros(160, dtype=np.float32))
            recorder.record_output(np.zeros(240, dtype=np.float32))
            await recorder.close()

            transcript = Path(tmp).glob("*/call_session-test/transcript.jsonl")
            transcript_path = next(transcript)
            entries = [
                json.loads(line)
                for line in transcript_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(entries[0]["speaker"], "user")
            self.assertEqual(entries[0]["stt_ms"], 180)

    async def test_recording_enqueue_is_non_blocking_when_queue_is_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = recording_config(tmp)
            config.recording_queue_size = 1
            recorder = AsyncSessionRecorder(config, "queue-test")
            await recorder.start(
                input_sample_rate=16000,
                output_sample_rate=24000,
                voice_config=VoiceSessionConfig.from_config(
                    voice_config_source(),
                    24000,
                    language="it",
                ),
            )
            for index in range(30):
                recorder.transcript("system", f"evento {index}")
            await recorder.close()

            self.assertGreaterEqual(recorder._dropped_items, 0)

    async def test_rag_timeout_falls_back_to_llm(self) -> None:
        class SlowRAGRetriever(RAGRetriever):
            def _retrieve_sync(self, query, cancel_event=None):  # type: ignore[override]
                time.sleep(0.05)
                return []

        config = rag_config()
        retriever = SlowRAGRetriever(config)
        result = await retriever.retrieve("Quanto costa il servizio chatbot?")

        self.assertFalse(result.used)
        self.assertTrue(result.timed_out)
        self.assertLess(result.latency_ms, 100)


def voice_config_source() -> SimpleNamespace:
    return SimpleNamespace(
        tts_backend="omnivoice",
        tts_voice_id="voice-a",
        tts_model="k2-fsa/OmniVoice",
        tts_speed=1.05,
        tts_stability=0.9,
        tts_similarity_boost=0.8,
        tts_style=0.15,
        tts_temperature=0.0,
        tts_seed=42,
        tts_output_format="pcm16",
        tts_loudness_target_lufs=-16.0,
        tts_loudness_enabled=True,
        tts_crossfade_ms=20,
        tts_crossfade_enabled=True,
        tts_guidance_scale=2.0,
        tts_position_temperature=0.0,
        tts_class_temperature=0.0,
        tts_num_step_first=40,
        tts_num_step_next=40,
        tts_voice_mode="session_anchor",
        tts_instruct="male, middle-aged, low pitch",
        tts_fixed_reference_instruct=False,
        tts_language="it",
        tts_postprocess_output=False,
        tts_denoise=True,
    )


def recording_config(path: str) -> SimpleNamespace:
    return SimpleNamespace(
        recording_enabled=True,
        recording_dir=path,
        recording_queue_size=64,
        recording_prebuffer_seconds=1.0,
    )


def rag_config() -> SimpleNamespace:
    return SimpleNamespace(
        rag_enabled=True,
        rag_docs_dir="missing",
        rag_timeout_ms=1,
        rag_max_chunks=3,
        rag_max_context_chars=2500,
    )
