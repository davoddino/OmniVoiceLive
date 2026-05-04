from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TEXT = (
    "Ciao! Certo, ti aiuto volentieri. "
    "Con CavadaLabs possiamo creare un chatbot per il tuo sito, "
    "collegarlo ai contenuti aziendali e renderlo semplice da aggiornare. "
    "Partiamo dalle tue esigenze e scegliamo insieme la soluzione piu adatta."
)


@dataclass(frozen=True)
class VoiceCandidate:
    number: int
    instruct: str
    speed: float
    num_step: int
    guidance_scale: float
    position_temperature: float


CANDIDATES = [
    VoiceCandidate(1, "female, moderate pitch", 1.06, 40, 1.8, 2.0),
    VoiceCandidate(2, "female, high pitch", 1.05, 40, 1.8, 2.0),
    VoiceCandidate(3, "female, young adult, moderate pitch", 1.08, 40, 1.8, 2.0),
    VoiceCandidate(4, "female, young adult, high pitch", 1.06, 40, 1.8, 3.0),
    VoiceCandidate(5, "female, middle-aged, moderate pitch", 1.04, 40, 2.0, 2.0),
    VoiceCandidate(6, "female, middle-aged, high pitch", 1.04, 40, 1.8, 2.5),
    VoiceCandidate(7, "female, low pitch", 1.06, 40, 1.8, 2.0),
    VoiceCandidate(8, "female, young adult, low pitch", 1.08, 40, 1.8, 2.0),
    VoiceCandidate(9, "male, moderate pitch", 1.07, 40, 1.8, 2.0),
    VoiceCandidate(10, "male, high pitch", 1.06, 40, 1.8, 2.0),
    VoiceCandidate(11, "male, young adult, moderate pitch", 1.08, 40, 1.8, 2.0),
    VoiceCandidate(12, "male, young adult, high pitch", 1.06, 40, 1.8, 3.0),
    VoiceCandidate(13, "male, middle-aged, moderate pitch", 1.05, 40, 2.0, 2.0),
    VoiceCandidate(14, "male, middle-aged, low pitch", 1.05, 40, 2.0, 2.0),
    VoiceCandidate(15, "male, middle-aged, high pitch", 1.04, 40, 1.8, 2.5),
    VoiceCandidate(16, "male, young adult, low pitch", 1.08, 40, 1.8, 2.0),
    VoiceCandidate(17, "female", 1.08, 36, 1.6, 4.0),
    VoiceCandidate(18, "male", 1.08, 36, 1.6, 4.0),
    VoiceCandidate(19, "female, teenager, moderate pitch", 1.08, 36, 1.6, 4.0),
    VoiceCandidate(20, "male, teenager, moderate pitch", 1.08, 36, 1.6, 4.0),
]


def parse_args() -> argparse.Namespace:
    from live_tts.config import load_env_files

    load_env_files()
    parser = argparse.ArgumentParser(
        description="Generate 20 OmniVoice voice-design candidates as numbered WAV files."
    )
    parser.add_argument(
        "--output-dir",
        default="voice_candidates",
        help="Directory where WAV files and manifest.csv are written.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LIVE_TTS_MODEL", "k2-fsa/OmniVoice"),
        help="OmniVoice model id or local path.",
    )
    parser.add_argument(
        "--device-map",
        default=os.getenv("LIVE_TTS_DEVICE_MAP", "cuda:0"),
        help="Device map passed to OmniVoice.from_pretrained.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default=os.getenv("LIVE_TTS_DTYPE", "float16"),
        help="Torch dtype for the model.",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("LIVE_TTS_LANGUAGE", "it"),
        help="Language code passed to OmniVoice.generate.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text spoken by every candidate.",
    )
    return parser.parse_args()


def main() -> None:
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(torch, args.dtype)
    print(f"Loading {args.model} on {args.device_map} with dtype={args.dtype}")
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=args.device_map,
        dtype=dtype,
    )
    sample_rate = int(model.sampling_rate or 24000)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "number",
                "filename",
                "instruct",
                "speed",
                "num_step",
                "guidance_scale",
                "position_temperature",
                "text",
            ],
        )
        writer.writeheader()

        for candidate in CANDIDATES:
            filename = f"{candidate.number:02d}.wav"
            path = output_dir / filename
            print(
                f"[{candidate.number:02d}/20] {filename} "
                f"instruct={candidate.instruct!r} speed={candidate.speed}"
            )
            audio = model.generate(
                text=args.text,
                language=args.language,
                instruct=candidate.instruct,
                num_step=candidate.num_step,
                speed=candidate.speed,
                guidance_scale=candidate.guidance_scale,
                position_temperature=candidate.position_temperature,
                class_temperature=0.0,
                denoise=True,
                postprocess_output=False,
            )
            sf.write(path, audio[0], sample_rate)
            writer.writerow(
                {
                    "number": candidate.number,
                    "filename": filename,
                    "instruct": candidate.instruct,
                    "speed": candidate.speed,
                    "num_step": candidate.num_step,
                    "guidance_scale": candidate.guidance_scale,
                    "position_temperature": candidate.position_temperature,
                    "text": args.text,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Done. Listen to WAV files in {output_dir}")
    print(f"Manifest: {manifest_path}")


def resolve_dtype(torch_module, value: str):
    if value == "float16":
        return torch_module.float16
    if value == "bfloat16":
        return torch_module.bfloat16
    if value == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {value}")


if __name__ == "__main__":
    main()
