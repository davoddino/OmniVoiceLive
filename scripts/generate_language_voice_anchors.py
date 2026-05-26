from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


ANCHOR_TEXTS = {
    "it": "Ciao, sono pronto ad aiutarti con una voce chiara, calma e professionale.",
    "en": "Hello, I am ready to help you with a clear, calm, professional voice.",
    "de": "Hallo, ich bin bereit, Ihnen mit einer klaren, ruhigen und professionellen Stimme zu helfen.",
    "fr": "Bonjour, je suis pret a vous aider avec une voix claire, calme et professionnelle.",
    "es": "Hola, estoy listo para ayudarte con una voz clara, tranquila y profesional.",
    "pt": "Ola, estou pronto para ajudar com uma voz clara, calma e profissional.",
    "ro": "Buna ziua, sunt gata sa va ajut cu o voce clara, calma si profesionala.",
    "sq": "Pershendetje, jam gati t'ju ndihmoj me nje ze te qarte, te qete dhe profesional.",
    "ru": "Здравствуйте, я готов помочь вам спокойным, четким и профессиональным голосом.",
    "uk": "Вітаю, я готовий допомогти вам спокійним, чітким і професійним голосом.",
    "pl": "Dzien dobry, jestem gotowy pomoc spokojnym, wyraznym i profesjonalnym glosem.",
    "sr": "Dobar dan, spreman sam da pomognem jasnim, mirnim i profesionalnim glasom.",
    "hr": "Dobar dan, spreman sam pomoci jasnim, mirnim i profesionalnim glasom.",
    "bs": "Dobar dan, spreman sam pomoci jasnim, smirenim i profesionalnim glasom.",
    "ar": "مرحبا، أنا مستعد لمساعدتك بصوت واضح وهادئ ومهني.",
    "zh": "你好，我已经准备好用清晰、平静、专业的声音帮助你。",
    "hi": "नमस्ते, मैं स्पष्ट, शांत और पेशेवर आवाज़ में आपकी मदद करने के लिए तैयार हूँ।",
    "ur": "سلام، میں صاف، پرسکون اور پیشہ ورانہ آواز میں آپ کی مدد کے لیے تیار ہوں۔",
    "sw": "Habari, niko tayari kukusaidia kwa sauti wazi, tulivu na ya kitaalamu.",
}


def parse_args() -> argparse.Namespace:
    from live_tts.config import load_env_files
    from live_tts.languages import SUPPORTED_LANGUAGES

    load_env_files()
    parser = argparse.ArgumentParser(
        description=(
            "Generate one fixed-reference OmniVoice anchor WAV per language and "
            "write voice_candidates/language_anchors/manifest.json."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "LIVE_TTS_LANGUAGE_REFERENCE_DIR",
            "voice_candidates/language_anchors",
        ),
        help="Directory where WAV files and manifest.json are written.",
    )
    parser.add_argument(
        "--languages",
        default=",".join(SUPPORTED_LANGUAGES),
        help="Comma-separated language codes to generate.",
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
        "--instruct",
        default=os.getenv("LIVE_TTS_INSTRUCT", "male, middle-aged, low pitch"),
        help="OmniVoice voice-design instruct. Must use valid comma-separated items.",
    )
    parser.add_argument("--speed", type=float, default=float(os.getenv("LIVE_TTS_SPEED", "1.05")))
    parser.add_argument("--num-step", type=int, default=int(os.getenv("LIVE_TTS_NUM_STEP_FIRST", "40")))
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=float(os.getenv("LIVE_TTS_GUIDANCE_SCALE", "2.0")),
    )
    parser.add_argument("--position-temperature", type=float, default=0.0)
    parser.add_argument("--class-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=int(os.getenv("LIVE_TTS_SEED", "14")))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate WAV files even if they already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep generating remaining languages if one language fails.",
    )
    return parser.parse_args()


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    from live_tts.languages import normalize_language_code, omnivoice_tts_language

    args = parse_args()
    languages = [
        normalize_language_code(code.strip(), default="")
        for code in args.languages.split(",")
        if code.strip()
    ]
    languages = [code for code in dict.fromkeys(languages) if code in ANCHOR_TEXTS]
    if not languages:
        raise SystemExit("No supported languages selected.")

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

    manifest: dict[str, object] = {
        "version": 1,
        "sample_rate": sample_rate,
        "model": args.model,
        "instruct": args.instruct,
        "num_step": args.num_step,
        "speed": args.speed,
        "guidance_scale": args.guidance_scale,
        "position_temperature": args.position_temperature,
        "class_temperature": args.class_temperature,
        "references": {},
    }
    references: dict[str, dict[str, object]] = {}

    for index, language in enumerate(languages, start=1):
        text = ANCHOR_TEXTS[language]
        filename = f"{language}.wav"
        path = output_dir / filename
        if path.is_file() and not args.overwrite:
            print(f"[{index:02d}/{len(languages):02d}] {language}: exists, keeping {path}")
            references[language] = {
                "language": language,
                "audio": filename,
                "text": text,
            }
            continue

        print(f"[{index:02d}/{len(languages):02d}] {language}: generating {path}")
        random.seed(args.seed + index)
        np.random.seed(args.seed + index)
        torch.manual_seed(args.seed + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + index)

        try:
            audio = model.generate(
                text=text,
                language=omnivoice_tts_language(language),
                instruct=args.instruct,
                num_step=args.num_step,
                speed=args.speed,
                guidance_scale=args.guidance_scale,
                position_temperature=args.position_temperature,
                class_temperature=args.class_temperature,
                denoise=True,
                postprocess_output=False,
            )
        except Exception:
            if not args.continue_on_error:
                raise
            print(f"[{index:02d}/{len(languages):02d}] {language}: failed")
            continue

        waveform = np.asarray(audio[0], dtype=np.float32).reshape(-1)
        sf.write(path, waveform, sample_rate)
        references[language] = {
            "language": language,
            "audio": filename,
            "text": text,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["references"] = references
    manifest_path = output_dir / "manifest.json"
    tmp_path = output_dir / ".manifest.json.tmp"
    tmp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(manifest_path)
    print(f"Done. Generated/registered {len(references)} references.")
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
