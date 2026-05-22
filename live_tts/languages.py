from __future__ import annotations


LANGUAGE_OPTIONS: dict[str, tuple[str, str]] = {
    "it": ("Italiano", "Italian"),
    "en": ("English", "English"),
    "de": ("Tedesco (Deutsch)", "German"),
    "fr": ("Francese (Français)", "French"),
    "es": ("Spagnolo (Español)", "Spanish"),
    "pt": ("Portoghese (Português)", "Portuguese"),
    "ro": ("Rumeno (Română)", "Romanian"),
    "sq": ("Albanese (Shqip)", "Albanian"),
    "ru": ("Russo (Русский)", "Russian"),
    "uk": ("Ucraino (Українська)", "Ukrainian"),
    "pl": ("Polski", "Polish"),
    "sr": ("Serbo/Croato/Bosniaco", "Serbian/Croatian/Bosnian"),
    "hr": ("Croato (Hrvatski)", "Croatian"),
    "bs": ("Bosniaco (Bosanski)", "Bosnian"),
    "ar": ("Arabo (العربية)", "Arabic"),
    "zh": ("Cinese semplificato (简体中文)", "Simplified Chinese"),
    "hi": ("Hindi (हिन्दी)", "Hindi"),
    "ur": ("Urdu (اردو)", "Urdu"),
    "sw": ("Swahili (Kiswahili)", "Swahili"),
}

SUPPORTED_LANGUAGES = {
    code: label for code, (label, _english_name) in LANGUAGE_OPTIONS.items()
}
SOURCE_LANGUAGE_OPTIONS = {"auto": "Auto rilevamento", **SUPPORTED_LANGUAGES}

LANGUAGE_ALIASES = {
    "cn": "zh",
    "zh-cn": "zh",
    "zh_hans": "zh",
    "zh-hans": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "al": "sq",
    "alb": "sq",
    "ua": "uk",
    "sh": "sr",
    "hbs": "sr",
    "sr-hr": "sr",
    "serbo-croatian": "sr",
    "serbo/croatian": "sr",
}

OMNIVOICE_TTS_LANGUAGE_OVERRIDES = {
    # OmniVoice does not accept bare "ar", but its language map contains
    # "standard arabic" -> "arb".
    "ar": "Standard Arabic",
}


def normalize_language_code(
    value: object,
    default: str = "it",
    *,
    allow_auto: bool = False,
) -> str:
    code = str(value or "").strip().lower().replace("_", "-")
    if allow_auto and code in {"", "auto", "detect", "auto-detect", "autodetect"}:
        return "auto"
    code = LANGUAGE_ALIASES.get(code, code)
    fallback = LANGUAGE_ALIASES.get(default, default)
    if fallback not in SUPPORTED_LANGUAGES:
        fallback = "it"
    return code if code in SUPPORTED_LANGUAGES else fallback


def language_label(code: str | None) -> str:
    normalized = normalize_language_code(code, default="it")
    return SUPPORTED_LANGUAGES.get(normalized, SUPPORTED_LANGUAGES["it"])


def language_english_name(code: str | None) -> str:
    normalized = normalize_language_code(code, default="it")
    return LANGUAGE_OPTIONS.get(normalized, LANGUAGE_OPTIONS["it"])[1]


def source_language_name(code: str | None) -> str:
    normalized = normalize_language_code(code, default="auto", allow_auto=True)
    if normalized == "auto":
        return "the auto-detected source language"
    return language_english_name(normalized)


def omnivoice_tts_language(code: str | None) -> str:
    normalized = normalize_language_code(code, default="it")
    return OMNIVOICE_TTS_LANGUAGE_OVERRIDES.get(normalized, normalized)
