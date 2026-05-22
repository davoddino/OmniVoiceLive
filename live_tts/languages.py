from __future__ import annotations


LANGUAGE_OPTIONS: dict[str, tuple[str, str]] = {
    "it": ("Italiano", "Italian"),
    "en": ("English", "English"),
    "de": ("Deutsch", "German"),
    "fr": ("Français", "French"),
    "es": ("Español", "Spanish"),
    "pt": ("Português", "Portuguese"),
    "ro": ("Română", "Romanian"),
    "sq": ("Shqip", "Albanian"),
    "ru": ("Русский", "Russian"),
    "uk": ("Українська", "Ukrainian"),
    "pl": ("Polski", "Polish"),
    "sr": ("Serbo/Croato/Bosniaco", "Serbian/Croatian/Bosnian"),
    "hr": ("Hrvatski", "Croatian"),
    "bs": ("Bosanski", "Bosnian"),
    "ar": ("العربية", "Arabic"),
    "zh": ("简体中文", "Simplified Chinese"),
    "hi": ("हिन्दी", "Hindi"),
    "ur": ("اردو", "Urdu"),
    "sw": ("Kiswahili", "Swahili"),
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
