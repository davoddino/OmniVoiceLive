import json
import io
import wave
import re
import requests
from pathlib import Path


LLM_URL = "http://192.168.0.20:8001/v1/chat/completions"
TTS_URL = "http://127.0.0.1:8010/synthesize"
MODEL = "qwen3.6-35b"

OUT = Path("cavadatts_response.wav")


CAVADA_CONTEXT = """
Cerca di scrivere meno di 100 parole come risposta, se non necessarie di più!
Non serve che rispondi sempre con il contesto sotto. Solo quando chiesto

CavadaLabs è un provider italiano di soluzioni di Intelligenza Artificiale per PMI.
Porta modelli AI open-source e ottimizzati dentro le aziende, collegandoli ai sistemi esistenti.

CavadaLabs offre:
OCR documentale, trascrizioni automatiche, knowledge management con LLM,
server GPU personalizzati, integrazione con ERP, CRM e database,
soluzioni on-premise o in cloud privato.

Il target principale sono PMI italiane, manifattura, logistica e studi professionali.

Regole di risposta:
Rispondi sempre in italiano naturale e parlato.
Scrivi testo pensato per essere letto ad alta voce.
Non usare markdown.
Non usare asterischi.
Non usare elenchi puntati.
Non usare titoli.
Non usare grassetto.
Non usare parentesi se puoi evitarle.
Non usare simboli decorativi.
Non scrivere codice.
Usa frasi brevi, fluide e complete.
Spiega come se stessi parlando con una persona.
"""


def remove_think_only(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text


def fix_encoding(text: str) -> str:
    """
    Corregge casi tipo:
    Ã¨ -> è
    lâ -> l'
    giÃ -> già
    """
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def clean_stream_text(text: str) -> str:
    return remove_think_only(fix_encoding(text))


def llm_stream(prompt: str):
    payload = {
        "model": MODEL,
        "stream": True,
        "temperature": 0,
        "top_p": 1,
        "messages": [
            {"role": "system", "content": CAVADA_CONTEXT},
            {"role": "user", "content": prompt},
        ],
    }

    with requests.post(LLM_URL, json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.encoding = "utf-8"

        for raw_line in r.iter_lines(decode_unicode=False):
            if not raw_line:
                continue

            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                line = raw_line.decode("latin-1")

            if line.startswith("data: "):
                line = line[6:]

            if line == "[DONE]":
                break

            try:
                data = json.loads(line)
                delta = data["choices"][0].get("delta", {})
                text = delta.get("content")

                if text:
                    yield clean_stream_text(text)

            except Exception:
                continue


def tts_segments(text_iter):
    buffer = ""

    strong_delimiters = ".?!…"
    soft_delimiters = ",;:"

    min_first_chars = 100
    min_next_chars = 140
    max_chars = 280

    first = True

    for piece in text_iter:
        piece = clean_stream_text(piece)

        if not piece:
            continue

        print(piece, end="", flush=True)
        buffer += piece

        min_chars = min_first_chars if first else min_next_chars

        cut = max(buffer.rfind(d) for d in strong_delimiters)

        if cut >= min_chars:
            segment = clean_stream_text(buffer[: cut + 1]).strip()
            buffer = buffer[cut + 1 :]

            first = False

            if segment:
                yield segment

            continue

        if len(buffer) >= max_chars:
            soft_cut = max(buffer.rfind(d, 0, max_chars) for d in soft_delimiters)

            if soft_cut >= min_chars:
                cut = soft_cut
            else:
                cut = buffer.rfind(" ", 0, max_chars)

            if cut > 0:
                segment = clean_stream_text(buffer[: cut + 1]).strip()
                buffer = buffer[cut + 1 :]

                first = False

                if segment:
                    yield segment

    tail = clean_stream_text(buffer).strip()
    if tail:
        yield tail


def synthesize_segment(text: str) -> bytes:
    print(f"\n\n[TTS] {text}\n", flush=True)

    r = requests.post(
        TTS_URL,
        json={"text": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.content


def wav_to_pcm(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()

    return pcm, sr, channels, width


def write_wav(path: Path, pcm_chunks, sr=24000, channels=1, width=2):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(sr)

        for pcm in pcm_chunks:
            wf.writeframes(pcm)


def main():
    prompt = input("Tu: ").strip()

    print("\nLLM:\n", flush=True)

    pcm_chunks = []
    sr = 24000
    channels = 1
    width = 2

    for segment in tts_segments(llm_stream(prompt)):
        wav = synthesize_segment(segment)
        pcm, sr, channels, width = wav_to_pcm(wav)
        pcm_chunks.append(pcm)

    write_wav(OUT, pcm_chunks, sr, channels, width)

    print(f"\n\nAudio salvato in: {OUT}")


if __name__ == "__main__":
    main()