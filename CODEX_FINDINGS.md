# CODEX findings - Live TTS call-center upgrade

Data analisi: 2026-05-05

## Architettura attuale

- Entrypoint: `live_tts/app.py` crea la app FastAPI, carica `LiveTTSConfig`, istanzia `STTService`, `LLMStreamer` e `BaseTTS`, poi apre `/ws`.
- Acquisizione audio: browser in `live_tts/static/app.js`, con `recorder-worklet.js` o fallback `ScriptProcessor`, invia frame float32 via WebSocket.
- Playback: browser in `live_tts/static/app.js` e `player-worklet.js`, riceve PCM16 dal server, ricampiona al sample rate del browser e accoda l'audio. Il client cancella la coda su barge-in.
- VAD server: `live_tts/audio.py`, classe `AudioTurnDetector`, usa RMS fisso (`LIVE_TTS_VAD_THRESHOLD`), start/end ms, preroll e durata massima.
- Barge-in client: `live_tts/static/app.js`, usa RMS locale con threshold configurato dal server.
- STT: `live_tts/stt.py`, HTTP esterno se configurato, altrimenti `faster-whisper` locale.
- LLM: `live_tts/llm.py`, streaming OpenAI-compatible. Il prompt di sistema arriva da `live_tts/prompts/cavadalabs_voice.md`.
- Chunking risposta: `live_tts/segmenter.py`, `LiveTextSegmenter`, alimentato dai delta LLM in `RealtimeSession._respond`.
- TTS: `live_tts/tts.py`, `OmniVoiceTTS.synthesize`; con `voice_design`, `turn_anchor` o `session_anchor`. I parametri sono in `LiveTTSConfig`.
- Salvataggio: al momento non c'e' un recorder/session writer persistente per audio, transcript, metriche o CRM event log.

## Criticita' audio

- La voce puo' cambiare tra chunk perche' ogni segmento e' una generazione OmniVoice separata. In modalita' `voice_design` non viene riusato un prompt vocale generato; in `session_anchor` il riuso esiste, ma non e' il default in `live_tts.env`.
- Non esiste un oggetto `VoiceSessionConfig` persistente con tutti i parametri effettivi della voce per sessione; i parametri sono globali e non vengono loggati/salvati per chiamata.
- Il chunking corrente usa soglie molto alte (`280/900` primo chunk, `900/1800` successivi) e puo' tagliare su spazi se supera il massimo. Non protegge numeri, email, URL, importi o codici.
- Il TTS riceve testo streaming pulito solo parzialmente: rimuove `<think>`, ma non normalizza markdown, emoji, bullet, simboli, slash o forme poco pronunciabili.
- Non c'e' normalizzazione loudness tra waveform TTS consecutive; picchi o RMS diversi possono produrre salti di volume percepiti.
- Gli edge fade esistono, ma non c'e' crossfade tra chunk. Il playback invia frame in ordine, ma non mantiene statistiche di chunk generati/riprodotti/cancellati lato server.
- Il VAD server usa soglia fissa, quindi in ambienti rumorosi puo' partire tardi, tagliare parlato o confondere rumore per voce. Manca calibrazione iniziale e hysteresis start/continue.
- Recording, transcript JSONL, metadata e analytics non sono presenti; vanno aggiunti in modo asincrono fuori dal path live.

## Piano minimo di intervento

1. Aggiungere `VoiceSessionConfig` e usarla una sola volta per sessione, con parametri TTS stabili, sample rate, output format e target loudness.
2. Rendere default la continuita' vocale su `session_anchor` nella configurazione locale, senza togliere la possibilita' di tornare a `voice_design`.
3. Sostituire il segmenter con un sentence accumulator piu' conservativo: confini naturali, min/max caratteri ragionevoli, protezione per email, URL, numeri, date, importi e codici.
4. Aggiungere normalizzazione testo TTS prima della sintesi.
5. Normalizzare RMS/loudness dei chunk TTS a target stabile e aggiungere crossfade opzionale 10-30 ms prima dell'invio.
6. Rendere il VAD adattivo con calibrazione rumore iniziale, hysteresis start/continue e nuove variabili env semplici.
7. Aggiungere writer asincrono locale per input/output wav, transcript JSONL, metadata, metriche e event log CRM append-only.
8. Aggiungere RAG opzionale con timeout rigido e fallback LLM-only.
9. Aggiornare prompt call-center e test leggeri per segmenter, normalizzazione, config sessione TTS, writer, timeout RAG e cancellazione coda.

## File che intendo modificare

- `live_tts/config.py`
- `live_tts/audio.py`
- `live_tts/segmenter.py`
- `live_tts/tts.py`
- `live_tts/llm.py`
- `live_tts/session.py`
- `live_tts/app.py`
- `live_tts/prompts/cavadalabs_voice.md`
- `live_tts/static/app.js`
- `live_tts/static/player-worklet.js`
- `live_tts.env`
- nuovi moduli leggeri in `live_tts/` per registrazione, RAG, metriche/eventi e configurazione vocale
- nuovi test o script di verifica sotto `tests/`
