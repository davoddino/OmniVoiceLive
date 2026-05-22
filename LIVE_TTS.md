# OmniVoice Live TTS Call-Center Project

Questo progetto trasforma la repo OmniVoice in una soluzione realtime da call-center:
una pagina web HTTPS/WSS ascolta l'utente, trascrive, genera una risposta LLM,
sintetizza con OmniVoice e interrompe subito la riproduzione quando l'utente parla sopra.

L'obiettivo non e' una demo minima. La base deve essere vendibile: componenti isolati,
contratti espliciti, cancellazione robusta, metriche di latenza e possibilita' di
sostituire STT, LLM o TTS senza riscrivere il frontend.

## Vincoli Di Prodotto

- Voce: riferimento sintetico OmniVoice, non voce umana clonata.
- Profilo vocale iniziale: candidato `voice_candidates/14.wav`, generato da
  OmniVoice con `instruct="male, middle-aged, low pitch"`.
- Prima risposta: deve restare veloce, ma senza spezzare troppo il parlato: la
  stabilita' del timbro vale piu' della primissima sillaba immediata.
- Conversazione: l'assistente deve fermarsi quando l'utente parla sopra.
- Browser: riproduzione automatica dello stream audio dopo il gesto iniziale
  "Avvia chiamata".
- Backend: un modello OmniVoice caldo su GPU, mai ricaricato per richiesta.
- Vendibilita': configurazione per azienda, log, metriche, stati chiari, errori
  espliciti e interfaccia operativa.

## Architettura

```text
Browser HTTPS
  AudioWorklet microfono
  VAD locale per barge-in rapido
  AudioWorklet player con jitter buffer
  UI call-center
        |
        | WSS /ws
        v
FastAPI Realtime Gateway
  session state
  turn_id
  VAD server
  cancel/flush su barge-in
  metriche di latenza
        |
        +--> STT adapter
        |      HTTP whisper.py oppure faster-whisper in-process
        |
        +--> LLM adapter
        |      OpenAI-compatible streaming endpoint
        |
        +--> OmniVoice TTS worker
               modello caldo
               voce voice-design
               micro-segmenti
               PCM 24 kHz verso browser
```

## Perche' Micro-Segmenti

OmniVoice non e' un TTS causale sample-per-sample: `generate()` produce audio alla
fine della generazione del testo richiesto. Per un'esperienza live efficace usiamo
micro-segmenti testuali:

1. L'LLM produce token in streaming.
2. Il segmenter emette subito una prima frase corta o clausola stabile.
3. OmniVoice sintetizza quel segmento mentre l'LLM continua a generare testo.
4. Il backend invia PCM frame-by-frame al browser.
5. I segmenti successivi vengono accodati e riprodotti senza player manuale.

Questa strategia evita di aspettare la risposta completa e mantiene cancellazione
semplice: su barge-in invalidiamo il `turn_id` e scartiamo tutto cio' che arriva tardi.

## Stati Della Sessione

```text
idle
  -> listening
  -> user_speaking
  -> transcribing
  -> thinking
  -> speaking
  -> interrupted
  -> listening
```

Il browser mostra questi stati, ma il backend rimane autoritativo per turni,
cancellazione e scarto dei chunk vecchi.

## Protocollo WebSocket

### Client -> Server

```json
{"type":"session.start","sample_rate":48000}
```

Inizializza la sessione audio. Dopo questo messaggio il client puo' inviare frame
binary `Float32Array` mono nel sample rate dichiarato.

```json
{"type":"barge_in"}
```

Richiede stop immediato della risposta corrente. Il browser deve anche svuotare
subito il proprio buffer audio.

```json
{"type":"session.stop"}
```

Chiude logicamente la chiamata.

### Server -> Client

```json
{"type":"session.ready","session_id":"...","tts_sample_rate":24000}
```

La sessione e' pronta. I binary frame dal server sono PCM signed 16-bit little-endian
mono a `tts_sample_rate`.

```json
{"type":"vad.speech_start","turn_id":2}
{"type":"vad.speech_end","duration_ms":1420}
{"type":"stt.final","turn_id":2,"text":"...","latency_ms":230}
{"type":"assistant.text_delta","turn_id":2,"text":"..."}
{"type":"assistant.audio_start","turn_id":2,"segment_index":0,"text":"..."}
{"type":"assistant.audio_end","turn_id":2,"segment_index":0}
{"type":"assistant.done","turn_id":2,"latency_ms":950}
{"type":"turn.cancelled","turn_id":2,"reason":"barge_in"}
```

## Barge-In

La cancellazione ha due livelli:

- Browser: appena il VAD locale rileva voce utente mentre l'assistente parla,
  ferma l'AudioWorklet player e svuota il buffer. Questo da' stop percepito
  sotto i 100 ms.
- Server: invalida il `turn_id`, imposta un cancel event, ferma LLM/TTS futuri
  e scarta ogni audio generato per turni vecchi.

Una generazione OmniVoice gia' entrata in `model.generate()` potrebbe terminare
comunque perche' PyTorch non espone una cancellazione hard pulita a meta' kernel.
Per questo i segmenti devono restare brevi: il costo massimo di una generazione
sprecata rimane controllato.

## Prima Risposta Veloce

Configurazione prevista:

- primo segmento: 60-150 caratteri circa;
- segmenti successivi: 90-220 caratteri circa;
- `num_step` primo segmento: 40;
- `num_step` successivi: 40 di default;
- confini naturali di frase con protezione per email, numeri, URL e codici;
- riferimento vocale sintetico fisso per tutti i chunk;
- loudness smoothing e crossfade leggero tra chunk;
- modello caricato una volta all'avvio;
- warmup TTS all'avvio;
- trim edge leggero dei segmenti audio per rimuovere padding/fade artificiali.

## Configurazione

Le variabili principali:

`live_tts` carica automaticamente `.env` o `live_tts.env` dalla root della repo.
`LIVE_TTS_ENV_FILE=/percorso/file.env` permette di indicare un file diverso.

```text
LIVE_TTS_HOST=0.0.0.0
LIVE_TTS_PORT=8020
LIVE_TTS_LOG_LEVEL=info
LIVE_TTS_SSL_CERTFILE=
LIVE_TTS_SSL_KEYFILE=
LIVE_TTS_TTS_BACKEND=omnivoice
LIVE_TTS_MODEL=k2-fsa/OmniVoice
LIVE_TTS_DEVICE_MAP=cuda:0
LIVE_TTS_DTYPE=float16
LIVE_TTS_LANGUAGE=it
LIVE_TTS_INSTRUCT=male, middle-aged, low pitch
LIVE_TTS_VOICE_MODE=fixed_reference
LIVE_TTS_VOICE_ID=cavadalabs_it_male_calm_v1
LIVE_TTS_REFERENCE_AUDIO=voice_candidates/14.wav
LIVE_TTS_REFERENCE_TEXT=Ciao! Certo, ti aiuto volentieri. Con CavadaLabs possiamo creare un chatbot per il tuo sito, collegarlo ai contenuti aziendali e renderlo semplice da aggiornare. Partiamo dalle tue esigenze e scegliamo insieme la soluzione piu adatta.
LIVE_TTS_REFERENCE_PREPROCESS=false
LIVE_TTS_FIXED_REFERENCE_INSTRUCT=false
LIVE_TTS_NUM_STEP_FIRST=40
LIVE_TTS_NUM_STEP_NEXT=40
LIVE_TTS_SPEED=1.05
LIVE_TTS_GUIDANCE_SCALE=2.2
LIVE_TTS_STABILITY=0.90
LIVE_TTS_SIMILARITY_BOOST=0.80
LIVE_TTS_STYLE=0.15
LIVE_TTS_TEMPERATURE=0.0
LIVE_TTS_SEED=14
LIVE_TTS_POSITION_TEMPERATURE=0.15
LIVE_TTS_CLASS_TEMPERATURE=0.0
LIVE_TTS_POSTPROCESS_OUTPUT=false
LIVE_TTS_DENOISE=true
LIVE_TTS_OUTPUT_FORMAT=pcm16
LIVE_TTS_LOUDNESS_TARGET_LUFS=-16.0
LIVE_TTS_LOUDNESS_NORMALIZATION=true
LIVE_TTS_CROSSFADE=true
LIVE_TTS_CROSSFADE_MS=10
LIVE_TTS_SELF_CONDITION=true
LIVE_TTS_ANCHOR_MIN_SECONDS=1.6
LIVE_TTS_ANCHOR_MAX_SECONDS=6.0
LIVE_TTS_SESSION_VOICE_ANCHOR=false
LIVE_TTS_STARTUP_VOICE_ANCHOR=false
LIVE_TTS_STARTUP_ANCHOR_TEXT=Parlo in italiano con voce maschile, calma, chiara e professionale.

LIVE_TTS_QWEN_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
LIVE_TTS_QWEN_MODE=custom_voice
LIVE_TTS_QWEN_SPEAKER=Aiden
LIVE_TTS_QWEN_INSTRUCT=Speak in Italian with a warm, confident, lively call-center tone. Keep pronunciation clear and natural.
LIVE_TTS_QWEN_DEVICE_MAP=cuda:0
LIVE_TTS_QWEN_DTYPE=bfloat16
LIVE_TTS_QWEN_ATTN_IMPLEMENTATION=
LIVE_TTS_QWEN_URL=
LIVE_TTS_QWEN_AUTO_START=true
LIVE_TTS_QWEN_WORKER_HOST=127.0.0.1
LIVE_TTS_QWEN_WORKER_PORT=8031
LIVE_TTS_QWEN_WORKER_DIR=.live_tts_workers/qwen3_tts
LIVE_TTS_QWEN_WORKER_PYTHON=
LIVE_TTS_QWEN_WORKER_INSTALL=true
LIVE_TTS_QWEN_WORKER_START_TIMEOUT_S=900
LIVE_TTS_QWEN_WORKER_REQUEST_TIMEOUT_S=120

LIVE_TTS_STT_BACKEND=auto
LIVE_TTS_STT_URL=
LIVE_TTS_STT_LANGUAGE=it
LIVE_TTS_STT_LEAD_PADDING_MS=280
LIVE_TTS_STT_TAIL_PADDING_MS=120
LIVE_TTS_WHISPER_MODEL=small
LIVE_TTS_WHISPER_DEVICE=cuda
LIVE_TTS_WHISPER_COMPUTE_TYPE=int8_float16

LIVE_TTS_LLM_BACKEND=openai
LIVE_TTS_LLM_URL=http://192.168.0.20:8001/v1/chat/completions
LIVE_TTS_LLM_MODEL=qwen3.6-35b
LIVE_TTS_SYSTEM_PROMPT_FILE=live_tts/prompts/cavadalabs_voice.md
LIVE_TTS_LLM_TEMPERATURE=0.3

LIVE_TTS_MODE=agent
LIVE_TTS_TRANSLATOR_TIMING=immediate
LIVE_TTS_TRANSLATOR_SOURCE_LANGUAGE=auto
LIVE_TTS_TRANSLATOR_TARGET_LANGUAGE=it
LIVE_TTS_TRANSLATOR_IMMEDIATE_BUFFER_MS=1400

LIVE_TTS_SEGMENT_MIN_FIRST_CHARS=60
LIVE_TTS_SEGMENT_MAX_FIRST_CHARS=150
LIVE_TTS_SEGMENT_MIN_NEXT_CHARS=90
LIVE_TTS_SEGMENT_MAX_NEXT_CHARS=220

LIVE_TTS_VAD_THRESHOLD=0.020
LIVE_TTS_VAD_ADAPTIVE=true
LIVE_TTS_VAD_NOISE_CALIBRATION_MS=1000
LIVE_TTS_VAD_START_MULTIPLIER=2.2
LIVE_TTS_VAD_CONTINUE_MULTIPLIER=1.4
LIVE_TTS_VAD_MIN_SPEECH_MS=180
LIVE_TTS_VAD_END_SILENCE_MS=500
LIVE_TTS_VAD_START_MS=180
LIVE_TTS_VAD_END_MS=500
LIVE_TTS_VAD_PREROLL_MS=700

LIVE_TTS_CLIENT_BARGE_THRESHOLD=0.022
LIVE_TTS_CLIENT_BARGE_STOP_MS=80
LIVE_TTS_CLIENT_BARGE_COMMIT_MS=140
LIVE_TTS_CLIENT_BARGE_COOLDOWN_MS=900

LIVE_TTS_RAG_ENABLED=false
LIVE_TTS_RAG_DOCS_DIR=rag_docs
LIVE_TTS_RAG_TIMEOUT_MS=300
LIVE_TTS_RAG_MAX_CHUNKS=3
LIVE_TTS_RAG_MAX_CONTEXT_CHARS=2500
LIVE_TTS_RAG_FALLBACK_TO_LLM=true

LIVE_TTS_RECORDING_ENABLED=true
LIVE_TTS_RECORDING_DIR=recordings
LIVE_TTS_RECORDING_QUEUE_SIZE=256
LIVE_TTS_RECORDING_PREBUFFER_SECONDS=3.0
```

La modalita' della sessione si sceglie dall'interfaccia prima di avviare la
chiamata. `Agente` mantiene il comportamento CAVADALABS normale: il browser invia
`language` in `session.start` e il backend la usa per Whisper, istruzione LLM e
OmniVoice.

`Traduttore` usa invece una lingua sorgente (`source_language`, anche `auto`) e
una lingua target (`target_language`). Il prompt CAVADALABS viene escluso e il LLM
riceve solo istruzioni di traduzione fedele. Il timing `parla subito` chiude i
pezzi audio con un buffer configurabile (`LIVE_TTS_TRANSLATOR_IMMEDIATE_BUFFER_MS`,
default 1400 ms) per evitare traduzioni di parole isolate. Il timing `fine parlato`
usa il VAD normale e aspetta turni piu' completi.

Se usi `whisper.py` come server HTTP, anche `/transcribe` accetta il campo form
`language=auto`, quindi Whisper puo' auto-rilevare la lingua sorgente.

Per non perdere l'attacco delle frasi, il browser conserva un piccolo preroll di
microfono anche mentre il WebSocket si sta aprendo. Il server aggiunge poi 280 ms
di silenzio prima dell'audio passato a Whisper e mantiene 700 ms di preroll VAD.
Questi valori non rendono il barge-in piu' sensibile: servono solo a non tagliare
le prime sillabe.

## Avvio

Con `.env` configurato:

```bash
uv run python -m live_tts
```

Poi aprire:

```text
http://127.0.0.1:8020
```

Per microfono e autoplay in produzione va servito dietro HTTPS. In locale i browser
accettano normalmente `localhost`/`127.0.0.1` come secure context per il microfono.
Per usare un iPhone sulla rete locale, segui [LOCAL_HTTPS.md](LOCAL_HTTPS.md).

## Selettore Engine TTS

L'interfaccia espone un selettore per confrontare engine diversi mantenendo uguali
STT, LLM, prompt, segmenter, VAD, barge-in e player. Il server mantiene un solo
engine attivo alla volta:

- `OmniVoice`: default, caricato all'avvio.
- `Qwen3-TTS`: opzionale, avviato in un worker isolato solo quando selezionato.

Quando cambi engine, il turno corrente viene cancellato, il modello precedente
viene scaricato o il worker viene terminato, la GPU viene liberata e l'interfaccia
mostra `loading`, `ready` o `failed`. L'audio del microfono viene messo in pausa
durante il caricamento e riparte quando l'engine e' pronto.

Qwen non viene installato nella venv principale. Il server crea automaticamente
una venv isolata sotto `.live_tts_workers/qwen3_tts` e installa le dipendenze da
`more_requirement.txt` al primo uso. Questo evita il conflitto tra `qwen-tts` e
le versioni `transformers` richieste da OmniVoice.

Per preinstallare manualmente il worker Qwen:

```bash
uv venv .live_tts_workers/qwen3_tts/.venv
uv pip install --python .live_tts_workers/qwen3_tts/.venv/bin/python -r more_requirement.txt
```

Qwen3-TTS usa di default il modello `0.6B-CustomVoice` per ridurre il rischio di
OOM. Per provare voice design invece dei preset:

```text
LIVE_TTS_QWEN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
LIVE_TTS_QWEN_MODE=voice_design
LIVE_TTS_QWEN_INSTRUCT=Speak in Italian with a warm, confident, lively call-center tone.
```

## Stabilita' Del Timbro

La modalita' consigliata e' `fixed_reference`: il file `voice_candidates/14.wav`
viene letto una volta all'avvio e trasformato in un prompt vocale OmniVoice
riusabile. Non viene riprodotto all'utente e non e' una risposta cachata: ogni
chunk resta generato live dal testo corrente, ma usa sempre lo stesso riferimento
di voce.

```text
LIVE_TTS_VOICE_MODE=fixed_reference
LIVE_TTS_REFERENCE_AUDIO=voice_candidates/14.wav
LIVE_TTS_REFERENCE_TEXT=Ciao! Certo, ti aiuto volentieri. Con CavadaLabs possiamo creare un chatbot per il tuo sito, collegarlo ai contenuti aziendali e renderlo semplice da aggiornare. Partiamo dalle tue esigenze e scegliamo insieme la soluzione piu adatta.
LIVE_TTS_REFERENCE_PREPROCESS=false
LIVE_TTS_FIXED_REFERENCE_INSTRUCT=false
LIVE_TTS_INSTRUCT=male, middle-aged, low pitch
LIVE_TTS_STARTUP_VOICE_ANCHOR=false
LIVE_TTS_SESSION_VOICE_ANCHOR=false
LIVE_TTS_NUM_STEP_FIRST=40
LIVE_TTS_NUM_STEP_NEXT=40
LIVE_TTS_SPEED=1.05
LIVE_TTS_GUIDANCE_SCALE=2.2
LIVE_TTS_SEED=14
LIVE_TTS_POSITION_TEMPERATURE=0.15
LIVE_TTS_CLASS_TEMPERATURE=0.0
LIVE_TTS_LOUDNESS_NORMALIZATION=true
LIVE_TTS_CROSSFADE_MS=10
```

In `fixed_reference` il riferimento audio resta il condizionamento principale.
`LIVE_TTS_FIXED_REFERENCE_INSTRUCT=false` evita di aggiungere token di instruct a
ogni chunk, proteggendo TTFA. Puoi metterlo a `true` solo per un test A/B se il
reference resta coerente con `LIVE_TTS_INSTRUCT`.

`LIVE_TTS_REFERENCE_PREPROCESS=false` mantiene il file 14 esattamente come e'
stato generato. `LIVE_TTS_POSITION_TEMPERATURE=0.15` riduce la varianza tra
chunk senza bloccare completamente l'espressivita'; se serve massima coerenza
puoi scendere a `0.0`.

Se `voice_candidates/14.wav` non esiste ancora sulla macchina di avvio:

```bash
uv run python scripts/generate_voice_candidates.py
```

Per tornare alla modalita' senza riferimento fisso:

```text
LIVE_TTS_VOICE_MODE=voice_design
LIVE_TTS_STARTUP_VOICE_ANCHOR=false
```

## Barge-In Locale

Il browser ferma subito il player quando il microfono supera la soglia locale
mentre l'assistente sta parlando. Dopo pochi millisecondi di voce confermata invia
anche `barge_in` al server e scarta eventuali frame audio vecchi arrivati in
ritardo.

```text
LIVE_TTS_CLIENT_BARGE_THRESHOLD=0.022
LIVE_TTS_CLIENT_BARGE_STOP_MS=80
LIVE_TTS_CLIENT_BARGE_COMMIT_MS=140
LIVE_TTS_CLIENT_BARGE_COOLDOWN_MS=900
```

Se si interrompe troppo facilmente per eco dagli speaker, alza
`LIVE_TTS_CLIENT_BARGE_THRESHOLD` a `0.026` o `0.030`. Se invece non interrompe
abbastanza rapidamente, scendi verso `0.018`.

## Hardening Per Produzione

Prima di venderlo a molte aziende, i prossimi step sono:

- autenticazione tenant e rate limit WebSocket;
- TLS terminato da reverse proxy;
- metriche Prometheus o OpenTelemetry;
- tracing per turni: STT latency, first LLM token, first TTS audio, barge-in delay;
- storage opzionale transcript con policy privacy;
- prompt/policy per azienda;
- monitor GPU e coda TTS;
- healthcheck profondo con warmup periodico;
- deployment separato per gateway, STT, LLM e TTS worker se il carico cresce.

La struttura implementata in `live_tts/` mantiene questi confini: gateway,
sessione, STT, LLM, segmenter, TTS e audio sono moduli separati.
