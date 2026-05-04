# OmniVoice Live TTS Call-Center Project

Questo progetto trasforma la repo OmniVoice in una soluzione realtime da call-center:
una pagina web HTTPS/WSS ascolta l'utente, trascrive, genera una risposta LLM,
sintetizza con OmniVoice e interrompe subito la riproduzione quando l'utente parla sopra.

L'obiettivo non e' una demo minima. La base deve essere vendibile: componenti isolati,
contratti espliciti, cancellazione robusta, metriche di latenza e possibilita' di
sostituire STT, LLM o TTS senza riscrivere il frontend.

## Vincoli Di Prodotto

- Voce: voice design OmniVoice, non voice cloning.
- Profilo vocale iniziale: lo stesso stile di `test.py`, cioe' italiano con
  `instruct="female, low pitch"`.
- Prima risposta: deve partire il prima possibile, privilegiando un primo segmento
  breve e sintetizzato con meno diffusion steps.
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

- primo segmento: 28-70 caratteri circa;
- segmenti successivi: 90-220 caratteri circa;
- `num_step` primo segmento: 16;
- `num_step` successivi: 24 di default;
- temperatura deterministica per ridurre variazioni tra segmenti;
- modello caricato una volta all'avvio;
- warmup TTS all'avvio;
- trim edge leggero dei segmenti audio per rimuovere padding/fade artificiali.

## Configurazione

Le variabili principali:

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
LIVE_TTS_INSTRUCT=female, low pitch
LIVE_TTS_NUM_STEP_FIRST=16
LIVE_TTS_NUM_STEP_NEXT=24
LIVE_TTS_SPEED=1.05
LIVE_TTS_SELF_CONDITION=true
LIVE_TTS_ANCHOR_MIN_SECONDS=0.45

LIVE_TTS_STT_BACKEND=auto
LIVE_TTS_STT_URL=
LIVE_TTS_WHISPER_MODEL=small
LIVE_TTS_WHISPER_DEVICE=cuda
LIVE_TTS_WHISPER_COMPUTE_TYPE=int8_float16

LIVE_TTS_LLM_BACKEND=openai
LIVE_TTS_LLM_URL=http://192.168.0.20:8001/v1/chat/completions
LIVE_TTS_LLM_MODEL=qwen3.6-35b
```

Per testare solo trasporto audio e UI senza GPU:

```text
LIVE_TTS_TTS_BACKEND=mock
LIVE_TTS_STT_BACKEND=mock
LIVE_TTS_LLM_BACKEND=mock
```

## Avvio

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

## Stabilita' Del Timbro

In modalita' voice-design pura, generare ogni micro-segmento in modo indipendente
puo' cambiare il timbro tra chunk. Per mantenere la prima risposta veloce senza
usare una voce esterna clonata, `live_tts` usa self-conditioning per turno:

1. il primo segmento viene generato con `instruct="female, low pitch"`;
2. se e' abbastanza lungo, il suo audio diventa un'ancora vocale temporanea;
3. i segmenti successivi dello stesso turno usano quell'ancora;
4. l'ancora viene scartata al turno successivo.

Variabili:

```text
LIVE_TTS_SELF_CONDITION=true
LIVE_TTS_ANCHOR_MIN_SECONDS=0.45
```

Questo non richiede una voce clonata dell'utente o di una persona reale: stabilizza
la voce auto-generata dal primo chunk della risposta.

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
