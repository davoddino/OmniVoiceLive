# HTTPS Locale Per iPhone

Questa guida crea un certificato locale valido per aprire OmniVoice Live da iPhone:

```text
https://IP_DEL_MAC:8020
```

Il microfono su iPhone richiede HTTPS con un certificato considerato affidabile.
Per un test locale la strada piu' pulita e' `mkcert`, perche' crea una root CA locale
che puoi installare anche su iPhone.

## 1. Installa mkcert Sul Mac

```bash
brew install mkcert
mkcert -install
```

## 2. Trova L'IP LAN Del Mac

Su Wi-Fi di solito:

```bash
ipconfig getifaddr en0
```

Su Ethernet potrebbe essere:

```bash
ipconfig getifaddr en1
```

Esempio:

```text
192.168.1.50
```

## 3. Crea Il Certificato Server

Sostituisci `192.168.1.50` con l'IP reale del Mac.

```bash
mkdir -p certs

mkcert \
  -cert-file certs/live_tts.pem \
  -key-file certs/live_tts-key.pem \
  localhost 127.0.0.1 192.168.1.50 "$(hostname).local"
```

Il certificato deve includere l'IP che userai da iPhone. Se l'IP cambia, ricrea il
certificato.

## 4. Installa La Root CA Su iPhone

Trova la root CA creata da `mkcert`:

```bash
mkcert -CAROOT
```

Dentro quella cartella c'e' `rootCA.pem`. Invia **solo** `rootCA.pem` all'iPhone
via AirDrop, email o Files. Non inviare mai `live_tts-key.pem`.

Su iPhone:

1. Apri il file `rootCA.pem` e installa il profilo.
2. Vai in `Impostazioni`.
3. Apri `Generali`.
4. Apri `VPN e gestione dispositivo` e installa il profilo, se richiesto.
5. Vai in `Generali` -> `Info` -> `Impostazioni attendibilita certificati`.
6. Abilita la fiducia completa per la root CA di `mkcert`.

## 5. Avvia Whisper

In un terminale:

```bash
uv run uvicorn whisper:app --host 127.0.0.1 --port 8000
```

Whisper puo' restare HTTP su localhost, perche' viene chiamato dal backend
`live_tts`, non direttamente dall'iPhone.

## 6. Avvia Live TTS In HTTPS

Sostituisci `192.168.1.50` con l'IP reale del Mac.

```bash
LIVE_TTS_HOST=0.0.0.0 \
LIVE_TTS_PORT=8020 \
LIVE_TTS_LOG_LEVEL=info \
LIVE_TTS_SSL_CERTFILE=certs/live_tts.pem \
LIVE_TTS_SSL_KEYFILE=certs/live_tts-key.pem \
LIVE_TTS_STT_BACKEND=http \
LIVE_TTS_STT_URL=http://127.0.0.1:8000/transcribe \
LIVE_TTS_LLM_BACKEND=openai \
LIVE_TTS_LLM_URL=http://192.168.0.20:8001/v1/chat/completions \
LIVE_TTS_LLM_MODEL=qwen3.6-35b \
LIVE_TTS_TTS_BACKEND=omnivoice \
LIVE_TTS_MODEL=k2-fsa/OmniVoice \
LIVE_TTS_DEVICE_MAP=cuda:0 \
LIVE_TTS_DTYPE=float16 \
LIVE_TTS_LANGUAGE=it \
LIVE_TTS_INSTRUCT="female, low pitch" \
uv run python -m live_tts
```

Apri da iPhone:

```text
https://192.168.1.50:8020
```

## 7. Debug Rapido

Dal Mac:

```bash
curl -k https://127.0.0.1:8020/health
```

Da iPhone:

- deve essere sulla stessa rete Wi-Fi del Mac;
- devi usare `https://`, non `http://`;
- l'IP nell'URL deve essere incluso nel certificato;
- macOS Firewall deve permettere connessioni in ingresso a Python/Uvicorn;
- se Safari mostra warning certificato, la root CA non e' stata fidata
  completamente su iPhone.

## Note Di Produzione

Per clienti veri e accesso fuori dalla LAN, usa un dominio reale e un certificato
pubblico, oppure termina TLS su un reverse proxy. Il certificato `mkcert` serve solo
per sviluppo locale e demo in rete privata.
