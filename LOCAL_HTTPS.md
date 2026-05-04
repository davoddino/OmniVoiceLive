# HTTPS Locale Su Ubuntu Per iPhone

Questa guida serve per avviare OmniVoice Live su Ubuntu e aprirlo da iPhone:

```text
https://IP_UBUNTU:8020
```

Su iPhone microfono e autoplay richiedono HTTPS. Per una demo locale in LAN usiamo
`mkcert`, che crea una root CA locale da installare anche su iPhone.

## 1. Installa mkcert Su Ubuntu

Prova prima con `apt`:

```bash
sudo apt update
sudo apt install -y mkcert libnss3-tools
mkcert -install
```

Se `apt` non trova `mkcert`, installa almeno `libnss3-tools` e poi installa `mkcert`
dal pacchetto/release adatto alla tua macchina:

```bash
sudo apt update
sudo apt install -y libnss3-tools curl ca-certificates
```

Poi usa il binario Linux corretto per la tua architettura e mettilo in
`/usr/local/bin/mkcert`. Su server x86_64 di solito e' `linux-amd64`.

Verifica:

```bash
mkcert -version
```

## 2. Trova L'IP LAN Di Ubuntu

Metodo rapido:

```bash
hostname -I
```

Prendi l'IP della rete Wi-Fi/LAN, per esempio:

```text
192.168.1.50
```

Metodo piu' esplicito:

```bash
ip addr
```

L'iPhone deve essere sulla stessa rete.

## 3. Crea Il Certificato Server

Sostituisci `192.168.1.50` con l'IP reale di Ubuntu:

```bash
mkdir -p certs

mkcert \
  -cert-file certs/live_tts.pem \
  -key-file certs/live_tts-key.pem \
  localhost 127.0.0.1 192.168.1.50
```

Se userai anche un hostname locale, aggiungilo nello stesso comando, per esempio:

```bash
mkcert \
  -cert-file certs/live_tts.pem \
  -key-file certs/live_tts-key.pem \
  localhost 127.0.0.1 192.168.1.50 ubuntu-live.local
```

Il certificato deve includere esattamente l'IP o hostname che aprirai da iPhone.
Se l'IP cambia, ricrea il certificato.

## 4. Installa La Root CA Su iPhone

Trova la cartella della root CA:

```bash
mkcert -CAROOT
```

Dentro quella cartella c'e' `rootCA.pem`.

Invia **solo** `rootCA.pem` all'iPhone. Non inviare mai
`certs/live_tts-key.pem`.

Opzione comoda da Ubuntu: servi temporaneamente la cartella della CA in HTTP sulla LAN.
Sostituisci `192.168.1.50` con l'IP reale.

```bash
cd "$(mkcert -CAROOT)"
python3 -m http.server 8088 --bind 0.0.0.0
```

Da iPhone apri:

```text
http://192.168.1.50:8088/rootCA.pem
```

Installa il profilo. Poi su iPhone:

1. Apri `Impostazioni`.
2. Vai in `Generali`.
3. Apri `VPN e gestione dispositivo` e installa il profilo, se richiesto.
4. Vai in `Generali` -> `Info` -> `Impostazioni attendibilita certificati`.
5. Abilita la fiducia completa per la root CA di `mkcert`.

Dopo aver installato la CA, ferma il server temporaneo con `CTRL+C`.

## 5. Apri Il Firewall Ubuntu

Se usi `ufw`:

```bash
sudo ufw allow 8020/tcp
sudo ufw status
```

Se vuoi scaricare la root CA da iPhone con il server temporaneo sopra:

```bash
sudo ufw allow 8088/tcp
```

Puoi rimuovere la regola 8088 dopo l'installazione della CA:

```bash
sudo ufw delete allow 8088/tcp
```

## 6. Avvia Whisper

In un terminale:

```bash
uv run uvicorn whisper:app --host 127.0.0.1 --port 8000
```

Whisper resta HTTP su localhost perche' viene chiamato dal backend `live_tts`, non
direttamente dall'iPhone.

Verifica da Ubuntu:

```bash
curl http://127.0.0.1:8000/health
```

## 7. Avvia Live TTS In HTTPS

`live_tts` carica automaticamente `.env` o `live_tts.env` dalla root della repo.
Il file [.env.example](.env.example) contiene la configurazione Ubuntu completa.
Una volta preparato `.env`, in un secondo terminale avvii solo:

```bash
uv run python -m live_tts
```

Da Ubuntu verifica:

```bash
curl -k https://127.0.0.1:8020/health
curl -k https://192.168.0.20:8020/health
```

Da iPhone apri:

```text
https://192.168.1.50:8020
```

## 8. Debug Rapido

Se iPhone non apre la pagina:

- controlla che iPhone e Ubuntu siano sulla stessa rete;
- usa `https://`, non `http://`;
- controlla che l'IP nell'URL sia incluso nel certificato;
- controlla `sudo ufw status`;
- prova da un altro PC: `curl -k https://IP_UBUNTU:8020/health`;
- controlla che `live_tts` stia ascoltando su `0.0.0.0`, non solo su `127.0.0.1`.
- controlla che `/health` mostri `"websocket_support": true`.

Se la UI mostra `WebSocket non aperto code=1006` e nel server non compare
`session connected`, il problema e' quasi sempre supporto WebSocket mancante in
Uvicorn o blocco TLS/browser. Prima verifica:

```bash
uv run python -c "import websockets; print(websockets.__version__)"
curl -k https://IP_UBUNTU:8020/health
```

Se `websockets` manca:

```bash
uv sync
```

oppure:

```bash
uv add websockets
```

Se Safari mostra warning certificato:

- la root CA non e' installata;
- oppure non e' stata abilitata in `Impostazioni attendibilita certificati`;
- oppure hai ricreato i certificati e devi reinstallare la nuova root CA.

Se il microfono non parte:

- assicurati di essere su HTTPS;
- consenti il microfono a Safari;
- ricarica la pagina;
- guarda il messaggio `SISTEMA` nella UI. Su iPhone puo' comparire:

```text
INFO: AudioWorklet non disponibile, uso fallback iOS.
```

Questo e' previsto: il frontend usa un fallback compatibile quando Safari non
espone `AudioWorklet`.

## Note Di Produzione

`mkcert` serve solo per sviluppo locale e demo in rete privata. Per clienti veri
usa un dominio reale e certificati pubblici, oppure termina TLS su un reverse proxy
come Nginx, Caddy o Traefik.
