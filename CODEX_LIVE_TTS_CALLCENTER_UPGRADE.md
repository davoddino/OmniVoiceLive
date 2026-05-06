# Prompt operativo per Codex — Upgrade Live TTS Call Center

## Obiettivo
Portare il progetto Live TTS a un livello superiore senza introdurre complessità non necessaria. Il sistema attuale funziona già bene: la priorità assoluta è migliorare la qualità, la stabilità e la coerenza dell'audio generato dal modello TTS live, in particolare evitando variazioni di tono, timbro, volume, prosodia o identità vocale tra chunk consecutivi.

Lavora in modo continuativo fino al 6 maggio alle ore 08:00, senza fermarti volontariamente dopo la prima modifica. Non usare `sleep` o attese passive. Se completi una parte, passa subito alla successiva priorità utile. Puoi cercare su internet, leggere documentazione aggiornata, confrontare soluzioni tecniche e implementare la migliore opzione compatibile con il progetto. Continua a implementare, testare e rifinire senza stop, mantenendo bassa la complessità.

## Principio guida
Non trasformare il progetto in un mostro enterprise. Migliora ciò che impatta direttamente:

1. qualità voce;
2. continuità audio tra chunk;
3. latenza;
4. robustezza audio in ambienti rumorosi;
5. registrazione e salvataggio audio/trascrizioni;
6. prompt più adatto a un call-center;
7. modalità LLM semplice e modalità RAG veloce;
8. analytics leggeri post-call.

Ogni modifica deve essere semplice, misurabile, reversibile e compatibile con il flusso live.

---

# Priorità 0 — Analisi iniziale obbligatoria

Prima di modificare codice:

1. ispeziona la struttura del repository;
2. identifica il punto esatto in cui avvengono:
   - acquisizione audio;
   - VAD / threshold rumore;
   - STT;
   - chiamata al modello linguistico;
   - chunking della risposta;
   - TTS;
   - playback;
   - salvataggio eventuale;
3. individua dove nasce il problema della voce che cambia tra chunk;
4. scrivi un breve file `CODEX_FINDINGS.md` con:
   - architettura attuale;
   - criticità audio;
   - piano minimo di intervento;
   - file che intendi modificare.

Non fare refactor ampi. Prima correggi la pipeline critica.

---

# Priorità 1 — Stabilità timbro/tono tra chunk TTS

Questo è il problema principale.

## Obiettivo
La voce deve sembrare una singola voce continua, non tante generazioni separate. Tra chunk non devono cambiare:

- timbro;
- tono medio;
- volume percepito;
- velocità;
- prosodia;
- energia;
- stile;
- identità vocale.

## Azioni da implementare

### 1. Configurazione TTS persistente per sessione
Crea un oggetto di configurazione vocale persistente per ogni chiamata/sessione, per esempio:

```ts
VoiceSessionConfig {
  provider: string
  voiceId: string
  model: string
  speed: number
  stability: number
  similarityBoost?: number
  style?: number
  temperature?: number
  seed?: number
  sampleRate: number
  outputFormat: string
  loudnessTargetLUFS: number
}
```

La configurazione deve essere creata una sola volta a inizio sessione e riutilizzata per tutti i chunk.

### 2. Evitare parametri random per chunk
Se il provider TTS supporta seed, voice stability, style, temperature o similarity, non ricalcolarli a ogni chunk. Imposta valori stabili e conservativi.

Indicazione generale:

- `temperature` bassa o disattivata se disponibile;
- `stability` alta;
- `style` basso o medio-basso;
- stessa `voiceId`;
- stesso formato audio;
- stesso sample rate;
- stesso codec.

### 3. Chunking semantico migliore
Non spezzare frasi in punti casuali. Spezza solo su confini naturali:

- punto;
- punto interrogativo;
- punto esclamativo;
- punto e virgola;
- due punti;
- fine frase ragionevole;
- massimo caratteri configurabile.

Evita chunk troppo corti. I chunk troppo piccoli peggiorano naturalezza e coerenza.

Implementa una soglia minima consigliata:

- chunk minimo: 80-120 caratteri, salvo frasi urgenti;
- chunk massimo: 250-450 caratteri;
- non tagliare dentro numeri, date, indirizzi email, codici cliente, importi, URL.

### 4. Sentence buffer prima del TTS
Il modello può streammare token, ma il TTS non deve ricevere token grezzi. Usa un `SentenceAccumulator`:

`LLM stream -> SentenceAccumulator -> TTS queue -> playback`

Il buffer deve inviare al TTS solo testo sufficientemente stabile.

### 5. Pre-normalizzazione testo
Prima del TTS normalizza:

- spazi multipli;
- emoji;
- markdown;
- elenchi puntati troppo spezzati;
- abbreviazioni ambigue;
- numeri telefonici;
- importi;
- date;
- sigle;
- punteggiatura eccessiva.

Il TTS deve ricevere testo parlabile, non testo da chat.

### 6. Crossfade leggero tra chunk
Se la pipeline audio lo permette, aggiungi un crossfade minimo tra chunk consecutivi:

- 10-30 ms;
- solo se non introduce latenza percepibile;
- opzionale e disattivabile da config.

### 7. Loudness normalization
Normalizza i chunk allo stesso loudness percepito, per esempio target circa -16 LUFS o RMS equivalente se LUFS non è disponibile. Evita salti di volume tra chunk.

Non introdurre librerie pesanti se non necessarie. Se esiste già una pipeline audio, usa quella.

### 8. Coda playback robusta
La coda audio deve:

- mantenere ordine dei chunk;
- evitare overlap errato;
- permettere interruzione per barge-in;
- cancellare chunk pendenti se l'utente parla;
- loggare chunk generati, riprodotti, cancellati.

---

# Priorità 2 — Qualità audio input e threshold rumore

## Obiettivo
Rendere l'ascolto più robusto in ambienti rumorosi senza complicare il sistema.

## Azioni

1. Implementa threshold rumore adattivo se non esiste già.
2. Calibra il rumore di fondo nei primi 500-1500 ms della chiamata.
3. Usa hysteresis:
   - soglia più alta per iniziare parlato;
   - soglia più bassa per continuare parlato;
   - timeout breve per fine parlato.
4. Aggiungi config semplice:

```env
VAD_NOISE_CALIBRATION_MS=1000
VAD_START_MULTIPLIER=2.2
VAD_CONTINUE_MULTIPLIER=1.4
VAD_MIN_SPEECH_MS=180
VAD_END_SILENCE_MS=500
```

5. Se esistono già noise suppression, AGC o echo cancellation, assicurati che siano attivi e configurabili.
6. Non aggiungere pipeline DSP complessa se non necessaria.

---

# Priorità 3 — Prompt più da call-center

## Obiettivo
Migliorare il comportamento del modello linguistico affinché sembri un operatore professionale, chiaro, sintetico e orientato alla risoluzione.

## Requisiti del prompt

Il prompt deve istruire il modello a:

- parlare come operatore call-center professionale;
- essere cortese ma non prolisso;
- fare una domanda alla volta;
- non fare monologhi;
- confermare i dati critici;
- chiedere chiarimenti quando necessario;
- non inventare policy o informazioni aziendali;
- usare frasi brevi e adatte alla voce;
- evitare markdown, liste lunghe, parentesi inutili e formati non pronunciabili;
- gestire interruzioni dell'utente;
- ammettere quando non sa qualcosa;
- passare a fallback umano solo nei casi necessari e solo se già supportato dal sistema;
- non cambiare identità, tono o lingua senza motivo;
- mantenere una voce coerente, calma, efficiente.

## Prompt base consigliato

Integra o adatta questo system prompt:

```text
Sei un assistente vocale per call-center. Devi parlare in modo naturale, professionale, breve e orientato alla risoluzione.

Regole fondamentali:
- Rispondi con frasi brevi, facili da pronunciare ad alta voce.
- Fai una sola domanda alla volta.
- Non usare markdown, elenchi lunghi, tabelle o formattazioni da testo scritto.
- Non fare monologhi: massimo 2-3 frasi prima di lasciare spazio all'utente.
- Se devi verificare un dato, chiedilo in modo semplice e confermalo.
- Se non hai informazioni sufficienti, chiedi chiarimento invece di inventare.
- Non inventare policy, prezzi, disponibilità, procedure o dati cliente.
- Mantieni tono calmo, competente e cordiale.
- Se l'utente è arrabbiato, riconosci il problema e porta la conversazione al passo successivo concreto.
- Se l'utente interrompe, fermati e rispondi all'ultimo messaggio dell'utente.
- Ottimizza le risposte per TTS: punteggiatura naturale, niente simboli inutili, niente testo difficile da pronunciare.

Obiettivo: risolvere la richiesta dell'utente nel minor numero di turni possibile, senza sembrare robotico e senza aumentare la latenza.
```

## Prompt per TTS friendliness
Aggiungi una fase leggera di normalizzazione della risposta prima del TTS, oppure istruisci il modello così:

```text
Scrivi sempre in forma parlata. Evita slash, parentesi, markdown, sigle non spiegate e numerazioni complesse. Usa punteggiatura naturale per guidare il parlato.
```

---

# Priorità 4 — Modalità LLM semplice e modalità RAG

Servono entrambe.

## Modalità A — Fast LLM only
Percorso più veloce possibile:

`audio -> STT -> prompt call-center -> LLM -> sentence buffer -> TTS -> playback`

Questa modalità deve essere default quando non serve conoscenza esterna.

## Modalità B — Fast RAG
Percorso con recupero documenti:

`audio -> STT -> intent check leggero -> retrieval rapido -> prompt con contesto -> LLM -> sentence buffer -> TTS -> playback`

## Requisiti RAG

1. RAG deve essere opzionale e disattivabile.
2. Retrieval deve avere timeout rigido.
3. Se il retrieval è lento, procedi con fallback LLM semplice.
4. Non bloccare la chiamata live inutilmente.
5. Il contesto RAG deve essere corto e rilevante.
6. Non inserire documenti lunghi nel prompt.
7. Logga quando RAG è usato e quanto tempo impiega.

Config consigliata:

```env
RAG_ENABLED=true
RAG_TIMEOUT_MS=300
RAG_MAX_CHUNKS=3
RAG_MAX_CONTEXT_CHARS=2500
RAG_FALLBACK_TO_LLM=true
```

---

# Priorità 5 — Registrazione audio e trascrizioni

## Obiettivo
Salvare audio e trascrizioni senza interferire con la latenza live.

## Regola fondamentale
Il salvataggio deve essere asincrono e non deve bloccare TTS, STT o playback.

## Requisiti

1. Avvia registrazione appena la sessione/chiamata parte, prima del TTS live.
2. Se possibile, mantieni un pre-buffer audio di alcuni secondi.
3. Salva audio input utente e, se possibile, audio output TTS.
4. Salva trascrizione turn-by-turn:
   - timestamp;
   - speaker: user/assistant/system;
   - testo;
   - latenza STT;
   - latenza LLM;
   - latenza TTS;
   - chunk id;
   - eventuale uso RAG.
5. Salva un file metadata per ogni chiamata.
6. Non fare upload remoto sincrono nel path live.
7. Usa una coda locale o writer async.

Struttura consigliata:

```text
recordings/
  2026-05-05/
    call_<session_id>/
      input.wav
      output_tts.wav
      mixed.wav
      transcript.jsonl
      metadata.json
      latency_metrics.json
```

## Transcript JSONL esempio

```json
{"ts":"2026-05-05T12:00:01.120Z","speaker":"user","text":"Buongiorno, vorrei informazioni.","stt_ms":180}
{"ts":"2026-05-05T12:00:01.620Z","speaker":"assistant","text":"Buongiorno, certo. Mi dica pure di cosa ha bisogno.","llm_ms":220,"tts_ms":310,"chunk_id":1}
```

---

# Priorità 6 — CRM separato dal live path

Il CRM non deve stare nel percorso live.

## Regola
Durante la chiamata live non fare chiamate CRM bloccanti.

## Architettura corretta

`live call -> local event log -> post-call worker -> CRM sync`

## Azioni

1. Crea eventi locali append-only.
2. Dopo la chiamata, genera summary e payload CRM.
3. Sincronizza CRM in background o manualmente.
4. Se CRM fallisce, non deve impattare chiamata o registrazione.
5. Lascia interfaccia/adattatore per futuri CRM, ma non implementare integrazioni pesanti se non richieste.

---

# Priorità 7 — Analytics leggeri

Implementa analytics post-call semplici, non enterprise.

Metriche utili:

- durata chiamata;
- numero turni;
- tempo medio STT;
- tempo medio LLM;
- tempo medio TTS;
- tempo al primo audio;
- numero interruzioni/barge-in;
- numero chunk TTS;
- chunk cancellati;
- uso RAG sì/no;
- errori provider;
- fallback usati;
- stima esito: risolto / non risolto / escalation.

Salva in `latency_metrics.json` o database locale se già presente.

---

# Priorità 8 — Escalation semplice, solo se non complica

Implementa solo una logica minimale se già compatibile con il sistema.

Trigger possibili:

- bassa confidence STT/LLM;
- utente ripete lo stesso problema più volte;
- errore tecnico;
- richiesta non supportata;
- utente chiede esplicitamente un operatore.

Non costruire un sistema complesso di escalation. Basta un flag/evento:

```json
{"event":"escalation_requested","reason":"user_requested_human"}
```

---

# Cose da NON fare

Non implementare ora:

- simulatore chiamate;
- compliance enterprise avanzata;
- dashboard complessa da supervisor;
- marketplace;
- CRM sincrono nel live path;
- modalità human-like latency artificiale;
- pause finte o ritardi intenzionali;
- refactor architetturali enormi;
- nuove dipendenze pesanti non necessarie;
- sistemi multi-agent complessi;
- UI enorme se prima non è sistemata la voce.

---

# UI: miglioramenti ammessi ma secondari

Se resta tempo, migliora l'interfaccia solo dove aiuta il debug e il controllo qualità:

- stato sessione live;
- livello rumore;
- stato VAD;
- chunk TTS in coda;
- provider TTS attivo;
- latenza STT/LLM/TTS;
- pulsante start/stop recording;
- link a registrazione e trascrizione;
- indicatore RAG on/off.

Non fare redesign estetico se la voce non è stata prima migliorata.

---

# Test obbligatori

Aggiungi o aggiorna test leggeri per:

1. chunking semantico;
2. normalizzazione testo TTS;
3. stabilità config TTS per sessione;
4. salvataggio transcript JSONL;
5. timeout RAG;
6. writer async non bloccante;
7. cancellazione coda TTS su barge-in, se presente.

Se non esiste framework test, crea script di verifica semplici.

---

# Criteri di successo

Il lavoro è riuscito se:

- la voce mantiene stesso timbro tra chunk;
- i chunk non sembrano frasi scollegate;
- il volume resta costante;
- la latenza non peggiora sensibilmente;
- il rumore di fondo viene gestito meglio;
- audio e trascrizioni vengono salvati senza bloccare il live;
- LLM-only e RAG sono entrambe disponibili;
- CRM resta fuori dal percorso live;
- analytics minimi sono salvati;
- il codice resta semplice e leggibile.

---

# Output finale richiesto a Codex

Alla fine produci:

1. riepilogo modifiche;
2. file modificati;
3. come avviare il sistema;
4. variabili env nuove;
5. test eseguiti;
6. problemi rimasti;
7. prossimi step consigliati solo se strettamente necessari.

Non fermarti dopo un'analisi superficiale. Implementa, testa, misura e migliora finché ci sono interventi utili e compatibili con la semplicità del progetto.
