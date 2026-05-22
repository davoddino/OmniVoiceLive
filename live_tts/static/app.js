const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const stateLabel = document.getElementById("stateLabel");
const sessionLabel = document.getElementById("sessionLabel");
const meterFill = document.getElementById("meterFill");
const conversation = document.getElementById("conversation");
const sttLatency = document.getElementById("sttLatency");
const ttsLatency = document.getElementById("ttsLatency");
const turnLabel = document.getElementById("turnLabel");
const bufferLabel = document.getElementById("bufferLabel");
const timerLabel = document.getElementById("timerLabel");
const transportLabel = document.getElementById("transportLabel");
const connectionLabel = document.getElementById("connectionLabel");
const audioModeLabel = document.getElementById("audioModeLabel");
const voiceModeLabel = document.getElementById("voiceModeLabel");
const sessionModeSelect = document.getElementById("sessionModeSelect");
const sourceLanguageControl = document.getElementById("sourceLanguageControl");
const sourceLanguageSelect = document.getElementById("sourceLanguageSelect");
const languageSelectLabel = document.getElementById("languageSelectLabel");
const languageSelect = document.getElementById("languageSelect");
const translationTimingControl = document.getElementById("translationTimingControl");
const translationTimingSelect = document.getElementById("translationTimingSelect");
const ttsEngineSelect = document.getElementById("ttsEngineSelect");

const LANGUAGE_LABELS = {
  it: "Italiano",
  en: "English",
  de: "Tedesco (Deutsch)",
  fr: "Francese (Français)",
  es: "Spagnolo (Español)",
  pt: "Portoghese (Português)",
  ro: "Rumeno (Română)",
  sq: "Albanese (Shqip)",
  ru: "Russo (Русский)",
  uk: "Ucraino (Українська)",
  pl: "Polski",
  sr: "Serbo/Croato/Bosniaco",
  hr: "Croato (Hrvatski)",
  bs: "Bosniaco (Bosanski)",
  ar: "Arabo (العربية)",
  zh: "Cinese semplificato (简体中文)",
  hi: "Hindi (हिन्दी)",
  ur: "Urdu (اردو)",
  sw: "Swahili (Kiswahili)",
};

const SOURCE_LANGUAGE_LABELS = {
  auto: "Auto rilevamento",
  ...LANGUAGE_LABELS,
};

const ENGINE_LABELS = {
  omnivoice: "OmniVoice",
  qwen3_tts: "Qwen3-TTS",
};

let socket = null;
let audioContext = null;
let recorderNode = null;
let recorderSinkNode = null;
let playerNode = null;
let mediaStream = null;
let ttsSampleRate = 24000;
let fallbackPlayerQueue = [];
let fallbackPlayerOffset = 0;
let fallbackQueuedSamples = 0;
let fallbackBufferLastAt = 0;
let activeAssistantMessage = null;
let assistantSpeaking = false;
let assistantPlaybackActive = false;
let assistantDonePending = false;
let bargeSent = false;
let audioPlaybackBlocked = false;
let activeAudioTurnId = null;
let blockedAudioTurnId = null;
let playerBufferedMs = 0;
let currentTurnId = null;
let socketWasOpen = false;
let socketOpenTimer = null;
let cleaningUp = false;
let callStartedAt = 0;
let timerInterval = null;
let sessionShortId = null;
let sessionStarted = false;
let audioStreamingPaused = false;
let pendingSessionStart = false;
let pendingMicFrames = [];
let pendingMicSamples = 0;
let clientBargeEnabled = true;
let activeSessionMode = "agent";
let activeTranslationTiming = "immediate";
let translatorImmediateBufferMs = 2000;

const pendingMicMaxMs = 1400;

const clientVad = {
  threshold: 0.012,
  stopMs: 20,
  commitMs: 45,
  cooldownMs: 700,
  speechMs: 0,
  interruptMs: 0,
  lastBargeAt: Number.NEGATIVE_INFINITY,
};

startButton.addEventListener("click", startCall);
stopButton.addEventListener("click", stopCall);
sessionModeSelect.addEventListener("change", updateModeControls);
sourceLanguageSelect.addEventListener("change", switchToTranslatorFromLanguageControl);
languageSelect.addEventListener("change", updateModeControls);
translationTimingSelect.addEventListener("change", updateModeControls);
ttsEngineSelect.addEventListener("change", onTtsEngineChange);
updateModeControls();

async function startCall() {
  try {
    cleaningUp = false;
    setState("connecting", "Connessione");
    startButton.disabled = true;
    sessionModeSelect.disabled = true;
    sourceLanguageSelect.disabled = true;
    languageSelect.disabled = true;
    translationTimingSelect.disabled = true;
    ttsEngineSelect.disabled = true;
    transportLabel.textContent =
      location.protocol === "https:" ? "HTTPS/WSS" : "HTTP/WS";
    connectionLabel.textContent = "Avvio";
    updateEngineStatus({
      engine: selectedEngine(),
      status: "connecting",
      sample_rate: ttsSampleRate,
    });

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      throw new Error("Web Audio API non disponibile in questo browser.");
    }
    addSystemMessage("INFO: Creo AudioContext.");
    audioContext = new AudioContextClass({ latencyHint: "interactive" });
    addSystemMessage(
      `INFO: AudioContext state=${audioContext.state}, sampleRate=${audioContext.sampleRate}.`,
    );

    addSystemMessage("INFO: Richiedo accesso al microfono.");
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    addSystemMessage("INFO: Microfono autorizzato.");
    const source = audioContext.createMediaStreamSource(mediaStream);

    const canUseWorklet = Boolean(audioContext.audioWorklet && window.AudioWorkletNode);
    if (canUseWorklet) {
      try {
        await setupAudioWorkletGraph(source);
        audioModeLabel.textContent = "AudioWorklet";
        addSystemMessage("INFO: AudioWorklet attivo.");
      } catch (error) {
        addSystemMessage(
          `INFO: AudioWorklet non disponibile, uso fallback iOS. Dettaglio: ${error.message}`,
        );
        setupScriptProcessorGraph(source);
        audioModeLabel.textContent = "Fallback iOS";
      }
    } else {
      addSystemMessage("INFO: AudioWorklet non disponibile, uso fallback iOS.");
      setupScriptProcessorGraph(source);
      audioModeLabel.textContent = "Fallback iOS";
    }

    connectWebSocket();
    resumeAudioContextWithTimeout();
  } catch (error) {
    cleanup("Avvio non riuscito");
    setState("error", "Errore");
    addSystemMessage(error.message || "Impossibile avviare audio e microfono.");
  }
}

function stopCall() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "session.stop" }));
  }
  cleanup("Chiamata terminata");
}

function cleanup(label) {
  cleaningUp = true;
  if (socketOpenTimer) {
    clearTimeout(socketOpenTimer);
    socketOpenTimer = null;
  }
  if (playerNode) {
    clearPlayer();
  }
  if (recorderNode) {
    try {
      recorderNode.disconnect();
    } catch (_error) {
      // Already disconnected.
    }
  }
  if (recorderSinkNode) {
    try {
      recorderSinkNode.disconnect();
    } catch (_error) {
      // Already disconnected.
    }
  }
  if (playerNode) {
    try {
      playerNode.disconnect();
    } catch (_error) {
      // Already disconnected.
    }
  }
  if (mediaStream) {
    for (const track of mediaStream.getTracks()) {
      track.stop();
    }
  }
  if (
    socket &&
    (socket.readyState === WebSocket.OPEN ||
      socket.readyState === WebSocket.CONNECTING)
  ) {
    socket.close();
  }
  if (audioContext && audioContext.state !== "closed") {
    audioContext.close().catch(() => {});
  }
  socket = null;
  socketWasOpen = false;
  sessionShortId = null;
  sessionStarted = false;
  audioStreamingPaused = false;
  pendingSessionStart = false;
  audioContext = null;
  mediaStream = null;
  recorderNode = null;
  recorderSinkNode = null;
  playerNode = null;
  fallbackPlayerQueue = [];
  fallbackPlayerOffset = 0;
  fallbackQueuedSamples = 0;
  pendingMicFrames = [];
  pendingMicSamples = 0;
  clientBargeEnabled = true;
  activeSessionMode = selectedMode();
  activeTranslationTiming = selectedTranslationTiming();
  playerBufferedMs = 0;
  assistantSpeaking = false;
  assistantPlaybackActive = false;
  assistantDonePending = false;
  bargeSent = false;
  audioPlaybackBlocked = false;
  activeAudioTurnId = null;
  blockedAudioTurnId = null;
  resetClientVad();
  stopTimer();
  startButton.disabled = false;
  sessionModeSelect.disabled = false;
  sourceLanguageSelect.disabled = false;
  languageSelect.disabled = false;
  translationTimingSelect.disabled = false;
  ttsEngineSelect.disabled = false;
  stopButton.disabled = true;
  sessionLabel.textContent = label;
  connectionLabel.textContent = "Offline";
  audioModeLabel.textContent = "Audio in attesa";
  updateModeControls();
  setState("idle", "Idle");
  setTimeout(() => {
    cleaningUp = false;
  }, 0);
}

function onMicFrame(event) {
  const frame = event.data;
  updateClientVad(frame);
  if (
    !socket ||
    socket.readyState !== WebSocket.OPEN ||
    !sessionStarted ||
    audioStreamingPaused
  ) {
    rememberPendingMicFrame(frame);
    return;
  }
  socket.send(frame.buffer);
}

async function setupAudioWorkletGraph(source) {
  await audioContext.audioWorklet.addModule("/static/recorder-worklet.js");
  await audioContext.audioWorklet.addModule("/static/player-worklet.js");

  playerNode = new AudioWorkletNode(audioContext, "player-worklet");
  playerNode.connect(audioContext.destination);
  playerNode.port.onmessage = (event) => {
    if (event.data.type === "buffer") {
      updatePlayerBuffer(event.data.ms);
    }
  };

  recorderNode = new AudioWorkletNode(audioContext, "recorder-worklet");
  source.connect(recorderNode);
  recorderSinkNode = audioContext.createGain();
  recorderSinkNode.gain.value = 0;
  recorderNode.connect(recorderSinkNode);
  recorderSinkNode.connect(audioContext.destination);
  recorderNode.port.onmessage = onMicFrame;
}

function setupScriptProcessorGraph(source) {
  playerNode = audioContext.createScriptProcessor(1024, 0, 1);
  playerNode.onaudioprocess = onFallbackPlayerProcess;
  playerNode.connect(audioContext.destination);

  recorderNode = audioContext.createScriptProcessor(2048, 1, 1);
  recorderNode.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const frame = new Float32Array(input.length);
    frame.set(input);

    const output = event.outputBuffer.getChannelData(0);
    output.fill(0);

    onMicFrame({ data: frame });
  };
  source.connect(recorderNode);
  recorderNode.connect(audioContext.destination);
}

function connectWebSocket() {
  const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${wsProto}//${location.host}/ws`;
  addSystemMessage(`INFO: Apro WebSocket ${wsUrl}`);
  connectionLabel.textContent = "Connessione";

  socketWasOpen = false;
  socket = new WebSocket(wsUrl);
  const ws = socket;
  let wsOpened = false;
  ws.binaryType = "arraybuffer";

  socketOpenTimer = setTimeout(() => {
    if (socket === ws && ws.readyState !== WebSocket.OPEN) {
      addSystemMessage(
        `INFO: WebSocket ancora non aperto dopo 5s, readyState=${ws.readyState}.`,
      );
      setState("error", "Errore");
      ws.close();
    }
  }, 5000);

  ws.onopen = () => {
    if (socket !== ws) {
      return;
    }
    wsOpened = true;
    socketWasOpen = true;
    if (socketOpenTimer) {
      clearTimeout(socketOpenTimer);
      socketOpenTimer = null;
    }
    addSystemMessage("INFO: WebSocket aperto.");
    connectionLabel.textContent = "Online";
    pendingSessionStart = true;
    audioStreamingPaused = true;
    ws.send(
      JSON.stringify({
        type: "tts.engine.select",
        engine: selectedEngine(),
      }),
    );
    startTimer();
  };

  ws.onmessage = onSocketMessage;
  ws.onclose = (event) => {
    if (socket !== ws) {
      return;
    }
    if (socketOpenTimer) {
      clearTimeout(socketOpenTimer);
      socketOpenTimer = null;
    }
    if (cleaningUp) {
      return;
    }
    const reason = event.reason ? ` reason=${event.reason}` : "";
    if (wsOpened || socketWasOpen || event.code === 1000) {
      addSystemMessage(`INFO: WebSocket chiuso code=${event.code}${reason}.`);
      connectionLabel.textContent = "Offline";
      cleanup("Chiamata terminata");
    } else {
      cleanup("WebSocket non aperto");
      setState("error", "Errore");
      addSystemMessage(`INFO: WebSocket non aperto code=${event.code}${reason}.`);
    }
  };

  ws.onerror = () => {
    if (socket !== ws) {
      return;
    }
    addSystemMessage("INFO: Errore WebSocket. Controlla certificato, IP e firewall.");
    connectionLabel.textContent = "Errore";
    setState("error", "Errore");
  };
}

function rememberPendingMicFrame(frame) {
  if (!audioContext || !frame || frame.length === 0) {
    return;
  }
  const copy = new Float32Array(frame.length);
  copy.set(frame);
  pendingMicFrames.push(copy);
  pendingMicSamples += copy.length;

  const maxSamples = Math.max(
    1,
    Math.round((audioContext.sampleRate * pendingMicMaxMs) / 1000),
  );
  while (pendingMicSamples > maxSamples && pendingMicFrames.length > 0) {
    const dropped = pendingMicFrames.shift();
    pendingMicSamples -= dropped ? dropped.length : 0;
  }
}

function flushPendingMicFrames() {
  if (!socket || socket.readyState !== WebSocket.OPEN || pendingMicFrames.length === 0) {
    pendingMicFrames = [];
    pendingMicSamples = 0;
    return;
  }
  const frameCount = pendingMicFrames.length;
  const durationMs = audioContext
    ? Math.round((pendingMicSamples / audioContext.sampleRate) * 1000)
    : 0;
  for (const frame of pendingMicFrames) {
    socket.send(frame.buffer);
  }
  pendingMicFrames = [];
  pendingMicSamples = 0;
  addSystemMessage(
    `INFO: Inviato preroll microfono iniziale ${durationMs} ms (${frameCount} frame).`,
  );
}

function onTtsEngineChange() {
  updateEngineStatus({
    engine: selectedEngine(),
    status: socket && socket.readyState === WebSocket.OPEN ? "loading" : "selected",
    sample_rate: ttsSampleRate,
  });
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  audioStreamingPaused = true;
  clearPlayer();
  socket.send(
    JSON.stringify({
      type: "tts.engine.select",
      engine: selectedEngine(),
    }),
  );
}

function sendSessionStart() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  pendingSessionStart = false;
  activeSessionMode = selectedMode();
  activeTranslationTiming = selectedTranslationTiming();
  const modeLabel =
    activeSessionMode === "translator" ? "traduttore live" : "agente CavadaLabs";
  const targetLabel = LANGUAGE_LABELS[selectedLanguage()] || selectedLanguage();
  const sourceLabel =
    SOURCE_LANGUAGE_LABELS[selectedSourceLanguage()] || selectedSourceLanguage();
  addSystemMessage(
    activeSessionMode === "translator"
      ? `INFO: Avvio modalità ${modeLabel}: ${sourceLabel} -> ${targetLabel}, timing=${activeTranslationTiming}.`
      : `INFO: Avvio modalità ${modeLabel}, lingua=${targetLabel}.`,
  );
  socket.send(
    JSON.stringify({
      type: "session.start",
      sample_rate: audioContext ? audioContext.sampleRate : 48000,
      mode: activeSessionMode,
      language: selectedLanguage(),
      source_language: selectedSourceLanguage(),
      target_language: selectedLanguage(),
      translation_timing: activeTranslationTiming,
      tts_engine: selectedEngine(),
    }),
  );
}

async function resumeAudioContextWithTimeout(timeoutMs = 1200) {
  if (!audioContext) {
    return;
  }
  addSystemMessage(`INFO: Resume AudioContext da state=${audioContext.state}.`);
  try {
    const result = await Promise.race([
      audioContext.resume().then(() => "ok"),
      new Promise((resolve) => {
        setTimeout(() => resolve("timeout"), timeoutMs);
      }),
    ]);
    addSystemMessage(
      `INFO: Resume AudioContext result=${result}, state=${audioContext.state}.`,
    );
  } catch (error) {
    addSystemMessage(`INFO: Resume AudioContext fallito: ${error.message}.`);
  }
}

function updateClientVad(frame) {
  let sum = 0;
  for (let i = 0; i < frame.length; i += 1) {
    sum += frame[i] * frame[i];
  }
  const rms = Math.sqrt(sum / Math.max(1, frame.length));
  const level = Math.min(100, Math.round((rms / 0.08) * 100));
  meterFill.style.width = `${level}%`;

  if (rms >= clientVad.threshold) {
    clientVad.speechMs += (frame.length / audioContext.sampleRate) * 1000;
  } else {
    clientVad.speechMs = 0;
  }

  const now = performance.now();
  if (!clientBargeEnabled) {
    clientVad.interruptMs = 0;
    return;
  }
  if (
    (isAssistantAudioActive() || audioPlaybackBlocked) &&
    rms >= clientVad.threshold &&
    now - clientVad.lastBargeAt > clientVad.cooldownMs
  ) {
    clientVad.interruptMs += (frame.length / audioContext.sampleRate) * 1000;
  } else if (rms < clientVad.threshold) {
    clientVad.interruptMs = 0;
  }

  if (
    isAssistantAudioActive() &&
    clientVad.interruptMs >= clientVad.stopMs &&
    !audioPlaybackBlocked
  ) {
    blockAssistantAudioPlayback(rms);
  }

  if (
    audioPlaybackBlocked &&
    clientVad.interruptMs >= clientVad.commitMs &&
    !bargeSent &&
    now - clientVad.lastBargeAt > clientVad.cooldownMs
  ) {
    bargeSent = true;
    clientVad.lastBargeAt = now;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "barge_in" }));
    }
    setState("interrupted", "Interrotto");
  }
}

function blockAssistantAudioPlayback(rms) {
  if (!clientBargeEnabled) {
    return;
  }
  audioPlaybackBlocked = true;
  blockedAudioTurnId = activeAudioTurnId || currentTurnId;
  assistantSpeaking = false;
  assistantPlaybackActive = false;
  assistantDonePending = false;
  clearPlayer();
  bufferLabel.textContent = "0 ms";
  sendBargeIn();
  addSystemMessage(
    `INFO: Barge-in locale rms=${rms.toFixed(4)} soglia=${clientVad.threshold}.`,
  );
  connectionLabel.textContent = "Interrotto";
  setState("interrupted", "Interrotto");
}

function resetClientVad() {
  clientVad.speechMs = 0;
  clientVad.interruptMs = 0;
}

function isAssistantAudioActive() {
  return assistantSpeaking || assistantPlaybackActive || playerBufferedMs > 0;
}

function sendBargeIn() {
  if (!clientBargeEnabled) {
    return;
  }
  const now = performance.now();
  if (bargeSent || now - clientVad.lastBargeAt <= clientVad.cooldownMs) {
    return;
  }
  bargeSent = true;
  clientVad.lastBargeAt = now;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "barge_in" }));
  }
}

function updatePlayerBuffer(ms) {
  playerBufferedMs = Math.max(0, Number(ms) || 0);
  bufferLabel.textContent = `${playerBufferedMs} ms`;
  if (playerBufferedMs > 0 && !audioPlaybackBlocked) {
    assistantPlaybackActive = true;
  }
  if (playerBufferedMs === 0 && assistantDonePending && !audioPlaybackBlocked) {
    assistantPlaybackActive = false;
    finishAssistantPlayback();
  } else if (playerBufferedMs === 0 && !assistantSpeaking) {
    assistantPlaybackActive = false;
  }
}

function finishAssistantPlayback() {
  assistantSpeaking = false;
  assistantPlaybackActive = false;
  assistantDonePending = false;
  audioPlaybackBlocked = false;
  activeAudioTurnId = null;
  blockedAudioTurnId = null;
  resetClientVad();
  connectionLabel.textContent = "Online";
  setState("listening", "Ascolto");
}

function onSocketMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    enqueueAudio(event.data);
    return;
  }

  const message = JSON.parse(event.data);
  switch (message.type) {
    case "session.ready":
      ttsSampleRate = message.tts_sample_rate;
      configureClientBargeIn(message);
      configureLanguages(message.languages, message.source_languages);
      configureSessionDefaults(message);
      configureTtsEngines(message.tts_engines, message.tts_engine);
      updateEngineStatus(message.tts_engine);
      sessionShortId = message.session_id.slice(0, 8);
      updateSessionLabel({
        mode: selectedMode(),
        language: message.language || selectedLanguage(),
        sourceLanguage: selectedSourceLanguage(),
        targetLanguage: selectedLanguage(),
        translationTiming: selectedTranslationTiming(),
      });
      break;
    case "session.started":
      sessionStarted = true;
      audioStreamingPaused = false;
      configureClientBargeIn(message);
      activeSessionMode = message.mode || selectedMode();
      activeTranslationTiming = message.translation_timing || selectedTranslationTiming();
      if (message.language) {
        updateSessionLabel({
          mode: activeSessionMode,
          language: message.language,
          sourceLanguage: message.source_language || selectedSourceLanguage(),
          targetLanguage: message.target_language || message.language,
          translationTiming: activeTranslationTiming,
        });
      }
      if (message.tts_sample_rate) {
        ttsSampleRate = message.tts_sample_rate;
      }
      updateEngineStatus(message.tts_engine);
      connectionLabel.textContent = "Online";
      ttsEngineSelect.disabled = false;
      stopButton.disabled = false;
      flushPendingMicFrames();
      setState("listening", "Ascolto");
      break;
    case "tts.engine.loading":
      audioStreamingPaused = true;
      ttsEngineSelect.disabled = true;
      clearPlayer();
      updateEngineStatus({
        engine: message.engine,
        status: "loading",
        sample_rate: ttsSampleRate,
      });
      connectionLabel.textContent = "Carico TTS";
      setState("connecting", "Carico voce");
      break;
    case "tts.engine.ready":
      audioStreamingPaused = false;
      if (message.tts_sample_rate || message.sample_rate) {
        ttsSampleRate = message.tts_sample_rate || message.sample_rate;
      }
      configureTtsEngines(message.tts_engines, message);
      updateEngineStatus(message);
      ttsEngineSelect.disabled = false;
      if (sessionStarted) {
        flushPendingMicFrames();
        connectionLabel.textContent = "Online";
        setState("listening", "Ascolto");
      } else if (pendingSessionStart) {
        sendSessionStart();
      }
      break;
    case "tts.engine.error":
      audioStreamingPaused = false;
      ttsEngineSelect.disabled = false;
      configureTtsEngines(message.tts_engines, {
        engine: message.engine,
        status: "failed",
      });
      updateEngineStatus({
        engine: message.engine,
        status: "failed",
        error: message.message,
      });
      addSystemMessage(message.message || "Engine TTS non disponibile.");
      connectionLabel.textContent = "Errore TTS";
      setState("error", "Errore");
      if (!sessionStarted) {
        startButton.disabled = false;
        languageSelect.disabled = false;
        stopButton.disabled = true;
      }
      break;
    case "audio.meter":
      updateServerMeter(message.rms);
      break;
    case "vad.speech_start":
      connectionLabel.textContent = "Utente";
      setState("user-speaking", "Utente parla");
      break;
    case "vad.speech_end":
      connectionLabel.textContent = "STT";
      setState("transcribing", "Trascrivo");
      break;
    case "stt.final":
      currentTurnId = message.turn_id;
      turnLabel.textContent = String(message.turn_id);
      sttLatency.textContent = `${message.latency_ms} ms`;
      if (message.text) {
        addMessage("user", "Utente", message.text);
      }
      connectionLabel.textContent = "LLM";
      setState("thinking", "Elaboro");
      break;
    case "assistant.thinking":
      prepareAssistantMessage(message.turn_id);
      setState("thinking", "Elaboro");
      break;
    case "assistant.text_delta":
      appendAssistantText(message.turn_id, message.text);
      break;
    case "assistant.audio_start":
      activeAudioTurnId = message.turn_id;
      if (blockedAudioTurnId === message.turn_id) {
        audioPlaybackBlocked = true;
        clearPlayer();
        break;
      }
      audioPlaybackBlocked = false;
      assistantSpeaking = true;
      assistantPlaybackActive = false;
      assistantDonePending = false;
      bargeSent = false;
      resetClientVad();
      connectionLabel.textContent = "TTS";
      setState("speaking", "Rispondo");
      break;
    case "assistant.audio_ready":
      ttsLatency.textContent = `${message.latency_ms} ms`;
      break;
    case "assistant.done":
      assistantSpeaking = false;
      assistantDonePending = true;
      if (!assistantPlaybackActive && playerBufferedMs === 0) {
        finishAssistantPlayback();
      }
      break;
    case "turn.cancelled":
      assistantSpeaking = false;
      assistantPlaybackActive = false;
      assistantDonePending = false;
      activeAudioTurnId = null;
      resetClientVad();
      clearPlayer();
      audioPlaybackBlocked = blockedAudioTurnId === message.turn_id;
      if (!audioPlaybackBlocked) {
        blockedAudioTurnId = null;
      }
      connectionLabel.textContent = "Interrotto";
      setState("interrupted", "Interrotto");
      break;
    case "turn.empty":
      connectionLabel.textContent = "Online";
      setState("listening", "Ascolto");
      break;
    case "error":
      addSystemMessage(message.message || "Errore non specificato.");
      connectionLabel.textContent = "Errore";
      setState("error", "Errore");
      break;
    default:
      break;
  }
}

function selectedLanguage() {
  return languageSelect ? languageSelect.value : "it";
}

function selectedSourceLanguage() {
  return sourceLanguageSelect ? sourceLanguageSelect.value : "auto";
}

function selectedMode() {
  return sessionModeSelect ? sessionModeSelect.value : "agent";
}

function selectedTranslationTiming() {
  return translationTimingSelect ? translationTimingSelect.value : "immediate";
}

function selectedEngine() {
  return ttsEngineSelect ? ttsEngineSelect.value : "omnivoice";
}

function updateSessionLabel(options) {
  const language = typeof options === "string" ? options : options.language;
  const mode = typeof options === "string" ? selectedMode() : options.mode;
  const sourceLanguage =
    typeof options === "string" ? selectedSourceLanguage() : options.sourceLanguage;
  const targetLanguage =
    typeof options === "string" ? language : options.targetLanguage || language;
  const timing =
    typeof options === "string" ? selectedTranslationTiming() : options.translationTiming;
  const label = LANGUAGE_LABELS[language] || language || "Italiano";
  const targetLabel = LANGUAGE_LABELS[targetLanguage] || targetLanguage || label;
  const sourceLabel =
    SOURCE_LANGUAGE_LABELS[sourceLanguage] || sourceLanguage || "Auto rilevamento";
  const modeLabel =
    mode === "translator"
      ? `Traduttore ${sourceLabel} -> ${targetLabel}${
          timing === "immediate" ? " · parla subito" : " · fine parlato"
        }`
      : `Agente · ${label}`;
  if (sessionShortId) {
    sessionLabel.textContent = `Sessione ${sessionShortId} · ${modeLabel}`;
  } else {
    sessionLabel.textContent = `Sessione ${modeLabel}`;
  }
}

function configureLanguages(languages, sourceLanguages) {
  if (languages && typeof languages === "object") {
    Object.assign(LANGUAGE_LABELS, languages);
    populateSelect(languageSelect, languages, selectedLanguage());
  }
  if (sourceLanguages && typeof sourceLanguages === "object") {
    Object.assign(SOURCE_LANGUAGE_LABELS, sourceLanguages);
    populateSelect(sourceLanguageSelect, sourceLanguages, selectedSourceLanguage());
  } else if (languages && typeof languages === "object") {
    populateSelect(
      sourceLanguageSelect,
      { auto: SOURCE_LANGUAGE_LABELS.auto, ...languages },
      selectedSourceLanguage(),
    );
  }
}

function populateSelect(select, options, selected) {
  if (!select || !options || typeof options !== "object") {
    return;
  }
  const fallback = selected || select.value;
  select.innerHTML = "";
  for (const [value, label] of Object.entries(options)) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  }
  if ([...select.options].some((option) => option.value === fallback)) {
    select.value = fallback;
  }
}

function configureSessionDefaults(message) {
  if (pendingSessionStart || sessionStarted) {
    if (Number.isFinite(message.translator_immediate_buffer_ms)) {
      translatorImmediateBufferMs = message.translator_immediate_buffer_ms;
    }
    updateModeControls();
    return;
  }
  if (message.mode && sessionModeSelect) {
    sessionModeSelect.value = message.mode;
  }
  if (message.translation_timing && translationTimingSelect) {
    translationTimingSelect.value = message.translation_timing;
  }
  if (message.source_language && sourceLanguageSelect) {
    sourceLanguageSelect.value = message.source_language;
  }
  if (message.target_language && languageSelect) {
    languageSelect.value = message.target_language;
  } else if (message.language && languageSelect) {
    languageSelect.value = message.language;
  }
  if (Number.isFinite(message.translator_immediate_buffer_ms)) {
    translatorImmediateBufferMs = message.translator_immediate_buffer_ms;
  }
  updateModeControls();
}

function updateModeControls() {
  const translator = selectedMode() === "translator";
  sourceLanguageControl.classList.toggle("hidden", !translator);
  translationTimingControl.classList.toggle("hidden", !translator);
  languageSelectLabel.textContent = translator ? "A" : "Lingua agente";
  if (translator && selectedTranslationTiming() === "immediate") {
    audioModeLabel.textContent = `Traduttore: buffer ${translatorImmediateBufferMs} ms`;
  } else if (!sessionStarted) {
    audioModeLabel.textContent = "Audio in attesa";
  }
}

function switchToTranslatorFromLanguageControl() {
  if (sessionStarted || pendingSessionStart || selectedMode() === "translator") {
    updateModeControls();
    return;
  }
  sessionModeSelect.value = "translator";
  updateModeControls();
}

function configureTtsEngines(engines, activeStatus) {
  if (!ttsEngineSelect || !Array.isArray(engines) || engines.length === 0) {
    return;
  }
  const selected = activeStatus && activeStatus.engine ? activeStatus.engine : selectedEngine();
  ttsEngineSelect.innerHTML = "";
  for (const engine of engines) {
    const option = document.createElement("option");
    option.value = engine.id;
    option.textContent = engine.label || ENGINE_LABELS[engine.id] || engine.id;
    ttsEngineSelect.appendChild(option);
  }
  if ([...ttsEngineSelect.options].some((option) => option.value === selected)) {
    ttsEngineSelect.value = selected;
  }
}

function updateEngineStatus(status) {
  if (!status) {
    return;
  }
  const engine = status.engine || selectedEngine();
  const label = ENGINE_LABELS[engine] || engine || "TTS";
  if (ttsEngineSelect && ttsEngineSelect.value !== engine) {
    const exists = [...ttsEngineSelect.options].some((option) => option.value === engine);
    if (exists) {
      ttsEngineSelect.value = engine;
    }
  }

  const state = status.status || "ready";
  const sampleRate = status.sample_rate || status.tts_sample_rate || ttsSampleRate;
  if (state === "ready") {
    voiceModeLabel.textContent = `${label} pronto · ${sampleRate} Hz`;
  } else if (state === "loading") {
    voiceModeLabel.textContent = `${label} in caricamento`;
  } else if (state === "failed") {
    voiceModeLabel.textContent = `${label} errore`;
  } else if (state === "selected") {
    voiceModeLabel.textContent = `${label} selezionato`;
  } else {
    voiceModeLabel.textContent = `${label} · ${state}`;
  }
}

function enqueueAudio(arrayBuffer) {
  if (!playerNode || !audioContext || audioPlaybackBlocked) {
    return;
  }
  assistantPlaybackActive = true;
  const pcm = new Int16Array(arrayBuffer);
  const floats = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i += 1) {
    floats[i] = Math.max(-1, Math.min(1, pcm[i] / 32768));
  }
  const resampled = resampleLinear(floats, ttsSampleRate, audioContext.sampleRate);
  enqueuePlayerSamples(resampled);
}

function configureClientBargeIn(message) {
  if (typeof message.client_barge_enabled === "boolean") {
    clientBargeEnabled = message.client_barge_enabled;
  }
  if (Number.isFinite(message.client_barge_threshold)) {
    clientVad.threshold = message.client_barge_threshold;
  }
  if (Number.isFinite(message.client_barge_stop_ms)) {
    clientVad.stopMs = message.client_barge_stop_ms;
  }
  if (Number.isFinite(message.client_barge_commit_ms)) {
    clientVad.commitMs = message.client_barge_commit_ms;
  }
  if (Number.isFinite(message.client_barge_cooldown_ms)) {
    clientVad.cooldownMs = message.client_barge_cooldown_ms;
  }
}

function clearPlayer() {
  if (playerNode && playerNode.port) {
    playerNode.port.postMessage({ type: "clear" });
  }
  fallbackPlayerQueue = [];
  fallbackPlayerOffset = 0;
  fallbackQueuedSamples = 0;
  playerBufferedMs = 0;
  bufferLabel.textContent = "0 ms";
}

function enqueuePlayerSamples(samples) {
  if (playerNode && playerNode.port) {
    playerNode.port.postMessage(
      { type: "audio", samples: samples.buffer },
      [samples.buffer],
    );
    return;
  }

  fallbackPlayerQueue.push(samples);
  fallbackQueuedSamples += samples.length;
  updateFallbackBufferLabel();
}

function onFallbackPlayerProcess(event) {
  const output = event.outputBuffer.getChannelData(0);
  for (let i = 0; i < output.length; i += 1) {
    if (fallbackPlayerQueue.length === 0) {
      output[i] = 0;
      continue;
    }

    const head = fallbackPlayerQueue[0];
    output[i] = head[fallbackPlayerOffset] || 0;
    fallbackPlayerOffset += 1;
    fallbackQueuedSamples = Math.max(0, fallbackQueuedSamples - 1);

    if (fallbackPlayerOffset >= head.length) {
      fallbackPlayerQueue.shift();
      fallbackPlayerOffset = 0;
    }
  }

  const now = performance.now();
  if (now - fallbackBufferLastAt > 100) {
    fallbackBufferLastAt = now;
    updateFallbackBufferLabel();
  }
}

function updateFallbackBufferLabel() {
  if (!audioContext) {
    updatePlayerBuffer(0);
    return;
  }
  updatePlayerBuffer(
    Math.round((fallbackQueuedSamples / audioContext.sampleRate) * 1000),
  );
}

function resampleLinear(input, fromRate, toRate) {
  if (fromRate === toRate) {
    return input;
  }
  const ratio = fromRate / toRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const pos = i * ratio;
    const left = Math.floor(pos);
    const right = Math.min(input.length - 1, left + 1);
    const frac = pos - left;
    output[i] = input[left] * (1 - frac) + input[right] * frac;
  }
  return output;
}

function prepareAssistantMessage(turnId) {
  if (activeAssistantMessage && activeAssistantMessage.dataset.turnId === String(turnId)) {
    return;
  }
  activeAssistantMessage = addMessage("assistant", "Assistente", "");
  activeAssistantMessage.dataset.turnId = String(turnId);
}

function appendAssistantText(turnId, text) {
  prepareAssistantMessage(turnId);
  const body = activeAssistantMessage.querySelector(".messageText");
  body.textContent += text;
  conversation.scrollTop = conversation.scrollHeight;
}

function addMessage(role, title, text) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.innerHTML = `
    <div class="messageHead">
      <span>${title}</span>
      <span>${new Date().toLocaleTimeString()}</span>
    </div>
    <div class="messageText"></div>
  `;
  node.querySelector(".messageText").textContent = text;
  conversation.appendChild(node);
  conversation.scrollTop = conversation.scrollHeight;
  return node;
}

function addSystemMessage(text) {
  addMessage("system", "Sistema", text);
}

function setState(className, label) {
  document.body.className = `state-${className}`;
  stateLabel.textContent = label;
}

function startTimer() {
  if (timerInterval) {
    return;
  }
  callStartedAt = Date.now();
  updateTimer();
  timerInterval = setInterval(updateTimer, 1000);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  timerLabel.textContent = "00:00";
}

function updateTimer() {
  const elapsed = Math.max(0, Math.floor((Date.now() - callStartedAt) / 1000));
  const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const seconds = String(elapsed % 60).padStart(2, "0");
  timerLabel.textContent = `${minutes}:${seconds}`;
}

function updateServerMeter(rms) {
  if (!Number.isFinite(rms)) {
    return;
  }
  const level = Math.min(100, Math.round((rms / 0.08) * 100));
  meterFill.style.width = `${level}%`;
}
