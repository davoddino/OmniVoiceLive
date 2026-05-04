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
let bargeSent = false;
let currentTurnId = null;
let socketWasOpen = false;
let socketOpenTimer = null;
let cleaningUp = false;
let callStartedAt = 0;
let timerInterval = null;

const clientVad = {
  threshold: 0.018,
  speechMs: 0,
  lastBargeAt: 0,
};

startButton.addEventListener("click", startCall);
stopButton.addEventListener("click", stopCall);

async function startCall() {
  try {
    cleaningUp = false;
    setState("connecting", "Connessione");
    startButton.disabled = true;
    transportLabel.textContent =
      location.protocol === "https:" ? "HTTPS/WSS" : "HTTP/WS";
    connectionLabel.textContent = "Avvio";
    voiceModeLabel.textContent = "Stabilizzata per turno";

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
  audioContext = null;
  mediaStream = null;
  recorderNode = null;
  recorderSinkNode = null;
  playerNode = null;
  fallbackPlayerQueue = [];
  fallbackPlayerOffset = 0;
  fallbackQueuedSamples = 0;
  assistantSpeaking = false;
  bargeSent = false;
  stopTimer();
  startButton.disabled = false;
  stopButton.disabled = true;
  sessionLabel.textContent = label;
  connectionLabel.textContent = "Offline";
  audioModeLabel.textContent = "Audio in attesa";
  setState("idle", "Idle");
  setTimeout(() => {
    cleaningUp = false;
  }, 0);
}

function onMicFrame(event) {
  const frame = event.data;
  updateClientVad(frame);
  if (!socket || socket.readyState !== WebSocket.OPEN) {
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
      bufferLabel.textContent = `${event.data.ms} ms`;
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
  socket.binaryType = "arraybuffer";

  socketOpenTimer = setTimeout(() => {
    if (socket && socket.readyState !== WebSocket.OPEN) {
      addSystemMessage(
        `INFO: WebSocket ancora non aperto dopo 10s, readyState=${readyStateLabel(socket.readyState)}.`,
      );
      addSystemMessage(`INFO: Test diagnostico: ${location.origin}/ws-test`);
      setState("error", "Errore");
      socket.close();
    }
  }, 10000);

  socket.onopen = () => {
    socketWasOpen = true;
    if (socketOpenTimer) {
      clearTimeout(socketOpenTimer);
      socketOpenTimer = null;
    }
    addSystemMessage("INFO: WebSocket aperto.");
    connectionLabel.textContent = "Online";
    socket.send(
      JSON.stringify({
        type: "session.start",
        sample_rate: audioContext ? audioContext.sampleRate : 48000,
      }),
    );
    setState("listening", "Ascolto");
    startTimer();
    stopButton.disabled = false;
  };

  socket.onmessage = onSocketMessage;
  socket.onclose = (event) => {
    if (socketOpenTimer) {
      clearTimeout(socketOpenTimer);
      socketOpenTimer = null;
    }
    if (cleaningUp) {
      return;
    }
    const reason = event.reason ? ` reason=${event.reason}` : "";
    if (socketWasOpen) {
      addSystemMessage(`INFO: WebSocket chiuso code=${event.code}${reason}.`);
      connectionLabel.textContent = "Offline";
      cleanup("Chiamata terminata");
    } else {
      cleanup("WebSocket non aperto");
      setState("error", "Errore");
      addSystemMessage(`INFO: WebSocket non aperto code=${event.code}${reason}.`);
      addSystemMessage(`INFO: Apri ${location.origin}/ws-test e premi Test WebSocket.`);
    }
  };

  socket.onerror = () => {
    addSystemMessage(
      "INFO: Errore WebSocket. Controlla trust certificato iOS, IP, rete e firewall.",
    );
    connectionLabel.textContent = "Errore";
    setState("error", "Errore");
  };
}

function readyStateLabel(value) {
  switch (value) {
    case WebSocket.CONNECTING:
      return "CONNECTING";
    case WebSocket.OPEN:
      return "OPEN";
    case WebSocket.CLOSING:
      return "CLOSING";
    case WebSocket.CLOSED:
      return "CLOSED";
    default:
      return String(value);
  }
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
  if (
    assistantSpeaking &&
    clientVad.speechMs >= 80 &&
    !bargeSent &&
    now - clientVad.lastBargeAt > 800
  ) {
    bargeSent = true;
    clientVad.lastBargeAt = now;
    clearPlayer();
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "barge_in" }));
    }
    setState("interrupted", "Interrotto");
  }
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
      sessionLabel.textContent = `Sessione ${message.session_id.slice(0, 8)}`;
      break;
    case "session.started":
      connectionLabel.textContent = "Online";
      setState("listening", "Ascolto");
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
      assistantSpeaking = true;
      bargeSent = false;
      connectionLabel.textContent = "TTS";
      setState("speaking", "Rispondo");
      break;
    case "assistant.audio_ready":
      ttsLatency.textContent = `${message.latency_ms} ms`;
      break;
    case "assistant.done":
      assistantSpeaking = false;
      bargeSent = false;
      connectionLabel.textContent = "Online";
      setState("listening", "Ascolto");
      break;
    case "turn.cancelled":
      assistantSpeaking = false;
      clearPlayer();
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

function enqueueAudio(arrayBuffer) {
  if (!playerNode || !audioContext) {
    return;
  }
  const pcm = new Int16Array(arrayBuffer);
  const floats = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i += 1) {
    floats[i] = Math.max(-1, Math.min(1, pcm[i] / 32768));
  }
  const resampled = resampleLinear(floats, ttsSampleRate, audioContext.sampleRate);
  enqueuePlayerSamples(resampled);
}

function clearPlayer() {
  if (playerNode && playerNode.port) {
    playerNode.port.postMessage({ type: "clear" });
  }
  fallbackPlayerQueue = [];
  fallbackPlayerOffset = 0;
  fallbackQueuedSamples = 0;
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
    bufferLabel.textContent = "0 ms";
    return;
  }
  bufferLabel.textContent = `${Math.round(
    (fallbackQueuedSamples / audioContext.sampleRate) * 1000,
  )} ms`;
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
