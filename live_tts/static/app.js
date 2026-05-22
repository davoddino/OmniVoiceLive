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
const listenerCountLabel = document.getElementById("listenerCountLabel");
const timelineTitle = document.getElementById("timelineTitle");
const roomBadge = document.getElementById("roomBadge");

const roomTabs = [...document.querySelectorAll(".roomTab")];
const assistantRoom = document.getElementById("assistantRoom");
const translatorRoom = document.getElementById("translatorRoom");
const eventRoom = document.getElementById("eventRoom");
const roomPanels = {
  assistant: assistantRoom,
  translator: translatorRoom,
  event: eventRoom,
};

const languageSelect = document.getElementById("languageSelect");
const assistantPromptModeSelect = document.getElementById("assistantPromptModeSelect");
const customPromptWrap = document.getElementById("customPromptWrap");
const customPromptInput = document.getElementById("customPromptInput");
const ttsEngineSelect = document.getElementById("ttsEngineSelect");
const sourceLanguageSelect = document.getElementById("sourceLanguageSelect");
const translatorTargetLanguageSelect = document.getElementById(
  "translatorTargetLanguageSelect",
);
const translationTimingSelect = document.getElementById("translationTimingSelect");
const translatorTtsEngineSelect = document.getElementById("translatorTtsEngineSelect");

const eventSpeakerRoleButton = document.getElementById("eventSpeakerRoleButton");
const eventListenerRoleButton = document.getElementById("eventListenerRoleButton");
const eventSpeakerPanel = document.getElementById("eventSpeakerPanel");
const eventListenerPanel = document.getElementById("eventListenerPanel");
const eventSourceLanguageSelect = document.getElementById("eventSourceLanguageSelect");
const eventTargetLanguageSelect = document.getElementById("eventTargetLanguageSelect");
const eventTtsEngineSelect = document.getElementById("eventTtsEngineSelect");
const eventCodeInput = document.getElementById("eventCodeInput");
const eventCodeDisplay = document.getElementById("eventCodeDisplay");

const targetLanguageSelects = [
  languageSelect,
  translatorTargetLanguageSelect,
  eventTargetLanguageSelect,
];
const sourceLanguageSelects = [sourceLanguageSelect, eventSourceLanguageSelect];
const engineSelects = [ttsEngineSelect, translatorTtsEngineSelect, eventTtsEngineSelect];

const LANGUAGE_LABELS = {
  it: "Italian",
  en: "English",
  de: "German",
  fr: "French",
  es: "Spanish",
  pt: "Portuguese",
  ro: "Romanian",
  sq: "Albanian",
  ru: "Russian",
  uk: "Ukrainian",
  pl: "Polish",
  sr: "Serbian/Croatian/Bosnian",
  hr: "Croatian",
  bs: "Bosnian",
  ar: "Arabic",
  zh: "Simplified Chinese",
  hi: "Hindi",
  ur: "Urdu",
  sw: "Swahili",
};

const SOURCE_LANGUAGE_LABELS = {
  auto: "Auto-detect",
  ...LANGUAGE_LABELS,
};

const ENGINE_LABELS = {
  omnivoice: "OmniVoice",
  qwen3_tts: "Qwen3-TTS",
};

const ROOM_TITLES = {
  assistant: "Virtual Assistant",
  translator: "Instant Translator",
  event: "Event Interpreter",
};

let socket = null;
let activeProtocol = "";
let activeRoom = "assistant";
let activeEventRole = "speaker";
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
let activeEventMessages = new Map();
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
assistantPromptModeSelect.addEventListener("change", updateRoomControls);
translationTimingSelect.addEventListener("change", updateRoomControls);
languageSelect.addEventListener("change", updateRoomControls);
translatorTargetLanguageSelect.addEventListener("change", updateRoomControls);
sourceLanguageSelect.addEventListener("change", updateRoomControls);
eventSourceLanguageSelect.addEventListener("change", updateRoomControls);
eventTargetLanguageSelect.addEventListener("change", onEventTargetLanguageChange);
eventCodeInput.addEventListener("input", () => {
  eventCodeInput.value = normalizeEventCode(eventCodeInput.value);
});

for (const tab of roomTabs) {
  tab.addEventListener("click", () => switchRoom(tab.dataset.room));
}

for (const button of [eventSpeakerRoleButton, eventListenerRoleButton]) {
  button.addEventListener("click", () => switchEventRole(button.dataset.role));
}

for (const select of engineSelects) {
  select.addEventListener("change", () => {
    updateEngineStatus({
      engine: selectedEngine(),
      status: socket && socket.readyState === WebSocket.OPEN ? "loading" : "selected",
      sample_rate: ttsSampleRate,
    });
    if (
      activeProtocol !== "realtime" ||
      !socket ||
      socket.readyState !== WebSocket.OPEN
    ) {
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
  });
}

updateRoomControls();

async function startCall() {
  if (activeRoom === "event" && activeEventRole === "listener") {
    await startEventListener();
    return;
  }
  if (activeRoom === "event") {
    await startEventSpeaker();
    return;
  }
  await startRealtimeCall();
}

async function startRealtimeCall() {
  try {
    beginStartup("Connecting");
    activeProtocol = "realtime";
    activeSessionMode = activeRoom === "translator" ? "translator" : "agent";
    clientBargeEnabled = activeSessionMode !== "translator";
    await setupInputAudioGraph();
    connectRealtimeWebSocket();
    resumeAudioContextWithTimeout();
  } catch (error) {
    cleanup("Start failed");
    setState("error", "Error");
    addSystemMessage(error.message || "Unable to start audio and microphone.");
  }
}

async function startEventSpeaker() {
  try {
    beginStartup("Starting event");
    activeProtocol = "event_speaker";
    clientBargeEnabled = false;
    await setupInputAudioGraph();
    connectEventSpeakerWebSocket();
    resumeAudioContextWithTimeout();
  } catch (error) {
    cleanup("Event start failed");
    setState("error", "Error");
    addSystemMessage(error.message || "Unable to start the event speaker.");
  }
}

async function startEventListener() {
  const code = normalizeEventCode(eventCodeInput.value);
  if (!code) {
    addSystemMessage("Enter an event code before joining.");
    setState("error", "Error");
    return;
  }
  try {
    beginStartup("Joining event");
    activeProtocol = "event_listener";
    clientBargeEnabled = false;
    await setupPlaybackOnlyGraph();
    connectEventListenerWebSocket(code);
    resumeAudioContextWithTimeout();
  } catch (error) {
    cleanup("Join failed");
    setState("error", "Error");
    addSystemMessage(error.message || "Unable to join the event.");
  }
}

function beginStartup(label) {
  cleaningUp = false;
  setState("connecting", label);
  setControlsDisabled(true);
  startButton.disabled = true;
  stopButton.disabled = true;
  transportLabel.textContent =
    location.protocol === "https:" ? "HTTPS/WSS" : "HTTP/WS";
  connectionLabel.textContent = "Starting";
  listenerCountLabel.textContent = "0";
  updateEngineStatus({
    engine: selectedEngine(),
    status: "connecting",
    sample_rate: ttsSampleRate,
  });
}

function stopCall() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    if (activeProtocol === "event_speaker") {
      socket.send(JSON.stringify({ type: "event.host.stop" }));
    } else if (activeProtocol === "event_listener") {
      socket.send(JSON.stringify({ type: "event.listener.leave" }));
    } else {
      socket.send(JSON.stringify({ type: "session.stop" }));
    }
  }
  cleanup("Session ended");
}

function cleanup(label) {
  cleaningUp = true;
  if (socketOpenTimer) {
    clearTimeout(socketOpenTimer);
    socketOpenTimer = null;
  }
  clearPlayer();
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
  activeProtocol = "";
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
  playerBufferedMs = 0;
  assistantSpeaking = false;
  assistantPlaybackActive = false;
  assistantDonePending = false;
  bargeSent = false;
  audioPlaybackBlocked = false;
  activeAudioTurnId = null;
  blockedAudioTurnId = null;
  activeAssistantMessage = null;
  activeEventMessages = new Map();
  resetClientVad();
  stopTimer();
  setControlsDisabled(false);
  startButton.disabled = false;
  stopButton.disabled = true;
  sessionLabel.textContent = label;
  connectionLabel.textContent = "Offline";
  audioModeLabel.textContent = "Audio idle";
  eventCodeDisplay.textContent = activeEventRole === "speaker" ? "Not hosted" : "-";
  listenerCountLabel.textContent = "0";
  updateRoomControls();
  setState("idle", "Idle");
  setTimeout(() => {
    cleaningUp = false;
  }, 0);
}

async function setupInputAudioGraph() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("Web Audio API is not available in this browser.");
  }
  audioContext = new AudioContextClass({ latencyHint: "interactive" });
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });
  addSystemMessage("Microphone ready.");
  const source = audioContext.createMediaStreamSource(mediaStream);
  const canUseWorklet = Boolean(audioContext.audioWorklet && window.AudioWorkletNode);
  if (canUseWorklet) {
    try {
      await setupAudioWorkletGraph(source);
      audioModeLabel.textContent = "AudioWorklet";
      return;
    } catch (error) {
      addSystemMessage(`Audio fallback enabled: ${error.message}`);
    }
  }
  setupScriptProcessorGraph(source);
  audioModeLabel.textContent = "Fallback audio";
}

async function setupPlaybackOnlyGraph() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("Web Audio API is not available in this browser.");
  }
  audioContext = new AudioContextClass({ latencyHint: "interactive" });
  const canUseWorklet = Boolean(audioContext.audioWorklet && window.AudioWorkletNode);
  if (canUseWorklet) {
    try {
      await audioContext.audioWorklet.addModule("/static/player-worklet.js?v=2");
      playerNode = new AudioWorkletNode(audioContext, "player-worklet");
      playerNode.connect(audioContext.destination);
      playerNode.port.onmessage = (event) => {
        if (event.data.type === "buffer") {
          updatePlayerBuffer(event.data.ms);
        }
      };
      audioModeLabel.textContent = "Playback ready";
      return;
    } catch (error) {
      addSystemMessage(`Playback fallback enabled: ${error.message}`);
    }
  }
  playerNode = audioContext.createScriptProcessor(1024, 0, 1);
  playerNode.onaudioprocess = onFallbackPlayerProcess;
  playerNode.connect(audioContext.destination);
  audioModeLabel.textContent = "Fallback playback";
}

async function setupAudioWorkletGraph(source) {
  await audioContext.audioWorklet.addModule("/static/recorder-worklet.js?v=2");
  await audioContext.audioWorklet.addModule("/static/player-worklet.js?v=2");

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

function connectRealtimeWebSocket() {
  const ws = openSocket("/ws", "Connecting");
  pendingSessionStart = false;
  ws.onopen = () => {
    if (socket !== ws) {
      return;
    }
    markSocketOpen("Online");
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
  ws.onmessage = onRealtimeSocketMessage;
  bindSocketCloseHandlers(ws);
}

function connectEventSpeakerWebSocket() {
  const ws = openSocket("/ws/event/speaker", "Starting event");
  audioStreamingPaused = true;
  ws.onopen = () => {
    if (socket !== ws) {
      return;
    }
    markSocketOpen("Online");
    ws.send(
      JSON.stringify({
        type: "event.host.start",
        sample_rate: audioContext ? audioContext.sampleRate : 48000,
        source_language: eventSourceLanguageSelect.value,
        tts_engine: selectedEngine(),
      }),
    );
    startTimer();
  };
  ws.onmessage = onEventSocketMessage;
  bindSocketCloseHandlers(ws);
}

function connectEventListenerWebSocket(code) {
  const ws = openSocket("/ws/event/listener", "Joining event");
  ws.onopen = () => {
    if (socket !== ws) {
      return;
    }
    markSocketOpen("Online");
    ws.send(
      JSON.stringify({
        type: "event.listener.join",
        event_code: code,
        target_language: eventTargetLanguageSelect.value,
      }),
    );
    startTimer();
  };
  ws.onmessage = onEventSocketMessage;
  bindSocketCloseHandlers(ws);
}

function openSocket(path, label) {
  const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${wsProto}//${location.host}${path}`;
  addSystemMessage("Connecting to server.");
  connectionLabel.textContent = label;
  socketWasOpen = false;
  socket = new WebSocket(wsUrl);
  const ws = socket;
  ws.binaryType = "arraybuffer";
  socketOpenTimer = setTimeout(() => {
    if (socket === ws && ws.readyState !== WebSocket.OPEN) {
      addSystemMessage(`WebSocket did not open after 5s, readyState=${ws.readyState}.`);
      setState("error", "Error");
      ws.close();
    }
  }, 5000);
  return ws;
}

function markSocketOpen(label) {
  socketWasOpen = true;
  if (socketOpenTimer) {
    clearTimeout(socketOpenTimer);
    socketOpenTimer = null;
  }
  connectionLabel.textContent = label;
}

function bindSocketCloseHandlers(ws) {
  let wsOpened = false;
  ws.addEventListener("open", () => {
    wsOpened = true;
  });
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
      addSystemMessage(`WebSocket closed code=${event.code}${reason}.`);
      connectionLabel.textContent = "Offline";
      cleanup("Session ended");
    } else {
      cleanup("WebSocket did not open");
      setState("error", "Error");
      addSystemMessage(`WebSocket did not open code=${event.code}${reason}.`);
    }
  };
  ws.onerror = () => {
    if (socket !== ws) {
      return;
    }
    addSystemMessage("WebSocket error. Check certificate, host, and firewall.");
    connectionLabel.textContent = "Error";
    setState("error", "Error");
  };
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
  if (durationMs > 0 && frameCount > 0) {
    audioModeLabel.textContent = "Microphone ready";
  }
}

function sendSessionStart() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  pendingSessionStart = false;
  activeSessionMode = activeRoom === "translator" ? "translator" : "agent";
  activeTranslationTiming = translationTimingSelect.value;
  const payload = {
    type: "session.start",
    sample_rate: audioContext ? audioContext.sampleRate : 48000,
    mode: activeSessionMode,
    language: selectedLanguage(),
    source_language: sourceLanguageSelect.value,
    target_language: selectedLanguage(),
    translation_timing: activeTranslationTiming,
    tts_engine: selectedEngine(),
  };
  if (activeSessionMode === "agent") {
    payload.agent_prompt_mode = assistantPromptModeSelect.value;
    if (assistantPromptModeSelect.value === "custom") {
      payload.agent_prompt = customPromptInput.value;
    }
  }
  const modeLabel =
    activeSessionMode === "translator" ? "Instant Translator" : "Virtual Assistant";
  addSystemMessage(`Starting ${modeLabel}.`);
  socket.send(JSON.stringify(payload));
}

async function resumeAudioContextWithTimeout(timeoutMs = 1200) {
  if (!audioContext) {
    return;
  }
  try {
    await Promise.race([
      audioContext.resume(),
      new Promise((resolve) => {
        setTimeout(resolve, timeoutMs);
      }),
    ]);
  } catch (error) {
    addSystemMessage(`Audio resume failed: ${error.message}.`);
  }
}

function onRealtimeSocketMessage(event) {
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
      updateSessionLabel();
      break;
    case "session.started":
      sessionStarted = true;
      audioStreamingPaused = false;
      configureClientBargeIn(message);
      activeSessionMode = message.mode || selectedMode();
      activeTranslationTiming = message.translation_timing || translationTimingSelect.value;
      if (message.tts_sample_rate) {
        ttsSampleRate = message.tts_sample_rate;
      }
      updateEngineStatus(message.tts_engine);
      updateSessionLabel(message);
      connectionLabel.textContent = "Online";
      stopButton.disabled = false;
      flushPendingMicFrames();
      setState("listening", "Listening");
      break;
    case "tts.engine.loading":
      audioStreamingPaused = true;
      setEngineControlsDisabled(true);
      clearPlayer();
      updateEngineStatus({
        engine: message.engine,
        status: "loading",
        sample_rate: ttsSampleRate,
      });
      connectionLabel.textContent = "Loading TTS";
      setState("connecting", "Loading voice");
      break;
    case "tts.engine.ready":
      audioStreamingPaused = false;
      if (message.tts_sample_rate || message.sample_rate) {
        ttsSampleRate = message.tts_sample_rate || message.sample_rate;
      }
      configureTtsEngines(message.tts_engines, message);
      updateEngineStatus(message);
      setEngineControlsDisabled(false);
      if (sessionStarted) {
        flushPendingMicFrames();
        connectionLabel.textContent = "Online";
        setState("listening", "Listening");
      } else if (pendingSessionStart) {
        sendSessionStart();
      }
      break;
    case "tts.engine.error":
      audioStreamingPaused = false;
      setEngineControlsDisabled(false);
      configureTtsEngines(message.tts_engines, {
        engine: message.engine,
        status: "failed",
      });
      updateEngineStatus({
        engine: message.engine,
        status: "failed",
        error: message.message,
      });
      addSystemMessage(message.message || "TTS engine is not available.");
      connectionLabel.textContent = "TTS Error";
      setState("error", "Error");
      if (!sessionStarted) {
        startButton.disabled = false;
        stopButton.disabled = true;
      }
      break;
    case "audio.meter":
      updateServerMeter(message.rms);
      break;
    case "vad.speech_start":
      connectionLabel.textContent = "User";
      setState("user-speaking", "User speaking");
      break;
    case "vad.speech_end":
      connectionLabel.textContent = "STT";
      setState("transcribing", "Transcribing");
      break;
    case "stt.final":
      currentTurnId = message.turn_id;
      turnLabel.textContent = String(message.turn_id);
      sttLatency.textContent = `${message.latency_ms} ms`;
      if (message.text) {
        addMessage(
          activeSessionMode === "translator" ? "source" : "user",
          activeSessionMode === "translator" ? "Source" : "You",
          message.text,
        );
      }
      connectionLabel.textContent = "LLM";
      setState("thinking", "Thinking");
      break;
    case "assistant.thinking":
      prepareAssistantMessage(message.turn_id);
      setState("thinking", "Thinking");
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
      setState("speaking", activeSessionMode === "translator" ? "Speaking" : "Replying");
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
      connectionLabel.textContent = "Interrupted";
      setState("interrupted", "Interrupted");
      break;
    case "turn.empty":
      connectionLabel.textContent = "Online";
      setState("listening", "Listening");
      break;
    case "error":
      addSystemMessage(message.message || "Unspecified error.");
      connectionLabel.textContent = "Error";
      setState("error", "Error");
      break;
    default:
      break;
  }
}

function onEventSocketMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    enqueueAudio(event.data);
    return;
  }
  const message = JSON.parse(event.data);
  switch (message.type) {
    case "event.speaker.ready":
    case "event.listener.ready":
      configureLanguages(message.languages, message.source_languages);
      configureTtsEngines(message.tts_engines, message.tts_engine);
      if (message.tts_sample_rate) {
        ttsSampleRate = message.tts_sample_rate;
      }
      updateEngineStatus(message.tts_engine || { engine: selectedEngine(), status: "ready" });
      break;
    case "tts.engine.loading":
      audioStreamingPaused = true;
      setEngineControlsDisabled(true);
      updateEngineStatus({
        engine: message.engine,
        status: "loading",
        sample_rate: ttsSampleRate,
      });
      connectionLabel.textContent = "Loading TTS";
      setState("connecting", "Loading voice");
      break;
    case "tts.engine.ready":
      if (message.tts_sample_rate || message.sample_rate) {
        ttsSampleRate = message.tts_sample_rate || message.sample_rate;
      }
      configureTtsEngines(message.tts_engines, message);
      updateEngineStatus(message);
      setEngineControlsDisabled(false);
      break;
    case "tts.engine.error":
      addSystemMessage(message.message || "TTS engine is not available.");
      updateEngineStatus({ engine: message.engine, status: "failed" });
      setState("error", "Error");
      break;
    case "event.host.started":
      sessionStarted = true;
      audioStreamingPaused = false;
      sessionShortId = message.event_code;
      eventCodeDisplay.textContent = message.event_code;
      listenerCountLabel.textContent = String(message.listener_count || 0);
      if (message.tts_sample_rate) {
        ttsSampleRate = message.tts_sample_rate;
      }
      updateEngineStatus(message.tts_engine);
      updateSessionLabel(message);
      setEngineControlsDisabled(true);
      stopButton.disabled = false;
      flushPendingMicFrames();
      connectionLabel.textContent = "Hosting";
      setState("listening", "Ready");
      break;
    case "event.listener.joined":
      sessionStarted = true;
      audioStreamingPaused = false;
      sessionShortId = message.event_code;
      eventCodeDisplay.textContent = message.event_code;
      listenerCountLabel.textContent = String(message.listener_count || 1);
      if (message.tts_sample_rate) {
        ttsSampleRate = message.tts_sample_rate;
      }
      updateSessionLabel(message);
      setEngineControlsDisabled(true);
      stopButton.disabled = false;
      connectionLabel.textContent = "Listening";
      setState("listening", "Listening");
      audioModeLabel.textContent = `Listening in ${
        LANGUAGE_LABELS[message.target_language] || message.target_language
      }`;
      break;
    case "event.listener.updated":
      audioModeLabel.textContent = `Listening in ${
        LANGUAGE_LABELS[message.target_language] || message.target_language
      }`;
      break;
    case "event.listener_count":
      listenerCountLabel.textContent = String(message.listener_count || 0);
      break;
    case "audio.meter":
      updateServerMeter(message.rms);
      break;
    case "event.speech_start":
      connectionLabel.textContent = activeEventRole === "speaker" ? "Speaker" : "Live";
      setState("user-speaking", activeEventRole === "speaker" ? "Speaking" : "Live");
      break;
    case "event.speech_end":
      connectionLabel.textContent = "STT";
      setState("transcribing", "Transcribing");
      break;
    case "event.turn.started":
      currentTurnId = message.turn_id;
      turnLabel.textContent = String(message.turn_id);
      break;
    case "event.source_text":
      currentTurnId = message.turn_id;
      turnLabel.textContent = String(message.turn_id);
      if (Number.isFinite(message.latency_ms)) {
        sttLatency.textContent = `${message.latency_ms} ms`;
      }
      if (message.text) {
        addMessage("source", "Speaker", message.text);
      }
      connectionLabel.textContent = "Translating";
      setState("thinking", "Translating");
      break;
    case "event.translation.started":
      prepareEventTranslationMessage(message.turn_id, message.target_language);
      break;
    case "event.translation_delta":
      appendEventTranslationText(
        message.turn_id,
        message.target_language,
        message.text || "",
      );
      break;
    case "event.translation_final":
      finishEventTranslation(message);
      break;
    case "event.audio_start":
      activeAudioTurnId = message.turn_id;
      assistantSpeaking = true;
      assistantPlaybackActive = false;
      assistantDonePending = false;
      connectionLabel.textContent = "Audio";
      setState("speaking", "Playing");
      break;
    case "event.audio_ready":
      if (Number.isFinite(message.latency_ms)) {
        ttsLatency.textContent = `${message.latency_ms} ms`;
      }
      break;
    case "event.audio_end":
      assistantDonePending = true;
      if (!assistantPlaybackActive && playerBufferedMs === 0) {
        finishAssistantPlayback();
      }
      break;
    case "event.first_audio":
      if (Number.isFinite(message.latency_ms)) {
        ttsLatency.textContent = `${message.latency_ms} ms`;
      }
      break;
    case "event.turn.done":
      if (Number.isFinite(message.latency_ms)) {
        connectionLabel.textContent = "Online";
      }
      setState("listening", activeEventRole === "speaker" ? "Ready" : "Listening");
      break;
    case "event.no_listeners":
      addSystemMessage(message.message || "No listeners are connected.");
      connectionLabel.textContent = "Waiting";
      setState("listening", "Ready");
      break;
    case "event.turn.empty":
      connectionLabel.textContent = "Online";
      setState("listening", activeEventRole === "speaker" ? "Ready" : "Listening");
      break;
    case "event.ended":
      addSystemMessage(`Event ended: ${message.reason || "closed"}.`);
      cleanup("Event ended");
      break;
    case "error":
      addSystemMessage(message.message || "Unspecified error.");
      connectionLabel.textContent = "Error";
      setState("error", "Error");
      break;
    default:
      break;
  }
}

function updateClientVad(frame) {
  if (!audioContext || !frame) {
    return;
  }
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
    setState("interrupted", "Interrupted");
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
  addSystemMessage("Playback interrupted.");
  connectionLabel.textContent = "Interrupted";
  setState("interrupted", "Interrupted");
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
  if (activeProtocol === "event_listener") {
    connectionLabel.textContent = "Listening";
    setState("listening", "Listening");
  } else if (activeProtocol === "event_speaker") {
    connectionLabel.textContent = "Hosting";
    setState("listening", "Ready");
  } else {
    connectionLabel.textContent = "Online";
    setState("listening", "Listening");
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
  activeAssistantMessage = addMessage(
    "assistant",
    activeSessionMode === "translator" ? "Translation" : "Assistant",
    "",
  );
  activeAssistantMessage.dataset.turnId = String(turnId);
}

function appendAssistantText(turnId, text) {
  prepareAssistantMessage(turnId);
  const body = activeAssistantMessage.querySelector(".messageText");
  body.textContent += text;
  conversation.scrollTop = conversation.scrollHeight;
}

function eventMessageKey(turnId, targetLanguage) {
  return `${turnId || "turn"}:${targetLanguage || "target"}`;
}

function prepareEventTranslationMessage(turnId, targetLanguage) {
  const key = eventMessageKey(turnId, targetLanguage);
  if (activeEventMessages.has(key)) {
    return activeEventMessages.get(key);
  }
  const label = LANGUAGE_LABELS[targetLanguage] || targetLanguage || "Target";
  const node = addMessage("assistant", `Translation · ${label}`, "");
  node.dataset.turnId = String(turnId || "");
  node.dataset.targetLanguage = targetLanguage || "";
  activeEventMessages.set(key, node);
  return node;
}

function appendEventTranslationText(turnId, targetLanguage, text) {
  const node = prepareEventTranslationMessage(turnId, targetLanguage);
  node.querySelector(".messageText").textContent += text;
  conversation.scrollTop = conversation.scrollHeight;
}

function finishEventTranslation(message) {
  const key = eventMessageKey(message.turn_id, message.target_language);
  const node = activeEventMessages.get(key);
  if (node) {
    const body = node.querySelector(".messageText");
    if (!body.textContent.trim()) {
      body.textContent = message.text || "";
    }
    return;
  }
  const label = LANGUAGE_LABELS[message.target_language] || message.target_language;
  addMessage("assistant", `Translation · ${label}`, message.text || "");
}

function addMessage(role, title, text) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.innerHTML = `
    <div class="messageHead">
      <span></span>
      <span></span>
    </div>
    <div class="messageText"></div>
  `;
  const head = node.querySelectorAll(".messageHead span");
  head[0].textContent = title;
  head[1].textContent = new Date().toLocaleTimeString();
  node.querySelector(".messageText").textContent = text;
  conversation.appendChild(node);
  conversation.scrollTop = conversation.scrollHeight;
  return node;
}

function addSystemMessage(text) {
  addMessage("system", "System", text);
}

function switchRoom(room) {
  if (!room || room === activeRoom || sessionStarted || pendingSessionStart) {
    return;
  }
  activeRoom = room;
  for (const tab of roomTabs) {
    tab.classList.toggle("active", tab.dataset.room === room);
  }
  for (const [name, panel] of Object.entries(roomPanels)) {
    panel.classList.toggle("active", name === room);
  }
  timelineTitle.textContent = ROOM_TITLES[room];
  roomBadge.textContent = room === "event" ? "Event" : room === "translator" ? "Translator" : "Assistant";
  activeAssistantMessage = null;
  activeEventMessages = new Map();
  updateRoomControls();
  updateSessionLabel();
}

function switchEventRole(role) {
  if (!role || role === activeEventRole || sessionStarted || pendingSessionStart) {
    return;
  }
  activeEventRole = role;
  eventSpeakerRoleButton.classList.toggle("active", role === "speaker");
  eventListenerRoleButton.classList.toggle("active", role === "listener");
  eventSpeakerPanel.classList.toggle("active", role === "speaker");
  eventListenerPanel.classList.toggle("active", role === "listener");
  eventCodeDisplay.textContent = role === "speaker" ? "Not hosted" : "-";
  updateRoomControls();
}

function updateRoomControls() {
  const promptCustom = assistantPromptModeSelect.value === "custom";
  customPromptWrap.classList.toggle("hidden", !promptCustom);
  if (activeRoom === "assistant") {
    startButton.textContent = "Start assistant";
    audioModeLabel.textContent = sessionStarted ? audioModeLabel.textContent : "Audio idle";
  } else if (activeRoom === "translator") {
    startButton.textContent = "Start translator";
    if (!sessionStarted && translationTimingSelect.value === "immediate") {
      audioModeLabel.textContent = `Translator buffer ${translatorImmediateBufferMs} ms`;
    }
  } else if (activeEventRole === "speaker") {
    startButton.textContent = "Host event";
    if (!sessionStarted) {
      audioModeLabel.textContent = "Speaker audio idle";
    }
  } else {
    startButton.textContent = "Join event";
    if (!sessionStarted) {
      audioModeLabel.textContent = "Listener audio idle";
    }
  }
  updateSessionLabel();
}

function setControlsDisabled(disabled) {
  for (const tab of roomTabs) {
    tab.disabled = disabled;
  }
  for (const button of [eventSpeakerRoleButton, eventListenerRoleButton]) {
    button.disabled = disabled;
  }
  for (const control of [
    languageSelect,
    assistantPromptModeSelect,
    customPromptInput,
    ttsEngineSelect,
    sourceLanguageSelect,
    translatorTargetLanguageSelect,
    translationTimingSelect,
    translatorTtsEngineSelect,
    eventSourceLanguageSelect,
    eventTargetLanguageSelect,
    eventTtsEngineSelect,
    eventCodeInput,
  ]) {
    control.disabled = disabled;
  }
}

function setEngineControlsDisabled(disabled) {
  const shouldDisable = disabled || (sessionStarted && activeRoom === "event");
  for (const select of engineSelects) {
    select.disabled = shouldDisable;
  }
}

function selectedMode() {
  return activeRoom === "translator" ? "translator" : "agent";
}

function selectedLanguage() {
  return activeRoom === "translator"
    ? translatorTargetLanguageSelect.value
    : languageSelect.value;
}

function selectedEngine() {
  if (activeRoom === "translator") {
    return translatorTtsEngineSelect.value;
  }
  if (activeRoom === "event") {
    return eventTtsEngineSelect.value;
  }
  return ttsEngineSelect.value;
}

function updateSessionLabel(message = {}) {
  if (activeRoom === "event") {
    const code = message.event_code || sessionShortId;
    if (activeEventRole === "speaker") {
      sessionLabel.textContent = code
        ? `Event ${code} · Speaker`
        : "Event speaker setup";
    } else {
      const language = message.target_language || eventTargetLanguageSelect.value;
      sessionLabel.textContent = code
        ? `Event ${code} · Listening in ${LANGUAGE_LABELS[language] || language}`
        : "Event listener setup";
    }
    return;
  }

  const language = message.language || selectedLanguage();
  const targetLanguage = message.target_language || selectedLanguage();
  const sourceLanguage = message.source_language || sourceLanguageSelect.value;
  const timing = message.translation_timing || translationTimingSelect.value;
  if (activeRoom === "translator") {
    const sourceLabel = SOURCE_LANGUAGE_LABELS[sourceLanguage] || sourceLanguage;
    const targetLabel = LANGUAGE_LABELS[targetLanguage] || targetLanguage;
    const timingLabel = timing === "immediate" ? "speak immediately" : "end of speech";
    sessionLabel.textContent = sessionShortId
      ? `Session ${sessionShortId} · ${sourceLabel} to ${targetLabel} · ${timingLabel}`
      : `${sourceLabel} to ${targetLabel} · ${timingLabel}`;
    return;
  }
  const label = LANGUAGE_LABELS[language] || language;
  const promptLabel =
    assistantPromptModeSelect.value === "custom" ? "custom prompt" : "CavadaLabs prompt";
  sessionLabel.textContent = sessionShortId
    ? `Session ${sessionShortId} · ${label} · ${promptLabel}`
    : `${label} · ${promptLabel}`;
}

function configureLanguages(languages, sourceLanguages) {
  if (languages && typeof languages === "object") {
    const targetOptions = {};
    for (const [value, label] of Object.entries(languages)) {
      targetOptions[value] = LANGUAGE_LABELS[value] || label;
    }
    for (const select of targetLanguageSelects) {
      populateSelect(select, targetOptions, select.value);
    }
  }
  const sourceOptions = {};
  const sourceCatalog =
    sourceLanguages && typeof sourceLanguages === "object"
      ? sourceLanguages
      : { auto: "Auto-detect", ...(languages || LANGUAGE_LABELS) };
  for (const [value, label] of Object.entries(sourceCatalog)) {
    sourceOptions[value] = SOURCE_LANGUAGE_LABELS[value] || LANGUAGE_LABELS[value] || label;
  }
  for (const select of sourceLanguageSelects) {
    populateSelect(select, sourceOptions, select.value);
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
    updateRoomControls();
    return;
  }
  if (message.translation_timing) {
    translationTimingSelect.value = message.translation_timing;
  }
  if (message.source_language) {
    sourceLanguageSelect.value = message.source_language;
  }
  if (message.target_language) {
    translatorTargetLanguageSelect.value = message.target_language;
  } else if (message.language) {
    languageSelect.value = message.language;
  }
  if (Number.isFinite(message.translator_immediate_buffer_ms)) {
    translatorImmediateBufferMs = message.translator_immediate_buffer_ms;
  }
  updateRoomControls();
}

function configureTtsEngines(engines, activeStatus) {
  if (!Array.isArray(engines) || engines.length === 0) {
    return;
  }
  const selected = activeStatus && activeStatus.engine ? activeStatus.engine : selectedEngine();
  for (const select of engineSelects) {
    const previous = select.value || selected;
    select.innerHTML = "";
    for (const engine of engines) {
      const option = document.createElement("option");
      option.value = engine.id;
      option.textContent = engine.label || ENGINE_LABELS[engine.id] || engine.id;
      select.appendChild(option);
    }
    const wanted = [...select.options].some((option) => option.value === previous)
      ? previous
      : selected;
    if ([...select.options].some((option) => option.value === wanted)) {
      select.value = wanted;
    }
  }
}

function updateEngineStatus(status) {
  if (!status) {
    return;
  }
  const engine = status.engine || selectedEngine();
  const label = ENGINE_LABELS[engine] || engine || "TTS";
  const state = status.status || "ready";
  const sampleRate = status.sample_rate || status.tts_sample_rate || ttsSampleRate;
  if (state === "ready") {
    voiceModeLabel.textContent = `${label} ready · ${sampleRate} Hz`;
  } else if (state === "loading" || state === "connecting") {
    voiceModeLabel.textContent = `${label} loading`;
  } else if (state === "failed") {
    voiceModeLabel.textContent = `${label} error`;
  } else if (state === "selected") {
    voiceModeLabel.textContent = `${label} selected`;
  } else {
    voiceModeLabel.textContent = `${label} · ${state}`;
  }
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

function onEventTargetLanguageChange() {
  if (
    activeProtocol !== "event_listener" ||
    !sessionStarted ||
    !socket ||
    socket.readyState !== WebSocket.OPEN
  ) {
    updateRoomControls();
    return;
  }
  socket.send(
    JSON.stringify({
      type: "event.listener.update_language",
      target_language: eventTargetLanguageSelect.value,
    }),
  );
  updateRoomControls();
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

function normalizeEventCode(value) {
  return String(value || "")
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "")
    .slice(0, 12);
}
