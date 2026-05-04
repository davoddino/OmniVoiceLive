class RecorderWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.bufferSize = Math.max(128, Math.round(sampleRate * 0.02));
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) {
      return true;
    }

    const channel = input[0];
    for (let i = 0; i < channel.length; i += 1) {
      this.buffer.push(channel[i]);
    }

    while (this.buffer.length >= this.bufferSize) {
      const frame = new Float32Array(this.bufferSize);
      for (let i = 0; i < this.bufferSize; i += 1) {
        frame[i] = this.buffer.shift();
      }
      this.port.postMessage(frame, [frame.buffer]);
    }

    return true;
  }
}

registerProcessor("recorder-worklet", RecorderWorklet);
