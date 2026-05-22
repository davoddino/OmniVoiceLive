class PlayerWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.queuedSamples = 0;
    this.playing = false;
    this.prebufferSamples = Math.max(1, Math.round(sampleRate * 0.06));
    this.port.onmessage = (event) => {
      const message = event.data;
      if (message.type === "audio") {
        const data = new Float32Array(message.samples);
        this.queue.push(data);
        this.queuedSamples += data.length;
        if (!this.playing && this.queuedSamples >= this.prebufferSamples) {
          this.playing = true;
        }
      } else if (message.type === "clear") {
        this.queue = [];
        this.offset = 0;
        this.queuedSamples = 0;
        this.playing = false;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || !output[0]) {
      return true;
    }

    const channel = output[0];
    if (!this.playing) {
      if (this.queuedSamples >= this.prebufferSamples) {
        this.playing = true;
      } else {
        channel.fill(0);
        this.port.postMessage({
          type: "buffer",
          ms: Math.round((this.queuedSamples / sampleRate) * 1000),
        });
        return true;
      }
    }

    for (let i = 0; i < channel.length; i += 1) {
      if (this.queue.length === 0) {
        channel[i] = 0;
        this.playing = false;
        continue;
      }

      const head = this.queue[0];
      channel[i] = head[this.offset] || 0;
      this.offset += 1;
      this.queuedSamples -= 1;

      if (this.offset >= head.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }

    this.port.postMessage({
      type: "buffer",
      ms: Math.round((this.queuedSamples / sampleRate) * 1000),
    });

    return true;
  }
}

registerProcessor("player-worklet", PlayerWorklet);
