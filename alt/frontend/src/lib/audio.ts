// Browser audio: 24 kHz PCM16 capture through an AudioWorklet, and gapless playback that tracks how
// much of each agent item was actually played (needed for truncate on barge-in).

const RATE = 24000;

const WORKLET = `
class Capture extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Int16Array(2400); this.n = 0; }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.buffer.slice(0)); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("capture", Capture);
`;

export class Mic {
  private ctx?: AudioContext;
  private stream?: MediaStream;

  async start(onFrame: (pcm: ArrayBuffer) => void) {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    this.ctx = new AudioContext({ sampleRate: RATE });
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await this.ctx.audioWorklet.addModule(url);
    const node = new AudioWorkletNode(this.ctx, "capture");
    node.port.onmessage = (e) => onFrame(e.data as ArrayBuffer);
    this.ctx.createMediaStreamSource(this.stream).connect(node);
  }

  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.ctx?.close();
  }
}

function b64ToPcm(b64: string): Float32Array<ArrayBuffer> {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const pcm = new Int16Array(bytes.buffer);
  const out = new Float32Array(new ArrayBuffer(pcm.length * 4));
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i] / 0x8000;
  return out;
}

type Item = { chunks: { start: number; dur: number }[] };

export class Player {
  readonly ctx = new AudioContext({ sampleRate: RATE });
  private sources = new Set<AudioBufferSourceNode>();
  private items = new Map<string, Item>();
  private queueEnd = 0;
  private idleTimer?: number;
  onIdle?: () => void;
  onActive?: (active: boolean) => void;

  play(itemId: string, b64: string) {
    const data = b64ToPcm(b64);
    const buf = this.ctx.createBuffer(1, data.length, RATE);
    buf.copyToChannel(data, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    // gpt-live-1 streams audio continuously in real time (silence included): keep a small jitter buffer, and
    // re-buffer 150 ms when the queue ran dry so network jitter does not click.
    const start = Math.max(this.ctx.currentTime + (this.queueEnd < this.ctx.currentTime ? 0.15 : 0.02), this.queueEnd);
    src.start(start);
    this.queueEnd = start + buf.duration;
    const it = this.items.get(itemId) ?? { chunks: [] };
    it.chunks.push({ start, dur: buf.duration });
    this.items.set(itemId, it);
    this.sources.add(src);
    src.onended = () => {
      this.sources.delete(src);
      if (this.sources.size === 0) this.scheduleIdle();
    };
    window.clearTimeout(this.idleTimer);
    this.onActive?.(true);
  }

  private scheduleIdle() {
    window.clearTimeout(this.idleTimer);
    this.idleTimer = window.setTimeout(() => {
      this.onActive?.(false);
      this.onIdle?.();
    }, 300);
  }

  /** Stop everything; return the item that was playing and how many ms of it were heard. */
  flush(): { itemId: string; playedMs: number } | null {
    const now = this.ctx.currentTime;
    let current: { itemId: string; playedMs: number } | null = null;
    // Played = the full length of chunks that finished plus the part of the one playing now. Summing chunks
    // (not "now - first start") stays right when the queue drained between chunks.
    for (const [id, it] of this.items) {
      if (!it.chunks.some((c) => c.start <= now && now < c.start + c.dur)) continue;
      let played = 0;
      for (const c of it.chunks) played += Math.max(0, Math.min(c.dur, now - c.start));
      current = { itemId: id, playedMs: Math.round(played * 1000) };
    }
    this.sources.forEach((s) => s.stop());
    this.sources.clear();
    this.queueEnd = 0;
    this.onActive?.(false);
    return current;
  }

  close() {
    this.flush();
    void this.ctx.close();
  }
}

export function wsUrl(path: string) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${path}`;
}
