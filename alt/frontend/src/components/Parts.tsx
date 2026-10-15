import { useEffect, useRef, useState } from "react";
import { exportSession, mmss, StageMark, stageLabel, VoiceInfo } from "../lib/api";

// Interview flow as a stepper: done stages show the time they took against their target, the current
// stage pulses while the call is live.
export function StageStepper(props: {
  order: string[]; current: string | null; log: StageMark[]; targets: Record<string, number>;
  elapsed: number; live: boolean;
}) {
  const { order, current, log, targets, elapsed, live } = props;
  const startOf = (s: string) => log.find((m) => m.stage === s)?.t;
  const idx = current ? order.indexOf(current) : -1;
  return (
    <ol className="stepper">
      {order.map((s, i) => {
        const start = startOf(s);
        const next = order.slice(i + 1).map(startOf).find((t) => t !== undefined);
        // A stage counts as done only if the call actually entered it; one passed over (the call ended early, or
        // time ran out) shows as skipped, not as done.
        const state = i === idx ? (live ? "now" : "last") : start !== undefined ? "done" : i < idx ? "skipped" : "todo";
        const spent = start === undefined ? null : (next ?? (i === idx ? elapsed : start)) - start;
        const over = spent !== null && targets[s] && spent > targets[s] * 1.25;
        return (
          <li key={s} className={`step ${state}`}>
            <span className="dot">{state === "done" ? "✓" : state === "skipped" ? "–" : i + 1}</span>
            <span className="step-label">{stageLabel(s)}</span>
            <span className={`step-time mono ${over ? "over" : ""}`}>
              {spent !== null ? mmss(spent) : "—"}
              <small> / {mmss(targets[s] ?? 0)}</small>
            </span>
          </li>
        );
      })}
    </ol>
  );
}

export function Stat(props: { label: string; value: string; sub?: string; ratio?: number; tone?: string;
  children?: React.ReactNode }) {
  const { label, value, sub, ratio, tone, children } = props;
  return (
    <div className={`stat ${tone ?? ""}`}>
      <span className="eyebrow">{label}</span>
      <span className="stat-value">{value}{sub && <small>{sub}</small>}</span>
      {ratio !== undefined && (
        <span className="bar"><span style={{ width: `${Math.min(100, Math.max(0, ratio * 100))}%` }} /></span>
      )}
      {children}
    </div>
  );
}

export function StatusPill({ status, reason }: { status: string; reason?: string | null }) {
  const live = status === "live";
  return (
    <span className={`pill ${live ? "live" : status}`} title={reason ?? ""}>
      {live && <span className="pulse" />}
      {live ? "Live" : status === "ended" ? "Ended" : status}
    </span>
  );
}

// Models: the voice model, its built-in transcriber, and the LangGraph brain.
export function ModelCards({ voice }: { voice: VoiceInfo }) {
  const iv = voice.interviewer;
  const tr = voice.transcriber;
  const br = voice.brain;
  return (
    <div className="models">
      {iv && (
        <div className="model">
          <div className="model-head"><span className="eyebrow">Voice model · talks and listens</span><span className="tag">full duplex</span></div>
          <p className="model-name mono">{iv.model}</p>
          <dl>
            <dt>Reasoning</dt><dd>{iv.reasoning_effort}</dd>
            <dt>Voice</dt><dd>{iv.voice}</dd>
            {iv.turn_detection && (<><dt>Turn detection</dt>
              <dd>{iv.turn_detection.type}{iv.turn_detection.eagerness ? ` · ${iv.turn_detection.eagerness}` : ""}</dd></>)}
            {iv.noise_reduction && (<><dt>Noise reduction</dt><dd>{iv.noise_reduction}</dd></>)}
            {iv.barge_in && (<><dt>Interruptions</dt><dd>{iv.barge_in}</dd></>)}
          </dl>
        </div>
      )}
      {tr && (
        <div className="model">
          <div className="model-head"><span className="eyebrow">Transcription model</span><span className="tag fixed">fixed</span></div>
          <p className="model-name mono">{tr.model}</p>
          <dl>
            <dt>Language</dt><dd>{tr.language ?? "auto"}</dd>
            <dt>Role</dt><dd>Candidate text for the record and for the brain; the voice model listens to the audio itself</dd>
          </dl>
        </div>
      )}
      {br && (
        <div className="model">
          <div className="model-head"><span className="eyebrow">Brain · decides and thinks</span><span className="tag">per turn</span></div>
          <p className="model-name mono">{br.model}</p>
          <dl>
            <dt>Control</dt><dd>{br.graph}: routing, cards, stages and checks in code</dd>
            <dt>Thinking</dt><dd>Interprets each turn, extracts evidence, writes follow-ups, flags risks</dd>
            <dt>Deadline</dt><dd>{br.deadline_s} s per turn, then a deterministic fallback</dd>
          </dl>
        </div>
      )}
    </div>
  );
}

export function ExportMenu({ id, baseName }: { id: string; baseName: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  const go = (f: "md" | "json") => { setOpen(false); void exportSession(id, f, baseName); };
  return (
    <div className="menu" ref={ref}>
      <button className="primary" onClick={() => setOpen(!open)}>Export ▾</button>
      {open && (
        <div className="menu-pop">
          <button onClick={() => go("md")}><b>Markdown</b><span>Readable transcript, evidence and flags</span></button>
          <button onClick={() => go("json")}><b>JSON</b><span>Full record for downstream processing</span></button>
        </div>
      )}
    </div>
  );
}
