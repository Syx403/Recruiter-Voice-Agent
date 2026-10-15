import { useEffect, useMemo, useRef, useState } from "react";
import { ExportMenu, ModelCards, StageStepper, StatusPill } from "../components/Parts";
import { Player, wsUrl } from "../lib/audio";
import { cleanName, decideBrief, Entry, Evidence, Flag, getSession, Layout, listSessions, LiveState, mmss, Pipeline,
  SessionRecord, SessionRow, stageLabel, TraceStep, TurnMetric, writeBrief } from "../lib/api";

// One screening call, live or from history: the full conversation, the evidence the
// interviewer recorded, flags, models, cost, and exports. Live updates stream over the monitor socket.

type Tab = "brief" | "card" | "plan" | "evidence" | "flags" | "brain" | "models";

const median = (xs: (number | null | undefined)[]) => {
  const v = xs.filter((x): x is number => typeof x === "number").sort((a, b) => a - b);
  return v.length ? v[Math.floor(v.length / 2)] : null;
};
const secs = (x: number | null | undefined) => (typeof x === "number" ? `${x.toFixed(1)} s` : "—");

export default function Session({ id }: { id: string }) {
  const [rec, setRec] = useState<SessionRecord>();
  const [live, setLive] = useState<LiveState>();
  const [entries, setEntries] = useState<Record<string, Entry>>({});
  const [error, setError] = useState("");
  const [tab, setTab] = useState<Tab>("card");
  const [showNoise, setShowNoise] = useState(false);
  const [mirror, setMirror] = useState(false);
  const player = useRef<Player | undefined>(undefined);
  const mirrorRef = useRef(false);
  // Recruiters see the interview; the planner's internals (Brain, models, cost, latency, tool calls) are a toggle.
  const [tech, setTech] = useState(() => { try { return localStorage.getItem("tech") === "1"; } catch { return false; } });
  function toggleTech() {
    setTech(!tech);
    try { localStorage.setItem("tech", tech ? "0" : "1"); } catch { /* storage unavailable */ }
  }
  const wsRef = useRef<WebSocket | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  const [tick, setTick] = useState(0);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const lastState = useRef(0);

  useEffect(() => {
    let ws: WebSocket | undefined;
    getSession(id)
      .then((r) => {
        setRec(r);
        if (r.session.status === "ended") setTab("brief");
        setEntries(Object.fromEntries(r.timeline.map((e) => [e.id, e])));
        if (r.session.status === "live" || r.session.status === "created") {
          ws = new WebSocket(wsUrl(`/ws/monitor/${id}`));
          wsRef.current = ws;
          ws.onmessage = (ev) => {
            const m = JSON.parse(ev.data);
            if (m.type === "state") {
              setLive(m.state);
              lastState.current = performance.now();
            }
            else if (m.type === "timeline") setEntries((x) => ({ ...x, [m.entry.id]: { ...x[m.entry.id], ...m.entry } }));
            else if (m.type === "audio" && mirrorRef.current) player.current?.play(m.item_id, m.pcm);
            else if (m.type === "flush") player.current?.flush();
            else if (m.type === "ended") getSession(id).then(setRec);
          };
        }
      })
      .catch((e) => setError(String(e)));
    listSessions().then(setSessions);
    return () => ws?.close();
  }, [id]);

  // Tick the clock between server pushes (every 5 s) while the call is live.
  useEffect(() => {
    const h = window.setInterval(() => setTick((x) => x + 1), 1000);
    return () => window.clearInterval(h);
  }, []);

  const timeline = useMemo(() => Object.values(entries).sort((a, b) => a.t - b.t), [entries]);

  useEffect(() => {
    if (follow.current) bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [timeline]);

  if (error) return <div className="page"><p className="error">{error}</p></div>;
  if (!rec) return <div className="page"><p className="notice">Loading session…</p></div>;

  const isLive = !!live && !live.ended && live.status === "live";
  const status = live?.status ?? rec.session.status;
  const names = live?.names ?? rec.names;
  const voice = live?.voice ?? rec.voice;
  const drift = isLive && lastState.current ? (performance.now() - lastState.current) / 1000 : 0;
  const elapsed = (live?.clock.elapsed_s ?? rec.elapsed_s ?? 0) + (tick >= 0 ? drift : 0);
  const cost = live?.cost_usd ?? rec.session.cost_usd;
  const order = live?.order ?? rec.order;
  const stageLog = live?.stage_log ?? rec.stage_log;
  const current = live?.stage ?? stageLog[stageLog.length - 1]?.stage ?? null;
  const evidence = live?.ledger ?? rec.evidence;
  const flags = live?.flags ?? rec.flags;
  const consistency = live?.consistency ?? rec.consistency;
  const others = sessions.filter((o) => o.candidate_id === rec.session.candidate_id && o.id !== id).slice(0, 5);
  const noiseCount = timeline.filter((e) => e.kind === "noise").length;
  const turns: TurnMetric[] = live?.turns ?? rec.turns ?? [];
  const split = live?.cost_split ?? rec.cost_split;
  const firstSound = median(turns.map((m) => m.first_sound_s));
  const nextLine = median(turns.map((m) => m.next_line_s));

  function toggleMirror() {
    if (!player.current) player.current = new Player();
    mirrorRef.current = !mirror;
    setMirror(!mirror);
    // The server streams interviewer audio to this page only while mirroring (audio can crowd out state updates).
    wsRef.current?.send(JSON.stringify({ type: "mirror", on: !mirror }));
  }

  return (
    <div className="session">
      <div className="topbar">
        <a className="back" href="#/">← All interviews</a>
        <div className="s-actions">
          <label className="check small"><input type="checkbox" checked={tech} onChange={toggleTech} /> Technical details</label>
          {isLive && (
            <button className={mirror ? "primary" : "ghost"} onClick={toggleMirror}>
              {mirror ? "Mirroring interviewer audio" : "Mirror interviewer audio"}
            </button>
          )}
          <ExportMenu id={id} baseName={`screen_${cleanName(names.candidate ?? rec.session.candidate_id).replace(/\s+/g, "_")}_${id}`} />
        </div>
      </div>

      <section className="s-strip">
        <div className="s-who">
          <h1>{cleanName(rec.candidate.name ?? names.candidate ?? rec.session.candidate_id)}</h1>
          <span className="muted">{rec.role.title ?? names.role} · {rec.role.company ?? names.company}
            {rec.role.location ? ` · ${rec.role.location}` : ""} · interviewer {names.interviewer} ({names.persona})</span>
        </div>
        <div className="s-meta">
          <span><span className="eyebrow">Duration</span><b className="mono">{mmss(elapsed)}</b><small> / 15:00</small></span>
          <span><span className="eyebrow">Stage</span><b>{current ? stageLabel(current) : "—"}</b></span>
          <span><span className="eyebrow">Status</span>
            <StatusPill status={status} reason={live?.end_reason ?? rec.session.end_reason} />
            {isLive && <small className="muted"> {live?.agent_speaking ? "interviewer speaking" : live?.candidate_speaking ? "candidate speaking" : "listening"}</small>}
          </span>
          {tech && <>
            <span><span className="eyebrow">Cost</span><b className="mono">${cost.toFixed(3)}</b>
              {split && <small className="muted"> voice {split.voice.toFixed(3)} · brain {split.brain.toFixed(4)}</small>}</span>
            <span><span className="eyebrow">First sound / next question</span>
              <b className="mono">{secs(firstSound)} / {secs(nextLine)}</b><small className="muted"> median</small></span>
            <span className="code muted small">{id}</span>
          </>}
        </div>
        {(rec.candidate.cv_summary || others.length > 0 || (!isLive && rec.session.end_reason)) && (
          <details className="s-more">
            <summary>More about this candidate and call</summary>
            {!isLive && rec.session.end_reason && <p className="small">End: {rec.session.end_reason}</p>}
            {rec.candidate.cv_summary && <p className="small">{rec.candidate.cv_summary}</p>}
            {others.map((o) => (
              <a key={o.id} className="small" href={`#/sessions/${o.id}`}>
                {new Date((o.started_at ?? o.created_at) * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "short" })}
                <span className="mono"> · {o.elapsed_s ? mmss(o.elapsed_s) : "—"} · {o.status}</span>
              </a>
            ))}
          </details>
        )}
      </section>

      <StageStepper order={order} current={current} log={stageLog} targets={live?.stage_targets ?? rec.stage_targets}
                    elapsed={elapsed} live={isLive} />

      <div className="s-grid">
        <section className="thread"
                 onScroll={(e) => {
                   const el = e.currentTarget;
                   follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
                 }}>
          <div className="thread-head">
            <h3>Conversation</h3>
            {noiseCount > 0 && (
              <label className="check small">
                <input type="checkbox" checked={showNoise} onChange={(e) => setShowNoise(e.target.checked)} />
                Show {noiseCount} ignored non-speech sound{noiseCount > 1 ? "s" : ""}
              </label>
            )}
          </div>
          {timeline.length === 0 && <p className="notice">No conversation yet.</p>}
          {timeline.filter((e) => tech || (e.kind !== "tool" && e.kind !== "note"))
            .map((e) => <Line key={e.id} e={e} showNoise={showNoise} />)}
          {rec.recording && !isLive && (
            <div className="recording">
              <span className="eyebrow">Recording · left candidate, right interviewer</span>
              <audio controls preload="none" src={`/api/sessions/${id}/recording`} />
            </div>
          )}
          <div ref={bottom} />
        </section>

        <aside className="side">
          <nav className="tabs">
            {((isLive ? ["card", "plan", "evidence", "flags"] : ["brief", "plan", "evidence", "flags"]).concat(tech ? ["brain", "models"] : []) as Tab[]).map((t) => (
              <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>
                {{ brief: "Brief", card: "Question", plan: "Questions", evidence: `Evidence (${evidence.length})`, flags: `Flags (${flags.length})`,
                   brain: `Brain (${turns.length})`, models: "Models" }[t]}
              </button>
            ))}
          </nav>
          {tab === "card" && <CardPane live={live} />}
          {tab === "plan" && <PlanPane layout={live?.layout ?? rec.layout ?? null} pipeline={rec.pipeline ?? null} />}
          {tab === "evidence" && <EvidencePane evidence={evidence} />}
          {tab === "flags" && <FlagsPane flags={flags} consistency={consistency} />}
          {tab === "brief" && <BriefPane sid={id} pipeline={rec.pipeline ?? null} ended={rec.session.status === "ended"}
                                          onChange={() => getSession(id).then(setRec)} />}
          {tab === "brain" && <BrainPane turns={turns} />}
          {tab === "models" && <ModelCards voice={voice} />}
        </aside>
      </div>
    </div>
  );
}

function heardSplit(e: Entry): [string, string] {
  if (!e.interrupted || !e.audio_ms) return [e.text, ""];
  const cut = Math.floor(e.text.length * Math.min(1, (e.heard_ms ?? 0) / e.audio_ms));
  const i = e.text.lastIndexOf(" ", cut);
  const at = i > 0 ? i : cut;
  return [e.text.slice(0, at), e.text.slice(at)];
}

function Line({ e, showNoise }: { e: Entry; showNoise: boolean }) {
  if (e.kind === "stage") {
    return <div className="divider"><span>{stageLabel(e.stage ?? e.text)}</span><span className="mono">{mmss(e.t)}</span></div>;
  }
  if (e.kind === "noise") {
    if (!showNoise) return null;
    return <div className="noise mono">{mmss(e.t)} · non-speech sound ignored{e.text ? ` (“${e.text}”)` : ""}</div>;
  }
  if (e.kind === "tool") return <ToolChip e={e} />;
  if (e.kind === "note") return <NoteChip e={e} />;
  const agent = e.kind === "agent";
  const [heard, unheard] = agent ? heardSplit(e) : [e.text, ""];
  return (
    <div className={`msg ${agent ? "agent" : "cand"}`}>
      <div className="who">
        <span className="avatar">{agent ? "AI" : "C"}</span>
        <span className="mono t">{mmss(e.t)}</span>
      </div>
      <div className="bubble">
        <p>
          {heard}
          {unheard && <span className="unheard" title="Not heard: the candidate interrupted">{unheard}</span>}
          {e.partial && <span className="caret" />}
        </p>
        {e.interrupted && <span className="badge">interrupted at {mmss((e.heard_ms ?? 0) / 1000)}</span>}
        {agent && e.role === "bridge" && <span className="badge bridge">bridging while the brain thinks</span>}
        {!agent && (
          <span className="src">
            {e.intent && e.intent !== "answer" && <span className="intent" title={e.intent_reason}>{e.intent.replace(/_/g, " ")}</span>}
            transcribed
          </span>
        )}
      </div>
    </div>
  );
}

function ToolChip({ e }: { e: Entry }) {
  const [open, setOpen] = useState(false);
  const out = e.output ?? {};
  const status = String(out.status ?? "");
  const args = e.arguments ?? {};
  const summary = args.field ? `${args.field} = ${JSON.stringify(args.value)}`
    : args.to ? `→ ${stageLabel(String(args.to))}` : e.name === "check_quota_consistency" ? String(out.status ?? "") : "";
  return (
    <div className={`tool ${status}`}>
      <button className="tool-row" onClick={() => setOpen(!open)}>
        <span className="mono name">{e.name}</span>
        <span className="sum">{summary}</span>
        <span className={`st ${status}`}>{status}</span>
      </button>
      {open && <pre className="mono">{JSON.stringify({ arguments: args, output: out }, null, 2)}</pre>}
    </div>
  );
}

function NoteChip({ e }: { e: Entry }) {
  const [open, setOpen] = useState(false);
  const timing = [e.brain_ms !== undefined ? `brain ${(e.brain_ms / 1000).toFixed(1)} s` : "",
    typeof e.first_sound_s === "number" ? `first sound ${e.first_sound_s.toFixed(1)} s` : "",
    typeof e.next_line_s === "number" ? `next line ${e.next_line_s.toFixed(1)} s` : ""].filter(Boolean).join(" · ");
  return (
    <div className="note">
      <button className="tool-row" onClick={() => setOpen(!open)}>
        <span className="mono name">brain</span>
        <span className="sum">{(e.intent ?? "").replace(/_/g, " ")}{e.followup_by_brain ? " · follow-up written by the LLM" : ""}
          {timing && ` · ${timing}`}</span>
      </button>
      {e.trace && <Trace steps={e.trace} />}
      {open && (
        <div className="note-text">
          <p>{e.text}</p>
          {!!e.dropped?.length && <p className="muted">Dropped at the trust boundary: {e.dropped.join("; ")}</p>}
          {!!e.problems?.length && <p className="muted">Degraded: {e.problems.join("; ")}</p>}
        </div>
      )}
    </div>
  );
}

const LLM_NODES = new Set(["interpret"]);

function Trace({ steps }: { steps: TraceStep[] }) {
  return (
    <div className="trace">
      {steps.map((s, i) => (
        <span key={i} className={`node ${LLM_NODES.has(s.node) ? "llm" : "code"} ${s.status ?? ""}`}
              title={JSON.stringify(s)}>
          {s.node}{s.ms !== null && s.ms !== undefined && s.ms >= 5 ? ` ${s.ms} ms` : ""}{s.status === "degraded" ? " · degraded" : ""}
        </span>
      ))}
    </div>
  );
}

function BrainPane({ turns }: { turns: TurnMetric[] }) {
  return (
    <div className="pane-body">
      <p className="eyebrow">Per turn (LangGraph)</p>
      <div className="trace legend">
        <span className="node code">prepare</span><span className="node llm">interpret · Haiku</span>
        <span className="node code">check</span><span className="node code">route →</span>
        <span className="node code">record → advance | clarify | redirect | defer | close</span>
        <span className="node code">speak</span>
      </div>
      <p className="notice small">Purple is the LLM (judgement); grey is code (control). The voice model speaks while
        this runs, then says the line the graph returns.</p>
      {!turns.length && <p className="notice small">No turns yet.</p>}
      {!!turns.length && (
        <table className="table">
          <thead><tr><th>#</th><th>kind</th><th>brain</th><th>first sound</th><th>next line</th></tr></thead>
          <tbody>
            {turns.map((m) => (
              <tr key={m.turn}>
                <td className="mono">{m.turn}</td>
                <td>{m.kind?.replace(/_/g, " ")}{m.degraded ? " (degraded)" : ""}</td>
                <td className="mono">{(m.brain_ms / 1000).toFixed(1)} s</td>
                <td className="mono">{secs(m.first_sound_s)}</td>
                <td className="mono">{m.sent ? secs(m.next_line_s) : "voice"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function CardPane({ live }: { live?: LiveState }) {
  if (!live?.card) return <p className="notice small">The current question card shows here during a live call.</p>;
  return (
    <div className="pane-body">
      <p className="mono muted">{live.card.id} · follow-ups {live.card.probes_used}/{live.card.max_probes}</p>
      <p className="serif">{live.card.primary}</p>
    </div>
  );
}

function EvidencePane({ evidence }: { evidence: Evidence[] }) {
  const groups = evidence.reduce<Record<string, Evidence[]>>((g, e) => ({ ...g, [e.dimension]: [...(g[e.dimension] ?? []), e] }), {});
  if (!evidence.length) return <p className="notice small">Nothing recorded yet.</p>;
  return (
    <div className="pane-body">
      {Object.entries(groups).map(([dim, list]) => (
        <div key={dim} className="ev-group">
          <p className="eyebrow">{dim.replace(/_/g, " ")}</p>
          {list.map((e, i) => (
            <div key={i} className="ev">
              <span className="mono f">{e.field}</span>
              <span>{e.declined ? <i>declined</i> : e.value}</span>
              <span className="q">“{e.quote}”</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function FlagsPane({ flags, consistency }: { flags: Flag[]; consistency: Record<string, unknown>[] }) {
  return (
    <div className="pane-body">
      {consistency.map((c, i) => (
        <div key={i} className={`flag ${c.status === "inconsistent" ? "code" : "ok"}`}>
          <b>Quota check: {String(c.status)}</b>
          {c.ratio !== undefined && <span> · claimed / implied = {String(c.ratio)}</span>}
          {c.readback !== undefined && <p>{String(c.readback)}</p>}
        </div>
      ))}
      {!flags.length && !consistency.length && <p className="notice small">No flags.</p>}
      {flags.map((f, i) => (
        <div key={i} className={`flag ${f.source}`}>
          <b>{f.type.replace(/_/g, " ")}</b> <span className="muted">· {f.source} · {mmss(f.t)}</span>
          <p>{f.detail}</p>
        </div>
      ))}
    </div>
  );
}


const COMP_LABEL: Record<string, string> = {
  deal_execution: "Deal execution", performance_credibility: "Performance credibility",
  technical_credibility: "Technical credibility", outcome_orientation: "Outcome orientation",
  ownership_learning: "Ownership and learning",
};
const REC_LABEL: Record<string, string> = {
  recommend: "Recommend", conditional: "Conditional", not_recommended: "Not recommended",
  needs_verification: "Needs verification",
};

// The hiring-manager brief: the LLM scored with quotes, code grounded the quotes and computed the total; the recruiter
// decides (LangGraph interrupt in the candidate pipeline).
function BriefPane({ sid, pipeline, ended, onChange }: { sid: string; pipeline: Pipeline | null; ended: boolean;
  onChange: () => void }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const writing = !!pipeline && !pipeline.brief && !pipeline.brief_error &&
    ["briefing", "in_call", "invited"].includes(pipeline.status) && ended;
  useEffect(() => {
    if (!writing) return;
    const h = window.setInterval(onChange, 3000);
    return () => window.clearInterval(h);
  }, [writing, onChange]);
  if (!ended) return <p className="notice small">The brief is written when the call ends.</p>;
  if (!pipeline) {
    return (
      <div className="pane-body">
        <p className="notice small">This call has no brief yet.</p>
        <button className="primary" disabled={busy} onClick={async () => { setBusy(true); await writeBrief(sid); onChange(); }}>
          {busy ? "Writing the brief…" : "Write the brief"}
        </button>
      </div>
    );
  }
  if (writing) return <p className="notice small">Writing the brief… (score → ground quotes → compute)</p>;
  if (pipeline.brief_error) return <p className="error">Brief failed: {pipeline.brief_error}</p>;
  const b = pipeline.brief!, c = pipeline.computed!;
  const waiting = pipeline.next?.includes("review");
  async function decide(d: string) {
    setBusy(true);
    await decideBrief(pipeline!.id, d, note);
    setBusy(false);
    onChange();
  }
  return (
    <div className="pane-body brief">
      <div className="brief-head">
        <span className="brief-total">{c.total_25 ?? "—"}<small> / 25</small></span>
        <span className={`rec ${c.recommendation}`}>{REC_LABEL[c.recommendation] ?? c.recommendation}</span>
      </div>
      <p className="serif">{b.summary}</p>
      {[...c.notes, ...c.knockouts_to_review.map((k) => `Knockout to review: ${k.replace(/_/g, " ")}`)].map((n, i) =>
        <p key={i} className="muted small">⚠ {n}</p>)}
      <p className="eyebrow">Competencies</p>
      {b.competencies.map((x) => {
        const s = c.scores[x.id];
        return (
          <div key={x.id} className="comp">
            <div className="comp-head">
              <b>{COMP_LABEL[x.id] ?? x.id}</b>
              <span className="muted small"> · weight {c.weights[x.id]}</span>
              <span className="mono comp-score">{s == null ? "not observed" : `${s} / 5`}</span>
            </div>
            <span className="bar"><span style={{ width: `${((s ?? 0) / 5) * 100}%` }} /></span>
            {x.evidence.map((q, i) => <p key={i} className="q">“{q}”</p>)}
            <p className="muted small">{x.rationale}</p>
          </div>
        );
      })}
      {(c.requirement_fit ?? []).length > 0 && (<>
        <p className="eyebrow">Must-have requirements: CV before the call</p>
        {c.requirement_fit!.map((r) => (
          <div key={r.requirement_id} className="fit"><span className={`st ${r.status}`}>{r.status.replace("_", " ")}</span>
            <b>{r.requirement_id}</b> <span className="muted small">{r.note}</span></div>
        ))}
      </>)}
      {(c.cv_check ?? []).length > 0 && (<>
        <p className="eyebrow">CV vs what the candidate said (code)</p>
        <table className="table small"><tbody>
          {c.cv_check!.map((r) => (
            <tr key={r.claim}><td className="mono">{r.field}</td><td>{r.cv}</td><td>{r.said ?? "—"}</td>
              <td><span className={`st ${r.status === "differs" ? "flag" : r.status === "match" ? "pass" : "unknown"}`}>
                {r.status.replace("_", " ")}</span></td></tr>
          ))}
        </tbody></table>
      </>)}
      <p className="eyebrow">Fit checks</p>
      {b.fit_checks.map((f) => (
        <div key={f.id} className="fit"><span className={`st ${f.status}`}>{f.status}</span>
          <b>{f.id.replace(/_/g, " ")}</b> <span className="muted small">{f.note}</span></div>
      ))}
      {([["Risks", b.risks], ["Next-round focus", b.next_round_focus], ["Not obtained", b.missing_information]] as const)
        .map(([title, list]) => list.length > 0 && (
          <div key={title}><p className="eyebrow">{title}</p><ul>{list.map((x, i) => <li key={i}>{x}</li>)}</ul></div>
        ))}
      <p className="eyebrow">Recruiter decision</p>
      {waiting ? (
        <div className="decide">
          <textarea placeholder="Note for the hiring manager (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
          <div className="decide-row">
            <button className="primary" disabled={busy} onClick={() => decide("approve")}>Approve for next round</button>
            <button className="ghost" disabled={busy} onClick={() => decide("verify")}>Request verification</button>
            <button className="ghost" disabled={busy} onClick={() => decide("reject")}>Reject</button>
          </div>
        </div>
      ) : (
        <p><b>{pipeline.decision?.decision ?? pipeline.status}</b>{pipeline.decision?.note ? ` — ${pipeline.decision.note}` : ""}</p>
      )}
    </div>
  );
}


// What this call asks and why: the seed, the stage order, each card's wording variant, and which
// questions the candidate's approved plan personalised or added.
function PlanPane({ layout, pipeline }: { layout: Layout | null; pipeline: Pipeline | null }) {
  if (!layout) return <p className="notice small">This call predates per-session plans.</p>;
  return (
    <div className="pane-body">
      <p className="small">Seed <span className="mono">{layout.seed}</span> · market {layout.market ?? "—"}
        {pipeline && <> · <a href={`#/pipelines/${pipeline.id}`}>question plan →</a></>}</p>
      {layout.plan_summary && <p className="serif small">{layout.plan_summary}</p>}
      <p className="muted small">Same fields for every candidate of the role; the order and the A/B wording come from the
        seed, personalised wording and extra probes from the approved plan.</p>
      {layout.order.map((st) => (
        <div key={st} className="plan-stage">
          <p className="eyebrow">{stageLabel(st)}</p>
          {(layout.cards[st] ?? []).map((c) => (
            <div key={c.id} className="plan-card">
              <span className="mono small">{c.id}
                {c.variant && <span className="tag">{c.variant}</span>}
                {c.personalised && <span className="tag rust">personalised</span>}
                {c.extra && <span className="tag rust">extra probe</span>}
                {c.role_question && <span className="tag">role question</span>}
                {c.requirements.length > 0 && <span className="muted"> · {c.requirements.join(", ")}</span>}</span>
              <p className="small">{c.text}</p>
              {c.standard && <p className="muted small">Standard: {c.standard}</p>}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
