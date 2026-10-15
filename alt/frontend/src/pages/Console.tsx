import { useEffect, useState } from "react";
import { AppConfig, cancelPipeline, cleanName, createInvite, getConfig, InviteRow, listInvites, listPipelines, listSessions, mmss,
  Pipeline, SessionRow } from "../lib/api";

// The recruiter's console: invite a candidate (role → interviewer style → candidate → reviewed questions → link),
// follow interviews, read outcomes. Technical views (models, architecture, spend) live on the System page.

const OPENING_STATUS: Record<string, [string, string]> = {
  approved: ["Ready", "tone-good"], review: ["Needs your review", "tone-warn"], draft: ["Needs your review", "tone-warn"],
  none: ["Not set up", "tone-muted"], input_error: ["Job description incomplete", "tone-bad"],
};
const OUTCOME: Record<string, [string, string]> = {
  recommend: ["Recommended", "tone-good"], conditional: ["Conditional", "tone-warn"],
  needs_verification: ["Needs verification", "tone-warn"], not_recommended: ["Not recommended", "tone-bad"],
};

function endLabel(s: SessionRow): string {
  if (s.status === "live") return "In progress";
  const r = s.end_reason ?? "";
  if (r.includes("non-cooperative")) return "Ended: not engaging";
  if (r.includes("abusive")) return "Ended: abusive language";
  if (r.includes("asked to stop")) return "Ended by the candidate";
  if (r.includes("no consent")) return "No consent";
  if (r.includes("time")) return "Completed (time limit)";
  return s.status === "ended" ? "Completed" : s.status;
}

/** role title → [candidate id → calls], newest first at every level. */
function groupByRole(rows: SessionRow[]): [string, [string, SessionRow[]][]][] {
  const roles = new Map<string, Map<string, SessionRow[]>>();
  for (const r of rows) {
    const role = r.names.role ?? r.role_id;
    if (!roles.has(role)) roles.set(role, new Map());
    const cands = roles.get(role)!;
    if (!cands.has(r.candidate_id)) cands.set(r.candidate_id, []);
    cands.get(r.candidate_id)!.push(r);
  }
  return [...roles.entries()].map(([role, cands]) => [role, [...cands.entries()]]);
}

export default function Console() {
  const [cfg, setCfg] = useState<AppConfig>();
  const [role, setRole] = useState("");
  const [persona, setPersona] = useState("");
  const [cand, setCand] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [invites, setInvites] = useState<InviteRow[]>([]);
  const [pipes, setPipes] = useState<Pipeline[]>([]);

  useEffect(() => {
    getConfig().then((c) => {
      setCfg(c);
      setRole(c.defaults.role);
      setPersona(c.defaults.persona);
      setCand(c.candidates[0]?.id ?? "");
    });
    const refresh = () => {
      listSessions().then(setSessions);
      listInvites().then(setInvites);
      listPipelines().then(setPipes);
    };
    refresh();
    const h = window.setInterval(refresh, 3000);     // statuses change when candidates join or finish
    return () => window.clearInterval(h);
  }, []);

  if (!cfg) return <div className="page"><p className="notice">Loading…</p></div>;

  async function invite() {
    setErr("");
    setBusy(true);
    try {
      const p = await createInvite({ candidate_id: cand, role_id: role, persona_id: persona });
      location.hash = `#/pipelines/${p.id}`;
    } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  }

  const link = (i: InviteRow) => `${location.origin}${i.url}`;
  const live = invites.filter((i) => i.status === "in_call");
  const waiting = invites.filter((i) => i.status === "invited");
  const opening = cfg.roles.find((r) => r.id === role);
  const toReview = pipes.filter((p) => p.status === "plan_review");
  async function cancel(pid: string, who: string) {
    if (!window.confirm(`Cancel the interview for ${who}? An issued link will stop working.`)) return;
    try { await cancelPipeline(pid); } catch (e) { setErr((e as Error).message); }
    listPipelines().then(setPipes);
    listInvites().then(setInvites);
  }
  const candidate = cfg.candidates.find((c) => c.id === cand);

  return (
    <div className="page wide">
      <header className="masthead row-between">
        <div>
          <p className="eyebrow">Cygnify · Talent screening</p>
          <h1>Screening interviews</h1>
          <p className="lede">Invite a candidate to a structured first-round voice interview. The questions are prepared
            from the job description and the candidate's CV; you review them before the invitation goes out.</p>
        </div>
        <a className="muted small" href="#/system">System &amp; architecture →</a>
      </header>

      {live.length > 0 && (
        <section className="card live-now">
          <h3><span className="pulse" /> Interviews in progress</h3>
          {live.map((i) => (
            <a key={i.token} className="live-row" href={`#/sessions/${i.session_id}`}>
              <b>{i.candidate}</b><span className="muted">{i.role}</span><span className="btn primary">Watch →</span>
            </a>
          ))}
        </section>
      )}

      <div className="console-grid">
        <section className="card form-card">
          <h3>Invite a candidate</h3>
          <label>
            <span className="eyebrow">Job opening</span>
            <select value={role} onChange={(e) => setRole(e.target.value)}>
              {cfg.roles.map((r) => <option key={r.id} value={r.id}>{r.title} · {r.company}</option>)}
            </select>
          </label>
          <label>
            <span className="eyebrow">Interviewer style</span>
            <select value={persona} onChange={(e) => setPersona(e.target.value)}>
              {cfg.personas.map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
            </select>
          </label>
          <label>
            <span className="eyebrow">Candidate</span>
            <select value={cand} onChange={(e) => setCand(e.target.value)}>
              {cfg.candidates.map((c) => <option key={c.id} value={c.id}>{c.name}{c.location ? ` · ${c.location}` : ""}</option>)}
            </select>
          </label>
          {candidate?.headline && <p className="muted small">{candidate.headline}</p>}
          {opening && opening.status !== "approved" ? (
            <p className="notice small">This job opening's interview setup needs your review first.{" "}
              <a href={`#/roles/${role}`}>Review it →</a></p>
          ) : (
            <button className="primary" disabled={busy} onClick={invite}>
              {busy ? "Preparing questions…" : "Prepare interview questions"}</button>
          )}
          {err && <p className="error small">{err}</p>}

          {toReview.length > 0 && (
            <div className="waiting">
              <span className="eyebrow">Questions waiting for your review</span>
              {toReview.map((p) => (
                <div key={p.id} className="wait-row">
                  <span>{p.candidate} · <span className="muted">{p.role}</span></span>
                  <span className="row-actions">
                    <a className="ghost small btn" href={`#/pipelines/${p.id}`}>Review</a>
                    <button className="ghost small" onClick={() => cancel(p.id, p.candidate ?? "")}>Cancel</button>
                  </span>
                </div>
              ))}
            </div>
          )}
          {waiting.length > 0 && (
            <div className="waiting">
              <span className="eyebrow">Invited, not joined yet</span>
              {waiting.map((i) => {
                const p = pipes.find((x) => x.token === i.token);
                return (
                  <div key={i.token} className="wait-row">
                    <span>{i.candidate} · <span className="muted">{i.role}</span></span>
                    <span className="row-actions">
                      <button className="ghost small" onClick={() => void navigator.clipboard.writeText(link(i))}>Copy link</button>
                      {p && <a className="ghost small btn" href={`#/pipelines/${p.id}`}>Questions</a>}
                      {p && <button className="ghost small" onClick={() => cancel(p.id, i.candidate)}>Cancel</button>}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="card">
          <h3>Job openings</h3>
          {cfg.roles.map((r) => {
            const [label, tone] = OPENING_STATUS[r.status] ?? [r.status, "tone-muted"];
            return (
              <a key={r.id} className="role-row" href={`#/roles/${r.id}`}>
                <span><b>{r.title}</b><span className="muted small"> · {r.company} · {r.location}</span></span>
                <span className={`chip ${tone}`}>{label}</span>
              </a>
            );
          })}
          <p className="muted small">Open a job to see how its interview is set up: what each requirement is checked
            with, the role-play, and what the interviewer may tell candidates.</p>
        </section>
      </div>

      <section className="card">
        <h3>Interviews by job</h3>
        {sessions.length === 0 && <p className="muted small">No interviews yet.</p>}
        {groupByRole(sessions).map(([roleTitle, byCand]) => (
          <details key={roleTitle} className="role-group" open>
            <summary>
              <span className="role-title">{roleTitle}</span>
              <span className="muted">{byCand.length} candidate{byCand.length > 1 ? "s" : ""}</span>
            </summary>
            {byCand.map(([cid, calls]) => (
              <div key={cid} className="cand-group">
                <div className="cand-head">
                  <span className="avatar small">{cleanName(calls[0].names.candidate ?? cid).slice(0, 1)}</span>
                  <b>{cleanName(calls[0].names.candidate ?? cid)}</b>
                </div>
                <table className="table sessions">
                  <tbody>
                    {calls.map((s) => {
                      const [out, tone] = OUTCOME[s.recommendation ?? ""] ?? ["—", "tone-muted"];
                      return (
                        <tr key={s.id} onClick={() => (location.hash = `#/sessions/${s.id}`)}>
                          <td>{new Date((s.started_at ?? s.created_at) * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}</td>
                          <td className="muted">{s.names.persona ?? s.persona_id}</td>
                          <td className="mono">{s.elapsed_s ? mmss(s.elapsed_s) : "—"}</td>
                          <td>{endLabel(s)}</td>
                          <td>{s.recommendation ? <span className={`chip ${tone}`}>{out}{s.total_25 != null ? ` · ${s.total_25}/25` : ""}</span>
                            : <span className="muted small">no brief yet</span>}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ))}
          </details>
        ))}
      </section>
    </div>
  );
}
