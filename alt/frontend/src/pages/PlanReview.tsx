import { useCallback, useEffect, useState } from "react";
import { approvePlan, cancelPipeline, getPipeline, Pipeline, stageLabel } from "../lib/api";

// The candidate step of the question generator: candidate.md × role.md → a question plan. The recruiter
// reviews it; the interview link is issued only after approval (LangGraph interrupt in the candidate pipeline).

const STATUS_LABEL: Record<string, string> = {
  shown: "shown", partial: "partial", not_shown: "not shown", unclear: "unclear",
};

export default function PlanReview({ id }: { id: string }) {
  const [p, setP] = useState<Pipeline>();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [copied, setCopied] = useState(false);
  const load = useCallback(() => getPipeline(id).then(setP), [id]);
  useEffect(() => {
    void load();
    const h = window.setInterval(load, 4000);
    return () => window.clearInterval(h);
  }, [load]);
  if (!p) return <div className="page"><p className="notice">Loading…</p></div>;

  const plan = p.plan;
  const waiting = p.next?.includes("approve_plan");
  const link = p.url ? `${location.origin}${p.url}` : "";
  const claims = Object.fromEntries((plan?.claims ?? []).map((c) => [c.id, c]));
  async function cancel() {
    if (!window.confirm("Cancel this interview? An issued link will stop working.")) return;
    try { setP(await cancelPipeline(id)); } catch (e) { setErr((e as Error).message); }
  }
  const cancellable = ["plan_review", "invited"].includes(p.status) && p.invite_status !== "in_call";
  async function approve() {
    setBusy(true);
    setErr("");
    try { setP(await approvePlan(id)); } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  }

  return (
    <div className="page wide">
      <header className="masthead">
        <p className="eyebrow"><a href="#/">← Console</a> · Interview questions</p>
        <h1>{p.candidate} <span className="muted">for</span> {p.role}</h1>
        <p className="lede">{plan?.summary || p.parsed?.headline}</p>
      </header>

      {(p.input_issues ?? []).length > 0 && (
        <section className="card"><h3>The documents could not be used</h3>
          {p.input_issues!.map((i, k) => <p key={k} className="error small">✗ {i}</p>)}</section>
      )}

      <div className="role-grid">
        <section className="card">
          <h3>CV against the job's must-haves</h3>
          <p className="muted small">Read from the candidate's CV and the job description. Years of experience are counted
            from the CV's dates; the CV's own words are quoted as the basis.</p>
          <table className="table small">
            <tbody>
              {(plan?.requirement_fit ?? []).map((r) => (
                <tr key={r.requirement_id}>
                  <td className="mono">{r.requirement_id}</td>
                  <td><span className={`st ${r.status}`}>{STATUS_LABEL[r.status] ?? r.status}</span></td>
                  <td>{r.note}{r.cv_basis && <div className="q small">“{r.cv_basis}”</div>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="eyebrow">CV figures the interview will check (never read out before the candidate gives them)</p>
          <table className="table small">
            <tbody>
              {(plan?.claims ?? []).filter((c) => c.check === "numeric").map((c) => (
                <tr key={c.id}><td className="mono">{c.id}</td><td className="mono">{c.field}</td><td>{c.text}</td></tr>
              ))}
            </tbody>
          </table>
          {(p.parsed?.warnings ?? []).length > 0 && (
            <details><summary className="muted small">Input warnings ({p.parsed!.warnings!.length})</summary>
              <ul className="small muted">{p.parsed!.warnings!.map((w, i) => <li key={i}>{w}</li>)}</ul></details>
          )}
        </section>

        <section className="card">
          <h3>Questions for this candidate</h3>
          <p className="muted small">Every candidate for this job is asked about the same things. For this candidate,
            some questions are anchored on their own experience, and up to three extra questions follow up on gaps.</p>
          <p className="eyebrow">Questions anchored on this CV</p>
          {(plan?.cards ?? []).length === 0 && <p className="muted small">None: the standard wording is used.</p>}
          {(plan?.cards ?? []).map((c) => (
            <div key={c.card} className="qcard">
              <span className="mono small">{c.card}{c.verifies.length ? ` · checks ${c.verifies.join(", ")}` : ""}</span>
              <p>“{c.primary}”</p><p className="muted small">{c.rationale}</p>
            </div>
          ))}
          <p className="eyebrow">Extra follow-up questions</p>
          {(plan?.extra_probes ?? []).length === 0 && <p className="muted small">None.</p>}
          {(plan?.extra_probes ?? []).map((x, i) => (
            <div key={i} className="qcard">
              <span className="mono small">{stageLabel(x.stage)} · {x.field}{x.requirement_id ? ` · ${x.requirement_id}` : ""}
                {x.verifies.map((v) => ` · ${claims[v]?.field ?? v}`)}</span>
              <p>“{x.question}”</p><p className="muted small">{x.rationale}</p>
            </div>
          ))}
          {(p.plan_issues ?? []).length > 0 && (
            <details><summary className="muted small">Automatic checks ({p.plan_issues!.length} questions removed)</summary>
              <ul className="small muted">{p.plan_issues!.map((x, i) => <li key={i}>{x}</li>)}</ul></details>
          )}
          {err && <p className="error">{err}</p>}
          {waiting ? (
            <div className="decide-row">
              <button className="primary" disabled={busy} onClick={approve}>{busy ? "Issuing the link…" : "Approve and create the interview link"}</button>
              <button className="ghost" disabled={busy} onClick={cancel}>Cancel interview</button>
            </div>
          ) : p.status === "cancelled" ? (
            <p className="notice small">This interview was cancelled.</p>
          ) : p.url ? (
            <div className="created">
              <span className="eyebrow">Interview link ({p.invite_status})</span>
              <div className="link-row">
                <input readOnly value={link} onFocus={(e) => e.target.select()} />
                <button className="ghost" onClick={() => { void navigator.clipboard.writeText(link); setCopied(true); }}>
                  {copied ? "Copied" : "Copy"}</button>
              </div>
              {cancellable && <button className="ghost small" onClick={cancel}>Cancel interview</button>}
              <p className="notice small">It works once. <a href={p.url} target="_blank">Open it as the candidate (test) ↗</a>
                {p.session_id && <> · <a href={`#/sessions/${p.session_id}`}>Open the session →</a></>}</p>
            </div>
          ) : <p className="muted small">Status: {p.status}</p>}
        </section>
      </div>
    </div>
  );
}
