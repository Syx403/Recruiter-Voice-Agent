import { useCallback, useEffect, useState } from "react";
import { compileRole, decideRole, getRole, RoleDetail, Screen } from "../lib/api";

// The role step of the question generator: role.md (the structured JD) → a screen configuration that
// every candidate for the role shares. The model proposes, code validates, the recruiter approves here.

const COMP_LABEL: Record<string, string> = {
  deal_execution: "Deal execution", performance_credibility: "Performance credibility",
  technical_credibility: "Technical credibility", outcome_orientation: "Outcome orientation",
  ownership_learning: "Ownership & learning",
};
const CHECK_LABEL: Record<string, string> = {
  screen: "the standard cards", cv: "the CV", question: "a role question",
};

export default function Role({ id }: { id: string }) {
  const [r, setR] = useState<RoleDetail>();
  const [busy, setBusy] = useState("");
  const [note, setNote] = useState("");
  const [err, setErr] = useState("");
  const load = useCallback(() => getRole(id).then(setR), [id]);
  useEffect(() => { void load(); }, [load]);
  if (!r) return <div className="page"><p className="notice">Loading…</p></div>;

  const waiting = r.next.includes("role_review");
  const shown: Screen | null = (waiting ? r.draft : r.approved ?? r.draft) ?? null;
  const parsed = r.parsed;
  async function run(label: string, fn: () => Promise<RoleDetail>) {
    setBusy(label);
    setErr("");
    try { setR(await fn()); } catch (e) { setErr((e as Error).message); }
    setBusy("");
  }

  return (
    <div className="page wide">
      <header className="masthead">
        <p className="eyebrow"><a href="#/">← Console</a> · Job opening · interview setup</p>
        <h1>{parsed?.title ?? id}</h1>
        <p className="lede">
          {String(parsed?.company ?? "")} · {String(parsed?.location ?? "")} · market preset <b>{String(parsed?.market ?? "")}</b>
        </p>
      </header>

      <div className="role-grid">
        <section className="card">
          <h3>Job description</h3>
          <p className="muted small">Structured from the public posting. Source:{" "}
            {shown?.source?.source_url ? <a href={shown.source.source_url} target="_blank">public posting ↗</a> : "—"}</p>
          {(r.input_issues ?? []).map((i, k) => <p key={k} className="error small">✗ {i}</p>)}
          {(parsed?.warnings ?? []).map((w, k) => <p key={k} className="muted small">⚠ {w}</p>)}
          <p className="eyebrow">Requirements</p>
          <ul className="reqs">
            {(parsed?.requirements ?? []).map((q) => (
              <li key={q.id}><span className={`tag ${q.kind}`}>{q.id}</span> {q.text}</li>
            ))}
          </ul>
          <details><summary className="muted small">Structured job description (role.md)</summary><pre className="md">{r.role_md}</pre></details>
        </section>

        <section className="card">
          <div className="row-between">
            <h3>Interview setup {waiting ? "· needs your review" : r.approved ? "· in use" : "· draft"}</h3>
            <button className="ghost small" disabled={!!busy} onClick={() => run("compile", () => compileRole(id))}>
              {busy === "compile" ? "Generating…" : shown ? "Regenerate" : "Generate"}
            </button>
          </div>
          {r.error && <p className="error">{r.error}</p>}
          {err && <p className="error">{err}</p>}
          {!shown && <p className="notice small">No interview setup yet. Generate one from the job description.</p>}
          {shown && <ScreenView s={shown} issues={waiting ? r.issues ?? [] : shown.validation_issues ?? []} />}
          {waiting && (
            <div className="decide">
              <textarea placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
              <div className="decide-row">
                <button className="primary" disabled={!!busy} onClick={() => run("approve", () => decideRole(id, "approve", note))}>
                  Use this setup for every candidate of this job
                </button>
                <button className="ghost" disabled={!!busy} onClick={() => run("reject", () => decideRole(id, "reject", note))}>
                  Reject
                </button>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function ScreenView({ s, issues }: { s: Screen; issues: string[] }) {
  return (
    <div className="screen">
      <p className="eyebrow">How each must-have requirement is checked</p>
      <table className="table small reqcheck">
        <tbody>
          {s.requirements.map((q) => (
            <tr key={q.id}>
              <td className="mono">{q.id}</td>
              <td>{q.text}<div className="muted small">Look for: {q.look_for}</div></td>
              <td className="check-col">{CHECK_LABEL[q.check]}{q.years_from_cv ? " · years counted by code" : ""}
                {q.evidence_fields.length > 0 && <div className="muted small mono">{q.evidence_fields.join(", ")}</div>}
                {(q.questions ?? []).length > 0 && <div className="muted small">+ {q.questions!.join(", ")}</div>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="eyebrow">Role questions (asked to every candidate)</p>
      {s.role_questions.length === 0
        ? <p className="muted small">None: the standard AE cards already cover every must-have requirement.</p>
        : s.role_questions.map((q) => (
          <div key={q.field} className="qcard"><span className="mono small">{q.field} · {q.requirement_ids.join(", ")}</span>
            <p>“{q.question}”</p><p className="muted small">Strong answer: {q.look_for}</p></div>
        ))}

      <p className="eyebrow">Role-play objection</p>
      <div className="qcard">
        <p className="small"><b>{s.objection_scenario.role_played}</b></p>
        <p>“{s.objection_scenario.concern}”</p>
        <p className="muted small">If thin: “{s.objection_scenario.probe_if_weak}”</p>
        <p className="muted small">Strong answer: {s.objection_scenario.strong_answer}</p>
      </div>

      <p className="eyebrow">What matters most for this job (weights)</p>
      {Object.entries(s.competency_weights).map(([k, v]) => (
        <div key={k} className="wrow"><span>{COMP_LABEL[k] ?? k}</span>
          <span className="bar"><span style={{ width: `${v * 250}%` }} /></span><span className="mono">{v.toFixed(2)}</span></div>
      ))}
      <p className="muted small">{s.weights_rationale}</p>

      <p className="eyebrow">Job facts the answers are compared with</p>
      <p className="small">
        {String(s.fit.motion)} · {String(s.fit.segment)} · {s.arrangement_phrase} · cycle{" "}
        {Array.isArray(s.fit.sales_cycle_days) ? `${(s.fit.sales_cycle_days as number[]).join("–")} days` : "not stated"}
        {" "}({String(s.fit.sales_cycle_basis)}) · notice ≤ {String(s.fit.max_notice_days)} days ·
        {" "}work eligibility {s.fit.work_eligibility_asked ? "asked" : "not asked"} · pay band: {String(s.fit.ote ?? "not stated")} (never spoken)
      </p>

      <p className="eyebrow">What the interviewer may tell candidates</p>
      <ul className="small">{s.faq.map((f, i) => <li key={i}><b>{f.q}</b> — {f.a}</li>)}</ul>
      <p className="eyebrow">Next-round suggestions</p>
      <ul className="small">{s.next_round_suggestions.map((x, i) => <li key={i}>{x}</li>)}</ul>
      {issues.length > 0 && (
        <details><summary className="muted small">Automatic checks ({issues.length} adjustments)</summary>
          <ul className="small muted">{issues.map((x, i) => <li key={i}>{x}</li>)}</ul></details>
      )}
    </div>
  );
}
