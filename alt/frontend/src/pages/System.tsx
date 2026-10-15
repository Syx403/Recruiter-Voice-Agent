import { useEffect, useState } from "react";
import { PipelineDiagram, ScreenerDiagram } from "../components/Architecture";
import { ModelCards, Stat } from "../components/Parts";
import { AppConfig, getConfig, listPipelines, Pipeline } from "../lib/api";

// The technical view, kept off the recruiter's console: models, architecture, pipelines and spend.

const LAYERS: [string, string][] = [
  ["Inputs", "role.md (a structured job description) and candidate.md (a structured CV), produced upstream by a JD extractor and the sourcing agent (docs/INPUT_SPEC.md)."],
  ["Role step", "Claude Haiku 5.5 proposes the job's interview setup: weights, the role-play objection, how each must-have requirement is checked, up to two job-specific questions. Code validates it; the recruiter approves it once for every candidate of the job."],
  ["Candidate step", "Claude Haiku 5.5 reads the CV against the job description: requirement fit, personalised wording, up to three extra probes. Code enforces blind recall (no CV number is read out before the candidate states it) and the same fields for every candidate."],
  ["The call", "gpt-live-1 talks and listens (full duplex). A LangGraph brain runs once per candidate turn: Claude Haiku 5.5 judges the answer, code decides what is asked next, records evidence and checks numbers."],
  ["The brief", "Claude Haiku 5.5 scores with quotes; code grounds the quotes, computes the total and the recommendation; the recruiter decides."],
];

export default function System() {
  const [cfg, setCfg] = useState<AppConfig>();
  const [pipes, setPipes] = useState<Pipeline[]>([]);
  useEffect(() => {
    getConfig().then(setCfg);
    listPipelines().then(setPipes);
  }, []);
  if (!cfg) return <div className="page"><p className="notice">Loading…</p></div>;
  return (
    <div className="page wide">
      <header className="masthead">
        <p className="eyebrow"><a href="#/">← Console</a> · System</p>
        <h1>System &amp; architecture</h1>
        <p className="lede">How the screener works, which models it uses, and what it has spent.</p>
      </header>
      <section className="card">
        <h3>1 · The recruiting pipeline, and where this prototype sits</h3>
        <PipelineDiagram />
      </section>
      <section className="card">
        <h3>2 · The screening agent (built here)</h3>
        <ScreenerDiagram />
      </section>
      <div className="role-grid">
        <section className="card">
          <h3>Layers in one line each</h3>
          <dl className="call-facts">
            {LAYERS.flatMap(([k, v]) => [<dt key={`t${k}`}>{k}</dt>, <dd key={`d${k}`}>{v}</dd>])}
          </dl>
        </section>
        <div className="side-stack">
          <section className="card"><h3>Models</h3><ModelCards voice={cfg.voice_info} /></section>
          <section className="card">
            <h3>API spend (this prototype)</h3>
            <Stat label="Spent" value={`$${cfg.budget.spent.toFixed(2)}`} sub={` / $${cfg.budget.total}`}
                  ratio={cfg.budget.spent / cfg.budget.total} />
          </section>
        </div>
      </div>
      <section className="card">
        <h3>Candidate pipelines</h3>
        <table className="table small">
          <tbody>
            {pipes.map((p) => (
              <tr key={p.id} onClick={() => (location.hash = `#/pipelines/${p.id}`)}>
                <td className="mono">{p.id}</td><td>{p.candidate}</td><td className="muted">{p.role}</td>
                <td>{p.status}</td><td className="mono muted">{p.session_id ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
