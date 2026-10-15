// Two diagrams for the System page: the whole recruiting pipeline (where this prototype sits), then the screener
// module itself. Plain boxes and arrows in HTML so they reflow on a narrow screen.

type Status = "built" | "contract" | "roadmap" | "existing";
const TAG: Record<Status, [string, string]> = {
  built: ["Built here", "tone-good"], contract: ["Interface defined", "tone-warn"],
  roadmap: ["Roadmap", "tone-muted"], existing: ["Cygnify already has", "tone-muted"],
};

function Box({ title, body, status, wide }: { title: string; body: string; status?: Status; wide?: boolean }) {
  return (
    <div className={`ab ${status ?? ""} ${wide ? "wide" : ""}`}>
      <b>{title}</b>
      <span>{body}</span>
      {status && <span className={`chip ${TAG[status][1]}`}>{TAG[status][0]}</span>}
    </div>
  );
}
const Arrow = ({ label }: { label?: string }) => <div className="aa">{label && <span>{label}</span>}</div>;

export function PipelineDiagram() {
  return (
    <div className="arch">
      <div className="arow">
        <Box title="1 · Job intake" body="Public JD → structured role.md (requirements, market, comp, process)" status="contract" />
        <Arrow label="role.md" />
        <Box title="2 · Find" body="Search candidate sources, match profiles to the role, rank a shortlist" status="existing" />
        <Arrow label="candidate.md" />
        <Box title="3 · Engage" body="Personalised outreach, reply handling, consent, scheduling the screen" status="roadmap" />
        <Arrow label="invite link" />
      </div>
      <div className="arow">
        <Box title="4 · Screen (voice)" body="Questions from JD × CV, recruiter-approved; 15-min structured voice screen" status="built" wide />
        <Arrow label="evidence" />
        <Box title="5 · Brief" body="Scores with quotes, CV check, must-haves, what to verify next; recruiter approves" status="built" />
        <Arrow label="approved brief" />
        <Box title="6 · Hiring manager / ATS" body="Final interview with the brief; record kept in the ATS" status="roadmap" />
      </div>
      <p className="muted small">Steps 2–3 are Cygnify's sourcing agent's job. This prototype defines what it hands over
        (docs/INPUT_SPEC.md) and builds steps 4–5, the part that needs a voice.</p>
    </div>
  );
}

export function ScreenerDiagram() {
  return (
    <div className="arch">
      <p className="eyebrow">Before the call — LangGraph pipeline with recruiter approvals (state saved in SQLite)</p>
      <div className="arow">
        <Box title="role.md" body="the job" />
        <Arrow />
        <Box title="Role step" body="Haiku proposes how to screen this job → code validates → recruiter approves once per job" />
        <Arrow />
        <Box title="candidate.md" body="the CV" />
        <Arrow />
        <Box title="Candidate step" body="Haiku reads CV against JD → personalised questions → code validates → recruiter approves → link" />
      </div>
      <p className="eyebrow">During the call — one LangGraph run per candidate turn</p>
      <div className="arow">
        <Box title="Candidate (browser)" body="microphone and speaker" />
        <Arrow label="audio" />
        <Box title="gpt-live-1" body="listens and speaks, full duplex; yields to interruptions; decides nothing" />
        <Arrow label="transcript" />
        <Box title="Orchestrator" body="detects the end of a turn, merges fragments, tracks what was actually heard" />
      </div>
      <div className="arow brain">
        <Box title="prepare" body="build the context for this turn" />
        <Arrow />
        <Box title="interpret · Haiku" body="turn type, evidence with quotes, follow-up, red flags" />
        <Arrow />
        <Box title="check · code" body="trust boundary: quotes must be the candidate's words" />
        <Arrow />
        <Box title="record → advance · code" body="evidence ledger; card engine picks the next question; quota and CV checks" />
        <Arrow />
        <Box title="speak" body="next line back to gpt-live-1 (paraphrase, or word for word)" />
      </div>
      <div className="arow mem">
        <Box title="Context (what the models see)" body="Voice: persona and rules + the current question. Brain: AE rules, this stage's skill and rubric anchors, market and job layers (cached) + this turn: question, missing fields, captured facts, last 6 lines" wide />
        <Box title="Memory (what the system keeps)" body="Session state in code: stage, question cards, evidence ledger with quotes, flags, strikes. After the call: transcript, recording, brief in the store. Across days: pipeline threads in the checkpointer" wide />
      </div>
      <p className="eyebrow">After the call</p>
      <div className="arow">
        <Box title="Brief" body="Haiku scores 5 competencies with quotes → code keeps only quotes found in the transcript, computes /25 and the recommendation" />
        <Arrow />
        <Box title="Recruiter" body="approves or rejects the brief (LangGraph interrupt)" />
      </div>
    </div>
  );
}
