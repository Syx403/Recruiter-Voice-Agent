# Architecture

The screener covers one step of a recruiting pipeline: the first-round voice screen and the brief that comes out
of it. Sourcing, outreach and document extraction are upstream and not built; they hand over two Markdown
documents (`role.md`, `candidate.md`, see [INPUT_SPEC.md](INPUT_SPEC.md)).

## 1. Layers of content

| Layer | Content | Changes with | Where |
|---|---|---|---|
| Generic AE | stages, question cards and their fields, five competencies with 1/3/5 anchors, skill SOPs | never per role | `content/stages.yaml`, `content/rubric.yaml`, `content/skills/` |
| Market | currency and scale words, financial year, how compensation is asked, work eligibility, notice norms | country | `content/markets/` |
| Role | `role.md` → screen configuration (approved once per job) | each job | `content/roles/<id>/` |
| Candidate | `candidate.md` × the role → question plan (approved per candidate) | each candidate | pipeline database |

AE hiring knowledge (SDR seats are not closing seats, ARR vs TCV, new-logo share, MEDDPICC, blind recall of
numbers) is written as rules for every role. Nothing is written for one test role or CV.

## 2. Pipelines

Both are durable LangGraph graphs with a SQLite checkpointer (`alt/pipeline/graph.py`, `runs/alt/pipeline.sqlite`).
⏸ marks an `interrupt()` where a recruiter, or the end of the call, resumes the thread.

```
Role:      role_intake (parse role.md) → role_compile (Haiku) → role_validate (code) ─(rule breaks, once)→ role_repair
           → role_review ⏸ (recruiter) → role_publish (content/roles/<id>/screen.yaml)

Candidate: intake (parse candidate.md) → plan (Haiku: CV × JD) → validate (code) → approve_plan ⏸ (recruiter)
           → invite (one-time link) → await_call ⏸ (resumed when the call ends) → brief (score → ground quotes →
           repair once → compute in code) → review ⏸ (recruiter) → finalise
```

A document the parser rejects stops the pipeline before any model call.

### Role step (`alt/pipeline/screen.py`)

The model proposes only judgement calls; everything else comes from `role.md` in code.

| Proposed by the model | Validated in code |
|---|---|
| competency weights + rationale | each 0.05–0.40, sum 1.00, else rubric defaults |
| the role-play objection (buyer, concern, push, what a strong answer does) | lengths; required parts |
| how each [must] requirement is checked: by named card fields, from the CV, or by a role question | every [must] id once; fields exist |
| at most two role questions, asked to every candidate of the role | open, ≤ 35 words, stands alone, no protected attribute, serves a requirement the cards do not cover |
| cycle and deal-size ranges, labelled `jd` / `inferred` / `unknown` | an inferred range is never treated as the employer's |
| FAQ answers for candidate questions | every answer grounded in `role.md` |

### Candidate step (`alt/pipeline/plan.py`)

Output: a short summary, requirement fit (`shown` / `partial` / `not_shown` / `unclear`, quoting the CV),
personalised wording for some cards, and at most three extra probes. Code validates it:

- **Blind recall.** No question states an amount, a percentage or any other number from a CV claim. Code compares
  the candidate's spoken figures with the claims after the answer; only then may a neutral follow-up quote the CV.
- Open questions only, one question mark, ≤ 35 words, stand alone, no protected attribute, no name that is not in
  the inputs.
- Only personalisable cards change, and only their wording: the fields a card captures are the same for every
  candidate of a role.
- "N+ years" requirements are decided by code from the CV's dates, not by the model.

## 3. The call

```
browser (candidate) ⇄ alt/app.py websocket ⇄ alt/orchestrator.py ⇄ gpt-live-1 (alt/live_adapter.py)
                                                   │
                                                   └─ per candidate turn → alt/brain/graph.py (LangGraph + Haiku)
```

**gpt-live-1** is full duplex: it hears the candidate directly, takes turns, handles interruptions and keeps
talking naturally while the brain thinks. It was chosen over a cascade (separate transcription model + text
controller + voice) because the transcription model mis-heard non-native accents, and every decision downstream
inherited the error; gpt-live-1 understood them.

**The orchestrator** (`alt/orchestrator.py`) segments the transcript stream into candidate turns, waits for the end
of a turn (longer when the sentence sounds unfinished or the candidate asks for time), and sends the brain's lines
to the voice model. A brain result is provisional until its line is heard: if the candidate keeps talking first,
the turn is rolled back and decided again on the merged answer. The interviewer opens the call without waiting
for the candidate.

**The brain** (`alt/brain/graph.py`) runs once per candidate turn:

```
START → prepare → interpret ─(error/timeout: degrade)─→ check ─route─┬→ record ─┬→ advance ─┐
                                                                     │          └→ close ───┤
                                                                     ├→ clarify ────────────┤
                                                                     ├→ redirect ───────────┤
                                                                     ├→ defer ──────────────┤→ speak → END
                                                                     ├→ close ──────────────┤
                                                                     ├→ wait ───────────────┤
                                                                     └→ nudge ─(or record)──┘
```

`wait` holds when the answer sounds unfinished; `nudge` offers the question again once when the candidate stopped
mid-thought and went silent.

- `interpret` is the only LLM node (Claude Haiku 5.5, strict JSON, deadline in `alt/config.yaml`). It judges the answer, extracts
  evidence, drafts a contextual follow-up, answers role questions from the brief only, and raises red flags.
- `check` is the trust boundary: code validates everything the model returned. Routers are pure functions.
- The card engine (`backend/session.py`) decides what is asked next. A card is done when its fields are captured
  or its probes are spent.
- Engagement is judged as `engaged` / `weak_but_genuine` / `low_effort` / `not_engaging`. A sincere weak answer is
  recorded as weak; a candidate who will not engage is brought back, warned, then the call ends.

### Number checks

1. **Arithmetic consistency** (`backend/tools/consistency.py`): once quota, attainment, average deal and deal count
   are captured, code checks that `quota × attainment ≈ average deal × deals`. If not, a neutral clarifying question
   is inserted. Spoken amounts ("four crore", "1.2 million") are normalised in code (`backend/normalise.py`).
2. **CV check** (`alt/pipeline/cvcheck.py`): stated figures are compared with the CV's claims; a mismatch inserts a
   neutral follow-up that may then quote the CV.
3. **Expected ranges**: when the screen configuration has a range from the JD, out-of-range values produce a note
   in the check result. Not yet used for a follow-up question, and empty for both demo roles (see the README's
   limitations).

## 4. The brief (`alt/pipeline/brief.py`)

Haiku scores each competency 1–5 with quotes from the transcript. Code then:

- drops any score whose quote is not found in the transcript;
- caps performance credibility at 3 if the arithmetic check was inconsistent;
- computes a weighted total out of 25 with the role's weights;
- sets the recommendation: `recommend` (≥ 20), `conditional` (≥ 16), `not_recommended`, or `needs_verification`
  when key evidence is missing (`content/rubric.yaml`);
- lists knockouts for the recruiter, never applying them automatically.

The recruiter approves or overrides the brief.

## 5. Storage and cost

- `runs/alt/screener.sqlite`: sessions, evidence, transcript, flags. `runs/alt/pipeline.sqlite`: pipeline threads.
  `runs/<session_id>/`: event log, recordings, brief. All local and gitignored.
- `runs/costs.jsonl`: every model call's usage and cost. Caps in `alt/config.yaml` (per session and in total).
- Typical cost: about USD 0.05 per minute for the voice model and about USD 0.001 per minute for Haiku.

## 6. Shared core and the earlier loop

`backend/` holds the shared core used by the app: session state and card engine, evidence, coverage, quota and
stage tools, the number normaliser, the store, exports, recording and the cost ledger. It also contains an earlier
directive-driven loop (`backend/orchestrator.py`, `backend/controller.py`, `backend/adapters/openai_realtime.py`:
gpt-realtime-2.1 voicing lines chosen by a text controller). The app does not serve it, but imports helpers from
those modules, so they stay.
