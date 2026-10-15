# Guide for coding agents

Read in this order:
1. `README.md`: what this is and how to run it.
2. `docs/ARCHITECTURE.md`: the pipelines, the per-turn brain, the checks, and where each lives in code.
3. `docs/INPUT_SPEC.md`: the two input documents (`role.md`, `candidate.md`).

## Commands

```bash
uv sync --group alt
uv run --group alt pytest -q                              # no API calls; run after every change
(cd alt/frontend && npm ci && npm run build)              # the server serves alt/frontend/dist; rebuild after UI edits
uv run --group alt uvicorn alt.app:app --port 8100
```

Anything that calls a model costs money: the evals in `alt/eval/` (cost noted in each file's docstring), the
"Prepare interview questions" and brief steps, and every voice call (about USD 0.05 per minute). State the cost
before running one. Usage is logged to `runs/costs.jsonl`; the cap is in `alt/config.yaml`.

## Where things are

- Call flow: `alt/orchestrator.py` (turn segmentation, holds, interruptions, sending lines to the voice model).
- Per-turn brain: `alt/brain/graph.py` (LangGraph nodes and routers) and `alt/brain/llm.py` (the Haiku prompt and
  its context).
- Pipelines before and after the call: `alt/pipeline/` (`inputs.py` parsers, `screen.py` role step, `plan.py`
  candidate step, `cvcheck.py`, `brief.py`, `graph.py` the durable LangGraph pipelines).
- API: `alt/app.py`. UI: `alt/frontend/src/pages/`.
- Shared core: `backend/session.py` (card engine), `backend/tools/` (evidence, quota consistency, coverage, stages),
  `backend/normalise.py` (spoken numbers), `backend/store.py` (SQLite).
- Interview content: `content/stages.yaml` (stages, cards, fields), `content/rubric.yaml`, `content/skills/`,
  `content/markets/`, `content/roles/<id>/`.

## Rules that must not break

- Synthetic or public data only. Never add real candidates' personal data.
- Keys stay server-side (environment or macOS Keychain via `backend/keystore.py`). Never print, log or commit them.
- Code decides what is asked; the model decides how to say it. The same cards and fields for every candidate of a
  role; a question plan may change wording, never which fields a card captures.
- Blind recall: no question reads a CV number before the candidate has stated it.
- No protected attributes (age, gender, family, religion, caste, nationality, health, …) in inputs, prompts or
  questions; the input parsers remove them.
- The model never computes scores, totals or recommendations; code does (`alt/pipeline/brief.py`), and a recruiter
  makes every decision.
- Do not tune prompts or validators to the test candidates or roles; examples in prompts come from outside the
  test set.
- Never commit `runs/` (recordings, logs, databases, cost ledger).
