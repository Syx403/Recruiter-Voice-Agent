export type Option = { id: string; title?: string; name?: string };

export type VoiceInfo = {
  interviewer?: {
    model: string; reasoning_effort: string; voice: string;
    turn_detection?: { type: string; eagerness?: string } | null;
    noise_reduction?: string | null; barge_in?: string | null;
  };
  transcriber?: { model: string; language?: string | null };
  brain?: { model: string; graph: string; deadline_s: number };
};

/** One node of a brain run (LangGraph), as streamed in a note's trace. */
export type TraceStep = { node: string; ms: number | null; status?: string; kind?: string; llm_ms?: number;
  note?: string | null; next?: string; recorded?: number; rejected?: number; dropped?: number; end?: string; error?: string };
/** Per candidate turn: how long the candidate waited for the first sound and for the next line. */
export type TurnMetric = { turn: number; kind: string; brain_ms: number; llm_ms: number | null; degraded: boolean;
  sent: boolean; first_sound_s: number | null; next_line_s: number | null };

export type RoleOption = { id: string; title: string; company?: string; location?: string; market?: string;
  status: "approved" | "draft" | "review" | "none" | "input_error"; thread?: string | null; issues?: string[] };
export type CandidateOption = { id: string; name: string; headline?: string; location?: string; issues?: string[] };

export type AppConfig = {
  defaults: { persona: string; role: string };
  voice: Record<string, unknown>;
  voice_info: VoiceInfo;
  roles: RoleOption[];
  personas: Option[];
  candidates: CandidateOption[];
  budget: { total: number; spent: number };
};

export type Names = { role?: string; company?: string; persona?: string; interviewer?: string; candidate?: string };

export type SessionRow = {
  id: string; candidate_id: string; role_id: string; persona_id: string; model: string; status: string;
  cost_usd: number; created_at: number; started_at: number | null; end_reason: string | null;
  names: Names; elapsed_s: number | null; recommendation?: string | null; total_25?: number | null;
};

export type Entry = {
  id: string; kind: "agent" | "candidate" | "noise" | "tool" | "stage" | "note"; t: number; stage?: string; text: string;
  intent?: string; intent_reason?: string; role?: "bridge" | "directed"; trace?: TraceStep[];
  dropped?: string[]; problems?: string[]; followup_by_brain?: boolean; brain_ms?: number;
  first_sound_s?: number | null; next_line_s?: number | null; specific?: boolean;
  partial?: boolean; interrupted?: boolean; heard_ms?: number; audio_ms?: number; card?: string;
  name?: string; arguments?: Record<string, unknown>; output?: Record<string, unknown>;
};

export type Evidence = { t: number; stage: string; card: string | null; dimension: string; field: string;
  value: string; quote: string; declined: boolean; normalised: unknown };
export type Flag = { t: number; type: string; detail: string; source: string };
export type StageMark = { stage: string; t: number };

export type CandidateProfile = { id: string; name?: string; cv_summary?: string; locale?: string; synthetic?: boolean };
export type RoleProfile = { id: string; title?: string; company?: string; location?: string; context?: string };

export type SessionRecord = {
  session: SessionRow & { model: string };
  candidate: CandidateProfile; role: RoleProfile;
  names: Names; voice: VoiceInfo; elapsed_s: number | null; stage_log: StageMark[]; order: string[];
  stage_targets: Record<string, number>; cap_usd: number | null; recording: boolean;
  timeline: Entry[]; evidence: Evidence[]; flags: Flag[]; consistency: Record<string, unknown>[];
  overrides: unknown[]; refusals: unknown[]; rejections: unknown[];
  turns?: TurnMetric[]; cost_split?: { voice: number; brain: number } | null;
  pipeline?: Pipeline | null; layout?: Layout | null;
};

export type BriefCompetency = { id: string; score: number; confidence: string; evidence: string[]; rationale: string };
export type Brief = {
  summary: string; competencies: BriefCompetency[]; fit_checks: { id: string; status: string; note: string }[];
  key_evidence: string[]; risks: string[]; next_round_focus: string[]; missing_information: string[];
};
export type CvRow = { claim: string; field: string; cv: string; window: string; said: string | null;
  status: "match" | "differs" | "not_obtained" | "not_comparable" };
export type ReqFit = { requirement_id: string; status: string; cv_basis: string; note: string; by?: string };
export type Computed = {
  scores: Record<string, number | null>; total_25: number | null; recommendation: string; not_observed: string[];
  knockouts_to_review: string[]; notes: string[]; weights: Record<string, number>;
  cv_check?: CvRow[]; cv_differs?: CvRow[]; requirement_fit?: ReqFit[];
};
export type Claim = { id: string; field: string; value: string | number; unit: string; window: string; text: string;
  check: string };
export type Plan = {
  candidate_id: string; role_id: string; summary: string; claims: Claim[]; requirement_fit: ReqFit[];
  cards: { card: string; primary: string; rationale: string; verifies: string[]; fields?: string[] }[];
  extra_probes: { stage: string; field: string; question: string; requirement_id: string; verifies: string[];
    rationale: string }[];
  source: string;
};
export type Pipeline = {
  id: string; status: string; token: string | null; session_id: string | null; next?: string[];
  candidate?: string; candidate_id?: string; role?: string; role_id?: string; url?: string | null;
  invite_status?: string | null; updated_at?: number;
  plan?: Plan | null; plan_issues?: string[] | null; input_issues?: string[] | null;
  parsed?: { name: string; headline: string; warnings?: string[]; experience?: { title: string; company: string;
    months: number | null; closing: boolean; start: string | null; end: string | null }[] } | null;
  brief?: Brief | null; computed?: Computed | null; brief_error?: string | null;
  decision?: { decision: string; note: string; t: number } | null; unmatched?: string[] | null;
};

export type Requirement = { id: string; text: string; check: "screen" | "cv" | "question"; evidence_fields: string[];
  questions?: string[]; look_for: string; years_from_cv?: boolean };
export type Screen = {
  role_id: string; title: string; company: string; location: string; arrangement_phrase: string; market_id: string;
  country: string; currency: string; buyer_personas: string[]; products: string[]; team_leadership: string;
  fit: Record<string, unknown>; expected_ranges: Record<string, unknown>; requirements: Requirement[];
  nice_to_have: { id: string; text: string }[]; role_questions: { question: string; requirement_ids: string[];
    look_for: string; field: string }[];
  competency_weights: Record<string, number>; weights_rationale: string;
  objection_scenario: { role_played: string; concern: string; probe_if_weak: string; strong_answer: string };
  objection_rationale: string; faq: { q: string; a: string }[]; next_round_suggestions: string[];
  validation_issues?: string[]; approval?: { by: string; at: number; note: string };
  source?: { role_md_sha: string; source_url: string; compiled_at: number; model: string };
};
export type RoleDetail = {
  id: string; role_md: string; parsed: (Record<string, unknown> & { warnings?: string[]; title: string;
    requirements: { id: string; kind: string; text: string }[] }) | null;
  input_issues?: string[] | null; approved: Screen | null; draft: Screen | null; issues?: string[] | null;
  error?: string | null; thread: string | null; status: string; next: string[];
};

export async function getRole(id: string): Promise<RoleDetail> {
  return (await fetch(`/api/roles/${id}`)).json();
}
export async function compileRole(id: string): Promise<RoleDetail> {
  return (await fetch(`/api/roles/${id}/compile`, { method: "POST" })).json();
}
export async function decideRole(id: string, decision: string, note: string): Promise<RoleDetail> {
  const r = await fetch(`/api/roles/${id}/decision`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision, note }),
  });
  if (!r.ok) throw new Error((await r.json()).detail ?? "failed");
  return r.json();
}
export async function getPipeline(id: string): Promise<Pipeline> {
  return (await fetch(`/api/pipelines/${id}`)).json();
}
export async function listPipelines(): Promise<Pipeline[]> {
  return (await fetch("/api/pipelines")).json();
}
export async function cancelPipeline(id: string): Promise<Pipeline> {
  const r = await fetch(`/api/pipelines/${id}/cancel`, { method: "POST" });
  if (!r.ok) throw new Error((await r.json()).detail ?? "failed");
  return r.json();
}
export async function approvePlan(id: string): Promise<Pipeline> {
  const r = await fetch(`/api/pipelines/${id}/plan`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: "{}" });
  if (!r.ok) throw new Error((await r.json()).detail ?? "failed");
  return r.json();
}

export type CardLayout = { id: string; variant: string | null; personalised: boolean; extra: boolean;
  role_question: boolean; text: string; standard: string | null; verifies: string[]; requirements: string[] };
export type Layout = { seed: number; order: string[]; market: string | null; role_id: string | null;
  plan_summary: string; cards: Record<string, CardLayout[]> };

export async function writeBrief(sid: string): Promise<Pipeline> {
  return (await fetch(`/api/sessions/${sid}/brief`, { method: "POST" })).json();
}

export async function decideBrief(pid: string, decision: string, note: string): Promise<Pipeline> {
  const r = await fetch(`/api/pipelines/${pid}/decision`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision, note }),
  });
  return r.json();
}

export type LiveState = {
  id: string; stage: string; order: string[]; stage_log: StageMark[]; stage_targets: Record<string, number>;
  status: string; cost_usd: number; cap_usd: number; names: Names; voice: VoiceInfo;
  card: { id: string; primary: string; probes_used: number; max_probes: number } | null;
  clock: { elapsed_s: number; remaining_s: number; wrap_up: boolean; must_close: boolean };
  ledger: Evidence[]; flags: Flag[]; consistency: Record<string, unknown>[];
  overrides: unknown[]; refusals: unknown[]; rejections: unknown[]; ended: boolean; end_reason: string | null;
  candidate_speaking: boolean; agent_speaking: boolean;
  turns?: TurnMetric[]; cost_split?: { voice: number; brain: number }; layout?: Layout;
};

export async function getConfig(): Promise<AppConfig> {
  return (await fetch("/api/config")).json();
}

export type InviteRow = {
  token: string; role_id: string; persona_id: string; candidate_id: string; created_at: number;
  status: "invited" | "in_call" | "completed" | "cancelled"; session_id: string | null; url: string; candidate: string; role: string;
};

/** Starts the candidate pipeline (plan → recruiter review); the link is issued after the plan is approved. */
export async function createInvite(body: { candidate_id: string; role_id: string; persona_id: string }): Promise<Pipeline> {
  const r = await fetch("/api/invites", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json()).detail ?? "failed");
  return r.json();
}

export async function listInvites(): Promise<InviteRow[]> {
  return (await fetch("/api/invites")).json();
}

export async function listSessions(): Promise<SessionRow[]> {
  return (await fetch("/api/sessions")).json();
}

export async function getSession(id: string): Promise<SessionRecord> {
  const r = await fetch(`/api/sessions/${id}`);
  if (!r.ok) throw new Error(`session ${id} not found`);
  return r.json();
}

export const mmss = (s: number | null | undefined) => {
  const v = Math.max(0, Math.floor(s ?? 0));
  return `${Math.floor(v / 60)}:${String(v % 60).padStart(2, "0")}`;
};

export const stageLabel = (s: string) =>
  ({ consent: "Consent", background: "Background", deal_deep_dive: "Deal deep dive",
     performance_verification: "Performance", ai_objection_roleplay: "AI objection role-play",
     fit_and_logistics: "Fit & logistics", candidate_questions: "Candidate questions", close: "Close" } as
    Record<string, string>)[s] ?? s.replace(/_/g, " ");

export const cleanName = (n?: string) => (n ?? "").replace(" (synthetic)", "");

type SavePicker = (o: { suggestedName: string; types: { description: string; accept: Record<string, string[]> }[] }) =>
  Promise<{ createWritable: () => Promise<{ write: (b: Blob) => Promise<void>; close: () => Promise<void> }> }>;

/** Export a session. Chromium browsers open the system "Save as" dialog so the user picks the folder;
 *  other browsers fall back to a normal download. */
export async function exportSession(id: string, format: "md" | "json", baseName: string) {
  const r = await fetch(`/api/sessions/${id}/export?format=${format}`);
  const blob = await r.blob();
  const name = `${baseName}.${format}`;
  const picker = (window as unknown as { showSaveFilePicker?: SavePicker }).showSaveFilePicker;
  if (picker) {
    try {
      const handle = await picker({
        suggestedName: name,
        types: [format === "md"
          ? { description: "Markdown", accept: { "text/markdown": [".md"] } }
          : { description: "JSON", accept: { "application/json": [".json"] } }],
      });
      const w = await handle.createWritable();
      await w.write(blob);
      await w.close();
      return;
    } catch (e) {
      if ((e as Error).name === "AbortError") return;   // the user cancelled the dialog
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
