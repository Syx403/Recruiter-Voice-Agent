import { useEffect, useRef, useState } from "react";
import { Mic, Player, wsUrl } from "../lib/audio";

// Candidate interview page: consent, mic and a minimal state indicator only.
type Phase = "loading" | "consent" | "connecting" | "live" | "ended" | "used" | "error";
type Info = { status: string; role: string; company: string; interviewer: string; candidate: string };

export default function Call({ token }: { token: string }) {
  const [phase, setPhase] = useState<Phase>("loading");
  const [info, setInfo] = useState<Info>();
  const [agree, setAgree] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [error, setError] = useState("");
  const refs = useRef<{ ws?: WebSocket; mic?: Mic; player?: Player }>({});

  useEffect(() => {
    fetch(`/api/interview/${token}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("not found"))))
      .then((i: Info) => {
        setInfo(i);
        setPhase(i.status === "invited" ? "consent" : "used");
      })
      .catch(() => setPhase("error"));
    return () => {
      refs.current.mic?.stop();
      refs.current.player?.close();
      refs.current.ws?.close();
    };
  }, [token]);

  async function join() {
    setPhase("connecting");
    const player = new Player();
    const mic = new Mic();
    const ws = new WebSocket(wsUrl(`/ws/interview/${token}`));
    ws.binaryType = "arraybuffer";
    refs.current = { ws, mic, player };
    // gpt-live-1 streams audio continuously (silence included); the server says when the interviewer speaks.
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "audio") player.play(msg.item_id, msg.pcm);
      else if (msg.type === "speaking") setSpeaking(!!msg.on);
      else if (msg.type === "flush") {
        const cur = player.flush();
        if (cur) ws.send(JSON.stringify({ type: "playback", item_id: cur.itemId, played_ms: cur.playedMs }));
      } else if (msg.type === "ended") {
        mic.stop();
        setPhase("ended");
      } else if (msg.type === "error") {
        setError(String(msg.error?.message ?? msg.error));
      }
    };
    ws.onclose = () => setPhase((p) => (p === "live" || p === "connecting" ? "ended" : p));
    ws.onopen = async () => {
      try {
        await mic.start((pcm) => ws.readyState === WebSocket.OPEN && ws.send(pcm));
        setPhase("live");
      } catch (err) {
        setError(String(err));
        setPhase("error");
      }
    };
  }

  function hangup() {
    refs.current.ws?.send(JSON.stringify({ type: "hangup" }));
  }

  return (
    <div className="call">
      <p className="eyebrow">Cygnify · Screening interview</p>
      {phase === "loading" && <p className="notice">Loading…</p>}
      {(phase === "used" || (phase === "error" && !info)) && (
        <div className="call-box center">
          <h2>{phase === "used" ? "This interview link is no longer valid" : "Interview link not found"}</h2>
          <p className="lede">Please contact your recruiter if you need a new link.</p>
        </div>
      )}
      {phase === "consent" && info && (
        <div className="call-card">
          <h2>Hi {info.candidate}, welcome</h2>
          <p className="lede">Thanks for making time for your first-round screening interview.</p>
          <dl className="call-facts">
            <dt>Role</dt><dd>{info.role}</dd>
            <dt>Company</dt><dd>{info.company}</dd>
            <dt>Interviewer</dt><dd>{info.interviewer}, an AI interviewer working with Cygnify</dd>
            <dt>Length</dt><dd>About fifteen minutes</dd>
          </dl>
          <ul className="call-notes">
            <li>The call is recorded, and a human recruiter reviews it before any decision.</li>
            <li>Please use headphones in a quiet place.</li>
            <li>Take your time when you answer; you can ask the interviewer to repeat a question.</li>
          </ul>
          <label className="check">
            <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
            I understand and agree to a recorded AI screening call.
          </label>
          <button className="primary" disabled={!agree} onClick={join}>Join the call</button>
        </div>
      )}
      {(phase === "connecting" || phase === "live") && (
        <div className="call-box center">
          {info && <p className="call-role"><b>{info.role}</b><span>{info.company} · with {info.interviewer}</span></p>}
          <div className={`orb ${phase === "live" ? (speaking ? "speaking" : "listening") : ""}`} />
          <p className="state">{phase === "connecting" ? "Connecting…" : speaking ? "Interviewer speaking" : "Listening"}</p>
          <button className="ghost" onClick={hangup}>End call</button>
        </div>
      )}
      {phase === "ended" && (
        <div className="call-box center">
          <h2>Thank you</h2>
          <p className="lede">The call has ended. A recruiter will be in touch.</p>
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}
