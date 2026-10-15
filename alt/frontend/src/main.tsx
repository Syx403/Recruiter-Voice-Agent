import "@fontsource-variable/hanken-grotesk";
import "@fontsource-variable/jetbrains-mono";
import "@fontsource-variable/newsreader";
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import Console from "./pages/Console";
import PlanReview from "./pages/PlanReview";
import Role from "./pages/Role";
import Session from "./pages/Session";
import System from "./pages/System";
import "./styles.css";

// The recruiter console. Hash routes: #/ dashboard, #/sessions/<id> session (live or history), #/roles/<id> a role's
// screen configuration, #/pipelines/<id> a candidate's question plan.
// The candidate page is a separate app at /interview/<token> (interview.tsx).
function App() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const on = () => setHash(location.hash);
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const [, page, id] = hash.replace(/^#/, "").split("/");
  if ((page === "sessions" || page === "monitor") && id) return <Session key={id} id={id} />;
  if (page === "roles" && id) return <Role key={id} id={id} />;
  if (page === "pipelines" && id) return <PlanReview key={id} id={id} />;
  if (page === "system") return <System />;
  return <Console />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
