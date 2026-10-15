// The candidate app: a separate entry with only the call page. It never loads recruiter code.
import "@fontsource-variable/hanken-grotesk";
import "@fontsource-variable/newsreader";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Call from "./pages/Call";
import "./styles.css";

const token = location.pathname.split("/").filter(Boolean)[1] ?? "";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Call token={token} />
  </StrictMode>,
);
