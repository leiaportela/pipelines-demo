import { useCallback, useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";
import { getJSON } from "./api.js";
import Churn from "./pages/Churn.jsx";
import Clv from "./pages/Clv.jsx";
import Demand from "./pages/Demand.jsx";
import Overview from "./pages/Overview.jsx";
import { day } from "./format.js";

const TABS = ["Overview", "Churn", "CLV", "Demand"];

function Mark() {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" aria-hidden="true">
      <path d="M1 4h6v8h5V4h9" fill="none" stroke="var(--shell-ink)" strokeWidth="2" strokeLinejoin="miter" />
    </svg>
  );
}

function readHash() {
  const [path, query] = window.location.hash.replace(/^#\/?/, "").split("?");
  const q = new URLSearchParams(query || "");
  return { tab: TABS.find((t) => t.toLowerCase() === path) || TABS[0], run: q.get("run"), theme: q.get("theme") };
}

export default function App() {
  const [state, setState] = useState(readHash);
  const [theme, setTheme] = useState(() => readHash().theme || localStorage.getItem("theme") || "light");
  const [runs, setRuns] = useState([]);

  // Charts read CSS tokens while rendering, so the theme attribute must be set before children render.
  document.documentElement.dataset.theme = theme;
  useEffect(() => {
    localStorage.setItem("theme", theme);
  }, [theme]);
  useEffect(() => {
    const onHash = () => setState(readHash());
    window.addEventListener("hashchange", onHash);
    getJSON("/api/runs").then((d) => setRuns(d.runs)).catch(() => setRuns([]));
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = useCallback((next) => {
    const s = { ...state, ...next };
    const q = new URLSearchParams();
    if (s.run) q.set("run", s.run);
    const hash = `#/${s.tab.toLowerCase()}${q.toString() ? `?${q}` : ""}`;
    window.history.replaceState(null, "", hash);
    setState({ ...s, theme: null });
  }, [state]);

  const latest = runs.find((r) => r.status === "SUCCEEDED");
  const common = { runId: state.run, latestId: latest?.run_id, theme };
  const Page = { Overview, Churn, CLV: Clv, Demand }[state.tab];

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand"><Mark />Vertex AI pipeline</div>
          <nav className="tabs" role="tablist" aria-label="Models">
            {TABS.map((t) => (
              <button key={t} type="button" role="tab" className="tab" aria-selected={state.tab === t} onClick={() => go({ tab: t })}>{t}</button>
            ))}
          </nav>
          <div className="topbar-right">
            {runs.length > 0 && (
              <label className="run-pick">
                <span className="sr-only">Run</span>
                <select value={state.run || ""} onChange={(e) => go({ run: e.target.value || null })} aria-label="Run">
                  <option value="">Latest successful run</option>
                  {runs.map((r) => (
                    <option key={r.run_id} value={r.run_id}>{day(r.data_as_of)}: {r.run_id}{r.status === "FAILED" ? " (failed)" : ""}</option>
                  ))}
                </select>
              </label>
            )}
            <button type="button" className="icon-btn" aria-label={theme === "light" ? "Switch to blueprint (dark)" : "Switch to drawing (light)"}
              onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
              {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
            </button>
          </div>
        </div>
      </header>
      <main className="sheet">
        <Page key={`${state.tab}:${state.run || ""}`} {...common} onRun={(run) => go({ run })} onTab={(tab) => go({ tab })} />
      </main>
    </>
  );
}
