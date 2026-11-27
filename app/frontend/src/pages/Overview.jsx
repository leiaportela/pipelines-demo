import { ExternalLink } from "lucide-react";
import { Box, SectionHead, methodRow } from "../components/Box.jsx";
import { Freshness, Page, Status } from "../components/Page.jsx";
import { day, duration, num, pct, stamp, words } from "../format.js";
import { useApi } from "../useApi.js";

// The pipeline runs many small tasks; the drawing shows them grouped into stages (the fixtures use one task per stage).
const STAGES = [
  { key: "features", where: "in", label: "Build features", tasks: ["build_features", "build_churn_train", "build_churn_score", "build_clv_train", "build_clv_score", "build_demand_daily"] },
  { key: "check", where: "in", label: "Check data", tasks: ["check_data", "data_checks"] },
  { key: "churn-train", where: "churn", label: "AutoML model", tasks: ["train_automl_churn", "build_churn_automl_train", "build_churn_automl_score", "check_automl_data", "resolve_default_model", "require_default_model", "get_default_model"] },
  { key: "churn-batch", where: "churn", label: "Batch prediction", tasks: ["batch_predict_churn", "batch_predict_all", "select_top_customers", "batch_predict_explained"] },
  { key: "clv-train", where: "clv", label: "Tuning and training", tasks: ["train_clv_tuning", "build_clv_model_train", "build_clv_model_score", "tune_xgboost", "register_best_trial"] },
  { key: "clv-batch", where: "clv", label: "Batch prediction", tasks: ["batch_predict_clv", "get_clv_model", "model_batch_predict", "export_clv_score", "load_clv_predictions"] },
  { key: "demand-train", where: "demand", label: "ARIMA_PLUS training", tasks: ["train_demand_arima", "train_holdout_arima_plus", "label_holdout_model", "forecast_holdout", "evaluate_holdout", "train_production_arima_plus", "label_production_model"] },
  { key: "demand-fc", where: "demand", label: "Forecast", tasks: ["forecast_demand", "forecast_30_days", "check_forecast"] },
  { key: "publish", where: "out", label: "Publish", tasks: ["publish", "publish_results"] },
];

function groupSteps(steps) {
  const known = new Map(STAGES.flatMap((g) => g.tasks.map((t) => [t, g])));
  const buckets = new Map();
  for (const s of steps) {
    const g = known.get(s.step) || { key: `other-${s.step}`, where: s.branch === "shared" ? "in" : s.branch, label: words(s.step), tasks: [] };
    if (!buckets.has(g.key)) buckets.set(g.key, { ...g, rows: [] });
    buckets.get(g.key).rows.push(s);
  }
  const order = (g) => { const i = STAGES.findIndex((x) => x.key === g.key); return i === -1 ? 99 : i; };
  return [...buckets.values()].sort((a, b) => order(a) - order(b)).map((g) => {
    const st = g.rows.map((r) => r.status);
    const status = st.includes("FAILED") ? "FAILED" : st.every((x) => x === "SKIPPED") ? "SKIPPED" : "SUCCEEDED";
    const t0 = g.rows.map((r) => Date.parse(r.started_at)).filter(Number.isFinite);
    const t1 = g.rows.map((r) => Date.parse(r.ended_at)).filter(Number.isFinite);
    const span = t0.length && t1.length ? (Math.max(...t1) - Math.min(...t0)) / 1000 : null;
    const sum = g.rows.reduce((a, r) => a + (Number(r.duration_seconds) || 0), 0);
    return { ...g, step: g.key, status, duration_seconds: span ?? sum };
  });
}

const BRANCHES = [
  { key: "churn", label: "Churn", tab: "Churn", color: "--s-work" },
  { key: "clv", label: "CLV", tab: "CLV", color: "--s-verification" },
  { key: "demand", label: "Demand", tab: "Demand", color: "--s-setup" },
];

function Step({ step, max, color, note }) {
  const skipped = step.status === "SKIPPED";
  return (
    <div className={`step${skipped ? " skipped" : ""}`}>
      <div className="step-head">{step.label}</div>
      <div className="step-body">
        <Status status={step.status} />
        <div className="step-dur num">{skipped ? note || "Not run" : duration(step.duration_seconds)}</div>
        <div className="step-bar" aria-hidden="true">
          <span style={{ width: `${skipped || !max ? 0 : Math.max(3, (step.duration_seconds / max) * 100)}%`, background: `var(${color || "--ink-3"})` }} />
        </div>
      </div>
    </div>
  );
}

function Pipeline({ steps, run, models }) {
  const methods = Object.fromEntries(Object.entries(models).map(([k, m]) => [k, m.training_method]));
  const stages = groupSteps(steps);
  const churn = models.churn;
  const reused = churn && churn.trained_in_run === false;
  const stage = (g) => (reused && g.key === "churn-train" ? { ...g, status: "SKIPPED" } : g);
  const reusedNote = reused ? `Reused model v${churn.model_version ?? "?"}, trained ${day(churn.trained_at)}` : null;
  const max = Math.max(...stages.map((g) => g.duration_seconds || 0));
  const at = (where) => stages.filter((g) => g.where === where).map(stage);
  return (
    <div className="pipeline">
      <div className="stage in" aria-label="Prepare">
        {at("in").map((s) => <Step key={s.key} step={s} max={max} />)}
      </div>
      <div className="lanes">
        {BRANCHES.map((b) => (
          <div className="lane" key={b.key}>
            <div className="lane-label"><span className="swatch" style={{ background: `var(${b.color})` }} />{b.label}<span className="lane-method">{methods[b.key]}</span></div>
            <div className="lane-steps">
              {at(b.key).map((s) => <Step key={s.key} step={s} max={max} color={b.color} note={s.key === "churn-train" ? reusedNote : null} />)}
            </div>
          </div>
        ))}
      </div>
      <div className="stage out" aria-label="Publish">
        {at("out").map((s) => <Step key={s.key} step={s} max={max} />)}
        <div className="step-note num">Run took {duration((Date.parse(run.ended_at) - Date.parse(run.started_at)) / 1000)}</div>
      </div>
    </div>
  );
}

function ModelBox({ b, m, onOpen }) {
  const v = m.headline_metric_value;
  const value = m.headline_metric_name === "MAPE" ? `${num(v, 1)}%` : num(v, String(m.headline_metric_name).startsWith("AUC-PR") ? 3 : 1);
  return (
    <Box
      title={`${b.label}: ${m.training_method}`}
      value={<>{value} <span className="muted unit">{m.headline_metric_name}</span></>}
      aside={<button type="button" className="issue-link" onClick={() => onOpen(b.tab)}>Open {b.label}</button>}
      rows={[
        m.model_version != null ? ["Version", m.model_version, b.color] : ["Method", "Managed by BigQuery ML", b.color],
        ["Trained", m.trained_in_run === false ? `${day(m.trained_at)}, reused` : stamp(m.trained_at)],
        methodRow(m),
      ].filter(Boolean)}
    />
  );
}

export default function Overview({ runId, latestId, onRun, onTab }) {
  const api = useApi("overview", runId);
  return (
    <Page api={api}>
      {(d) => (
        <>
          <section className="section" style={{ paddingTop: 4 }}>
            <div className="section-head">
              <h1>Overview</h1>
              <span className="section-note">One Vertex AI pipeline trains three models, each with a different Google Cloud method.</span>
            </div>
            <Freshness run={d.run} latestId={latestId} />
            <div className="boxes models">
              {BRANCHES.map((b) => d.models[b.key] && <ModelBox key={b.key} b={b} m={d.models[b.key]} onOpen={onTab} />)}
            </div>
          </section>
          <section className="section">
            <SectionHead title="The pipeline, last run" note="Status and duration per step. Bars are scaled to the slowest step." />
            <Pipeline steps={d.steps} run={d.run} models={d.models} />
          </section>
          <section className="section">
            <SectionHead title="Run history" note="Select a run to open every page on it." />
            <div className="table-scroll">
              <table className="history">
                <thead>
                  <tr>{["Run", "Started", "Status", "Duration", "AutoML", "Source data up to", "Console"].map((h) => <th key={h}>{h}</th>)}</tr>
                </thead>
                <tbody>
                  {d.history.map((r) => (
                    <tr key={r.run_id} tabIndex={0} aria-selected={r.run_id === d.run.run_id}
                      onClick={() => onRun(r.run_id)} onKeyDown={(e) => e.key === "Enter" && onRun(r.run_id)}>
                      <td>{r.run_id}</td>
                      <td className="num">{stamp(r.started_at)}</td>
                      <td><Status status={r.status} /></td>
                      <td className="num">{duration((Date.parse(r.ended_at) - Date.parse(r.started_at)) / 1000)}</td>
                      <td>{r.train_automl ? "Trained" : "Reused"}</td>
                      <td className="num">{day(r.data_as_of)}</td>
                      <td>
                        {r.console_url && (
                          <a href={r.console_url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
                            Vertex AI <ExternalLink size={12} aria-hidden="true" />
                          </a>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="section-note" style={{ marginTop: 8 }}>
              {d.history.filter((r) => r.status === "SUCCEEDED").length} of {d.history.length} runs succeeded ({pct(d.history.filter((r) => r.status === "SUCCEEDED").length / d.history.length)}).
            </p>
          </section>
        </>
      )}
    </Page>
  );
}
