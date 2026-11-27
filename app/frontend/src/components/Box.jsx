import { ExternalLink } from "lucide-react";
import { duration, num, stamp } from "../format.js";

// The row that applies to each training method, from what the run published: a node-hour or trial budget when
// there is one, else the size of the evaluation (test customers) or of the model set (one series per category).
export function methodRow(model) {
  if (model.budget) return [model.trial_count ? "Tuning" : "Budget", model.budget];
  const k = model.metrics || {};
  const series = k.series ?? k.categories;
  if (series != null) return ["Series", `${num(series)} categories, one each`];
  if (k.test_customers != null) return ["Test set", `${num(k.test_customers)} customers`];
  return null;
}

export function Box({ title, value, rows = [], aside, small }) {
  return (
    <section className="box" aria-label={title}>
      <div className="box-head"><span>{title}</span>{aside}</div>
      <div className="box-body">
        <div className={`box-value num${small ? " small" : ""}`}>{value}</div>
        {rows.map(([k, v, sw]) => (
          <div className="box-row" key={k}>
            <span>{sw && <span className="swatch" style={{ background: `var(${sw})` }} />}{k}</span>
            <b className="num">{v}</b>
          </div>
        ))}
      </div>
    </section>
  );
}

export function TrainBox({ model }) {
  const reused = model.trained_in_run === false;
  const rows = [
    ["Method", model.training_method],
    ["Machine", model.machine_type],
    methodRow(model),
    ["Training time", reused ? "Reused, not retrained in this run" : duration(model.training_duration_seconds)],
    ["Trained", stamp(model.trained_at)],
    ["Version", model.model_version],
  ].filter((r) => r && r[1] != null && r[1] !== "");
  return (
    <section className="box train" aria-label="How it was trained">
      <div className="box-head">
        <span>How it was trained</span>
        {model.console_url && (
          <a className="issue-link" href={model.console_url} target="_blank" rel="noreferrer">
            Open in the console <ExternalLink size={12} aria-hidden="true" />
          </a>
        )}
      </div>
      <div className="box-body">
        <div className="train-method">{model.training_method}</div>
        {rows.slice(1).map(([k, v]) => (
          <div className="box-row" key={k}><span>{k}</span><b className="num">{v}</b></div>
        ))}
      </div>
    </section>
  );
}

export function SectionHead({ title, note }) {
  return (
    <div className="section-head">
      <h2>{title}</h2>
      {note && <span className="section-note">{note}</span>}
    </div>
  );
}

export function Empty({ title, children }) {
  return <div className="empty"><h3>{title}</h3>{children}</div>;
}
