import { CheckCircle2, MinusCircle, XCircle } from "lucide-react";
import { day, stamp } from "../format.js";
import { Empty } from "./Box.jsx";

export function Status({ status }) {
  const s = String(status || "").toUpperCase();
  const [Icon, cls] = s === "SUCCEEDED" ? [CheckCircle2, "pass"] : s === "FAILED" ? [XCircle, "fail"] : [MinusCircle, "skip"];
  return (
    <span className={`status ${cls}`}>
      <Icon size={14} aria-hidden="true" />{s.charAt(0) + s.slice(1).toLowerCase()}
    </span>
  );
}

export function Freshness({ run, latestId, cutoff }) {
  if (!run) return null;
  return (
    <p className="fresh-line num">
      <span>Run <b>{run.run_id}</b></span>
      <Status status={run.status} />
      <span>started {stamp(run.started_at)}</span>
      <span>source data up to {day(run.data_as_of)}</span>
      {cutoff && <span>forecast trained to {day(cutoff.date)}</span>}
      {latestId && run.run_id !== latestId && <span className="older">Not the latest run</span>}
    </p>
  );
}

export function Loading() {
  return (
    <div aria-busy="true" aria-label="Loading">
      <div className="boxes">{[0, 1, 2, 3].map((i) => <div key={i} className="skeleton" style={{ height: 108 }} />)}</div>
      <div className="skeleton" style={{ height: 340, marginTop: 28 }} />
    </div>
  );
}

export function Page({ api, children }) {
  if (api.error) {
    return (
      <div className="alert" role="alert">
        <div>
          <b>Couldn't load this page.</b> {api.error}
          <div style={{ marginTop: 8 }}><button type="button" className="btn" onClick={api.retry}>Try again</button></div>
        </div>
      </div>
    );
  }
  if (!api.data) return <Loading />;
  if (api.data.empty) {
    return (
      <Empty title="No pipeline run has been published yet">
        The dashboard fills in when a pipeline run succeeds and publishes its results. Check back after the next run.
      </Empty>
    );
  }
  return children(api.data);
}

export function NoModel({ name }) {
  return (
    <Empty title={`No ${name} model in this run`}>
      This run didn't publish a {name} model, so there is nothing to show. Pick a successful run in the run selector.
    </Empty>
  );
}
