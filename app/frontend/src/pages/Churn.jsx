import { useState } from "react";
import Chart from "../components/Chart.jsx";
import { Box, SectionHead, TrainBox } from "../components/Box.jsx";
import { Freshness, NoModel, Page } from "../components/Page.jsx";
import { chartBase } from "../chartBase.js";
import { num, pct, signed, words } from "../format.js";
import { useApi } from "../useApi.js";

function curveOption(c, pts, marker, opts) {
  const line = pts.map((p) => [p.x, p.y]);
  return {
    textStyle: c.textStyle,
    grid: c.gridX,
    tooltip: { ...c.tooltip, trigger: "axis", valueFormatter: (v) => Number(v).toFixed(3) },
    xAxis: { type: "value", min: 0, max: 1, name: opts.xName, nameLocation: "middle", nameGap: 28, nameTextStyle: c.nameStyle, axisLabel: c.axisLabel, splitLine: c.splitLine, axisLine: c.axisLine },
    yAxis: { type: "value", min: 0, max: 1, name: opts.yName, nameTextStyle: c.yNameStyle, axisLabel: c.axisLabel, splitLine: c.splitLine, axisLine: c.axisLine },
    series: [
      {
        type: "line", name: opts.name, data: line, showSymbol: false, lineStyle: { color: c.work, width: 2.5 }, itemStyle: { color: c.work },
        markLine: {
          silent: true, symbol: "none", label: { color: c.ink2, fontSize: 12, formatter: opts.refLabel, position: opts.refPos || "insideEndTop" },
          lineStyle: { color: c.ink3, type: "dashed", width: 1 }, data: [opts.ref],
        },
      },
      {
        type: "scatter", name: "Selected threshold", data: [marker], symbolSize: 13, z: 5,
        itemStyle: { color: c.rework, borderColor: c.sheet, borderWidth: 2 },
        label: { show: true, position: opts.labelPos, color: c.ink, fontWeight: 600, fontSize: 12, formatter: opts.markLabel },
      },
    ],
  };
}

function Matrix({ row }) {
  const total = row.tp + row.fp + row.tn + row.fn;
  const cell = (v, cls) => (
    <td className={cls}><b className="num">{num(v)}</b><span className="muted num">{pct(v / total, 1)}</span></td>
  );
  return (
    <table className="matrix">
      <caption className="sr-only">Confusion matrix for the class "reorders" at the selected threshold</caption>
      <thead>
        <tr><th /><th scope="col">Predicted to reorder</th><th scope="col">Predicted not to reorder</th></tr>
      </thead>
      <tbody>
        <tr><th scope="row">Reordered</th>{cell(row.tp, "hit")}{cell(row.fn, "miss")}</tr>
        <tr><th scope="row">Did not reorder</th>{cell(row.fp, "miss")}{cell(row.tn, "hit")}</tr>
      </tbody>
    </table>
  );
}

function nearest(rows, target) {
  let best = 0;
  rows.forEach((r, i) => { if (Math.abs(r.threshold - target) < Math.abs(rows[best].threshold - target)) best = i; });
  return best;
}

function View({ d, theme, latestId }) {
  const m = d.model;
  const ths = d.confusion.filter((r) => r.threshold >= 0 && r.threshold <= 1);
  const [idx, setIdx] = useState(() => nearest(ths, m?.metrics?.threshold ?? 0.5));
  if (!m) return <NoModel name="churn" />;
  const c = chartBase();
  const row = ths[Math.min(idx, ths.length - 1)];
  const th = row ? row.threshold : 0;
  const hasDetail = ths.length > 0;
  const top = d.customers.slice(0, 30);
  const scores = top.map((u) => Number(u.churn_score)).filter(Number.isFinite);
  const spread = scores.length ? { lo: num(Math.min(...scores), 4), hi: num(Math.max(...scores), 4), tight: Math.max(...scores) - Math.min(...scores) < 0.01 } : null;
  const prec = row ? row.tp / (row.tp + row.fp) : 0;
  const rec = row ? row.tp / (row.tp + row.fn) : 0;
  const fpr = row ? row.fp / (row.fp + row.tn) : 0;
  const f1 = prec + rec ? (2 * prec * rec) / (prec + rec) : 0;
  const base = m.metrics.positive_rate;
  const totalImp = d.importance.reduce((a, f) => a + (Number(f.importance) || 0), 0);
  const maxImp = Math.max(...d.importance.map((f) => Number(f.importance) || 0), 0);
  return (
    <>
      <section className="section" style={{ paddingTop: 4 }}>
        <div className="section-head">
          <h1>Churn</h1>
          <span className="section-note">Predicts which customers who have ordered before will place no order in the next 90 days.</span>
        </div>
        <Freshness run={d.run} latestId={latestId} />
        <div className="boxes four">
          <Box title="AUC-PR, class reorders" value={num(m.metrics.auc_pr, 3)} rows={[["Baseline (reorder rate)", pct(base, 0)], ["Test customers", num(m.metrics.test_customers)]]} />
          <Box title="AUC-ROC, class reorders" value={num(m.metrics.auc_roc, 3)} rows={[["Log loss", num(m.metrics.log_loss, 3)]]} />
          <Box title="At the selected threshold" value={<>{pct(prec)} <span className="muted unit">precision</span></>} rows={[["Recall", pct(rec)], ["F1", num(f1, 2)]]} />
          <TrainBox model={m} />
        </div>
      </section>
      {hasDetail ? (
        <>
          <section className="section">
            <SectionHead title="How good it is, at your threshold" note={`Charts score the class "reorders": customers who order again (the minority, ${pct(base, 0)}). The red marker is the threshold below.`} />
            <div className="threshold">
              <label htmlFor="th">Decision threshold <b className="num">{th.toFixed(3)}</b></label>
              <input id="th" type="range" min="0" max={ths.length - 1} step="1" value={Math.min(idx, ths.length - 1)} onChange={(e) => setIdx(Number(e.target.value))} />
              <span className="section-note">Customers scoring above it are predicted to reorder.</span>
            </div>
            <div className="trio">
              <div>
                <h3>Precision-recall <span className="muted">class: reorders</span></h3>
                <Chart theme={theme} label="Precision-recall curve" height={300}
                  option={curveOption(c, d.curves.pr, [rec, prec], { name: "Precision", xName: "Recall (reorders)", yName: "Precision (reorders)", ref: { yAxis: base }, refLabel: `Reorder rate ${pct(base)}`, refPos: "insideStartTop", labelPos: "top", markLabel: () => `${pct(prec)} / ${pct(rec)}` })} />
              </div>
              <div>
                <h3>ROC <span className="muted">class: reorders</span></h3>
                <Chart theme={theme} label="ROC curve" height={300}
                  option={curveOption(c, d.curves.roc, [fpr, rec], { name: "True positive rate", xName: "False positive rate", yName: "True positive rate (reorders)", ref: [{ coord: [0, 0] }, { coord: [1, 1] }], refLabel: "", labelPos: "right", markLabel: () => `${pct(rec)} / ${pct(fpr)}` })} />
              </div>
              <div>
                <h3>Confusion matrix <span className="muted">class: reorders</span></h3>
                <Matrix row={row} />
                <p className="section-note" style={{ marginTop: 8 }}>
                  Predicts {num(row.tp + row.fp)} customers will reorder; {num(row.tp)} of the {num(row.tp + row.fn)} who did.
                </p>
              </div>
            </div>
          </section>
          <section className="section">
            <div className="split">
              <div>
                <SectionHead title="What drives the score" note="Global feature importance for the model, share of total." />
                <ol className="bars">
                  {d.importance.map((f) => (
                    <li key={f.feature}>
                      <span className="bar-name">{words(f.feature)}</span>
                      <span className="bar-track" aria-hidden="true"><span style={{ width: `${maxImp ? (f.importance / maxImp) * 100 : 0}%` }} /></span>
                      <b className="num">{pct(f.importance / totalImp, 1)}</b>
                    </li>
                  ))}
                </ol>
              </div>
              <div>
                <SectionHead title="Most at risk" note={`Listed by P(churned), highest first: the probability of no order in 90 days, the opposite of the "reorders" class above. Reasons explain that score. Top ${top.length} shown.`} />
                {spread && (
                  <p className="section-note">
                    {spread.lo === spread.hi
                      ? `All ${top.length} scores are ${spread.hi}: the model cannot separate them, so read this list as a group, not an order.`
                      : spread.tight
                        ? `These ${top.length} scores sit between ${spread.lo} and ${spread.hi}: the model scores them almost equally, so read this list as a group, not an order.`
                        : `These ${top.length} scores sit between ${spread.lo} and ${spread.hi}.`}
                  </p>
                )}
                <div className="table-scroll">
                  <table>
                    <thead><tr><th>Rank</th><th>Customer</th><th>P(churned)</th><th className="left">Top reasons</th></tr></thead>
                    <tbody>
                      {top.map((u) => (
                        <tr key={u.user_id}>
                          <td className="num">{u.rank}</td>
                          <td className="num">{u.user_id}</td>
                          <td className="num total">{num(u.churn_score, 4)}</td>
                          <td className="left reasons">
                            {[1, 2, 3].map((i) => (
                              <span key={i}>{words(u[`reason_${i}_feature`])} <b className="num">{signed(u[`reason_${i}_value`], 3)}</b></span>
                            ))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </section>
        </>
      ) : (
        <section className="section"><div className="empty"><h3>No detail rows for this run</h3>Only the headline metrics were published. Pick a more recent run.</div></section>
      )}
    </>
  );
}

export default function Churn({ runId, latestId, theme }) {
  const api = useApi("churn", runId);
  return <Page api={api}>{(d) => <View d={d} theme={theme} latestId={latestId} />}</Page>;
}
