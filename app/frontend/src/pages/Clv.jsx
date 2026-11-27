import Chart from "../components/Chart.jsx";
import { Box, SectionHead, TrainBox } from "../components/Box.jsx";
import { Freshness, NoModel, Page } from "../components/Page.jsx";
import { chartBase } from "../chartBase.js";
import { num, pct, signed, usd, words } from "../format.js";
import { useApi } from "../useApi.js";

const axis = (c, extra) => ({ axisLabel: c.axisLabel, splitLine: c.splitLine, axisLine: c.axisLine, nameTextStyle: c.nameStyle, ...extra });
const money = (v) => (Math.abs(v) >= 1000 ? `$${num(v / 1000, 0)}k` : `$${num(v)}`);

function scatterOption(c, sample) {
  // Predictions are far tighter than the actuals, so each axis fits its own range; the diagonal runs as far as both reach.
  const xmax = Math.ceil(Math.max(...sample.map((r) => r.predicted)) / 25) * 25;
  const ymax = Math.ceil(Math.max(...sample.map((r) => r.actual)) / 50) * 50;
  const diag = Math.min(xmax, ymax);
  return {
    textStyle: c.textStyle, grid: c.gridX,
    tooltip: { ...c.tooltip, trigger: "item", formatter: (p) => `Predicted ${usd(p.value[0])}<br/>Actual ${usd(p.value[1])}` },
    xAxis: { type: "value", name: "Predicted spend, next 180 days", nameLocation: "middle", nameGap: 28, min: 0, max: xmax, ...axis(c, { axisLabel: { ...c.axisLabel, formatter: money } }) },
    yAxis: { type: "value", name: "Actual spend", min: 0, max: ymax, ...axis(c, { nameTextStyle: c.yNameStyle, axisLabel: { ...c.axisLabel, formatter: money } }) },
    series: [
      {
        type: "scatter", name: "Customers", symbolSize: 6, data: sample.map((r) => [r.predicted, r.actual]),
        itemStyle: { color: c.teal, opacity: 0.85 },
        markLine: { silent: true, symbol: "none", label: { show: true, formatter: "Perfect prediction", color: c.ink2, position: "insideEndTop" }, lineStyle: { color: c.ink3, type: "dashed", width: 1 }, data: [[{ coord: [0, 0] }, { coord: [diag, diag] }]] },
      },
    ],
  };
}

function groupedBars(c, cats, series, fmt) {
  return {
    textStyle: c.textStyle, grid: { ...c.grid, top: 40 },
    legend: { top: 0, left: 0, icon: "rect", itemWidth: 10, itemHeight: 10, textStyle: { color: c.ink2, fontSize: 12 } },
    tooltip: { ...c.tooltip, valueFormatter: fmt },
    xAxis: { type: "category", data: cats, ...axis(c, { axisTick: { show: false } }) },
    yAxis: { type: "value", ...axis(c, { axisLabel: { ...c.axisLabel, formatter: fmt } }) },
    series: series.map((s) => ({
      type: "bar", name: s.name, data: s.data, itemStyle: { color: s.color }, barMaxWidth: 28,
      label: { show: s.label, position: "top", color: c.ink, fontSize: 12, fontWeight: 600, formatter: (p) => fmt(p.value) },
    })),
  };
}

function trialsOption(c, trials, baseline) {
  const best = trials.find((t) => t.is_best);
  return {
    textStyle: c.textStyle, grid: c.gridX,
    tooltip: {
      ...c.tooltip, trigger: "item",
      formatter: (p) => {
        const t = trials[p.dataIndex];
        return `Trial ${t.trial_id}: RMSE ${num(t.rmse, 2)}<br/>${Object.entries(t.hyperparameters || {}).map(([k, v]) => `${words(k)} ${Number.isInteger(Number(v)) ? Number(v) : Number(v).toPrecision(3)}`).join("<br/>")}`;
      },
    },
    xAxis: { type: "category", name: "Trial", nameLocation: "middle", nameGap: 28, data: trials.map((t) => t.trial_id), ...axis(c, { axisTick: { show: false } }) },
    yAxis: { type: "value", name: "Validation RMSE ($)", scale: true, min: (v) => Math.floor((v.min - 0.2) * 4) / 4, max: (v) => Math.ceil((Math.max(v.max, baseline || 0) + 0.1) * 4) / 4, interval: 0.25, ...axis(c, { nameTextStyle: c.yNameStyle }) },
    series: [
      {
        type: "scatter", data: trials.map((t) => ({ value: t.rmse, symbolSize: t.is_best ? 16 : 10, itemStyle: { color: t.is_best ? c.rework : c.teal, borderColor: c.sheet, borderWidth: 1.5 } })),
        label: { show: false },
        markPoint: best ? {
          symbol: "circle", symbolSize: 0, silent: true,
          label: { show: true, color: c.ink, fontWeight: 600, fontSize: 12, position: "top", offset: Number(best.trial_id) >= trials.length - 1 ? [-18, -14] : [0, -14], align: Number(best.trial_id) >= trials.length - 1 ? "right" : "center", backgroundColor: c.sheet, padding: [1, 4], formatter: `Best: trial ${best.trial_id}, ${num(best.rmse, 1)}` },
          data: [{ coord: [best.trial_id, best.rmse] }],
        } : undefined,
        markLine: baseline ? {
          silent: true, symbol: "none", lineStyle: { color: c.ink3, type: "dashed", width: 1 },
          label: { color: c.ink2, fontSize: 12, formatter: `Baseline ${num(baseline, 1)}`, position: "insideEndTop" }, data: [{ yAxis: baseline }],
        } : undefined,
      },
    ],
  };
}

function Honest({ k, d }) {
  const gain = 1 - k.rmse / k.baseline_rmse;
  const byRank = [...d.deciles].sort((a, b) => a.decile - b.decile);
  const rises = byRank.length > 2 && byRank[0].actual_revenue > byRank[byRank.length - 1].actual_revenue;
  return (
    <p className="callout">
      <b>How to read this.</b> Spend in the next 180 days is hard to predict for one customer: the model explains {pct(k.r2, 1)} of the
      spread (R²) and lowers RMSE by {pct(gain, 1)} against always guessing the mean.
      {rises && ` It is more useful for ranking customers than for sizing a single customer's spend: actual revenue is higher in the top predicted decile than in the bottom one (lift ${num(byRank[0].lift, 1)}× against ${num(byRank[byRank.length - 1].lift, 1)}×).`}
    </p>
  );
}

function View({ d, theme, latestId }) {
  const m = d.model;
  if (!m) return <NoModel name="CLV" />;
  const c = chartBase();
  const k = m.metrics;
  const detail = d.sample.length > 0;
  // The contract numbers deciles from the highest predicted spend (D1); draw them from low to high.
  const deciles = [...d.deciles].sort((a, b) => b.decile - a.decile);
  const best = d.trials.find((t) => t.is_best);
  return (
    <>
      <section className="section" style={{ paddingTop: 4 }}>
        <div className="section-head">
          <h1>CLV</h1>
          <span className="section-note">Predicts how much each customer will spend in the next 180 days.</span>
        </div>
        <Freshness run={d.run} latestId={latestId} />
        {k.r2 != null && k.r2 < 0.1 && <Honest k={k} d={d} />}
        <div className="boxes four">
          <Box title="RMSE" value={usd(k.rmse, 1)} rows={[["Mean baseline RMSE", usd(k.baseline_rmse, 1)], ["Lower than baseline by", pct(1 - k.rmse / k.baseline_rmse, 1)], ["MAE", usd(k.mae, 1)]]} />
          <Box title="R²" value={num(k.r2, 3)} rows={[["Test customers", num(k.test_customers)]]} />
          <Box title="Best tuning trial" value={best ? `Trial ${best.trial_id}` : "–"}
            rows={best ? [["Of", `${d.trials.length} trials`], ["Validation RMSE", usd(best.rmse, 1)]] : []} />
          <TrainBox model={m} />
        </div>
      </section>
      {detail ? (
        <>
          <section className="section">
            <div className="split even">
              <div>
                <SectionHead title="Predicted vs actual" note="One dot per test customer, sample." />
                <Chart theme={theme} label="Predicted against actual spend" height={340} option={scatterOption(c, d.sample)} />
              </div>
              <div>
                <SectionHead title="Revenue by predicted decile" note="Total revenue for each tenth of customers, ranked by predicted spend (D1 = highest predicted spend, drawn last)." />
                <Chart theme={theme} label="Predicted and actual revenue by decile" height={340}
                  option={groupedBars(c, deciles.map((r) => `D${r.decile}`), [
                    { name: "Predicted", data: deciles.map((r) => r.predicted_revenue), color: c.work },
                    { name: "Actual", data: deciles.map((r) => r.actual_revenue), color: c.teal },
                  ], money)} />
              </div>
            </div>
          </section>
          <section className="section">
            <div className="split even">
              <div>
                <SectionHead title="Error by customer segment" note="Bias above zero means over-predicting." />
                <div className="table-scroll">
                  <table>
                    <thead><tr><th>Segment</th><th>Customers</th><th>MAE</th><th>RMSE</th><th>Bias</th></tr></thead>
                    <tbody>
                      {d.segments.map((s) => (
                        <tr key={s.segment}>
                          <td>{s.segment}</td><td className="num">{num(s.customers)}</td>
                          <td className="num">{usd(s.mae, 1)}</td><td className="num total">{usd(s.rmse, 1)}</td>
                          <td className="num">{signed(s.bias, 1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <div>
                <SectionHead title="Hyperparameter tuning" note={`${d.trials.length} trials, lowest RMSE wins.`} />
                <Chart theme={theme} label="Tuning trials by validation RMSE" height={300} option={trialsOption(c, d.trials, k.baseline_rmse)} />
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

export default function Clv({ runId, latestId, theme }) {
  const api = useApi("clv", runId);
  return <Page api={api}>{(d) => <View d={d} theme={theme} latestId={latestId} />}</Page>;
}
