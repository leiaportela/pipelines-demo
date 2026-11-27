import { useMemo, useState } from "react";
import Chart from "../components/Chart.jsx";
import { Box, SectionHead, TrainBox } from "../components/Box.jsx";
import { Freshness, NoModel, Page } from "../components/Page.jsx";
import { chartBase } from "../chartBase.js";
import { day, num, pct } from "../format.js";
import { useApi } from "../useApi.js";

// Legend glyphs: lines are thin bars centred in the 10 px legend box (extra points pin the bounding box height).
const PIN = "M0,0 L0,.01 M0,10 L0,10.01";
const LINE = `path://M0,4.5 L18,4.5 L18,5.5 L0,5.5 Z ${PIN}`;
const LINE_BOLD = `path://M0,3.5 L18,3.5 L18,6.5 L0,6.5 Z ${PIN}`;

// Band fill opacity. The dashed edges carry the contrast: edge vs fill is 3.7:1 (light) and 3.4:1 (dark).
const BAND_FILL = 0.2;
const withAlpha = (hex, a) => {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},${a})`;
};

function option(c, rows, asOf) {
  const dates = rows.map((r) => r[0]);
  const actual = rows.map((r) => r[1]);
  const fc = rows.map((r) => r[2]);
  const lower = rows.map((r) => r[3]);
  const width = rows.map((r) => (r[4] != null && r[3] != null ? r[4] - r[3] : null));
  return {
    textStyle: c.textStyle, grid: { ...c.grid, top: 64 },
    legend: { top: 0, left: 0, icon: "rect", itemWidth: 18, itemHeight: 10, textStyle: { color: c.ink2, fontSize: 12 }, data: [
      { name: "Actual units", icon: LINE, itemStyle: { color: c.ink } },
      { name: "Forecast", icon: LINE_BOLD, itemStyle: { color: c.work } },
      { name: "90% interval", icon: "rect", itemStyle: { color: withAlpha(c.work, BAND_FILL), borderColor: c.work, borderWidth: 1.5, borderType: "dashed" } },
    ] },
    tooltip: {
      ...c.tooltip,
      // The band is two stacked series (lower edge, then the width); show real bounds, never the raw width.
      formatter: (params) => {
        const i = params[0].dataIndex;
        const [date, actualV, fcV, lo, hi] = rows[i];
        const line = (name, v, color) => `<div style="display:flex;justify-content:space-between;gap:18px"><span><span style="display:inline-block;width:8px;height:8px;margin-right:6px;background:${color}"></span>${name}</span><b>${v == null ? "–" : num(v, 1)}</b></div>`;
        return `<div style="margin-bottom:4px">${day(date)}</div>${actualV == null ? "" : line("Actual units", actualV, c.ink)}${line("Forecast", fcV, c.work)}${fcV == null ? "" : line("90% upper", hi, c.work) + line("90% lower", lo, c.work)}`;
      },
    },
    xAxis: { type: "category", data: dates, boundaryGap: false, axisTick: { show: false }, axisLabel: { ...c.axisLabel, formatter: (v) => day(v), hideOverlap: true }, axisLine: c.axisLine },
    yAxis: { type: "value", name: "Units per day", nameTextStyle: c.yNameStyle, axisLabel: c.axisLabel, splitLine: c.splitLine, axisLine: c.axisLine },
    series: [
      { type: "line", name: "lower", data: lower, stack: "band", symbol: "none", lineStyle: { color: c.work, width: 1.5, type: "dashed" }, itemStyle: { color: c.work }, silent: true },
      { type: "line", name: "90% interval", data: width, stack: "band", symbol: "none", lineStyle: { color: c.work, width: 1.5, type: "dashed" }, areaStyle: { color: c.work, opacity: BAND_FILL }, itemStyle: { color: c.work } },
      { type: "line", name: "Actual units", data: actual, symbol: "none", lineStyle: { color: c.ink, width: 2 }, itemStyle: { color: c.ink } },
      {
        type: "line", name: "Forecast", data: fc, symbol: "none", lineStyle: { color: c.work, width: 2.5 }, itemStyle: { color: c.work },
        markLine: { silent: true, symbol: "none", lineStyle: { color: c.ink3, type: "dashed", width: 1 }, label: { color: c.ink2, fontSize: 12, formatter: `Source data up to ${day(asOf)}`, position: "end", rotate: 0, distance: 4 }, data: [{ xAxis: asOf }] },
      },
    ],
  };
}

function View({ d, theme, latestId }) {
  const m = d.model;
  const cats = Object.keys(d.series);
  const mape = useMemo(() => Object.fromEntries(d.holdout.map((h) => [h.category, h])), [d]);
  const sorted = useMemo(() => [...d.holdout].sort((a, b) => a.mape - b.mape), [d]);
  const [cat, setCat] = useState(null);
  if (!m) return <NoModel name="demand" />;
  const k = m.metrics;
  const current = cat && d.series[cat] ? cat : cats[0];
  const c = chartBase();
  const maxMape = Math.max(...d.holdout.map((h) => h.mape), 0);
  const h = mape[current];
  return (
    <>
      <section className="section" style={{ paddingTop: 4 }}>
        <div className="section-head">
          <h1>Demand</h1>
          <span className="section-note">Predicts daily units sold per product category for the next 30 days.</span>
        </div>
        <Freshness run={d.run} latestId={latestId} cutoff={d.data_cutoff} />
        {d.data_cutoff && (
          <p className="cutoff">
            {d.data_cutoff.excluded_days > 0
              ? <>The forecast leaves out the last {d.data_cutoff.excluded_days} days of source data (up to {day(d.run.data_as_of)}): the synthetic dataset ramps up sharply at its end.</>
              : <>The forecast uses all source data.</>}
          </p>
        )}
        {k.coverage_mean != null && k.coverage_mean < 0.8 && (
          <p className="callout">
            <b>How to read this.</b> The 90% band is too narrow: on the holdout it contained {pct(k.coverage_mean)} of actual days, not 90%.
            Treat the shaded band as a lower bound on the real uncertainty. Mean holdout error is {num(k.mape_mean, 1)}% (MAPE), and it varies a lot by category.
          </p>
        )}
        <div className="boxes four">
          <Box title="Holdout MAPE, mean" value={`${num(k.mape_mean, 1)}%`} rows={[["Categories", num(k.categories ?? k.series)], ["Holdout", d.holdout[0] ? `${d.holdout[0].holdout_days} days` : "–"]]} />
          <Box title="90% interval coverage" value={pct(k.coverage_mean)} rows={[["Target", "90%"]]} />
          <Box title={`${current || "Category"}, MAPE`} value={h ? `${num(h.mape, 1)}%` : "–"} rows={h ? [["Interval coverage", pct(h.coverage)], ["Holdout", `${h.holdout_days} days`]] : []} />
          <TrainBox model={m} />
        </div>
      </section>
      {cats.length ? (
        <section className="section">
          <div className="split demand">
            <div>
              <div className="section-head">
                <h2>Actuals and forecast</h2>
                <label className="picker">Category
                  <select className="select" value={current} onChange={(e) => setCat(e.target.value)}>
                    {cats.map((x) => <option key={x}>{x}</option>)}
                  </select>
                </label>
              </div>
              <Chart theme={theme} label={`Actual units and forecast for ${current}`} height={380} option={option(c, d.series[current], d.run.data_as_of)} />
            </div>
            <div>
              <SectionHead title="Holdout MAPE by category" note="Lower is better. Select a row to chart it." />
              <ol className="bars pick">
                {sorted.map((x) => (
                  <li key={x.category}>
                    <button type="button" aria-pressed={x.category === current} onClick={() => setCat(x.category)}>
                      <span className="bar-name">{x.category}</span>
                      <span className="bar-track" aria-hidden="true"><span style={{ width: `${(x.mape / maxMape) * 100}%` }} /></span>
                      <b className="num">{num(x.mape, 1)}%</b>
                    </button>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </section>
      ) : (
        <section className="section"><div className="empty"><h3>No detail rows for this run</h3>Only the headline metrics were published. Pick a more recent run.</div></section>
      )}
    </>
  );
}

export default function Demand({ runId, latestId, theme }) {
  const api = useApi("demand", runId);
  return <Page api={api}>{(d) => <View d={d} theme={theme} latestId={latestId} />}</Page>;
}
