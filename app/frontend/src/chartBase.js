import { cssVar } from "./cssVar.js";

/** Shared ECharts pieces read from the live theme tokens (call inside render, after the theme is applied). */
export function chartBase() {
  const ink = cssVar("--ink");
  const ink2 = cssVar("--ink-2");
  const ink3 = cssVar("--ink-3");
  const rule = cssVar("--rule");
  const sheet = cssVar("--sheet");
  const font = "'Archivo Variable', Archivo, system-ui, sans-serif";
  return {
    ink, ink2, ink3, rule, sheet,
    accent: cssVar("--accent"),
    work: cssVar("--s-work"),
    teal: cssVar("--s-verification"),
    rework: cssVar("--s-rework"),
    violet: cssVar("--s-setup"),
    textStyle: { fontFamily: font, color: ink2, fontSize: 12 },
    tooltip: {
      trigger: "axis",
      backgroundColor: sheet,
      borderColor: cssVar("--rule-strong"),
      borderWidth: 1,
      padding: [8, 10],
      textStyle: { color: ink, fontFamily: font, fontSize: 12 },
      extraCssText: `box-shadow:${cssVar("--shadow")};border-radius:2px;`,
    },
    axisLabel: { color: ink2, fontSize: 12, fontFamily: font },
    nameStyle: { color: ink2, fontSize: 12, fontFamily: font },
    yNameStyle: { color: ink2, fontSize: 12, fontFamily: font, align: "left" },
    splitLine: { lineStyle: { color: rule, width: 1 } },
    axisLine: { lineStyle: { color: cssVar("--rule-strong") } },
    grid: { left: 8, right: 16, top: 36, bottom: 8, containLabel: true },
    gridX: { left: 8, right: 16, top: 36, bottom: 40, containLabel: true },
  };
}
