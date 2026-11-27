import ReactEChartsCore from "echarts-for-react/lib/core";
import { BarChart, LineChart, ScatterChart } from "echarts/charts";
import { GridComponent, LegendComponent, MarkLineComponent, MarkPointComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([BarChart, LineChart, ScatterChart, GridComponent, LegendComponent, MarkLineComponent, MarkPointComponent, TooltipComponent, CanvasRenderer]);

export default function Chart({ option, height = 300, theme, label }) {
  return (
    <div role="img" aria-label={label}>
      <ReactEChartsCore key={theme} echarts={echarts} option={option} notMerge style={{ height }} opts={{ renderer: "canvas" }} />
    </div>
  );
}
