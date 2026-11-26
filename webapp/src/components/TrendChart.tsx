import type { AirSnapshot } from "../domain/airQuality";

type TrendChartProps = {
  snapshot: AirSnapshot;
};

const chartWidth = 720;
const chartHeight = 210;
const padding = { top: 20, right: 12, bottom: 28, left: 44 };

export default function TrendChart({ snapshot }: TrendChartProps) {
  const values = [
    ...snapshot.history.map((point) => point.pm25),
    snapshot.pm25,
  ];
  const maxValue = Math.max(50, Math.ceil(Math.max(...values) / 10) * 10);
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;
  const x = (index: number) =>
    padding.left + (index / (values.length - 1)) * innerWidth;
  const y = (value: number) =>
    padding.top + innerHeight - (value / maxValue) * innerHeight;
  const points = values.map((value, index) => `${x(index)},${y(value)}`);
  const lastX = x(values.length - 1);
  const lastY = y(snapshot.pm25);

  return (
    <div className="trend-chart">
      <svg
        viewBox={`0 0 ${chartWidth} ${chartHeight}`}
        role="img"
        aria-label="Biểu đồ PM2.5 trong 24 giờ qua và dự báo hiện tại"
      >
        {[0, 0.5, 1].map((fraction) => {
          const value = Math.round(maxValue * fraction);
          const lineY = y(value);
          return (
            <g key={fraction}>
              <line
                x1={padding.left}
                x2={chartWidth - padding.right}
                y1={lineY}
                y2={lineY}
                className="chart-grid"
              />
              <text x={padding.left - 10} y={lineY + 4} className="chart-label">
                {value}
              </text>
            </g>
          );
        })}
        <polygon
          points={[
            `${x(0)},${chartHeight - padding.bottom}`,
            ...points,
            `${lastX},${chartHeight - padding.bottom}`,
          ].join(" ")}
          className="chart-area"
        />
        <polyline points={points.join(" ")} className="chart-line" />
        <circle cx={lastX} cy={lastY} r="5" className="chart-current" />
        <text
          x={padding.left}
          y={chartHeight - 5}
          className="chart-label chart-label--edge"
        >
          24 giờ trước
        </text>
        <text
          x={chartWidth - padding.right}
          y={chartHeight - 5}
          textAnchor="end"
          className="chart-label chart-label--edge"
        >
          Hiện tại · AI
        </text>
      </svg>
    </div>
  );
}
