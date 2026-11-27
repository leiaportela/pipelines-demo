const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// BigQuery can hand numbers over as strings (NUMERIC, INT64 in JSON) or NULL; read them all the same way.
export function n(v) {
  if (v == null || v === "") return null;
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

export function num(v, digits = 0) {
  v = n(v);
  if (v == null) return "–";
  return Number(v).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function usd(v, digits = 0) {
  v = n(v);
  if (v == null) return "–";
  return `${v < 0 ? "−" : ""}$${num(Math.abs(v), digits)}`;
}

export function pct(v, digits = 0) {
  v = n(v);
  if (v == null) return "–";
  return `${(v * 100).toFixed(digits)}%`;
}

export function day(d) {
  if (!d) return "–";
  const p = String(d).slice(0, 10).split("-");
  return `${MONTHS[Number(p[1]) - 1]} ${Number(p[2])}`;
}

export function stamp(ts) {
  if (!ts) return "–";
  const d = new Date(ts);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${hh}:${mm} UTC`;
}

export function duration(s) {
  s = n(s);
  if (s == null) return "–";
  const t = Math.round(s);
  if (t < 60) return `${t} s`;
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  if (h) return `${h} h ${String(m).padStart(2, "0")} min`;
  return `${m} min ${String(t % 60).padStart(2, "0")} s`;
}

export function words(s) {
  const t = String(s || "").replace(/_/g, " ");
  return t.charAt(0).toUpperCase() + t.slice(1);
}

export function signed(v, digits = 2) {
  v = n(v);
  if (v == null) return "–";
  return `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(digits)}`;
}
