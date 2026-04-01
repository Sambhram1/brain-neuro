import { useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

function SignalChart({ signal }) {
  if (!signal?.length) return null;
  const min = Math.min(...signal);
  const max = Math.max(...signal);
  const range = max - min || 1;
  const W = 600, H = 110, PAD = 8;
  const pts = signal
    .map((v, i) => {
      const x = PAD + (i / (signal.length - 1)) * (W - PAD * 2);
      const y = PAD + (1 - (v - min) / range) * (H - PAD * 2);
      return `${x},${y}`;
    })
    .join(" ");
  const fillPts = `${PAD},${H - PAD} ${pts} ${W - PAD},${H - PAD}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" preserveAspectRatio="none">
      <defs>
        <linearGradient id="gl" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#00e5ff" stopOpacity="0.2" />
          <stop offset="100%" stopColor="#00e5ff" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={fillPts} fill="url(#gl)" />
      <polyline points={pts} fill="none" stroke="#00e5ff" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

function SampleTable({ sample }) {
  if (!sample?.length) return null;
  return (
    <div className="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>TR</th>
            {sample[0].map((_, i) => <th key={i}>v{i}</th>)}
          </tr>
        </thead>
        <tbody>
          {sample.map((row, i) => (
            <tr key={i}>
              <td className="idx">t={i}s</td>
              {row.map((v, j) => <td key={j}>{v.toFixed(4)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const STEPS = [
  { t: 0,   label: "Downloading video via yt-dlp" },
  { t: 8,   label: "Extracting audio & stripping chapters" },
  { t: 20,  label: "Transcribing speech with WhisperX (CPU — slow)" },
  { t: 120, label: "Extracting video features with V-JEPA2" },
  { t: 240, label: "Predicting fMRI brain responses" },
];

function LoadingSteps() {
  const [elapsed, setElapsed] = useState(0);
  useState(() => {
    const t = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => clearInterval(t);
  });
  const active = [...STEPS].reverse().find(s => elapsed >= s.t) || STEPS[0];
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  return (
    <div className="loading-wrap">
      <span className="spin spin-lg" />
      <div className="loading-info">
        <div className="loading-step">{active.label}…</div>
        <div className="loading-time">
          {mins > 0 ? `${mins}m ` : ""}{secs}s elapsed · CPU inference takes 10–20 min for a reel
        </div>
        <div className="loading-bar">
          <div className="loading-bar-fill" style={{ width: `${Math.min(elapsed / 1200 * 100, 95)}%` }} />
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="stat-l">{label}</div>
      <div className="stat-v">{value}</div>
    </div>
  );
}

export default function App() {
  const [url, setUrl] = useState("");
  const [phase, setPhase] = useState("idle");
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  async function run() {
    const trimmed = url.trim();
    if (!trimmed) return;
    setPhase("loading");
    setResult(null);
    setErr("");
    try {
      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: trimmed }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
      setPhase("done");
    } catch (e) {
      setErr(e.message);
      setPhase("error");
    }
  }

  function dlJson(data, name) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: name,
    });
    a.click();
  }

  return (
    <div className="app">
      <header className="header">
        <div className="badge">META · TRIBE v2 · fMRI PREDICTION</div>
        <h1>Neural Response Analyzer</h1>
        <p className="sub">
          Paste a video URL → model returns{" "}
          <code>[a, b]</code>
        </p>
      </header>

      <div className="input-wrap">
        <input
          className="url-in"
          type="url"
          placeholder="https://example.com/video.mp4"
          value={url}
          onChange={e => setUrl(e.target.value)}
          onKeyDown={e => e.key === "Enter" && run()}
          spellCheck={false}
          autoComplete="off"
        />
        <button className="run-btn" onClick={run} disabled={phase === "loading"}>
          {phase === "loading" ? <span className="spin" /> : "▶ Run"}
        </button>
      </div>

      {phase === "loading" && <LoadingSteps />}

      {phase === "error" && (
        <div className="err-box">✗ {err}</div>
      )}

      {phase === "done" && result && (
        <div className="results">
          <div className="ret-label">
            Return value &nbsp;<code>[a, b]</code>
          </div>

          <div className="panels">
            {/* ── A: preds ── */}
            <div className="panel">
              <div className="panel-hd">
                <span className="key">a</span>
                <div>
                  <div className="panel-title">preds</div>
                  <div className="panel-desc">{result.a.description}</div>
                </div>
              </div>

              <div className="stats">
                <Stat label="Shape" value={`(${result.a.shape[0]}, ${result.a.shape[1].toLocaleString()})`} />
                <Stat label="Timesteps" value={result.a.timesteps} />
                <Stat label="Vertices" value={result.a.vertices.toLocaleString()} />
                <Stat label="Mean" value={result.a.mean.toFixed(5)} />
                <Stat label="Peak |act|" value={result.a.peak_abs.toFixed(5)} />
                <Stat label="Range" value={`${result.a.min.toFixed(3)} → ${result.a.max.toFixed(3)}`} />
              </div>

              <div className="chart-lbl">Mean cortical activation / TR (1 s)</div>
              <SignalChart signal={result.a.signal} />

              <div className="chart-lbl" style={{ marginTop: "1.4rem" }}>
                Raw sample — first 5 TR × 8 vertices
              </div>
              <SampleTable sample={result.a.sample} />

              <button className="dl-btn" onClick={() => dlJson(result.a.signal, "preds_signal.json")}>
                ↓ preds_signal.json
              </button>
            </div>

            {/* ── B: segments ── */}
            <div className="panel">
              <div className="panel-hd">
                <span className="key">b</span>
                <div>
                  <div className="panel-title">segments</div>
                  <div className="panel-desc">{result.b.description}</div>
                </div>
              </div>

              <div className="stats">
                <Stat label="Total Segments" value={result.b.count} />
              </div>

              <div className="seg-list">
                {result.b.data.map((seg, i) => (
                  <div className="seg-row" key={i}>
                    <span className="seg-i">#{String(i).padStart(2, "0")}</span>
                    <span className="seg-body">
                      {typeof seg === "object"
                        ? Object.entries(seg)
                            .slice(0, 5)
                            .map(([k, v]) => (
                              <span key={k}>
                                <span className="seg-k">{k}:</span>
                                {String(v).slice(0, 60)}&nbsp;&nbsp;
                              </span>
                            ))
                        : String(seg).slice(0, 120)}
                    </span>
                  </div>
                ))}
              </div>

              <button className="dl-btn" onClick={() => dlJson(result.b.data, "segments.json")}>
                ↓ segments.json
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
