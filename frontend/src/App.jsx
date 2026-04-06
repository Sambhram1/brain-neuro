import { useState } from "react";
import "./App.css";

// When served from the backend (same-origin), use "" (relative paths).
// When served from Vercel/elsewhere, user can click the Backend URL to configure it.
// For HF Spaces: set VITE_API_URL=https://<your-username>-<space-name>.hf.space
const DEFAULT_API = import.meta.env.VITE_API_URL || "";

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

const STEPS_URL = [
  { t: 0,   label: "Downloading video to browser via yt-dlp" },
  { t: 8,   label: "Extracting audio & stripping chapters" },
  { t: 20,  label: "Transcribing speech with WhisperX (CPU — slow)" },
  { t: 120, label: "Extracting video features with V-JEPA2" },
  { t: 240, label: "Predicting fMRI brain responses" },
];

const STEPS_FILE = [
  { t: 0,   label: "Uploading video to backend" },
  { t: 5,   label: "Extracting audio & stripping chapters" },
  { t: 20,  label: "Transcribing speech with WhisperX (CPU — slow)" },
  { t: 120, label: "Extracting video features with V-JEPA2" },
  { t: 240, label: "Predicting fMRI brain responses" },
];

function LoadingSteps({ steps }) {
  const [elapsed, setElapsed] = useState(0);
  useState(() => {
    const t = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => clearInterval(t);
  });
  const active = [...steps].reverse().find(s => elapsed >= s.t) || steps[0];
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
  const [mode, setMode] = useState("url"); // "url" | "file"
  const [file, setFile] = useState(null);
  const [phase, setPhase] = useState("idle");
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");
  const [apiUrl, setApiUrl] = useState(() => localStorage.getItem("api_url") || DEFAULT_API);
  const [showApiInput, setShowApiInput] = useState(false);

  function saveApiUrl(val) {
    const trimmed = val.trim().replace(/\/$/, "");
    localStorage.setItem("api_url", trimmed);
    setApiUrl(trimmed);
  }

  async function run() {
    if (mode === "url" && !url.trim()) return;
    if (mode === "file" && !file) return;
    setPhase("loading");
    setResult(null);
    setErr("");
    try {
      // Pre-flight: verify backend is reachable
      try {
        const hc = await fetch(`${apiUrl}/health`, { signal: AbortSignal.timeout(10000) });
        if (!hc.ok) throw new Error();
      } catch {
        throw new Error(
          `Cannot reach backend at ${apiUrl} — is the Colab notebook still running? ` +
          `Check that Cell 7 is active and the tunnel URL is correct.`
        );
      }

      let videoFile = file;

      if (mode === "url") {
        // Step 1: backend downloads via yt-dlp and streams the video back
        const dlRes = await fetch(`${apiUrl}/fetch-video?url=${encodeURIComponent(url.trim())}`);
        if (!dlRes.ok) {
          const j = await dlRes.json().catch(() => ({}));
          throw new Error(j.detail || `Download failed: HTTP ${dlRes.status}`);
        }
        const blob = await dlRes.blob();
        videoFile = new File([blob], "video.mp4", { type: "video/mp4" });
      }

      // Step 2: upload video to backend for model processing
      const form = new FormData();
      form.append("file", videoFile);
      const res = await fetch(`${apiUrl}/analyze-upload`, { method: "POST", body: form });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
      setPhase("done");
    } catch (e) {
      setErr(e.message || "Network error — check backend URL and Colab notebook");
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
        <div className="api-config">
          <span className="api-label">Backend: </span>
          {showApiInput ? (
            <input
              className="api-in"
              defaultValue={apiUrl}
              onBlur={e => { saveApiUrl(e.target.value); setShowApiInput(false); }}
              onKeyDown={e => { if (e.key === "Enter") { saveApiUrl(e.target.value); setShowApiInput(false); } }}
              autoFocus
            />
          ) : (
            <span className="api-url" onClick={() => setShowApiInput(true)}>{apiUrl}</span>
          )}
        </div>
      </header>

      <div className="mode-tabs">
        <button className={`mode-tab${mode === "url" ? " active" : ""}`} onClick={() => setMode("url")}>
          URL
        </button>
        <button className={`mode-tab${mode === "file" ? " active" : ""}`} onClick={() => setMode("file")}>
          Upload file
        </button>
      </div>

      <div className="input-wrap">
        {mode === "url" ? (
          <input
            className="url-in"
            type="url"
            placeholder="https://youtube.com/shorts/... or Instagram/TikTok URL"
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === "Enter" && run()}
            spellCheck={false}
            autoComplete="off"
          />
        ) : (
          <label className="file-label">
            <input
              type="file"
              accept="video/*,.mp4,.mov,.avi,.mkv,.webm"
              className="file-in"
              onChange={e => setFile(e.target.files[0] || null)}
            />
            <span className="file-name">{file ? file.name : "Choose a video file…"}</span>
          </label>
        )}
        <button className="run-btn" onClick={run} disabled={phase === "loading"}>
          {phase === "loading" ? <span className="spin" /> : "▶ Run"}
        </button>
      </div>

      {phase === "loading" && <LoadingSteps steps={mode === "file" ? STEPS_FILE : STEPS_URL} />}

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
