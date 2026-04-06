import streamlit as st
import tempfile
import os
import numpy as np
import json
import time

# ── HF Spaces: disable hf_transfer everywhere ────────────────────────
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ── HF Spaces: auto-login with HF_TOKEN secret ───────────────────────
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    try:
        import huggingface_hub
        huggingface_hub.login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        pass

# ── Install whisperx + hf_transfer at startup (too large for build) ──
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "whisperx @ git+https://github.com/m-bain/whisperX.git",
     "faster-whisper", "hf_transfer"],
    check=False,
)

# ── Patch whisperx subprocess: bypass uvx, force int8 on CPU ─────────
def _patch_whisperx_subprocess():
    import subprocess as _sp, torch
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    def _fix(cmd):
        if not isinstance(cmd, (list, tuple)):
            return cmd
        cmd = list(cmd)
        # Replace "uvx whisperx ..." with "python -m whisperx ..."
        # so it uses the pip-installed whisperx, not uvx's broken sandbox
        if len(cmd) >= 2 and ("uvx" in str(cmd[0])) and ("whisperx" in str(cmd[1])):
            cmd = [sys.executable, "-m", "whisperx"] + cmd[2:]
        if not any("whisperx" in str(a) for a in cmd[:4]):
            return cmd
        # Force int8 compute on CPU (float16 is CUDA-only)
        if compute_type != "float16":
            out, i = [], 0
            while i < len(cmd):
                if str(cmd[i]) == "--compute_type":
                    i += 2
                else:
                    out.append(cmd[i])
                    i += 1
            out += ["--compute_type", "int8"]
            cmd = out
        return cmd

    def _fix_env(kw):
        env = kw.get("env") or os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        kw["env"] = env
        return kw

    _orig_run = _sp.run
    _orig_popen = _sp.Popen

    def _run(cmd, *a, **kw):
        fixed = _fix(cmd)
        if isinstance(cmd, (list, tuple)) and any("whisperx" in str(x) for x in cmd[:4]):
            _fix_env(kw)
        return _orig_run(fixed, *a, **kw)

    def _popen(cmd, *a, **kw):
        fixed = _fix(cmd)
        if isinstance(cmd, (list, tuple)) and any("whisperx" in str(x) for x in cmd[:4]):
            _fix_env(kw)
        return _orig_popen(fixed, *a, **kw)

    _sp.run = _run
    _sp.Popen = _popen

_patch_whisperx_subprocess()

st.set_page_config(
    page_title="TRIBE v2 — Neural Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne+Mono&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  /* ── Reset & base ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: #07090d !important;
    color: #c8cfd8 !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  [data-testid="stHeader"] { background: transparent !important; }
  [data-testid="stToolbar"] { display: none !important; }
  [data-testid="stSidebar"] { display: none !important; }
  div.block-container { padding: 2.5rem 3rem !important; max-width: 1400px; margin: auto; }

  /* ── Typography ── */
  .mono { font-family: 'Syne Mono', monospace !important; }

  /* ── Header ── */
  .nra-header {
    border-bottom: 1px solid #1a2030;
    padding-bottom: 1.4rem;
    margin-bottom: 2.5rem;
  }
  .nra-badge {
    display: inline-block;
    background: #00e5ff18;
    border: 1px solid #00e5ff44;
    color: #00e5ff;
    font-family: 'Syne Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    padding: 3px 10px;
    border-radius: 2px;
    margin-bottom: 0.7rem;
  }
  .nra-title {
    font-size: 2rem !important;
    font-weight: 300 !important;
    color: #edf0f4 !important;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem 0 !important;
  }
  .nra-sub {
    font-size: 0.82rem;
    color: #4a5568;
    letter-spacing: 0.03em;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1.5px dashed #1e2d45 !important;
    border-radius: 6px !important;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploader"]:hover { border-color: #00e5ff55 !important; }
  [data-testid="stFileUploader"] label { color: #4a5568 !important; }
  [data-testid="stFileUploader"] section { padding: 2rem !important; }

  /* ── Button ── */
  .stButton > button {
    background: #00e5ff !important;
    color: #07090d !important;
    border: none !important;
    font-family: 'Syne Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.6rem 2rem !important;
    border-radius: 3px !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  /* ── Stat card ── */
  .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.2rem 0; }
  .stat-card {
    background: #0d1117;
    border: 1px solid #1a2030;
    border-radius: 5px;
    padding: 1rem 1.2rem;
  }
  .stat-label { font-size: 0.65rem; letter-spacing: 0.12em; color: #3d4f66; text-transform: uppercase; margin-bottom: 0.3rem; }
  .stat-val { font-family: 'Syne Mono', monospace; font-size: 1.1rem; color: #00e5ff; }

  /* ── Section header ── */
  .sect-head {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3d4f66;
    border-bottom: 1px solid #111820;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }

  /* ── Segment row ── */
  .seg-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid #111820;
  }
  .seg-idx {
    font-family: 'Syne Mono', monospace;
    font-size: 0.7rem;
    color: #00e5ff66;
    min-width: 2.5rem;
    padding-top: 2px;
  }
  .seg-body { font-size: 0.8rem; color: #8896aa; line-height: 1.5; }
  .seg-key { color: #c8cfd8; margin-right: 0.4rem; }

  /* ── Result panel ── */
  .result-panel {
    background: #0d1117;
    border: 1px solid #1a2030;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    height: 100%;
  }

  /* ── Download ── */
  [data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid #1e2d45 !important;
    color: #4a90b8 !important;
    font-family: 'Syne Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.45rem 1.2rem !important;
    border-radius: 3px !important;
    width: auto !important;
  }
  [data-testid="stDownloadButton"] > button:hover { border-color: #4a90b8 !important; }

  /* ── Chart ── */
  [data-testid="stVegaLiteChart"] { background: transparent !important; }
  [data-testid="stLineChart"] svg { background: transparent !important; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #00e5ff !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #07090d; }
  ::-webkit-scrollbar-thumb { background: #1a2030; border-radius: 2px; }

  /* ── Success ── */
  [data-testid="stAlert"] {
    background: #00e5ff10 !important;
    border: 1px solid #00e5ff33 !important;
    color: #00e5ff !important;
    border-radius: 4px !important;
    font-family: 'Syne Mono', monospace !important;
    font-size: 0.75rem !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="nra-header">
  <div class="nra-badge">META · TRIBE v2 · fMRI PREDICTION</div>
  <div class="nra-title">Neural Response Analyzer</div>
  <div class="nra-sub">Upload a video · predict cortical brain activity · inspect raw signals</div>
</div>
""", unsafe_allow_html=True)

# ── Pre-check: verify LLaMA 3.2 access before loading full model ────────
def _check_llama_access():
    """Fail fast with a clear message if LLaMA 3.2 is not accessible."""
    try:
        from huggingface_hub import model_info
        model_info("meta-llama/Llama-3.2-3B")
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err or "gated" in err.lower():
            st.error(
                "**LLaMA 3.2 access denied.** Two things needed:\n\n"
                "1. Accept the license at "
                "[meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)\n"
                "2. Add your HF token as a Space secret named `HF_TOKEN` in Settings"
            )
            st.stop()
        # Other errors (network etc.) — let it proceed and fail later
_check_llama_access()

# ── Model loader ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from tribev2 import TribeModel
    return TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

# ── Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop a video file here or click to browse",
    type=["mp4", "mkv", "avi", "mov", "webm"],
    label_visibility="visible",
)

if uploaded:
    st.video(uploaded)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    run = st.button("▶  RUN TRIBE v2 ANALYSIS")

    if run:
        # Save to temp file
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            video_path = tmp.name

        with st.spinner("Loading model weights…"):
            model = load_model()

        try:
            with st.spinner("Extracting multimodal events from video…"):
                df = model.get_events_dataframe(video_path=video_path)

            with st.spinner("Predicting fMRI brain responses…"):
                preds, segments = model.predict(events=df)
        except Exception as e:
            os.unlink(video_path)
            import traceback
            st.error(f"**Model error:** {e}\n\n```\n{traceback.format_exc()}\n```")
            st.stop()

        os.unlink(video_path)

        # ── Stats ────────────────────────────────────────────────────────
        n_t, n_v = preds.shape
        mean_act = float(preds.mean())
        peak_act = float(np.abs(preds).max())

        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-label">Timesteps</div>
            <div class="stat-val">{n_t}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Cortical Vertices</div>
            <div class="stat-val">{n_v:,}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Segments</div>
            <div class="stat-val">{len(segments)}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Mean Activation</div>
            <div class="stat-val">{mean_act:.4f}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Peak |Activation|</div>
            <div class="stat-val">{peak_act:.4f}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Value Range</div>
            <div class="stat-val">{preds.min():.3f} → {preds.max():.3f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two panels ───────────────────────────────────────────────────
        col1, col2 = st.columns([1.1, 0.9], gap="large")

        with col1:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown('<div class="sect-head">01 · Brain Signal Predictions (preds)</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.78rem;color:#4a5568;margin-bottom:1rem">shape&nbsp;<span class="mono" style="color:#c8cfd8">({n_t}, {n_v:,})</span> — mean activation per TR (1 s)</div>', unsafe_allow_html=True)

            chart_data = {"Activation": preds.mean(axis=1).tolist()}
            st.line_chart(chart_data, height=220, use_container_width=True)

            # sample of first 5 timesteps × first 8 vertices
            st.markdown('<div style="font-size:0.7rem;letter-spacing:0.08em;color:#3d4f66;text-transform:uppercase;margin:1rem 0 0.5rem">Raw sample — first 5 TR × 8 vertices</div>', unsafe_allow_html=True)
            import pandas as pd
            sample = pd.DataFrame(
                preds[:5, :8],
                columns=[f"v{i}" for i in range(8)],
                index=[f"t={i}s" for i in range(5)],
            ).round(4)
            st.dataframe(sample, use_container_width=True, height=180)

            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

            buf = __import__("io").BytesIO()
            np.save(buf, preds)
            st.download_button("↓  Download preds.npy", buf.getvalue(), "preds.npy", "application/octet-stream")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown('<div class="sect-head">02 · Temporal Segments</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.78rem;color:#4a5568;margin-bottom:1rem">{len(segments)} segments · video decomposed into events with timing</div>', unsafe_allow_html=True)

            seg_html = ""
            for i, seg in enumerate(segments):
                if isinstance(seg, dict):
                    inner = "".join(
                        f'<span class="seg-key">{k}:</span>{v}  '
                        for k, v in list(seg.items())[:4]
                    )
                else:
                    inner = str(seg)[:120]
                seg_html += f'<div class="seg-row"><div class="seg-idx">#{i:02d}</div><div class="seg-body">{inner}</div></div>'

            st.markdown(
                f'<div style="max-height:460px;overflow-y:auto">{seg_html}</div>',
                unsafe_allow_html=True,
            )

            seg_json = json.dumps(
                [s if isinstance(s, dict) else str(s) for s in segments],
                indent=2, default=str,
            )
            st.download_button(
                "↓  Download segments.json",
                seg_json.encode(),
                "segments.json",
                "application/json",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.success("Analysis complete — both outputs returned by TRIBE v2")
