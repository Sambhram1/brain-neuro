import sys, tempfile, os, asyncio, shutil, subprocess, gc
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
MODEL_DIR = str(Path(__file__).parent / "model_weights")
CACHE_DIR = os.environ.get("HF_HOME", str(Path(__file__).parent / "cache"))

# ── HF Spaces: login with HF_TOKEN secret if available ─────────────
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    try:
        import huggingface_hub
        huggingface_hub.login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        pass

def _patch_whisperx_subprocess():
    """
    tribev2 calls whisperx as a CLI subprocess with --compute_type float16.
    float16 is only supported on CUDA. Intercept all subprocess calls and
    replace float16 with int8 when no GPU is available.
    """
    import subprocess as _sp, torch
    # float16 only works on CUDA GPUs; MPS and CPU need int8
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    if compute_type == "float16":
        return  # CUDA GPU present — no patch needed

    def _fix(cmd):
        if not isinstance(cmd, (list, tuple)):
            return cmd
        if not any("whisperx" in str(a) for a in cmd[:3]):
            return cmd
        out, i = [], 0
        while i < len(cmd):
            if str(cmd[i]) == "--compute_type":
                i += 2          # drop flag + old value
            else:
                out.append(cmd[i])
                i += 1
        out += ["--compute_type", "int8"]
        return out

    _orig_run   = _sp.run
    _orig_popen = _sp.Popen

    def _run(cmd, *a, **kw):   return _orig_run(_fix(cmd), *a, **kw)
    def _popen(cmd, *a, **kw): return _orig_popen(_fix(cmd), *a, **kw)

    _sp.run   = _run
    _sp.Popen = _popen


# Apply patch at import time so it's active before any model call
_patch_whisperx_subprocess()


def get_model():
    global _model
    if _model is None:
        import pathlib
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath
        from tribev2.demo_utils import TribeModel
        _model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    return _model

class Req(BaseModel):
    url: str

@app.get("/health")
def health():
    return {"status": "ok"}

def _download_video(url: str, tmp_dir: str) -> str:
    import subprocess, sys
    out_tmpl = os.path.join(tmp_dir, "video.%(ext)s")
    args = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", out_tmpl,
        "--quiet", "--no-warnings",
    ]
    cookies = Path(__file__).parent / "cookies.txt"
    if cookies.exists():
        args += ["--cookies", str(cookies)]
    args.append(url)
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"yt-dlp exited {r.returncode}")
    mp4s = [f for f in os.listdir(tmp_dir) if f.endswith(".mp4")]
    if not mp4s:
        raise RuntimeError("yt-dlp ran but no .mp4 found")
    return os.path.join(tmp_dir, mp4s[0])


@app.post("/analyze")
async def analyze(req: Req):
    tmp_dir = tempfile.mkdtemp()
    try:
        loop = asyncio.get_event_loop()
        video_path = await loop.run_in_executor(None, _download_video, req.url, tmp_dir)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, f"Download failed: {e}")

    # Run model
    try:
        loop = asyncio.get_event_loop()
        preds, segments = await loop.run_in_executor(None, _run_model, video_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, f"Model error: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return _format_result(preds, segments)


def _run_model(video_path: str):
    model = get_model()
    df = model.get_events_dataframe(video_path=video_path)
    result = model.predict(events=df)
    gc.collect()  # free intermediate tensors on memory-constrained envs
    return result


def _format_result(preds, segments):
    n_t, n_v = preds.shape
    return {
        "a": {
            "label": "preds",
            "description": f"Predicted fMRI activity — {n_t} TRs × {n_v:,} vertices",
            "shape": [n_t, n_v],
            "timesteps": n_t,
            "vertices": n_v,
            "mean": float(preds.mean()),
            "min": float(preds.min()),
            "max": float(preds.max()),
            "peak_abs": float(np.abs(preds).max()),
            "signal": preds.mean(axis=1).tolist(),
            "sample": preds[:5, :8].tolist(),
        },
        "b": {
            "label": "segments",
            "description": f"{len(segments)} event segments extracted from video",
            "count": len(segments),
            "data": [s if isinstance(s, dict) else str(s) for s in segments],
        }
    }


@app.get("/fetch-video")
async def fetch_video(url: str):
    """Download a video URL via yt-dlp and stream it back to the frontend."""
    tmp_dir = tempfile.mkdtemp()
    try:
        loop = asyncio.get_event_loop()
        video_path = await loop.run_in_executor(None, _download_video, url, tmp_dir)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, f"Download failed: {e}")

    def stream_and_cleanup():
        try:
            with open(video_path, "rb") as f:
                yield from f
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        stream_and_cleanup(),
        media_type="video/mp4",
        headers={"Content-Disposition": "attachment; filename=video.mp4"},
    )


@app.post("/analyze-upload")
async def analyze_upload(file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, "video.mp4")
    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    try:
        loop = asyncio.get_event_loop()
        preds, segments = await loop.run_in_executor(None, _run_model, video_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, f"Model error: {e}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return _format_result(preds, segments)


# ── Serve frontend from /static, with index.html as fallback ────────
_STATIC = Path(__file__).parent / "static"
if _STATIC.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_STATIC / "assets")), name="assets")

    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        """Serve frontend files; fall back to index.html for SPA routing."""
        file = _STATIC / path
        if file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(_STATIC / "index.html"))
