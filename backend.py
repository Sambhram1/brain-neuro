import sys, tempfile, os, asyncio, shutil, subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
MODEL_DIR = str(Path(__file__).parent / "model_weights")
CACHE_DIR = str(Path(__file__).parent / "cache")

def _patch_whisperx():
    """Force whisperx to use int8 on CPU, float16 only on CUDA."""
    import torch
    try:
        import whisperx.asr as _asr
        _orig = _asr.load_model
        def _patched(*args, **kwargs):
            if not torch.cuda.is_available():
                kwargs["compute_type"] = "int8"
            elif kwargs.get("compute_type", "float16") == "float16":
                # Verify the GPU actually supports float16
                try:
                    torch.zeros(1, dtype=torch.float16, device="cuda")
                except Exception:
                    kwargs["compute_type"] = "int8"
            return _orig(*args, **kwargs)
        _asr.load_model = _patched
    except Exception:
        pass


def get_model():
    global _model
    if _model is None:
        import pathlib
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath
        _patch_whisperx()
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
    return model.predict(events=df)


def _format_result(preds, segments):
    n_t, n_v = preds.shape
    return {
        "a": {
            "label": "preds",
            "shape": [n_t, n_v],
            "signal": preds.mean(axis=1).tolist(),
            "sample": preds[:5, :8].tolist(),
            "stats": {"mean": float(preds.mean()), "min": float(preds.min()), "max": float(preds.max())}
        },
        "b": {
            "label": "segments",
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
