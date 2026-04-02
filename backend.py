import tempfile, os, asyncio, shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
MODEL_DIR = str(Path(__file__).parent / "model_weights")
CACHE_DIR = str(Path(__file__).parent / "cache")

def get_model():
    global _model
    if _model is None:
        import pathlib, os
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath
        from tribev2 import TribeModel
        _model = TribeModel.from_pretrained(MODEL_DIR, cache_folder=CACHE_DIR, device="cpu")
    return _model

class Req(BaseModel):
    url: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(req: Req):
    tmp_dir = tempfile.mkdtemp()
    out = os.path.join(tmp_dir, "video.%(ext)s")

    # Download via yt-dlp (invoked through Node.js downloader.js)
    import subprocess
    script = Path(__file__).parent / "downloader.js"
    try:
        loop = asyncio.get_event_loop()
        def _dl():
            r = subprocess.run(
                ["node", str(script), req.url, tmp_dir],
                capture_output=True, text=True
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip())
            return r.stdout.strip()
        video_path = await loop.run_in_executor(None, _dl)
    except Exception as e:
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
