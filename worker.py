"""Self-hosted GPU training worker for the fruit-phenotyping platform.

Runs on the user's workstation (RTX PRO 4000 Blackwell, native Windows). It only
makes OUTBOUND HTTPS calls to the backend, so it works behind home/lab NAT with
no port-forwarding or tunnel. The loop is:

    claim a queued job  ->  download its data bundle  ->  YOLO train + val
    ->  embed the new train images  ->  upload weights + metrics + embeddings

The backend owns the job queue and runs the eval gate / promotion when the
artifacts arrive. This file talks to the backend purely over HTTP and never
imports any backend module, so the whole ``watermelon_trainer`` folder can be
copied to the training computer as-is (see SETUP.md).
"""

import io
import json
import os
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import embed as embed_helper

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BACKEND_URL = os.getenv("BACKEND_URL", "").rstrip("/")
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SECONDS", "15"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "120"))
BUNDLE_TIMEOUT = float(os.getenv("BUNDLE_TIMEOUT_SECONDS", "600"))
WORK_DIR = Path(os.getenv("WORK_DIR", str(Path(__file__).resolve().parent / "_work")))
DEVICE = os.getenv("TRAIN_DEVICE", "0")  # "0" = first CUDA GPU, "cpu" for testing

# Hyperparameter defaults; per-job values from the bundle override these.
DEFAULT_EPOCHS = int(os.getenv("DEFAULT_EPOCHS", "100"))
DEFAULT_IMGSZ = int(os.getenv("DEFAULT_IMGSZ", "1024"))
DEFAULT_BATCH = int(os.getenv("DEFAULT_BATCH", "4"))
DEFAULT_PATIENCE = int(os.getenv("DEFAULT_PATIENCE", "20"))

_HEADERS = {"X-Worker-Token": WORKER_TOKEN}


def _log(msg: str) -> None:
    print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Backend HTTP contract
# --------------------------------------------------------------------------- #
def claim_next_job():
    url = f"{BACKEND_URL}/train_jobs/next"
    resp = requests.get(url, headers=_HEADERS, timeout=HTTP_TIMEOUT)
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        return None
    return data.get("job")


def download_bundle(job_id: str, dest: Path) -> Path:
    url = f"{BACKEND_URL}/train_jobs/{job_id}/bundle"
    resp = requests.get(url, headers=_HEADERS, timeout=BUNDLE_TIMEOUT, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return dest


def send_heartbeat(job_id: str, *, progress=None, message=None, status=None) -> None:
    url = f"{BACKEND_URL}/train_jobs/{job_id}/heartbeat"
    payload = {}
    if progress is not None:
        payload["progress"] = float(progress)
    if message is not None:
        payload["message"] = str(message)
    if status is not None:
        payload["status"] = str(status)
    try:
        requests.post(url, headers=_HEADERS, json=payload, timeout=30)
    except Exception as exc:
        _log(f"heartbeat failed (non-fatal): {exc}")


def upload_artifacts(job_id: str, weights_path: Path, metrics: dict, embeddings_path: Path) -> dict:
    url = f"{BACKEND_URL}/train_jobs/{job_id}/artifacts"
    files = {
        "weights": ("best.pt", open(weights_path, "rb"), "application/octet-stream"),
        "embeddings": ("embeddings.npz", open(embeddings_path, "rb"), "application/octet-stream"),
    }
    data = {"metrics": json.dumps(metrics)}
    try:
        resp = requests.post(url, headers=_HEADERS, files=files, data=data, timeout=BUNDLE_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    finally:
        for _, (_, fh, _) in files.items():
            try:
                fh.close()
            except Exception:
                pass


def report_failure(job_id: str, message: str) -> None:
    send_heartbeat(job_id, status="failed", message=message[:500])


def describe_claim_error(exc: Exception) -> str:
    """Return an actionable message for queue-claim failures."""
    response = getattr(exc, "response", None)
    if response is not None:
        status = response.status_code
        if status == 404:
            parsed = urlparse(BACKEND_URL)
            return (
                f"{exc} | The backend at {parsed.netloc or BACKEND_URL} does not expose "
                "/train_jobs/next. This usually means BACKEND_URL points at the wrong "
                "Space, the backend Space is still running an older deployment, or the "
                "training endpoints were not pushed/rebuilt there yet."
            )
        if status == 401:
            return (
                f"{exc} | Worker token was rejected. Check that WORKER_TOKEN in .env "
                "matches the Hugging Face Space secret and that the Space was restarted."
            )
    return str(exc)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _extract_bundle(zip_path: Path, work: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work)
    job_meta = {}
    meta_path = work / "job.json"
    if meta_path.exists():
        job_meta = json.loads(meta_path.read_text())
    return job_meta


def _train_image_paths(work: Path):
    images_dir = work / "images" / "train"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)


def run_job(job: dict) -> None:
    from ultralytics import YOLO

    job_id = job["job_id"]
    hp = job.get("hyperparams") or {}
    epochs = int(hp.get("epochs", DEFAULT_EPOCHS))
    imgsz = int(hp.get("imgsz", DEFAULT_IMGSZ))
    batch = int(hp.get("batch", DEFAULT_BATCH))
    patience = int(hp.get("patience", DEFAULT_PATIENCE))

    _log(f"claimed job {job_id} (expert={job.get('expert_id')} mode={job.get('mode')})")
    send_heartbeat(job_id, progress=1, message="Downloading data bundle...", status="running")

    work = WORK_DIR / job_id
    if work.exists():
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    zip_path = work / "bundle.zip"
    download_bundle(job_id, zip_path)
    job_meta = _extract_bundle(zip_path, work)

    data_yaml = work / "data.yaml"
    base_weights = work / "base.pt"
    if not data_yaml.exists():
        raise FileNotFoundError("bundle missing data.yaml")
    if not base_weights.exists():
        raise FileNotFoundError("bundle missing base.pt")

    send_heartbeat(job_id, progress=3, message=f"Training {epochs} epochs @ imgsz={imgsz}...")
    model = YOLO(str(base_weights))

    def _on_epoch_end(trainer):
        try:
            cur = int(getattr(trainer, "epoch", 0)) + 1
            total = int(getattr(trainer, "epochs", epochs)) or epochs
            pct = 3 + int(92 * cur / max(total, 1))
            send_heartbeat(job_id, progress=min(pct, 95), message=f"Epoch {cur}/{total}")
        except Exception:
            pass

    model.add_callback("on_fit_epoch_end", _on_epoch_end)

    run_dir = work / "runs"
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=DEVICE,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        verbose=False,
    )

    # Locate best weights produced by training.
    best_weights = run_dir / "train" / "weights" / "best.pt"
    if not best_weights.exists():
        last = run_dir / "train" / "weights" / "last.pt"
        best_weights = last if last.exists() else base_weights

    has_val = (work / "images" / "val").exists() and any((work / "images" / "val").iterdir())
    metrics = {"epochs": epochs, "imgsz": imgsz, "has_val": bool(has_val)}
    if has_val:
        send_heartbeat(job_id, progress=96, message="Evaluating candidate on val split...")
        cand_metrics = YOLO(str(best_weights)).val(data=str(data_yaml), imgsz=imgsz, device=DEVICE, verbose=False)
        metrics.update(_extract_metrics(cand_metrics, results))
        # Evaluate the incumbent (base.pt, shipped in the bundle) on the SAME val
        # split so the backend gate is a fair comparison without needing a GPU.
        send_heartbeat(job_id, progress=97, message="Evaluating incumbent on val split...")
        try:
            base_metrics = YOLO(str(base_weights)).val(data=str(data_yaml), imgsz=imgsz, device=DEVICE, verbose=False)
            base_extracted = _extract_metrics(base_metrics, None)
            metrics["base_primary_metric"] = base_extracted.get("primary_metric", 0.0)
            metrics["base_seg_map"] = base_extracted.get("seg_map")
        except Exception:
            traceback.print_exc()
            metrics["base_primary_metric"] = None
    else:
        # No val split (very small dataset): nothing to gate against.
        metrics["primary_metric"] = None
        metrics["base_primary_metric"] = None

    send_heartbeat(job_id, progress=98, message="Embedding train images...")
    emb_path = work / "embeddings.npz"
    n_emb = embed_helper.save_reference_npz(emb_path, _train_image_paths(work))
    metrics["reference_embedding_count"] = n_emb

    send_heartbeat(job_id, progress=99, message="Uploading artifacts...")
    result = upload_artifacts(job_id, best_weights, metrics, emb_path)
    _log(f"job {job_id} done: {result}")

    import shutil
    shutil.rmtree(work, ignore_errors=True)


def _extract_metrics(val_metrics, train_results) -> dict:
    """Pull seg + box mAP out of an Ultralytics results object, defensively."""
    out = {}
    try:
        box = getattr(val_metrics, "box", None)
        seg = getattr(val_metrics, "seg", None)
        if seg is not None:
            out["seg_map"] = float(getattr(seg, "map", float("nan")))
            out["seg_map50"] = float(getattr(seg, "map50", float("nan")))
        if box is not None:
            out["box_map"] = float(getattr(box, "map", float("nan")))
            out["box_map50"] = float(getattr(box, "map50", float("nan")))
        rd = getattr(val_metrics, "results_dict", None)
        if isinstance(rd, dict):
            out["results_dict"] = {k: float(v) for k, v in rd.items() if _is_num(v)}
    except Exception:
        traceback.print_exc()
    # Primary gate metric: prefer seg mAP50-95, fall back to box.
    out["primary_metric"] = out.get("seg_map", out.get("box_map", 0.0))
    return out


def _is_num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def _validate_config() -> bool:
    ok = True
    if not BACKEND_URL:
        _log("ERROR: BACKEND_URL is not set (see .env.example).")
        ok = False
    if not WORKER_TOKEN:
        _log("ERROR: WORKER_TOKEN is not set (see .env.example).")
        ok = False
    return ok


def _report_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            _log(f"CUDA available: {torch.cuda.get_device_name(0)} (device={DEVICE})")
        else:
            _log("WARNING: CUDA not available; training will run on CPU (slow). Set up the GPU per SETUP.md.")
    except Exception as exc:
        _log(f"Could not query torch/CUDA: {exc}")


def main() -> int:
    if not _validate_config():
        return 2
    _log(f"Trainer worker starting. Backend={BACKEND_URL}, poll={POLL_INTERVAL}s")
    _report_gpu()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        job = None
        try:
            job = claim_next_job()
        except Exception as exc:
            _log(f"claim failed: {describe_claim_error(exc)}")
            time.sleep(POLL_INTERVAL)
            continue

        if not job:
            time.sleep(POLL_INTERVAL)
            continue

        try:
            run_job(job)
        except Exception as exc:
            traceback.print_exc()
            try:
                report_failure(job.get("job_id", ""), f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            # Brief pause so a persistently failing job doesn't hot-loop.
            time.sleep(min(POLL_INTERVAL, 10))


if __name__ == "__main__":
    sys.exit(main() or 0)
