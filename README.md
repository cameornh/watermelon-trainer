# watermelon_trainer

Self-hosted GPU training worker for the fruit-phenotyping platform. It runs on
your own workstation, pulls queued fine-tune / training jobs from the backend,
trains a YOLO-seg model on the GPU, and uploads the resulting weights, metrics,
and reference embeddings back. The backend runs the eval gate and promotes the
model. No Google Colab, no Google Drive, no paid GPU cloud.

This folder is fully self-contained: it talks to the backend only over HTTP and
does not import any backend code, so you can copy the whole `watermelon_trainer`
directory to the training computer on its own.

## How it fits together

```
backend Space (CPU)                        your workstation (GPU)
  POST /experts/{id}/finetune  --enqueue-->  job queue
  GET  /train_jobs/next        <--claim----  worker.py poll loop
  GET  /train_jobs/{id}/bundle --bundle---->  YOLO(base).train() + .val()
  POST /train_jobs/{id}/artifacts <--upload-  weights + metrics + embeddings
  eval gate -> promote -> hot-reload
```

All connections are outbound from the workstation, so it works behind NAT with
no port-forwarding or tunnel.

## Files

- `worker.py` - poll loop / entrypoint (`python -m watermelon_trainer.worker` or `python worker.py`)
- `embed.py` - standalone DINOv2 reference-embedding helper (kept in sync with the backend embedder)
- `requirements.txt` - cu128 PyTorch + Ultralytics + helpers
- `.env.example` - copy to `.env` and fill in `BACKEND_URL` + `WORKER_TOKEN`
- `SETUP.md` - full step-by-step setup for a native Windows + Blackwell GPU machine

## Quick start (after the environment is set up - see SETUP.md)

```powershell
Copy-Item .env.example .env
notepad .env               # fill in BACKEND_URL and WORKER_TOKEN
.\.venv\Scripts\python.exe worker.py
```

The worker logs each claimed job and its progress. Leave it running; it will
keep draining the queue. See `SETUP.md` for GPU driver setup, the cu128 install,
verifying CUDA, and optional autostart.
