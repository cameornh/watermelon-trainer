# Training computer setup (native Windows + RTX PRO 4000 Blackwell)

This guide sets up the self-hosted GPU training worker directly on Windows. It
does **not** require WSL, Ubuntu, Docker, inbound networking, port-forwarding, or
admin rights for the Python environment itself.

- **Target machine:** Windows 10/11, NVIDIA RTX PRO 4000 Blackwell Workstation
  Edition or similar CUDA-capable NVIDIA GPU.
- **Runtime:** native Windows Python virtual environment.
- **Connectivity:** outbound HTTPS only. The worker calls the backend; nothing
  connects in.

> You can copy this entire `watermelon_trainer` folder to the machine as-is. It
> does not import any backend code.

---

## 0. What you'll need

- The backend Space URL, for example:
  `https://PPAL-SongLab-UGA-fruit-analyzer.hf.space`
- A `WORKER_TOKEN` secret. This must be the same value on the backend Space and
  in the worker's local `.env` file.
- A recent NVIDIA Windows driver already installed on the computer.
- Python 3.10, 3.11, or 3.12 available for your Windows user account.

If you do not have admin rights, the two likely blockers are the NVIDIA driver
and Python itself. The worker does not need admin rights once those are present.
If `nvidia-smi` or `python --version` is missing, ask the computer admin/IT to
install a recent NVIDIA RTX driver and Python for you, or install Python using a
per-user installer if your policy allows it.

Generate a worker token on any machine with Python:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 1. Confirm the NVIDIA driver

Open PowerShell, not as Administrator unless your organization requires it:

```powershell
nvidia-smi
```

You should see the RTX PRO 4000 and a CUDA version. For Blackwell / `sm_120`,
use a recent driver that supports CUDA 12.8-era PyTorch wheels.

If `nvidia-smi` is not recognized or the GPU does not appear, this is a machine
setup issue rather than a worker issue. You will need an admin/IT person to
install or update the NVIDIA RTX / workstation driver.

---

## 2. Confirm Python

In PowerShell:

```powershell
py -0p
python --version
```

Use Python 3.10, 3.11, or 3.12. If the `py` launcher exists, prefer it because
it makes version selection explicit. For example:

```powershell
py -3.11 --version
```

If Python is not installed and you do not have admin rights, try the official
Python Windows installer with **Install for me only** if allowed by your
computer policy. Otherwise, ask IT to install Python.

---

## 3. Copy the worker folder

Put the `watermelon_trainer` folder somewhere in your Windows user directory,
for example:

```powershell
mkdir "$env:USERPROFILE\platform" -Force
Copy-Item "$env:USERPROFILE\Downloads\watermelon_trainer" "$env:USERPROFILE\platform\" -Recurse -Force
cd "$env:USERPROFILE\platform\watermelon_trainer"
```

Adjust the source path if you transferred the folder some other way.

---

## 4. Create a virtual environment

From inside the `watermelon_trainer` folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

If `py -3.11` does not work but `python` does:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

You can activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, either run this for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

or skip activation and call `.\.venv\Scripts\python.exe` directly in every
command.

---

## 5. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pulls `torch` and `torchvision` from the **cu128** wheel
index, which is what the Blackwell `sm_120` GPU needs. Older CUDA wheel builds
can install successfully but fail at runtime with:

```text
no kernel image is available for execution on the device
```

The first run will also download the DINOv2 embedder weights from Torch Hub.

---

## 6. Verify CUDA from native Windows Python

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

Expected:

- `cuda True`
- the RTX PRO 4000 device name
- `capability (12, 0)` for Blackwell

If CUDA is false but `nvidia-smi` works, reinstall the dependencies from
`requirements.txt` and make sure the cu128 PyTorch wheels were installed. If
CUDA is still false, the NVIDIA driver is probably too old.

---

## 7. Set the backend secret

On the backend Hugging Face Space, add a **Secret** named `WORKER_TOKEN` with
the value generated in step 0. Restart the Space so it picks up the secret.
Without this, the backend rejects every worker request.

---

## 8. Create the worker `.env`

From inside `watermelon_trainer`, create `.env`:

```powershell
@"
BACKEND_URL=https://PPAL-SongLab-UGA-fruit-analyzer.hf.space
WORKER_TOKEN=paste_the_same_worker_token_here
TRAIN_DEVICE=0

# Optional defaults. Individual platform jobs can override these.
DEFAULT_EPOCHS=100
DEFAULT_IMGSZ=1024
DEFAULT_BATCH=4
DEFAULT_PATIENCE=20
POLL_INTERVAL_SECONDS=15
"@ | Set-Content .env
```

Edit `.env` with Notepad if needed:

```powershell
notepad .env
```

Use the dev backend URL instead if you intentionally want this worker to drain
the dev queue.

---

## 9. Run the worker

```powershell
.\.venv\Scripts\python.exe worker.py
```

Expected startup:

```text
[worker HH:MM:SS] Trainer worker starting. Backend=https://...
[worker HH:MM:SS] CUDA available: NVIDIA RTX PRO 4000 ... (device=0)
```

Leave it running. When you click **Fine-tune from corrections** or onboard a new
fruit in the platform, the worker will claim the job, download the bundle,
train and validate on the GPU, upload weights/metrics/embeddings, and let the
backend promote or reject the candidate.

For a CPU-only dry run:

```powershell
notepad .env
```

Set:

```text
TRAIN_DEVICE=cpu
```

This verifies the queue loop but training will be very slow.

---

## 10. Optional autostart without admin rights

Use Task Scheduler if your account is allowed to create user-level tasks.

1. Open **Task Scheduler**.
2. Choose **Create Basic Task...**.
3. Trigger: **When I log on**.
4. Action: **Start a program**.
5. Program/script:

```text
powershell.exe
```

6. Arguments:

```text
-NoProfile -ExecutionPolicy Bypass -Command "cd $env:USERPROFILE\platform\watermelon_trainer; .\.venv\Scripts\python.exe worker.py"
```

This starts the worker whenever your Windows user logs in. It will not run while
the computer is off or before your user session exists.

---

## 11. Troubleshooting

- **`nvidia-smi` is not recognized**: the NVIDIA driver is missing or not on
  PATH. This usually requires admin/IT help.
- **`torch.cuda.is_available()` is false**: the NVIDIA driver is too old, or
  CPU-only PyTorch was installed. Reinstall with
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- **`no kernel image is available for execution on the device`**: PyTorch is not
  new enough for Blackwell. Reinstall from `requirements.txt`, which uses the
  cu128 wheel index.
- **PowerShell cannot activate `.venv`**: use
  `.\.venv\Scripts\python.exe worker.py` directly, or set
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current
  PowerShell window.
- **401 Unauthorized when claiming jobs**: `WORKER_TOKEN` mismatch between the
  worker `.env` and the backend Space secret, or the backend Space was not
  restarted after adding the secret.
- **Out of GPU memory**: lower `DEFAULT_BATCH` to `2` or lower `DEFAULT_IMGSZ`
  to `768` in `.env`, or override them per job from the platform.
- **Corporate firewall blocks downloads**: install dependencies on a network
  that allows PyPI and PyTorch wheel downloads, or ask IT to allow outbound HTTPS
  to PyPI, `download.pytorch.org`, Hugging Face, and GitHub/Torch Hub.
