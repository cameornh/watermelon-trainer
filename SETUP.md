# Training computer setup (Windows 11 + WSL2 + RTX PRO 4000 Blackwell)

This is the complete, step-by-step guide to turn your workstation into the GPU
training worker for the fruit-phenotyping platform. Follow it once. After that,
the worker just runs and drains the platform's training queue.

- **Target machine:** Windows 11 Pro, NVIDIA RTX PRO 4000 Blackwell Workstation
Edition (24 GB), 24 GB system RAM.
- **Runtime:** WSL2 (Ubuntu). The GPU is used from inside WSL via the Windows
NVIDIA driver. **Do not install a separate NVIDIA driver inside WSL.**
- **Connectivity:** outbound HTTPS only. No port-forwarding, no public IP, no
tunnel. The worker calls the backend; nothing connects in.

> You can copy this entire `watermelon_trainer` folder to the machine as-is. It
> does not import any backend code.

---

## 0. What you'll need

- Admin access on the Windows machine.
- The backend Space URL, e.g. `https://PPAL-SongLab-UGA-fruit-analyzer.hf.space`.
- A `WORKER_TOKEN` secret (any long random string). You will set the SAME value
on the backend Space and in the worker's `.env`. Generate one, e.g.:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

---

## 1. Update the NVIDIA Windows driver

The Blackwell GPU (compute capability **sm_120**) needs a recent driver that
also provides CUDA inside WSL2.

1. Download the latest **NVIDIA RTX / Workstation** driver for the "RTX PRO 4000
  Blackwell" from nvidia.com and install it on **Windows** (not in WSL).
2. Reboot.
3. Open PowerShell and confirm the driver sees the GPU:
  ```powershell
   nvidia-smi
  ```
   You should see the RTX PRO 4000 and a CUDA version (>= 12.8 recommended).

---

## 2. Enable WSL2 and install Ubuntu

In an **admin PowerShell**:

```powershell
wsl --install -d Ubuntu
wsl --set-default-version 2
```

Reboot if prompted, then launch "Ubuntu" from the Start menu and create your
Linux username/password. Confirm you are on WSL2:

```powershell
wsl -l -v      # VERSION should be 2 for Ubuntu
```

Update the distro (inside Ubuntu):

```bash
sudo apt update && sudo apt upgrade -y
```

---

## 3. Verify the GPU is visible inside WSL

Inside Ubuntu:

```bash
nvidia-smi
```

This must list the RTX PRO 4000. If it does not:

- make sure the **Windows** driver from step 1 is installed,
- update WSL itself from PowerShell: `wsl --update`, then `wsl --shutdown` and
reopen Ubuntu.

Again: do **not** `apt install` any `nvidia-driver-`* inside WSL. WSL uses the
Windows driver. You only need the CUDA *runtime* that ships inside the PyTorch
wheels (installed in step 6).

---

## 4. Install Python and tooling (inside Ubuntu)

```bash
sudo apt install -y python3 python3-venv python3-pip git
python3 --version    # 3.10, 3.11, or 3.12 are all fine
```

---

## 5. Copy the worker folder over

Put the `watermelon_trainer` folder somewhere in your Linux home (not on the
slow `/mnt/c` Windows mount). For example, from Ubuntu:

```bash
mkdir -p ~/platform
cp -r /mnt/c/Users/<you>/Downloads/watermelon_trainer ~/platform/
cd ~/platform/watermelon_trainer
```

(Adjust the source path to wherever you transferred the folder.)

---

## 6. Create a venv and install dependencies (cu128)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pulls `torch`/`torchvision` from the **cu128** wheel index,
which is required for the Blackwell `sm_120` GPU. Older cu121 builds will fail at
runtime with `no kernel image is available for execution on the device`.

The first run will also download the DINOv2 embedder weights from torch hub.

---

## 7. Verify CUDA + the Blackwell GPU

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

Expected:

- `cuda True`
- the RTX PRO 4000 device name
- `capability (12, 0)` (sm_120)

If `cuda` is `False`, revisit steps 1 and 3.

---

## 8. Set the backend secret

On the backend Hugging Face Space, add a **Secret** named `WORKER_TOKEN` with the
value you generated in step 0 (Space → Settings → Variables and secrets). Restart
the Space so it picks up the secret. Without it, the backend rejects every worker
request (this is the intended safe default).

---

## 9. Configure the worker `.env`

```bash
cp .env.example .env
nano .env
```

Set at least:

```
BACKEND_URL=https://PPAL-SongLab-UGA-fruit-analyzer.hf.space
WORKER_TOKEN=<the same token you set on the backend>
TRAIN_DEVICE=0
```

You can tune `DEFAULT_EPOCHS`, `DEFAULT_IMGSZ`, `DEFAULT_BATCH`, and
`POLL_INTERVAL_SECONDS` later. Note: each enqueued job can override the
hyperparameters, so these are only fallbacks.

---

## 10. Run the worker

```bash
source .venv/bin/activate
python worker.py
```

You should see:

```
[worker HH:MM:SS] Trainer worker starting. Backend=https://...
[worker HH:MM:SS] CUDA available: NVIDIA RTX PRO 4000 ... (device=0)
```

Leave it running. When you click **Fine-tune from corrections** (or onboard a new
fruit) in the platform, the worker will claim the job, download the data bundle,
train + validate on the GPU, and upload the result. The backend then runs the
eval gate and promotes the model if it beats the incumbent — all visible in the
Labeling Studio training-status bar.

To do a quick dry run without a GPU (e.g. on a laptop) set `TRAIN_DEVICE=cpu`;
training will be slow but the full loop will work.

---

## 11. (Optional) Autostart

### Option A — systemd inside WSL2

Modern WSL2 supports systemd. Ensure it is enabled in `/etc/wsl.conf`:

```ini
[boot]
systemd=true
```

(Then `wsl --shutdown` from PowerShell and reopen Ubuntu.)

Create `~/.config/systemd/user/wm-trainer.service`:

```ini
[Unit]
Description=Watermelon training worker
After=network-online.target

[Service]
WorkingDirectory=%h/platform/watermelon_trainer
ExecStart=%h/platform/watermelon_trainer/.venv/bin/python worker.py
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

Enable it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now wm-trainer.service
loginctl enable-linger $USER   # keep it running when you're not logged in
journalctl --user -u wm-trainer -f   # follow logs
```

### Option B — Windows Task Scheduler

Create a Basic Task that runs at logon with the action:

- Program: `wsl.exe`
- Arguments: `-d Ubuntu -- bash -lc "cd ~/platform/watermelon_trainer && ./.venv/bin/python worker.py"`

---

## 12. Troubleshooting

- `**no kernel image is available for execution on the device**` — you installed
a non-cu128 PyTorch. Reinstall: `pip uninstall -y torch torchvision` then
`pip install -r requirements.txt` (which uses the cu128 index).
- `**torch.cuda.is_available()` is False** — Windows driver missing/old, or WSL
needs `wsl --update` + `wsl --shutdown`. Do not install a driver inside WSL.
- `**xFormers is not available` warnings** — harmless; DINOv2 falls back to plain
attention.
- **401 Unauthorized when claiming jobs** — `WORKER_TOKEN` mismatch between the
worker `.env` and the backend Space secret, or the Space wasn't restarted after
adding the secret.
- **Out of GPU memory** — lower `DEFAULT_BATCH` (e.g. 2) and/or `DEFAULT_IMGSZ`
(e.g. 768) in `.env`, or set them per job from the platform.
- **Slow file access** — keep the folder in the Linux filesystem (`~/...`), not
under `/mnt/c/...`.

