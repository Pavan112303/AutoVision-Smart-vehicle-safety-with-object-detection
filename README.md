# YOLOv5 Web Interface

Browser-based real-time detection, built on Ultralytics YOLOv5 and a Flask backend.

---

## Overview

This project wraps the upstream `ultralytics/yolov5` codebase with a Flask UI (`app.py`) so that you can upload media, stream from a webcam, and download annotated outputs without touching the CLI. The server initializes a `DetectMultiBackend` model once, keeps it resident on CPU/GPU, and serves requests through `/upload`, `/results/<file>`, `/clear`, and `/health` routes.

---

## How It Works

1. **Startup** – `python app.py --weights yolov5s.pt --device 0` loads the requested weights via `DetectMultiBackend`, warms the model to avoid first-request latency, and builds upload/result directories.
2. **Request Flow** – Files submitted in the UI are saved to `uploads/`, preprocessed to 640×640 with padding (`preprocess_image`), inferred with the loaded model, and drawn using `Annotator`.
3. **Outputs** – Images land in `results/<uuid>_*.jpg`; videos are re-encoded to MP4 with bounding boxes and detection summaries. JSON metadata (classes, confidences, counts) is returned with each response.
4. **Live Camera** – When enabled, a background thread opens the default webcam, performs the same preprocessing/inference loop, and streams JPEG bytes.

---

## Prerequisites

- Python 3.9–3.12
- pip 23+
- PyTorch with CUDA (optional but strongly recommended)
- Git, FFmpeg (optional, only for certain video codecs)

Install the core dependencies inside a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate           # PowerShell
pip install --upgrade pip wheel
pip install flask torch torchvision torchaudio opencv-python numpy matplotlib ultralytics
```

Download the weights you plan to serve (e.g., `yolov5s.pt`) into the repository root. Other weights can be dropped into the same directory and referenced via `--weights`.

---

## Running the Server

```bash
cd yolov5
python app.py --weights yolov5s.pt --device 0 --host 0.0.0.0 --port 5000
```

- Use `--device cpu` to force CPU inference.
- Lower-end GPUs benefit from `--weights yolov5n.pt`.
- Logs list the active device, weight file, and camera availability.

---

## Using the Web UI

- **Upload media** – Drag/drop or browse for images (PNG/JPG/BMP/GIF) and videos (MP4/AVI/MOV/MKV/WebM). Adjust confidence, IoU, and image size before submitting.
- **Monitor progress** – The UI displays per-object statistics, total counts, and an annotated preview once processing completes.
- **Download results** – Use the provided link to fetch processed media from `results/`.
- **Clear storage** – Trigger the “Clear Files” action to wipe `uploads/` and `results/` directories without restarting the server.
- **Health check** – Hit `/health` to confirm the model is resident and whether the camera thread is active (useful for probes).

---

## Project Structure

```
yolov5/
├── app.py                # Flask server + detection pipeline
├── templates/            # Jinja templates (UI)
├── static/               # CSS/JS and client assets
├── models/, utils/, ...  # Upstream YOLOv5 core
├── data/, segment/, classify/  # Ancillary tasks (training, segmentation, classification)
├── uploads/              # Runtime user uploads (auto-created)
├── results/              # Runtime annotated outputs (auto-created)
└── runs/                 # Experiment artifacts from CLI scripts
```

---

## Generated & Optional Files

| Path | Why it can be removed |
| --- | --- |
| `.vscode/settings.json` | Local editor preferences; regenerate per developer. |
| `__pycache__/` | Python bytecode caches; recreated automatically. |
| `uploads/` | Only contains user-submitted files from the current session. |
| `results/` | Derived artifacts (annotated media) already downloadable from the UI. |
| `runs/detect/exp*` | Historical experiment logs from CLI runs; not needed for the web app. |
| `yolov5x.pt` | Extra-large weights; keep only the weight files you actively serve to reduce repo size. |

Add the above to `.gitignore` or delete them before packaging/deploying.

---

## Extending the App

- Swap templates in `templates/index.html` to rebrand the UI.
- Expose additional model parameters by forwarding extra form fields into `detect_objects`.
- Wire RTSP/RTMP inputs by adding loader logic in `detect_objects` similar to the video branch.

---

## Troubleshooting

- **Model fails to load** – Ensure the `.pt` file resides next to `app.py` and matches your PyTorch build (CUDA vs CPU).
- **CUDA OOM** – Reduce `--img-size`, switch to smaller weights, or run with `--device cpu`.
- **Browser can’t reach server** – Verify `host`/`port` flags and that Windows Firewall allows inbound connections.

---

**YOLOv5 Web Interface – making computer vision accessible through the browser.**