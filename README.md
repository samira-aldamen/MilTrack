#  MilTrack — UAV Military Vehicle Detection & Tracking

> A Computer Vision System for Aerial Military Vehicle Detection and Tracking  
> **Course:** AI447 — Computer Vision | Deep Learning – Spring 2025/2026  
> **Instructor:** Dr. Abdullah Alamaren  
> **University:** Jordan University of Science and Technology

| Name | ID |
|---|---|
| Wajd Almadi | 172834 |
| Samira Aldamen | 170587 |
| Batool Mahashi | 172901 |

---

##  Project Overview

MilTrack is an end-to-end system for detecting and tracking military ground vehicles (cars, buses, trucks) in UAV aerial footage. It uses **YOLOv11m** for object detection and **ByteTrack** for multi-object tracking, deployed through a **Gradio** web interface.

The system operates on the **UAVDT benchmark dataset** and handles three environmental conditions: day, night, and fog — across low, medium, and high UAV flight altitudes.

---

##  Project Structure

```
MilTrack/
│
├── app/
│   └── app.py                  # Gradio web interface (main entry point)
├── cli/
│   └── track.py                # CLI interface for batch processing
├── training/
│   └── cv_project.py           # Training / experimentation script (exported from Colab)
├── assets/demo/
│   ├── tracked_v4.mp4          # Demo output video
│   └── tracked_v4.csv          # Demo detection/tracking log
├── docs/
│   ├── Project_Report.docx
│   ├── PowerPoint_Slides.pptx
│   └── Best_Model_Weight_Info.docx
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── .gitignore
```

> **Note:** The trained `best.pt` weights file itself is not included in this repository (see `docs/Best_Model_Weight_Info.docx` for details on how it was produced). Place your own `best.pt` alongside `app/app.py` and `cli/track.py` before running, or update the path in those scripts.

---

##  Installation & Setup

### Prerequisites

- Python 3.8 or higher
- A GPU is strongly recommended (NVIDIA Tesla T4 or A100 for real-time performance)
- Google Colab is the recommended environment

### Step 1 — Clone or Download the Project

```bash
git clone <your-repo-url>
cd MilTrack
```

Or simply upload all files to your Google Colab session.

### Step 2 — Install Required Libraries

```bash
pip install ultralytics gradio opencv-python
```

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv11 model loading, inference, and ByteTrack integration |
| `gradio` | Web interface framework |
| `opencv-python` | Image and video I/O and processing |

### Step 3 — Add Model Weights

Make sure `best.pt` (the trained model weights file) is placed in the **same directory** as `app/app.py` (and `cli/track.py` for the CLI).

---

##  How to Run the System

### Option A — Gradio Web Interface (Recommended for Demo)

```bash
python app/app.py
```

This will launch a Gradio app. Since it runs with `share=True`, a **public URL** will be printed in the terminal — open it in any browser without any port configuration.

**Running on Google Colab:**

```python
!python app/app.py
```

The public Gradio URL will appear in the cell output. Click it to open the interface.

---

### Option B — Command-Line Interface (Recommended for Batch Processing)

```bash
python cli/track.py --source <path_to_video_or_image> [options]
```

**Example:**

```bash
# Process a video
python cli/track.py --source input_video.mp4

# Process an image
python cli/track.py --source image.jpg --conf-high 0.40
```

**Available CLI Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--source` | *(required)* | Path to input image or video file |
| `--conf-high` | `0.35` | High confidence threshold for confirmed detections |
| `--conf-low` | `0.20` | Low confidence threshold for ByteTrack recovery stage |
| `--iou` | `0.45` | NMS IoU overlap threshold |
| `--imgsz` | `640` | Inference resolution (`640` or `832`) |
| `--output` | auto-generated | Path to save annotated output file |

---

##  How to Use the Interface

### Detection Settings Panel (applies to both tabs)

| Control | Default | Range | Description |
|---|---|---|---|
| **Confidence High (τ_high)** | 0.35 | 0.10 – 0.90 | Threshold for confirmed bounding boxes (shown as solid colored boxes with labels) |
| **Confidence Low (τ_low)** | 0.20 | 0.05 – 0.80 | ByteTrack second-stage recovery threshold (shown as faded thin boxes) |
| **IoU Threshold (NMS)** | 0.45 | 0.10 – 0.90 | Controls overlap suppression — lower values remove more overlapping boxes |
| **Inference Resolution** | 640 | 640 or 832 | Resolution for YOLOv11 forward pass — 832 improves small object detection at ~70% extra inference time |
| **TTA** | ON | ON / OFF | Test-Time Augmentation (flip + scale ensemble) — auto-disabled for videos to preserve track ID consistency |

---

###  Image Tab — Single Frame Detection

**Input format:** JPG or PNG image

**Steps:**
1. Click on the **Image** tab
2. Upload your image (drag & drop or click to upload)
3. Adjust detection settings if needed
4. (Optional) Enable **TTA** for better detection accuracy
5. Click **"Run Detection"**

**Expected Output:**
- Annotated image with **colored bounding boxes**, **class labels**, **track IDs**, and a **HUD overlay**
- A **downloadable CSV file** containing detection logs
- A summary showing: total detected objects, unique tracks, and per-class counts (car / bus / truck)

---

###  Video Tab — Multi-Object Tracking

**Input format:** MP4, AVI, or MOV

**Steps:**
1. Click on the **Video** tab
2. Upload your video file
3. Adjust detection settings as needed
4. Click **"Run Tracker"**

**Expected Output:**
- Annotated output video (H.264 compressed via ffmpeg) with:
  - Bounding boxes for each detected vehicle
  - Unique **Track IDs** maintained across frames
  - **Motion trajectories** drawn as overlays
  - HUD information (class counts, FPS, etc.)
- The output video is displayed in the browser and available for download

>  **Note:** TTA is automatically disabled in video mode to maintain ByteTrack's Kalman filter state and consistent track IDs across frames.

---

##  Detected Classes

MilTrack detects the following vehicle categories:

| Class | Description |
|---|---|
| `car` | Standard civilian/military cars |
| `bus` | Buses and large passenger vehicles |
| `truck` | Military and cargo trucks |

---

##  Important Notes & Limitations

### Known Limitations

1. **Class Imbalance:** The system may under-detect trucks and buses. The UAVDT dataset is heavily skewed toward cars (~80%), which causes the model to lean toward car predictions. Lowering `τ_high` below the default can improve truck/bus recall.

2. **Real-Time Performance:**
   - **YOLOv11l** achieves ~67 FPS on NVIDIA T4 — well above the 30 FPS target 
   - **RT-DETR-l** achieves ~26 FPS on NVIDIA T4 — slightly below real-time threshold 

3. **Input Resolution:** All models are trained at 640×640. High-altitude vehicles may be as small as 20×20 pixels in the model's feature space, which can reduce detection confidence.

4. **Track ID Persistence:** Track IDs are reset between sessions. After each video inference, the tracker is reinitialized (`model.predictor = None`) to prevent ID contamination across different input videos.

5. **GPU Requirement:** While the system can run on CPU, real-time performance (≥30 FPS) requires a CUDA-capable NVIDIA GPU.

### Tips for Better Results

- For **dense convoy scenes**, lower the IoU threshold to reduce overlapping boxes
- For **high-altitude footage**, switch inference resolution to **832** for better small object detection
- To improve **truck/bus detection**, try reducing `τ_high` to around 0.25–0.30
- For **batch processing** of multiple UAVDT test sequences, use `track.py` (CLI) instead of the Gradio interface

---

##  Performance Benchmarks

| Model | mAP@0.5 | FPS (T4) |
|---|---|---|
| YOLOv11n |0.733 |~100
| YOLOv11m | 0.788 |~67
| YOLOv11l | 0.786 | ~67 |
| RT-DETR-l | 0.723 | ~26 |

**Tracking targets (UAVDT test set):**
- MOTA ≥ 55%
- IDF1 ≥ 60%

---

##  Dataset

The system is trained and evaluated on the **UAVDT Benchmark** — a large-scale UAV video dataset with labeled military and civilian vehicles under varied environmental settings (day/night/fog) and camera altitudes (low/medium/high).

---

##  Reproducibility

All experiments are reproducible using the provided `best.pt` model weights and configuration files. The system follows IEEE 830-1998 requirement conventions for full traceability.

---

*MilTrack — AI447 Computer Vision Project | JUST

