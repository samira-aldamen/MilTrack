"""
MilTrack — app.py
Gradio Web Interface for YOLOv11m + ByteTrack UAV Vehicle Detection & Tracking


Project: MilTrack — UAV Vehicle Detection & Tracking (UAVDT)

Features:
  - Image mode  : upload image → annotated image output + CSV download
  - Video mode  : upload video → annotated video output + CSV download
  - Threshold sliders : conf_high, conf_low, iou
  - imgsz toggle : 640 (fast) or 832 (better small objects)
  - TTA toggle   : image mode only

Usage (Colab):
  !pip install ultralytics gradio opencv-python-headless -q
  !python app.py
"""

import csv
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
WEIGHTS_PATH  = "best.pt"          # place best.pt next to app.py
MODEL_LABEL   = "YOLOv11m"

DEFAULT_CONF_HIGH = 0.35
DEFAULT_CONF_LOW  = 0.20
DEFAULT_IOU       = 0.45
DEFAULT_IMGSZ     = 640
TRAJ_MAX_LEN      = 10   # reduced from 60 → short clean trails, no tangle

#   Verify mapping with A2's convert_uavdt_to_yolo.py
CLASS_NAMES  = {0: "car", 1: "truck", 2: "bus"}
CLASS_COLORS = {
    0: (0,   200, 255),   # car   — amber (BGR)
    1: (60,  60,  255),   # truck — red
    2: (0,   255, 120),   # bus   — green
}
DEFAULT_COLOR = (200, 200, 200)

# ─────────────────────────────────────────────
#  LOAD MODEL (once at startup)
# ─────────────────────────────────────────────
print(f"[MilTrack] Loading {MODEL_LABEL} from {WEIGHTS_PATH} …")
model = YOLO(WEIGHTS_PATH)
print("[MilTrack] Model ready ✓")


# ══════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════

def draw_box(frame, track_id, cls_id, box, conf):
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
    label = f"{CLASS_NAMES.get(cls_id, 'obj')} #{track_id}  {conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_faded_box(frame, cls_id, box):
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
    faded = tuple(int(c * 0.5) for c in color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), faded, 1)


def draw_trajectories(frame, trajectories):
    for _, pts in trajectories.items():
        if len(pts) < 2:
            continue
        color = CLASS_COLORS.get(pts[0][2], DEFAULT_COLOR)
        for i in range(1, len(pts)):
            alpha = int(255 * i / len(pts))
            faded = tuple(int(c * alpha / 255) for c in color)
            cv2.line(frame, pts[i - 1][:2], pts[i][:2], faded, 2)


def draw_hud(frame, frame_idx, fps, n_tracks, imgsz, use_tta):
    tta_tag  = " +TTA"    if use_tta      else ""
    size_tag = f" @{imgsz}" if imgsz != 640 else ""
    cv2.rectangle(frame, (0, 0), (340, 80), (20, 20, 20), -1)
    cv2.putText(frame,
                f"MilTrack | {MODEL_LABEL}{tta_tag}{size_tag} + ByteTrack",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Frame: {frame_idx:05d}  FPS: {fps:5.1f}",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Active tracks: {n_tracks}",
                (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1)
    cv2.putText(frame, f"UAVDT  car/truck/bus  |  MilTrack v2",
                (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1)


# ══════════════════════════════════════════════
#  CSV HELPER
# ══════════════════════════════════════════════

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class_id", "class_name",
                    "confidence", "x1", "y1", "x2", "y2"])
        w.writerows(rows)


# ══════════════════════════════════════════════
#  IMAGE INFERENCE
# ══════════════════════════════════════════════

def run_image(
    image_path: str,
    conf_high: float,
    conf_low: float,
    iou: float,
    imgsz: int,
    use_tta: bool,
):
    model.predictor = None   # reset tracker state between runs

    frame = cv2.imread(image_path)
    if frame is None:
        raise gr.Error("❌ Cannot read image file.")

    results = model.track(
        frame,
        persist  = False,
        conf     = conf_low,
        iou      = iou,
        imgsz    = imgsz,
        augment  = use_tta,
        tracker  = "bytetrack.yaml",
        verbose  = False,
    )[0]

    rows = []
    n    = 0
    if results.boxes is not None and results.boxes.id is not None:
        for box, tid, cid, conf in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.id.cpu().numpy().astype(int),
            results.boxes.cls.cpu().numpy().astype(int),
            results.boxes.conf.cpu().numpy(),
        ):
            tid, cid = int(tid), int(cid)
            if conf >= conf_high:
                draw_box(frame, tid, cid, box, conf)
            else:
                draw_faded_box(frame, cid, box)
            rows.append([0, tid, cid, CLASS_NAMES.get(cid, "obj"),
                         round(float(conf), 4), *map(int, box)])
            n += 1

    draw_hud(frame, 0, 0.0, n, imgsz, use_tta)

    # Save outputs
    tmp_dir  = tempfile.mkdtemp()
    img_out  = os.path.join(tmp_dir, "tracked.jpg")
    csv_out  = os.path.join(tmp_dir, "tracked.csv")
    cv2.imwrite(img_out, frame)
    write_csv(csv_out, rows)

    # Convert BGR → RGB for Gradio display
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    stats = _stats_text(rows, is_video=False)
    return rgb, csv_out, stats


# ══════════════════════════════════════════════
#  VIDEO INFERENCE
# ══════════════════════════════════════════════

def run_video(
    video_path: str,
    conf_high: float,
    conf_low: float,
    iou: float,
    imgsz: int,
):
    model.predictor = None   # reset ByteTrack state between runs

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("❌ Cannot open video file.")

    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_dir   = tempfile.mkdtemp()
    raw_out   = os.path.join(tmp_dir, "tracked_raw.mp4")
    final_out = os.path.join(tmp_dir, "tracked.mp4")
    csv_out   = os.path.join(tmp_dir, "tracked.csv")

    writer = cv2.VideoWriter(
        raw_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src, (W, H),
    )

    trajectories = defaultdict(list)
    all_rows  = []
    frame_idx = 0
    t0        = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t_frame = time.perf_counter()

        results = model.track(
            frame,
            persist  = True,
            conf     = conf_low,
            iou      = iou,
            imgsz    = imgsz,
            augment  = False,        # TTA disabled for video
            tracker  = "bytetrack.yaml",
            verbose  = False,
        )[0]

        active = 0
        if results.boxes is not None and results.boxes.id is not None:
            for box, tid, cid, conf in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.id.cpu().numpy().astype(int),
                results.boxes.cls.cpu().numpy().astype(int),
                results.boxes.conf.cpu().numpy(),
            ):
                tid, cid = int(tid), int(cid)
                if conf >= conf_high:
                    draw_box(frame, tid, cid, box, conf)
                else:
                    draw_faded_box(frame, cid, box)

                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                traj = trajectories[tid]
                traj.append((cx, cy, cid))
                if len(traj) > TRAJ_MAX_LEN:
                    traj.pop(0)

                active += 1
                all_rows.append([frame_idx, tid, cid,
                                  CLASS_NAMES.get(cid, "obj"),
                                  round(float(conf), 4),
                                  *map(int, box)])

        draw_trajectories(frame, trajectories)
        elapsed  = time.perf_counter() - t_frame
        live_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        draw_hud(frame, frame_idx, live_fps, active, imgsz, False)
        writer.write(frame)

    cap.release()
    writer.release()

    # Convert to H.264 for browser playback
    os.system(
        f"ffmpeg -y -i {raw_out} -vcodec libx264 -crf 23 "
        f"{final_out} -loglevel quiet"
    )
    if not os.path.exists(final_out):
        final_out = raw_out   # fallback if ffmpeg unavailable

    write_csv(csv_out, all_rows)

    total_time = time.perf_counter() - t0
    avg_fps    = frame_idx / total_time if total_time > 0 else 0
    stats = _stats_text(all_rows, is_video=True,
                        n_frames=frame_idx, avg_fps=avg_fps)
    return final_out, csv_out, stats


# ══════════════════════════════════════════════
#  STATS SUMMARY TEXT
# ══════════════════════════════════════════════

def _stats_text(rows, is_video=False, n_frames=1, avg_fps=0.0):
    counts = {0: 0, 1: 0, 2: 0}
    ids    = set()
    for r in rows:
        cid = r[2]
        counts[cid] = counts.get(cid, 0) + 1
        ids.add(r[1])

    lines = [
        f"**Total detections :** {len(rows)}",
        f"**Unique track IDs:** {len(ids)}",
        f"**car detections  :** {counts.get(0,0)}",
        f"**truck detections:** {counts.get(1,0)}",
        f"**bus detections  :** {counts.get(2,0)}",
    ]
    if is_video:
        lines += [
            f"**Frames processed:** {n_frames}",
            f"**Avg FPS         :** {avg_fps:.1f}",
        ]
    return "\n".join(lines)


# ══════════════════════════════════════════════
#  GRADIO WRAPPERS
# ══════════════════════════════════════════════

def gradio_image(image, conf_high, conf_low, iou, imgsz, use_tta):
    if image is None:
        raise gr.Error(" Please upload an image first.")
    # Gradio passes numpy RGB → save to tmp file
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(tmp.name, bgr)
    return run_image(tmp.name, conf_high, conf_low, iou, int(imgsz), use_tta)


def gradio_video(video, conf_high, conf_low, iou, imgsz):
    if video is None:
        raise gr.Error(" Please upload a video first.")
    return run_video(video, conf_high, conf_low, iou, int(imgsz))


# ══════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════

CSS = """
#title { text-align: center; }
.stat-box { font-family: monospace; font-size: 14px; }
"""

with gr.Blocks(title="MilTrack — UAV Vehicle Tracker", css=CSS) as demo:

    gr.Markdown(
        """
        #  MilTrack — UAV Military Vehicle Detection & Tracking
        **YOLOv11m + ByteTrack  |  UAVDT Dataset  |  Classes: car · truck · bus**
        """,
        elem_id="title",
    )

    # ── SHARED CONTROLS ──────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("###  Detection Settings")
            conf_high = gr.Slider(0.10, 0.90, value=DEFAULT_CONF_HIGH,
                                  step=0.05, label="Confidence High (τ_high)",
                                  info="Solid box threshold")
            conf_low  = gr.Slider(0.05, 0.80, value=DEFAULT_CONF_LOW,
                                  step=0.05, label="Confidence Low (τ_low)",
                                  info="ByteTrack recovery threshold")
            iou       = gr.Slider(0.10, 0.90, value=DEFAULT_IOU,
                                  step=0.05, label="IoU Threshold (NMS)")
            imgsz     = gr.Radio([640, 832],
                                  value=DEFAULT_IMGSZ,
                                  label="Inference Resolution",
                                  info="832 = better small objects, ~70% slower")

    # ── TABS: IMAGE / VIDEO ───────────────────────────────────
    with gr.Tabs():

        # ── IMAGE TAB ────────────────────────────────────────
        with gr.Tab("🖼️  Image"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(
                        type="numpy", label="Upload Image (JPG / PNG)",
                        height=320,
                    )
                    use_tta = gr.Checkbox(
                        value=True, label="Enable TTA (Test-Time Augmentation)",
                        info="Improves recall on small objects — no retraining needed",
                    )
                    img_btn = gr.Button("🚀 Run Detection", variant="primary")

                with gr.Column():
                    img_output  = gr.Image(label="Annotated Output", height=320)
                    img_csv     = gr.File(label="📥 Download CSV Log")
                    img_stats   = gr.Markdown(elem_classes="stat-box")

            img_btn.click(
                fn      = gradio_image,
                inputs  = [img_input, conf_high, conf_low, iou, imgsz, use_tta],
                outputs = [img_output, img_csv, img_stats],
            )

        # ── VIDEO TAB ────────────────────────────────────────
        with gr.Tab("🎬  Video"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video (MP4 / AVI / MOV)")
                    gr.Markdown(
                        "> ⚠️ TTA is **disabled** for video — it breaks ByteTrack "
                        "track IDs across frames. Use **imgsz=832** instead for "
                        "better small-object recall."
                    )
                    vid_btn = gr.Button("🚀 Run Tracking", variant="primary")

                with gr.Column():
                    vid_output = gr.Video(label="Annotated Output")
                    vid_csv    = gr.File(label="📥 Download CSV Log")
                    vid_stats  = gr.Markdown(elem_classes="stat-box")

            vid_btn.click(
                fn      = gradio_video,
                inputs  = [vid_input, conf_high, conf_low, iou, imgsz],
                outputs = [vid_output, vid_csv, vid_stats],
            )

    # ── FOOTER ───────────────────────────────────────────────
    gr.Markdown(
        """
        ---
        **MilTrack** · Computer Vision Course AI447 · Spring 2025-2026  
        Dataset: [UAVDT](https://sites.google.com/view/grli-uavdt) · 
        Model: YOLOv11m (Config B) · Tracker: ByteTrack
        """
    )

# ══════════════════════════════════════════════
#  LAUNCH
# ══════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        share          = True,    # generates public URL — useful on Colab
        server_name    = "0.0.0.0",
        server_port    = 7860,
        show_error     = True,
    )
