

import argparse
import csv
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  TRACKING HYPERPARAMETERS
# ─────────────────────────────────────────────
CONF_HIGH       = 0.35   # τ_high  — primary association threshold
CONF_LOW        = 0.20   # τ_low   — second-pass recovery threshold
IOU_THRESHOLD   = 0.45   # NMS IoU threshold
TRACK_BUFFER    = 30     # frames to keep a lost track alive
MIN_HITS        = 3      # min detections before track is confirmed
MAX_AGE         = 30     # max frames without match before track is deleted

#   VERIFY THIS WITH A2's convert_uavdt_to_yolo.py
CLASS_NAMES = {0: "car", 1: "bus", 2: "truck"}

# Colour palette per class (BGR)
CLASS_COLORS = {
    0: (0,   200, 255),   # car   — amber
    1: (0,   255, 120),   # bus   — green
    2: (60,  60,  255),   # truck — red
}
DEFAULT_COLOR = (200, 200, 200)

TRAJ_MAX_LEN = 10   # reduced from 60 → short clean trails, no tangle


# ══════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════

def xyxy_to_tlwh(box: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] → [top-left-x, top-left-y, w, h]."""
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1])


def draw_bounding_box(frame, track_id, cls_id, box, conf):
    """Draw labelled bounding box on frame."""
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS.get(int(cls_id), DEFAULT_COLOR)
    label = f"{CLASS_NAMES.get(int(cls_id), 'obj')} #{track_id}  {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_trajectories(frame, trajectories):
    """Draw motion trails for each tracked object."""
    for track_id, pts in trajectories.items():
        if len(pts) < 2:
            continue
        color = CLASS_COLORS.get(pts[0][2], DEFAULT_COLOR)   # pts stores cls_id
        for i in range(1, len(pts)):
            alpha = int(255 * i / len(pts))
            faded = tuple(int(c * alpha / 255) for c in color)
            cv2.line(frame, pts[i - 1][:2], pts[i][:2], faded, 2)


def draw_hud(frame, frame_idx, fps, n_tracks, arch_name: str = "YOLOv11m"):
    """Overlay frame counter, FPS and active track count."""
    cv2.rectangle(frame, (0, 0), (280, 64), (20, 20, 20), -1)
    cv2.putText(frame, f"MilTrack  |  {arch_name} + ByteTrack",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(frame, f"Frame: {frame_idx:05d}  FPS: {fps:5.1f}",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(frame, f"Active tracks: {n_tracks}",
                (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 100), 1)


# ══════════════════════════════════════════════
#  CORE TRACKING FUNCTION
# ══════════════════════════════════════════════

def run_tracking(
    model: YOLO,
    source: str,
    output_path: str,
    conf_high: float = CONF_HIGH,
    conf_low: float  = CONF_LOW,
    iou_thresh: float = IOU_THRESHOLD,
    save_csv: bool = True,
    show: bool = False,
    arch_name: str = "YOLOv11m",
):
    """
    Run ByteTrack inference on an image or video source.

    Parameters
    ----------
    model       : loaded YOLO model
    source      : path to image or video file
    output_path : where to save the annotated output
    conf_high   : high-confidence threshold (τ_high)
    conf_low    : low-confidence threshold  (τ_low)
    iou_thresh  : NMS IoU threshold
    save_csv    : whether to write per-frame CSV log
    show        : display live preview window
    """

    src = Path(source)
    is_image = src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # ── Image mode ──────────────────────────────────────────
    if is_image:
        print(f"[MilTrack] Image mode → {src.name}")
        model.predictor = None   # reset tracker state
        frame = cv2.imread(str(src))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {src}")

        results = model.track(
            frame,
            persist=False,
            conf=conf_low,
            iou=iou_thresh,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        rows = []
        if results.boxes is not None and results.boxes.id is not None:
            for box, track_id, cls_id, conf in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.id.cpu().numpy().astype(int),
                results.boxes.cls.cpu().numpy().astype(int),
                results.boxes.conf.cpu().numpy(),
            ):
                draw_bounding_box(frame, track_id, cls_id, box, conf)
                rows.append([0, track_id, cls_id, CLASS_NAMES.get(cls_id, "obj"),
                              round(float(conf), 4), *map(int, box)])

        draw_hud(frame, 0, 0.0, len(rows), arch_name)
        cv2.imwrite(output_path, frame)
        print(f"[MilTrack] Saved annotated image → {output_path}")

        if save_csv:
            csv_path = str(Path(output_path).with_suffix(".csv"))
            _write_csv(csv_path, rows)

        if show:
            cv2.imshow("MilTrack", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ── Video mode ───────────────────────────────────────────
    print(f"[MilTrack] Video mode → {src.name}")
    model.predictor = None   # reset ByteTrack state between runs
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {src}")

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src,
        (W, H),
    )

    trajectories: dict[int, list] = defaultdict(list)   # track_id → [(cx,cy,cls)]
    all_rows = []
    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t_frame = time.perf_counter()

        results = model.track(
            frame,
            persist=True,           # maintain ByteTrack state across frames
            conf=conf_low,
            iou=iou_thresh,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        active = 0
        if results.boxes is not None and results.boxes.id is not None:
            for box, track_id, cls_id, conf in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.id.cpu().numpy().astype(int),
                results.boxes.cls.cpu().numpy().astype(int),
                results.boxes.conf.cpu().numpy(),
            ):
                if conf < conf_low:
                    continue

                tid, cid = int(track_id), int(cls_id)

                # High/low confidence split for ByteTrack two-stage logic
                # (Ultralytics ByteTrack handles this internally;
                #  we respect both thresholds at output stage)
                if conf >= conf_high:
                    # Confirmed detection — draw solid box
                    draw_bounding_box(frame, tid, cid, box, conf)
                else:
                    # Low-confidence recovery — draw dashed-style (thinner, faded)
                    x1, y1, x2, y2 = map(int, box)
                    color = CLASS_COLORS.get(cid, DEFAULT_COLOR)
                    faded = tuple(int(c * 0.55) for c in color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), faded, 1)

                # Trajectory centroid
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

        elapsed = time.perf_counter() - t_frame
        live_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        draw_hud(frame, frame_idx, live_fps, active, arch_name)

        writer.write(frame)

        if show:
            cv2.imshow("MilTrack", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[MilTrack] User interrupted.")
                break

        if frame_idx % 50 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  Frame {frame_idx}/{total}  ({pct:.1f}%)  "
                  f"active={active}  fps={live_fps:.1f}")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    total_time = time.perf_counter() - t0
    print(f"\n[MilTrack] Done — {frame_idx} frames in {total_time:.1f}s  "
          f"({frame_idx/total_time:.1f} fps avg)")
    print(f"[MilTrack] Output saved → {out_path}")

    if save_csv:
        csv_path = str(out_path.with_suffix(".csv"))
        _write_csv(csv_path, all_rows)


# ══════════════════════════════════════════════
#  CSV LOGGER
# ══════════════════════════════════════════════

def _write_csv(path: str, rows: list):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class_id", "class_name",
                    "confidence", "x1", "y1", "x2", "y2"])
        w.writerows(rows)
    print(f"[MilTrack] CSV log saved → {path}")


# ══════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="MilTrack — YOLOv11m + ByteTrack UAV vehicle tracker"
    )
    p.add_argument("--weights",  required=True,
                   help="Path to YOLOv11m trained weights (.pt)")
    p.add_argument("--source",   required=True,
                   help="Path to input image or video file")
    p.add_argument("--output",   default=None,
                   help="Output file path (auto-named if omitted)")
    p.add_argument("--arch-name", default="YOLOv11m",
                   help="Architecture label shown in HUD (default: YOLOv11m)")
    p.add_argument("--conf-high", type=float, default=CONF_HIGH,
                   help=f"High-conf threshold τ_high (default {CONF_HIGH})")
    p.add_argument("--conf-low",  type=float, default=CONF_LOW,
                   help=f"Low-conf threshold τ_low (default {CONF_LOW})")
    p.add_argument("--iou",       type=float, default=IOU_THRESHOLD,
                   help=f"NMS IoU threshold (default {IOU_THRESHOLD})")
    p.add_argument("--no-csv",    action="store_true",
                   help="Skip CSV log generation")
    p.add_argument("--show",      action="store_true",
                   help="Display live preview window")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Auto-generate output path if not given
    src = Path(args.source)
    if args.output is None:
        suffix = src.suffix if src.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"} else ".mp4"
        args.output = str(src.parent / f"{src.stem}_tracked{suffix}")

    print("=" * 56)
    print("  MilTrack  |  YOLOv11m (Config B)  +  ByteTrack")
    print("=" * 56)
    print(f"  Weights   : {args.weights}")
    print(f"  Source    : {args.source}")
    print(f"  Output    : {args.output}")
    print(f"  τ_high    : {args.conf_high}   τ_low: {args.conf_low}")
    print(f"  IoU thresh: {args.iou}")
    print("=" * 56)

    model = YOLO(args.weights)

    run_tracking(
        model       = model,
        source      = args.source,
        output_path = args.output,
        conf_high   = args.conf_high,
        conf_low    = args.conf_low,
        iou_thresh  = args.iou,
        save_csv    = not args.no_csv,
        show        = args.show,
        arch_name   = args.arch_name,
    )


if __name__ == "__main__":
    main()
