"""
People Flow Detection: YOLOv8 + ByteTrack + two-line IN/OUT counting + heatmap.

Install:
    pip install ultralytics supervision opencv-python numpy
    (ffmpeg must be installed for the H.264 conversion; Colab has it already)

Run:
    python people_flow.py
"""
import os
import shutil
import subprocess
import urllib.request

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# ---------------------------------------------------------------- config
VIDEO_URL = "https://media.roboflow.com/supervision/video-examples/people-walking.mp4"
VIDEO_IN = "people-walking.mp4"
VIDEO_RAW = "_raw_output.mp4"              # temporary (mp4v, not widely playable)
VIDEO_OUT = "output_people_flow.mp4"       # final H.264 file (plays everywhere)
HEATMAP_OUT = "final_heatmap.png"
HEATMAP_OVERLAY_OUT = "final_heatmap_overlay.png"

MODEL_WEIGHTS = "yolov8n.pt"   # try yolov8s.pt / yolov8m.pt for better accuracy
CONF = 0.3

# Two horizontal lines, as (x1, y1, x2, y2) in pixels.
# Set to None to auto-place them at 40% and 60% of frame height, or paste
# the coordinates you get from https://polygonzone.roboflow.com/ here.
UPPER_LINE = None   # e.g. (0, 400, 1920, 400)
LOWER_LINE = None   # e.g. (0, 650, 1920, 650)

UPPER_COLOR = (0, 255, 0)     # green (BGR) -> IN line
LOWER_COLOR = (0, 0, 255)     # red   (BGR) -> OUT line


# ---------------------------------------------------------------- helpers
def ensure_video():
    if not os.path.exists(VIDEO_IN):
        print("Downloading video...")
        urllib.request.urlretrieve(VIDEO_URL, VIDEO_IN)


def make_heat_kernel(radius=25):
    """Gaussian blob added to the heat accumulator at each person's center."""
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * (radius / 2.5) ** 2))
    return kernel.astype(np.float32)


def add_heat(heat, cx, cy, kernel):
    r = kernel.shape[0] // 2
    h, w = heat.shape
    x1, x2 = max(cx - r, 0), min(cx + r + 1, w)
    y1, y2 = max(cy - r, 0), min(cy + r + 1, h)
    if x1 >= x2 or y1 >= y2:
        return
    kx1, ky1 = x1 - (cx - r), y1 - (cy - r)
    kx2, ky2 = kx1 + (x2 - x1), ky1 + (y2 - y1)
    heat[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]


def draw_counter_panel(frame, count_in, count_out):
    cv2.rectangle(frame, (10, 10), (250, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"IN : {count_in}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, UPPER_COLOR, 2)
    cv2.putText(frame, f"OUT: {count_out}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, LOWER_COLOR, 2)


def convert_to_h264(src, dst):
    """Re-encode to H.264 so the video plays in browsers, phones and players."""
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found. Keeping the mp4v file; it may not play.")
        os.replace(src, dst)
        return
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         "-preset", "fast", "-movflags", "+faststart", dst],
        check=True,
    )
    os.remove(src)


# ---------------------------------------------------------------- main
def main():
    ensure_video()

    cap = cv2.VideoCapture(VIDEO_IN)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {W}x{H} @ {fps:.1f} fps")

    upper = UPPER_LINE or (0, int(H * 0.40), W, int(H * 0.40))
    lower = LOWER_LINE or (0, int(H * 0.60), W, int(H * 0.60))
    y_upper, y_lower = upper[1], lower[1]
    print(f"Upper line (IN):  {upper}")
    print(f"Lower line (OUT): {lower}")

    model = YOLO(MODEL_WEIGHTS)
    tracker = sv.ByteTrack(frame_rate=int(fps))

    writer = cv2.VideoWriter(
        VIDEO_RAW, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )

    heat = np.zeros((H, W), dtype=np.float32)
    kernel = make_heat_kernel()
    last_frame = None

    prev_pos = {}        # track_id -> previous (cx, cy)
    counted_in = set()   # IDs already counted as IN
    counted_out = set()  # IDs already counted as OUT

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Detect people (COCO class 0 = person)
        result = model(frame, classes=[0], conf=CONF, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # 2) Track with ByteTrack -> persistent IDs
        detections = tracker.update_with_detections(detections)

        for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)
            tid = int(tid)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 3) Counting logic: compare previous and current center y
            if tid in prev_pos:
                py = prev_pos[tid][1]
                # moving down and crossing the upper line -> IN
                if py < y_upper <= cy and tid not in counted_in:
                    counted_in.add(tid)
                # moving up and crossing the lower line -> OUT
                if py > y_lower >= cy and tid not in counted_out:
                    counted_out.add(tid)
            prev_pos[tid] = (cx, cy)

            # 4) Heatmap accumulation
            add_heat(heat, cx, cy, kernel)

            # Drawing
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        cv2.line(frame, upper[:2], upper[2:], UPPER_COLOR, 3)
        cv2.line(frame, lower[:2], lower[2:], LOWER_COLOR, 3)
        cv2.putText(frame, "IN line", (W - 140, y_upper - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, UPPER_COLOR, 2)
        cv2.putText(frame, "OUT line", (W - 150, y_lower + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, LOWER_COLOR, 2)
        draw_counter_panel(frame, len(counted_in), len(counted_out))

        writer.write(frame)
        last_frame = frame.copy()

    cap.release()
    writer.release()

    # Convert to a playable H.264 video
    convert_to_h264(VIDEO_RAW, VIDEO_OUT)

    # 5) Final heatmap
    norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm = cv2.GaussianBlur(norm, (0, 0), 15)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite(HEATMAP_OUT, colored)

    if last_frame is not None:
        overlay = cv2.addWeighted(last_frame, 0.5, colored, 0.5, 0)
        cv2.imwrite(HEATMAP_OVERLAY_OUT, overlay)

    print(f"Final -> IN: {len(counted_in)} | OUT: {len(counted_out)}")
    print(f"Saved: {VIDEO_OUT}, {HEATMAP_OUT}, {HEATMAP_OVERLAY_OUT}")


if __name__ == "__main__":
    main()