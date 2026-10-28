# People Flow Detection using Object Tracking & Heatmap Visualization

Counts people crossing two horizontal lines in a video, shows live IN/OUT counters, and produces a motion/presence heatmap.

**Input video:** https://media.roboflow.com/supervision/video-examples/people-walking.mp4 (1920x1080, 25 fps)

## Setup & Run

```bash
pip install ultralytics supervision opencv-python numpy
python people_flow.py
```

`ffmpeg` must be installed (it is already available in Google Colab). The script downloads the video automatically and writes:

| File | Description |
|---|---|
| `output_people_flow.mp4` | H.264 video with boxes, IDs, color-coded lines and live IN/OUT counters |
| `final_heatmap.png` | Standalone heatmap |
| `final_heatmap_overlay.png` | Heatmap overlaid on a clean frame of the scene |

## Detection Method

- **Detector:** YOLOv8n (Ultralytics, COCO pre-trained), filtered to class `0` (person), confidence >= 0.3.
- **Tracker:** ByteTrack (via `supervision`), which assigns a persistent ID to each person across frames.
- **Position:** the center of each bounding box, `(cx, cy)`, is the tracked point.
- A dictionary `prev_pos = {track_id: (cx, cy)}` stores each person's previous position.

## Line Coordinates

Two horizontal lines span the full frame width (frame size 1920x1080).

| Line | Color | Purpose | Coordinates `(x1, y1, x2, y2)` |
|---|---|---|---|
| Upper | Green | IN | `(0, 432, 1920, 432)` |
| Lower | Red | OUT | `(0, 648, 1920, 648)` |

These are placed at 40% and 60% of the frame height. Custom coordinates (for example from
https://polygonzone.roboflow.com/) can be set with `UPPER_LINE` and `LOWER_LINE` at the top of `people_flow.py`.

## IN / OUT Logic

For each tracked person, with previous center y = `py` and current center y = `cy`:

- **IN:** `py < y_upper <= cy`. The person was above the upper line and is now at or below it (moving **down**).
- **OUT:** `py > y_lower >= cy`. The person was below the lower line and is now at or above it (moving **up**).

Each ID is counted at most once per direction (`counted_in`, `counted_out` sets), so jitter near a line does not cause double counting.

**Final result on this video:** IN = 12, OUT = 7.

## Heatmap

At every frame, a Gaussian blob is added at each person's bounding-box center to an accumulator the size of the frame. After the last frame the accumulator is blurred, square-root scaled (so weak walking paths stay visible), normalized to the 99.5th percentile (so a few hotspots do not dominate), and colorized with the JET colormap (red = most presence).

- **Vertical streaks:** walking paths, since people mostly move top to bottom.
- **Red/yellow spots:** places where people stood still for a long time. The heatmap measures presence duration, so these spots are brighter than places people only walked through.

The overlay blends the heatmap onto a clean first frame, and only where there is real activity.

## Notes

- Use `yolov8s.pt` or `yolov8m.pt` for better accuracy at the cost of speed.
- Adjust the line positions to match your scene so people actually cross them.
- Tracker ID switches can occasionally happen when people overlap, so the counts (IN = 12, OUT = 7) are approximate.
- The video is first written with OpenCV (`mp4v`) and then re-encoded to H.264 with ffmpeg so it plays in browsers and on phones.