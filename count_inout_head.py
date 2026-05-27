"""Real-time head-only line-crossing counter.

Pipeline:
    webcam / video / RTSP
        -> YOLOv8 head detector (Abcfsa pre-trained on CrowdHuman)
        -> Ultralytics built-in ByteTrack (persistent IDs)
        -> supervision.LineZone (counts IN / OUT crossings)
        -> SQLite log via db.py (one row per crossing)

Run:
    python count_inout_head.py --line v
    python count_inout_head.py --source path/to/video.mp4 --line h --save out.mp4

Keys (in viewer window):
    q / Esc   quit
    r         reset on-screen counters (DB rows are not affected)
"""

import argparse
import atexit

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

import db


# -------- helpers --------------------------------------------------------------

def parse_source(s):
    """Cast '0' / '1' to int (webcam index); leave paths / URLs untouched."""
    return int(s) if s.isdigit() else s


def parse_line(spec, frame_w, frame_h):
    """Convert --line spec into two endpoint tuples.

    Forms:
        'h'              -> horizontal middle
        'v'              -> vertical middle
        'x1,y1,x2,y2'    -> absolute pixel coords
    """
    if not spec:
        return None
    if spec.lower() == 'h':
        y = frame_h // 2
        return (10, y), (frame_w - 10, y)
    if spec.lower() == 'v':
        x = frame_w // 2
        return (x, 10), (x, frame_h - 10)
    parts = [int(p) for p in spec.split(',')]
    if len(parts) != 4:
        raise ValueError(f"--line needs 4 ints 'x1,y1,x2,y2', got: {spec}")
    return (parts[0], parts[1]), (parts[2], parts[3])


def pick_line_interactively(first_frame):
    """Show first frame; user clicks 2 points to define the counting line."""
    pts = []
    base = first_frame.copy()
    win = 'Draw counting line - CLICK 2 POINTS, then press any key'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def render():
        img = base.copy()
        # Big, obvious banner on top so the user understands the screen is interactive.
        cv2.rectangle(img, (0, 0), (img.shape[1], 80), (0, 0, 0), -1)
        msg = f'CLICK 2 POINTS to draw the counting line   ({len(pts)}/2)'
        cv2.putText(img, msg, (16, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2)
        if len(pts) == 2:
            cv2.putText(img, 'Press any key to START', (16, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for p in pts:
            cv2.circle(img, p, 8, (0, 255, 255), -1)
        if len(pts) == 2:
            cv2.line(img, pts[0], pts[1], (0, 255, 255), 3)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            render()

    render()
    cv2.setMouseCallback(win, on_mouse)
    while len(pts) < 2:
        if cv2.waitKey(20) == 27:
            cv2.destroyWindow(win)
            raise SystemExit("Aborted: line not set")
    cv2.waitKey(0)
    cv2.destroyWindow(win)
    return pts[0], pts[1]


def detections_from_result(result):
    """Wrap an Ultralytics Results object into a supervision.Detections."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return sv.Detections.empty()
    tracker_id = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    return sv.Detections(
        xyxy=boxes.xyxy.cpu().numpy(),
        confidence=boxes.conf.cpu().numpy(),
        class_id=boxes.cls.cpu().numpy().astype(int),
        tracker_id=tracker_id,
    )


def build_line_zone(p1, p2, anchor):
    """Construct a LineZone with the chosen triggering anchor."""
    anchor_map = {
        'center': sv.Position.CENTER,
        'bottom': sv.Position.BOTTOM_CENTER,
        'top':    sv.Position.TOP_CENTER,
    }
    if anchor == 'corners':
        return sv.LineZone(start=sv.Point(*p1), end=sv.Point(*p2))
    return sv.LineZone(
        start=sv.Point(*p1), end=sv.Point(*p2),
        triggering_anchors=[anchor_map[anchor]],
    )


# -------- main ----------------------------------------------------------------

def main(opt):
    db.init_db()
    session_id = db.start_session(source=opt.source, weights=opt.weights)
    atexit.register(db.end_session, session_id)
    print(f'DB session: {session_id}')

    # Low-spec preset: switch to the nano model at a smaller input size,
    # unless the user explicitly chose their own --weights / --img-size.
    if opt.lite:
        if opt.weights == 'yolov8_head_medium.pt':
            opt.weights = 'yolov8_head_nano.pt'
        if opt.img_size == 640:
            opt.img_size = 320
        print(f'Lite mode: {opt.weights} @ img-size {opt.img_size}')

    model = YOLO(opt.weights)
    device = opt.device
    if device != 'cpu' and not torch.cuda.is_available():
        print('CUDA not available - falling back to CPU (will be slow).')
        device = 'cpu'
    if device != 'cpu':
        model.to(f'cuda:{device}')

    cap = cv2.VideoCapture(parse_source(opt.source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {opt.source}")
    # On slow machines the capture buffer backs up with stale frames and the
    # view lags behind reality; keep only the newest frame so it stays live.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, first = cap.read()
    if not ok:
        raise SystemExit("Cannot read first frame")
    fh, fw = first.shape[:2]

    p1, p2 = parse_line(opt.line, fw, fh) if opt.line else pick_line_interactively(first)
    if opt.flip:
        p1, p2 = p2, p1
    line_zone = build_line_zone(p1, p2, opt.anchor)

    line_annot  = sv.LineZoneAnnotator(
        thickness=2,
        display_in_count=False,
        display_out_count=False,
    )
    box_annot   = sv.BoxAnnotator(thickness=2)
    label_annot = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annot = sv.TraceAnnotator(thickness=2, trace_length=30)

    writer = None
    if opt.save:
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(opt.save, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps_out, (fw, fh))

    # For video files: rewind so the first real frame isn't lost to the picker.
    # For webcams / RTSP streams: cap.set(POS_FRAMES, 0) can stall the capture,
    # and rewind isn't meaningful for a live stream anyway — just keep reading.
    is_stream = opt.source.isdigit() or opt.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')
    )
    if not is_stream:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    win = 'Head In/Out Counter (q to quit, r to reset)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    seen_ids = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Inference + tracking (Ultralytics handles ByteTrack via persist=True)
        result = model.track(
            frame,
            persist=True,
            conf=opt.conf_thres,
            iou=opt.iou_thres,
            imgsz=opt.img_size,
            tracker='bytetrack.yaml',
            verbose=False,
        )[0]

        detections = detections_from_result(result)

        # Only trigger LineZone when we actually have tracker IDs — otherwise
        # supervision emits a "Line zone counting skipped" warning every frame.
        if detections.tracker_id is not None:
            crossed_in, crossed_out = line_zone.trigger(detections)
            seen_ids.update(int(t) for t in detections.tracker_id)

            ts = db.now_iso()
            rows = []
            for i, fired in enumerate(crossed_in):
                if fired:
                    rows.append((ts, 'in',  int(detections.tracker_id[i]), session_id))
            for i, fired in enumerate(crossed_out):
                if fired:
                    rows.append((ts, 'out', int(detections.tracker_id[i]), session_id))
            db.log_events(rows)

        # Build labels (#ID conf) when tracked, else (head conf)
        if detections.tracker_id is not None:
            labels = [f"#{tid} {conf:.2f}"
                      for tid, conf in zip(detections.tracker_id, detections.confidence)]
        else:
            labels = [f"head {conf:.2f}" for conf in detections.confidence]

        # Pick the canvas: the real camera frame, or (--blank) a black
        # background so only the boxes / line / traces show.
        canvas = np.zeros_like(frame) if opt.blank else frame

        # Draw annotations
        if detections.tracker_id is not None and len(detections) > 0:
            canvas = trace_annot.annotate(canvas, detections=detections)
        canvas = box_annot.annotate(canvas, detections=detections)
        if len(detections) > 0:
            canvas = label_annot.annotate(canvas, detections=detections, labels=labels)
        canvas = line_annot.annotate(canvas, line_counter=line_zone)

        # On-screen overlay: only TOTAL in top-left
        cv2.rectangle(canvas, (0, 0), (300, 70), (0, 0, 0), -1)
        cv2.putText(canvas, f'TOTAL : {line_zone.in_count}', (12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

        cv2.imshow(win, canvas)
        if writer is not None:
            writer.write(canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            line_zone.in_count = 0
            line_zone.out_count = 0
            seen_ids.clear()

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f'Session id       : {session_id}')
    print(f'Frames processed : {frame_idx}')
    print(f'IN     : {line_zone.in_count}')
    print(f'OUT    : {line_zone.out_count}')
    print(f'TOTAL  : {line_zone.in_count}   (IN only)')
    print(f'Unique : {len(seen_ids)}   (distinct tracker IDs seen)')


# -------- CLI ------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description='Real-time head counter (IN/OUT line crossing).')
    p.add_argument('--weights', type=str, default='yolov8_head_medium.pt',
                   help='yolov8_head_medium.pt (accurate, default) or yolov8_head_nano.pt (fast)')
    p.add_argument('--source', type=str, default='0',
                   help='webcam index (0/1/...), video path, or RTSP URL')
    p.add_argument('--img-size',  type=int,   default=640)
    p.add_argument('--conf-thres', type=float, default=0.60,
                   help='detection confidence threshold (0-1)')
    p.add_argument('--iou-thres',  type=float, default=0.45,
                   help='NMS IoU threshold (0-1)')
    p.add_argument('--device', default='0', help="'0' for GPU, 'cpu' for CPU")
    p.add_argument('--line', type=str, default='',
                   help="'x1,y1,x2,y2', or 'h' / 'v'. Empty = click 2 points on first frame.")
    p.add_argument('--flip', action='store_true', help='swap IN/OUT direction')
    p.add_argument('--blank', action='store_true',
                   help='hide the camera image; draw boxes/line on a black background')
    p.add_argument('--lite', action='store_true',
                   help='low-spec preset: nano model @ img-size 320 (faster, less accurate)')
    p.add_argument('--anchor', choices=['center', 'bottom', 'top', 'corners'],
                   default='center',
                   help='which bbox point triggers crossing (default: center)')
    p.add_argument('--save', type=str, default='',
                   help='output mp4 path; empty = no save')
    return p


if __name__ == '__main__':
    main(build_parser().parse_args())
