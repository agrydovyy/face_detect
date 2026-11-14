"""Single 4K RTSP camera: person (YOLO + tracker) and face (YuNet) detection
with automatic dataset collection.

Key points:
- A background thread reads the stream and keeps only the latest frame,
  so latency does not build up and the UI never blocks on the camera.
- Open/read timeouts + automatic reconnect inside the reader thread.
- Q / ESC / closing the window always work, even when the camera is down.
- Persons are tracked (ByteTrack), so every person gets a track id.
- For every track, person crops and face crops are cut from the ORIGINAL
  full-resolution frame and saved by a background writer thread:

    dataset/<session>/persons/track_0001/f00001234.jpg
    dataset/<session>/faces/track_0001/f00001234.jpg
    dataset/<session>/metadata.csv

  The same file name in persons/ and faces/ means "same frame".
  Folder per track = ImageFolder-compatible, easy to merge/label later.
- Saving is throttled and quality-filtered (size, confidence, sharpness,
  not cut by the frame edge) so you do not get thousands of duplicates.
"""
import csv
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

# Must be set before OpenCV opens the stream.
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')

import config
import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
YUNET_PATH = BASE / 'models' / 'face_detection_yunet_2023mar.onnx'
MODEL_NAME = 'yolo11n.pt'
TRACKER = 'bytetrack.yaml'

# Analysis AND display width (1280 / 1600 = faster, 3840 = full 4K, slow).
PROCESS_WIDTH = 3840
YOLO_IMGSZ = PROCESS_WIDTH  # keep equal to PROCESS_WIDTH, multiple of 32
PERSON_CONF = 0.35
FACE_CONF = 0.70
RETRY_SECONDS = 3
TIMEOUT_MS = 5000

# True: search faces inside person boxes on the full-res frame (better for 4K,
# also links a face to a person). False: run YuNet on the whole small frame.
FACES_IN_PERSON_CROPS = True
CROP_TOP_FRACTION = 0.5   # upper part of the person box where the face is
CROP_PAD = 0.10           # extra padding around the crop (fraction of size)
MIN_CROP_SIDE = 40        # skip crops smaller than this (px, original frame)

# ---- Dataset collection -------------------------------------------------
SAVE_DATASET = True
DATASET_DIR = BASE / 'dataset'
SAVE_INTERVAL_S = 1.0     # min seconds between saves for one track and kind
MAX_PER_TRACK = 60        # max images per track and kind (per session)
JPEG_QUALITY = 95
QUEUE_SIZE = 64           # if the disk is slower, extra images are dropped

PERSON_SAVE_CONF = 0.50
MIN_PERSON_H = 160        # px, in the ORIGINAL frame
PERSON_MARGIN = 0.05      # extra margin around the box (fraction)
SKIP_EDGE_PERSONS = True  # skip people cut by the frame border
EDGE_PX = 3

FACE_SAVE_SCORE = 0.80
MIN_FACE_SIDE = 48        # px, in the ORIGINAL frame
FACE_MARGIN = 0.25

# Blur filter: Laplacian variance of the crop downscaled to <=128 px.
# Values are written to metadata.csv, so tune these after a first run.
MIN_SHARP_PERSON = 8.0
MIN_SHARP_FACE = 15.0

WINDOW = 'RTSP 4K | Persons + Faces | Q/ESC=exit'


# --------------------------------------------------------------------------
# Stream reader
# --------------------------------------------------------------------------
def open_stream(url):
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_MS,
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_MS]
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    except TypeError:  # old OpenCV without params support
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        # Advisory only: FFmpeg backend may ignore it.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class StreamReader(threading.Thread):
    """Reads frames continuously, keeps only the newest one, reconnects on failure."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self._url = url
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stop_event = threading.Event()
        self.connected = False

    def run(self):
        while not self._stop_event.is_set():
            print('[RTSP] Connecting...')
            cap = open_stream(self._url)
            if not cap.isOpened():
                cap.release()
                print('[RTSP] Connection failed. Retrying...')
                self._stop_event.wait(RETRY_SECONDS)
                continue
            print('[RTSP] Connected')
            self.connected = True
            try:
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        print('[RTSP] Frame read failed. Reconnecting...')
                        break
                    with self._lock:
                        self._frame = frame
                        self._seq += 1
            finally:
                self.connected = False
                cap.release()
            self._stop_event.wait(RETRY_SECONDS)

    def get(self, last_seq):
        """Return (seq, frame); frame is None if nothing new since last_seq."""
        with self._lock:
            if self._seq == last_seq:
                return last_seq, None
            return self._seq, self._frame

    def stop(self):
        self._stop_event.set()


# --------------------------------------------------------------------------
# Dataset saving
# --------------------------------------------------------------------------
META_HEADER = ['time', 'frame_seq', 'kind', 'track_id', 'file',
               'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2',
               'score', 'sharpness', 'src_w', 'src_h']


def sharpness(img):
    """Variance of Laplacian on a small grayscale copy (higher = sharper)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 128.0 / max(gray.shape[:2])
    if s < 1.0:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_with_margin(frame, box, margin):
    """Crop box (x1, y1, x2, y2 in frame coords) + margin. Returns (crop, coords) or None."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 - bw * margin))
    cy1 = int(max(0, y1 - bh * margin))
    cx2 = int(min(w, x2 + bw * margin))
    cy2 = int(min(h, y2 + bh * margin))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


class DatasetSaver:
    """Throttled, quality-filtered saving of crops. Disk I/O runs in its own thread."""

    def __init__(self, root):
        session = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.root = Path(root) / session
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / 'metadata.csv'
        self._q = queue.Queue(maxsize=QUEUE_SIZE)
        self._last = {}     # (kind, track_id) -> last save time
        self._counts = {}   # (kind, track_id) -> saved images
        self.totals = {'person': 0, 'face': 0}
        self.dropped = 0
        # Not a daemon: on exit we wait until the queue is written out.
        self._thread = threading.Thread(target=self._run, name='DatasetWriter')
        self._thread.start()
        print(f'[DATASET] Saving to {self.root}')

    def try_save(self, kind, tid, frame, box, score, seq, now):
        """kind: 'person' | 'face'; box in ORIGINAL frame coords. Returns True if queued."""
        key = (kind, tid)
        if self._counts.get(key, 0) >= MAX_PER_TRACK:
            return False
        if now - self._last.get(key, 0.0) < SAVE_INTERVAL_S:
            return False

        margin = PERSON_MARGIN if kind == 'person' else FACE_MARGIN
        res = crop_with_margin(frame, box, margin)
        if res is None:
            return False
        crop, (cx1, cy1, cx2, cy2) = res

        sharp = sharpness(crop)
        min_sharp = MIN_SHARP_PERSON if kind == 'person' else MIN_SHARP_FACE
        if sharp < min_sharp:
            return False

        rel = Path(f'{kind}s') / f'track_{tid:04d}' / f'f{seq:08d}.jpg'
        src_h, src_w = frame.shape[:2]
        row = [datetime.fromtimestamp(now).isoformat(timespec='milliseconds'),
               seq, kind, tid, rel.as_posix(), cx1, cy1, cx2, cy2,
               f'{score:.3f}', f'{sharp:.1f}', src_w, src_h]
        try:
            # .copy(): the frame is modified later by drawing.
            self._q.put_nowait((self.root / rel, crop.copy(), row))
        except queue.Full:
            self.dropped += 1
            return False

        self._last[key] = now
        self._counts[key] = self._counts.get(key, 0) + 1
        self.totals[kind] += 1
        return True

    def _run(self):
        is_new = not self.meta_path.exists()
        with open(self.meta_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(META_HEADER)
            while True:
                item = self._q.get()
                if item is None:
                    break
                path, img, row = item
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    ok, buf = cv2.imencode('.jpg', img,
                                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if not ok:
                        continue
                    path.write_bytes(buf.tobytes())  # works with non-ASCII paths
                    writer.writerow(row)
                    f.flush()
                except OSError as e:
                    print(f'[DATASET] Write error: {e}')

    def close(self):
        self._q.put(None)
        self._thread.join()
        print(f'[DATASET] Done: {self.totals["person"]} person, '
              f'{self.totals["face"]} face images, dropped: {self.dropped}')


def assign_faces_to_persons(faces, boxes):
    """Map person index -> best face (x, y, w, h, score) whose center is inside that box.

    If a face lies inside several boxes, the smallest box wins.
    """
    best = {}
    if len(boxes) == 0:
        return best
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for face in faces:
        x, y, fw, fh, score = face
        cx, cy = x + fw / 2, y + fh / 2
        cands = [i for i, b in enumerate(boxes)
                 if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
        if not cands:
            continue
        i = min(cands, key=lambda k: areas[k])
        if i not in best or score > best[i][4]:
            best[i] = face
    return best


def save_detections(saver, frame, boxes, confs, ids, faces, scale, seq):
    """Cut crops from the ORIGINAL frame and hand them to the saver.

    boxes/faces are in 'small' coordinates; scale converts them to the original.
    Must be called BEFORE drawing (with PROCESS_WIDTH >= source width the
    small frame is the same array as the original one).
    """
    now = time.time()
    frame_h, frame_w = frame.shape[:2]
    face_of = assign_faces_to_persons(faces, boxes)

    for i, (box, conf, tid) in enumerate(zip(boxes, confs, ids)):
        tid = int(tid)
        if tid < 0:
            continue  # tracker has not assigned an id yet

        ob = np.asarray(box, dtype=np.float32) * scale
        person_h = ob[3] - ob[1]
        cut = (ob[0] <= EDGE_PX or ob[1] <= EDGE_PX or
               ob[2] >= frame_w - EDGE_PX or ob[3] >= frame_h - EDGE_PX)

        if (conf >= PERSON_SAVE_CONF and person_h >= MIN_PERSON_H
                and not (SKIP_EDGE_PERSONS and cut)):
            saver.try_save('person', tid, frame, ob, float(conf), seq, now)

        face = face_of.get(i)
        if face is not None:
            x, y, fw, fh, score = face
            fb = np.array([x, y, x + fw, y + fh], dtype=np.float32) * scale
            if (score >= FACE_SAVE_SCORE and (fb[2] - fb[0]) >= MIN_FACE_SIDE
                    and (fb[3] - fb[1]) >= MIN_FACE_SIDE):
                saver.try_save('face', tid, frame, fb, float(score), seq, now)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def resize_to_width(image, width):
    h, w = image.shape[:2]
    if w <= width:
        return image
    return cv2.resize(image, (width, round(h * width / w)),
                      interpolation=cv2.INTER_AREA)


def draw_box(frame, coords, label, color):
    x1, y1, x2, y2 = map(int, coords)
    height, width = frame.shape[:2]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)


def draw_stats(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


def no_signal_canvas(width=PROCESS_WIDTH, height=720):
    canvas = np.zeros((height, width, 3), np.uint8)
    cv2.putText(canvas, 'NO SIGNAL - reconnecting...', (40, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def detect_faces_full(face_detector, small):
    """YuNet on the whole downscaled frame. Returns list of (x, y, w, h, score)."""
    h, w = small.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(small)
    if faces is None:
        return []
    return [(f[0], f[1], f[2], f[3], float(f[-1])) for f in faces]


def detect_faces_in_persons(face_detector, frame, person_boxes_small, scale):
    """YuNet on crops of the original frame. Returns faces in SMALL-frame coords."""
    orig_h, orig_w = frame.shape[:2]
    result = []
    for (x1, y1, x2, y2) in person_boxes_small:
        # Person box in original coordinates, upper part only.
        ox1, oy1, ox2, oy2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        bw, bh = ox2 - ox1, oy2 - oy1
        oy2 = oy1 + bh * CROP_TOP_FRACTION
        bh = oy2 - oy1
        cx1 = int(max(0, ox1 - bw * CROP_PAD))
        cy1 = int(max(0, oy1 - bh * CROP_PAD))
        cx2 = int(min(orig_w, ox2 + bw * CROP_PAD))
        cy2 = int(min(orig_h, oy2 + bh * CROP_PAD))
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw < MIN_CROP_SIDE or ch < MIN_CROP_SIDE:
            continue
        crop = frame[cy1:cy2, cx1:cx2]
        face_detector.setInputSize((cw, ch))
        _, faces = face_detector.detect(crop)
        if faces is None:
            continue
        for f in faces:
            fx, fy, fw, fh = f[:4]
            result.append(((fx + cx1) / scale, (fy + cy1) / scale,
                           fw / scale, fh / scale, float(f[-1])))
    return result


def key_pressed_exit():
    return (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q'), 27)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    rtsp_url = (getattr(config, 'RTSP_URL', None) or '').strip()
    if not rtsp_url:
        raise SystemExit('RTSP_URL is empty. Set it in config.py or the environment.')
    if not YUNET_PATH.is_file() or YUNET_PATH.stat().st_size < 100_000:
        raise SystemExit(f'YuNet model missing/invalid: {YUNET_PATH}. See README.md.')

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else 'cpu'
    precision = 16 if use_cuda else 32
    print(f'[INIT] YOLO device: {device} (0 = first CUDA GPU), FP16: {use_cuda}')

    local_model = BASE / 'models' / MODEL_NAME
    model_path = str(local_model) if local_model.is_file() else MODEL_NAME
    print(f'[INIT] YOLO model: {model_path}, tracker: {TRACKER}')
    model = YOLO(model_path)

    face_detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH), '', (320, 320), FACE_CONF, 0.3, 5000)

    # Warm-up through track() so the tracker is registered from the first call.
    model.track(np.zeros((720, PROCESS_WIDTH, 3), np.uint8), persist=True,
                tracker=TRACKER, classes=[0], imgsz=YOLO_IMGSZ, device=device,
                quantize=precision, verbose=False)

    saver = DatasetSaver(DATASET_DIR) if SAVE_DATASET else None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print('[INFO] Press Q or ESC to exit')

    reader = StreamReader(rtsp_url)
    reader.start()

    last_seq = 0
    shown_once = False
    showing_no_signal = False
    fps_ema = 0.0
    last_loop = time.perf_counter()

    try:
        while True:
            last_seq, frame = reader.get(last_seq)

            if frame is None:
                # No new frame: stay responsive, show placeholder if disconnected.
                if not reader.connected and not showing_no_signal:
                    cv2.imshow(WINDOW, no_signal_canvas())
                    showing_no_signal = True
                    shown_once = True
                if key_pressed_exit():
                    break
                if shown_once and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                time.sleep(0.005)
                continue

            showing_no_signal = False
            t0 = time.perf_counter()
            orig_h, orig_w = frame.shape[:2]
            small = resize_to_width(frame, PROCESS_WIDTH)
            h, w = small.shape[:2]
            scale = orig_w / w  # same factor for x and y (aspect preserved)

            # --- Persons (COCO class 0) + tracking ---
            pred = model.track(small, persist=True, tracker=TRACKER,
                               classes=[0], conf=PERSON_CONF,
                               imgsz=YOLO_IMGSZ, device=device,
                               quantize=precision, verbose=False)[0]
            n = len(pred.boxes)
            if n:
                xyxy = pred.boxes.xyxy.cpu().numpy()
                confs = pred.boxes.conf.cpu().numpy()
                if pred.boxes.id is not None:
                    ids = pred.boxes.id.int().cpu().numpy()
                else:
                    ids = np.full(n, -1, dtype=int)
            else:
                xyxy = np.empty((0, 4), np.float32)
                confs = np.empty((0,), np.float32)
                ids = np.empty((0,), dtype=int)
            t_yolo = time.perf_counter()

            # --- Faces ---
            if FACES_IN_PERSON_CROPS:
                faces = detect_faces_in_persons(face_detector, frame, xyxy, scale)
            else:
                faces = detect_faces_full(face_detector, small)
            t_face = time.perf_counter()

            # --- Save crops (BEFORE drawing: small may be the same array as frame) ---
            if saver is not None:
                save_detections(saver, frame, xyxy, confs, ids, faces, scale, last_seq)

            # --- Draw on the small frame ---
            for box, conf, tid in zip(xyxy, confs, ids):
                tag = f'#{int(tid)} ' if tid >= 0 else ''
                draw_box(small, box, f'{tag}Person {conf:.2f}', (0, 220, 0))
            for (x, y, fw, fh, score) in faces:
                draw_box(small, (x, y, x + fw, y + fh),
                         f'Face {score:.2f}', (255, 180, 0))

            # FPS over the whole loop (includes decode wait, display, waitKey).
            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_loop, 1e-6)
            last_loop = now
            fps_ema = inst_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst_fps

            stats = (f'Src {orig_w}x{orig_h} | Analyze {w}x{h} | '
                     f'P {len(xyxy)} F {len(faces)} | '
                     f'YOLO {1000 * (t_yolo - t0):.0f}ms | '
                     f'Face {1000 * (t_face - t_yolo):.0f}ms | '
                     f'{fps_ema:.1f} FPS')
            if saver is not None:
                stats += (f' | Saved P {saver.totals["person"]} '
                          f'F {saver.totals["face"]}')
            draw_stats(small, stats)

            cv2.imshow(WINDOW, small)
            shown_once = True
            if key_pressed_exit():
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted')
    finally:
        reader.stop()
        reader.join(timeout=TIMEOUT_MS / 1000 + 1)
        if saver is not None:
            saver.close()  # waits until all queued images are written
        cv2.destroyAllWindows()
        print('[INFO] Stopped')


if __name__ == '__main__':
    main()
