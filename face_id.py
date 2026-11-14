"""Face identification on top of the existing YuNet detector.

Pipeline
--------
  face image --YuNet--> 5 landmarks --similarity warp--> aligned 112x112
             --ArcFace (ONNX)--> 512-d L2-normalised embedding
             --cosine similarity vs. gallery--> identity / Unknown

No training is needed: the labelled dataset (identities/person_001/faces/*.jpg)
is turned into a gallery of embeddings. New people are added by rebuilding the
gallery, not by retraining.

Files needed in ./models/
-------------------------
  face_detection_yunet_2023mar.onnx   (already used by main_new.py)
  w600k_r50.onnx                      ArcFace R50, from InsightFace "buffalo_l"
                                      pack (buffalo_l.zip, GitHub releases of
                                      deepinsight/insightface, tag v0.7).
                                      Lighter alternative: w600k_mbf.onnx from
                                      buffalo_s.zip. Check the InsightFace model
                                      license: pretrained models are for
                                      non-commercial research use.
Dependencies: pip install onnxruntime   (or onnxruntime-gpu for CUDA)

Commands
--------
  python face_id.py build    --identities identities --out gallery.npz
  python face_id.py evaluate --gallery gallery.npz
  python face_id.py identify photo_or_folder --gallery gallery.npz
Live use inside main_new.py:
  python main_new.py --identify --gallery gallery.npz
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
YUNET_PATH = BASE / 'models' / 'face_detection_yunet_2023mar.onnx'
ARCFACE_PATH = BASE / 'models' / 'w600k_r50.onnx'
GALLERY_PATH = BASE / 'gallery.npz'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Cosine-similarity threshold between a face and an identity (0.35..0.50 is
# typical for ArcFace). Calibrate it with `evaluate` on YOUR data.
DEFAULT_THRESHOLD = 0.40
TOPK = 3                  # identity score = mean of its TOPK best similarities
OUTLIER_SIM = 0.25        # build: image looks unlike the rest of its identity
SAME_PERSON_SIM = 0.45    # build: two identities look like one person

# Standard ArcFace 112x112 landmark template: image-left eye, image-right eye,
# nose tip, image-left mouth corner, image-right mouth corner. YuNet returns
# its landmarks in the same order.
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                             [56.0252, 71.7366], [41.5493, 92.3655],
                             [70.7299, 92.2041]], dtype=np.float32)


# --------------------------------------------------------------------------
# Alignment + embedding
# --------------------------------------------------------------------------
def similarity_transform(src, dst):
    """Least-squares similarity transform (Umeyama). Returns a 2x3 matrix."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / n
    u, sv, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    rot = u @ np.diag(d) @ vt
    var_s = (s_c ** 2).sum() / n
    scale = (sv * d).sum() / max(var_s, 1e-12)
    t = mu_d - scale * rot @ mu_s
    return np.hstack([scale * rot, t[:, None]]).astype(np.float32)


def align_face(image, landmarks5):
    """Warp the face so its landmarks match the ArcFace template (112x112)."""
    lm = np.asarray(landmarks5, np.float32).reshape(5, 2)
    if lm[0, 0] > lm[1, 0]:  # upright face: first eye must be the image-left one
        lm = lm[[1, 0, 2, 4, 3]]
    m = similarity_transform(lm, ARCFACE_TEMPLATE)
    return cv2.warpAffine(image, m, (112, 112), borderValue=0.0)


class ArcFace:
    """ArcFace ONNX embedder (w600k_r50 / w600k_mbf, 112x112 -> 512-d)."""

    def __init__(self, path=ARCFACE_PATH):
        import onnxruntime as ort
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'ArcFace model not found: {path}')
        available = ort.get_available_providers()
        providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                     if p in available]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.providers = self.session.get_providers()

    def embed_aligned(self, aligned_bgr):
        blob = cv2.dnn.blobFromImage(aligned_bgr, 1.0 / 127.5, (112, 112),
                                     (127.5, 127.5, 127.5), swapRB=True)
        out = self.session.run(None, {self.input_name: blob})[0][0]
        return (out / (np.linalg.norm(out) + 1e-9)).astype(np.float32)

    def embed(self, image, landmarks5):
        return self.embed_aligned(align_face(image, landmarks5))


class YuNetDetector:
    """YuNet wrapper that returns landmarks; also works on small face crops."""

    def __init__(self, path=YUNET_PATH, score=0.6):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'YuNet model not found: {path}')
        self.det = cv2.FaceDetectorYN.create(str(path), '', (320, 320),
                                             score, 0.3, 5000)

    def detect(self, img, min_side=160, max_side=1920):
        """Returns [(x, y, w, h, score, landmarks 5x2)] in `img` coordinates."""
        h, w = img.shape[:2]
        m = max(h, w)
        s = 1.0
        if m < min_side:
            s = min_side / m          # tiny crops: upscale for the detector
        elif m > max_side:
            s = max_side / m
        work = img
        if s != 1.0:
            interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
            work = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
        self.det.setInputSize((work.shape[1], work.shape[0]))
        _, faces = self.det.detect(work)
        out = []
        if faces is None:
            return out
        for f in faces:
            lm = (f[4:14].reshape(5, 2) / s).astype(np.float32)
            out.append((f[0] / s, f[1] / s, f[2] / s, f[3] / s,
                        float(f[-1]), lm))
        return out


def read_image(path):
    """cv2.imread that also works with non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------
# Gallery
# --------------------------------------------------------------------------
class Gallery:
    """Embeddings of known people. Identity score = mean of its TOPK best sims."""

    def __init__(self, names, embs, labels, paths=None):
        self.names = [str(n) for n in names]
        self.embs = np.asarray(embs, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.paths = list(paths) if paths is not None else [''] * len(self.embs)
        self._idx = [np.where(self.labels == k)[0] for k in range(len(self.names))]

    def __len__(self):
        return len(self.embs)

    def scores(self, emb):
        """Score of every identity (-1 for identities without images)."""
        sims = self.embs @ emb
        out = np.full(len(self.names), -1.0, dtype=np.float32)
        for k, idx in enumerate(self._idx):
            if len(idx):
                top = np.sort(sims[idx])[::-1][:TOPK]
                out[k] = float(top.mean())
        return out

    def identify(self, emb):
        """Returns (best identity name, its score, scores of all identities)."""
        sc = self.scores(emb)
        i = int(np.argmax(sc))
        return self.names[i], float(sc[i]), sc

    def save(self, path):
        np.savez(path, embs=self.embs, labels=self.labels,
                 names=np.array(self.names), paths=np.array(self.paths))

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        return cls(data['names'].tolist(), data['embs'], data['labels'],
                   data['paths'].tolist())


# --------------------------------------------------------------------------
# Live use: identity per track with voting over recent frames
# --------------------------------------------------------------------------
class TrackIdentifier:
    """Keeps a stable identity for every track id.

    Each update compares the face with the gallery; the track label is the
    majority of the last `window` results (a result below the threshold is a
    vote for "Unknown"). This removes single-frame flicker.
    """

    UNKNOWN = 'Unknown'

    def __init__(self, gallery, embedder, threshold=DEFAULT_THRESHOLD,
                 interval_s=0.5, window=8):
        self.gallery = gallery
        self.embedder = embedder
        self.threshold = threshold
        self.interval_s = interval_s
        self.window = window
        self._votes = {}    # track id -> deque[(name or None, score)]
        self._last = {}     # track id -> time of last update
        self._label = {}    # track id -> (name, score)

    def due(self, tid, now):
        return now - self._last.get(tid, 0.0) >= self.interval_s

    def label(self, tid):
        """(name, score) or None if the track has not been identified yet."""
        return self._label.get(tid)

    def update(self, tid, image, landmarks5, now):
        """Embed the face (image = full-res frame, landmarks in its coords)."""
        self._last[tid] = now
        emb = self.embedder.embed(image, landmarks5)
        name, score, _ = self.gallery.identify(emb)
        hist = self._votes.setdefault(tid, deque(maxlen=self.window))
        hist.append((name if score >= self.threshold else None, score))

        tally = {}
        for n, s in hist:
            if n is not None:
                tally.setdefault(n, []).append(s)
        label = (self.UNKNOWN, score)
        if tally:
            best = max(tally, key=lambda n: (len(tally[n]), sum(tally[n])))
            if len(tally[best]) * 2 > len(hist):   # strict majority
                label = (best, float(np.mean(tally[best])))
        if self._label.get(tid, (None,))[0] != label[0]:
            print(f'[ID] track #{tid} -> {label[0]} ({label[1]:.2f})')
        self._label[tid] = label
        return label


# --------------------------------------------------------------------------
# CLI: build / evaluate / identify
# --------------------------------------------------------------------------
def list_images(folder):
    return sorted(p for p in Path(folder).rglob('*')
                  if p.suffix.lower() in IMG_EXT)


def largest(dets):
    return max(dets, key=lambda d: d[2] * d[3])


def cmd_build(args):
    root = Path(args.identities)
    ident_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not ident_dirs:
        raise SystemExit(f'No identity folders in {root}')
    detector = YuNetDetector(score=args.min_score)
    embedder = ArcFace(args.arcface)
    print(f'[BUILD] ArcFace providers: {embedder.providers}')

    names, embs, labels, paths = [], [], [], []
    print(f'\n{"identity":<22}{"images":>8}{"embedded":>10}{"no face":>9}')
    for d in ident_dirs:
        faces_dir = d / 'faces' if (d / 'faces').is_dir() else d
        files = list_images(faces_dir)
        k = len(names)
        names.append(d.name)
        ok = 0
        for f in files:
            img = read_image(f)
            dets = detector.detect(img) if img is not None else []
            if not dets:
                continue
            lm = largest(dets)[5]
            embs.append(embedder.embed(img, lm))
            labels.append(k)
            paths.append(f.relative_to(root).as_posix())
            ok += 1
        print(f'{d.name:<22}{len(files):>8}{ok:>10}{len(files) - ok:>9}')
    if not embs:
        raise SystemExit('No faces were embedded. Check the models and images.')

    gallery = Gallery(names, np.stack(embs), labels, paths)
    gallery.save(args.out)
    print(f'\n[BUILD] Saved {len(gallery)} embeddings, '
          f'{len(names)} identities -> {args.out}')

    # --- Quality report: wrong labels / unmerged tracks ---
    print('\n[CHECK] Images that look unlike the rest of their identity '
          f'(mean similarity < {OUTLIER_SIM}):')
    flagged = 0
    for k, idx in enumerate(gallery._idx):
        if len(idx) < 3:
            continue
        sub = gallery.embs[idx]
        sim = sub @ sub.T
        mean_other = (sim.sum(1) - 1.0) / (len(idx) - 1)
        for j in np.where(mean_other < OUTLIER_SIM)[0]:
            print(f'   {gallery.paths[idx[j]]}  ({mean_other[j]:.2f})')
            flagged += 1
    if not flagged:
        print('   none')

    print(f'\n[CHECK] Identity pairs that look like one person '
          f'(centroid similarity > {SAME_PERSON_SIM}) - maybe unmerged tracks:')
    cents = []
    for idx in gallery._idx:
        c = gallery.embs[idx].mean(0) if len(idx) else np.zeros(gallery.embs.shape[1])
        cents.append(c / (np.linalg.norm(c) + 1e-9))
    pairs = 0
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            s = float(cents[a] @ cents[b])
            if s > SAME_PERSON_SIM:
                print(f'   {names[a]} <-> {names[b]}: {s:.2f}')
                pairs += 1
    if not pairs:
        print('   none')


def cmd_evaluate(args):
    g = Gallery.load(args.gallery)
    n_id = len(g.names)
    thresholds = np.round(np.arange(0.20, 0.71, 0.05), 2)
    print(f'[EVAL] {len(g)} embeddings, {n_id} identities')
    print('NOTE: folds are contiguous blocks (files sorted by name) to limit '
          'near-duplicate leakage between neighbouring frames. Metrics are '
          'optimistic if one person has only one track/session.')

    # --- Closed-set: k-fold by contiguous blocks, per identity ---
    fold_id = np.full(len(g), -1)
    for idx in g._idx:
        n = len(idx)
        if n >= 2:
            k = min(args.folds, n)
            fold_id[idx] = (np.arange(n) * k) // n
    best_score, best_ok = [], []
    for f in range(args.folds):
        test = np.where(fold_id == f)[0]
        if len(test) == 0:
            continue
        keep = np.ones(len(g), bool)
        keep[test] = False
        sub = Gallery(g.names, g.embs[keep], g.labels[keep])
        for i in test:
            name, score, _ = sub.identify(g.embs[i])
            best_score.append(score)
            best_ok.append(name == g.names[g.labels[i]])
    best_score = np.array(best_score)
    best_ok = np.array(best_ok)
    if len(best_score) == 0:
        raise SystemExit('Not enough images per identity to evaluate.')
    print(f'\n[EVAL] Closed-set top-1 accuracy (no threshold): '
          f'{100 * best_ok.mean():.1f}%  ({len(best_ok)} test images)')

    # --- Open-set: leave one identity out, it must be rejected ---
    far = None
    if n_id >= 2:
        rejected_scores = []
        for k, idx in enumerate(g._idx):
            if not len(idx):
                continue
            keep = g.labels != k
            others = Gallery(g.names, g.embs[keep], g.labels[keep])
            for i in idx:
                rejected_scores.append(others.identify(g.embs[i])[1])
        rejected_scores = np.array(rejected_scores)
        far = {t: float((rejected_scores >= t).mean()) for t in thresholds}
    else:
        print('[EVAL] Only one identity: open-set test skipped.')

    print(f'\n{"thr":>5}{"accepted":>10}{"acc.accepted":>14}{"TAR":>7}'
          f'{"FAR (unknown accepted)":>26}')
    rows = []
    for t in thresholds:
        acc_mask = best_score >= t
        accepted = acc_mask.mean()
        acc_acc = best_ok[acc_mask].mean() if acc_mask.any() else float('nan')
        tar = (best_ok & acc_mask).mean()
        f_ = far[t] if far else float('nan')
        rows.append((t, tar, f_))
        print(f'{t:>5.2f}{100 * accepted:>9.1f}%{100 * acc_acc:>13.1f}%'
              f'{100 * tar:>6.1f}%{100 * f_:>25.1f}%')

    if far:
        ok_rows = [r for r in rows if r[2] <= 0.01]
        # Smallest threshold with FAR <= 1%, but never below 0.30: ArcFace
        # similarities under ~0.3 are practically never the same person.
        ok_rows = [r for r in ok_rows if r[0] >= 0.30]
        pick = min(ok_rows, key=lambda r: r[0]) if ok_rows else \
            max(rows, key=lambda r: r[1] - r[2])
        print(f'\n[EVAL] Suggested threshold: {pick[0]:.2f} '
              f'(TAR {100 * pick[1]:.1f}%, FAR {100 * pick[2]:.1f}%)')
        print('       TAR = known person correctly named; FAR = a person that '
              'is NOT in the gallery accepted as someone.')


def cmd_identify(args):
    gallery = Gallery.load(args.gallery)
    detector = YuNetDetector(score=args.min_score)
    embedder = ArcFace(args.arcface)
    target = Path(args.path)
    files = list_images(target) if target.is_dir() else [target]
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = read_image(f)
        if img is None:
            print(f'{f}: cannot read')
            continue
        dets = detector.detect(img)
        if not dets:
            print(f'{f}: no face')
            continue
        vis = img.copy()
        for k, (x, y, w, h, score, lm) in enumerate(dets):
            emb = embedder.embed(img, lm)
            name, sim, sc = gallery.identify(emb)
            order = np.argsort(sc)[::-1][:2]
            second = ', '.join(f'{gallery.names[j]} {sc[j]:.2f}' for j in order)
            label = name if sim >= args.threshold else 'Unknown'
            print(f'{f.name}: face {k} -> {label} ({sim:.2f})   top: {second}')
            if out_dir:
                color = (0, 220, 0) if label != 'Unknown' else (0, 140, 255)
                cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)),
                              color, 2)
                cv2.putText(vis, f'{label} {sim:.2f}', (int(x), max(15, int(y) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if out_dir:
            ok, buf = cv2.imencode('.jpg', vis)
            if ok:
                (out_dir / f'{f.stem}_id.jpg').write_bytes(buf.tobytes())


def main():
    ap = argparse.ArgumentParser(description='Face identification tools')
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='build gallery from identities/ folder')
    b.add_argument('--identities', default='identities')
    b.add_argument('--out', default=str(GALLERY_PATH))
    b.add_argument('--arcface', default=str(ARCFACE_PATH))
    b.add_argument('--min-score', type=float, default=0.6,
                   help='YuNet score for faces inside the crops')
    b.set_defaults(fn=cmd_build)

    e = sub.add_parser('evaluate', help='accuracy + threshold calibration')
    e.add_argument('--gallery', default=str(GALLERY_PATH))
    e.add_argument('--folds', type=int, default=5)
    e.set_defaults(fn=cmd_evaluate)

    i = sub.add_parser('identify', help='identify faces in an image or folder')
    i.add_argument('path')
    i.add_argument('--gallery', default=str(GALLERY_PATH))
    i.add_argument('--arcface', default=str(ARCFACE_PATH))
    i.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    i.add_argument('--min-score', type=float, default=0.7)
    i.add_argument('--out', help='folder for annotated images')
    i.set_defaults(fn=cmd_identify)

    args = ap.parse_args()
    try:
        args.fn(args)
    except FileNotFoundError as e:
        sys.exit(str(e))


if __name__ == '__main__':
    main()
