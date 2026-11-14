## main_new.py: single 4K RTSP camera, person + face detection and dataset collection

- Added a background `StreamReader` thread that keeps only the latest frame (no latency build-up), with open/read timeouts and automatic reconnect. Q / ESC / closing the window work even when the camera is down, and a "NO SIGNAL" placeholder is shown while reconnecting.
- Added person detection and tracking with YOLO11n + ByteTrack (COCO class 0), so every person gets a persistent track id. Runs on CUDA with FP16 when available, otherwise on CPU.
- Added face detection with YuNet. Two modes are available via `FACES_IN_PERSON_CROPS`: search faces inside the upper part of each person box on the full-resolution frame (better for 4K, also links a face to a person), or run on the whole downscaled frame.
- Added `assign_faces_to_persons`, which links each face to a person box (the smallest box containing the face center wins, the highest-scoring face is kept).
- Added automatic dataset collection (`DatasetSaver`):
  - person and face crops are cut from the original full-resolution frame and written by a background thread, so disk I/O never blocks the video loop;
  - structure: `dataset/<session>/persons|faces/track_XXXX/fNNNNNNNN.jpg` plus `metadata.csv`; the same file name in `persons/` and `faces/` means the same frame, and one folder per track is ImageFolder-compatible;
  - throttling and limits: minimum interval per track, maximum images per track and kind, bounded queue (extra images are dropped if the disk is slow);
  - quality filters: confidence/score, minimum size, not cut by the frame edge, and a blur filter (Laplacian variance on a downscaled crop);
  - `metadata.csv` stores time, frame number, kind, track id, file, crop coordinates, score, sharpness and source size, so thresholds can be tuned after a first run;
  - the writer is flushed and joined on exit, so no queued images are lost.
- Added an on-screen stats bar: source/analysis resolution, person and face counts, YOLO and face timings, smoothed FPS and saved image counters.
- All tunable parameters (`PROCESS_WIDTH`, confidence thresholds, crop margins, save intervals and limits, sharpness thresholds) are grouped as constants at the top of the file.

## update main_new.py: face-gated dataset collection

Changes compared to the previous version (`main_new_old.py`):

- **Pair-based saving.** A sample is now saved only when a tracked person has a detected face inside their box with a score >= the face threshold. People without a (good) face are never saved. Each sample is a person crop plus a face crop cut from the same original full-resolution frame (`persons/track_XXXX/` and `faces/track_XXXX/`, same file name = same frame). A track folder appears when the track gets its first valid face.
- **Configurable face probability threshold.** `FACE_SAVE_SCORE` (default 0.80) can be overridden from the command line: `python main_new.py --face-score 0.9`. The value is validated to be in the 0..1 range.
- **Automatic detector threshold.** The YuNet detection threshold is lowered to `min(FACE_CONF, face_score)`, so faces allowed by the save threshold are not filtered out by the detector.
- **New `metadata.csv` format.** One row per pair instead of one row per image: `time, frame_seq, track_id, person_file, face_file`, person and face box coordinates, `person_conf`, `face_score`, `person_sharp`, `face_sharp`, `src_w`, `src_h`.
- **`DatasetSaver` rework.** The throttle (`SAVE_INTERVAL_S`) and the per-track limit (`MAX_PER_TRACK`) now count pairs per track. `try_save_pair` replaces `try_save`. The writer thread saves both images of a pair, and the CSV row is written only if both JPEGs were written successfully. A log line is printed when a new track gets its first sample.
- **Quality filters simplified.** The blur filter is applied to the face only (`MIN_SHARP_FACE`); the person blur filter (`MIN_SHARP_PERSON`) was removed. Person sharpness is still logged to the CSV. `SKIP_EDGE_PERSONS` now defaults to `False`, so people cut by the frame border are saved if their face is valid.
- **UI.** Faces below the save threshold are drawn in gray, faces that qualify for saving in orange. The stats bar and the final log now show the number of saved tracks and person+face pairs instead of separate person and face counters.
- **Docs.** The module docstring was rewritten to describe the saving rules, the dataset layout and the new CLI option.

## update main_new.py: dataset layout grouped by track

Changes compared to `main_new.py`:

- **New dataset folder structure.** Each track now has its own folder containing both crop types:
  `dataset/<YYYYMMDD_HHMMSS>/track_XXXX/persons/fNNNNNNNN.jpg` and
  `dataset/<YYYYMMDD_HHMMSS>/track_XXXX/faces/fNNNNNNNN.jpg`.
  Previously the layout was `persons/track_XXXX/` and `faces/track_XXXX/`. Now all images of one person visit are in a single folder, which makes it easier to review, label or delete one track.
- **Session + track as a unique key.** Track ids restart in every session, so the session folder name (start time) together with the track id uniquely identifies one person visit.
- `metadata.csv` keeps the same columns; `person_file` and `face_file` now contain the new relative paths (`track_XXXX/persons/...`, `track_XXXX/faces/...`).
- Module docstring updated to describe the new layout.

## update main_new.py: resolution-independent overlay

Changes compared to `main_new_1.py`:

- **Constant on-screen overlay size.** Fonts, line thickness and paddings are multiplied by `ui_scale()` (frame pixels per screen pixel, based on the real window width via `cv2.getWindowImageRect`). Text and boxes keep the same visible size for any stream resolution (e.g. 4K) and any window size. If OpenCV cannot report the window size, `UI_REF_WIDTH` is used as a fallback.
- **Resizable window with correct aspect ratio.** The window is created with `WINDOW_KEEPRATIO` and an initial width of `WINDOW_INIT_WIDTH` (1280 px). After the first frame its height is adjusted once to the stream's aspect ratio.
- **New bounding box labels.** `draw_label` draws a filled label (dark text on the box color) above the box. If there is no room above, the label is placed inside the box, and it is always kept within the frame.
- **New status bar.** `draw_status` replaces the full-width top bar with a compact semi-transparent panel in the top-right corner. It has two right-aligned lines:
  - resolution (source -> analysis), FPS, YOLO and face detection time;
  - person and face counts, plus saved tracks and pairs.
  The font shrinks automatically on very narrow frames.
- **"NO SIGNAL" screen.** The placeholder now has a 16:9 height computed from the width and uses the same scaling as the rest of the overlay.
- **Overlay settings.** New constants: `WINDOW_INIT_WIDTH`, `UI_REF_WIDTH`, `FONT`, `BOX_FONT`, `BOX_LINE`, `STATUS_FONT`. Below-threshold faces are now drawn in a lighter gray (150, 150, 150) for better visibility.


## face_id.py (new): face identification with ArcFace

Identifies people on top of the existing YuNet detector. No training is needed: the labelled dataset (`identities/person_001/faces/*.jpg`) is turned into a gallery of embeddings, and new people are added by rebuilding the gallery.

Pipeline: face -> YuNet 5 landmarks -> similarity warp to 112x112 -> ArcFace (ONNX, `w600k_r50.onnx`, 512-d L2-normalised embedding) -> cosine similarity against the gallery -> identity or `Unknown`.

- **Alignment and embedding.** `similarity_transform` (Umeyama) + `align_face` warp the face to the standard ArcFace landmark template. `ArcFace` runs the ONNX model on CUDA when available, otherwise on CPU.
- **`YuNetDetector` wrapper.** Returns landmarks and also works on small face crops (small crops are upscaled, very large images are downscaled before detection).
- **`Gallery`.** Stores embeddings, labels and file paths in a `.npz` file. The score of an identity is the mean of its `TOPK` (3) best similarities. Supports `save`, `load`, `scores` and `identify`.
- **`TrackIdentifier`.** Keeps a stable identity per tracker id. Each track is identified at most every `interval_s` seconds, and the label is a strict majority vote over the last `window` results (results below the threshold vote for `Unknown`), which removes single-frame flicker.
- **CLI commands:**
  - `build`: builds the gallery from the `identities/` folder and prints a quality report (images that look unlike the rest of their identity, and identity pairs that look like one person, i.e. possibly unmerged tracks);
  - `evaluate`: closed-set top-1 accuracy (k-fold by contiguous blocks), open-set test (leave one identity out), TAR/FAR table over thresholds and a suggested threshold;
  - `identify`: identifies faces in an image or folder and can save annotated images.
- Default similarity threshold is `DEFAULT_THRESHOLD = 0.40` and should be calibrated with `evaluate` on real data.
- Requires `onnxruntime` (or `onnxruntime-gpu`) and the ArcFace model in `models/`.

## main_new.py: live identification mode


- **New `--identify` mode.** `python main_new.py --identify --gallery gallery.npz` compares every tracked person who has a good face with the gallery. New options: `--identify`, `--gallery FILE` (implies `--identify`) and `--id-threshold T` (minimum cosine similarity; defaults to `face_id.DEFAULT_THRESHOLD`). If the gallery or the model is missing, the program exits with a clear message.
- **`Face` namedtuple.** Faces now carry `x, y, w, h, score` plus `lm`, the 5 YuNet landmarks in original-frame coordinates (needed for ArcFace alignment). `detect_faces_full` and `detect_faces_in_persons` return `Face` objects, and all other code reads only the first five fields, so the saving logic is unchanged.
- **Identification step in the main loop.** It runs before drawing (to use clean pixels) and only for faces assigned to a tracked person, with score >= the save threshold, big enough (`MIN_FACE_SIDE`) and due according to `ID_INTERVAL_S` (0.5 s). The track label is the majority vote over the last `ID_WINDOW` (8) results.
- **Overlay.** Known people are drawn as `#3 Name 0.62`, unidentified people keep the default green `#3 Person 0.91`, and people identified as Unknown are drawn in orange. The status bar shows `ID: N people` when identification is on.
- **Startup log.** Prints the number of people and embeddings in the gallery, the threshold and the ONNX provider used by ArcFace.
- Module docstring extended with a short "Identification" section.