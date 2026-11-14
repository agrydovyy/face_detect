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