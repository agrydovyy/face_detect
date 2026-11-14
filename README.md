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