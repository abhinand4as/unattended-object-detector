"""Per-frame detection plumbing: person detection, luggage-label collapsing,
detection-array assembly, and video/image/webcam frame iteration.

This module is where the two detectors (closed-set COCO for people,
open-vocabulary for luggage — see the module docstring in detector.py for
why they're split that way) get reduced to one common shape: a plain
(N, 6) [x1, y1, x2, y2, confidence, class_id] numpy array, which is what
BoxMOT's tracker.update() expects regardless of which model produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .constants import COCO_PERSON_CLASS_ID, LUGGAGE_CLASS_ID, PERSON_CLASS_ID

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# A raw open-vocabulary detection: (matched_prompt, confidence, (x1, y1, x2, y2)).
RawDetection = Tuple[str, float, Tuple[int, int, int, int]]


def detect_person(model: YOLO, frame: np.ndarray, conf: float, device: Optional[str]) -> np.ndarray:
    """Run the closed-set COCO detector for the person class only.

    Returns an (N, 6) array in the same [x1, y1, x2, y2, conf, class_id]
    shape build_dets() produces for luggage, so the two can be concatenated
    directly into one detections array for the tracker.
    """
    result = model.predict(frame, conf=conf, classes=[COCO_PERSON_CLASS_ID], device=device, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()[:, None]
    class_ids = np.full((len(boxes), 1), PERSON_CLASS_ID, dtype=np.float32)
    return np.concatenate([xyxy, confidences, class_ids], axis=1).astype(np.float32)


def make_luggage_classifier(luggage_prompts: List[str]) -> Callable[[str], Optional[int]]:
    """Build a function mapping a raw open-vocabulary prompt match to LUGGAGE_CLASS_ID,
    or None to drop it (negative prompts — see prompts.NEGATIVE_PROMPTS).

    A closure rather than a plain module-level function because the set of
    luggage prompts is a runtime choice (--luggage-prompts can override the
    built-in vocabulary), not a fixed constant.
    """
    luggage_set = set(luggage_prompts)

    def classify(label: str) -> Optional[int]:
        return LUGGAGE_CLASS_ID if label in luggage_set else None

    return classify


def _nms_per_class(dets: np.ndarray, iou_thresh: float = 0.6) -> np.ndarray:
    """Collapse near-duplicate same-class boxes down to the single highest-confidence one.

    Needed because YOLO-World runs non-max suppression per *prompt*, not per
    collapsed track class: "backpack", "duffel bag", and "handbag" firing on
    the exact same physical bag all survive as separate boxes, since each
    was the top scorer for its own prompt. Once every luggage prompt
    collapses to one LUGGAGE_CLASS_ID for tracking (see
    make_luggage_classifier), that reads as several overlapping detections
    of "the same object" and — confirmed empirically — produces multiple
    separate track ids for what is physically one bag. Ordinary
    class-scoped NMS on the already-collapsed classes fixes it.
    """
    keep_rows = []
    for cls_id in np.unique(dets[:, 5]):
        mask = dets[:, 5] == cls_id
        boxes = dets[mask, :4]
        scores = dets[mask, 4]
        # cv2.dnn.NMSBoxes wants boxes as (x, y, w, h), not (x1, y1, x2, y2).
        xywh = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
        keep_idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=iou_thresh)
        for i in np.array(keep_idx).flatten():
            keep_rows.append(dets[mask][i])

    if not keep_rows:
        return np.empty((0, 6), dtype=np.float32)
    return np.array(keep_rows, dtype=np.float32)


def build_luggage_dets(raw_dets: List[RawDetection], classify: Callable[[str], Optional[int]]) -> np.ndarray:
    """Convert OpenVocabDetector.detect()'s output into a tracker-ready (N, 6) array.

    Drops anything classify() maps to None (negative prompts), then
    de-duplicates same-object detections via _nms_per_class.
    """
    rows = []
    for label, conf, (x1, y1, x2, y2) in raw_dets:
        cls_id = classify(label)
        if cls_id is not None:
            rows.append([x1, y1, x2, y2, conf, cls_id])

    if not rows:
        return np.empty((0, 6), dtype=np.float32)
    return _nms_per_class(np.array(rows, dtype=np.float32))


def iter_frames(source_arg: str) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield (source_path, frame) pairs for a webcam index, video file, or single image.

    A bare digit string (e.g. "0") is treated as a webcam index. Anything
    else must be an existing path: an image file (read once, yielded once)
    or a video file (read frame by frame). No directory-of-images mode —
    unlike a plain detection test harness, ownership/abandonment tracking
    needs one continuous scene, not a batch of unrelated stills.
    """
    if source_arg.isdigit():
        cap = cv2.VideoCapture(int(source_arg))
        if not cap.isOpened():
            raise SystemExit(f"cannot open webcam {source_arg}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield source_arg, frame
        finally:
            cap.release()
        return

    path = Path(source_arg)
    if not path.exists():
        raise SystemExit(f"source not found: {path}")

    if path.suffix.lower() in IMAGE_EXT:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"unreadable image: {path}")
        yield str(path), frame
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield str(path), frame
    finally:
        cap.release()
