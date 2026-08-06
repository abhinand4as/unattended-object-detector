"""BoxMOT tracking backend: tracker construction, per-frame timing, and rendering/saving.

Talks to BoxMOT (https://github.com/mikel-brostrom/boxmot) at the low level
via `create_tracker` rather than its high-level `BoxMOT`/`Detector`/
`ReIDModel` facade — that facade is broken in the currently pinned release
(importing it raises `ModuleNotFoundError: No module named 'boxmot.data'`,
confirmed by inspecting the installed package). The low-level path
(`create_tracker` + `tracker.update(dets, frame)`) is fully functional and
is all this pipeline needs.

Default tracker: occluboost. In BoxMOT's own benchmark table it posts the
best HOTA/MOTA/IDF1 on MOT17 and SportsMOT of any tracker in the repo, and
it's purpose-built to hold a track's identity through occlusion — exactly
the failure mode that matters here: a person crouches to set down a bag
behind another person, or is briefly blocked by someone walking past, and
must keep the same ID afterward for "person X left, bag still there" logic
to hold.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from boxmot.trackers.registry import create_tracker

from .constants import LUGGAGE_CLASS_ID
from .ownership import LuggageState, bottom_center

# Every tracker BoxMOT ships that this pipeline supports selecting via --tracker.
TRACKER_CHOICES = [
    "occluboost",
    "botsort",
    "boosttrack",
    "strongsort",
    "deepocsort",
    "hybridsort",
    "bytetrack",
    "ocsort",
    "sfsort",
]

# Subset of the above that support appearance-based ReID matching at all.
REID_CAPABLE_TRACKERS = {"occluboost", "botsort", "boosttrack", "strongsort", "deepocsort", "hybridsort"}


def build_tracker(
    tracker_name: str,
    *,
    reid_weights: str,
    use_reid: bool,
    device: str | None,
    reappear_window: float,
    luggage_conf: float,
    fps: float | None,
):
    """Construct a BoxMOT tracker, with two fixes layered on top of its shipped defaults.

    1. GTA-based full-disappearance recovery (occluboost only). occluboost's
       shipped config disables GTA (Global Track Association) by default and
       caps its alive-phase recovery window (gta_max_gap) far below how long
       a track survives before dying (max_age) — leaving a "dead zone" where
       a person who's fully left the frame and come back gets a new track id
       even though GTA is nominally on. Enabling GTA and widening gta_max_gap
       to exceed max_age closes that gap: a person can be fully out of frame
       for up to `reappear_window` seconds and keep their id on return.
    2. Confidence-threshold override (occluboost only). occluboost's shipped
       config requires ~0.57 confidence to match a track and ~0.71 to start a
       new one. Open-vocabulary luggage detections run far lower than that by
       design (see open_vocab.Config.conf's default of 0.10) — left alone,
       every luggage detection is silently dropped and no luggage track is
       ever created, regardless of how good the boxes are. Both thresholds
       are tied to `luggage_conf`, the lower of the two detectors' floors;
       person detections (from the closed-set COCO model) run at much higher
       confidence and clear that floor easily, so this doesn't loosen
       anything on the person side.

    Args:
        tracker_name: One of TRACKER_CHOICES.
        reid_weights: ReID model weights (auto-downloaded by BoxMOT on first use).
        use_reid: Whether to actually build a ReID backend for this tracker.
        device: "cpu", a CUDA device index like "0", or None for auto.
        reappear_window: Seconds a person can be fully out of frame and still
            keep their track id on return (occluboost only; see fix 1 above).
        luggage_conf: The open-vocabulary detector's own confidence floor,
            reused here as the tracker's floor too (see fix 2 above).
        fps: Video framerate if known (recorded footage), else None (live
            stream) — used only to convert reappear_window from seconds to
            frames, since occluboost's internal aging counters are frame-based.
    """
    tracker_kwargs = None
    if tracker_name == "occluboost":
        # occluboost's own max_age defaults to 146 frames; gta_max_gap must
        # meet or exceed that for the alive-phase recovery window to cover a
        # track's entire lifetime with no gap. 150 is a safe floor for videos
        # with unknown/low fps.
        gta_max_gap = max(round(reappear_window * (fps or 30.0)), 150)
        tracker_kwargs = {
            "det_thresh": luggage_conf,
            "new_track_thresh": luggage_conf,
            "gta_enabled": True,
            "gta_max_gap": gta_max_gap,
        }
    return create_tracker(
        tracker_name,
        reid_weights=reid_weights if use_reid else None,
        device=device,
        tracker_kwargs=tracker_kwargs,
    )


def id_to_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, visually distinct BGR color per track id (for person boxes)."""
    hue = (track_id * 37) % 180
    b, g, r = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def probe_fps(source_path: str) -> float | None:
    """Read a video file's own framerate, or None if it can't be determined
    (e.g. the source is a webcam index, not a file path)."""
    cap = cv2.VideoCapture(source_path)
    fps = None
    if cap.isOpened():
        probed = cap.get(cv2.CAP_PROP_FPS)
        if probed > 0:
            fps = probed
    cap.release()
    return fps


class FrameClock:
    """Elapsed seconds per processed frame.

    Uses the video's own timeline when fps is known (deterministic
    regardless of how fast this machine actually processes each frame —
    important for the ownership/abandonment timers in ownership.py, which
    must reflect video time, not wall-clock processing time). Falls back to
    real wall-clock time for live streams (webcam), where there is no fixed
    framerate to rely on.
    """

    def __init__(self, fps: float | None):
        self.fps = fps
        self._t = 0.0
        self._last_wall: float | None = None

    def tick(self) -> float:
        if self.fps:
            self._t += 1.0 / self.fps
        else:
            now = time.time()
            self._t += 0.0 if self._last_wall is None else now - self._last_wall
            self._last_wall = now
        return self._t


def draw_tracks(
    frame: np.ndarray,
    tracks: np.ndarray,
    class_names: dict,
    luggage_state: dict[int, LuggageState] | None = None,
    luggage_class_ids: set[int] = frozenset({LUGGAGE_CLASS_ID}),
    debug: bool = False,
) -> np.ndarray:
    """Draw current tracks onto a copy of `frame`.

    Person boxes get a per-id color and an "id:<n> person <conf>" label,
    with "[owner]" appended if that person currently owns a piece of
    luggage. Luggage boxes are color-coded by ownership status:
        gray    "(pre-existing, ignored)" — flagged static, see ownership.py
        cyan    "(observing)"             — still inside the ownership window
        green   "owner:<id>"              — owned, owner currently nearby
        red     "UNATTENDED! owner id:<n>" — owner has been away too long
    A thin line connects a piece of luggage to its owner once one is known,
    colored to match the luggage box (green = attended, red = unattended).

    `debug` controls how much gets drawn once ownership tracking is active
    (`luggage_state` is not None):
        True  — everything: every person and every piece of luggage,
                including bystanders and static/pre-existing luggage.
        False — only what matters for the abandonment story: people who are
                a confirmed owner (tagged "[owner]"), and luggage that is
                observing/owned/unattended (static "pre-existing" luggage
                and non-owning bystanders are omitted). The owner-connector
                line is unaffected either way, since both its endpoints are
                already being drawn in this mode.
    With `luggage_state=None` (e.g. --no-luggage-logic, no ownership concept
    to filter by), `debug` has no effect — plain per-class boxes are always
    drawn for every track.

    `tracks` is BoxMOT's raw (M, 8) output: rows of
    [x1, y1, x2, y2, track_id, conf, cls, det_ind].
    """
    annotated = frame.copy()
    tracking_luggage = luggage_state is not None
    show_all = debug or not tracking_luggage

    # Pre-compute every track's ground point (for the owner-connector line)
    # and the set of currently-owning person ids (for the "[owner]" tag and
    # the non-debug person filter), both needed while drawing the *other*
    # class's boxes below.
    centers: dict[int, tuple[float, float]] = {}
    owner_ids: set[int] = set()
    if tracking_luggage:
        for x1, y1, x2, y2, track_id, _conf, _cls, _ in tracks:
            centers[int(track_id)] = bottom_center((x1, y1, x2, y2))
        owner_ids = {s.owner_id for s in luggage_state.values() if s.owner_id is not None}

    for x1, y1, x2, y2, track_id, conf, cls, _ in tracks:
        track_id, cls = int(track_id), int(cls)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))

        if tracking_luggage and cls in luggage_class_ids:
            state = luggage_state.get(track_id)
            is_static = state is not None and state.is_static
            if not show_all and (state is None or is_static):
                continue  # non-debug: skip unclassified/pre-existing luggage

            if is_static:
                color, label, thickness = (128, 128, 128), f"Luggage id:{track_id} (pre-existing, ignored)", 1
            elif state is None or state.owner_id is None:
                color, label, thickness = (0, 220, 220), f"Luggage id:{track_id} (observing)", 2
            elif state.unattended:
                color, label, thickness = (0, 0, 255), f"Luggage id:{track_id} UNATTENDED! owner id:{state.owner_id}", 3
            else:
                color, label, thickness = (0, 200, 0), f"Luggage id:{track_id} owner:{state.owner_id}", 2

            cv2.rectangle(annotated, p1, p2, color, thickness)
            cv2.putText(annotated, label, (p1[0], max(p1[1] - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if state is not None and state.owner_id in centers:
                line_color = (0, 0, 255) if state.unattended else (0, 200, 0)
                cv2.line(annotated, tuple(map(int, centers[track_id])), tuple(map(int, centers[state.owner_id])), line_color, 1)
        else:
            if not show_all and track_id not in owner_ids:
                continue  # non-debug: only show confirmed owners

            color = id_to_color(track_id)
            label = f"id:{track_id} {class_names[cls]} {conf:.2f}"
            if track_id in owner_ids:
                label += " [owner]"
            cv2.rectangle(annotated, p1, p2, color, 2)
            cv2.putText(annotated, label, (p1[0], max(p1[1] - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return annotated


class OutputWriter:
    """Saves annotated output: images as individual files, video/webcam frames
    appended to a single .mp4 per source, keyed by the frame's own source path."""

    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._video_writers: dict[str, cv2.VideoWriter] = {}
        self._image_counts: dict[str, int] = {}

    def write(self, source_path: str, frame: np.ndarray) -> None:
        if Path(source_path).suffix.lower() in self.IMAGE_EXT:
            self._write_image(source_path, frame)
        else:
            self._write_video_frame(source_path, frame)

    def _write_image(self, source_path: str, frame: np.ndarray) -> None:
        stem = Path(source_path).stem
        count = self._image_counts.get(stem, 0)
        self._image_counts[stem] = count + 1
        suffix = "" if count == 0 else f"_{count}"
        cv2.imwrite(str(self.out_dir / f"{stem}{suffix}.jpg"), frame)

    def _write_video_frame(self, source_path: str, frame: np.ndarray) -> None:
        writer = self._video_writers.get(source_path)
        if writer is None:
            fps = probe_fps(source_path) or 30.0
            h, w = frame.shape[:2]
            name = Path(source_path).stem or "webcam"
            writer = cv2.VideoWriter(str(self.out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            self._video_writers[source_path] = writer
        writer.write(frame)

    def close(self) -> None:
        for writer in self._video_writers.values():
            writer.release()
