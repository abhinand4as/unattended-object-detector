"""Unattended-object detection: closed-set person detection + open-vocabulary
luggage detection + tracking + ownership/abandonment alerts.

Standalone package — this module and its siblings (constants.py,
open_vocab.py, ownership.py, tracking.py, detection.py) carry everything
needed to run; nothing here imports scripts from outside this package.

Why two different detectors for person vs. luggage
----------------------------------------------------
COCO's own luggage classes (backpack/handbag/suitcase) miss a lot of real
luggage shapes — duffel bags, cardboard boxes, sacks, trolley bags — no
fixed training set covers that whole space. Open-vocabulary detection
(YOLO-World or YOLOE, see open_vocab.py/model.py) sidesteps that: describe
what you're looking for in words, and the model finds it.

But COCO's own person class is already excellent — high, well-calibrated
confidence — and there's no reason to pay open-vocabulary detection's noise
tax on something a closed-set detector already does better and faster. So
detection is a hybrid, not a single model:

    person  -> YOLO26l (COCO, closed-set)      -- fast, high-confidence, reliable
    luggage -> OpenVocabDetector (open_vocab.py) -- broad vocabulary where it's needed

Dropping "person" from the open-vocabulary vocabulary also sharpens luggage
precision on its own (fewer competing prompts for the model's text encoder
to disambiguate against), and person detections at high confidence skip the
tracker's multi-frame tentative-track period entirely — only the
lower-confidence luggage class still needs it.

All luggage-ish prompts ("suitcase", "backpack", "duffel bag", ...) collapse
into ONE tracked class ("luggage") once tracking starts — the specific
prompt that fired no longer matters, only "is this luggage". See
tracking.py for tracker construction and rendering, and ownership.py for
the actual abandonment logic.

Usage
-----
    uv sync

    uv run unattended-object-detector --source video.mp4
    uv run unattended-object-detector --source 0 --no-show
    uv run unattended-object-detector --source video.mp4 \\
        --luggage-prompts "duffel bag" "jute sack" "tiffin carrier"
    uv run unattended-object-detector --source video.mp4 --tracker botsort
    uv run unattended-object-detector --source video.mp4 --weights yoloe-26l-seg.pt

Notes
-----
* First run downloads YOLO26l (~50MB) plus whichever open-vocabulary weights
  --weights selects (YOLO-World: ~25-100MB + the CLIP text encoder ~350MB;
  YOLOE: its own, larger segmentation checkpoint, no separate CLIP download).
  All need network access. See model.py for how the backend is chosen.
* Only a video file, webcam index, or single image are supported as
  --source — see detection.iter_frames for why.
* --luggage-conf defaults far lower than --person-conf because open-vocab
  models score noticeably lower than closed-set detectors on the same
  object; see open_vocab.Config's docstring.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import OwnershipConfig
from .constants import CLASS_NAMES, LUGGAGE_CLASS_ID, PERSON_CLASS_ID
from .detection import build_luggage_dets, detect_person, iter_frames, make_luggage_classifier
from .open_vocab import Config as OpenVocabConfig, OpenVocabDetector
from .ownership import LuggageOwnershipTracker
from .prompts import LUGGAGE_PROMPTS, NEGATIVE_PROMPTS
from .tracking import FrameClock, OutputWriter, REID_CAPABLE_TRACKERS, TRACKER_CHOICES, build_tracker, draw_tracks, probe_fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unattended-object detection: closed-set person detection + "
        "open-vocabulary luggage detection + BoxMOT tracking + ownership/abandonment alerts."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video file, webcam index, or single image (default: 0).",
    )

    person_group = parser.add_argument_group("person detector (closed-set, COCO)")
    person_group.add_argument(
        "--person-model", default="yolo26l.pt", help="Detection weights for person only (default: %(default)s)."
    )
    person_group.add_argument(
        "--person-conf",
        type=float,
        default=0.25,
        help="Person confidence threshold (default: 0.25 — COCO person detection is reliably high-confidence).",
    )

    luggage_group = parser.add_argument_group("luggage detector (open-vocabulary)")
    luggage_group.add_argument(
        "--weights",
        default="yolov8s-worldv2.pt",
        help=(
            "Open-vocabulary weights: a YOLO-World checkpoint (yolov8{s,m,l,x}-worldv2.pt) or "
            "a YOLOE one (yoloe-{n,s,m,l,x}-seg.pt, e.g. yoloe-26l-seg.pt) — the backend is "
            "picked automatically from this filename, see model.py (default: %(default)s)."
        ),
    )
    luggage_group.add_argument(
        "--luggage-prompts",
        nargs="+",
        default=None,
        help="Override the luggage vocabulary (default: the built-in list in prompts.py).",
    )
    luggage_group.add_argument(
        "--no-negatives",
        action="store_true",
        help="Drop the negative prompts (chair/table/door/...). Usually makes luggage detection worse.",
    )
    luggage_group.add_argument(
        "--luggage-conf",
        type=float,
        default=0.10,
        help="Luggage confidence threshold (default: 0.10 — open-vocab models score lower than closed-set ones).",
    )
    luggage_group.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold (default: 0.50).")
    luggage_group.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640).")
    luggage_group.add_argument("--device", default=None, help="Device to run on, e.g. 'cpu' or '0' (default: auto).")

    tracker_group = parser.add_argument_group("tracker (see tracking.py)")
    tracker_group.add_argument(
        "--tracker",
        default="occluboost",
        choices=TRACKER_CHOICES,
        help="BoxMOT tracker (default: occluboost — best benchmarked HOTA/IDF1, built for occlusion).",
    )
    tracker_group.add_argument(
        "--reid-weights",
        default="osnet_x0_25_msmt17.pt",
        help="ReID weights for appearance matching (auto-downloaded).",
    )
    tracker_group.add_argument(
        "--no-reid",
        action="store_true",
        help="Disable appearance-based ReID even on trackers that support it.",
    )
    tracker_group.add_argument(
        "--reappear-window",
        type=float,
        default=8.0,
        help="Seconds a person can be fully out of frame and still keep their ID on return (occluboost only; default: 8.0).",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--no-show", action="store_true", help="Disable the live tracking window.")
    output_group.add_argument("--no-save", action="store_true", help="Disable saving annotated output.")
    output_group.add_argument(
        "--project",
        default="runs/predict",
        help="Directory to save results under, relative to the current working directory "
        "(default: %(default)s).",
    )
    output_group.add_argument("--name", default="predict", help="Subdirectory name for this run (default: predict).")
    output_group.add_argument(
        "--debug",
        action="store_true",
        help="Draw every person and every piece of luggage (including bystanders and "
        "pre-existing/static luggage). Without this flag, only confirmed owners and "
        "observing/owned/unattended luggage are drawn — see tracking.draw_tracks.",
    )

    # Defaults for the group below come from OwnershipConfig (config.py) — the
    # single source of truth for these numbers, so tuning them for a new
    # camera/scene means editing one file rather than keeping this parser
    # and ownership.LuggageOwnershipTracker's own defaults in sync by hand.
    default_ownership = OwnershipConfig()
    ownership_group = parser.add_argument_group("luggage ownership (see ownership.py, config.py)")
    ownership_group.add_argument(
        "--no-luggage-logic",
        action="store_true",
        help="Disable owner assignment / unattended-object detection; just detect and track normally.",
    )
    ownership_group.add_argument(
        "--owner-distance-factor",
        type=float,
        default=default_ownership.near_distance_factor,
        help=(
            "'Nearby' radius while determining ownership, as a multiple of the candidate "
            "person's own box height rather than a fixed pixel value — see config.py's "
            "docstring for why (default: %(default)s)."
        ),
    )
    ownership_group.add_argument(
        "--owner-window",
        type=float,
        default=default_ownership.owner_window,
        help="Seconds to observe nearby people before fixing an owner (default: %(default)s).",
    )
    ownership_group.add_argument(
        "--away-distance-factor",
        type=float,
        default=default_ownership.away_distance_factor,
        help=(
            "Distance beyond which the owner is 'away', as a multiple of the owner's own "
            "current box height rather than a fixed pixel value (default: %(default)s)."
        ),
    )
    ownership_group.add_argument(
        "--away-time",
        type=float,
        default=default_ownership.away_time,
        help="Seconds away before the luggage is flagged unattended (default: %(default)s).",
    )
    ownership_group.add_argument(
        "--static-grace-period",
        type=float,
        default=default_ownership.static_grace_period,
        help=(
            "Luggage first seen within this many seconds of stream start is treated as pre-existing scene "
            "furniture (e.g. a false-positive detection on a cabinet), not abandoned luggage, and exempted "
            "from owner/unattended logic entirely (default: %(default)s)."
        ),
    )
    ownership_group.add_argument(
        "--min-reference-height",
        type=float,
        default=default_ownership.min_reference_height,
        help=(
            "Pixel floor under the person-height 'ruler' that --owner-distance-factor/"
            "--away-distance-factor multiply against, guarding against degenerate near-zero-"
            "height boxes (default: %(default)s)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Detectors ------------------------------------------------------- #
    person_model = YOLO(args.person_model)

    luggage_prompts = args.luggage_prompts or LUGGAGE_PROMPTS
    negative_prompts = [] if args.no_negatives else NEGATIVE_PROMPTS
    open_vocab_prompts = luggage_prompts + negative_prompts  # no "person" — see module docstring

    open_vocab_cfg = OpenVocabConfig(
        weights=args.weights,
        conf=args.luggage_conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device or "",
        prompts=open_vocab_prompts,
    )
    luggage_detector = OpenVocabDetector(open_vocab_cfg)
    print(f"encoding {len(open_vocab_prompts)} prompts through CLIP ...")
    t0 = time.time()
    luggage_detector.set_prompts(open_vocab_prompts)
    print(f"  done in {time.time() - t0:.1f}s")

    classify_luggage_label = make_luggage_classifier(luggage_prompts)

    # --- Tracker + supporting infrastructure ------------------------------ #
    fps = None if args.source.isdigit() else probe_fps(args.source)
    tracker = build_tracker(
        args.tracker,
        reid_weights=args.reid_weights,
        use_reid=(args.tracker in REID_CAPABLE_TRACKERS) and not args.no_reid,
        device=args.device,
        reappear_window=args.reappear_window,
        luggage_conf=args.luggage_conf,
        fps=fps,
    )
    output = None if args.no_save else OutputWriter(Path(args.project) / args.name)
    clock = FrameClock(fps)

    ownership_tracker = None
    if not args.no_luggage_logic:
        ownership_tracker = LuggageOwnershipTracker(
            OwnershipConfig(
                near_distance_factor=args.owner_distance_factor,
                owner_window=args.owner_window,
                away_distance_factor=args.away_distance_factor,
                away_time=args.away_time,
                static_grace_period=args.static_grace_period,
                min_reference_height=args.min_reference_height,
            )
        )

    # --- Main loop ---------------------------------------------------------#
    try:
        for source_path, frame in iter_frames(args.source):
            person_dets = detect_person(person_model, frame, args.person_conf, args.device)
            raw_luggage_dets = luggage_detector.detect(frame)
            luggage_dets = build_luggage_dets(raw_luggage_dets, classify_luggage_label)
            dets = np.concatenate([person_dets, luggage_dets], axis=0)

            tracks = tracker.update(dets, frame)

            luggage_state = None
            if ownership_tracker is not None:
                t = clock.tick()
                luggage_boxes = {
                    int(tid): (x1, y1, x2, y2) for x1, y1, x2, y2, tid, _conf, cls, _ in tracks if int(cls) == LUGGAGE_CLASS_ID
                }
                person_boxes = {
                    int(tid): (x1, y1, x2, y2) for x1, y1, x2, y2, tid, _conf, cls, _ in tracks if int(cls) == PERSON_CLASS_ID
                }
                for event in ownership_tracker.update(t, luggage_boxes, person_boxes):
                    print(event)
                luggage_state = ownership_tracker.luggage

            annotated = draw_tracks(
                frame, tracks, CLASS_NAMES, luggage_state, luggage_class_ids={LUGGAGE_CLASS_ID}, debug=args.debug
            )

            if not args.no_show:
                cv2.imshow("Unattended object detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if output is not None:
                output.write(source_path, annotated)
    finally:
        if output is not None:
            output.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
