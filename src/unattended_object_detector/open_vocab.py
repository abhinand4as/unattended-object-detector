"""Open-vocabulary luggage detector.

Why open-vocabulary at all: COCO's own luggage classes (backpack, handbag,
suitcase) miss a lot of real-world luggage shapes — duffel bags, cardboard
boxes, sacks, trolley bags — because no fixed training set covers the full
space of "things a person might carry and leave behind". Open-vocabulary
detection sidesteps that: you describe what you're looking for in words, and
the model finds it. Adding a new luggage type is adding a string to a list,
no retraining involved.

This is a pared-down, standalone version of the detector wrapper originally
prototyped in inference_openvocab.py: only what this pipeline actually
calls (set_prompts once, then detect() per frame) is kept. The original
file's ROI-classification and negative-label-drawing helpers exist to
support a three-mode CLI test harness for tuning prompts — genuinely useful
there, but dead weight in an always-on detection pipeline, so they were
left out rather than carried over unused.

Which underlying model actually runs — YOLO-World or the newer YOLOE family
— is model.py's job, picked from cfg.weights's filename (see its module
docstring). This module stays model-agnostic: it only calls the small
common surface model.VocabBackend exposes (set_classes/predict/
resolve_label), never anything YOLO-World- or YOLOE-specific directly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import model as model_backends
from .prompts import LUGGAGE_PROMPTS, NEGATIVE_PROMPTS


@dataclass
class Config:
    """Open-vocabulary detector configuration.

    Only the fields OpenVocabDetector actually reads are here — no
    display/labelling options, since this pipeline draws its own unified
    "luggage" tag downstream (see tracking.draw_tracks) rather than the raw
    prompt match.
    """

    # YOLO-World: yolov8{s,m,l,x}-worldv2.pt (x is most accurate, slowest).
    # YOLOE: yoloe-{n,s,m,l,x}-seg.pt (e.g. yoloe-26l-seg.pt) — a newer
    # open-vocabulary family; prefer the plain "-seg" checkpoint over
    # "-seg-pf" ("prompt-free"), which ignores set_prompts entirely. The
    # backend is picked automatically from this filename — see model.py.
    weights: str = "yolov8s-worldv2.pt"
    conf: float = 0.10                   # deliberately low: open-vocab models score lower than closed-set detectors on the same object
    iou: float = 0.50
    imgsz: int = 640
    device: str = ""                     # "" = auto, else "cpu" or a CUDA device index like "0"
    prompts: List[str] = field(default_factory=lambda: LUGGAGE_PROMPTS + NEGATIVE_PROMPTS)


class OpenVocabDetector:
    """Thin, model-agnostic wrapper that keeps the expensive vocabulary-encoding
    step out of the per-frame path, regardless of which backend model.py loads.
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self._backend: Optional[model_backends.VocabBackend] = None
        self._active_prompts: List[str] = []

    def _ensure_backend(self) -> model_backends.VocabBackend:
        """Load the model on first use only, so constructing a detector (e.g. to
        read --help) never requires network access or GPU weights."""
        if self._backend is None:
            self._backend = model_backends.build_backend(self.cfg.weights)
        return self._backend

    def set_prompts(self, prompts: Sequence[str]) -> None:
        """Encode `prompts` and install them as the model's vocabulary.

        This is the ONE expensive step (hundreds of milliseconds) — it must be
        called once, before the per-frame loop starts, never inside it. Calling
        it again with an unchanged prompt list is a cheap no-op.
        """
        prompts = list(prompts)
        if prompts == self._active_prompts:
            return
        backend = self._ensure_backend()
        try:
            backend.set_classes(prompts)
        except Exception as exc:  # noqa: BLE001 — surface any failure as a clear, actionable message
            sys.exit(
                f"set_classes() failed: {exc}\n\n"
                "This almost always means the model's text encoder could not be downloaded "
                "(CLIP for YOLO-World, YOLOE's own text encoder for YOLOE). Ensure network "
                "access; for YOLO-World specifically:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            )
        self._active_prompts = prompts

    def detect(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Run detection on one BGR image (as produced by cv2.imread/VideoCapture).

        Returns a list of (matched_prompt, confidence, (x1, y1, x2, y2)) tuples,
        sorted by confidence descending. The matched prompt is whichever entry
        in cfg.prompts scored highest for that box — could be a luggage prompt
        or a negative prompt; the caller decides what to do with each (see
        detection.classify_label).
        """
        backend = self._ensure_backend()
        result = backend.predict(
            image,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device or None,
        )

        detections = []
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            label = backend.resolve_label(names[cls_id])
            detections.append((label, confidence, (x1, y1, x2, y2)))
        detections.sort(key=lambda d: -d[1])
        return detections