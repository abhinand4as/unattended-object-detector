"""Open-vocabulary model backends: YOLO-World and YOLOE.

Ultralytics ships more than one open-vocabulary detector family, and they
don't quite share an API for installing a text vocabulary:

    YOLOWorld.set_classes(names)                        # names only
    YOLOE.set_classes(names, embeddings=None)            # embeddings
                                                          # auto-computed
                                                          # via get_text_pe()
                                                          # if omitted

YOLOE also flatly rejects any class name containing a space
(`assert " " not in classes` inside YOLOE.set_classes) — a real problem
for this pipeline, whose vocabulary (prompts.py) is full of multi-word
phrases like "duffel bag" or "an unattended object on the floor". Both
quirks are absorbed here, one per backend class, so open_vocab.py's
OpenVocabDetector can call set_classes()/predict() the same way regardless
of which model actually loaded — and so a third backend later means adding
one more class here, not touching the caller.

Backend selection is by weights filename (anything containing "yoloe"
picks YoloeBackend; everything else — yolov8*-world*.pt, yolov8*-worldv2.pt,
... — picks YoloWorldBackend), so switching models is just changing
--weights, no separate flag to keep in sync.
"""

from __future__ import annotations

import sys
from typing import Protocol, Sequence


class VocabBackend(Protocol):
    """Common shape every open-vocabulary backend below implements."""

    def set_classes(self, prompts: Sequence[str]) -> None: ...

    def predict(self, image, *, conf: float, iou: float, imgsz: int, device: str | None): ...

    def resolve_label(self, raw_label: str) -> str:
        """Map a label read off a Results object back to the prompt string
        the caller originally passed to set_classes() — an identity mapping
        unless the backend had to rewrite prompts to satisfy its own API
        (see YoloeBackend)."""
        ...


class YoloWorldBackend:
    """Wraps ultralytics.YOLOWorld (YOLO-World v2 and earlier)."""

    def __init__(self, weights: str):
        try:
            from ultralytics import YOLOWorld
        except ImportError:
            sys.exit("ultralytics not installed. Run: pip install ultralytics")
        self._model = YOLOWorld(weights)

    def set_classes(self, prompts: Sequence[str]) -> None:
        self._model.set_classes(list(prompts))

    def predict(self, image, *, conf: float, iou: float, imgsz: int, device: str | None):
        return self._model.predict(image, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]

    def resolve_label(self, raw_label: str) -> str:
        return raw_label


def _sanitize(prompt: str) -> str:
    """Rewrite a prompt to satisfy YOLOE.set_classes's no-spaces rule.
    Reversed by YoloeBackend.resolve_label via the map set_classes() builds."""
    return prompt.replace(" ", "_")


class YoloeBackend:
    """Wraps ultralytics.YOLOE (e.g. yoloe-26l-seg.pt) — a newer
    open-vocabulary family. Every released YOLOE checkpoint is a
    segmentation model (filenames always carry a "-seg" suffix); this
    pipeline only reads Results.boxes, so the extra mask output is simply
    unused. Prefer the plain "-seg" variant over "-seg-pf" ("prompt-free")
    for this pipeline — "-pf" ignores set_classes and always detects its
    own large built-in vocabulary instead of the luggage-specific prompts
    in prompts.py.
    """

    def __init__(self, weights: str):
        try:
            from ultralytics import YOLOE
        except ImportError:
            sys.exit(
                "ultralytics with YOLOE support not installed. YOLOE ships in newer "
                "ultralytics releases — run: pip install -U ultralytics"
            )
        self._model = YOLOE(weights)
        self._label_map: dict[str, str] = {}

    def set_classes(self, prompts: Sequence[str]) -> None:
        original = list(prompts)
        sanitized = [_sanitize(p) for p in original]
        self._label_map = dict(zip(sanitized, original))
        # embeddings=None (the default) makes YOLOE compute text
        # embeddings itself via get_text_pe() — no separate call needed.
        self._model.set_classes(sanitized)

    def predict(self, image, *, conf: float, iou: float, imgsz: int, device: str | None):
        return self._model.predict(image, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]

    def resolve_label(self, raw_label: str) -> str:
        return self._label_map.get(raw_label, raw_label)


def _select_backend_class(weights: str) -> type:
    return YoloeBackend if "yoloe" in weights.lower() else YoloWorldBackend


def build_backend(weights: str) -> VocabBackend:
    """Load `weights` through whichever backend its filename selects."""
    return _select_backend_class(weights)(weights)