"""Unit tests for the open-vocabulary backend-selection/sanitization logic
in model.py -- the parts that don't require downloading model weights.
Backend __init__ (which loads real weights) is intentionally not exercised
here; see model.py's module docstring for why the split exists at all.
"""

from unattended_object_detector.model import YoloeBackend, YoloWorldBackend, _sanitize, _select_backend_class


def test_select_backend_class_by_filename():
    assert _select_backend_class("yolov8s-worldv2.pt") is YoloWorldBackend
    assert _select_backend_class("yolov8x-worldv2.pt") is YoloWorldBackend
    assert _select_backend_class("yoloe-26l-seg.pt") is YoloeBackend
    assert _select_backend_class("yoloe-11m-seg-pf.pt") is YoloeBackend
    # Case-insensitive, since it's just a filename a user typed at the CLI.
    assert _select_backend_class("YOLOE-26L-SEG.PT") is YoloeBackend


def test_sanitize_replaces_spaces_for_yoloe_compatibility():
    # YOLOE.set_classes asserts " " not in classes -- multi-word prompts
    # from prompts.py (e.g. "duffel bag") would otherwise crash it.
    assert _sanitize("duffel bag") == "duffel_bag"
    assert _sanitize("an unattended object on the floor") == "an_unattended_object_on_the_floor"
    assert _sanitize("suitcase") == "suitcase"


def test_yoloe_backend_resolves_sanitized_labels_back_to_originals():
    # Build a YoloeBackend without touching __init__ (which would try to
    # load real weights) -- only set_classes()/resolve_label()'s pure
    # label-mapping logic is under test here.
    backend = YoloeBackend.__new__(YoloeBackend)
    backend._label_map = {}

    original_prompts = ["duffel bag", "suitcase", "an unattended object on the floor"]
    sanitized = [_sanitize(p) for p in original_prompts]
    backend._label_map = dict(zip(sanitized, original_prompts))

    for original, raw in zip(original_prompts, sanitized):
        assert backend.resolve_label(raw) == original

    # A label that was never in the vocabulary (e.g. a stale/unexpected
    # class id) should pass through unchanged rather than raising.
    assert backend.resolve_label("unknown_label") == "unknown_label"


def test_yoloworld_backend_resolve_label_is_identity():
    backend = YoloWorldBackend.__new__(YoloWorldBackend)
    assert backend.resolve_label("duffel bag") == "duffel bag"