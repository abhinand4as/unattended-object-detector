"""Open-vocabulary prompt vocabulary for the luggage detector.

Kept in its own module, separate from open_vocab.py's model-wrapping code,
so the vocabulary — the one thing most worth experimenting with — can be
edited without touching any detection logic. These strings ARE the
"training data" for open-vocabulary detection: edit freely, there is no
retraining step, just a re-encode through CLIP (see
open_vocab.OpenVocabDetector.set_prompts). Override LUGGAGE_PROMPTS at the
CLI with --luggage-prompts instead of editing this file, if you'd rather
not commit to a change; NEGATIVE_PROMPTS has no CLI override, since it
isn't meant to vary per run — see its own docstring below for why.

Deliberately NOT "person": person detection is handled entirely by the
separate closed-set COCO detector (see detector.py's module docstring and
ARCHITECTURE.md §2), at higher confidence than open-vocabulary scoring ever
reaches. Adding "person" here would make it one more prompt CLIP has to
disambiguate luggage against, for no benefit — the hybrid design exists
specifically to avoid paying that cost.
"""

from __future__ import annotations

from typing import List

LUGGAGE_PROMPTS: List[str] = [
    "suitcase", "backpack", "handbag", "duffel bag", "shopping bag",
    "cardboard box", "plastic bag", "briefcase", "trolley bag",
    "luggage", "sack", "package", "parcel", "container",
    # Catch-all: covers objects that don't match any noun above but still
    # fit the core pattern — sitting alone on the floor, no owner nearby.
    # Broader net, so also more prone to false positives on ordinary floor
    # clutter; worth validating against your own footage before relying on
    # it in production.
    "an unattended object on the floor",
]

LUGGAGE_PROMPTS: List[str] = [
    "suitcase", "backpack", "handbag", "duffel bag", "shopping bag",
    "plastic bag", "briefcase", "trolley bag",
    "luggage", "sack", "package", "parcel", "container",
    # Catch-all: covers objects that don't match any noun above but still
    # fit the core pattern — sitting alone on the floor, no owner nearby.
    # Broader net, so also more prone to false positives on ordinary floor
    # clutter; worth validating against your own footage before relying on
    # it in production.
    "an unattended object on the floor",
]

# Negative prompts. Fed to the detector alongside the luggage vocabulary
# (see open_vocab.Config.prompts) so it has something to compete against,
# but never tracked — filtered out in detection.classify_label. Not
# optional in practice: without them, an open-vocabulary model has no way
# to say "this is NOT luggage" and will force-fit every blob onto the
# nearest luggage prompt.
NEGATIVE_PROMPTS: List[str] = ["chair", "table", "door", "trash can", "floor", "wall", "person", "human"]