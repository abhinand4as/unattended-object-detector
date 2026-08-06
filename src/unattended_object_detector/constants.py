"""Shared class-id constants for the unattended-object detection pipeline.

Only two tracked classes exist in this pipeline: PERSON (from the closed-set
COCO detector) and LUGGAGE (from the open-vocabulary detector — many
different text prompts like "backpack", "duffel bag", "cardboard box" all
collapse into this single id once tracking starts; see
detection.classify_label for where that collapsing happens).

Kept in their own module, separate from detection.py/tracking.py, so those
two modules can both depend on the class ids without depending on each
other — avoids a circular import between them.
"""

PERSON_CLASS_ID = 0
LUGGAGE_CLASS_ID = 1

# Human-readable names, keyed by the ids above. Used only for drawing.
CLASS_NAMES = {
    PERSON_CLASS_ID: "person",
    LUGGAGE_CLASS_ID: "luggage",
}

# The closed-set COCO detector's own class id for "person" (index 0 in the
# standard 80-class COCO list). Kept as a distinct name from PERSON_CLASS_ID
# even though both happen to be 0 right now, so the two meanings — "COCO's
# numbering" vs. "this pipeline's numbering" — never get silently conflated
# if either one changes independently.
COCO_PERSON_CLASS_ID = 0
