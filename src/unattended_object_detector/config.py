"""Tunable configuration for the luggage ownership/abandonment state machine.

Kept in its own module, separate from ownership.py's state-machine logic —
mirrors how prompts.py holds the open-vocabulary prompt vocabulary apart
from open_vocab.py's model-wrapping code, so every number worth tuning for
a given camera/scene lives in one place. See ownership.py's module
docstring for what each phase of the three-phase state machine actually
does with these values.

Each field also has a matching CLI flag (--owner-distance, --owner-window,
--away-distance, --away-time, --static-grace-period — see detector.py) to
override it for a single run without editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OwnershipConfig:
    """One instance configures one LuggageOwnershipTracker.

    All distances are in pixels (image space, not real-world units — see
    ARCHITECTURE.md's known limitations on scene-geometry sensitivity), all
    durations in seconds.
    """

    # Phase 2 (ownership window): a person within this radius of a piece of
    # luggage accrues proximity time toward becoming its owner.
    near_distance: float = 100.0

    # Phase 2: how long to observe nearby people, from when the luggage
    # track first appears, before fixing whoever accrued the most proximity
    # time as the owner.
    owner_window: float = 3.0

    # Phase 3 (abandonment watch): owner-to-luggage distance beyond this
    # counts as "away".
    away_distance: float = 120.0

    # Phase 3: continuous away-time, in seconds, before the luggage is
    # flagged unattended.
    away_time: float = 10.0

    # Phase 1 (static-furniture check): luggage first seen within this many
    # seconds of stream start is exempted as pre-existing scene furniture,
    # never entering phases 2/3 at all.
    static_grace_period: float = 3.0