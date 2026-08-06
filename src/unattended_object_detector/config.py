"""Tunable configuration for the luggage ownership/abandonment state machine.

Kept in its own module, separate from ownership.py's state-machine logic —
mirrors how prompts.py holds the open-vocabulary prompt vocabulary apart
from open_vocab.py's model-wrapping code, so every number worth tuning for
a given camera/scene lives in one place. See ownership.py's module
docstring for what each phase of the three-phase state machine actually
does with these values.

Each field also has a matching CLI flag (--owner-distance-factor,
--owner-window, --away-distance-factor, --away-time,
--static-grace-period, --min-reference-height — see detector.py) to
override it for a single run without editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OwnershipConfig:
    """One instance configures one LuggageOwnershipTracker.

    near_distance_factor and away_distance_factor are NOT pixel distances —
    a fixed pixel threshold silently means a different real-world distance
    depending on camera position/zoom (an object twice as close to the
    camera covers roughly twice as many pixels for the same real-world
    gap), so it needs re-tuning every time a camera moves. Instead, both
    are unitless multipliers of a "ruler" computed per pair at check time:
    the relevant person's own bounding-box height. A real person's height
    is roughly constant (~1.7m) regardless of who they are, so how tall
    their box is *in that part of the frame* is a decent local proxy for
    "how many pixels currently represent about a meter here" — bigger near
    the camera, smaller far from it, in the same proportion any other
    real-world distance in that spot of the frame would be. Expressing
    thresholds as a multiple of that ruler keeps them holding roughly the
    same real-world meaning across the frame and across camera setups,
    without any calibration step. It's still an approximation (assumes
    roughly-constant real person height, and a person roughly upright
    facing the camera) — see ARCHITECTURE.md's known limitations for where
    it can still drift (e.g. a person crouching or lying down shrinks their
    own ruler).

    owner_window, away_time, and static_grace_period are plain durations,
    in seconds.
    """

    # Phase 2 (ownership window): a person is "near" a piece of luggage if
    # their bottom-center distance to it is within this many multiples of
    # *that person's own* box height. ~0.5-0.7 is roughly "within arm's
    # reach plus a bit" — validate against your own footage.
    near_distance_factor: float = 0.6

    # Phase 2: how long to observe nearby people, from when the luggage
    # track first appears, before fixing whoever accrued the most proximity
    # time as the owner.
    owner_window: float = 3.0

    # Phase 3 (abandonment watch): the owner is "away" once their distance
    # to the luggage exceeds this many multiples of *the owner's own
    # current* box height. ~1.2-1.5 is roughly "a couple of body-lengths
    # away" — validate against your own footage.
    away_distance_factor: float = 1.5

    # Phase 3: continuous away-time, in seconds, before the luggage is
    # flagged unattended.
    away_time: float = 30.0

    # Phase 1 (static-furniture check): luggage first seen within this many
    # seconds of stream start is exempted as pre-existing scene furniture,
    # never entering phases 2/3 at all.
    static_grace_period: float = 2.0

    # Safety floor, in pixels, under the person-height "ruler" used by
    # near_distance_factor/away_distance_factor above. Guards against a
    # degenerate near-zero-height box (a bad or partially occluded
    # detection) collapsing its threshold to ~0 and making that person
    # permanently "far" no matter how close they actually are.
    min_reference_height: float = 20.0