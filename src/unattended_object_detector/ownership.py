"""Luggage-ownership and unattended-object state machine.

Pure logic, no cv2/ultralytics/detector dependency — feed it per-frame
person/luggage track boxes (identity comes from the caller's tracker; see
tracking.py) and a monotonically increasing timestamp, and it tells you who
owns which luggage and when a piece of luggage has been abandoned. Being
dependency-free like this makes it trivial to unit test in isolation from
the (slow, GPU-bound) detection and tracking machinery.

Three phases per luggage track:

1. Static-furniture check: luggage first seen within `static_grace_period`
   seconds of the very first call to update() is assumed to already have
   been sitting there when observation started — scene furniture (a
   cabinet, a fixed bin, ...) that a detector mistook for luggage, not
   something a person just abandoned. It's marked `is_static` and
   permanently exempted from the two phases below. Without this, a
   persistent false-positive detection eventually gets *some* nearby person
   voted "owner" by pure chance and then flagged unattended the moment that
   unrelated bystander walks off.
2. Ownership window (`owner_window` seconds from when the luggage track
   first appears): every person within `near_distance_factor` × their own
   box height of the luggage accumulates time-spent-nearby. When the
   window closes, whoever accumulated the most time is fixed as the owner.
   If nobody was ever nearby, the luggage is left ownerless and phase 3
   never runs for it.
3. Abandonment watch: once an owner is fixed, track the owner-to-luggage
   distance on every update. If it stays above `away_distance_factor` ×
   the owner's own current box height continuously for `away_time`
   seconds, the luggage is flagged `unattended`. The flag is not latched —
   it clears automatically if the owner comes back within that same
   distance. A missing owner track (occluded, or has walked out of frame)
   counts as infinitely far away.

Distances in phases 2 and 3 are scaled by a person's own box height rather
than a flat pixel constant — see config.OwnershipConfig's docstring for
why (short version: a fixed pixel threshold means a different real-world
distance depending on camera position/zoom; box height is a local,
per-person "ruler" that tracks perspective automatically).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import OwnershipConfig

Box = tuple[float, float, float, float]
Point = tuple[float, float]


def bottom_center(box: Box) -> Point:
    """Approximate ground-contact point of a box: horizontal center, bottom edge.

    Used instead of the box centroid because it's a much better proxy for
    "where this object/person actually is on the floor" — a tall person's
    box centroid floats near their chest, nowhere near the ground.
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, y2


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _box_height(box: Box) -> float:
    """A box's height — used as a local "ruler" for scale-adaptive distance
    thresholds (see config.OwnershipConfig's docstring)."""
    _x1, y1, _x2, y2 = box
    return max(y2 - y1, 0.0)


@dataclass
class LuggageState:
    """Per-track state for one piece of luggage. One instance lives for the
    entire lifetime of a track id in LuggageOwnershipTracker.luggage."""

    first_seen: float
    is_static: bool = False
    proximity_time: dict[int, float] = field(default_factory=dict)
    owner_id: int | None = None
    away_since: float | None = None
    unattended: bool = False


class LuggageOwnershipTracker:
    """Drives the ownership/abandonment state machine described in the module
    docstring across an entire video stream, one update() call per frame."""

    def __init__(self, cfg: OwnershipConfig | None = None):
        cfg = cfg or OwnershipConfig()
        self.near_distance_factor = cfg.near_distance_factor
        self.owner_window = cfg.owner_window
        self.away_distance_factor = cfg.away_distance_factor
        self.away_time = cfg.away_time
        self.static_grace_period = cfg.static_grace_period
        self.min_reference_height = cfg.min_reference_height

        # Every luggage track id ever seen, keyed by that id, kept for the
        # lifetime of the tracker so callers (e.g. drawing code) can look up
        # a track's current status.
        self.luggage: dict[int, LuggageState] = {}

        self._last_t: float | None = None
        self._stream_start_t: float | None = None

    def _reference_height(self, box_height: float) -> float:
        """The person-height "ruler" a threshold factor multiplies against,
        floored at min_reference_height (see OwnershipConfig's docstring)."""
        return max(box_height, self.min_reference_height)

    def update(
        self,
        t: float,
        luggage_boxes: dict[int, Box],
        person_boxes: dict[int, Box],
    ) -> list[str]:
        """Advance the state machine by one frame.

        Args:
            t: Monotonically increasing timestamp in seconds (video-timeline
               time for recorded footage, wall-clock time for a live feed —
               see tracking.FrameClock).
            luggage_boxes: {track_id: (x1, y1, x2, y2)} for every luggage
               track visible this frame.
            person_boxes: {track_id: (x1, y1, x2, y2)} for every person
               track visible this frame.

        Returns:
            Human-readable event strings for anything that changed this
            frame (owner assigned, alert raised, alert cleared) — empty most
            frames. Meant to be printed/logged by the caller, not parsed.
        """
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        if self._stream_start_t is None:
            self._stream_start_t = t

        person_points = {pid: bottom_center(box) for pid, box in person_boxes.items()}
        person_heights = {pid: _box_height(box) for pid, box in person_boxes.items()}
        events: list[str] = []

        for lug_id, lug_box in luggage_boxes.items():
            state = self.luggage.get(lug_id)
            if state is None:
                is_static = (t - self._stream_start_t) <= self.static_grace_period
                state = LuggageState(first_seen=t, is_static=is_static)
                self.luggage[lug_id] = state
            lug_pt = bottom_center(lug_box)

            if state.is_static:
                continue  # permanently exempt — see module docstring, phase 1

            if state.owner_id is None:
                # Phase 2: still within (or just past) the observation window.
                if t - state.first_seen < self.owner_window:
                    for pid, ppt in person_points.items():
                        near_distance = self._reference_height(person_heights[pid]) * self.near_distance_factor
                        if _distance(lug_pt, ppt) <= near_distance:
                            state.proximity_time[pid] = state.proximity_time.get(pid, 0.0) + dt
                elif state.proximity_time:
                    state.owner_id = max(state.proximity_time, key=state.proximity_time.get)
                    events.append(f"Luggage id:{lug_id} -> owner assigned: person id:{state.owner_id}")
                continue

            # Phase 3: owner fixed — watch the distance between them and the luggage.
            # A missing owner box (occluded/out of frame) falls back to the
            # min_reference_height floor for the ruler; distance is already
            # infinite in that case, so it's always "away" regardless.
            owner_pt = person_points.get(state.owner_id)
            owner_height = person_heights.get(state.owner_id, 0.0)
            distance = _distance(lug_pt, owner_pt) if owner_pt is not None else math.inf
            away_distance = self._reference_height(owner_height) * self.away_distance_factor

            if distance <= away_distance:
                state.away_since = None
                if state.unattended:
                    state.unattended = False
                    events.append(f"Luggage id:{lug_id} -> owner id:{state.owner_id} returned, alert cleared")
            else:
                if state.away_since is None:
                    state.away_since = t
                elif not state.unattended and (t - state.away_since) >= self.away_time:
                    state.unattended = True
                    events.append(f"ALERT: Luggage id:{lug_id} left UNATTENDED by person id:{state.owner_id}")

        return events
