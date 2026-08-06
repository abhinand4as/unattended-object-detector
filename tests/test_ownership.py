"""Unit tests for the ownership/abandonment state machine.

ownership.py has no cv2/ultralytics/detector dependency by design (see its
module docstring and ARCHITECTURE.md §5) specifically so it can be tested
like this, without any of the slow, GPU-bound detection/tracking machinery.
"""

from unattended_object_detector.config import OwnershipConfig
from unattended_object_detector.ownership import LuggageOwnershipTracker


def test_static_grace_period_exempts_pre_existing_luggage():
    tracker = LuggageOwnershipTracker(OwnershipConfig(owner_window=5.0, static_grace_period=3.0))

    # Luggage present from the very first frame, with a person right next to
    # it the whole time, is scene furniture (§5 phase 1) -- never assigned an
    # owner, never flagged, no events at all.
    events = []
    for t in [0.0, 1.0, 2.0, 4.0, 10.0, 20.0]:
        events += tracker.update(t, luggage_boxes={1: (0, 0, 10, 10)}, person_boxes={1: (0, 0, 10, 10)})

    assert events == []
    assert tracker.luggage[1].is_static is True
    assert tracker.luggage[1].owner_id is None


def test_owner_assigned_to_person_nearest_during_observation_window():
    tracker = LuggageOwnershipTracker(OwnershipConfig(near_distance=50.0, owner_window=5.0, static_grace_period=0.0))

    # Luggage first appears at t=1 (past the zero-length grace period), so
    # it's a real candidate, not static furniture. Person 1 stays close for
    # the whole observation window; person 2 is only nearby briefly.
    tracker.update(1.0, luggage_boxes={}, person_boxes={})  # establishes stream start
    events = []
    events += tracker.update(2.0, luggage_boxes={9: (100, 100, 110, 110)}, person_boxes={1: (100, 100, 110, 110)})
    events += tracker.update(4.0, luggage_boxes={9: (100, 100, 110, 110)}, person_boxes={1: (100, 100, 110, 110), 2: (100, 100, 110, 110)})
    events += tracker.update(8.0, luggage_boxes={9: (100, 100, 110, 110)}, person_boxes={1: (100, 100, 110, 110)})

    assert any("owner assigned: person id:1" in e for e in events)
    assert tracker.luggage[9].owner_id == 1


def test_abandonment_alert_raised_then_cleared_on_return():
    tracker = LuggageOwnershipTracker(
        OwnershipConfig(near_distance=50.0, owner_window=1.0, away_distance=100.0, away_time=5.0, static_grace_period=0.0)
    )
    luggage_box = (500, 500, 510, 510)
    near_person = (500, 500, 510, 510)
    far_person = (0, 0, 10, 10)

    tracker.update(0.0, luggage_boxes={}, person_boxes={})
    tracker.update(1.0, luggage_boxes={5: luggage_box}, person_boxes={1: near_person})
    events = tracker.update(3.0, luggage_boxes={5: luggage_box}, person_boxes={1: near_person})
    assert tracker.luggage[5].owner_id == 1
    assert not any("ALERT" in e for e in events)

    # Owner walks away and stays away for >= away_time (5s): alert fires.
    events = []
    for t in [4.0, 6.0, 8.0, 9.0]:
        events += tracker.update(t, luggage_boxes={5: luggage_box}, person_boxes={1: far_person})
    assert any("ALERT" in e and "id:5" in e for e in events)
    assert tracker.luggage[5].unattended is True

    # Owner returns: alert clears.
    events = tracker.update(9.5, luggage_boxes={5: luggage_box}, person_boxes={1: near_person})
    assert any("alert cleared" in e for e in events)
    assert tracker.luggage[5].unattended is False


def test_ownerless_luggage_never_flagged():
    tracker = LuggageOwnershipTracker(OwnershipConfig(near_distance=10.0, owner_window=1.0, away_time=1.0, static_grace_period=0.0))
    luggage_box = (500, 500, 510, 510)
    far_person = (0, 0, 10, 10)  # always outside near_distance

    events = []
    tracker.update(0.0, luggage_boxes={}, person_boxes={})
    for t in [1.0, 2.0, 5.0, 10.0]:
        events += tracker.update(t, luggage_boxes={7: luggage_box}, person_boxes={1: far_person})

    assert events == []
    assert tracker.luggage[7].owner_id is None
