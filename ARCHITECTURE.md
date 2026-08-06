# Architecture — Unattended Object Detector

This document describes how the pipeline in this directory is structured, how a frame flows
through it end to end, and why the design is split the way it is — mainly two deliberate
decisions: a hybrid closed-set/open-vocabulary detector pair, and a dependency-free ownership
state machine kept separate from all the vision/tracking machinery.

---

## 1. Overview

The system decides, per tracked piece of luggage, whether it has been **abandoned**: left
alone by whoever brought it, for longer than a configurable grace period, after an
observation window established who that person was in the first place. It runs as one
continuous per-frame pipeline over a video file, webcam stream, or single image.

```
CLI (detector.main)
  └─ per-frame loop
       ├─ detect_person()         closed-set COCO detector (YOLO26l), person only
       ├─ OpenVocabDetector        open-vocabulary detector (YOLO-World + CLIP), luggage vocabulary
       │    └─ build_luggage_dets()  collapse many prompts -> one LUGGAGE class, de-dup via NMS
       ├─ tracker.update()         BoxMOT (occluboost by default) — assigns/maintains track ids
       ├─ LuggageOwnershipTracker.update()   ownership + abandonment state machine
       └─ draw_tracks()            render boxes, labels, owner-connector lines
```

Nine modules, each with a narrow, independent responsibility, all living under
`src/unattended_object_detector/` and talking to each other through package-relative imports
(e.g. `detector.py` does `from .tracking import ...`):

| Module | Responsibility |
|---|---|
| `constants.py` | The two tracked class ids (`PERSON`, `LUGGAGE`) and their display names. |
| `prompts.py` | The open-vocabulary luggage/negative prompt lists — edit here to experiment. |
| `config.py` | `OwnershipConfig` — every tunable in the ownership/abandonment state machine, edit here to tune. |
| `open_vocab.py` | Model-agnostic luggage detector wrapper: vocabulary encoding, per-frame detection. |
| `model.py` | YOLO-World / YOLOE backend implementations, picked by `--weights` filename. |
| `detection.py` | Turns both detectors' outputs into one common `(N, 6)` array shape; frame source iteration. |
| `tracking.py` | BoxMOT tracker construction/tuning, rendering, output saving, frame timing. |
| `ownership.py` | Pure ownership/abandonment state machine — no cv2/ultralytics/detector dependency. |
| `detector.py` | CLI wiring: argument parsing and the main per-frame loop tying the above together. |

`detector.main` is exposed as the `unattended-object-detector` console script
(`project.scripts` in `pyproject.toml`), installed into the project's venv by `uv sync`.
`ownership.py`'s dependency-freedom is what makes `tests/test_ownership.py` possible without
any of the GPU-bound detection/tracking machinery — see §5.

---

## 2. Why two different detectors for person vs. luggage

COCO's own luggage classes (backpack/handbag/suitcase) miss a lot of real luggage shapes —
duffel bags, cardboard boxes, sacks, trolley bags — no fixed training set covers that whole
space. Open-vocabulary detection (YOLO-World + CLIP) sidesteps that: describe what you're
looking for in words, and the model finds it. Adding a new luggage type is adding a string to
a list in `prompts.LUGGAGE_PROMPTS` (or via `--luggage-prompts`), no retraining involved.

But COCO's own person class is already excellent — high, well-calibrated confidence — and
there's no reason to pay open-vocabulary detection's noise tax on something a closed-set
detector already does better and faster. So detection is a hybrid, not a single model:

```
person  -> YOLO26l (COCO, closed-set)         -- fast, high-confidence, reliable
luggage -> OpenVocabDetector (open_vocab.py)  -- broad vocabulary where it's needed
```

Dropping "person" from the open-vocabulary vocabulary also sharpens luggage precision on its
own (fewer competing prompts for CLIP to disambiguate against), and person detections at high
confidence skip the tracker's multi-frame tentative-track period entirely — only the
lower-confidence luggage class still needs it (see §4, confidence-threshold override).

---

## 3. Detection

### 3.1 Person (`detection.detect_person`)

Runs YOLO26l restricted to COCO's `person` class id (`classes=[COCO_PERSON_CLASS_ID]`),
producing an `(N, 6)` array of `[x1, y1, x2, y2, conf, class_id]` rows, `class_id` remapped to
this pipeline's own `PERSON_CLASS_ID` — kept as a distinct constant from COCO's numbering (see
`constants.py`) even though both happen to be `0`, so the two meanings never get silently
conflated if either changes independently.

### 3.2 Luggage (`open_vocab.OpenVocabDetector`, `model.py`)

`OpenVocabDetector` is a thin, **model-agnostic** wrapper: it holds a `Config` (weights, conf,
iou, imgsz, device, prompts) and calls exactly three methods on whatever backend `model.py`
hands it — `set_classes`, `predict`, `resolve_label` (the `model.VocabBackend` protocol) —
never anything backend-specific. The expensive step — installing the prompt vocabulary
(`set_prompts`) — happens once before the frame loop starts, never per frame; calling it again
with an unchanged prompt list is a cheap no-op. Each `detect()` call returns
`(matched_prompt, confidence, box)` tuples sorted by confidence, one entry per detected box,
for whichever prompt (luggage or negative) scored highest.

**Backend selection (`model.py`).** Two backends today, both loaded lazily (constructing a
detector never requires network access — see `_ensure_backend`):

- `YoloWorldBackend` — wraps `ultralytics.YOLOWorld`. `set_classes(names)` takes plain strings.
- `YoloeBackend` — wraps `ultralytics.YOLOE`, a newer open-vocabulary family (e.g.
  `yoloe-26l-seg.pt`). Every released YOLOE checkpoint is a segmentation model (filenames
  always carry a `-seg` suffix); this pipeline only reads `Results.boxes`, so the mask output
  is simply unused. `YOLOE.set_classes` computes text embeddings itself via `get_text_pe()`
  when none are passed, so `YoloeBackend.set_classes` just calls `set_classes(names)` the same
  shape as YOLO-World — **except** YOLOE's implementation asserts no class name contains a
  space, which the multi-word prompts in `prompts.py` (`"duffel bag"`, `"an unattended object
  on the floor"`, ...) would violate outright. `YoloeBackend` sanitizes (`prompt.replace(" ",
  "_")`) before calling `set_classes`, keeps the sanitized→original mapping, and
  `resolve_label()` reverses it when `OpenVocabDetector.detect()` reads labels back off
  `Results.names` — so `detection.classify_label`'s matching against the original
  `prompts.LUGGAGE_PROMPTS` strings works identically regardless of which backend is loaded.

`build_backend(weights)` (`model.py`) picks the class purely from the weights filename —
anything containing `"yoloe"` selects `YoloeBackend`, everything else (`yolov8*-world*.pt`, ...)
selects `YoloWorldBackend` — so switching models is just changing `--weights`, with no separate
flag to keep in sync. Adding a third backend later means adding one more class to `model.py`,
not touching `OpenVocabDetector` or `detector.py`'s per-frame loop at all.

**Negative prompts** (`chair`, `table`, `door`, `trash can`, `floor`, `wall` — `prompts.
NEGATIVE_PROMPTS`) are fed to the detector alongside the luggage vocabulary but never tracked
— filtered out downstream. Not optional in practice: without them, an open-vocabulary model
has no way to say "this is NOT luggage" and force-fits every blob onto the nearest luggage
prompt. Deliberately excludes `"person"` — see `prompts.py`'s module docstring: person
detection is handled entirely by the separate closed-set COCO detector (§2), so adding it here
would only cost the text encoder one more competing prompt to disambiguate against, for no
benefit.

### 3.3 Collapsing prompts to one class (`detection.build_luggage_dets`)

All luggage-ish prompts ("suitcase", "backpack", "duffel bag", ...) collapse into **one**
tracked class (`LUGGAGE_CLASS_ID`) once tracking starts — the specific prompt that fired no
longer matters, only "is this luggage" (`make_luggage_classifier`).

This collapsing creates a side effect that needs its own fix: YOLO-World runs NMS *per prompt*,
not per collapsed class, so "backpack", "duffel bag", and "handbag" firing on the exact same
physical bag can all survive as separate boxes — each was the top scorer for its own prompt.
Once collapsed to one class, that reads as several overlapping detections of "the same object"
and — confirmed empirically — produces multiple separate track ids for one physical bag.
`detection._nms_per_class` runs ordinary class-scoped NMS (via `cv2.dnn.NMSBoxes`) on the
already-collapsed classes to fix it, before the array ever reaches the tracker.

### 3.4 Frame sources (`detection.iter_frames`)

Yields `(source_path, frame)` pairs for a webcam index (bare digit string), a single image, or
a video file. No directory-of-images mode: unlike a plain detection test harness, ownership/
abandonment tracking needs one continuous scene with a consistent timeline, not a batch of
unrelated stills.

---

## 4. Tracking (`tracking.py`)

Talks to BoxMOT at the low level via `create_tracker` + `tracker.update(dets, frame)`, not its
high-level facade — that facade is broken in the currently pinned release (raises
`ModuleNotFoundError: No module named 'boxmot.data'`). The low-level path is fully functional
and is all this pipeline needs.

**Default tracker: `occluboost`.** In BoxMOT's own benchmark table it posts the best
HOTA/MOTA/IDF1 on MOT17 and SportsMOT of any tracker in the repo, and it's purpose-built to
hold a track's identity through occlusion — exactly the failure mode that matters here: a
person crouches to set down a bag behind another person, or is briefly blocked by someone
walking past, and must keep the same ID afterward for "person X left, bag still there" logic
to hold.

`build_tracker` layers two fixes on top of occluboost's shipped defaults:

1. **GTA-based full-disappearance recovery.** occluboost disables GTA (Global Track
   Association) by default and caps its alive-phase recovery window (`gta_max_gap`) far below
   how long a track survives before dying (`max_age`) — leaving a "dead zone" where a person
   who's fully left the frame and come back gets a new track id even though GTA is nominally
   on. Enabling GTA and widening `gta_max_gap` to exceed `max_age` closes that gap: a person
   can be fully out of frame for up to `--reappear-window` seconds and keep their id on return.
2. **Confidence-threshold override.** occluboost's shipped config requires ~0.57 confidence to
   match a track and ~0.71 to start a new one. Open-vocabulary luggage detections run far
   lower than that by design (`--luggage-conf` defaults to 0.10) — left alone, every luggage
   detection would be silently dropped and no luggage track ever created, regardless of how
   good the boxes are. Both thresholds are tied to `luggage_conf`, the lower of the two
   detectors' floors; person detections clear it easily, so this doesn't loosen anything on
   the person side.

`FrameClock` (§5 below) and `probe_fps` exist so that time-based logic — both the tracker's
own frame-based aging counters and the ownership state machine's second-based timers — can be
driven off the video's own timeline when it's known, rather than wall-clock processing speed.

---

## 5. Ownership / abandonment state machine (`ownership.py`, `config.py`)

Pure logic, no cv2/ultralytics/detector dependency — fed per-frame person/luggage track boxes
(identity from the tracker) and a monotonically increasing timestamp. Dependency-free by
design, so it's trivial to unit test in isolation from the slow, GPU-bound detection/tracking
machinery.

`LuggageOwnershipTracker.__init__` takes a single `config.OwnershipConfig` (defaults to
`OwnershipConfig()` if omitted), mirroring `open_vocab.OpenVocabDetector`'s own `cfg`-taking
constructor (§3.2). Every number below — near/away distance factors, the owner and away
windows, the static grace period, the reference-height floor — is a field on that dataclass,
kept in its own module for the same reason `prompts.py` holds the vocabulary separately: one
place to tune per camera/scene without touching the state-machine logic itself. `detector.py`
builds an `OwnershipConfig` from the matching `--owner-distance-factor`/`--owner-window`/
`--away-distance-factor`/`--away-time`/`--static-grace-period`/`--min-reference-height` flags,
whose own argparse defaults are read from `OwnershipConfig()` — so the CLI help text and the
tracker's defaults can never drift out of sync.

Three phases per luggage track, driven by `LuggageOwnershipTracker.update()`:

1. **Static-furniture check.** Luggage first seen within `--static-grace-period` seconds of
   the very first `update()` call is assumed to have already been sitting there when
   observation started — scene furniture (a cabinet, a fixed bin, ...) that the detector
   mistook for luggage, not something a person just abandoned. It's marked `is_static` and
   permanently exempted from the two phases below. Without this, a persistent false-positive
   detection eventually gets *some* nearby person voted "owner" by pure chance and is then
   flagged unattended the moment that unrelated bystander walks off.
2. **Ownership window** (`--owner-window` seconds from when the luggage track first appears).
   Every person within `--owner-distance-factor` × *that person's own* box height of the
   luggage accumulates time-spent-nearby. When the window closes, whoever accumulated the most
   time is fixed as the owner. If nobody was ever nearby, the luggage stays ownerless and
   phase 3 never runs for it.
3. **Abandonment watch.** Once an owner is fixed, the owner-to-luggage distance is tracked on
   every update against `--away-distance-factor` × *the owner's own current* box height. If it
   stays above that continuously for `--away-time` seconds, the luggage is flagged
   `unattended`. The flag is not latched — it clears automatically if the owner comes back
   within that same distance. A missing owner track (occluded, or has walked out of frame)
   counts as infinitely far away regardless of the threshold.

Distance uses each box's **bottom-center point** (`bottom_center`), not its centroid — a much
better proxy for "where this object/person actually is on the floor" than a box centroid,
which for a tall person floats near their chest.

### 5.1 Scale-adaptive thresholds: why not a flat pixel distance

A fixed pixel threshold (the original design) silently means a different real-world distance
depending on camera position or zoom: an object twice as close to the camera covers roughly
twice as many pixels for the same real-world gap, so a constant tuned for one camera setup
needs re-tuning every time the camera moves — see the "known limitations" bullet this section
used to carry, now addressed here instead.

`near_distance_factor` and `away_distance_factor` are unitless multipliers instead, applied
against a "ruler" computed per pair *at check time*: the relevant person's own current
bounding-box height (`ownership._box_height`, floored at `min_reference_height` via
`LuggageOwnershipTracker._reference_height` to avoid a degenerate near-zero-height box
collapsing its own threshold to ~0). A real person's height is roughly constant regardless of
who they are, so how tall their box is *in that part of the frame* is a decent local proxy for
"how many pixels currently represent about a meter here" — it scales with perspective the same
way any other real-world distance at that spot would, with zero calibration required. Phase 2
uses each *candidate* person's own height (different people at different depths in the same
frame get different, correctly-scaled thresholds); phase 3 re-derives the ruler from the
*owner's* current box every update, so it keeps tracking correctly even as the owner walks
toward or away from the camera mid-scene.

This is an approximation, not a calibrated measurement: it assumes a roughly-constant real
person height and a person standing roughly upright and facing the camera. A person crouching,
lying down, or foreshortened at a steep camera angle shrinks their own ruler and makes the
resulting threshold too small for that moment — see §9's known limitations for where a fully
calibrated approach (e.g. a homography mapping image pixels to real-world ground-plane
coordinates) would do better, at the cost of a per-camera calibration step this design
deliberately avoids.

`update()` returns human-readable event strings for anything that changed that frame (owner
assigned, alert raised, alert cleared) — printed by `detector.main`, not parsed by anything
downstream.

---

## 6. Rendering and output (`tracking.draw_tracks`, `tracking.OutputWriter`)

`draw_tracks` renders every current track onto a copy of the frame. Person boxes get a
per-id deterministic color (`id_to_color`, hashed from the track id) and an
`id:<n> person <conf>` label, with `[owner]` appended if that person currently owns a piece of
luggage. Luggage boxes are color-coded by ownership status:

| Color | Label | Meaning |
|---|---|---|
| gray | `(pre-existing, ignored)` | flagged static (§5, phase 1) |
| cyan | `(observing)` | still inside the ownership window |
| green | `owner:<id>` | owned, owner currently nearby |
| red | `UNATTENDED! owner id:<n>` | owner has been away too long |

A thin connector line links a piece of luggage to its owner once one is known, colored to
match the luggage box (green = attended, red = unattended).

**`--debug` controls how much of this gets drawn**, once ownership tracking is active
(`--no-luggage-logic` not passed). Without `--debug` (the default), only what matters for the
abandonment story is drawn: people who are a *confirmed* owner (`state.owner_id`, phase 2 of
§5 — not people merely being observed as ownership candidates), and luggage in the
observing/owned/unattended states; bystanders and static/pre-existing luggage are skipped
entirely so the frame stays readable. The owner-connector line is unaffected by the flag
either way, since both its endpoints are already being drawn whenever it would appear. With
`--debug`, every person and every piece of luggage is drawn, exactly as described above. With
`luggage_state=None` (`--no-luggage-logic`), there's no ownership concept to filter by, so
`--debug` has no effect — plain per-class boxes are always drawn for every track.

`OutputWriter` saves annotated frames: video/webcam sources accumulate into one `.mp4` per
source (keyed by source path, opened lazily on first frame so its resolution/fps are known);
image sources are written as individual `.jpg` files.

---

## 7. Data flow

```
frame (BGR) ──┬─► detect_person() ──────────────┐
              │                                   │
              └─► OpenVocabDetector.detect() ──► build_luggage_dets()
                                                   │
                                    concat ────────┘
                                      │
                                      ▼
                              tracker.update(dets, frame)
                                      │
                                      ▼
                         tracks: [x1,y1,x2,y2,id,conf,cls,det_ind]
                          │                              │
                          ▼                              ▼
              LuggageOwnershipTracker.update()      draw_tracks()
              (per luggage/person track boxes,            │
               clock.tick() timestamp)                    │
                          │                                │
                          └──────► luggage_state ──────────┘
                                                            │
                                                            ▼
                                              annotated frame ──► imshow / OutputWriter
```

**Per-frame, transient:** raw detection arrays, the concatenated `dets` array, `tracks`, the
annotated frame.
**Cross-frame, retained:** the tracker's internal state (BoxMOT), `LuggageOwnershipTracker.
luggage` (one `LuggageState` per track id ever seen), `FrameClock`'s elapsed time, and
`OutputWriter`'s open video writers.

---

## 8. Extension points

- **New luggage vocabulary.** Add strings to `prompts.LUGGAGE_PROMPTS`/`NEGATIVE_PROMPTS` or
  pass `--luggage-prompts` (the only prompt-related CLI override — `NEGATIVE_PROMPTS` isn't
  meant to vary per run) — no retraining, just a text-encoder re-encode at startup.
- **Different open-vocabulary backend.** Switch `--weights` between a YOLO-World checkpoint and
  a YOLOE one (e.g. `yoloe-26l-seg.pt`) — `model.build_backend` picks the class from the
  filename automatically (§3.2). Adding a third family means adding one more class to
  `model.py` implementing `set_classes`/`predict`/`resolve_label`; `open_vocab.py` and
  `detector.py` don't need to change.
- **Different tracker.** Any of `tracking.TRACKER_CHOICES` works via `--tracker`; only
  `occluboost` gets the two tuning fixes in §4 (GTA widening, confidence-threshold override) —
  other trackers use BoxMOT's shipped defaults unmodified.
- **Multi-camera / re-identification across cameras.** Out of scope today — `OutputWriter` and
  the ownership tracker both assume one continuous single-camera scene per run.
- **Alerting integration.** `LuggageOwnershipTracker.update()`'s returned event strings are
  currently just printed; routing them to a message queue, webhook, or alarm system is a
  matter of replacing that `print(event)` call in `detector.main`.

## 9. Known limitations

- **Static-furniture detection is time-based, not re-evaluated.** A track marked `is_static`
  at creation stays exempt forever, even if it later turns out to be a real abandoned bag that
  happened to be flagged within the grace period after a mid-stream restart.
- **Ownership is decided once, within a fixed window, and never re-opened.** If the true owner
  wasn't within `--owner-distance-factor` × their own box height during the observation window
  (e.g. still walking into frame), the luggage stays ownerless for the rest of the run and
  never enters phase 3 at all.
- **Scale-adaptive thresholds are an approximation, not a calibrated measurement (§5.1).**
  `--owner-distance-factor`/`--away-distance-factor` scale with a person's own box height
  instead of a flat pixel radius, which fixes the worst of the old scene-geometry sensitivity
  (a threshold tuned on one camera no longer silently means a different real-world distance on
  another), but it still assumes a roughly-constant real person height and a person standing
  roughly upright, facing the camera. A crouching, lying-down, or steeply foreshortened person
  shrinks their own "ruler" and can make the threshold too tight for that moment. A wide-angle
  or oblique view can also distort a single person's box height itself (e.g. partial
  occlusion). The fully correct fix — a homography mapping image pixels to real-world
  ground-plane coordinates — would eliminate this, at the cost of a per-camera calibration step
  this design deliberately avoids.
- **Single combined luggage class.** Once prompts collapse to `LUGGAGE_CLASS_ID`, there's no
  way to tell downstream what kind of object it was (backpack vs. cardboard box) — only that
  it matched *some* luggage-ish prompt.
