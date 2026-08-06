# Unattended Object Detector

Unattended-object (abandoned-luggage) detection for video: closed-set person detection +
open-vocabulary luggage detection + [BoxMOT](https://github.com/mikel-brostrom/boxmot)
multi-object tracking + a per-track ownership/abandonment state machine.

```
person  -> YOLO26l (COCO, closed-set)        -- fast, high-confidence, reliable
luggage -> YOLO-World + CLIP (open-vocab)     -- broad vocabulary, described in plain English
              |
              v
        BoxMOT tracker (occluboost by default)
              |
              v
     ownership / abandonment state machine
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together and why the design is
split the way it is.

This is a self-contained `uv` package, laid out as:

```text
unattended_object_detector/
├── pyproject.toml
├── src/
│   └── unattended_object_detector/
│       ├── detector.py      # CLI entry point + main per-frame loop
│       ├── constants.py     # shared class ids
│       ├── prompts.py       # luggage/negative prompt vocabulary — edit here to experiment
│       ├── config.py        # ownership/abandonment tunables — edit here to tune
│       ├── detection.py     # per-frame detection plumbing, frame iteration
│       ├── open_vocab.py    # model-agnostic luggage detector wrapper
│       ├── model.py         # YOLO-World / YOLOE backends — picked by --weights filename
│       ├── tracking.py      # BoxMOT tracker construction, rendering, saving
│       └── ownership.py     # ownership/abandonment state machine (no cv2/ultralytics dep)
└── tests/
    ├── test_ownership.py
    └── test_model.py
```

Nothing here imports scripts from outside this directory.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11+ (uv will fetch a matching interpreter automatically)
- Network access on first run, to download:
  - YOLO26l weights (person detector, ~50MB)
  - YOLO-World weights (luggage detector, ~25-100MB depending on `--weights`)
  - the CLIP text encoder (~350MB)
  - ReID weights for the tracker (auto-downloaded by BoxMOT)

## Install

```bash
uv sync
```

This resolves and installs everything declared in `pyproject.toml` — including this package
itself (installed in editable mode, so edits under `src/` take effect immediately) and the CLIP
text encoder (pulled straight from its GitHub repo — it isn't published on PyPI).

## Usage

`uv sync` installs an `unattended-object-detector` console script into the project's venv:

```bash
# Video file, default tracker (occluboost), live preview window + saved output
uv run unattended-object-detector --source video.mp4

# Webcam, no preview window (e.g. headless/server use)
uv run unattended-object-detector --source 0 --no-show

# Custom luggage vocabulary — open-vocabulary detection needs no retraining,
# just new prompt strings
uv run unattended-object-detector --source video.mp4 \
    --luggage-prompts "duffel bag" "jute sack" "tiffin carrier"

# Different tracker
uv run unattended-object-detector --source video.mp4 --tracker botsort
```

Run `uv run unattended-object-detector --help` for the full set of options, grouped by stage:
person detector, luggage detector, tracker, output, and ownership/abandonment logic.

## Tests

```bash
uv run pytest
```

`ownership.py` — the ownership/abandonment state machine — has no cv2/ultralytics/detector
dependency by design, specifically so it can be unit tested without any of the slow, GPU-bound
detection/tracking machinery (see `tests/test_ownership.py`). `tests/test_model.py` covers
`model.py`'s backend-selection and prompt-sanitization logic the same way — no weights
downloaded, since it never constructs a real `YoloWorldBackend`/`YoloeBackend`.

### Supported sources

Only a video file, a webcam index, or a single image — no directory-of-images batch mode.
Ownership/abandonment tracking needs one continuous scene with a consistent timeline, not a
batch of unrelated stills (see `detection.iter_frames`).

### Output

Annotated frames are shown live (unless `--no-show`) and saved under
`runs/predict/<name>/` (unless `--no-save`): a `.mp4` per video/webcam source, or individual
`.jpg` files for an image source. Abandonment events are also printed to stdout as they occur,
e.g.:

```
Luggage id:3 -> owner assigned: person id:1
ALERT: Luggage id:3 left UNATTENDED by person id:1
Luggage id:3 -> owner id:1 returned, alert cleared
```

By default (no `--debug`), only what matters for the abandonment story is drawn: people who
are a confirmed owner (tagged `[owner]`), and luggage that is observing/owned/unattended —
bystanders and pre-existing/static luggage are left undrawn to keep the frame readable. Pass
`--debug` to draw everything instead: every person and every piece of luggage, including
bystanders and static luggage. This only affects rendering — detection, tracking, and the
printed events are identical either way. See `tracking.draw_tracks`.

## Experimenting with prompts

The luggage vocabulary lives in `src/unattended_object_detector/prompts.py`
(`LUGGAGE_PROMPTS`, `NEGATIVE_PROMPTS`) — open-vocabulary detection needs no retraining, so
editing that file and rerunning is the whole workflow. `--luggage-prompts` is the only
prompt-related CLI override — it replaces `LUGGAGE_PROMPTS` for that run only, without editing
the file (negatives still come from `prompts.py` unless `--no-negatives` is also passed):

```bash
uv run unattended-object-detector --source video.mp4 \
    --luggage-prompts "duffel bag" "jute sack" "tiffin carrier"
```

## Switching the luggage detector backend (YOLO-World / YOLOE)

`--weights` selects both the checkpoint *and* the backend that loads it — `model.py` picks
automatically from the filename, nothing else to configure:

```bash
# Default: YOLO-World v2
uv run unattended-object-detector --source video.mp4 --weights yolov8s-worldv2.pt

# YOLOE — a newer open-vocabulary family; any filename containing "yoloe" selects it.
# Every released YOLOE checkpoint is a segmentation model (filenames always end in
# "-seg" or "-seg-pf") -- this pipeline only reads its boxes, so the mask output is
# unused. Prefer the plain "-seg" variant: "-seg-pf" ("prompt-free") ignores
# --luggage-prompts entirely and always detects its own large built-in vocabulary instead.
uv run unattended-object-detector --source video.mp4 --weights yoloe-26l-seg.pt
```

Everything else — `--luggage-prompts`, `--luggage-conf`, `--iou`, `--imgsz`, `--device` — works
identically regardless of backend; `open_vocab.OpenVocabDetector` only ever calls the small
common surface `model.VocabBackend` exposes (`set_classes`/`predict`/`resolve_label`), never
anything backend-specific. One real API difference is handled transparently: YOLOE rejects any
class name containing a space, so multi-word prompts like `"duffel bag"` get sanitized before
being sent to it and mapped back to the original string when reading detections — see
`model.py`'s module docstring.

## Tuning ownership/abandonment

Every tunable in the ownership state machine lives in
`src/unattended_object_detector/config.py`'s `OwnershipConfig`:

| Field | CLI flag | Default | Meaning |
|---|---|---|---|
| `near_distance_factor` | `--owner-distance-factor` | 0.6 | "Nearby" radius while determining ownership, as a multiple of the candidate person's own box height |
| `owner_window` | `--owner-window` | 3.0 s | How long to observe nearby people before fixing an owner |
| `away_distance_factor` | `--away-distance-factor` | 1.2 | Owner-to-luggage distance beyond which the owner is "away", as a multiple of the owner's own current box height |
| `away_time` | `--away-time` | 10.0 s | Continuous away-time before luggage is flagged unattended |
| `static_grace_period` | `--static-grace-period` | 3.0 s | Luggage seen this soon after stream start is treated as pre-existing furniture, not abandoned |
| `min_reference_height` | `--min-reference-height` | 20 px | Floor under the box-height "ruler" the two factors above multiply against |

**`near_distance_factor`/`away_distance_factor` are not pixel distances** — they're unitless
multiples of a person's own bounding-box height, computed per pair at check time. A fixed pixel
threshold means a different real-world distance depending on where the camera is or how zoomed
in it is (an object twice as close to the camera covers roughly twice as many pixels for the
same real-world gap), so it silently needs re-tuning every time the camera moves. Since a real
person's height is roughly constant, their box height in a given part of the frame is a decent
local proxy for "how many pixels represent about a meter here" — it grows and shrinks with
perspective the same way any other real-world distance in that spot would. See
`config.OwnershipConfig`'s docstring for the full reasoning and its limits (e.g. a crouching
person shrinks their own ruler).

Edit `config.py` to change the defaults everywhere, or override any single field per run with
its CLI flag — same pattern as `--luggage-prompts` above. See `ownership.py`'s module
docstring (and `ARCHITECTURE.md` §5) for what each one actually does in the three-phase state
machine.

## Notes

- `--luggage-conf` defaults far lower than `--person-conf` (0.10 vs 0.25) because
  open-vocabulary models score noticeably lower than closed-set detectors on the same object —
  see `open_vocab.Config`'s docstring.
- `--tracker occluboost` (the default) posts the best HOTA/MOTA/IDF1 in BoxMOT's own benchmark
  table and is purpose-built to hold identity through occlusion — the failure mode that matters
  most here (a person briefly blocked while setting down a bag must keep the same track ID
  afterward). See `tracking.py`.
- First run downloads several hundred MB of weights; subsequent runs are cached locally by
  Ultralytics/BoxMOT/CLIP.
