# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `pyproject.toml` making this directory a runnable `uv` project (`uv sync`, `uv run
  unattended-object-detector ...`), with `boxmot`, `ultralytics`, `opencv-python`, `numpy`, and
  CLIP (via `tool.uv.sources`, pulled from its GitHub repo) as dependencies. Pinned to
  Python 3.11+.
- `README.md`, `ARCHITECTURE.md` documenting setup, usage, and the pipeline's design.
- `pytest` (dev dependency) and `tests/test_ownership.py`, covering all three phases of the
  ownership/abandonment state machine.
- `--debug` CLI flag. Without it, `draw_tracks` now only draws confirmed owners and
  observing/owned/unattended luggage, skipping bystanders and static/pre-existing luggage;
  with it, every person and every piece of luggage is drawn as before. Has no effect under
  `--no-luggage-logic`, where there's no ownership concept to filter by.
- `config.py` with `OwnershipConfig`, a dataclass listing every tunable in the ownership/
  abandonment state machine (`near_distance`, `owner_window`, `away_distance`, `away_time`,
  `static_grace_period`) in one place, edit-here-to-tune — mirrors `prompts.py`'s role for the
  luggage vocabulary.

### Changed

- Restructured into a proper src-layout package: the six modules moved from the project root
  into `src/unattended_object_detector/`, with an `__init__.py` and relative imports between
  them (previously bare top-level imports, e.g. `from constants import ...`, that only worked
  because Python adds a directly-run script's own directory to `sys.path`).
- Added a `hatchling` build backend and a `unattended-object-detector` console-script entry
  point (`project.scripts`), installed in editable mode by `uv sync`. Invocation changed from
  `uv run detector.py ...` to `uv run unattended-object-detector ...`.
- `--project`'s default output directory (`runs/predict`) is now resolved relative to the
  current working directory rather than the package's own install location — necessary once
  the pipeline runs as an installed console script that could be invoked from anywhere.
- Moved `LUGGAGE_PROMPTS`/`DISTRACTOR_PROMPTS` out of `open_vocab.py` into their own
  `prompts.py`, so the vocabulary can be experimented with in one place without touching
  detection logic. `open_vocab.py` and `detector.py` now import from `prompts.py`.
- Renamed `DISTRACTOR_PROMPTS` to `NEGATIVE_PROMPTS` (and `--no-distractors` to
  `--no-negatives`) in `prompts.py`. `"person"` remains deliberately excluded from it — person
  detection is handled entirely by the separate closed-set COCO detector (§2 of
  `ARCHITECTURE.md`), so adding it back would only cost CLIP one more competing prompt to
  disambiguate luggage against. `--luggage-prompts` remains the only prompt-related CLI
  override; `NEGATIVE_PROMPTS` has none, since it isn't meant to vary per run.
- `LuggageOwnershipTracker.__init__` now takes a single `OwnershipConfig` instead of five
  separate keyword arguments, mirroring `OpenVocabDetector`'s own `cfg`-taking constructor.
  `detector.py`'s ownership CLI flags (`--owner-distance`, `--owner-window`, `--away-distance`,
  `--away-time`, `--static-grace-period`) now read their argparse defaults from
  `OwnershipConfig()` instead of duplicating the literals, so the CLI help and the tracker's
  own defaults can't drift out of sync.

## [0.1.0] - 2026-08-06

### Added

- Initial standalone extraction of the unattended-object detection pipeline: `detector.py`
  (CLI + main loop), `constants.py` (shared class ids), `detection.py` (per-frame detection
  plumbing and frame iteration), `open_vocab.py` (YOLO-World + CLIP open-vocabulary luggage
  detector), `tracking.py` (BoxMOT tracker construction, rendering, output saving), and
  `ownership.py` (dependency-free ownership/abandonment state machine).
- Hybrid detection: closed-set COCO person detection (YOLO26l) + open-vocabulary luggage
  detection (YOLO-World/CLIP), unified into one tracked-class array for BoxMOT.
- `occluboost` as the default tracker, with GTA-based full-disappearance recovery and a
  confidence-threshold override tuned for open-vocabulary luggage detections.
- Three-phase luggage ownership/abandonment logic: static-furniture exemption, an
  observation-window owner vote, and a distance/time-based unattended-object alert.
