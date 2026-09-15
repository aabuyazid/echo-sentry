# Embedded model artifacts

`model.tflite` and `normalization.bin` here are embedded into the firmware image via
`EMBED_FILES` in `main/CMakeLists.txt` and read at boot by `inference.c` /
`inference.cc`.

**These are currently a PLACEHOLDER**, trained on synthetic audio
(`training/tests/test_pipeline_integration.py`'s `_hum()` generator), not real
collected data — comprehensive-plan.md's M4 (the data-collection campaign) hasn't
happened yet, since it needs actual hardware in an actual equipment room. They exist
so the firmware build, `EMBED_FILES`, and the on-device TFLM interpreter path can be
built and exercised end-to-end now, per that milestone's own note: "M1–M2 can start
immediately... M4 is the long pole."

**Once a real per-node model exists** (after M4 + `train.py` + `quantize.py` +
`calibrate.py` + `export.py` — see comprehensive-plan.md §4), replace both files with
that node's `model.tflite` and `normalization.bin` from its model directory, and
update `telemetry.c`'s `MODEL_ID_PLACEHOLDER` to read `model.json`'s `model_id`
instead (currently hardcoded — see the `TODO(M5)` there). Each node gets its own pair
of files; this directory holds whichever node the current build targets.
