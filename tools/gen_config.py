#!/usr/bin/env python3
"""Generate firmware headers and the training Python module from config/frontend.yaml.

Usage:
    uv run --project training python tools/gen_config.py

Produces three generated files, each marked GENERATED — DO NOT EDIT and gitignored:
  - firmware/main/gen/frontend_config.h   (C scalars/flags; frontend_core.cc)
  - firmware/main/gen/frontend_tables.h   (C LUTs; via tools/gen_frontend_tables.cc)
  - training/src/echo_sentry_train/frontend_config.py  (Python; model input shape only —
    NOT the DSP internals, which training no longer needs: features.py calls into the
    same compiled frontend_core.cc firmware runs, via tools/build_native_frontend.sh)

The config hash embedded in both remaining generated files is how a mismatch between
firmware and training gets caught at runtime (or in evaluate.py) instead of surfacing
as mysteriously bad anomaly scores — see comprehensive-plan.md §2.2 and §8 ("the
golden-vector test").
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "frontend.yaml"
C_HEADER_PATH = REPO_ROOT / "firmware" / "main" / "gen" / "frontend_config.h"
C_TABLES_PATH = REPO_ROOT / "firmware" / "main" / "gen" / "frontend_tables.h"
PY_MODULE_PATH = REPO_ROOT / "training" / "src" / "echo_sentry_train" / "frontend_config.py"
TABLE_GEN_SRC = REPO_ROOT / "tools" / "gen_frontend_tables.cc"
TABLE_GEN_BIN = REPO_ROOT / "tools" / ".gen_frontend_tables_bin"

HASH_LEN = 6  # hex chars — matches the "cfg" field width used in MQTT payloads / model.json


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of the config's semantic content, independent of YAML formatting/comments."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:HASH_LEN]


def derive(cfg: dict[str, Any]) -> dict[str, Any]:
    audio, frontend, model = cfg["audio"], cfg["frontend"], cfg["model"]

    ms_samples = frontend["window_size_ms"] * audio["sample_rate_hz"]
    if ms_samples % 1000 != 0:
        raise ValueError("window_size_ms * sample_rate_hz must be a whole number of samples")
    window_sample_count = ms_samples // 1000
    if window_sample_count & (window_sample_count - 1) != 0:
        raise ValueError(
            f"window_size_ms ({frontend['window_size_ms']}) * sample_rate_hz "
            f"({audio['sample_rate_hz']}) / 1000 = {window_sample_count} samples, which "
            "is not a power of 2 — required by firmware/main/frontend_core.cc; see "
            "config/frontend.yaml's comment on window_size_ms."
        )
    fft_length = window_sample_count  # no general zero-pad-to-next-pow2 case; see above
    fft_size_log2 = fft_length.bit_length() - 1
    correction_bits = fft_size_log2 - frontend["filter_bank_scaling_bits"] // 2

    mel_channels = frontend["mel_channels"]
    frames_per_vector = model["frames_per_vector"]
    clip_frames = model["clip_frames"]
    vector_stride = model["vector_stride"]
    if clip_frames < frames_per_vector:
        raise ValueError("model.clip_frames must be >= model.frames_per_vector")
    if vector_stride < 1:
        raise ValueError("model.vector_stride must be >= 1")
    vector_dim = mel_channels * frames_per_vector
    vectors_per_clip = ((clip_frames - frames_per_vector) // vector_stride) + 1

    # Neither this frontend nor its predecessor exposes a literal "disable noise
    # reduction" flag — see config/frontend.yaml's comment on enable_noise_reduction.
    # The documented no-op value is min_signal_remaining = 1.0 (always keep 100% of
    # the signal), proven exact in firmware/main/frontend_core.cc's header comment.
    configured_min_signal_remaining = frontend["filter_bank_min_signal_remaining"]
    effective_min_signal_remaining = (
        1.0 if not frontend["enable_noise_reduction"] else configured_min_signal_remaining
    )

    return {
        "window_sample_count": window_sample_count,
        "fft_length": fft_length,
        "fft_size_log2": fft_size_log2,
        "correction_bits": correction_bits,
        "vector_dim": vector_dim,
        "vectors_per_clip": vectors_per_clip,
        "effective_min_signal_remaining": effective_min_signal_remaining,
    }


def _bool_c(v: bool) -> str:
    return "true" if v else "false"


def render_c_header(cfg: dict[str, Any], derived: dict[str, Any], cfg_hash: str) -> str:
    audio, frontend, model, node = cfg["audio"], cfg["frontend"], cfg["model"], cfg["node"]
    return f"""\
// GENERATED FILE — DO NOT EDIT.
// Produced by tools/gen_config.py from config/frontend.yaml (hash {cfg_hash}).
// Re-run `uv run --project training python tools/gen_config.py` after changing that file.
//
// Scalars and flags consumed by firmware/main/frontend_core.cc. The LUTs it also
// needs live in the sibling gen/frontend_tables.h (tools/gen_frontend_tables.cc).

#pragma once

#include <stdbool.h>
#include <stdint.h>

// Hash of config/frontend.yaml at generation time. Firmware reports this in every MQTT
// score payload's "cfg" field; the hub quarantines any score whose cfg disagrees with
// the model it was scored against, and training stamps it into model.json at export.
#define ECHO_SENTRY_CONFIG_HASH "{cfg_hash}"

// --- audio ---
#define ECHO_SENTRY_SAMPLE_RATE_HZ {audio["sample_rate_hz"]}
#define ECHO_SENTRY_BIT_DEPTH     {audio["bit_depth"]}
#define ECHO_SENTRY_CHANNELS      {audio["channels"]}

// --- frontend shape ---
#define ECHO_SENTRY_WINDOW_SIZE_MS      {frontend["window_size_ms"]}
#define ECHO_SENTRY_WINDOW_STRIDE_MS    {frontend["window_stride_ms"]}
#define ECHO_SENTRY_WINDOW_SCALING_BITS {frontend["window_scaling_bits"]}
#define ECHO_SENTRY_MEL_CHANNELS        {frontend["mel_channels"]}

// ⚠️ Both MUST stay false — see comprehensive-plan.md §4.1. Enabling either silently
// suppresses stationary machinery sound, which is the signal this system exists to detect.
static const bool kEchoSentryEnableNoiseReduction = {_bool_c(frontend["enable_noise_reduction"])};
static const bool kEchoSentryEnablePcan = {_bool_c(frontend["enable_pcan"])};

// --- mel filterbank / spectral-subtraction runtime parameters ---
// (Design-time-only parameters — filter_bank_scaling_bits, filter_bank_alignment,
// filter_bank_channel_block_size, pcan_strength/offset/gain_bits — are baked into
// gen/frontend_tables.h's constant arrays and are not needed here.)
#define ECHO_SENTRY_FB_POST_SCALING_BITS        {frontend["filter_bank_post_scaling_bits"]}
#define ECHO_SENTRY_FB_SPECTRAL_SUBTRACTION_BITS {frontend["filter_bank_spectral_subtraction_bits"]}
#define ECHO_SENTRY_FB_SMOOTHING_BITS           {frontend["filter_bank_smoothing_bits"]}
#define ECHO_SENTRY_CORRECTION_BITS             {derived["correction_bits"]}
static const float kEchoSentryEvenSmoothing = {frontend["filter_bank_even_smoothing"]}f;
static const float kEchoSentryOddSmoothing = {frontend["filter_bank_odd_smoothing"]}f;
static const float kEchoSentryMinSignalRemaining = {derived["effective_min_signal_remaining"]}f;
static const bool kEchoSentryFilterBankClamping = {_bool_c(frontend["filter_bank_clamping"])};

// --- model input shape ---
#define ECHO_SENTRY_FRAMES_PER_VECTOR {model["frames_per_vector"]}
#define ECHO_SENTRY_CLIP_FRAMES       {model["clip_frames"]}
#define ECHO_SENTRY_VECTOR_STRIDE     {model["vector_stride"]}
#define ECHO_SENTRY_VECTOR_DIM        {derived["vector_dim"]}       // mel_channels * fpv
#define ECHO_SENTRY_VECTORS_PER_CLIP  {derived["vectors_per_clip"]} // from clip_frames/stride

// --- node defaults (overridable at runtime via NVS; see firmware/main/provisioning.c) ---
#define ECHO_SENTRY_ANOMALY_THRESHOLD_DEFAULT {node["anomaly_threshold_default"]}f
"""


def render_py_module(cfg: dict[str, Any], derived: dict[str, Any], cfg_hash: str) -> str:
    audio, frontend, model, node = cfg["audio"], cfg["frontend"], cfg["model"], cfg["node"]
    return f'''\
"""GENERATED FILE — DO NOT EDIT.

Produced by tools/gen_config.py from config/frontend.yaml (hash {cfg_hash}).
Re-run `uv run --project training python tools/gen_config.py` after changing that file.

Model input-shape constants only — the DSP internals (window/filterbank/PCAN
parameters) live solely in firmware/main/gen/frontend_config.h and
gen/frontend_tables.h now. training/src/echo_sentry_train/features.py calls into the
SAME compiled frontend_core.cc firmware runs (via tools/build_native_frontend.sh)
rather than recomputing features from these parameters in Python — see
comprehensive-plan.md §8, "the golden-vector test", and frontend_core.cc's header
comment for why that's a stronger bit-exactness guarantee than two independent
implementations of the same math.
"""

from __future__ import annotations

CONFIG_HASH = "{cfg_hash}"

# --- audio ---
SAMPLE_RATE_HZ = {audio["sample_rate_hz"]}
BIT_DEPTH = {audio["bit_depth"]}
CHANNELS = {audio["channels"]}

# --- frontend shape (informational — features.py gets the authoritative values
# straight from the compiled library at import time) ---
WINDOW_SIZE_MS = {frontend["window_size_ms"]}
WINDOW_STRIDE_MS = {frontend["window_stride_ms"]}
MEL_CHANNELS = {frontend["mel_channels"]}

# --- model input shape ---
FRAMES_PER_VECTOR = {model["frames_per_vector"]}
CLIP_FRAMES = {model["clip_frames"]}
VECTOR_STRIDE = {model["vector_stride"]}
VECTOR_DIM = {derived["vector_dim"]}  # mel_channels * frames_per_vector
VECTORS_PER_CLIP = {derived["vectors_per_clip"]}

QUANTIZATION = "{model["quantization"]}"

# --- node defaults ---
ANOMALY_THRESHOLD_DEFAULT = {node["anomaly_threshold_default"]}
'''


def build_table_gen_binary() -> Path:
    TABLE_GEN_BIN.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["clang++", "-std=c++17", "-O2", "-o", str(TABLE_GEN_BIN), str(TABLE_GEN_SRC)],
        check=True,
    )
    return TABLE_GEN_BIN


def generate_tables_header(cfg: dict[str, Any], derived: dict[str, Any]) -> str:
    frontend = cfg["frontend"]
    binary = build_table_gen_binary()
    args = [
        str(binary),
        f"--sample_rate={cfg['audio']['sample_rate_hz']}",
        f"--window_sample_count={derived['window_sample_count']}",
        f"--window_scaling_bits={frontend['window_scaling_bits']}",
        f"--num_channels={frontend['mel_channels']}",
        f"--lower_band_limit={frontend['lower_band_limit_hz']}",
        f"--upper_band_limit={frontend['upper_band_limit_hz']}",
        f"--fft_length={derived['fft_length']}",
        f"--filter_bank_scaling_bits={frontend['filter_bank_scaling_bits']}",
        f"--filter_bank_alignment={frontend['filter_bank_alignment']}",
        f"--filter_bank_channel_block_size={frontend['filter_bank_channel_block_size']}",
        f"--pcan_strength={frontend['pcan_strength']}",
        f"--pcan_offset={frontend['pcan_offset']}",
        f"--pcan_gain_bits={frontend['pcan_gain_bits']}",
        f"--pcan_smoothing_bits={frontend['pcan_smoothing_bits']}",
        f"--correction_bits={derived['correction_bits']}",
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout


def main() -> int:
    cfg = load_config()
    derived = derive(cfg)
    cfg_hash = config_hash(cfg)

    C_HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    PY_MODULE_PATH.parent.mkdir(parents=True, exist_ok=True)

    C_HEADER_PATH.write_text(render_c_header(cfg, derived, cfg_hash))
    PY_MODULE_PATH.write_text(render_py_module(cfg, derived, cfg_hash))
    C_TABLES_PATH.write_text(generate_tables_header(cfg, derived))

    print(f"config hash: {cfg_hash}")
    print(f"wrote {C_HEADER_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {C_TABLES_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {PY_MODULE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
