"""The Python half of the frontend — comprehensive-plan.md §8, "the golden-vector test".

Calls into firmware/main/frontend_core.cc, compiled natively for this machine by
tools/build_native_frontend.sh, via ctypes — the SAME compiled implementation
firmware runs on-device (cross-compiled for the ESP32-S3 from the identical source),
rather than a second Python implementation that merely aims to agree with it. See
frontend_core.cc's header comment for the full rationale and the two non-obvious
details (PCAN's hardcoded snr_shift, FilterbankConfig's unused fields) worth knowing
before touching either side.

This module is deliberately the ONLY place training computes log-mel features.
dataset.py, clean.py, calibrate.py, and evaluate.py all go through it — never through
librosa, which is exploration-only (see training/pyproject.toml).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from echo_sentry_train import frontend_config as fc

_LIB_PATH = Path(__file__).parent / "libechosentry_frontend.dylib"


def _load_library() -> ctypes.CDLL:
    if not _LIB_PATH.exists():
        raise FileNotFoundError(
            f"{_LIB_PATH} not found — run tools/build_native_frontend.sh first "
            "(requires firmware/managed_components/espressif__esp-tflite-micro to "
            "already be fetched: run `idf.py reconfigure` in firmware/ once)."
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.echo_sentry_frontend_create.argtypes = []
    lib.echo_sentry_frontend_create.restype = ctypes.c_void_p
    lib.echo_sentry_frontend_destroy.argtypes = [ctypes.c_void_p]
    lib.echo_sentry_frontend_destroy.restype = None
    lib.echo_sentry_frontend_reset.argtypes = [ctypes.c_void_p]
    lib.echo_sentry_frontend_reset.restype = None
    lib.echo_sentry_frontend_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
    ]
    lib.echo_sentry_frontend_process_frame.restype = None
    lib.echo_sentry_frontend_window_sample_count.argtypes = []
    lib.echo_sentry_frontend_window_sample_count.restype = ctypes.c_int
    lib.echo_sentry_frontend_mel_channels.argtypes = []
    lib.echo_sentry_frontend_mel_channels.restype = ctypes.c_int
    return lib


_lib = _load_library()

WINDOW_SAMPLE_COUNT = _lib.echo_sentry_frontend_window_sample_count()
MEL_CHANNELS = _lib.echo_sentry_frontend_mel_channels()
WINDOW_STRIDE_SAMPLES = fc.WINDOW_STRIDE_MS * fc.SAMPLE_RATE_HZ // 1000

if MEL_CHANNELS != fc.MEL_CHANNELS:
    raise RuntimeError(
        f"native frontend library reports mel_channels={MEL_CHANNELS}, but "
        f"frontend_config.py (from the same config/frontend.yaml) says {fc.MEL_CHANNELS} "
        "— rebuild the library: tools/build_native_frontend.sh"
    )


class FrontendSession:
    """One clip/file's worth of frontend state.

    Wraps a FrontendState handle: create it once per independent audio stream (a WAV
    file during training; conceptually the node's whole uptime in firmware — see
    frontend_core.h on why noise-reduction state is NOT reset per scored clip
    on-device). Use as a context manager or call close() explicitly.
    """

    def __init__(self) -> None:
        self._handle = _lib.echo_sentry_frontend_create()
        if not self._handle:
            raise RuntimeError("echo_sentry_frontend_create() failed")

    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        if audio_frame.dtype != np.int16 or audio_frame.shape != (WINDOW_SAMPLE_COUNT,):
            raise ValueError(
                f"expected int16 array of shape ({WINDOW_SAMPLE_COUNT},), got "
                f"{audio_frame.dtype} shape {audio_frame.shape}"
            )
        audio_frame = np.ascontiguousarray(audio_frame)
        out = np.zeros(MEL_CHANNELS, dtype=np.int16)
        _lib.echo_sentry_frontend_process_frame(
            self._handle,
            audio_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )
        return out

    def close(self) -> None:
        if self._handle:
            _lib.echo_sentry_frontend_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> FrontendSession:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def compute_log_mel_frames(pcm_int16: np.ndarray) -> np.ndarray:
    """PCM (1D int16, mono, SAMPLE_RATE_HZ) -> log-mel frames, shape (num_frames, MEL_CHANNELS).

    One FrontendSession per call — matches training's per-file granularity. Firmware's
    equivalent session spans its whole uptime, not per-clip; see FrontendSession's
    docstring. This asymmetry is safe today because noise reduction is disabled
    (config/frontend.yaml) and spectral subtraction is a proven no-op when disabled
    (frontend_core.cc's header comment) — the noise-estimate state this affects is
    otherwise irrelevant to the output. Re-examine this if noise reduction is ever
    enabled.
    """
    if pcm_int16.dtype != np.int16:
        raise ValueError(f"expected int16 PCM, got {pcm_int16.dtype}")
    if pcm_int16.ndim != 1:
        raise ValueError(f"expected 1D mono PCM, got shape {pcm_int16.shape}")

    num_frames = (
        0
        if len(pcm_int16) < WINDOW_SAMPLE_COUNT
        else (len(pcm_int16) - WINDOW_SAMPLE_COUNT) // WINDOW_STRIDE_SAMPLES + 1
    )
    frames = np.zeros((num_frames, MEL_CHANNELS), dtype=np.int16)

    with FrontendSession() as session:
        for i in range(num_frames):
            start = i * WINDOW_STRIDE_SAMPLES
            frames[i] = session.process_frame(pcm_int16[start : start + WINDOW_SAMPLE_COUNT])

    return frames


def frames_to_vectors(frames: np.ndarray) -> np.ndarray:
    """Concatenate FRAMES_PER_VECTOR consecutive frames into VECTOR_DIM-wide feature
    vectors, subsampled by VECTOR_STRIDE — comprehensive-plan.md §4.1.

    frames: (num_frames, MEL_CHANNELS) -> vectors: (num_vectors, VECTOR_DIM)
    """
    num_frames, mel = frames.shape
    if mel != fc.MEL_CHANNELS:
        raise ValueError(f"expected {fc.MEL_CHANNELS} mel channels, got {mel}")

    fpv = fc.FRAMES_PER_VECTOR
    stride = fc.VECTOR_STRIDE
    if num_frames < fpv:
        return np.empty((0, fc.VECTOR_DIM), dtype=frames.dtype)

    starts = range(0, num_frames - fpv + 1, stride)
    vectors = np.stack([frames[s : s + fpv].reshape(-1) for s in starts], axis=0)
    return vectors


def pcm_to_vectors(pcm_int16: np.ndarray) -> np.ndarray:
    """Convenience: PCM -> frames -> vectors, in one call."""
    return frames_to_vectors(compute_log_mel_frames(pcm_int16))
