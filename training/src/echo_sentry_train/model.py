"""Candidate architectures — comprehensive-plan.md §4.2.

    #0 pca_baseline   — non-negotiable honesty check; ship the AE only if it beats this
    #1 dense_ae       — PRIMARY. The DCASE/MIMII baseline verbatim. Ship this.
    #2 conv_ae        — small 2D conv AE; explore offline, promote only on a real gain
    #3 idnn           — predicts the center frame from its neighbours; insurance if
                        non-stationary (cyclic/intermittent) equipment joins the pilot

Architecture #4 (5-way node-ID self-supervised classifier) is deliberately not
implemented here — comprehensive-plan.md §4.2 flags it as inconsistent across
individual machines in the literature, and it needs a joint multi-node training loop
that's a genuinely separate effort from the other three, which all share the same
"reconstruct/predict a vector, score by error" shape. Evaluate it only if #1-#3 all
underperform on real collected data.

Every architecture here scores a batch of feature vectors the same way callers expect:
`.fit(X_train)` then `.score(X) -> np.ndarray` of one non-negative anomaly score per
row of X (higher = more anomalous), so train.py/evaluate.py/quantize.py don't need to
know which one they're holding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow import keras

from echo_sentry_train import frontend_config as fc


class AnomalyScorer(Protocol):
    def fit(self, x_train: np.ndarray, **kwargs) -> None: ...
    def score(self, x: np.ndarray) -> np.ndarray: ...


# ---- #0: PCA baseline (non-negotiable honesty check) -----------------------------


@dataclass
class PcaBaseline:
    n_components: int = 8
    _pca: PCA | None = None

    def fit(self, x_train: np.ndarray, **_kwargs) -> None:
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(x_train)

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self._pca is not None, "call fit() first"
        reconstructed = self._pca.inverse_transform(self._pca.transform(x))
        return np.mean((x - reconstructed) ** 2, axis=1)


# ---- #1: Dense autoencoder (PRIMARY — DCASE/MIMII baseline) ----------------------


def build_dense_autoencoder(input_dim: int = fc.VECTOR_DIM, hidden: int = 128,
                             bottleneck: int = 8) -> keras.Model:
    """input -> 128x4 -> 8 -> 128x4 -> input, BatchNorm + ReLU throughout — the exact
    DCASE Task 2 / MIMII baseline shape (comprehensive-plan.md §4.2)."""
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for _ in range(4):
        x = keras.layers.Dense(hidden)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(bottleneck)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    for _ in range(4):
        x = keras.layers.Dense(hidden)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    outputs = keras.layers.Dense(input_dim)(x)
    model = keras.Model(inputs, outputs, name="dense_ae")
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


class DenseAutoencoderScorer:
    def __init__(self, input_dim: int = fc.VECTOR_DIM, hidden: int = 128, bottleneck: int = 8):
        self.model = build_dense_autoencoder(input_dim, hidden, bottleneck)

    def fit(self, x_train: np.ndarray, *, x_val: np.ndarray | None = None, epochs: int = 100,
            batch_size: int = 512, patience: int = 10) -> keras.callbacks.History:
        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss" if x_val is not None else "loss", patience=patience,
            restore_best_weights=True,
        )]
        return self.model.fit(
            x_train, x_train, validation_data=(x_val, x_val) if x_val is not None else None,
            epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0, callbacks=callbacks,
        )

    def score(self, x: np.ndarray) -> np.ndarray:
        reconstructed = self.model.predict(x, verbose=0)
        return np.mean((x - reconstructed) ** 2, axis=1)


# ---- #2: Convolutional autoencoder (explore offline) ------------------------------


def build_conv_autoencoder(frames_per_vector: int = fc.FRAMES_PER_VECTOR,
                            mel_channels: int = fc.MEL_CHANNELS) -> keras.Model:
    """Small 2D conv AE over the (frames_per_vector, mel_channels) patch each feature
    vector already is, before flattening — weight sharing across frequency, at the
    cost of more arena and more quantization care (comprehensive-plan.md §4.2)."""
    inputs = keras.Input(shape=(frames_per_vector, mel_channels, 1))
    x = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D((1, 2), padding="same")(x)
    x = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(x)
    encoded = keras.layers.MaxPooling2D((1, 2), padding="same")(x)

    x = keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(encoded)
    x = keras.layers.UpSampling2D((1, 2))(x)
    x = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.UpSampling2D((1, 2))(x)
    outputs = keras.layers.Conv2D(1, (3, 3), padding="same", activation="linear")(x)

    # Pooling can overshoot mel_channels by one pixel depending on divisibility;
    # crop back to the exact input shape rather than silently mismatching it.
    outputs = keras.layers.Cropping2D(((0, outputs.shape[1] - frames_per_vector),
                                        (0, outputs.shape[2] - mel_channels)))(outputs)

    model = keras.Model(inputs, outputs, name="conv_ae")
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


class ConvAutoencoderScorer:
    def __init__(self, frames_per_vector: int = fc.FRAMES_PER_VECTOR,
                 mel_channels: int = fc.MEL_CHANNELS):
        self.frames_per_vector = frames_per_vector
        self.mel_channels = mel_channels
        self.model = build_conv_autoencoder(frames_per_vector, mel_channels)

    def _reshape(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(-1, self.frames_per_vector, self.mel_channels, 1)

    def fit(self, x_train: np.ndarray, *, x_val: np.ndarray | None = None, epochs: int = 100,
            batch_size: int = 512, patience: int = 10) -> keras.callbacks.History:
        xt = self._reshape(x_train)
        xv = self._reshape(x_val) if x_val is not None else None
        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss" if xv is not None else "loss", patience=patience,
            restore_best_weights=True,
        )]
        return self.model.fit(
            xt, xt, validation_data=(xv, xv) if xv is not None else None,
            epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0, callbacks=callbacks,
        )

    def score(self, x: np.ndarray) -> np.ndarray:
        xr = self._reshape(x)
        reconstructed = self.model.predict(xr, verbose=0)
        return np.mean((xr - reconstructed) ** 2, axis=(1, 2, 3))


# ---- #3: IDNN — predict the center frame from its neighbours ----------------------


def _center_frame_index(frames_per_vector: int = fc.FRAMES_PER_VECTOR) -> int:
    return frames_per_vector // 2


def split_context_target(x: np.ndarray, *, frames_per_vector: int = fc.FRAMES_PER_VECTOR,
                          mel_channels: int = fc.MEL_CHANNELS) -> tuple[np.ndarray, np.ndarray]:
    """(N, VECTOR_DIM) -> context (N, (frames_per_vector-1)*mel_channels), target (N, mel)."""
    frames = x.reshape(-1, frames_per_vector, mel_channels)
    center = _center_frame_index(frames_per_vector)
    context = np.concatenate([frames[:, :center], frames[:, center + 1 :]], axis=1)
    target = frames[:, center]
    return context.reshape(len(x), -1), target


def build_idnn(frames_per_vector: int = fc.FRAMES_PER_VECTOR,
               mel_channels: int = fc.MEL_CHANNELS, hidden: int = 64) -> keras.Model:
    context_dim = (frames_per_vector - 1) * mel_channels
    inputs = keras.Input(shape=(context_dim,))
    x = keras.layers.Dense(hidden, activation="relu")(inputs)
    x = keras.layers.Dense(hidden, activation="relu")(x)
    outputs = keras.layers.Dense(mel_channels)(x)
    model = keras.Model(inputs, outputs, name="idnn")
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


class IdnnScorer:
    def __init__(self, frames_per_vector: int = fc.FRAMES_PER_VECTOR,
                 mel_channels: int = fc.MEL_CHANNELS, hidden: int = 64):
        self.frames_per_vector = frames_per_vector
        self.mel_channels = mel_channels
        self.model = build_idnn(frames_per_vector, mel_channels, hidden)

    def fit(self, x_train: np.ndarray, *, x_val: np.ndarray | None = None, epochs: int = 100,
            batch_size: int = 512, patience: int = 10) -> keras.callbacks.History:
        ctx_t, tgt_t = split_context_target(x_train, frames_per_vector=self.frames_per_vector,
                                             mel_channels=self.mel_channels)
        val_data = None
        if x_val is not None:
            ctx_v, tgt_v = split_context_target(x_val, frames_per_vector=self.frames_per_vector,
                                                 mel_channels=self.mel_channels)
            val_data = (ctx_v, tgt_v)
        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_data is not None else "loss", patience=patience,
            restore_best_weights=True,
        )]
        return self.model.fit(
            ctx_t, tgt_t, validation_data=val_data, epochs=epochs, batch_size=batch_size,
            shuffle=True, verbose=0, callbacks=callbacks,
        )

    def score(self, x: np.ndarray) -> np.ndarray:
        ctx, tgt = split_context_target(x, frames_per_vector=self.frames_per_vector,
                                         mel_channels=self.mel_channels)
        predicted = self.model.predict(ctx, verbose=0)
        return np.mean((tgt - predicted) ** 2, axis=1)


def set_seed(seed: int = 42) -> None:
    tf.keras.utils.set_random_seed(seed)


# name -> constructor, shared by train.py/calibrate.py/evaluate.py/quantize.py/export.py.
# "pca" is intentionally excluded: it's the honesty-check control (evaluate.py only),
# never a candidate for on-device export.
ARCHS: dict[str, type] = {
    "dense_ae": DenseAutoencoderScorer,
    "conv_ae": ConvAutoencoderScorer,
    "idnn": IdnnScorer,
}


def load_trained_scorer(model_dir):
    """Reloads a saved Keras model into the right scorer wrapper, using the arch name
    recorded in train.py's train_metadata.json — so callers get back an object whose
    .score(x) means the same thing it did right after training, regardless of which
    architecture it is."""
    import json
    from pathlib import Path

    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "train_metadata.json").read_text())
    arch = metadata["arch"]
    if arch not in ARCHS:
        raise ValueError(f"arch {arch!r} has no on-device scorer (pca is eval-only)")

    keras_model = keras.models.load_model(model_dir / "model.keras")
    scorer = ARCHS[arch].__new__(ARCHS[arch])
    scorer.model = keras_model
    if arch in ("conv_ae", "idnn"):
        scorer.frames_per_vector = fc.FRAMES_PER_VECTOR
        scorer.mel_channels = fc.MEL_CHANNELS
    return scorer
