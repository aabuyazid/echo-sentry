# Echo Sentry — Comprehensive Implementation Plan

## Context

The development environment is finished and verified (ESP-IDF v5.5.5 on Python 3.12, TensorFlow
2.21 with a working `audio_microfrontend_op`, two independent `uv` projects, mosquitto, clangd).
The repo currently contains **scaffolding only** — `pyproject.toml`s, `uv.lock`s,
`config/frontend.yaml`, `.clangd`, and empty package directories. No application code exists yet.

This plan covers building the actual system: firmware, hub, training, and the data collection that
has to happen before any of the modelling is meaningful. It is grounded in the DCASE Task 2 /
MIMII anomalous-sound-detection literature, which is the closest well-studied analogue to this
problem, and it is shaped by three decisions made during planning:

| Decision | Consequence for this plan |
|---|---|
| **Continuous rotating machinery** (HVAC, pumps, fans, compressors) | Quasi-stationary audio. Dense autoencoder over log-mel is the right primary model. Slow-drift detection (CUSUM) matters more than fast transient detection. Baseline statistics are stable, so robust calibration works well. |
| **Cannot seed faults on the equipment** | There is no way to measure an in-situ AUC. Validation splits into *benchmark* correctness (public corpora) and *operational* trust (false-alarm rate). The system must also be built to **accumulate its own labeled anomalies over time** — see §3.4. |
| **Raw PCM is acceptable; speech is not a concern** | Nodes can keep a rolling audio buffer and upload an **evidence clip** whenever a score excursion occurs. This is the single highest-leverage design choice in the plan. It is still consistent with "no raw audio leaves the equipment room" — the Pi *is* in the room, and audio never crosses the `eth0` boundary to Teams or Power BI. |

---

## 1. Literature review — what we are borrowing, and from whom

The problem has a name in the research community: **unsupervised anomalous sound detection (ASD)
for machine condition monitoring**. It has had a dedicated DCASE challenge task since 2020, which
means there are published baselines, reference implementations, and eight years of failure modes
already catalogued. We should not invent anything we can copy.

### 1.1 The canonical baseline (MIMII → DCASE Task 2)

Hitachi's MIMII dataset paper established the recipe every subsequent system is measured against,
and the DCASE baseline has carried it forward essentially unchanged through the 2025 edition.
Reference implementation parameters, read directly from the source:

| Parameter | MIMII baseline | DCASE 2023–2025 baseline |
|---|---|---|
| `n_mels` | 64 | 128 |
| `n_fft` | 1024 (64 ms @ 16 kHz) | 1024 |
| `hop_length` | 512 (32 ms) | 512 |
| `power` | 2.0 | 2.0 |
| frames concatenated | 5 | 5 |
| input dim | 64 × 5 = **320** | 128 × 5 = 640 |
| architecture | dense AE | `input → 128×4 → 8 → 128×4 → input`, BatchNorm + ReLU throughout |
| loss / optimizer | MSE / Adam | MSE / Adam |
| epochs, batch | 50, 512 | similar |
| anomaly score | mean reconstruction MSE over all vectors in a clip | same (plus a selective-Mahalanobis mode) |

Two things to take from this. First, **the model is astonishingly small** — an 8-dimensional
bottleneck, five-frame context. Nobody needed a large network. Second, **the score is an average
over many short windows**, not one inference over a long window. That is a better fit for a
microcontroller than it looks, and it gives you a *distribution* of per-window errors per clip
rather than a single number, which the hub can exploit.

### 1.2 What performance to expect

MIMII's published benchmark, averaged across machine models, at +6 dB SNR:

| Machine type | AUC |
|---|---|
| Fan | ~0.94 |
| Pump | ~0.85 |
| Slide rail | ~0.78 |
| **Valve** | **~0.67** |

Valve performance collapses at −6 dB SNR to ~0.53 — barely above chance. The lesson is that
autoencoders model *stationary* sound well and *impulsive, cyclic* sound badly. Your equipment is
continuous rotating machinery, which is the favourable case — but it also means the published
0.90-ish numbers are the realistic ceiling, not 0.99, and the plan should not promise otherwise.

SNR dominates everything. In a shared equipment room with five machines running, each node hears
its neighbours. This is the primary justification for the cross-node common-mode rejection in §5.

### 1.3 Architectures beyond the baseline

The DCASE 2025 task summary (119 submissions, 35 teams) concluded that fine-tuning pre-trained
models, using frozen pre-trained models, and training small models from scratch can all be
competitive — the differentiators were "appropriate cost functions, anomaly score normalization,
and use of clean machine and noise sounds." **Score normalization is called out explicitly as one
of the three things that matters.** That is a hub-side concern, and §5 is built around it.

Self-supervised classification (train a network to identify *which machine* a clip came from; score
by how badly it fails) generally beats autoencoders on DCASE leaderboards. But the literature is
candid about the catch: MobileNetV2-style machine-ID classification achieved an average mAUC of
**59.73% versus 64.46% for IDNN**, an autoencoder-family method — it "easily fails on some
individual machines and lacks performance consistency." For a five-node pilot where one embarrassing
node ruins the demo, consistency beats peak AUC.

### 1.4 Quantization

Published INT8 post-training-quantization results for anomaly-detection autoencoders show
**AUROC degradation below 0.12%** with 51.7–74.9% size reduction, and 1.2–1.8× speedup on ARM.
Dense autoencoders quantize especially cleanly (74.9% reduction, the largest in that study). This
de-risks the int8 requirement substantially — but it must still be *measured*, not assumed (§4.4).

### 1.5 Score post-processing

The time-series anomaly-detection literature is unambiguous that smoothing raw anomaly scores
"improves performance in most settings and crucially never harms performance," with EMA and sliding
median/mean/max as the standard operators. One specific finding maps directly onto our problem: a
**median filter as a preprocessor followed by an EWMA control chart as the detector** significantly
improves detection rate while reducing false-alarm rate. That two-stage shape is exactly what §5
specifies, and the reason is intuitive — the median kills the dropped-wrench impulse, and the EWMA
then tracks genuine sustained change without that impulse smeared across its entire time constant.

### 1.6 Where the computation should live

Kalør et al., *Remote Anomaly Detection in Industry 4.0 Using Resource-Constrained Devices*, studied
precisely our architectural question — anomaly detection on acoustic/vibration signals over a
constrained wireless link — and found the coding/transmission trade-off depends on channel SNR.
Our situation resolves it trivially: the nodes are metres from the Pi's own access point with
ample bandwidth, so we are not forced into on-device-only inference by the link. **We choose
on-device inference anyway**, because it bounds the Pi's workload at five nodes' worth of JSON
rather than five continuous audio streams, and because it means a Wi-Fi outage degrades alerting
rather than detection.

### 1.7 Sources

- [DCASE 2025 Task 2 description (arXiv:2506.10097)](https://arxiv.org/abs/2506.10097)
- [DCASE 2023 Task 2 / first-shot domain-generalization baseline (arXiv:2303.00455)](https://arxiv.org/pdf/2303.00455)
- [MIMII dataset paper](https://www.alphaxiv.org/overview/1909.09347v1) · [MIMII baseline code](https://github.com/MIMII-hitachi/mimii_baseline/)
- [DCASE 2023 AE baseline implementation (nttcslab)](https://github.com/nttcslab/dcase2023_task2_baseline_ae)
- [Outlier-aware Inlier Modeling / MobileNetV2 vs IDNN comparison (arXiv:2309.07500)](https://arxiv.org/pdf/2309.07500)
- [Remote Anomaly Detection in Industry 4.0 Using Resource-Constrained Devices (arXiv:2110.05757)](https://arxiv.org/abs/2110.05757)
- [Unsupervised Anomaly Detection in Multivariate Time Series — score smoothing (arXiv:2503.23060)](https://arxiv.org/pdf/2503.23060)
- [Data-Driven Anomaly Detection for Time-Series Streaming Data — median + EWMA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7582627/)
- [Memory-Efficient CNN Autoencoder for TinyML Edge Devices — INT8 AUROC impact](https://doi.org/10.3390/fi18060286)
- [AWS: anomaly detection on industrial equipment using audio](https://aws.amazon.com/blogs/machine-learning/performing-anomaly-detection-on-industrial-equipment-using-audio-signals/)

---

## 2. Software architecture

### 2.1 System view

```
┌─ EQUIPMENT ROOM ────────────────────────────────────────────┐
│                                                              │
│  ESP32-S3 node ×5                        Raspberry Pi hub    │
│  ┌────────────────────────┐              ┌────────────────┐  │
│  │ I2S ─► ring (PSRAM)    │              │ mosquitto      │  │
│  │   │         │          │   MQTT       │   │            │  │
│  │   ▼         ▼          │  ─────────►  │   ▼            │  │
│  │ frontend  evidence     │   scores     │ ingest         │  │
│  │   │       buffer 30 s  │              │   ▼            │  │
│  │   ▼         │          │   HTTP       │ store (SQLite) │  │
│  │ TFLM AE     └──────────┼─ ─ ─ ─ ─►    │   ▼            │  │
│  │   │           .wav     │  evidence    │ detector  ◄── §5│  │
│  │   ▼                    │              │   ▼            │  │
│  │ score ──────────────────              │ alerting       │  │
│  └────────────────────────┘              │   ▼            │  │
│                                          │ sinks + outbox │  │
│  ▲ raw audio never crosses this boundary ─┴──────┬─────────┘  │
└───────────────────────────────────────────────────┼──────────┘
                                                    │ eth0, outbound HTTPS only
                                      Teams (Power Automate) · Power BI (push)
```

The room boundary is the privacy boundary. PCM moves freely *inside* it (node → Pi) and never
crosses it. Teams and Power BI receive scores, timestamps, and alert metadata only.

### 2.2 The three contracts

Everything else is implementation detail; these three interfaces are what must not drift.

**(a) `config/frontend.yaml` → generated code.** One file defines every signal-processing
parameter. `tools/gen_config.py` emits `firmware/main/gen/frontend_config.h` and
`training/src/echo_sentry_train/frontend_config.py`. Neither generated file is ever hand-edited,
and `gen_config.py` writes a hash of the YAML into both, which firmware reports in its telemetry
and training stamps into the model metadata. A mismatch is then detectable at runtime instead of
appearing as mysteriously bad scores.

**(b) MQTT topic and payload schema.**

| Topic | QoS | Retain | Cadence | Purpose |
|---|---|---|---|---|
| `echo-sentry/v1/<node>/score` | 1 | no | 10 s | The anomaly score and its supporting statistics |
| `echo-sentry/v1/<node>/health` | 0 | yes | 60 s | Heap, RSSI, uptime, I2S drop counter, clock skew |
| `echo-sentry/v1/<node>/status` | 1 | yes | on change | `online` / LWT `offline` |
| `echo-sentry/v1/<node>/evidence` | 1 | no | on excursion | Announces a clip uploaded over HTTP |
| `echo-sentry/v1/cmd/<node>` | 1 | no | hub → node | `set_threshold`, `capture`, `ota`, `reboot` |

Score payload (~220 bytes JSON; CBOR is a later optimization, not needed at 0.5 msg/s):

```json
{ "v":1, "node":"es-03", "t":1789435200, "seq":40213,
  "err_mean":0.0412, "err_p95":0.0981, "err_max":0.1533,
  "rms_dbfs":-38.2, "bands":[...16 int8 octave-band energies...],
  "drops":0, "model":"ae-2026w37-a1b2c3", "cfg":"f7e9d2", "fw":"1.2.0" }
```

`seq` is a monotonic counter — it lets the hub detect gaps that timestamps alone would hide.
`err_p95` and `err_max` come free from the per-clip distribution of window errors (§1.1) and give
the hub far more to work with than a mean alone. `bands` is a coarse 16-value spectral summary for
dashboard diagnostics; it is not invertible to audio.

**(c) The model artifact.** `model.tflite` plus a sidecar `model.json` carrying the config hash,
input/output quantization parameters, training corpus identifier, and the fp32↔int8 AUC delta
measured at export. Firmware refuses to run a model whose config hash does not match its own.

### 2.3 Tier 1 — Firmware (`firmware/`)

Single ESP-IDF project, one binary, with capture behaviour gated by a Kconfig option rather than a
separate application. This is deliberate: **the capture path must be byte-identical to the
production path**, or your training data comes from a subtly different sensor than your inference
does. Sharing `audio_capture.c` between modes guarantees that by construction.

FreeRTOS task structure:

| Task | Core | Prio | Responsibility |
|---|---|---|---|
| `i2s_capture` | 1 | 10 | Blocking `i2s_channel_read()` into DMA buffers; shift 24→16 bit; push to ring. **Never blocks on anything downstream** — on queue-full it drops and increments a counter. |
| `frontend` | 1 | 7 | Microfrontend log-mel per 64 ms window; emits frames to a frame ring |
| `inference` | 0 | 5 | Assembles 5-frame vectors, `interpreter->Invoke()`, accumulates per-clip error statistics |
| `telemetry` | 0 | 3 | Builds and publishes the score payload; handles `cmd/` topics |
| `net` | 0 | 2 | Wi-Fi + MQTT supervision, exponential-backoff reconnect, SNTP, OTA check |
| `evidence` | 0 | 4 | On trigger, snapshots the PCM ring and HTTP-POSTs a WAV to the Pi |

Source files under `firmware/main/`:

```
audio_capture.c   I2S std driver, ring buffer, drop accounting
frontend.c        microfrontend wrapper, reads gen/frontend_config.h
inference.cc      TFLM setup, arena, MicroMutableOpResolver, per-clip scoring
detector.c        local fast-path threshold (NVS-configurable) → triggers evidence
evidence.c        PCM ring snapshot + HTTP multipart upload
net.c             wifi, sntp, mqtt client, LWT, backoff
telemetry.c       payload assembly, cmd handling
provisioning.c    NVS: node_id, ssid/psk, threshold, hub address
ota.c             esp_https_ota against the Pi
app_main.c        init order, task creation, watchdog
gen/frontend_config.h   GENERATED — do not edit
model_data.cc           GENERATED by EMBED_FILES from model.tflite
```

Memory budget (ESP32-S3-DevKitC-1-**N8R8** — the 8 MB PSRAM variant is required for the evidence
buffer):

| Allocation | Size | Where |
|---|---|---|
| I2S DMA buffers (4 × 1024 frames × 4 B) | 16 KB | internal |
| Evidence PCM ring, 30 s @ 16 kHz 16-bit | 960 KB | **PSRAM** |
| Frame ring (62 frames × 64 bands × 2 B) | 8 KB | internal |
| TFLM tensor arena | ~32 KB | internal (`MALLOC_CAP_INTERNAL` — arena in PSRAM is far slower) |
| Model flatbuffer (~185 KB int8) | in flash | XIP, not RAM |
| Wi-Fi + MQTT + TLS stacks | ~60 KB | internal |

Compute budget, to be **measured at M2, not assumed**: 320→128×4→8→128×4→320 is ~182 k MACs per
vector. At 29 vectors per 2 s clip (stride-2 subsampling, see §4.1) that is ~5.3 M MACs per clip.
ESP-NN int8 kernels should land this in roughly 150–300 ms per 2 s clip — a 10–15% duty cycle.
If measurement comes in worse, the knobs in order of preference are: increase vector stride to 3,
drop hidden width 128→64, then lengthen the clip interval.

### 2.4 Tier 2 — Hub (`hub/src/echo_sentry_hub/`)

```
__main__.py    asyncio entrypoint, wiring, graceful shutdown
config.py      pydantic-settings; env + /etc/echo-sentry/config.toml
models.py      pydantic schemas for every MQTT payload (rejects malformed nodes loudly)
ingest.py      paho-mqtt v2 client thread → asyncio.Queue
store.py       SQLite (WAL), schema + migrations, retention
detector.py    ★ §5 — calibration, smoothing, hysteresis, CUSUM, common-mode rejection
alerting.py    dedup, cooldown, escalation, auto-resolve, feedback ingestion
evidence.py    aiohttp endpoint receiving WAV uploads; writes to /var/lib/echo-sentry/audio
forwarder.py   outbox drain loop with tenacity backoff
sinks/
  base.py      Protocol: async def send(event) -> None
  teams.py     Power Automate Workflow HTTP trigger, Adaptive Card
  powerbi.py   streaming push dataset REST
  console.py   dev sink
```

Concurrency: one asyncio event loop. `paho-mqtt` runs its own network thread via `loop_start()`
and hands messages across with `loop.call_soon_threadsafe()`. **paho-mqtt 2.x requires
`CallbackAPIVersion.VERSION2` in the `Client` constructor** — v1-style examples will raise
immediately. SQLite calls go through `asyncio.to_thread` to keep the loop responsive; at 0.5 msg/s
× 5 nodes this is comfortable with enormous headroom.

SQLite schema (the system of record):

```sql
CREATE TABLE score (
  id INTEGER PRIMARY KEY, node TEXT NOT NULL, ts INTEGER NOT NULL, seq INTEGER NOT NULL,
  err_mean REAL, err_p95 REAL, err_max REAL, rms_dbfs REAL,
  z REAL, z_corrected REAL, z_ewma REAL, cusum REAL,   -- detector outputs, written inline
  model TEXT, cfg TEXT, fw TEXT, raw TEXT,
  UNIQUE(node, seq));
CREATE INDEX idx_score_node_ts ON score(node, ts);

CREATE TABLE detector_state (
  node TEXT PRIMARY KEY, baseline_median REAL, baseline_mad REAL, baseline_updated INTEGER,
  ewma REAL, cusum REAL, state TEXT, state_entered INTEGER, median_buf TEXT);

CREATE TABLE alert (
  id INTEGER PRIMARY KEY, node TEXT, condition TEXT, severity TEXT,
  opened_ts INTEGER, closed_ts INTEGER, peak_z REAL,
  evidence_path TEXT, verdict TEXT, verdict_by TEXT, verdict_ts INTEGER);

CREATE TABLE outbox (                                   -- store-and-forward
  id INTEGER PRIMARY KEY, sink TEXT, payload TEXT, created_ts INTEGER,
  attempts INTEGER DEFAULT 0, last_error TEXT, sent_ts INTEGER);

CREATE TABLE evidence (
  id INTEGER PRIMARY KEY, node TEXT, ts INTEGER, path TEXT, bytes INTEGER,
  trigger TEXT, alert_id INTEGER, label TEXT);        -- label filled by §3.4 flywheel
```

Writing the detector's intermediate values (`z`, `z_ewma`, `cusum`) inline into the `score` row is
a deliberate choice: when you retune thresholds later you want to replay history and see exactly
what the detector saw, without recomputing it.

### 2.5 Tier 3 — Training (`training/src/echo_sentry_train/`)

```
frontend_config.py  GENERATED from config/frontend.yaml
features.py         microfrontend wrapper — the ONLY feature path used for training
dataset.py          manifest loading, time-based splits, tf.data pipelines
clean.py            §3.3 cleaning + manifest generation
model.py            architectures: dense_ae, idnn, conv_ae, pca_baseline
train.py            training loop, per-node model registry
calibrate.py        derives per-node baseline median/MAD and suggested thresholds
quantize.py         int8 PTQ with representative dataset
evaluate.py         AUC/pAUC on MIMII + DCASE; fp32 vs int8 delta; score distributions
export.py           model.tflite + model.json sidecar
```

And `tools/`:

```
gen_config.py       frontend.yaml → C header + Python module (+ config hash)
capture_server.py   TCP/HTTP receiver for node PCM → timestamped WAV + sidecar JSON
fake_node.py        synthetic MQTT nodes with configurable drift, spikes, dropout
replay.py           replay historical SQLite scores through detector.py for threshold tuning
label.py            CLI to review evidence clips and set evidence.label
```

`tools/replay.py` deserves emphasis. Because every score and every detector intermediate is
persisted, you can re-run months of history through a modified detector in seconds and count how
many alerts a threshold change would have produced. Threshold tuning without it is guesswork.

---

## 3. Data collection

This is the work that determines whether anything else succeeds, and it has to happen before
meaningful training can start.

### 3.1 Capture firmware

Enabled by `CONFIG_ECHO_SENTRY_CAPTURE_MODE`. Behaviour:

- Same `audio_capture.c` as production — identical I2S clock config, identical 24→16-bit shift,
  identical DMA sizing. Non-negotiable.
- Streams raw 16 kHz / 16-bit / mono PCM to `tools/capture_server.py` over a persistent TCP
  socket. That is 256 kbit/s per node — five nodes is 1.3 Mbit/s, trivial for the Pi's AP.
- Each 60-second segment is closed and reopened so a dropout costs one minute, not the session.
- Every segment carries a sidecar JSON: `node_id`, firmware version, config hash, start/end UTC
  (SNTP-synced), cumulative I2S drop count, and free-text `session_note` set at start.
- **The I2S drop counter is the integrity signal.** Any segment with a non-zero delta is quarantined
  in cleaning. Without this you cannot distinguish a genuine acoustic discontinuity from a buffer
  overrun, and you will train on glitches.

Capture can also run *alongside* production inference later — the evidence buffer (§3.4) is the
same mechanism at a lower duty cycle.

### 3.2 Campaign protocol

| Aspect | Requirement | Why |
|---|---|---|
| Duration | **14 continuous days** per node, 7 absolute minimum | Must cover the full weekly duty cycle. A model trained on Tuesday alarms every Saturday. |
| Coverage | All shifts, weekday + weekend, HVAC seasonal state noted | These are the dominant sources of legitimate score variation. |
| Mic placement | Fixed mount, documented distance/orientation/height, photographed | Moving a mic 30 cm invalidates the baseline. Photograph it so it can be restored after maintenance. |
| Machine states | Log start-up, steady, shutdown, and each load level | The model needs to see every normal state or each one becomes an anomaly. |
| Gain | INMP441 has fixed gain — just verify no clipping at the loudest state | Clipping is unrecoverable and silently destroys the high-frequency content you care about. |
| Simultaneity | **All five nodes record at once** | Required to characterize cross-talk and to calibrate the common-mode rejection in §5. |

**Operator logbook.** A `logbook.csv` maintained by whoever is on site — `timestamp_utc, node_or_all,
event, notes`. Entries like `2026-09-15T14:32Z, all, forklift passed`, `2026-09-16T09:00Z, es-03,
maintenance on pump`. This costs almost nothing and is the difference between a cleanable dataset
and an uncleanable one. Every unexplained score excursion you later investigate will be resolved by
this file or not at all.

### 3.3 Cleaning pipeline (`training/src/echo_sentry_train/clean.py`)

Runs over the raw WAV corpus and emits a **manifest CSV** — `path, node, start_utc, duration,
machine_state, reject_reason, split`. The manifest is the dataset; the WAVs are just storage.

Applied in order:

1. **Integrity** — non-zero I2S drop delta, truncated file, or duration mismatch → reject.
2. **Clipping** — >0.01% of samples at ≥0.99 full scale → reject.
3. **DC offset** — INMP441 has a measurable DC bias; high-pass at 20 Hz. The frontend's 50 Hz lower
   band limit already handles this for features, but the raw corpus should be clean for reuse.
4. **Dead mic / machine-off** — RMS below floor, or near-zero variance. **Label these, don't just
   drop them.** "Machine off" is a legitimate state you will need to recognize so the detector can
   suppress alerts rather than flag silence as anomalous.
5. **Impulse rejection** — per-window crest factor and kurtosis. Windows far above the node's
   baseline contain impacts (doors, dropped tools, forklifts). Exclude from the **training** set —
   the autoencoder must model steady state — but **retain them in a held-out `nuisance` set**.
   That set is how you tune §5's rejection logic; throwing it away is a mistake you cannot undo
   without re-collecting.
6. **Logbook join** — drop windows the operator marked as maintenance or abnormal.
7. **Split by time, never at random.** First 60% train / next 20% validation / last 20% test,
   chronologically, per node. Adjacent frames in a continuous recording are near-identical; a
   random split leaks the test set into training and produces a model that looks excellent and
   generalizes to nothing. This is the most common serious error in this problem domain.
8. **Domain-shift holdout** — reserve one or two complete days from a *different week* as a
   generalization set, mirroring DCASE's source/target domain structure. If AUC collapses here,
   the model has memorized a day rather than learned the machine.

### 3.4 Solving the "no anomalies" problem — the evidence flywheel

You cannot seed faults, so the pilot cannot report an in-situ AUC on day one. The plan is to make
the system **manufacture its own labeled dataset** while it runs:

1. Node keeps a 30-second rolling PCM ring in PSRAM at all times.
2. Any excursion above the local fast-path threshold — set *low*, deliberately, so it fires far more
   often than an alert does — snapshots the ring (15 s before, 15 s after) and POSTs it to the Pi.
3. The Pi stores the clip and links it to the score row and any alert.
4. `tools/label.py` presents unlabeled clips for review: listen, read the logbook, assign
   `normal / nuisance / suspicious / confirmed-fault`.
5. Labels feed three things: the detector's baseline (confirmed-normal periods extend it),
   the nuisance set (tunes rejection), and eventually a real in-situ ROC curve.

Add to this the retrospective path: because **100% of scores are retained forever** in SQLite,
the first time a monitored machine genuinely degrades or fails, you can go back and ask whether the
score rose beforehand and by how much. For a pilot that cannot seed faults, that single retrospective
is the most persuasive validation available — and it only exists if the retention and the evidence
capture are built in from the start rather than added after the failure.

---

## 4. Training

### 4.1 Feature frontend — changes to `config/frontend.yaml`

The current values were carried over from `micro_speech` keyword-spotting defaults. Machine
condition monitoring wants different ones, and the literature is specific about what they should be.

| Key | Current | Proposed | Rationale |
|---|---|---|---|
| `window_size_ms` | 30 | **64** | 1024 samples @ 16 kHz — matches MIMII/DCASE `n_fft=1024`. Longer window = better frequency resolution, which is what identifies tonal bearing/imbalance signatures. Stationary sound doesn't need speech-rate time resolution. |
| `window_stride_ms` | 20 | **32** | 512 samples — matches `hop_length=512`, 50% overlap. |
| `mel_channels` | 40 | **64** | MIMII baseline. Machine spectra are broader than speech; 40 bands is a speech-derived number. |
| `frames_per_inference` | 49 | *replaced* | See below. |
| **`frames_per_vector`** | — | **5** | New. Model input = 5 × 64 = **320 dims**, exactly the MIMII baseline. |
| **`clip_frames`** | — | **62** | New. ~2 s of context → 58 possible vectors per clip. |
| **`vector_stride`** | — | **2** | New. Score every 2nd vector → 29 inferences/clip. Halves compute at negligible statistical cost. |
| **`enable_noise_reduction`** | — | **false** | ⚠️ Critical. |
| **`enable_pcan`** | — | **false** | ⚠️ Critical. |

**The two `false` flags are the most important lines in the file.** The TFLM microfrontend is not
a plain log-mel extractor — it also includes spectral-subtraction noise reduction and PCAN
auto-gain, both designed to *normalize away stationary background* so a wake word stands out. For
machine condition monitoring the stationary background **is the signal**. Leaving these enabled
would systematically suppress exactly the degradation you are trying to detect, and the failure
would be silent: the model trains fine and simply never fires. Both must be explicitly disabled in
the C config and matched in the Python `audio_microfrontend_op` call.

### 4.2 Architectures to explore

Build in this order. Each is a small change to `model.py` against an unchanged pipeline.

| # | Model | Shape | Why explore it | Effort |
|---|---|---|---|---|
| 0 | **PCA / Mahalanobis** | linear, no NN | Non-negotiable control. If a 320→8 PCA reconstruction scores within 0.02 AUC of the autoencoder, the neural net has not earned its complexity and you should say so. Costs an afternoon. | Trivial |
| 1 | **Dense AE** ← *primary* | 320→128×4→8→128×4→320, BN+ReLU | The DCASE/MIMII baseline verbatim. Proven on exactly this problem, ~185 KB int8, quantizes cleanly, best-understood failure modes. **Ship this.** | Low |
| 2 | **Conv AE (2-D)** | small CNN over 64×5 mel patches | Weight sharing across frequency; usually a modest gain on stationary machinery. More arena, more quantization care. | Medium |
| 3 | **IDNN** | predict centre frame from the 4 surrounding | Beats dense AE on non-stationary machines (mAUC 64.46% vs MobileNetV2's 59.73% in the cited comparison). Insurance if cyclic equipment joins the pilot later. | Medium |
| 4 | **5-way node-ID classifier** | small CNN, score = −log p(correct node) | Self-supervised outlier-exposure family; often tops DCASE leaderboards. Feasible here because five nodes = five classes. **But** the literature is explicit that it "easily fails on some individual machines and lacks performance consistency" — a real risk in a five-node demo. Evaluate offline; do not ship first. | Medium-high |
| 5 | **Pre-trained embeddings + Mahalanobis** | frozen audio encoder, hub-side scoring | Competitive per DCASE 2025. Too large for the ESP32-S3, but viable as a *hub-side second opinion* on evidence clips only. Stretch goal. | High |

Decision rule: ship #1, keep #0 as an honesty check, evaluate #2–#4 offline on the collected data,
and only promote one if it beats #1 by >0.03 AUC *on the domain-shift holdout* — not on the test
split, where overfitting to a single deployment is easy.

### 4.3 Training protocol

- **Normal data only.** Unsupervised; the model never sees an anomaly.
- **Per-node models.** Five machines with different acoustics should not share one autoencoder —
  the DCASE "section" concept exists for exactly this reason. Five models, one registry, selected
  by `node_id` at flash time. They are ~185 KB each; there is no reason to economize.
- Adam, MSE, batch 512, ~50–100 epochs with early stopping on validation reconstruction error.
- Normalization statistics (per-mel-band mean/std) computed on the **training split only** and
  frozen into `frontend_config`. Computing them over the full dataset leaks test statistics.
- Log everything to `model.json`: corpus manifest hash, config hash, git SHA, split boundaries.

### 4.4 Quantization and export

Full-integer PTQ, `inference_input_type` and `inference_output_type` both `int8`, ops constrained
to `TFLITE_BUILTINS_INT8` — anything falling back to float will not run under TFLM. The
representative dataset must be drawn from the **training split**, ~500 vectors sampled across the
full range of normal operating states, not from one contiguous hour.

Embed via ESP-IDF's `EMBED_FILES` in `main/CMakeLists.txt`, not `xxd -i`.

Use `MicroMutableOpResolver` with only the ops the model actually needs (`FullyConnected`,
`Relu`, and whatever BatchNorm folds into) rather than `AllOpsResolver` — it saves tens of KB.

### 4.5 Evaluation — how the pilot proves it works without anomalies

Three independent tiers, because no single one is sufficient here:

**Tier A — pipeline correctness (public benchmarks).** Run the *entire* pipeline — features,
training, quantization, scoring — on MIMII and the DCASE 2023–2025 development sets. Report
AUC and pAUC (p=0.1) per machine type and SNR against published baselines.

| Gate | Target |
|---|---|
| MIMII fan @ +6 dB | AUC ≥ 0.92 (published ~0.94) |
| MIMII pump @ +6 dB | AUC ≥ 0.83 (published ~0.85) |
| Within published baseline | ≤ 0.02 AUC on every machine type |
| **fp32 → int8 delta** | **≤ 0.01 AUC** (literature suggests <0.12% is achievable) |

This is the load-bearing evidence. If the pipeline reproduces published numbers, the implementation
is correct, and the remaining uncertainty is about your specific machines rather than about the
software.

**Tier B — relative sensitivity (synthetic injection).** Mix MIMII anomalous clips into your own
recorded room audio at an SNR sweep (+6, 0, −6 dB) and plot ROC. This will not predict the true
detection rate, but it does answer "how far above the room noise floor must a fault be before we
see it," which is the question stakeholders actually ask.

**Tier C — operational trust (in situ).** These are the numbers that decide whether the pilot is
renewed:

| Metric | Target |
|---|---|
| False alerts per node per week | **< 1** |
| Day-over-day stability of the per-node score median | CV < 10% |
| Node uptime / score delivery completeness | > 99% |
| Score history retained | 100%, indefinitely |
| Evidence clips captured and labeled | every WATCH-or-above transition |

Be explicit with stakeholders that Tier C contains no detection-rate number and cannot until a
real fault occurs or fault seeding becomes possible. Stating that up front is much better than
being asked for it later.

---

## 5. Determining anomalous sound in the hub

This is `detector.py`, and it is where the pilot is won or lost. The model produces a noisy,
right-skewed scalar; the detector turns it into a decision somebody will be paged about. DCASE 2025
named **anomaly score normalization** as one of three differentiators among 119 submissions — the
processing below is that normalization, plus the temporal logic the challenge format doesn't cover.

Six stages, per node, applied in order.

### Stage 1 — Robust baseline calibration

Over a rolling 14-day window of scores from confirmed-normal periods:

```
median  = median(scores)
mad     = median(|scores − median|)
z       = 0.6745 × (score − median) / mad
```

**Use median and MAD, not mean and standard deviation.** Reconstruction-error distributions are
right-skewed, and any contamination in the baseline window inflates the standard deviation, which
*desensitizes* the detector exactly when it should be getting more sensitive. MAD has a 50%
breakdown point; a quarter of your baseline can be garbage and the estimate still holds. The 0.6745
factor rescales MAD to be comparable to a Gaussian σ, so `z` reads like a familiar z-score.

Recalibrate nightly, excluding any window inside an open alert or labeled non-normal.

### Stage 2 — Rolling median filter

Window of 5 samples (≈50 s at a 10 s cadence). Removes isolated single-clip impulses — a dropped
tool, a door, a forklift — before they reach the smoother. Cost: ~20 s of added latency, which is
irrelevant for degradation that develops over days.

### Stage 3 — Cross-node common-mode rejection

```
z_corrected = z_node − median(z over all reporting nodes)     # requires ≥3 nodes online
```

Five microphones in one room hear the same environment. If every node spikes simultaneously, the
cause is almost certainly environmental — HVAC cycling, a compressor elsewhere, a vehicle — and not
five machines failing at once. Subtracting the cross-node median is a differential measurement: it
cancels what is common and preserves what is local. Given §1.2's finding that ASD performance
collapses with falling SNR, and that each node's dominant noise source is its neighbours, this is
likely the highest-value single stage in the detector for a five-node room.

Fall back to raw `z` when fewer than three nodes are reporting, and record which mode was used.

### Stage 4 — EWMA smoothing

```
ewma ← α × z_corrected + (1 − α) × ewma        α = 0.02  (τ ≈ 50 samples ≈ 8 min)
```

Median-then-EWMA is the specific two-stage arrangement the literature reports as significantly
improving detection rate while reducing false alarms. The ordering matters: the median removes the
impulse *before* the EWMA would otherwise smear it across its whole time constant.

### Stage 5 — Hysteresis state machine with N-of-M persistence

A single threshold on a noisy signal produces flapping, and flapping destroys trust faster than
missed detections do.

| Transition | Condition |
|---|---|
| `NORMAL → WATCH` | `ewma ≥ 3.0` |
| `WATCH → ALARM` | `ewma ≥ 6.0` **and** 5 of the last 10 samples ≥ 6.0 |
| `ALARM → WATCH` | `ewma < 4.0` |
| `WATCH → NORMAL` | `ewma < 2.5` sustained 30 min |

Note the asymmetry: entry at 6.0, exit at 4.0. Separate entry and exit thresholds are what prevent
oscillation at the boundary. The N-of-M requirement is the false-alarm killer — it demands
persistence, which genuine mechanical degradation has and acoustic accidents do not.

`WATCH` does not page anyone. It logs, it triggers evidence capture, and it appears on the
dashboard. Only `ALARM` reaches Teams.

### Stage 6 — CUSUM for slow degradation

EWMA detects steps. Bearing wear is a *ramp* over weeks that may never produce a step large enough
to trip any instantaneous threshold, yet it is precisely what "predictive maintenance" promises.

```
S⁺ ← max(0, S⁺ + z_corrected − k)     k = 0.5     alarm at h = 5.0
```

CUSUM accumulates small persistent positive deviations. `k` is the slack — drift below 0.5σ is
ignored as noise — and `h` is the decision interval. This raises a separate, lower-urgency condition
(`degradation_trend`) that routes to a weekly digest rather than an immediate page. It is the
stage most likely to produce the pilot's headline result.

### Also required

- **Dead-man detection.** No score for >3 intervals → `node_offline`, corroborated by the MQTT
  last-will on `status`.
- **Machine-off suppression.** If `rms_dbfs` is below the node's learned "machine off" floor,
  suppress scoring entirely. A stopped machine is not an anomalous machine, and the autoencoder
  will happily report that silence is unlike anything it was trained on.
- **Model/config guard.** A score whose `cfg` hash disagrees with the hub's is quarantined, not
  scored.

### Tuning procedure

Do not guess the constants. After the first two weeks of live scores, run
`tools/replay.py` over the SQLite history with a grid over (α, median window, thresholds, N-of-M,
k, h) and pick the setting that yields **zero alerts on the known-normal period** with the largest
margin to the threshold. Then verify against the `nuisance` set from §3.3 — those recordings exist
for exactly this purpose. Re-run after any threshold change and diff the alert counts before
deploying.

---

## 6. Alerting

- **Teams via Power Automate Workflow HTTP trigger**, Adaptive Card. Classic O365 incoming-webhook
  connectors completed retirement earlier this year and are no longer an option.
- Card contents: node, machine, condition, severity, peak z, a sparkline of the last 24 h, the
  evidence clip reference, and **two buttons — `Confirm` and `False alarm`**. The verdict writes
  back to `alert.verdict` and feeds §3.4's flywheel. An alerting system with no feedback path
  cannot improve, and the operators who dismiss the alerts are the only people who know why.
- Cooldown per `(node, condition)`: 4 h. Auto-resolve with a closing card when the state machine
  returns to `NORMAL`.
- `degradation_trend` (CUSUM) → weekly digest, not a page.
- **Store-and-forward.** Every outbound message is an `outbox` row first; `forwarder.py` drains it
  with `tenacity` backoff. Ethernet egress will drop, and an alert lost to a transient network
  failure is worse than no system at all.
- Power BI streaming push dataset is a **view, not storage** (~10 k rows/hr, 75-column cap).
  SQLite remains the system of record; the dashboard pushes a downsampled 1-minute aggregate.

---

## 7. Milestones

| # | Milestone | Exit criterion |
|---|---|---|
| **M0** | Contracts | `gen_config.py` emits both headers from updated `frontend.yaml`; hashes match; committed |
| **M1** | Audio in | Capture firmware streams clean 16 kHz WAV to `capture_server.py`; **zero I2S drops over 1 h**; spectrogram of a known tone is correct |
| **M2** | Frontend parity ⚠️ | **Golden-vector test passes** — see §8. Gates everything downstream. |
| **M3** | Pipeline correctness | Tier A benchmarks met on MIMII/DCASE; int8 delta ≤ 0.01 AUC |
| **M4** | Collection campaign | 14 days × 5 nodes captured, cleaned, manifest built, splits frozen |
| **M5** | On-device inference | Per-node models trained and flashed; score published every 10 s; inference time measured and within budget |
| **M6** | Hub end-to-end | Ingest → SQLite → detector → Teams card, driven by `fake_node.py`; deployed to the Pi |
| **M7** | Tuning + pilot | Thresholds tuned via `replay.py`; two weeks live at <1 false alert/node/week |

M1–M2 can start immediately with one node on the bench. M4 is the long pole — start the campaign
as early as M2 passes, and do M3 in parallel while audio accumulates.

---

## 8. Verification

**The golden-vector test (M2) is the single most important check in this plan.**

Train/serve feature skew is the dominant failure mode in TinyML audio: if the C frontend and the
Python frontend differ in any parameter, the model trains beautifully and produces garbage on
hardware, with no error message and nothing obviously wrong to look at.

```
1. Generate a deterministic 10 s test signal (swept sine + white noise), save as WAV.
2. Python:   audio_microfrontend_op on that WAV with config/frontend.yaml params
             → golden.npy, committed to the repo.
3. Firmware: a `capture` command over MQTT plays the same WAV from flash through
             frontend.c and dumps the resulting frames over UART.
4. Compare:  assert max |c_frame − py_frame| ≤ 1 LSB across every frame and band.
```

Anything above 1 LSB means a parameter mismatch — check `enable_noise_reduction` and `enable_pcan`
first, since those are silent and asymmetric by default. Run this in CI on every `frontend.yaml`
change; it is cheap and it will save days.

Remaining checks:

| Tier | Check | Expected |
|---|---|---|
| Firmware | 1 h capture, drop counter | 0 |
| Firmware | Known 1 kHz tone through the full chain | Correct mel bin, correct level, no clipping |
| Firmware | `heap_caps_get_free_size`, arena high-water after 24 h | Stable, no leak |
| Firmware | Inference time per clip (`esp_timer`) | Within §2.3 budget; if not, apply the knobs in order |
| Firmware | Pull node power for 60 s | Reconnects with backoff; hub sees LWT then recovery; `seq` gap logged |
| Training | Tier A benchmark gates | §4.5 table |
| Training | PCA control vs dense AE | AE must beat PCA by >0.02 AUC or reconsider |
| Training | Shuffle labels and retrain | AUC → ~0.5. Catches leakage from a bad split. |
| Hub | `fake_node.py` with injected spike / ramp / dropout | Spike rejected by median; ramp caught by CUSUM; dropout raises `node_offline` |
| Hub | `fake_node.py` spiking all 5 nodes at once | **No alert** — common-mode rejection working |
| Hub | Kill network during an alert | Row stays in `outbox`, delivered on restore, no duplicate |
| Hub | Replay 14 days of known-normal history | Zero alerts |
| End-to-end | One real node → Pi AP → SNTP → score → tap near mic → evidence clip → Teams card | Full chain within the cooldown window |

---

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Train/serve feature skew | High if unguarded | M2 golden-vector test in CI; single config source |
| Microfrontend noise-reduction/PCAN left on | **High — silent failure** | Explicit `false` in YAML; asserted by the golden-vector test |
| Not enough normal-state diversity in 14 days | Medium | Operator logbook; extend campaign if a state is unobserved; the nuisance set is retained |
| Cross-talk between the five machines | High | Common-mode rejection (§5 Stage 3); characterized during simultaneous capture |
| No in-situ detection-rate number | **Certain** | Stated up front; Tier A/B/C framing; evidence flywheel builds toward one |
| Inference budget exceeded on device | Medium | Measured at M2 with three pre-agreed fallback knobs |
| Alert fatigue kills the pilot | Medium | Hysteresis + N-of-M + cooldown + WATCH-doesn't-page + feedback buttons |
| Model drifts with season/load | Medium | Rolling 14-day baseline recalibration; CUSUM reset on confirmed-normal |

---

## Execution order

1. ~~Write this document to `comprehensive-plan.md` at the repo root.~~ Done.
2. Update `config/frontend.yaml` per §4.1 (the two `false` flags especially).
3. Write `tools/gen_config.py` and generate both config artifacts — M0.
4. Then M1 onward, per the milestone table in §7.
