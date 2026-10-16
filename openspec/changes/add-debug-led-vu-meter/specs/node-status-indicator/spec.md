# Spec Delta

## Purpose

What a node communicates locally, on the device itself, about its own audio input and
capture-path health — without depending on a network, a hub, or host tooling. This is
the diagnostic channel available during hardware bring-up and in a field equipment room
where nothing is attached to the node.

## ADDED Requirements

### Requirement: Audio level is indicated on a logarithmic scale

The node SHALL derive its level indication from the signal's RMS level expressed in
dBFS, normalized between a configurable floor and ceiling, and SHALL NOT map linear
sample amplitude directly to indicator intensity. Microphone output spanning an
equipment room's ambient range occupies a small fraction of linear full scale, so a
linear mapping leaves the indicator visually dark across the entire range of interest.

#### Scenario: Typical equipment-room ambient reads mid-scale
- **WHEN** the microphone observes a steady level in the middle of the configured
  floor-to-ceiling dBFS window
- **THEN** the indicator shows a clearly intermediate intensity, distinguishable by eye
  from both its minimum-on and maximum intensities

#### Scenario: Level at or below the configured floor
- **WHEN** the measured level is at or below the configured floor in dBFS
- **THEN** the indicator shows its minimum-on intensity and does not go dark

#### Scenario: Level at or above the configured ceiling
- **WHEN** the measured level is at or above the configured ceiling in dBFS
- **THEN** the indicator shows its maximum configured intensity and does not wrap,
  overflow, or dim

### Requirement: Any present signal produces a visible indication

The node SHALL render every level above the configured floor as a non-zero, visually
perceptible indication. Perceived brightness is not linear in drive level, so the
mapping SHALL be perceptually corrected; correction alone is insufficient at the bottom
of the range, where available intensity resolution would otherwise round a real signal
down to fully off.

#### Scenario: Signal just above the floor remains visible
- **WHEN** the measured level is above the configured floor by a small margin, such as
  a live microphone sitting at its own self-noise level in a quiet room
- **THEN** the indicator is visibly lit, not fully dark

#### Scenario: Intensity increases with level
- **WHEN** the measured level rises monotonically from the floor to the ceiling
- **THEN** the rendered intensity increases monotonically and never decreases

### Requirement: Absence of signal is distinguishable from a quiet room

The node SHALL render a complete absence of signal differently from any present signal,
so that a disconnected, unpowered, or mis-configured microphone is distinguishable from
a working microphone in a silent room. A microphone that produces no useful data still
yields a normally advancing sample stream, so the sample values are the only available
discriminator.

#### Scenario: Microphone produces constant samples
- **WHEN** every sample in the measurement window is identically zero
- **THEN** the indicator is fully dark

#### Scenario: Live microphone in a silent room
- **WHEN** the microphone is working and the room is as quiet as it gets, so the
  measured level is the microphone's own noise floor
- **THEN** the indicator is lit at or near its minimum-on intensity, and is therefore
  distinguishable from the fully dark state

### Requirement: Level indication uses averaging time response

The node SHALL smooth its level indication over time with a fast rise and a slower
decay, on the order of a few hundred milliseconds to decay, so that the indication
tracks sustained level rather than instantaneous sample values. The monitored equipment
is quasi-stationary rotating machinery, for which an averaging indication is the
appropriate instrument.

#### Scenario: Steady machinery produces a steady indication
- **WHEN** the microphone observes a steady, continuous machine noise
- **THEN** the indicator holds a steady intensity without visible flicker or jitter

#### Scenario: Sudden onset is reflected promptly
- **WHEN** the level rises abruptly from quiet to loud
- **THEN** the indicator reaches the corresponding higher intensity within a small
  fraction of a second

#### Scenario: Sudden silence decays rather than cutting out
- **WHEN** a sustained loud sound stops abruptly
- **THEN** the indicator decays smoothly to the new level over a few hundred
  milliseconds rather than dropping instantly

### Requirement: Clipping is indicated distinctly from loud audio

The node SHALL indicate that samples have reached or approached digital full scale using
an indication distinct from its highest normal level indication, and SHALL hold that
indication long enough to be seen by an observer who is not watching continuously.
Clipping corrupts the feature frontend's input, and an averaging level indication cannot
reveal it.

#### Scenario: Samples reach full scale
- **WHEN** any sample in a measurement window reaches or exceeds the configured
  near-full-scale limit
- **THEN** the indicator shows its clipping indication, visually distinct from the
  normal maximum-level indication, and holds it long enough to be observed

#### Scenario: Loud but unclipped audio
- **WHEN** the measured level is at or above the configured ceiling but no sample
  reaches the near-full-scale limit
- **THEN** the indicator shows its normal maximum-level indication and not the clipping
  indication

### Requirement: Capture-path faults are indicated distinctly from audio level

The node SHALL indicate a stalled sample stream and a rising count of dropped samples,
each distinguishable from every level indication and from each other. These faults make
the level indication meaningless, so they SHALL take precedence over it rather than
being rendered as a level.

#### Scenario: Sample stream stops advancing
- **WHEN** no new samples have been captured for longer than the configured stall
  timeout
- **THEN** the indicator shows its stall indication instead of a level indication

#### Scenario: Dropped-sample count is rising
- **WHEN** the cumulative dropped-sample count increases between consecutive
  measurement windows
- **THEN** the indicator shows its drop indication, distinguishable from both the stall
  indication and every level indication

#### Scenario: Startup before the first samples arrive
- **WHEN** the node has just booted and no samples have been captured yet
- **THEN** the indicator does not show a stall or drop indication

#### Scenario: Capture path is healthy
- **WHEN** samples are advancing and the dropped-sample count is unchanged
- **THEN** the indicator shows a level indication

### Requirement: The indicator does not disturb audio capture

The indicator SHALL NOT alter the captured sample stream, SHALL NOT increase the
dropped-sample count, and SHALL NOT measurably raise the measured noise floor. It exists
to diagnose the capture path, so it is worthless if it perturbs what it measures.

#### Scenario: Drop count is unaffected
- **WHEN** the node runs for an extended period with the indicator active
- **THEN** the cumulative dropped-sample count is no higher than for an otherwise
  identical run without the indicator

#### Scenario: Noise floor is unaffected by indicator intensity
- **WHEN** the measured noise floor in a quiet room is compared between the indicator
  held dark and the indicator held at its maximum configured intensity
- **THEN** the two measurements agree within normal measurement variation

#### Scenario: Reader falling behind does not corrupt the indication
- **WHEN** the indicator's reader falls far enough behind the capture write position
  that the data it would read has been overwritten
- **THEN** the indicator resynchronizes to current data rather than indicating a level
  derived from a torn read

### Requirement: The indicator is available in every firmware build

The node SHALL provide the indicator in both the production build and the raw-audio
capture build. The capture build brings up no network by design, which makes the
indicator its only live diagnostic channel.

#### Scenario: Capture build with no network
- **WHEN** the node runs a capture build with no hub and no host console attached
- **THEN** the indicator reflects audio level and capture-path health

#### Scenario: Production build
- **WHEN** the node runs a production build performing on-device inference
- **THEN** the indicator reflects audio level and capture-path health, and inference,
  telemetry, and detector behavior are unchanged

### Requirement: The level mapping is calibratable without code changes

The node SHALL expose its floor level, ceiling level, maximum intensity, and indicator
output pin as build-time configuration. Datasheet-derived defaults require verification
against real hardware in a real room, and the indicator's output pin differs between
board revisions.

#### Scenario: Narrowing the level window after calibration
- **WHEN** an operator measures the actual ambient level and sets a different floor and
  ceiling in dBFS
- **THEN** the indicator's mapping uses the new window with no source changes

#### Scenario: Board revision with a different indicator pin
- **WHEN** the node is built for a board revision whose indicator is on a different pin
- **THEN** the pin is selected by configuration with no source changes
