"""Onset detection on a block stream.

Two detectors, deliberately different in character:

  RmsOnset       -- tracks the loudness envelope against an adaptive baseline.
                    Trivially cheap (~microseconds). Fires on anything that gets
                    suddenly louder, which in combat includes your own attacks,
                    music stings and footsteps. High recall, low precision.

  SpectralFlux   -- half-wave-rectified change in the magnitude spectrum. Fires
                    on a change in timbre, not just level, so it separates a
                    sharp whoosh from a swell in the music bed. ~50 us per block
                    at 512 points. This is the one to build on.

Both use an adaptive baseline rather than a fixed threshold, because game audio
levels drift with distance, music and mix. A fixed threshold tuned in one fight
will be wrong in the next.

The refractory period matters more than it looks: a single attack sound produces
several frames above threshold as it swells. Without a lockout you get five
"onsets" for one event and the downstream logic has to dedupe. 80 ms is roughly
half the parry window -- long enough to collapse one sound into one event, short
enough not to swallow a genuine second attack in a combo.
"""

from __future__ import annotations

import numpy as np


class _Base:
    """Common adaptive-threshold + refractory machinery."""

    def __init__(self, samplerate: int, blocksize: int, sensitivity: float = 3.0,
                 refractory_ms: float = 80.0, adapt: float = 0.02,
                 floor_frac: float = 0.12, peak_decay: float = 0.999) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.sensitivity = sensitivity
        self.adapt = adapt
        self.block_ms = blocksize / samplerate * 1000.0
        self.refractory_ns = int(refractory_ms * 1e6)
        self.floor_frac = floor_frac
        self.peak_decay = peak_decay
        self._mean = 0.0
        self._var = 0.0
        self._peak = 0.0
        self._n = 0
        self._lock_until_ns = 0
        self.last_value = 0.0
        self.last_threshold = 0.0

    def _score(self, block: np.ndarray) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        self._mean = self._var = self._peak = 0.0
        self._n = 0
        self._lock_until_ns = 0

    def push(self, block: np.ndarray, t_ns: int | None = None) -> bool:
        """Feed one block. True if this block starts an onset.

        The refractory period is wall-clock, not a block count. A block counter
        only advances while blocks are being fed, so a lockout set at the end of
        one burst survives any gap in feeding and silently eats the start of the
        next one -- which is exactly how this detector missed 64% of trials.
        """
        from ..clock import now_ns
        t = now_ns() if t_ns is None else t_ns
        v = self._score(block)
        self.last_value = v
        self._n += 1

        # Warm up on the first ~0.5 s before firing on anything.
        warm = self._n < max(8, int(500 / self.block_ms))
        std = self._var ** 0.5
        thr = self._mean + self.sensitivity * std

        # Absolute floor, scaled to the loudest thing seen recently. Against
        # digital silence the adaptive threshold collapses to zero and every
        # scrap of dither looks like a 100-sigma event; against a loud mix a
        # fixed floor would never trip. A fraction of the decaying peak is the
        # only form of this that survives both.
        floor = self._peak * self.floor_frac
        self.last_threshold = max(thr, floor)

        fired = False
        if t < self._lock_until_ns:
            pass
        elif not warm and v > thr and v > floor:
            fired = True
            self._lock_until_ns = t + self.refractory_ns

        # Only adapt on non-onset blocks: letting the transient into the baseline
        # raises the bar right when a combo's second hit needs to clear it.
        if not fired:
            d = v - self._mean
            self._mean += self.adapt * d
            self._var += self.adapt * (d * d - self._var)
        self._peak = max(self._peak * self.peak_decay, v)
        return fired


class RmsOnset(_Base):
    """Envelope detector. Cheap, noisy, useful as a sanity baseline."""

    name = "rms"

    def _score(self, block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(block * block) + 1e-12))


class SpectralFluxOnset(_Base):
    """Half-wave-rectified spectral flux. Fires on timbre change."""

    name = "flux"

    def __init__(self, samplerate: int, blocksize: int, sensitivity: float = 3.0,
                 refractory_ms: float = 80.0, adapt: float = 0.02,
                 floor_frac: float = 0.12, peak_decay: float = 0.999,
                 fmin: float = 200.0, fmax: float | None = None) -> None:
        super().__init__(samplerate, blocksize, sensitivity, refractory_ms,
                         adapt, floor_frac, peak_decay)
        self._window = np.hanning(blocksize).astype(np.float32)
        self._prev = np.zeros(blocksize // 2 + 1, dtype=np.float32)
        freqs = np.fft.rfftfreq(blocksize, 1.0 / samplerate)
        hi = fmax if fmax is not None else samplerate / 2
        # Band-limit: below ~200 Hz is music and ambience, the top octave is
        # mostly hiss. Attack transients sit in between.
        self._band = (freqs >= fmin) & (freqs <= hi)

    def reset(self) -> None:
        super().reset()
        self._prev[:] = 0.0

    def _score(self, block: np.ndarray) -> float:
        mag = np.abs(np.fft.rfft(block * self._window)).astype(np.float32)
        diff = mag - self._prev
        np.maximum(diff, 0.0, out=diff)          # half-wave rectify: onsets only
        self._prev = mag
        return float(diff[self._band].sum())


def build(kind: str, samplerate: int, blocksize: int, **kw) -> _Base:
    if kind == "rms":
        return RmsOnset(samplerate, blocksize, **kw)
    if kind == "flux":
        return SpectralFluxOnset(samplerate, blocksize, **kw)
    raise ValueError(f"unknown onset detector {kind!r}; use rms|flux")
