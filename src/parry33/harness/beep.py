"""Audio responder: emits a sharp burst at a known instant.

The audio analogue of flash_window. To measure how fast we can notice a sound
the system is playing, we have to play one ourselves and know exactly when it
left for the device.

Timing detail that matters: a blocking write() returns when samples are *queued*,
not when they are played, so timing around it measures the queue, not the sound.
Instead the output runs a callback that emits silence until armed, and stamps the
clock at the moment it actually hands the burst to the device. Everything after
that -- device buffer, mixer, loopback capture, our detection -- is what we are
trying to measure, and all of it is downstream of that stamp.

A broadband burst is deliberate: white noise puts energy across the whole
spectrum, so spectral flux sees an unambiguous edge. A pure tone would be easy
to miss in a band-limited detector.

Length matters as much as content. A 4 ms click is shorter than one capture
block (10.67 ms at 512 frames), so depending on alignment it either lands inside
one block or splits across two -- and a split burst produced 8x less spectral
flux, dropping it to the threshold and causing roughly half of all trials to be
missed. 20 ms guarantees at least one block is fully inside the burst regardless
of phase, and it is closer to a real attack sound anyway.
"""

from __future__ import annotations

import numpy as np

from ..clock import now_ns

_pa = None
try:
    import pyaudiowpatch as _pa
except ImportError:
    pass


class BeepEmitter:
    def __init__(self, samplerate: int = 48000, blocksize: int = 256,
                 burst_ms: float = 20.0, amplitude: float = 0.25,
                 device_index: int | None = None, seed: int = 7) -> None:
        if _pa is None:
            raise RuntimeError("pip install pyaudiowpatch")
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.amplitude = amplitude
        self.device_index = device_index
        self.channels = 2
        self._pa = None
        self._stream = None

        n = max(1, int(samplerate * burst_ms / 1000.0))
        rng = np.random.default_rng(seed)
        burst = rng.standard_normal(n).astype(np.float32)
        # Hard attack, quick decay: a ramped onset would blur the edge we time.
        burst *= np.linspace(1.0, 0.0, n, dtype=np.float32) ** 0.5
        self._burst = (burst / np.abs(burst).max() * amplitude).astype(np.float32)

        self._armed = False
        self._pos = 0
        self.t_emit_ns = 0

    @staticmethod
    def available() -> bool:
        return _pa is not None

    def start(self) -> "BeepEmitter":
        self._pa = _pa.PyAudio()
        if self.device_index is None:
            wasapi = self._pa.get_host_api_info_by_type(_pa.paWASAPI)
            self.device_index = wasapi["defaultOutputDevice"]
        info = self._pa.get_device_info_by_index(self.device_index)
        # WASAPI shared mode demands an exact match with the endpoint's mix
        # format -- asking a 7.1 HDMI output for 2 channels fails outright with
        # "Invalid number of channels". Take whatever the device declares and
        # duplicate the burst across every channel so it is audible regardless
        # of the speaker configuration.
        self.channels = int(info["maxOutputChannels"]) or 1
        self.samplerate = int(info["defaultSampleRate"])
        self.device_name = info["name"]

        def _cb(in_data, frame_count, time_info, status):
            out = np.zeros((frame_count, self.channels), dtype=np.float32)
            if self._armed:
                if self._pos == 0:
                    self.t_emit_ns = now_ns()   # the instant it leaves for the device
                take = min(frame_count, self._burst.size - self._pos)
                seg = self._burst[self._pos: self._pos + take]
                out[:take] = seg[:, None]
                self._pos += take
                if self._pos >= self._burst.size:
                    self._armed = False
                    self._pos = 0
            return (out.tobytes(), _pa.paContinue)

        self._stream = self._pa.open(
            format=_pa.paFloat32, channels=self.channels, rate=self.samplerate,
            output=True, output_device_index=self.device_index,
            frames_per_buffer=self.blocksize, stream_callback=_cb)
        self._stream.start_stream()
        return self

    def fire(self) -> None:
        """Arm one burst. Returns immediately; read t_emit_ns after it lands."""
        self.t_emit_ns = 0
        self._pos = 0
        self._armed = True

    @property
    def fired(self) -> bool:
        return self.t_emit_ns != 0

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
