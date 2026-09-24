"""WASAPI loopback capture.

Why audio is worth a whole subsystem: it does not go through the display
pipeline. The video path costs ~33 ms end-to-end on this rig and 97% of that is
structural -- compositor and Desktop Duplication quantising to the frame period,
none of it optimisable. Audio bypasses all of it. Samples arrive on the audio
clock regardless of what the GPU is doing, which also makes it immune to the
framerate collapse that cost us 3x capture throughput.

The community guidance for this game is explicitly audio-first: enemies emit a
distinct sound immediately before a blow, and players are told to react as that
sound peaks. If that holds, audio is not a supplement to the vision model, it is
the primary signal.

Architecture mirrors the video side: a callback thread fills a preallocated ring,
a consumer drains it. Unlike video, a missed audio block is a real loss -- there
is no "latest frame is good enough" -- so this is a sequential queue with an
overrun counter, not a latest-value register.
"""

from __future__ import annotations

import numpy as np

from ..clock import now_ns

_pa = None
try:
    import pyaudiowpatch as _pa
except ImportError:
    pass


class AudioRing:
    """Preallocated SPSC block queue. Producer is the audio callback."""

    __slots__ = ("_blocks", "_ts", "_cap", "_w", "_r", "overruns")

    def __init__(self, capacity: int, blocksize: int) -> None:
        self._cap = int(capacity)
        self._blocks = np.zeros((self._cap, blocksize), dtype=np.float32)
        self._ts = np.zeros(self._cap, dtype=np.int64)
        self._w = 0
        self._r = 0
        self.overruns = 0

    @property
    def pending(self) -> int:
        return self._w - self._r

    def write(self, samples: np.ndarray, t_ns: int) -> None:
        if self._w - self._r >= self._cap:
            self.overruns += 1
            self._r = self._w - self._cap + 1     # drop oldest
        i = self._w % self._cap
        self._blocks[i] = samples
        self._ts[i] = t_ns
        self._w += 1

    def read(self):
        """Next unread block as (samples, t_ns), or None."""
        if self._r >= self._w:
            return None
        i = self._r % self._cap
        out = (self._blocks[i], int(self._ts[i]))
        self._r += 1
        return out


class LoopbackCapture:
    """System-audio loopback on the default (or named) output device.

    Blocks are mono float32. Stereo is averaged down -- onset detection cares
    about the envelope, not the image.
    """

    def __init__(self, device_index: int | None = None, blocksize: int = 512,
                 ring_blocks: int = 256, samplerate: int | None = None) -> None:
        if _pa is None:
            raise RuntimeError("pip install pyaudiowpatch")
        self.blocksize = blocksize
        self._pa = None
        self._stream = None
        self.device_index = device_index
        self.samplerate = samplerate
        self.channels = 2
        self.device_name = "?"
        self.ring = AudioRing(ring_blocks, blocksize)
        self._t0 = 0

    @staticmethod
    def available() -> bool:
        return _pa is not None

    @staticmethod
    def list_devices():
        """All WASAPI loopback devices, default-output first."""
        if _pa is None:
            return []
        p = _pa.PyAudio()
        try:
            wasapi = p.get_host_api_info_by_type(_pa.paWASAPI)
            default = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
            out = []
            for lb in p.get_loopback_device_info_generator():
                out.append({
                    "index": lb["index"],
                    "name": lb["name"],
                    "channels": lb["maxInputChannels"],
                    "rate": int(lb["defaultSampleRate"]),
                    "is_default_output": default["name"] in lb["name"],
                })
            out.sort(key=lambda d: not d["is_default_output"])
            return out
        finally:
            p.terminate()

    def _resolve(self, p):
        if self.device_index is not None:
            return p.get_device_info_by_index(self.device_index)
        wasapi = p.get_host_api_info_by_type(_pa.paWASAPI)
        default = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        for lb in p.get_loopback_device_info_generator():
            if default["name"] in lb["name"]:
                return lb
        raise RuntimeError(
            "no loopback device matches the default output. Pass an explicit "
            "device index -- see `parry33 audio devices`.")

    def start(self) -> "LoopbackCapture":
        self._pa = _pa.PyAudio()
        dev = self._resolve(self._pa)
        self.device_index = dev["index"]
        self.device_name = dev["name"]
        self.channels = int(dev["maxInputChannels"])
        self.samplerate = self.samplerate or int(dev["defaultSampleRate"])
        ch = self.channels

        def _cb(in_data, frame_count, time_info, status):
            t = now_ns()
            a = np.frombuffer(in_data, dtype=np.float32)
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1)
            n = self.blocksize
            if a.size >= n:
                self.ring.write(a[:n], t)
            else:
                buf = np.zeros(n, dtype=np.float32)
                buf[: a.size] = a
                self.ring.write(buf, t)
            return (None, _pa.paContinue)

        self._stream = self._pa.open(
            format=_pa.paFloat32, channels=ch, rate=self.samplerate,
            input=True, input_device_index=self.device_index,
            frames_per_buffer=self.blocksize, stream_callback=_cb)
        self._t0 = now_ns()
        self._stream.start_stream()
        return self

    @property
    def block_ms(self) -> float:
        return self.blocksize / max(self.samplerate or 1, 1) * 1000.0

    def read(self):
        return self.ring.read()

    def drain(self):
        """Discard everything buffered. Call before a timed trial."""
        while self.ring.read() is not None:
            pass

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

    def close(self) -> None:
        self.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return (f"<LoopbackCapture {self.device_name!r} {self.samplerate} Hz "
                f"{self.channels}ch block={self.blocksize} "
                f"({self.block_ms:.2f} ms)>")
