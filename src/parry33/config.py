"""TOML-backed configuration.

config/default.toml is the base; config/local.toml (gitignored) overlays it so
per-machine ROIs and keybinds never end up in a commit.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@dataclass(slots=True)
class CaptureConfig:
    backend: str = "dxgi"
    monitor: int = 0
    gpu: int = 0
    mode: str = "poll"
    color: str = "BGRA"
    target_fps: int = 0
    poll_sleep_us: float = 0.0
    region: tuple | None = None

    @property
    def size(self):
        if self.region is None:
            return None
        left, top, right, bottom = self.region
        return (right - left, bottom - top)


@dataclass(slots=True)
class InputConfig:
    backend: str = "sendinput"
    hold_ms: float = 40.0
    async_release: bool = True
    bindings: dict = field(default_factory=dict)

    def binding(self, action: str, backend=None) -> str:
        be = backend or self.backend
        table = self.bindings.get(be, {})
        if action not in table:
            raise KeyError(f"no binding for {action!r} under [input.bindings.{be}]")
        return table[action]


@dataclass(slots=True)
class TimingConfig:
    spin_margin_us: float = 400.0
    high_priority: bool = True
    time_critical: bool = True
    freeze_gc: bool = True


@dataclass(slots=True)
class AudioConfig:
    device: int = -1          # -1 = follow the default output device
    blocksize: int = 256
    detector: str = "flux"
    sensitivity: float = 3.0
    refractory_ms: float = 80.0
    floor_frac: float = 0.12
    grunt_band: tuple = (150, 2500)
    clash_band: tuple = (6000, 24000)
    grunt_sensitivity: float = 1.8
    clash_sensitivity: float = 3.0

    @property
    def device_index(self):
        """None means 'resolve the default output' -- 0 is a real device."""
        return None if self.device is None or self.device < 0 else self.device


@dataclass(slots=True)
class BenchConfig:
    trials: int = 200
    warmup: int = 30
    seconds: float = 10.0


@dataclass(slots=True)
class GameConfig:
    parry_window_ms: float = 150.0
    fps: int = 40
    variable_fps: bool = True


@dataclass(slots=True)
class Config:
    refresh_hz: int = 144
    game: GameConfig = field(default_factory=GameConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    input: InputConfig = field(default_factory=InputConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)

    @property
    def frame_period_ms(self) -> float:
        return 1000.0 / max(self.refresh_hz, 1)

    @property
    def capture_period_ms(self) -> float:
        """What we can actually sample at: min(game fps, refresh)."""
        return 1000.0 / max(min(self.game.fps or self.refresh_hz, self.refresh_hz), 1)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path=None) -> Config:
    data: dict = {}
    paths = [Path(path)] if path else [CONFIG_DIR / "default.toml", CONFIG_DIR / "local.toml"]
    for p in paths:
        if p.exists():
            data = _deep_merge(data, tomllib.loads(p.read_text(encoding="utf-8")))

    cap = dict(data.get("capture", {}))
    if cap.get("region") is not None:
        cap["region"] = tuple(cap["region"])

    inp = dict(data.get("input", {}))
    inp["bindings"] = inp.get("bindings", {})

    return Config(
        refresh_hz=data.get("display", {}).get("refresh_hz", 144),
        game=GameConfig(**data.get("game", {})),
        capture=CaptureConfig(**cap),
        input=InputConfig(**inp),
        timing=TimingConfig(**data.get("timing", {})),
        audio=AudioConfig(**{k: (tuple(v) if k.endswith("_band") else v)
                         for k, v in data.get("audio", {}).items()}),
        bench=BenchConfig(**data.get("bench", {})),
    )
