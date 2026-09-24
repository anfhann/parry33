"""Does our injected keypress actually reach the game?

The detector can be perfect and the bot still does nothing if SendInput is being
filtered. This presses the parry key slowly, with a countdown, so you can watch
the character and see whether it reacts at all.

Run it, tab into the game, and stand somewhere safe in combat. If the character
never responds, injection is blocked and no amount of detector work will help.

Likely cause on this machine: the game is the Xbox Game Pass build, which runs
in an AppContainer sandbox with a read-only install (see docs -- UE4SS could not
attach for the same reason). Windows restricts synthetic input into protected
processes from a standard-rights process.

If it is blocked, the things to try, in order:
  1. Run this elevated (right-click the terminal -> Run as administrator).
     UIPI blocks input from lower to higher integrity levels.
  2. Use a virtual gamepad instead of the keyboard: ViGEmBus presents a real
     HID device at the driver level rather than synthesising key events, and is
     usually not filtered the same way. `pip install vgamepad` plus the ViGEmBus
     driver, then set backend = "gamepad" in config.
  3. The Steam build of the game, which is not sandboxed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parry33 import config as cfgmod          # noqa: E402
from parry33.input import base as inbase      # noqa: E402
from parry33.live import foreground_title     # noqa: E402


def main(n: int = 8, gap_s: float = 2.5, delay: int = 10) -> int:
    cfg = cfgmod.load()
    be = inbase.build(cfg.input)
    key = cfg.input.binding("parry")
    tok = be.token("parry")
    print(f"injection test: will press {key} ({getattr(tok, 'label', tok)}) "
          f"{n} times, {gap_s:.1f}s apart")
    print(f"hold {cfg.input.hold_ms:.0f} ms per press, backend {be.name}\n")
    print(f"tab into the game NOW and watch your character.")
    for r in range(delay, 0, -1):
        print(f"  starting in {r}...", end="\r", flush=True)
        time.sleep(1.0)
    print(" " * 30)
    for i in range(n):
        be.tap("parry")
        print(f"  press {i+1}/{n}   foreground: {foreground_title()[:40]!r}")
        time.sleep(gap_s)
    be.close()
    print("\n  Did the character react at all?")
    print("    YES -> injection works; the problem is detection or timing.")
    print("    NO  -> injection is blocked. Try running this elevated, then a")
    print("           virtual gamepad (ViGEmBus). See the docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
