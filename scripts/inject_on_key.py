"""Manual injection test, in combat.

Press '+' and it injects a chosen key. Lets you verify injection works during a
battle specifically -- the earlier overworld test (E opening an NPC dialogue)
proved the key reaches the game, but combat may handle input differently, and
that is where it matters.

    python scripts/inject_on_key.py            # '+' injects F (basic attack)
    python scripts/inject_on_key.py parry      # '+' injects E (parry)
    python scripts/inject_on_key.py parry 150  # ...holding it for 150 ms

A game that POLLS key state each tick can miss a 40 ms synthetic press entirely,
while the same press works fine for a menu that reads Windows messages. If a
longer hold fixes it, that was the cause.

F10 stops. Injected input is ignored by the hook, so the tool never mistakes its
own keystroke for yours.
"""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parry33 import config as cfgmod            # noqa: E402
from parry33.input import base as inbase        # noqa: E402
from parry33.input.keywatch import KeyWatcher   # noqa: E402
from parry33.live import foreground_title       # noqa: E402

_u32 = ctypes.WinDLL("user32", use_last_error=True)
VK_F10 = 0x79


def main(action: str = "attack", hold_ms: float | None = None) -> int:
    cfg = cfgmod.load()
    if hold_ms:
        cfg.input.hold_ms = hold_ms
    be = inbase.build(cfg.input)
    try:
        tok = be.token(action)
    except KeyError:
        print(f"no binding for {action!r}; try: attack, parry, dodge")
        return 1
    label = getattr(tok, "label", str(tok))
    print(f"press '+'  ->  inject {action} ({label})")
    print(f"hold {cfg.input.hold_ms:.0f} ms | F10 to stop")
    print("tab into the game now\n")

    n = 0
    with KeyWatcher(watch=("plus", "numplus")) as keys:
        while True:
            for t_key, name, down in keys.drain():
                if down:
                    be.tap(action)
                    n += 1
                    print(f"  injected {action} #{n}   "
                          f"foreground={foreground_title()[:34]!r}")
            if _u32.GetAsyncKeyState(VK_F10) & 0x8000:
                break
            time.sleep(0.005)
    be.close()
    print(f"\n  {n} injections sent.")
    print("  Did the character act each time? If yes, injection works in combat")
    print("  and the problem is detection/timing, not the input path.")
    return 0


if __name__ == "__main__":
    act = sys.argv[1] if len(sys.argv) > 1 else "attack"
    hold = float(sys.argv[2]) if len(sys.argv) > 2 else None
    raise SystemExit(main(act, hold))
