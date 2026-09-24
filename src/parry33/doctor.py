"""Environment preflight.

Checks the things that silently produce zero frames or zero input, which are the
two most expensive ways to lose an afternoon on this project.
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from ctypes import wintypes

from . import config as cfgmod

_u32 = ctypes.WinDLL("user32", use_last_error=True)
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_shell32 = ctypes.WinDLL("shell32", use_last_error=True)

OK, WARN, FAIL = "[ ok ]", "[warn]", "[FAIL]"


class DEVMODEW(ctypes.Structure):
    _fields_ = [
        ("dmDeviceName", wintypes.WCHAR * 32), ("dmSpecVersion", wintypes.WORD),
        ("dmDriverVersion", wintypes.WORD), ("dmSize", wintypes.WORD),
        ("dmDriverExtra", wintypes.WORD), ("dmFields", wintypes.DWORD),
        ("dmPositionX", ctypes.c_long), ("dmPositionY", ctypes.c_long),
        ("dmDisplayOrientation", wintypes.DWORD), ("dmDisplayFixedOutput", wintypes.DWORD),
        ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short),
        ("dmYResolution", ctypes.c_short), ("dmTTOption", ctypes.c_short),
        ("dmCollate", ctypes.c_short), ("dmFormName", wintypes.WCHAR * 32),
        ("dmLogPixels", wintypes.WORD), ("dmBitsPerPel", wintypes.DWORD),
        ("dmPelsWidth", wintypes.DWORD), ("dmPelsHeight", wintypes.DWORD),
        ("dmDisplayFlags", wintypes.DWORD), ("dmDisplayFrequency", wintypes.DWORD),
        ("dmICMMethod", wintypes.DWORD), ("dmICMIntent", wintypes.DWORD),
        ("dmMediaType", wintypes.DWORD), ("dmDitherType", wintypes.DWORD),
        ("dmReserved1", wintypes.DWORD), ("dmReserved2", wintypes.DWORD),
        ("dmPanningWidth", wintypes.DWORD), ("dmPanningHeight", wintypes.DWORD)]


def primary_display():
    _u32.EnumDisplaySettingsW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(DEVMODEW))
    _u32.EnumDisplaySettingsW.restype = wintypes.BOOL
    d = DEVMODEW()
    d.dmSize = ctypes.sizeof(DEVMODEW)
    if not _u32.EnumDisplaySettingsW(None, 0xFFFFFFFF, ctypes.byref(d)):
        return None
    return int(d.dmPelsWidth), int(d.dmPelsHeight), int(d.dmDisplayFrequency)


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", ctypes.c_long)]


class _PATH_SOURCE(ctypes.Structure):
    _fields_ = [("adapterId", LUID), ("id", wintypes.UINT),
                ("modeInfoIdx", wintypes.UINT), ("statusFlags", wintypes.UINT)]


class _RATIONAL(ctypes.Structure):
    _fields_ = [("Numerator", wintypes.UINT), ("Denominator", wintypes.UINT)]


class _PATH_TARGET(ctypes.Structure):
    _fields_ = [("adapterId", LUID), ("id", wintypes.UINT),
                ("modeInfoIdx", wintypes.UINT), ("outputTechnology", wintypes.UINT),
                ("rotation", wintypes.UINT), ("scaling", wintypes.UINT),
                ("refreshRate", _RATIONAL), ("scanLineOrdering", wintypes.UINT),
                ("targetAvailable", wintypes.BOOL), ("statusFlags", wintypes.UINT)]


class _PATH_INFO(ctypes.Structure):
    _fields_ = [("sourceInfo", _PATH_SOURCE), ("targetInfo", _PATH_TARGET),
                ("flags", wintypes.UINT)]


class _MODE_INFO(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_byte * 64)]   # opaque; we never read modes


class _DEVICE_INFO_HEADER(ctypes.Structure):
    _fields_ = [("type", wintypes.UINT), ("size", wintypes.UINT),
                ("adapterId", LUID), ("id", wintypes.UINT)]


class _ADVANCED_COLOR_INFO(ctypes.Structure):
    _fields_ = [("header", _DEVICE_INFO_HEADER), ("value", wintypes.UINT),
                ("colorEncoding", wintypes.UINT),
                ("bitsPerColorChannel", wintypes.UINT)]


_QDC_ONLY_ACTIVE_PATHS = 0x00000002
_GET_ADVANCED_COLOR_INFO = 9


def hdr_state():
    """Return (enabled, supported, bits_per_channel) for the active displays.

    HDR is the single most damaging display setting for Desktop Duplication.
    The legacy DuplicateOutput API hands back an 8-bit BGRA surface, so with HDR
    on Windows tone-maps the whole desktop down to SDR inside AcquireNextFrame,
    every frame. That collapses capture throughput -- observed here as a hard
    ~30 fps ceiling against a game rendering 75 -- and it is invisible unless
    you go looking for it.
    """
    try:
        n_path, n_mode = wintypes.UINT(), wintypes.UINT()
        if _u32.GetDisplayConfigBufferSizes(_QDC_ONLY_ACTIVE_PATHS,
                                            ctypes.byref(n_path),
                                            ctypes.byref(n_mode)) != 0:
            return None
        paths = (_PATH_INFO * n_path.value)()
        modes = (_MODE_INFO * n_mode.value)()
        if _u32.QueryDisplayConfig(_QDC_ONLY_ACTIVE_PATHS,
                                   ctypes.byref(n_path), paths,
                                   ctypes.byref(n_mode), modes, None) != 0:
            return None
        out = []
        for i in range(n_path.value):
            info = _ADVANCED_COLOR_INFO()
            info.header.type = _GET_ADVANCED_COLOR_INFO
            info.header.size = ctypes.sizeof(_ADVANCED_COLOR_INFO)
            info.header.adapterId = paths[i].targetInfo.adapterId
            info.header.id = paths[i].targetInfo.id
            if _u32.DisplayConfigGetDeviceInfo(ctypes.byref(info)) != 0:
                continue
            out.append((bool(info.value & 0x2), bool(info.value & 0x1),
                        int(info.bitsPerColorChannel)))
        return out
    except (AttributeError, OSError):
        return None


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def run(cfg=None) -> int:
    cfg = cfg or cfgmod.load()
    problems = 0
    print("parry33 doctor\n")

    v = sys.version_info
    print(f"{OK} python {v.major}.{v.minor}.{v.micro} ({'64' if sys.maxsize > 2**32 else '32'}-bit)")
    wv = sys.getwindowsversion()
    print(f"{OK} windows build {wv.build}")

    disp = primary_display()
    if disp:
        w, h, hz = disp
        period = 1000.0 / max(hz, 1)
        print(f"{OK} display {w}x{h} @ {hz} Hz -> {period:.2f} ms/frame")
        if hz != cfg.refresh_hz:
            problems += 1
            print(f"{WARN}   config says refresh_hz={cfg.refresh_hz}; "
                  f"update config/local.toml or the budget maths will be wrong")
        full_mb = w * h * 4 / 1e6
        print(f"       full-screen BGRA frame = {full_mb:.1f} MB "
              f"({full_mb * hz:.0f} MB/s at {hz} Hz) -- crop the ROI")
    else:
        problems += 1
        print(f"{FAIL} could not query display mode")

    hdr = hdr_state()
    if hdr:
        on = [b for e, s, b in hdr if e]
        if on:
            problems += 1
            print(f"{FAIL} HDR is ENABLED ({on[0]}-bit) -- this cripples capture")
            print("       The legacy Desktop Duplication API returns 8-bit BGRA, so")
            print("       Windows tone-maps the whole HDR desktop to SDR inside")
            print("       AcquireNextFrame every frame. Measured effect: ~30 fps")
            print("       ceiling against a game rendering 75.")
            print("       Turn it off: Settings > System > Display > Use HDR,")
            print("       or press Win+Alt+B. Turn it back on when you are done.")
        else:
            print(f"{OK} HDR off ({hdr[0][2]}-bit) -- correct for capture")

    from .util.prio import dpi_awareness, set_dpi_aware
    set_dpi_aware()
    aware = dpi_awareness()          # read the real state, not the setter result
    if aware == "per-monitor-aware":
        print(f"{OK} DPI awareness: {aware}")
    else:
        problems += 1
        print(f"{WARN} DPI awareness: {aware} -- on a scaled display Windows will "
              f"report virtualised coordinates and your ROI will land wrong")

    from .clock import PreciseTimer
    t = PreciseTimer()
    if t.supported:
        print(f"{OK} high-resolution waitable timer available (~0.5 ms)")
    else:
        problems += 1
        print(f"{WARN} no high-resolution timer; fell back to timeBeginPeriod(1)")
    t.close()

    elevated = bool(_shell32.IsUserAnAdmin())
    print(f"{OK if elevated else WARN} process elevation: "
          f"{'admin' if elevated else 'standard'}")
    if not elevated:
        print("       if the game runs elevated, UIPI silently drops SendInput from "
              "a standard-rights process. Run this elevated to match.")

    if _have("bettercam") or _have("dxcam"):
        from .capture.dxgi import DXGICapture
        impl = "bettercam" if _have("bettercam") else "dxcam"
        print(f"{OK} capture backend: {impl}")
        try:
            print("       " + DXGICapture.describe_devices().replace("\n", "\n       "))
        except Exception as e:  # noqa: BLE001 - diagnostics
            print(f"{WARN}   device enumeration failed: {e}")
    else:
        problems += 1
        print(f"{FAIL} no DXGI backend. pip install bettercam")

    if _have("vgamepad"):
        try:
            import vgamepad  # noqa: F401
            print(f"{OK} vgamepad importable (ViGEmBus driver present)")
        except Exception as e:  # noqa: BLE001
            problems += 1
            print(f"{FAIL} vgamepad installed but not usable: {e}")
            print("       install ViGEmBus: "
                  "https://github.com/nefarius/ViGEmBus/releases")
    else:
        print(f"{WARN} vgamepad missing -- gamepad injection unavailable "
              f"(sendinput still works)")

    print(f"\n{OK} config: capture={cfg.capture.backend}/{cfg.capture.mode} "
          f"region={cfg.capture.region} input={cfg.input.backend}")
    if cfg.capture.region is None:
        problems += 1
        print(f"{WARN}   region is null: you are capturing the whole screen. "
              f"Set one in config/local.toml.")
    else:
        w, h = cfg.capture.size
        mb = w * h * 4 / 1e6
        print(f"       roi {w}x{h} = {mb:.2f} MB/frame")
        print(f"       measured: ROI size has little effect on capture cost up to "
              f"1280x720 -- size it for the model, not the copy "
              f"(`bench capture --sweep` to confirm on this machine)")
    try:
        for a in ("parry", "dodge"):
            print(f"       binding {a}: {cfg.input.binding(a)}")
    except KeyError as e:
        problems += 1
        print(f"{FAIL} {e}")

    print(f"\n{'no blocking issues' if problems == 0 else f'{problems} item(s) need attention'}")
    print("\nreminders:")
    print("  * run the game borderless / windowed-fullscreen, not exclusive fullscreen")
    print("  * capture.gpu must be the adapter the game renders on")
    print("  * verify in-game keybinds match [input.bindings] before trusting a run")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
