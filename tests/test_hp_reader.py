"""Segmentation and OCR tests for the HP-readout reader.

This is the project's only non-circular ground truth: it says when an attack
landed without asking whether anyone reacted to it. It also took several wrong
turns to get right, each of which produced plausible-looking but wrong output
rather than an error, so the failure modes are pinned here directly.

Geometry is taken from real captures at 2560x1440: current-HP glyphs ~31 px tall
and 14 px wide, the slash and the "/ max" after it ~21 px, digits 1-3 px apart,
and a ~10 px gap before the slash. The digit band sits above a segment bar which
moves with the readout's zoom animation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

find_hits = pytest.importorskip("find_hits")


def _frame(spans, height=90, bar=None):
    """spans: (x0, width, glyph_height) -> (mask, gray) as the reader sees them.

    Glyphs are placed in a band with clear rows above and below, mimicking a
    real frame; `bar` optionally adds the wide segment bar underneath.
    """
    gray = np.zeros((height, 280), dtype=np.uint8)
    top = 30
    for i, (x0, w, h) in enumerate(spans):
        y = top + (40 - h) // 2
        gray[y:y + h, x0:x0 + w] = 255
        # Real digits have internal structure. A solid block normalises to the
        # zero vector (every pixel equals the mean), which correctly matches
        # nothing -- so a fixture of solid blocks tests nothing. Punch a hole
        # whose position varies per glyph to make them distinguishable.
        if h >= 12 and w >= 8:
            hy = y + 2 + i * 3            # unique per glyph, so none collide
            gray[hy:min(hy + 4, y + h - 2), x0 + 1 + (i % 2):x0 + w - 2] = 0
    if bar is not None:
        gray[bar:bar + 12, 10:250] = 255
    return gray > find_hits.BRIGHT, gray


def test_reads_current_hp_and_ignores_max():
    """Three tall digits, then the slash and a smaller max, which must be cut."""
    m, g = _frame([(115, 14, 31), (130, 14, 32), (149, 14, 31),   # "543"
                   (171, 8, 22),                                   # "/"
                   (187, 8, 21), (199, 8, 21), (210, 10, 20)])     # "1132"
    assert len(find_hits.glyphs(m, g)) == 3


def test_leading_noise_speck_does_not_abort_the_scan():
    """A 1px-tall speck left of the digits.

    An earlier version stopped at the first short glyph in order to find the
    slash, so a single speck of bright scenery at the left edge ended the scan
    and the frame read as empty. Real captures contain these constantly.
    """
    m, g = _frame([(3, 7, 1), (48, 4, 1),
                   (115, 14, 31), (130, 14, 32), (149, 14, 31),
                   (171, 8, 22), (187, 8, 21)])
    assert len(find_hits.glyphs(m, g)) == 3


def test_four_digit_value():
    m, g = _frame([(112, 14, 31), (128, 14, 31), (144, 14, 31), (160, 14, 31),
                   (182, 8, 22), (198, 8, 21)])
    assert len(find_hits.glyphs(m, g)) == 4


def test_segment_bar_below_is_excluded():
    """The bar under the number must not be read, or merged with the digits.

    The bar moves with the zoom: digits at rows 47-85 with the bar at 96-110
    when zoomed, 49-80 with the bar at 86-99 when not. A fixed crop that cleared
    it in one state included it in the other, and a bar touching a digit fused
    with it -- which made the same "0" measure 40 px tall beside a 28 px "1" and
    wrecked the height filter. Hence the band is located per frame.
    """
    m, g = _frame([(115, 14, 31), (130, 14, 32), (149, 14, 31)], bar=78)
    assert len(find_hits.glyphs(m, g)) == 3


def test_glyphs_are_scale_normalised():
    """The same count of glyphs at two zoom levels, each a fixed-size patch.

    Per-glyph normalisation is what survives the zoom; whole-readout
    normalisation does not, because the group's aspect ratio changes with both
    zoom and digit count.
    """
    small, gs = _frame([(115, 14, 28), (131, 14, 28), (147, 14, 28)])
    big, gb = _frame([(110, 18, 38), (130, 18, 38), (150, 18, 38)])
    a, b = find_hits.glyphs(small, gs), find_hits.glyphs(big, gb)
    assert len(a) == len(b) == 3
    assert all(x.shape == y.shape for x, y in zip(a, b))


def test_read_value_matches_templates():
    """End-to-end: seeded templates read the number back."""
    spans = [(112, 14, 31), (128, 14, 31), (144, 14, 31), (160, 14, 31)]
    m, g = _frame(spans)
    gl = find_hits.glyphs(m, g)
    assert len(gl) == 4
    tpl = {"7": [gl[0]], "1": [gl[1]], "3": [gl[2]], "9": [gl[3]]}
    assert find_hits.read_value(m, g, tpl) == 7139


def test_unconfident_match_returns_none():
    """A glyph unlike any template must yield no reading, not a wrong one.

    A wrong value invents an HP change; None merely makes the frame unreadable,
    and the caller reports the read rate.
    """
    m, g = _frame([(112, 14, 31), (128, 14, 31)])
    gl = find_hits.glyphs(m, g)
    bogus = {"5": [np.zeros_like(gl[0])]}
    assert find_hits.read_value(m, g, bogus) is None
