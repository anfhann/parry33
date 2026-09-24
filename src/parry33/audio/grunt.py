"""Learned attack-sound detector.

This replaces the hand-tuned spectral-flux onset for the *firing* decision.
Flux answers "did something transient just happen", which in a combat mix is
true ~3.4 times a second: footsteps, impacts, the player's own attacks and the
enemy's grunt all look alike to it. Measured on 728 s of labelled audio, no
sensitivity setting held useful coverage below ~1.4 false onsets/s, so the
trigger was effectively picking a random transient inside its gate -- which is
exactly the reported symptom: fires "quite early or quite late", each mode
internally consistent, neither related to the attack.

The model answers the question we actually care about:

    given the trailing 141 ms of audio ending now, should we press now?

Positives are moments the player pressed *and landed* (confirmed by a clash
within 200 ms). Every feature is strictly earlier than the decision instant, so
the clash -- which arrives ~32 ms *after* the press -- cannot leak in.

Measured leave-one-session-out over 16 sessions / 351 landed parries:

    AUC 0.947                          (shuffled-label floor 0.454)
    served, sliding every 10 ms:       49% of real attacks parried
    whiffs                             3.5 per minute
    timing error                       p50 +7 ms

Measured on the five long early sessions -- the only ones with enough landed
parries to hold out. Those are a small set of enemies. Nothing here has been
validated against an enemy the model was not trained on.

Where the player's presses sit relative to this model's score peak, which is
what makes the peak usable as a landmark at all:

    landed presses   p25 +8 ms, p50 +18 ms, p75 +30 ms
    land rate        81% within peak+0..50 ms, 64% at +50..100, 29% at +150..200

So the press should be scheduled close to peak+20 ms. The lookahead already
adds +30 ms, and the input path (loop wakeup, SendInput, the game sampling at
60 Hz) adds more on top -- which is what --lead-ms exists to subtract.

The gap between 0.947 AUC and 48% is not a discrepancy -- AUC is measured on
sampled negatives, while serving slides continuously. The served number is the
honest one. This is the same train/serve gap that cost us the first vision
model, so it is measured here deliberately rather than discovered in a fight.

Where the rest goes, measured at th=0.99:

    never crossed threshold    17%
    fired but eaten by lockout  7%
    crossed but off-target     28%   <- dominant loss

That last line is why near-miss negatives matter (see model training): a model
trained only against distant negatives learns "an attack is happening", not
"press now", so its score plateaus across the whole grunt and the peak lands
wherever noise puts it.

THE MECHANIC, which drives the firing policy:

    successful parry -> instant reframe, 0 ms. Parry again immediately.
    whiffed parry    -> 1500 ms lockout, during which real attacks are lost.

So the cost is sharply asymmetric: correct presses are free, wrong ones are
expensive and can cost the *following* attack too. The policy is therefore a
high threshold with a short refractory -- reluctant to fire, but able to fire
again at once for combos. An earlier version had this backwards (a permissive
threshold with a 1200 ms lock, on the mistaken belief that every parry locked
out), which discarded real attacks in combos.

The refractory here exists only to avoid double-firing on ONE attack sound; it
is not modelling a game rule.
"""

from __future__ import annotations

import numpy as np

SR = 48000
NFFT = 1024
HOP = 480                 # 10 ms at 48 kHz
NFRAMES = 13              # 141 ms of trailing context
NMEL = 40
FMIN, FMAX = 50, 16000

# Chosen by sweeping the real mechanic (parry free, whiff = 1500 ms) against
# 272 labelled attacks. Measured, refractory 250 ms:
#
#     0.80 -> 52% parried, 4.1 whiffs/min
#     0.90 -> 49% parried, 3.5 whiffs/min   <- here
#     0.95 -> 49% parried, 2.9 whiffs/min
#     0.99 -> 43% parried, 1.8 whiffs/min
#
# 0.80 is the maximum but only 3 points above 0.90 while firing 15% more often,
# and every extra whiff is also a stray keypress out in the world.
#
# An earlier 0.99 was carried over from the model trained without near-miss
# negatives, whose score plateaued so broadly that only near-certainty was
# usable. Sharpening the score moved the operating point, and leaving the old
# threshold in place cost 14 points of parry rate to buy a whiff reduction that
# is not worth having: whiff cost is already priced into "parried".
THRESHOLD = 0.90
REFRACTORY_MS = 250.0
# Firing on the first threshold crossing catches the rising edge of the sound
# and locks out its true peak. Waiting 30 ms and taking the local maximum moved
# the timing error from -26 ms to +2 ms. 60 ms overshoots.
LOOKAHEAD_MS = 30.0


def mel_filterbank(sr=SR, n_fft=NFFT, n_mels=NMEL, fmin=FMIN, fmax=FMAX):
    to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = to_hz(np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2))
    b = np.clip(np.floor((n_fft + 1) * pts / sr).astype(int), 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo = b[i]
        mid = max(b[i + 1], lo + 1)
        hi = min(max(b[i + 2], mid + 1), n_fft // 2 + 1)
        mid = min(mid, hi - 1)
        fb[i, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        fb[i, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    return fb


FB = mel_filterbank()
HANN = np.hanning(NFFT).astype(np.float32)


def mel_frames(pcm: np.ndarray, fb: np.ndarray | None = None) -> np.ndarray:
    """Whole-signal log-mel, one frame per HOP samples. Offline path."""
    fb = FB if fb is None else fb
    fr = np.lib.stride_tricks.sliding_window_view(pcm, NFFT)[::HOP]
    mag = np.abs(np.fft.rfft(fr * HANN, axis=1))
    return np.log(mag @ fb.T + 1e-6).astype(np.float32)


def stack(frames: np.ndarray, nframes: int | None = None) -> np.ndarray:
    """(n, n_mels) log-mel -> (n-nframes+1, n_mels*nframes) feature rows.

    Row i ends at frame i+nframes-1. Mel-major, to match how training rows are
    built (band 0 across all frames, then band 1, ...). Training and serving
    MUST agree here; a transposed layout scores as noise rather than failing.
    """
    nframes = NFRAMES if nframes is None else nframes
    v = np.lib.stride_tricks.sliding_window_view(frames, nframes, axis=0)
    return v.reshape(v.shape[0], -1)


class GruntStream:
    """Turns a stream of audio blocks into scheduled press times.

    Blocks arrive at whatever size the capture uses (256 samples = 5.33 ms);
    frames must be emitted every HOP samples (10 ms) regardless. Decoupling the
    two rates is this class's job, not the caller's.
    """

    def __init__(self, model, threshold: float = THRESHOLD,
                 lookahead_ms: float = LOOKAHEAD_MS,
                 refractory_ms: float = REFRACTORY_MS,
                 nframes: int | None = None, nmel: int | None = None,
                 fmin: float | None = None, fmax: float | None = None):
        # Geometry comes from the caller (in practice, from the saved model
        # bundle) rather than from module constants. A model trained on 34
        # frames served by a stream built for 13 is not a subtle degradation --
        # it is garbage in, and if the dimensions happen to line up it fails
        # silently rather than raising. Tying the shape to the model that was
        # actually loaded makes the two impossible to desynchronise.
        self.nframes = NFRAMES if nframes is None else int(nframes)
        self.nmel = NMEL if nmel is None else int(nmel)
        fmin = FMIN if fmin is None else fmin
        fmax = FMAX if fmax is None else fmax
        self.fb = (FB if (self.nmel == NMEL and fmin == FMIN and fmax == FMAX)
                   else mel_filterbank(n_mels=self.nmel, fmin=fmin, fmax=fmax))
        want = self.nmel * self.nframes
        have = getattr(model, "n_features_in_", None)
        if have is not None and have != want:
            raise ValueError(
                f"model expects {have} features but this stream produces "
                f"{want} ({self.nmel} mels x {self.nframes} frames). The model "
                f"was trained with a different geometry -- retrain, or pass the "
                f"nframes/nmel/fmin/fmax recorded in its bundle.")
        self.model = model
        self.threshold = threshold
        self.lookahead = max(0, int(round(lookahead_ms / 10.0)))
        self.lookahead_ms = lookahead_ms
        self.refractory_ns = int(refractory_ms * 1e6)
        self._buf = np.zeros(NFFT, dtype=np.float32)
        self._filled = 0
        # Frame boundaries are absolute sample counts, not "every HOP samples
        # from whenever we started". Offline frame k covers [k*HOP, k*HOP+NFFT),
        # so the first frame ends at NFFT and every subsequent one HOP later.
        # Counting hops from sample 0 instead would offset every streamed frame
        # against every trained one -- the same one-frame skew that silently
        # broke the vision features.
        self._n = 0
        self._next_end = NFFT
        self._frames = np.zeros((self.nframes, self.nmel), dtype=np.float32)
        self._nseen = 0
        self._pend = None                     # (best_p, best_t, frames_left)
        self._lock_until = 0
        self.last_p = 0.0
        # The score of the window that caused the most recent fire. Distinct
        # from last_p, which is simply the newest window -- after the lookahead
        # confirm those differ, and reporting last_p made fires look like they
        # triggered on scores far below the threshold.
        self.last_fire_p = 0.0
        self.scored = 0

    def push(self, samples: np.ndarray, t_ns: int):
        """Feed one audio block; returns a press time in ns, or None.

        t_ns stamps the START of the block, matching the capture convention.
        Frame times are derived from it by sample offset, so the press is
        anchored to the audio clock rather than to when the main loop got round
        to draining the ring.
        """
        n = len(samples)
        i = 0
        while i < n:
            take = min(self._next_end - self._n, n - i)
            self._append(samples[i:i + take])
            self._n += take
            i += take
            if self._n < self._next_end:
                break
            self._next_end += HOP
            if self._frame():
                fired = self._decide(t_ns + int(i / SR * 1e9))
                if fired is not None:
                    return fired
        return None

    def _append(self, chunk):
        k = len(chunk)
        if not k:
            return
        if k >= NFFT:
            self._buf[:] = chunk[-NFFT:]
            self._filled = NFFT
            return
        self._buf[:-k] = self._buf[k:]
        self._buf[-k:] = chunk
        self._filled = min(NFFT, self._filled + k)

    def _frame(self) -> bool:
        if self._filled < NFFT:
            return False
        mag = np.abs(np.fft.rfft(self._buf * HANN))
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = np.log(self.fb @ mag + 1e-6)
        self._nseen += 1
        return True

    def _decide(self, t_ns: int):
        if self._nseen < self.nframes:
            return None
        p = float(self.model.predict_proba(self._frames.T.reshape(1, -1))[0, 1])
        self.last_p = p
        self.scored += 1
        if self._pend is not None:
            best_p, best_t, left = self._pend
            if p > best_p:
                best_p, best_t = p, t_ns
            left -= 1
            if left > 0:
                self._pend = (best_p, best_t, left)
                return None
            self._pend = None
            self._lock_until = best_t + self.refractory_ns
            self.last_fire_p = best_p
            return best_t + int(self.lookahead_ms * 1e6)
        if p >= self.threshold and t_ns >= self._lock_until:
            if self.lookahead == 0:
                self._lock_until = t_ns + self.refractory_ns
                self.last_fire_p = p
                return t_ns + int(self.lookahead_ms * 1e6)
            self._pend = (p, t_ns, self.lookahead)
        return None
