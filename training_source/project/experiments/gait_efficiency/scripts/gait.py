"""Finger-cycle counter shared by the real-hand and the simulation analysis (gait_efficiency study).

One function, ``count_cycles``, is used unchanged on both sides:

1. keep the samples with ``t_start_s <= t <= t_end_s`` (time from the policy start; default: drop the first 3 s);
2. optionally keep only some joints (robustness check (a): one finger);
3. centre each joint and project on the first principal component, fitted on this window only
   (PCA does not depend on joint order or sign, so no sim-to-kit joint mapping is needed);
4. band-pass 0.12-1.2 Hz (2nd-order Butterworth, zero-phase ``filtfilt``) and take the Hilbert phase;
5. cycles = |unwrapped phase change over the window| / 2 pi;
6. rhythm = frequency of the largest FFT magnitude above the lower band edge, from the centred PC1 projection
   (Hann window, zero-padded to at least 4096 samples to refine the peak location);
7. per-cycle stroke amplitude = peak-to-peak of the band-passed PC1 signal (rad) between successive phase
   wraps (complete cycles only).

All joint positions are in rad; the PC1 projection is in rad along a unit vector in joint space.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

BAND_HZ = (0.12, 1.2)
T_START_S = 3.0
MIN_SAMPLES = 40  # below this the window is reported as too short (2 s at 20 Hz)


def pc1_projection(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
  """Centred projection on the first principal component; returns (projection, unit vector, variance share)."""
  x = q - q.mean(axis=0, keepdims=True)
  _, s, vt = np.linalg.svd(x, full_matrices=False)
  v = vt[0]
  share = float(s[0] ** 2 / np.sum(s ** 2)) if np.sum(s ** 2) > 0 else float("nan")
  return x @ v, v, share


def bandpass(x: np.ndarray, fs: float, band: tuple[float, float] = BAND_HZ, order: int = 2) -> np.ndarray:
  b, a = butter(order, [band[0], band[1]], btype="band", fs=fs)
  return filtfilt(b, a, x)


def dominant_frequency(x: np.ndarray, fs: float, f_min: float) -> float:
  x = (x - x.mean()) * np.hanning(len(x))
  n = max(4096, int(2 ** np.ceil(np.log2(len(x)))))
  mag = np.abs(np.fft.rfft(x, n=n))
  f = np.fft.rfftfreq(n, d=1.0 / fs)
  ok = f > f_min
  return float(f[ok][np.argmax(mag[ok])])


def count_cycles(q: np.ndarray, t: np.ndarray, *, t_start_s: float = T_START_S, t_end_s: float | None = None,
                 band: tuple[float, float] = BAND_HZ, joints: list[int] | None = None) -> dict:
  """Finger cycles of one run.  ``q``: (T, J) joint positions at the policy rate; ``t``: (T,) seconds from the
  policy start.  Returns cycles, rhythm (Hz), window length (s), per-cycle stroke amplitudes (rad) and the
  cumulative-cycle time series (for the camera check)."""
  q = np.asarray(q, dtype=np.float64)
  t = np.asarray(t, dtype=np.float64)
  end = t[-1] if t_end_s is None else min(float(t_end_s), t[-1])
  m = (t >= t_start_s) & (t <= end)
  out = {"t_start_s": float(t_start_s), "t_end_s": float(end), "n_samples": int(m.sum())}
  if m.sum() < MIN_SAMPLES:
    out.update(status="window too short", cycles=float("nan"), rhythm_hz=float("nan"), window_s=float(max(0.0, end - t_start_s)))
    return out
  qw, tw = q[m], t[m]
  if joints is not None:
    qw = qw[:, joints]
  fs = 1.0 / float(np.median(np.diff(tw)))
  p, v, share = pc1_projection(qw)
  y = bandpass(p, fs, band)
  ph = np.unwrap(np.angle(hilbert(y)))
  d = ph - ph[0]
  sgn = 1.0 if d[-1] >= 0 else -1.0
  cum = sgn * d / (2 * np.pi)  # cumulative cycles, positive in the net direction of the phase
  k = np.floor(cum)
  edges = np.flatnonzero(np.diff(k) != 0) + 1
  amps = [float(y[a:b].max() - y[a:b].min()) for a, b in zip(edges[:-1], edges[1:]) if b - a >= 2]
  out.update(status="ok", fs_hz=fs, window_s=float(tw[-1] - tw[0]), cycles=float(abs(d[-1]) / (2 * np.pi)),
             rhythm_hz=dominant_frequency(p, fs, band[0]), pc1_variance_share=share,
             pc1_raw_range_rad=float(p.max() - p.min()), stroke_amplitudes_rad=amps,
             stroke_amplitude_median_rad=float(np.median(amps)) if amps else float("nan"),
             t_series=tw.tolist(), cum_cycles_series=cum.tolist())
  return out
