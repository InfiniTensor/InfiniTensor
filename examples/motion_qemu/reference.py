"""Independent NumPy oracle using the delivered floating-point neuron equations."""

import numpy as np


def reservoir(raw, previous, initial, wi, wr):
    state = np.array(initial, dtype=np.float64, copy=True)
    current, v, ref = state
    previous = previous.copy()
    counts = np.zeros((len(raw), 113), np.int32)
    threshold = np.r_[np.full(90, 10240.0), np.full(23, 15360.0)]
    period = np.r_[np.full(90, 0.3), np.full(23, 0.5)]
    for row, x in enumerate(raw):
        delta = x - previous
        previous = x.copy()
        remaining = np.zeros(42)
        remaining[2 * np.arange(21) + (delta < 0)] = np.minimum(
            50, np.ceil(np.abs(delta) / 1e-5)
        )
        prev = np.zeros((113, 1))
        for _ in range(50):
            spikes = (remaining > 0).astype(np.float64)
            spikes[1::2] *= -1
            remaining -= remaining > 0
            current[:] += (spikes[:, None].T @ wi + prev.T @ wr).flatten() * 1024
            active = ref > 0
            v[:] = np.where(active, v - v // 128, v - v // 128 + current // 1)
            ref[:] = np.where(active, ref - 0.1, ref)
            current[:] = np.where(active, current, current - current // 16)
            fired = v >= threshold
            v[fired] = 0
            current[fired] = 0
            ref[fired] = period[fired]
            prev = fired.astype(np.float64)[:, None]
            counts[row] += fired
    return counts, state, previous
