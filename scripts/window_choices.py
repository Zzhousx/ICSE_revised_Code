"""Draw one uniformly selected candidate window per log using NumPy PCG64."""
import sys
import numpy as np
from scipy.io import loadmat, savemat

x = loadmat(sys.argv[1])
lengths = np.asarray(x['lengths']).reshape(-1).astype(int)
window, step, runs = (int(np.asarray(x[k]).item()) for k in ('window','step','runs'))
counts = []
for length in lengths:
    starts = [0] if length <= window else list(range(0, length - window + 1, step))
    if length > window and starts[-1] + window < length:
        starts.append(length - window)
    counts.append(len(starts))
choices = np.empty((len(counts), runs), dtype=np.int32)
for seed in range(runs):
    rng = np.random.default_rng(seed)
    for i, count in enumerate(counts):
        choices[i, seed] = rng.integers(count) + 1
savemat(sys.argv[2], {'choices': choices})
