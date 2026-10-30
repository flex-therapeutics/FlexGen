import torch
import numpy as np
from contextlib import contextmanager


@contextmanager
def sampling_rng(seed, devices):
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
