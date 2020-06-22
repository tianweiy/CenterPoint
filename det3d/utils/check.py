import numpy as np


def is_array_like(x):
    return isinstance(x, (list, tuple, np.ndarray))


def shape_mergeable(x, expected_shape):
    mergeable = True
    if is_array_like(x) and is_array_like(expected_shape):
        x = np.array(x)
        if len(x.shape) == len(expected_shape):
            for s, s_ex in zip(x.shape, expected_shape):
                if s_ex is not None and s != s_ex:
                    mergeable = False
                    break
    return mergeable
