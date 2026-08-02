import numpy as np

try:
    import cupy as xp
    GPU = True
except ImportError:
    import numpy as xp
    GPU = False

if GPU:
    compile_fn = xp.fuse()
else:
    try:
        from numba import njit
        compile_fn = njit
        NUMBA_AVAILABLE = True
    except ImportError:
        compile_fn = lambda f: f  # no-op decorator, just returns the plain function unchanged
        NUMBA_AVAILABLE = False


@compile_fn
def ReLU(x):
    return xp.maximum(0, x)

@compile_fn
def ReLU_derivative(x):
    return (x > 0).astype(xp.float32)


@compile_fn
def LeakyReLU(x, alpha=0.01):
    return xp.maximum(alpha * x, x)

@compile_fn
def LeakyReLU_derivative(x, alpha=0.01):
    return xp.where(x > 0, 1.0, alpha)


@compile_fn
def Sigmoid(x):
    return 1 / (1 + xp.exp(-x))

@compile_fn
def Sigmoid_derivative(x):
    s = 1 / (1 + xp.exp(-x))
    return s * (1 - s)


@compile_fn
def Tanh(x):
    return xp.tanh(x)

@compile_fn
def Tanh_derivative(x):
    return 1 - xp.tanh(x) ** 2


# softmax is NOT a candidate for fuse() or njit — see note below
def softmax(x):
    x = x - xp.max(x, axis=1, keepdims=True)
    exp = xp.exp(x)
    return exp / xp.sum(exp, axis=1, keepdims=True)


if not GPU and NUMBA_AVAILABLE:
    # Warm up njit compilation for every shape actually used in the network,
    # so the compile cost happens at import time, not during the first training batch.
    _warmup_shapes = [256, 128]  # match HIDDEN_LAYER from your main script

    for _size in _warmup_shapes:
        _dummy = np.zeros((1, _size), dtype=np.float32)
        _ = ReLU(_dummy)
        _ = ReLU_derivative(_dummy)
        _ = LeakyReLU(_dummy)
        _ = LeakyReLU_derivative(_dummy)
        _ = Sigmoid(_dummy)
        _ = Sigmoid_derivative(_dummy)
        _ = Tanh(_dummy)
        _ = Tanh_derivative(_dummy)

    del _warmup_shapes, _dummy, _size