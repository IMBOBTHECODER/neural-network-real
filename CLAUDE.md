# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A neural network library built **from scratch on NumPy/CuPy** — no PyTorch/TensorFlow. It implements a full MLP and CNN (forward, backprop, AdamW, cosine LR schedule) by hand, trained on the EMNIST-balanced dataset (47 classes, 28×28 grayscale). The point is the hand-written math, not a framework wrapper.

## Commands

```bash
pip install -r requirements.txt   # numpy, scikit-learn, cupy-cuda12x, numba
python test.py                    # trains the CNN in the __main__ block, then evaluates
```

There is no test suite, linter, or build step. `test.py`'s `__main__` block is the entry point — edit it to change the model, dataset split, or training run.

## CPU vs GPU: the `xp` convention

Every module does `import cupy as xp` and falls back to `import numpy as xp` on ImportError. **Always use `xp` for array ops** so code runs on both. `np` (real NumPy) is kept separately and used only for disk I/O in `save`/`load` (CuPy arrays are converted via `xp.asnumpy`).

Caveat: `test.py`'s `__main__` calls `xp.cuda.Stream.null.synchronize()` unconditionally — this **crashes on CPU-only machines** (NumPy has no `.cuda`). Remove/guard those lines when running without a GPU.

`activation.py` picks a JIT per backend: `cupy.fuse()` on GPU, `numba.njit` on CPU (with an import-time warmup loop over hidden-layer shapes), or a no-op decorator if neither is present. `softmax` is deliberately left un-JITed.

## Architecture

Two files:
- `activation.py` — activation functions and their derivatives (ReLU, LeakyReLU, Sigmoid, Tanh, softmax), each paired as `Fn` / `Fn_derivative`.
- `test.py` — everything else: `Config`, `DataLoader`, layers, `NeuralNetwork` (MLP), `CNN`.

**Layer interface.** `ConvLayer`, `PoolLayer`, and `ActivationLayer` all expose `forward(x)` / `backward(grad)` and operate on NCHW tensors `(N, C, H, W)`. `CNN` just chains them. Crucially, **each layer owns its own optimizer state and applies its own weight update inside `backward`** — there is no separate optimizer object. `ConvLayer.backward` computes gradients *and* calls `self._adam_update(...)` before returning the input gradient. Keep this pattern when adding layers.

**Convolution is im2col-based.** `ConvLayer._im2col` loops over kernel *positions* (tiny, e.g. 9 for 3×3), not output positions, turning conv into a single matmul. `_col2im` reverses it for the backward pass, summing gradients where patches overlapped. Both are heavily commented — read them before touching conv shapes.

**MLP (`NeuralNetwork`).** He-initialized weights, LeakyReLU hidden layers, softmax output, cross-entropy loss. `backprop` fuses delta computation *and* the AdamW update in one pass. Its `compute_final_gradient=True` flag makes it also return the gradient w.r.t. its *input* — this is how the CNN backprops from the MLP tail into the conv stack.

**CNN = conv stack + MLP tail.** `CNN.forward` reshapes the flat `(N, 784)` batch into `(N, 1, 28, 28)`, runs the conv layers, flattens, then calls `self.mlp.forward`. `_compute_flatten_size` walks the layer list to derive the MLP's input size — so the MLP is constructed *after* the conv layers are known. `CNN.__init__` injects optimizer hyperparameters from `Config` into each `ConvLayer` (they carry safe defaults otherwise).

**Config.** The `Config` dataclass holds architecture + optimizer + training hyperparameters. `cnn_layer` is set imperatively in `__main__` as a list of instantiated layer objects (not declared in the dataclass default).

**Optimizer.** AdamW (Adam + decoupled weight decay) with bias correction, plus a cosine-annealing LR schedule (`update_lr`) from `initial_lr` down to `1e-5`. Both the MLP and each ConvLayer implement this independently; `CNN.update_lr` fans out to all of them.

## Persistence

Only `NeuralNetwork` has `save`/`load` (to `.npz`, storing weights, biases, full Adam moment buffers, and hyperparameters). `CNN` has **no** save/load yet — conv kernels are not persisted. `model.npz` and `emnist1.npz` are gitignored (`*.npz`) but present as local checkpoints.

## License

GPL-3.0 (`COPYING`). Source files carry a copyright header.
