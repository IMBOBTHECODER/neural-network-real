# Copyright (C) 2026  Pham Tien Dat

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

import time
import pandas as pd
import math
from activation import LeakyReLU, LeakyReLU_derivative, softmax
from config import Config
from data_loader import DataLoader, get_data


def clip_grad_norm(grad, max_norm=5.0):
    norm = xp.sqrt(xp.sum(grad * grad))
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad

# Fused Functions

@compile_fn
def _adam_step(param, m, v, grad, lr, beta1, beta2, bc1, bc2, eps, wd):
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m_new / bc1
    v_hat = v_new / bc2
    denom = xp.sqrt(v_hat) + eps
    param_new = param * (1 - lr * wd) - lr * m_hat / denom
    return param_new, m_new, v_new


class ConvLayer():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.kernels = xp.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(xp.float32) * xp.float32(math.sqrt(2.0 / (in_channels * kernel_size * kernel_size)))

        self.bias = xp.zeros(out_channels, dtype=xp.float32)   # one per output channel

        self.stride = stride
        self.padding = padding

        # Adam state (per-layer, independent buffers)
        self.kernel_m = xp.zeros_like(self.kernels)
        self.kernel_v = xp.zeros_like(self.kernels)
        self.bias_m = xp.zeros_like(self.bias)
        self.bias_v = xp.zeros_like(self.bias)

        self.beta1_pow = 1.0
        self.beta2_pow = 1.0

        # Hyperparameter values injected by CNN.__init__, but give safe defaults
        self.learning_rate = 1e-3
        self.initial_lr = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.weight_decay = 1e-3
        self.grad_clip_norm = 5.0

    def _im2col(self, x, kH, kW, stride=1, padding=0):
        # Rearranges every sliding-window patch into a layout ready to become
        # matrix rows, so convolution can be done as one matmul instead of loops.
        #
        # x: (N, C, H, W)  — batch, channels, height, width (NCHW)
        # returns col of shape (N, C, kH, kW, H_out, W_out) plus the output dims

        N, C, H, W = x.shape

        if padding > 0:
            x = xp.pad(
                x,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode="constant",
            )

        H_pad, W_pad = x.shape[2], x.shape[3]        # use PADDED dims from here on

        # Output spatial size — how many positions the kernel stops at
        H_out = (H_pad - kH) // stride + 1
        W_out = (W_pad - kW) // stride + 1

        # 6D holder. Think of it as: for each position (i, j) inside the kernel
        # window, store the value that kernel-cell lands on, for every output
        # position, every channel, every image — all at once.
        col = xp.zeros((N, C, kH, kW, H_out, W_out), dtype=x.dtype)

        # KEY TRICK: loop over kernel positions (kH*kW = tiny, e.g. 9 for 3x3),
        # NOT over output positions (H_out*W_out = large). Each iteration grabs
        # one kernel-cell's value across ALL patches in a single strided slice.
        for i in range(kH):
            i_end = i + stride * H_out          # last row this offset reaches
            for j in range(kW):
                j_end = j + stride * W_out      # last col this offset reaches

                # One strided slice = "kernel-cell (i,j) for every output patch".
                # The ::stride steps by the conv stride so we land on exactly the
                # patch origins, not every pixel.
                col[:, :, i, j, :, :] = x[:, :, i:i_end:stride, j:j_end:stride]

        return col, H_out, W_out

    def _im2col_fast(self, x, kH, kW, stride=1, padding=0):
        N, C, H, W = x.shape

        if padding > 0:
            x = xp.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode="constant")

        H_pad, W_pad = x.shape[2], x.shape[3]

        windows = xp.lib.stride_tricks.sliding_window_view(x, (kH, kW), axis=(2, 3))
        windows = windows[:, :, ::stride, ::stride, :, :]

        H_out, W_out = windows.shape[2], windows.shape[3]
        col = windows.transpose(0, 1, 4, 5, 2, 3)   # -> (N, C, kH, kW, H_out, W_out)

        return col, H_out, W_out

    def _col2im(self, d_col, N, C, kH, kW, H, W):
        stride = self.stride
        padding = self.padding

        H_pad = H + 2 * padding
        W_pad = W + 2 * padding

        H_out = (H_pad - kH) // stride + 1
        W_out = (W_pad - kW) // stride + 1

        d_col = d_col.reshape(N, H_out, W_out, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)
        d_x = xp.zeros((N, C, H_pad, W_pad), dtype=d_col.dtype)

        for i in range(kH):
            i_end = i + stride * H_out
            for j in range(kW):
                j_end = j + stride * W_out
                d_x[:, :, i:i_end:stride, j:j_end:stride] += d_col[:, :, i, j, :, :]

        if padding > 0:
            d_x = d_x[:, :, padding:-padding, padding:-padding]
        return d_x


    def forward(self, x):
        # Full vectorized conv: im2col -> reshape -> one matmul -> reshape back.
        #
        # x:       (N, C_in, H, W)
        # kernels: (C_out, C_in, kH, kW)
        # returns: (N, C_out, H_out, W_out)

        self.x_shape = x.shape
        kernel_shape = self.kernels.shape

        N, C, H, W = self.x_shape
        C_out = kernel_shape[0]
        kH, kW = kernel_shape[2], kernel_shape[3]

        # Step 1: gather all patches (still in 6D "per kernel-cell" layout)
        col, self.H_out, self.W_out = self._im2col_fast(x, kH, kW, self.stride, self.padding)

        # Step 2: reshape into a 2D matrix where each ROW is one flattened patch.
        #
        # transpose(0, 4, 5, 1, 2, 3) reorders the axes to:
        #   (N, H_out, W_out,  C, kH, kW)
        # so the three axes a conv sums over (C, kH, kW) end up LAST and adjacent.
        # Then reshape collapses:
        #   - (N, H_out, W_out) -> rows   (one row per output patch)
        #   - (C, kH, kW)       -> cols   (the flattened patch = matmul contraction)
        self.col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * self.H_out * self.W_out, -1)

        # Flatten each kernel the SAME (C, kH, kW) way so its values line up with
        # the patch columns above. .T makes it (C*kH*kW, C_out) for the matmul.
        w_col = self.kernels.reshape(C_out, -1).T

        # Step 3: ONE matmul does every patch's dot product with every kernel.
        # (N*H_out*W_out, C*kH*kW) @ (C*kH*kW, C_out) -> (N*H_out*W_out, C_out)
        out = self.col @ w_col

        # Step 4: reshape the flat results back into image form.
        # We built rows as (N, H_out, W_out), so unflatten to that, put C_out last,
        # then transpose to channels-first (NCHW) to match the rest of the pipeline.
        out = out.reshape(N, self.H_out, self.W_out, C_out).transpose(0, 3, 1, 2)

        out = out + self.bias.reshape(1, C_out, 1, 1)      # bias broadcasts across N, H_out, W_out

        return out

    def backward(self, d_out):
        # d_out is the gradient flowing in from the layer after this one.
        # It has the same shape as this layer's OUTPUT: (N, C_out, H_out, W_out).
        # Our job: (1) find how to update our kernels, (2) find the gradient to
        # hand back to the layer BEFORE us. Then return (2).

        N, C, H, W = self.x_shape                       # the input shape we saw in forward
        C_out = self.kernels.shape[0]                   # number of kernels / output channels
        kH, kW = self.kernels.shape[2], self.kernels.shape[3]

        # Flatten d_out so it lines up with self.col (the flattened patches from forward).
        # We want one row per output position, matching how col was built.
        # (N, C_out, H_out, W_out) -> (N*H_out*W_out, C_out)
        d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(N * self.H_out * self.W_out, C_out)

        # KERNEL GRADIENT — "how should each kernel weight change?"
        # Rule: a weight's gradient = the input it multiplied, times the output gradient.
        # self.col holds those inputs; multiplying gives us the kernel gradient.
        # Same idea as the MLP's  activations.T @ delta.
        d_w_col = self.col.T @ d_out_flat               # (C*kH*kW, C_out)
        d_kernels = d_w_col.T.reshape(C_out, C, kH, kW) # shape it back like the kernels

        d_bias = xp.sum(d_out, axis=(0, 2, 3))     # sum over N, H_out, W_out -> shape (C_out,)

        # INPUT GRADIENT — "how did each input pixel affect the loss?"
        # Rule: an input's gradient = the kernel it multiplied, times the output gradient.
        # Same idea as the MLP's  delta @ weights.T.
        # This comes out in flattened-patch form and still needs unflattening (next line).
        w_col = self.kernels.reshape(C_out, -1)         # (C_out, C*kH*kW)
        d_col = d_out_flat @ w_col                      # (N*H_out*W_out, C*kH*kW)

        # col2im: turn that flattened-patch gradient back into a real image.
        # Because one pixel appeared in several patches, its contributions get SUMMED.
        d_x = self._col2im(d_col, N, C, kH, kW, H, W)   # (N, C, H, W) — matches the input

        # Gradient Clipping
        d_kernels = clip_grad_norm(d_kernels, self.grad_clip_norm)
        d_bias = clip_grad_norm(d_bias, self.grad_clip_norm)

        # Use the kernel gradient to actually update the kernels (Adam does this).
        # Note: this changes self.kernels; it does NOT touch d_x.
        self._adam_update(d_kernels, d_bias)

        # Hand the input gradient back so the previous layer can keep going.
        return d_x

    def _adam_update(self, d_kernels, d_bias):
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        bc1 = 1 - self.beta1_pow
        bc2 = 1 - self.beta2_pow

        self.kernels, self.kernel_m, self.kernel_v = _adam_step(
            self.kernels, self.kernel_m, self.kernel_v, d_kernels,
            self.learning_rate, self.beta1, self.beta2, bc1, bc2, self.eps, self.weight_decay
        )
        self.bias, self.bias_m, self.bias_v = _adam_step(
            self.bias, self.bias_m, self.bias_v, d_bias,
            self.learning_rate, self.beta1, self.beta2, bc1, bc2, self.eps, 0.0   # no decay on bias
        )

    def update_lr(self, epoch, total_epochs):
        min_lr = 1e-5
        self.learning_rate = (
            min_lr
            + (self.initial_lr - min_lr)
            * (1 + math.cos(math.pi * epoch / total_epochs))
            / 2
        )

    def get_state(self):
        return {
            'kernels': xp.asnumpy(self.kernels),
            'bias': xp.asnumpy(self.bias),
            'kernel_m': xp.asnumpy(self.kernel_m),
            'kernel_v': xp.asnumpy(self.kernel_v),
            'bias_m': xp.asnumpy(self.bias_m),
            'bias_v': xp.asnumpy(self.bias_v),
            'beta1_pow': self.beta1_pow,
            'beta2_pow': self.beta2_pow,
        }

    def load_state(self, state):
        self.kernels = xp.asarray(state['kernels'])
        self.bias = xp.asarray(state['bias'])
        self.kernel_m = xp.asarray(state['kernel_m'])
        self.kernel_v = xp.asarray(state['kernel_v'])
        self.bias_m = xp.asarray(state['bias_m'])
        self.bias_v = xp.asarray(state['bias_v'])
        self.beta1_pow = float(state['beta1_pow'])
        self.beta2_pow = float(state['beta2_pow'])


class PoolLayer():
    def __init__(self, pool_size, stride, training=False):
        self.training = training
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.in_shape = x.shape

        N, C, H, W = self.in_shape
        ps = self.pool_size

        # Crop the leftover edge so dimensions divide evenly by pool_size.
        # This matches the loop version, which ignored the unfillable row/col.
        H_crop = (H // ps) * ps      # 11 -> 10
        W_crop = (W // ps) * ps      # 11 -> 10
        x = x[:, :, :H_crop, :W_crop]

        H_out = H_crop // ps
        W_out = W_crop // ps

        x_reshaped = x.reshape(N, C, H_out, ps, W_out, ps)
        out = x_reshaped.max(axis=(3, 5))

        if self.training:
            max_expanded = out[:, :, :, None, :, None]
            self.mask = (x_reshaped == max_expanded).reshape(N, C, H_crop, W_crop).astype(xp.float32)

        return out
    
    def backward(self, grad):
        # grad: (N, C, H_out, W_out) — one gradient per pooling window
        # self.mask: (N, C, H_crop, W_crop) — 1.0 at each window's max position
        N, C, H_out, W_out = grad.shape
        ps = self.pool_size

        # Upsample grad: each output cell's gradient spread across its ps×ps window.
        # Insert size-1 axes where the within-window dims go, then broadcast.
        # (N, C, H_out, W_out) -> (N, C, H_out, 1, W_out, 1) -> repeat to (N, C, H_out, ps, W_out, ps)
        grad_up = grad[:, :, :, None, :, None]                       # (N, C, H_out, 1, W_out, 1)
        grad_up = xp.broadcast_to(grad_up, (N, C, H_out, ps, W_out, ps))
        grad_up = grad_up.reshape(N, C, H_out * ps, W_out * ps)      # (N, C, H_crop, W_crop)

        # Keep gradient only at the max positions (mask is 1 there, 0 elsewhere).
        d_input = grad_up * self.mask                                # (N, C, H_crop, W_crop)

        # Reverse-pad back to the original pre-crop size (dropped edge gets zero grad).
        N_, C_, H_orig, W_orig = self.in_shape
        if d_input.shape[2] != H_orig or d_input.shape[3] != W_orig:
            full = xp.zeros((N_, C_, H_orig, W_orig), dtype=d_input.dtype)
            full[:, :, :d_input.shape[2], :d_input.shape[3]] = d_input
            d_input = full

        return d_input


class ActivationLayer():
    def forward(self, x):
        self.last_input = x        # cache for backward (activation needs pre-activation values)
        return LeakyReLU(x)

    def backward(self, grad):
        return grad * LeakyReLU_derivative(self.last_input)


class BatchNormLayer():
    def __init__(self, num_channels, momentum, training=False):
        self.num_channels = num_channels
        self.momentum = momentum
        self.training = training

        # learnable parameters
        self.gamma = xp.ones(num_channels, dtype=xp.float32)
        self.beta = xp.zeros(num_channels, dtype=xp.float32)

        # Adam state (per-layer, independent buffers)
        self.gamma_m = xp.zeros_like(self.gamma)
        self.gamma_v = xp.zeros_like(self.gamma)
        self.beta_m = xp.zeros_like(self.beta)
        self.beta_v = xp.zeros_like(self.beta)

        self.beta1_pow = 1.0
        self.beta2_pow = 1.0

        # non-learned running statistics (eval-time use)
        self.running_mean = xp.zeros(num_channels, dtype=xp.float32)
        self.running_var = xp.ones(num_channels, dtype=xp.float32)   # init to 1, not 0

        # Hyperparameter values injected by CNN.__init__, but give safe defaults
        self.learning_rate = 1e-3
        self.initial_lr = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.weight_decay = 0.0
        self.grad_clip_norm = 5.0

    def forward(self, x):
        self.x = x

        if self.training:
            mean = xp.mean(x, axis=(0, 2, 3), keepdims=True)
            mean_sq = xp.mean(x * x, axis=(0, 2, 3), keepdims=True)
            self.mean = mean
            self.var = mean_sq - mean * mean

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean.reshape(-1)
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var.reshape(-1)
        else:
            self.mean = self.running_mean.reshape(1, -1, 1, 1)
            self.var = self.running_var.reshape(1, -1, 1, 1)

        gamma_r = self.gamma.reshape(1, -1, 1, 1)
        beta_r = self.beta.reshape(1, -1, 1, 1)

        self.x_norm = (x - self.mean) / xp.sqrt(self.var + self.eps)
        out = gamma_r * self.x_norm + beta_r

        return out

    def backward(self, d_out):
        # d_out: (N, C, H, W) — gradient w.r.t. this layer's output
        N, C, H, W = d_out.shape
        m = N * H * W   # number of elements averaged per channel

        gamma = self.gamma
        beta = self.beta
        x_norm = self.x_norm

        d_gamma = xp.sum(d_out * x_norm, axis=(0, 2, 3))
        d_beta = xp.sum(d_out, axis=(0, 2, 3))

        # x feeds into mean, var, AND x_norm directly — all three paths must be summed.
        gamma = self.gamma.reshape(1, C, 1, 1)
        std_inv = 1.0 / xp.sqrt(self.var + self.eps)   # cached from forward, reshaped to broadcast

        d_x_norm = d_out * gamma   # gradient w.r.t. x_norm, undoing the gamma multiply

        d_var = xp.sum(d_x_norm * (self.x - self.mean) * -0.5 * std_inv**3, axis=(0, 2, 3), keepdims=True)
        d_mean = xp.sum(d_x_norm * -std_inv, axis=(0, 2, 3), keepdims=True) \
                + d_var * xp.mean(-2.0 * (self.x - self.mean), axis=(0, 2, 3), keepdims=True)

        d_x = (d_x_norm * std_inv) \
            + (d_var * 2.0 * (self.x - self.mean) / m) \
            + (d_mean / m)

        # Gradient clipping
        d_gamma = clip_grad_norm(d_gamma, self.grad_clip_norm)
        d_beta = clip_grad_norm(d_beta, self.grad_clip_norm)

        self._adam_update(d_gamma, d_beta)

        return d_x

    def _adam_update(self, d_gamma, d_beta):
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        bc1 = 1 - self.beta1_pow
        bc2 = 1 - self.beta2_pow

        self.gamma, self.gamma_m, self.gamma_v = _adam_step(
            self.gamma, self.gamma_m, self.gamma_v, d_gamma,
            self.learning_rate, self.beta1, self.beta2, bc1, bc2, self.eps, 0.0
        )
        self.beta, self.beta_m, self.beta_v = _adam_step(
            self.beta, self.beta_m, self.beta_v, d_beta,
            self.learning_rate, self.beta1, self.beta2, bc1, bc2, self.eps, 0.0
        )

    def update_lr(self, epoch, total_epochs):
        min_lr = 1e-5
        self.learning_rate = (
            min_lr
            + (self.initial_lr - min_lr)
            * (1 + math.cos(math.pi * epoch / total_epochs))
            / 2
        )

    def get_state(self):
        return {
            'gamma': xp.asnumpy(self.gamma),
            'beta': xp.asnumpy(self.beta),
            'gamma_m': xp.asnumpy(self.gamma_m),
            'gamma_v': xp.asnumpy(self.gamma_v),
            'beta_m': xp.asnumpy(self.beta_m),
            'beta_v': xp.asnumpy(self.beta_v),
            'running_mean': xp.asnumpy(self.running_mean),
            'running_var': xp.asnumpy(self.running_var),
            'beta1_pow': self.beta1_pow,
            'beta2_pow': self.beta2_pow,
        }

    def load_state(self, state):
        self.gamma = xp.asarray(state['gamma'])
        self.beta = xp.asarray(state['beta'])
        self.gamma_m = xp.asarray(state['gamma_m'])
        self.gamma_v = xp.asarray(state['gamma_v'])
        self.beta_m = xp.asarray(state['beta_m'])
        self.beta_v = xp.asarray(state['beta_v'])
        self.running_mean = xp.asarray(state['running_mean'])
        self.running_var = xp.asarray(state['running_var'])
        self.beta1_pow = float(state['beta1_pow'])
        self.beta2_pow = float(state['beta2_pow'])


class NeuralNetwork():
    def __init__(self, input_node: int, hidden_layer: list[int], output_node: int,
                batch_size: int = 64, learning_rate: float = 1e-3, initial_lr: float = 1e-4,
                beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 1e-4, dropout_rate = 0.0, grad_clip_norm = 5.0):
        self.layers = [input_node,*hidden_layer, output_node]
        self.size = len(self.layers)

        self.training_step = 0
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.initial_lr = initial_lr

        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        self.grad_clip_norm = grad_clip_norm

        self.dropout_rate = dropout_rate
        self.training = True
        self.dropout_masks = [None] * (self.size - 1)

        # Bias Correction
        self.beta1_pow = 1.0
        self.beta2_pow = 1.0

        # Can be switched to Xavier when using Sigmoid, Tanh, etc
        # He initiallisation

        self.weights = [
            (
                xp.random.randn(self.layers[i], self.layers[i + 1]).astype(xp.float32)
                * xp.float32(math.sqrt(2.0 / self.layers[i]))
            )
            for i in range(self.size - 1)
        ]

        # Biases set to zero first
        self.biases = [
            xp.zeros(self.layers[i + 1], dtype=xp.float32)
            for i in range(self.size - 1)
        ]

        # Value after activation (filled per forward pass; rebound, not reused)
        self.activations = [None] * self.size

        # Value before activation (filled per forward pass; rebound, not reused)
        self.logits = [None] * (self.size - 1)

        # Adam's momentum and variance
        self.weight_m = [
            xp.zeros_like(w)
            for w in self.weights
        ]

        self.weight_v = [
            xp.zeros_like(w)
            for w in self.weights
        ]

        self.bias_m = [
            xp.zeros_like(b)
            for b in self.biases
        ]

        self.bias_v = [
            xp.zeros_like(b)
            for b in self.biases
        ]

    def forward(self, inputs):
        self.activations[0] = inputs

        for layer in range(1, self.size - 1):     # hidden layers only
            l = layer - 1
            x = self.activations[l]
            z = x @ self.weights[l] + self.biases[l]
            self.logits[l] = z

            a = LeakyReLU(z)

            # dropout after activation, training only
            if self.training and self.dropout_rate > 0:
                mask = (xp.random.rand(*a.shape) > self.dropout_rate).astype(xp.float32)
                a = a * mask / (1 - self.dropout_rate)
                self.dropout_masks[l] = mask

            self.activations[layer] = a

        # output layer — NO dropout, NO LeakyReLU, just softmax
        x = self.activations[-2]
        z = x @ self.weights[-1] + self.biases[-1]
        self.logits[-1] = z
        self.activations[-1] = softmax(z)
        return self.activations[-1]

    def backprop(self, prediction, target, compute_final_gradient = False):
        self.training_step += 1

        xp.clip(prediction, 1e-15, 1.0, out=prediction)

        loss = -xp.mean(xp.sum(target * xp.log(prediction), axis=1))

        # Compute deltas (same order as weights)
        delta = [None] * len(self.weights)

        delta[-1] = prediction - target
        for l in range(len(self.weights) - 2, -1, -1):
            delta[l] = (
                delta[l + 1] @ self.weights[l + 1].T
            ) * LeakyReLU_derivative(self.logits[l])

            if self.training and self.dropout_rate > 0 and self.dropout_masks[l] is not None:
                delta[l] = delta[l] * self.dropout_masks[l] / (1 - self.dropout_rate)

        final_delta = None
        if compute_final_gradient:
            final_delta = delta[0] @ self.weights[0].T

        lr = self.learning_rate
        eps = self.eps
        wd = self.weight_decay
        beta1 = self.beta1
        beta2 = self.beta2
        grad_clip_norm = self.grad_clip_norm

        self.beta1_pow *= beta1
        self.beta2_pow *= beta2

        bias_correction1 = 1 - self.beta1_pow
        bias_correction2 = 1 - self.beta2_pow

        for l in range(len(self.weights)):
            d = delta[l]
            batch_size = d.shape[0]

            gradient = self.activations[l].T @ d / batch_size
            bias_gradient = xp.sum(d, axis=0) / batch_size

            gradient = clip_grad_norm(gradient, grad_clip_norm)
            bias_gradient = clip_grad_norm(bias_gradient, grad_clip_norm)

            self.weights[l], self.weight_m[l], self.weight_v[l] = _adam_step(
                self.weights[l], self.weight_m[l], self.weight_v[l], gradient,
                lr, beta1, beta2, bias_correction1, bias_correction2, eps, wd
            )
            self.biases[l], self.bias_m[l], self.bias_v[l] = _adam_step(
                self.biases[l], self.bias_m[l], self.bias_v[l], bias_gradient,
                lr, beta1, beta2, bias_correction1, bias_correction2, eps, 0.0
            )

        if compute_final_gradient:
            return loss, final_delta

        return loss

    def update_lr(self, epoch, total_epochs):
        min_lr = 1e-5

        self.learning_rate = (
            min_lr
            + (self.initial_lr - min_lr)
            * (1 + math.cos(math.pi * epoch / total_epochs))
            / 2
        )

    def evaluate(self, loader):
        correct = 0
        total = 0
        total_loss = 0

        for x_batch, y_batch in loader:
            pred = self.forward(x_batch)

            # Avoid log(0)
            prediction = pred.copy()
            xp.clip(prediction, 1e-15, 1.0, out=prediction)

            loss = -xp.mean(xp.sum(y_batch * xp.log(prediction), axis=1))
            total_loss += loss

            pred_class = xp.argmax(pred, axis=1)
            true_class = xp.argmax(y_batch, axis=1)

            correct += xp.sum(pred_class == true_class)
            total += len(x_batch)

        return float(total_loss / len(loader)), float(correct / total)
    
    def train(self, loader, epochs):
        for epoch in range(epochs + 1):
            total_loss = 0

            self.update_lr(epoch, epochs)

            for x_batch, y_batch in loader:
                pred = self.forward(x_batch)
                loss = self.backprop(
                    pred,
                    y_batch
                )
                total_loss += loss

            if epoch % 2 == 0:
                print(
                    f"[Epoch {epoch}] "
                    f"LR={self.learning_rate:.6f} "
                    f"Loss={total_loss/len(loader):.4f}\n"
                )

    def get_state(self):
        return {
            'weights': [xp.asnumpy(w) for w in self.weights],
            'biases': [xp.asnumpy(b) for b in self.biases],
            'weight_m': [xp.asnumpy(m) for m in self.weight_m],
            'weight_v': [xp.asnumpy(v) for v in self.weight_v],
            'bias_m': [xp.asnumpy(m) for m in self.bias_m],
            'bias_v': [xp.asnumpy(v) for v in self.bias_v],
            'learning_rate': self.learning_rate,
            'initial_lr': self.initial_lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'grad_clip_norm': self.grad_clip_norm,
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'training_step': self.training_step,
            'beta1_pow': self.beta1_pow,
            'beta2_pow': self.beta2_pow,
        }

    def load_state(self, state):
        self.weights = [xp.asarray(w) for w in state['weights']]
        self.biases = [xp.asarray(b) for b in state['biases']]
        self.weight_m = [xp.asarray(m) for m in state['weight_m']]
        self.weight_v = [xp.asarray(v) for v in state['weight_v']]
        self.bias_m = [xp.asarray(m) for m in state['bias_m']]
        self.bias_v = [xp.asarray(v) for v in state['bias_v']]

        self.learning_rate = float(state['learning_rate'])
        self.initial_lr = float(state['initial_lr'])
        self.beta1 = float(state['beta1'])
        self.beta2 = float(state['beta2'])
        self.eps = float(state['eps'])
        self.weight_decay = float(state['weight_decay'])
        self.grad_clip_norm = float(state['grad_clip_norm'])
        self.dropout_rate = float(state['dropout_rate'])
        self.batch_size = int(state['batch_size'])
        self.training_step = int(state['training_step'])
        self.beta1_pow = float(state['beta1_pow'])
        self.beta2_pow = float(state['beta2_pow'])

    def save(self, path: str = "model.npz"):
        if not path.endswith(".npz"):
            path += ".npz"

        state = self.get_state()
        np.savez(path, state=np.array(state, dtype=object))

    def load(self, path: str = "model.npz"):
        if not path.endswith(".npz"):
            path += ".npz"

        data = np.load(path, allow_pickle=True)
        state = data['state'].item()   # .item() unwraps the 0-d object array back to the dict
        self.load_state(state)


class CNN():
    def __init__(self, config):
        self.input_shape = config.input_shape
        self.layers = config.cnn_layer

        # Inject optimizer hyperparameters into every conv layer (once, from config)
        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                layer.learning_rate = config.learning_rate
                layer.initial_lr = config.initial_lr
                layer.beta1 = config.beta1
                layer.beta2 = config.beta2
                layer.eps = config.eps
                layer.weight_decay = config.weight_decay
                layer.grad_clip_norm = config.grad_clip_norm

            if isinstance(layer, BatchNormLayer):
                layer.learning_rate = config.learning_rate
                layer.initial_lr = config.initial_lr
                layer.beta1 = config.beta1
                layer.beta2 = config.beta2
                layer.eps = config.eps
                layer.weight_decay = 0.0   # deliberately NOT config.weight_decay
                layer.grad_clip_norm = config.grad_clip_norm

        flatten_size = self._compute_flatten_size(config.input_shape)

        # The MLP tail — input_node is the FLATTEN SIZE, not 784
        self.mlp = NeuralNetwork(
            input_node=flatten_size,   # <-- must be the post-conv flatten size
            hidden_layer=config.hidden_layer,
            output_node=config.output_node,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            initial_lr=config.initial_lr,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
            dropout_rate=config.dropout_rate,
            grad_clip_norm=config.grad_clip_norm
        )

    def forward(self, x):
        # x arrives as (batch, 784) flat from your DataLoader
        batch = x.shape[0]
        x = x.reshape(batch, *self.input_shape)   # unflatten to image

        for layer in self.layers:
            x = layer.forward(x)          # conv/pool/activation, uniform interface

        self.flatten_shape = x.shape
        x = x.reshape(batch, -1)          # flatten for the MLP
        return self.mlp.forward(x)

    def backprop(self, prediction, target):
        loss, gradient_flat = self.mlp.backprop(prediction, target, True)

        grad = gradient_flat.reshape(self.flatten_shape)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return loss

    def update_lr(self, epoch, total_epochs):
        self.mlp.update_lr(epoch, total_epochs)     # the MLP tail
        for layer in self.layers:
            if hasattr(layer, "update_lr"):          # conv layers have it; pool/activation don't
                layer.update_lr(epoch, total_epochs)

    def train(self, loader, epochs):
        self.set_training(True)

        for epoch in range(epochs + 1):
            total_loss = 0

            self.update_lr(epoch, epochs)

            for x_batch, y_batch in loader:
                pred = self.forward(x_batch)
                loss = self.backprop(
                    pred,
                    y_batch
                )
                total_loss += loss

            if epoch % 1 == 0:
                print(
                    f"[Epoch {epoch}] "
                    f"Loss={total_loss/len(loader):.4f}\n"
                )

    def evaluate(self, loader):
        self.set_training(False)

        correct = 0
        total = 0
        total_loss = 0

        for x_batch, y_batch in loader:
            pred = self.forward(x_batch)          # full CNN forward
            prediction = pred.copy()
            xp.clip(prediction, 1e-15, 1.0, out=prediction)

            loss = -xp.mean(xp.sum(y_batch * xp.log(prediction), axis=1))
            total_loss += loss

            correct += xp.sum(xp.argmax(pred, axis=1) == xp.argmax(y_batch, axis=1))
            total += len(x_batch)

        self.set_training(True)           # restore training mode afterward
        return float(total_loss / len(loader)), float(correct / total)

    def _compute_flatten_size(self, input_shape):
        # input_shape: (channels, H, W) — e.g. (1, 28, 28)
        channels, height, width = input_shape

        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                k = layer.kernels.shape[2]
                stride = layer.stride
                padding = layer.padding
                height = (height + 2*padding - k) // stride + 1
                width  = (width  + 2*padding - k) // stride + 1
                channels = layer.kernels.shape[0]

            elif isinstance(layer, PoolLayer):
                p = layer.pool_size
                stride = layer.stride
                height = (height - p) // stride + 1
                width  = (width  - p) // stride + 1
                # channels unchanged for pooling

            # BatchNormLayer, ActivationLayer: no shape change, skip

        return channels * height * width

    def set_training(self, training):
        self.mlp.training = training
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = training

    def save(self, path: str = "cnn_model.npz"):
        if not path.endswith(".npz"):
            path += ".npz"

        layer_states = []
        for layer in self.layers:
            if hasattr(layer, "get_state"):
                layer_states.append(layer.get_state())
            else:
                layer_states.append(None)   # Pool/Activation — nothing to save

        np.savez(
            path,
            layer_states=np.array(layer_states, dtype=object),
        )

        # MLP gets its own file (reuses NeuralNetwork.save as-is)
        mlp_path = path.replace(".npz", "_mlp.npz")
        self.mlp.save(mlp_path)

    def load(self, path: str = "cnn_model.npz"):
        if not path.endswith(".npz"):
            path += ".npz"

        data = np.load(path, allow_pickle=True)
        layer_states = data["layer_states"]

        for layer, state in zip(self.layers, layer_states):
            if state is not None and hasattr(layer, "load_state"):
                layer.load_state(state)

        mlp_path = path.replace(".npz", "_mlp.npz")
        self.mlp.load(mlp_path)

def describe_layer(layer):
    if isinstance(layer, ConvLayer):
        return {
            'type': 'ConvLayer',
            'in_channels': int(layer.kernels.shape[1]),
            'out_channels': int(layer.kernels.shape[0]),
            'kernel_size': int(layer.kernels.shape[2]),
            'stride': layer.stride,
            'padding': layer.padding,
        }
    elif isinstance(layer, BatchNormLayer):
        return {
            'type': 'BatchNormLayer',
            'num_channels': layer.num_channels,
            'momentum': layer.momentum,
        }
    elif isinstance(layer, PoolLayer):
        return {
            'type': 'PoolLayer',
            'pool_size': layer.pool_size,
            'stride': layer.stride,
        }
    elif isinstance(layer, ActivationLayer):
        return {'type': 'ActivationLayer'}
    else:
        raise ValueError(f"Unknown layer type: {type(layer).__name__}")


def build_layer_from_description(desc):
    t = desc['type']
    if t == 'ConvLayer':
        return ConvLayer(desc['in_channels'], desc['out_channels'], desc['kernel_size'],
                        stride=desc['stride'], padding=desc['padding'])
    elif t == 'BatchNormLayer':
        return BatchNormLayer(desc['num_channels'], momentum=desc['momentum'])
    elif t == 'PoolLayer':
        return PoolLayer(desc['pool_size'], desc['stride'])
    elif t == 'ActivationLayer':
        return ActivationLayer()
    else:
        raise ValueError(f"Unknown layer type in description: {t}")


if __name__ == "__main__":
    config = Config()
    config.cnn_layer = [
        ConvLayer(1, 16, 3, padding=1),
        BatchNormLayer(16, momentum=0.9),
        ActivationLayer(),

        ConvLayer(16, 16, 3, padding=1),
        BatchNormLayer(16, momentum=0.9),
        ActivationLayer(),

        PoolLayer(2, 2),

        ConvLayer(16, 32, 3, padding=1),
        BatchNormLayer(32, momentum=0.9),
        ActivationLayer(),

        ConvLayer(32, 32, 3, padding=1),
        BatchNormLayer(32, momentum=0.9),
        ActivationLayer(),

        PoolLayer(2, 2),
    ]

    t0 = time.perf_counter()
    X_train, X_test, Y_train, Y_test = get_data("/kaggle/input/datasets/crawford/emnist/emnist-bymerge-train.csv", "/kaggle/input/datasets/crawford/emnist/emnist-bymerge-test.csv")

    # train_df = pd.read_csv('/kaggle/input/datasets/crawford/emnist/emnist-bymerge-train.csv', header=None)
    # test_df = pd.read_csv('/kaggle/input/datasets/crawford/emnist/emnist-bymerge-test.csv', header=None)

    # Y_train = train_df.iloc[:, 0].to_numpy().astype(np.int32)
    # X_train = train_df.iloc[:, 1:].to_numpy().astype(np.float32) / 255.0

    # Y_test = test_df.iloc[:, 0].to_numpy().astype(np.int32)
    # X_test = test_df.iloc[:, 1:].to_numpy().astype(np.float32) / 255.0

    # X_train, X_test = xp.asarray(X_train), xp.asarray(X_test)
    # Y_train, Y_test = xp.asarray(Y_train), xp.asarray(Y_test)

    train_loader = DataLoader(
        X_train, Y_train,
        batch_size=config.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        X_test, Y_test,
        batch_size=config.batch_size,
        shuffle=False
    )

    t1 = time.perf_counter()
    print(f"Data loaded in {t1 - t0:.6f} seconds")


    nn = CNN(config)


    xp.cuda.Stream.null.synchronize()
    t2 = time.perf_counter()

    nn.train(train_loader, config.epochs)

    xp.cuda.Stream.null.synchronize()
    print(f"Time taken: {time.perf_counter() - t2}s")

    # nn.save("emnist1.npz")

    loss, acc = nn.evaluate(test_loader)

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.2%}")