# Copyright (C) 2026  Pham Tien Dat

import numpy as np
try:
    import cupy as xp
except ImportError:
    import numpy as xp

import time
import math
from activation import LeakyReLU, LeakyReLU_derivative, softmax
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field


@dataclass
class Config:
    # Data
    input_shape: tuple = (1, 28, 28)
    num_classes: int = 47

    # CNN architecture
    cnn_layer: list = field(default_factory=lambda: [])

    # MLP architecture
    input_node: int = 784
    hidden_layer: list = field(default_factory=lambda: [256, 128])
    output_node: int = 47

    # Optimizer constants
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-3

    # Training
    batch_size: int = 128
    epochs: int = 5


class DataLoader:
    def __init__(self, X, Y, batch_size=64, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = int(xp.max(Y).item()) + 1

    def __iter__(self):
        indices = xp.arange(len(self.X))

        if self.shuffle:
            xp.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]

            x = self.X[batch]

            y = xp.eye(self.num_classes, dtype=xp.float32)[self.Y[batch]]

            yield x, y

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size


class ConvLayer():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.kernels = xp.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(xp.float32) * xp.float32(math.sqrt(2.0 / (in_channels * kernel_size * kernel_size)))

        self.stride = stride

        # Adam state (per-layer, independent buffers)
        self.kernel_m = xp.zeros_like(self.kernels)
        self.kernel_v = xp.zeros_like(self.kernels)
        self.beta1_pow = 1.0
        self.beta2_pow = 1.0

        # Hyperparameter values injected by CNN.__init__, but give safe defaults
        self.learning_rate = 1e-3
        self.initial_lr = config.learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.weight_decay = 1e-3

    def _im2col(self, x, kH, kW, stride=1):
        # Rearranges every sliding-window patch into a layout ready to become
        # matrix rows, so convolution can be done as one matmul instead of loops.
        #
        # x: (N, C, H, W)  — batch, channels, height, width (NCHW)
        # returns col of shape (N, C, kH, kW, H_out, W_out) plus the output dims

        N, C, H, W = x.shape

        # Output spatial size — how many positions the kernel stops at
        H_out = (H - kH) // stride + 1
        W_out = (W - kW) // stride + 1

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


    def _col2im(self, d_col, N, C, kH, kW, H, W):
        stride = self.stride

        H_out = (H - kH) // stride + 1
        W_out = (W - kW) // stride + 1

        d_col = d_col.reshape(N, H_out, W_out, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)
        d_x = xp.zeros((N, C, H, W), dtype=d_col.dtype)

        for i in range(kH):
            i_end = i + stride * H_out
            for j in range(kW):
                j_end = j + stride * W_out
                d_x[:, :, i:i_end:stride, j:j_end:stride] += d_col[:, :, i, j, :, :]

        return d_x


    def forward(self, x):
        # Full vectorized conv: im2col -> reshape -> one matmul -> reshape back.
        #
        # x:       (N, C_in, H, W)
        # kernels: (C_out, C_in, kH, kW)
        # returns: (N, C_out, H_out, W_out)

        self.x_shape = x.shape
        N, C, H, W = self.x_shape
        C_out = self.kernels.shape[0]
        kH, kW = self.kernels.shape[2], self.kernels.shape[3]

        # Step 1: gather all patches (still in 6D "per kernel-cell" layout)
        col, self.H_out, self.W_out = self._im2col(x, kH, kW, self.stride)

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

        # INPUT GRADIENT — "how did each input pixel affect the loss?"
        # Rule: an input's gradient = the kernel it multiplied, times the output gradient.
        # Same idea as the MLP's  delta @ weights.T.
        # This comes out in flattened-patch form and still needs unflattening (next line).
        w_col = self.kernels.reshape(C_out, -1)         # (C_out, C*kH*kW)
        d_col = d_out_flat @ w_col                      # (N*H_out*W_out, C*kH*kW)

        # col2im: turn that flattened-patch gradient back into a real image.
        # Because one pixel appeared in several patches, its contributions get SUMMED.
        d_x = self._col2im(d_col, N, C, kH, kW, H, W)   # (N, C, H, W) — matches the input

        # Use the kernel gradient to actually update the kernels (Adam does this).
        # Note: this changes self.kernels; it does NOT touch d_x.
        self._adam_update(d_kernels)

        # Hand the input gradient back so the previous layer can keep going.
        return d_x

    def _adam_update(self, d_kernels):
        lr = self.learning_rate
        eps = self.eps
        wd = self.weight_decay
        beta1 = self.beta1
        beta2 = self.beta2

        self.beta1_pow *= beta1
        self.beta2_pow *= beta2

        bias_correction1 = 1 - self.beta1_pow
        bias_correction2 = 1 - self.beta2_pow

        # Adam
        self.kernel_m = (
            beta1 * self.kernel_m
            + (1 - beta1) * d_kernels
        )

        self.kernel_v = (
            beta2 * self.kernel_v
            + (1 - beta2) * d_kernels * d_kernels
        )

        m_hat = self.kernel_m / bias_correction1
        v_hat = self.kernel_v / bias_correction2

        denom = xp.sqrt(v_hat)
        denom += eps

        # AdamW
        self.kernels *= (1 - lr * wd)

        self.kernels -= (
            lr
            * m_hat
            / denom
        )

    def update_lr(self, epoch, total_epochs):
        min_lr = 1e-5
        self.learning_rate = (
            min_lr
            + (self.initial_lr - min_lr)
            * (1 + math.cos(math.pi * epoch / total_epochs))
            / 2
        )


class PoolLayer():
    def __init__(self, pool_size, stride, training):
        self.training = training
        self.pool_size = pool_size
        self.stride = stride

    def _pool_single(self, inputs, pool_size, stride):
        y = (inputs.shape[0] - pool_size) // stride + 1
        x = (inputs.shape[1] - pool_size) // stride + 1

        output = xp.zeros((y, x), dtype=xp.float32)

        mask = None
        if self.training:                      # only build mask when we'll need it
            mask = xp.zeros_like(inputs)

        stride_i = 0
        for i in range(y):
            stride_j = 0
            for j in range(x):
                patch = inputs[stride_i:stride_i+pool_size, stride_j:stride_j+pool_size]
                output[i, j] = xp.max(patch)

                if self.training:
                    flat = xp.argmax(patch)
                    pi, pj = divmod(int(flat), pool_size)
                    mask[stride_i + pi, stride_j + pj] = 1.0

                stride_j += stride
            stride_i += stride

        return output, mask

    def _pool_multi(self, inputs, pool_size, stride):
        channels = inputs.shape[0]
        outputs = []
        masks = []
        for c in range(channels):
            result, mask = self._pool_single(inputs[c], pool_size, stride)
            outputs.append(result)
            masks.append(mask)
        return xp.array(outputs), xp.array(masks)

    def forward(self, x):
        # x: (batch, channels, H, W), uses self.pool_size, self.stride
        batch_size = x.shape[0]
        outputs = []
        masks = []
        for b in range(batch_size):
            result, mask = self._pool_multi(x[b], self.pool_size, self.stride)
            outputs.append(result)
            masks.append(mask)

        self.mask = xp.array(masks)      # (batch, channels, H, W) — same shape as input x
        return xp.array(outputs)         # (batch, channels, H_out, W_out)
    
    def backward(self, grad):
        # grad: (batch, channels, out_height, out_width) — one gradient per pooling window
        # self.mask: (batch, channels, in_height, in_width) — 1.0 at each window's max position
        d_input = xp.zeros_like(self.mask)   # input-shaped, gradient w.r.t. this layer's input

        batch_size, num_channels, out_height, out_width = grad.shape
        pool_size = self.pool_size
        stride = self.stride

        for image in range(batch_size):
            for channel in range(num_channels):
                for out_row in range(out_height):
                    for out_col in range(out_width):
                        row_start = out_row * stride
                        col_start = out_col * stride

                        window = (
                            slice(row_start, row_start + pool_size),
                            slice(col_start, col_start + pool_size),
                        )

                        d_input[image, channel][window] += (
                            grad[image, channel, out_row, out_col]
                            * self.mask[image, channel][window]
                        )

        return d_input


class ActivationLayer():
    def forward(self, x):
        self.last_input = x        # cache for backward (activation needs pre-activation values)
        return LeakyReLU(x)

    def backward(self, grad):
        return grad * LeakyReLU_derivative(self.last_input)


class NeuralNetwork():
    def __init__(self, input_node: int, hidden_layer: list[int], output_node: int,
                 batch_size: int = 64, learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 1e-4):
        self.training_step = 0
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.initial_lr = learning_rate

        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        # Bias Correction
        self.beta1_pow = 1.0
        self.beta2_pow = 1.0

        # Can be switched to Xavier when using Sigmoid, Tanh, etc
        # He initiallisation

        self.layers = [input_node,*hidden_layer, output_node]
        self.size = len(self.layers)

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

        # Value after activation
        self.activations = [
            xp.empty((batch_size, size), dtype=xp.float32)
            for size in self.layers
        ]

        # Value before activation
        self.logits = [
            xp.empty((batch_size, size), dtype=xp.float32)
            for size in self.layers[1:]
        ]

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

        # Calculate value of each node (for hidden layer)
        for layer in range(1, self.size - 1):
            l = layer - 1

            x = self.activations[l]
            z = x @ self.weights[l] + self.biases[l]

            self.activations[layer] = LeakyReLU(z)
            self.logits[l] = z

        # Calculate value of each node (for final/output layer)
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

        final_delta = None
        if compute_final_gradient:
            final_delta = delta[0] @ self.weights[0].T

        lr = self.learning_rate
        eps = self.eps
        wd = self.weight_decay
        beta1 = self.beta1
        beta2 = self.beta2

        self.beta1_pow *= beta1
        self.beta2_pow *= beta2

        bias_correction1 = 1 - self.beta1_pow
        bias_correction2 = 1 - self.beta2_pow

        for l in range(len(self.weights)):
            d = delta[l]
            batch_size = d.shape[0]

            gradient = self.activations[l].T @ d / batch_size
            bias_gradient = xp.sum(d, axis=0) / batch_size

            # Adam
            self.weight_m[l] = (
                beta1 * self.weight_m[l]
                + (1 - beta1) * gradient
            )

            self.bias_m[l] = (
                beta1 * self.bias_m[l]
                + (1 - beta1) * bias_gradient
            )

            self.weight_v[l] = (
                beta2 * self.weight_v[l]
                + (1 - beta2) * gradient * gradient
            )

            self.bias_v[l] = (
                beta2 * self.bias_v[l]
                + (1 - beta2) * bias_gradient * bias_gradient
            )

            m_hat = self.weight_m[l] / bias_correction1
            v_hat = self.weight_v[l] / bias_correction2

            bias_m_hat = self.bias_m[l] / bias_correction1
            bias_v_hat = self.bias_v[l] / bias_correction2

            denom = xp.sqrt(v_hat)
            denom += eps

            bias_denom = xp.sqrt(bias_v_hat)
            bias_denom += eps

            # AdamW
            self.weights[l] *= (1 - lr * wd)

            self.weights[l] -= (
                lr
                * m_hat
                / denom
            )

            self.biases[l] -= (
                lr
                * bias_m_hat
                / bias_denom
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

    def save(self, path: str = "model.npz"):
        if not path.endswith(".npz"):
            path += ".npz"

        np.savez(
            path,

            weights=np.array([xp.asnumpy(w) for w in self.weights], dtype=object),
            biases=np.array([xp.asnumpy(b) for b in self.biases], dtype=object),

            weights_m=np.array([xp.asnumpy(m) for m in self.weight_m], dtype=object),
            weights_v=np.array([xp.asnumpy(v) for v in self.weight_v], dtype=object),

            biases_m=np.array([xp.asnumpy(m) for m in self.bias_m], dtype=object),
            biases_v=np.array([xp.asnumpy(v) for v in self.bias_v], dtype=object),

            hyperparameter=np.array([
                self.learning_rate,
                self.initial_lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                self.batch_size
            ], dtype=np.float32),

            training_step=self.training_step,
            beta1_pow=self.beta1_pow,
            beta2_pow=self.beta2_pow
        )

    def load(self, path: str = "model.npz"):
        if not path.endswith(".npz"):
            path += ".npz"

        data = np.load(path, allow_pickle=True)

        self.weights = [xp.asarray(w) for w in data["weights"]]
        self.biases = [xp.asarray(b) for b in data["biases"]]

        self.weight_m = [xp.asarray(m) for m in data["weights_m"]]
        self.weight_v = [xp.asarray(v) for v in data["weights_v"]]

        self.bias_m = [xp.asarray(m) for m in data["biases_m"]]
        self.bias_v = [xp.asarray(v) for v in data["biases_v"]]

        (
            self.learning_rate,
            self.initial_lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            self.batch_size,
        ) = data["hyperparameter"]

        self.learning_rate = float(self.learning_rate)
        self.initial_lr = float(self.initial_lr)
        self.beta1 = float(self.beta1)
        self.beta2 = float(self.beta2)
        self.eps = float(self.eps)
        self.weight_decay = float(self.weight_decay)
        self.batch_size = int(self.batch_size)

        self.training_step = int(data["training_step"])
        self.beta1_pow = float(data["beta1_pow"])
        self.beta2_pow = float(data["beta2_pow"])


class CNN():
    def __init__(self, config):
        self.layers = config.cnn_layer

        # Inject optimizer hyperparameters into every conv layer (once, from config)
        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                layer.learning_rate = config.learning_rate
                layer.beta1 = config.beta1
                layer.beta2 = config.beta2
                layer.eps = config.eps
                layer.weight_decay = config.weight_decay

        flatten_size = self._compute_flatten_size(config.input_shape)

        # The MLP tail — input_node is the FLATTEN SIZE, not 784
        self.mlp = NeuralNetwork(
            input_node=flatten_size,   # <-- must be the post-conv flatten size
            hidden_layer=config.hidden_layer,
            output_node=config.output_node,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    def forward(self, x):
        # x arrives as (batch, 784) flat from your DataLoader
        batch = x.shape[0]
        x = x.reshape(batch, 1, 28, 28)   # unflatten to image

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

    def _compute_flatten_size(self, input_shape):
        # input_shape: (channels, H, W) — e.g. (1, 28, 28)
        channels, height, width = input_shape

        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                k = layer.kernels.shape[2]          # kernel size (kH)
                stride = layer.stride
                height = (height - k) // stride + 1
                width  = (width  - k) // stride + 1
                channels = layer.kernels.shape[0]   # out_channels

            elif isinstance(layer, PoolLayer):
                p = layer.pool_size
                stride = layer.stride
                height = (height - p) // stride + 1
                width  = (width  - p) // stride + 1
                # channels unchanged for pooling

            # ActivationLayer: no shape change, skip

        return channels * height * width

if __name__ == "__main__":
    config = Config()
    config.cnn_layer = [
        ConvLayer(1, 8, 3),
        ActivationLayer(),
        PoolLayer(2, 2, training=True),
        ConvLayer(8, 16, 3),
        ActivationLayer(),
        PoolLayer(2, 2, training=True),
    ]

    t0 = time.perf_counter()
    emnist = fetch_openml(
        "EMNIST_balanced",
        version=1,
        as_frame=False
    )

    X = emnist.data.astype(np.float32) / 255.0
    Y = emnist.target.astype(np.int32)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    # --- TEMP: tiny subset for a first correctness run ---
    X_train, Y_train = X_train[:256], Y_train[:256]
    X_test,  Y_test  = X_test[:128],  Y_test[:128]
    # -----------------------------------------------------

    X_train, X_test = xp.asarray(X_train), xp.asarray(X_test)
    Y_train, Y_test = xp.asarray(Y_train), xp.asarray(Y_test)

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

    # loss, acc = nn.evaluate(test_loader)

    # print(f"Loss: {loss:.4f}")
    # print(f"Accuracy: {acc:.2%}")