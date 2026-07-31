import numpy as np
import time
from activation import LeakyReLU, LeakyReLU_derivative, softmax
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, X, Y, batch_size=64, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = np.max(Y) + 1

    def __iter__(self):
        indices = np.arange(len(self.X))

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]

            x = self.X[batch]

            y = np.eye(self.num_classes, dtype=np.float32)[self.Y[batch]]

            yield x, y

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size

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
                np.random.randn(self.layers[i], self.layers[i + 1]).astype(np.float32)
                * np.float32(np.sqrt(2.0 / self.layers[i]))
            )
            for i in range(self.size - 1)
        ]

        # Biases set to zero first
        self.biases = [
            np.zeros(self.layers[i + 1], dtype=np.float32)
            for i in range(self.size - 1)
        ]

        # Value after activation
        self.activations = [
            np.empty((batch_size, size), dtype=np.float32)
            for size in self.layers
        ]

        # Value before activation
        self.logits = [
            np.empty((batch_size, size), dtype=np.float32)
            for size in self.layers[1:]
        ]

        # Adam's momentum and variance
        self.weight_m = [
            np.zeros_like(w)
            for w in self.weights
        ]

        self.weight_v = [
            np.zeros_like(w)
            for w in self.weights
        ]

        self.bias_m = [
            np.zeros_like(b)
            for b in self.biases
        ]

        self.bias_v = [
            np.zeros_like(b)
            for b in self.biases
        ]

    def forward(self, inputs: np.ndarray):
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

    def backprop(self, prediction, target):
        self.training_step += 1

        np.clip(prediction, 1e-15, 1.0, out=prediction)

        loss = -np.mean(np.sum(target * np.log(prediction), axis=1))

        # Compute deltas (same order as weights)
        delta = [None] * len(self.weights)

        delta[-1] = prediction - target

        for l in range(len(self.weights) - 2, -1, -1):
            delta[l] = (
                delta[l + 1] @ self.weights[l + 1].T
            ) * LeakyReLU_derivative(self.logits[l])

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
            bias_gradient = np.sum(d, axis=0) / batch_size

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

            denom = np.sqrt(v_hat)
            denom += eps

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
                / denom
            )

        return loss

    def update_lr(self, epoch, total_epochs):
        min_lr = 1e-5

        self.learning_rate = (
            min_lr
            + (self.initial_lr - min_lr)
            * (1 + np.cos(np.pi * epoch / total_epochs))
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
            np.clip(prediction, 1e-15, 1.0, out=prediction)

            loss = -np.mean(np.sum(y_batch * np.log(prediction), axis=1))
            total_loss += loss

            pred_class = np.argmax(pred, axis=1)
            true_class = np.argmax(y_batch, axis=1)

            correct += np.sum(pred_class == true_class)
            total += len(x_batch)

        return total_loss / len(loader), correct / total

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

    def save(self, path: str ="model.npz"):
        if path[-4:] != ".npz":
            path += ".npz"

        np.savez(
            path,
            weights=np.array(self.weights, dtype=object),
            biases=np.array(self.biases, dtype=object),

            weights_m=np.array(self.weight_m, dtype=object),
            weights_v=np.array(self.weight_v, dtype=object),
            biases_m=np.array(self.bias_m, dtype=object),
            biases_v=np.array(self.bias_v, dtype=object),

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

        self.weights = list(data["weights"])
        self.biases = list(data["biases"])

        self.weight_m = list(data["weights_m"])
        self.weight_v = list(data["weights_v"])

        self.bias_m = list(data["biases_m"])
        self.bias_v = list(data["biases_v"])

        (
            self.learning_rate,
            self.initial_lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            self.batch_size
        ) = data["hyperparameter"]

        self.training_step = int(data["training_step"])
        self.beta1_pow = float(data["beta1_pow"])
        self.beta2_pow = float(data["beta2_pow"])


if __name__ == "__main__":
    # Setup and training

    # CONFIG
    INPUT_NODE = 784
    HIDDEN_LAYER = [256, 128]
    OUTPUT_NODE = 47

    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64

    BETA1 = 0.9 # ratio between past and
    BETA2 = 0.999
    EPS = 1e-8
    WEIGHT_DECAY = 1e-4

    t0 = time.perf_counter()
    emnist = fetch_openml(
        "EMNIST_balanced",
        version=1,
        as_frame=False
    )

    X = emnist.data.astype(np.float32) / 255.0
    Y = emnist.target.astype(np.int32)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    train_loader = DataLoader(
        X_train,
        Y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        X_test,
        Y_test,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    t1 = time.perf_counter()
    print(f"Data loaded in {t1 - t0:.6f} seconds")

    nn = NeuralNetwork(
        INPUT_NODE, HIDDEN_LAYER, OUTPUT_NODE,
        BATCH_SIZE, LEARNING_RATE,
        BETA1, BETA2, EPS, WEIGHT_DECAY)

    t2 = time.perf_counter()
    nn.train(train_loader, 50)
    print(f"\nTime taken: {time.perf_counter() - t2:.6f}s")

    nn.save("emnist1.npz")

    loss, acc = nn.evaluate(test_loader)

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.2%}")