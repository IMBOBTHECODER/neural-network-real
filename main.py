# Copyright (C) 2026  Pham Tien Dat

import torch
from torch import nn
from torch.utils.data import  DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time
import math
from config import Config
from data_loader import CIFAR10Dataset, get_cifar10_data_torch


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.kernels = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size,
            dtype=torch.float32, device=device
        ) * math.sqrt(2.0 / (in_channels * kernel_size * kernel_size)))

        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32, device=device))   # one per output channel

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.kernels, bias=self.bias, stride=self.stride, padding=self.padding)


class PoolLayer(nn.Module):
    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=stride)

    def forward(self, x):
        return self.pool(x)


class ActivationLayer(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        return self.activation(x)


class BatchNormLayer(nn.Module):
    def __init__(self, num_channels, momentum, training=False):
        super().__init__()

        self.num_channels = num_channels
        self.momentum = momentum
        self.training = training

        # learnable parameters
        self.gamma = torch.nn.Parameter(torch.ones(num_channels, device=device))
        self.beta = torch.nn.Parameter(torch.zeros(num_channels, device=device))

        # non-learned running statistics (eval-time use)
        self.register_buffer("running_mean", torch.zeros(num_channels, dtype=torch.float32, device=device))
        self.register_buffer("running_var", torch.ones(num_channels, dtype=torch.float32, device=device))
        
        self.eps = 1e-8

    def forward(self, x):
        if self.training:
            mean = torch.mean(x, axis=(0, 2, 3), keepdims=True)
            mean_sq = torch.mean(x * x, axis=(0, 2, 3), keepdims=True)
            var = mean_sq - mean * mean

            momentum = self.momentum
            self.running_mean = momentum * self.running_mean + (1 - momentum) * mean.reshape(-1)
            self.running_var = momentum * self.running_var + (1 - momentum) * var.reshape(-1)
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)

        gamma_r = self.gamma.reshape(1, -1, 1, 1)
        beta_r = self.beta.reshape(1, -1, 1, 1)

        out = gamma_r * (x - mean) / torch.sqrt(var + self.eps) + beta_r

        return out


class ResidualBlock(nn.Module):
    def __init__(self, sub_layers):
        super().__init__()

        self.sub_layers = nn.ModuleList(sub_layers)   # was: self.sub_layers = sub_layers
        self.final_activation = ActivationLayer()

    def forward(self, x):
        out = x
        for layer in self.sub_layers:
            out = layer.forward(out)
        return self.final_activation.forward(out + x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.35):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        # NOTE: no softmax here — see explanation below

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv_stack = config.cnn_layer

        flatten_size = self._compute_flatten_size(config.input_shape)
        self.mlp = MLP(flatten_size, config.hidden_layers, config.output_node, config.dropout_rate)


    def forward(self, x):
        x = self.conv_stack(x)
        x = x.flatten(1)
        return self.mlp(x)

    def _compute_flatten_size(self, input_shape):
        # input_shape: (channels, H, W) — e.g. (3, 32, 32)
        self.eval()   # dummy pass shouldn't affect batchnorm running stats or dropout
        with torch.no_grad():   # no need to track gradients for this throwaway pass
            dummy = torch.zeros(1, *input_shape, device=device)   # batch size 1
            out = self.conv_stack(dummy)           # run through your actual conv/residual/pool layers
        self.train()   # restore training mode
        return out.numel() // out.shape[0]   # total elements, divided by batch size


def train(model, train_loader, optimizer, scheduler, loss_fn, epochs, device,
          checkpoint_path=None, checkpoint_every=5, config=None):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}")

        if checkpoint_path and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, config)
            print(f"  Saved checkpoint at epoch {epoch}")

def evaluate(model, loader, loss_fn, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)              # raw logits
            loss = loss_fn(output, y_batch)
            total_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    model.train()   # restore training mode for whoever calls this next
    return total_loss / len(loader), correct / total


def save_checkpoint(path, model, optimizer, scheduler, epoch, config):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']


if __name__ == "__main__":
    config = Config()
    config.cnn_layer = nn.Sequential(
        ConvLayer(3, 32, 3, padding=1), BatchNormLayer(32, momentum=0.9), nn.LeakyReLU(),
        ResidualBlock([
            ConvLayer(32, 32, 3, padding=1), BatchNormLayer(32, momentum=0.9), ActivationLayer(),
            ConvLayer(32, 32, 3, padding=1), BatchNormLayer(32, momentum=0.9),
        ]),
        ResidualBlock([
            ConvLayer(32, 32, 3, padding=1), BatchNormLayer(32, momentum=0.9), ActivationLayer(),
            ConvLayer(32, 32, 3, padding=1), BatchNormLayer(32, momentum=0.9),
        ]),
        nn.MaxPool2d(2, 2),
        ConvLayer(32, 64, 3, padding=1), BatchNormLayer(64, momentum=0.9), nn.LeakyReLU(),
        ResidualBlock([
            ConvLayer(64, 64, 3, padding=1), BatchNormLayer(64, momentum=0.9), ActivationLayer(),
            ConvLayer(64, 64, 3, padding=1), BatchNormLayer(64, momentum=0.9),
        ]),
        ResidualBlock([
            ConvLayer(64, 64, 3, padding=1), BatchNormLayer(64, momentum=0.9), ActivationLayer(),
            ConvLayer(64, 64, 3, padding=1), BatchNormLayer(64, momentum=0.9),
        ]),
        nn.MaxPool2d(2, 2),
    )

    t0 = time.perf_counter()
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=config.input_shape[1], padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    X_train, X_test, Y_train, Y_test = get_cifar10_data_torch('/kaggle/input/datasets/pankrzysiu/cifar10-python/cifar-10-batches-py')

    train_dataset = CIFAR10Dataset(X_train, Y_train, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = CIFAR10Dataset(X_test, Y_test, transform=None)   # no augmentation at eval
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle =False)

    t1 = time.perf_counter()
    print(f"Data loaded in {t1 - t0:.6f} seconds")


    model = CNN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)

    # start_epoch = load_checkpoint("cifar10_checkpoint.pt", model, optimizer, scheduler, device)

    train(model, train_loader, optimizer, scheduler, config.loss_fn,
        epochs=config.epochs, device=device,
        checkpoint_path="cifar10_checkpoint.pt", checkpoint_every=5, config=config)

    # after training, save a final checkpoint too
    save_checkpoint("cifar10_final.pt", model, optimizer, scheduler, config.epochs - 1, config)


    print(f"Time taken: {time.perf_counter() - t1}s")

    train_eval_dataset = CIFAR10Dataset(X_train, Y_train, transform=None)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=config.batch_size, shuffle=False)

    train_loss, train_acc = evaluate(model, train_eval_loader, config.loss_fn, device)
    loss, acc = evaluate(model, test_loader, config.loss_fn, device)

    print("Test:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.2%}\n")

    print("Train:")
    print(f"Loss: {train_loss:.4f}") 
    print(f"Accuracy: {train_acc:.2%}")