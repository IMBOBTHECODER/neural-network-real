from dataclasses import dataclass, field

@dataclass
class Config:
    # Data
    input_shape: tuple = (3, 32, 32)
    num_classes: int = 10

    # CNN architecture
    cnn_layer: list = field(default_factory=lambda: [])

    # MLP architecture
    input_node: int = 3072 # Unused
    hidden_layer: list = field(default_factory=lambda: [256])
    output_node: int = 10

    # Optimizer constants
    learning_rate: float = 1e-2 # Unused
    initial_lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 5e-4
    dropout_rate: float = 0.35
    grad_clip_norm: float = 5.0

    # Training
    batch_size: int = 256
    epochs: int = 50