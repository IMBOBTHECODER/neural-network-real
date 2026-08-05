import numpy as np
import json
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
    hidden_layer: list = field(default_factory=lambda: [128])
    output_node: int = 47

    # Optimizer constants
    learning_rate: float = 1e-3
    initial_lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 5e-4
    dropout_rate: float = 0.25
    grad_clip_norm: float = 5.0

    # Training
    batch_size: int = 256
    epochs: int = 50


def save_model(cnn, path):
    if not path.endswith(".npz"):
        path += ".npz"

    # Build the same config_dict as before
    config_dict = {
        'input_shape': list(cnn.config.input_shape),
        'num_classes': cnn.config.num_classes,
        'hidden_layer': cnn.config.hidden_layer,
        'input_node': cnn.config.input_node,
        'output_node': cnn.config.output_node,
        'learning_rate': cnn.config.learning_rate,
        'initial_lr': cnn.config.initial_lr,
        'beta1': cnn.config.beta1,
        'beta2': cnn.config.beta2,
        'eps': cnn.config.eps,
        'weight_decay': cnn.config.weight_decay,
        'dropout_rate': cnn.config.dropout_rate,
        'grad_clip_norm': cnn.config.grad_clip_norm,
        'batch_size': cnn.config.batch_size,
        'epochs': cnn.config.epochs,
        'cnn_layer_desc': [describe_layer(l) for l in cnn.layers],
    }
    config_json = json.dumps(config_dict)   # one string, holds the whole architecture

    # Gather every layer's weight state
    layer_states = [layer.get_state() if hasattr(layer, "get_state") else None for layer in cnn.layers]
    mlp_state = cnn.mlp.get_state()   # you'd give NeuralNetwork a get_state()/load_state() pair too,
                                       # mirroring what you built for ConvLayer/BatchNormLayer

    np.savez(
        path,
        config_json=config_json,             # the "header"
        layer_states=np.array(layer_states, dtype=object),
        mlp_state=np.array(mlp_state, dtype=object),
    )


def load_model(path):
    if not path.endswith(".npz"):
        path += ".npz"

    data = np.load(path, allow_pickle=True)

    config_dict = json.loads(str(data['config_json']))   # str() unwraps the 0-d array back to a plain string

    config = Config(
        input_shape=tuple(config_dict['input_shape']),
        num_classes=config_dict['num_classes'],
        hidden_layer=config_dict['hidden_layer'],
        input_node=config_dict['input_node'],
        output_node=config_dict['output_node'],
        learning_rate=config_dict['learning_rate'],
        initial_lr=config_dict['initial_lr'],
        beta1=config_dict['beta1'],
        beta2=config_dict['beta2'],
        eps=config_dict['eps'],
        weight_decay=config_dict['weight_decay'],
        dropout_rate=config_dict['dropout_rate'],
        grad_clip_norm=config_dict['grad_clip_norm'],
        batch_size=config_dict['batch_size'],
        epochs=config_dict['epochs'],
    )
    config.cnn_layer = [build_layer_from_description(d) for d in config_dict['cnn_layer_desc']]

    cnn = CNN(config)   # architecture now matches exactly, fresh random weights

    layer_states = data['layer_states']
    for layer, state in zip(cnn.layers, layer_states):
        if state is not None and hasattr(layer, "load_state"):
            layer.load_state(state)

    cnn.mlp.load_state(data['mlp_state'].item())   # .item() unwraps the 0-d object array

    return cnn