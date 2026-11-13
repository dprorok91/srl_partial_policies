import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_policy_network(
    input_dim: int = 3,
    num_actions: int = 5,
    hidden_layers: list[int] | None = None,
) -> keras.Model:
    """Build a small policy network for Pendulum with a configurable stack of dense layers.

    The default setup intentionally uses a single dense output layer to match the
    project goal: one network layer and no hidden layers.
    """
    if hidden_layers is None:
        hidden_layers = []

    inputs = keras.Input(shape=(input_dim,), dtype=tf.float32, name="observation")
    x = inputs

    for units in hidden_layers:
        x = layers.Dense(units, activation="relu", dtype=tf.float32, name=f"hidden_{units}")(x)

    logits = layers.Dense(num_actions, dtype=tf.float32, name="action_logits")(x)
    model = keras.Model(inputs=inputs, outputs=logits, name="pendulum_policy")
    return model


def action_values_array() -> np.ndarray:
    return np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
