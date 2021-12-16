import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, Activation
import tensorflow.keras.activations as activations
from typing import Callable, Union


def clipped_exp(x: tf.Tensor) -> tf.Tensor:
    """Applies a exp activation function clipped at 1e-5 and 1e6.
    Arguments:
        x: Input tensor
    Returns:
        Tensor, output of clipped exp transformation.
    """
    return tf.clip_by_value(K.exp(x), 1e-5, 1e6)


def clipped_softplus(x: tf.Tensor) -> tf.Tensor:
    """Applies a exp activation function clipped at 1e-4 and 1e4.
    Arguments:
        x: Input tensor
    Returns:
        Tensor, output of clipped softplus transformation.
    """
    return tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


ACTIVATIONS = {
    "prelu": PReLU(),
    "relu": ReLU(),
    "leaky_relu": LeakyReLU(),
    "selu": Activation("selu"),
    "linear": Activation("linear"),
    "sigmoid": Activation("sigmoid"),
    "hard_sigmoid": Activation("hard_sigmoid"),
    "clipped_exp": Activation(clipped_exp),
    "clipped_softplus": Activation(clipped_softplus),
}


def get(identifier: Union[Callable, str]) -> Callable:
    """Returns activation function
    Arguments:
        identifier: Function or string
    Returns:
        Function corresponding to the input string or input function.
    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif identifier in ACTIVATIONS.keys():
        return ACTIVATIONS.get(identifier)
    else:
        return activations.get(identifier)
