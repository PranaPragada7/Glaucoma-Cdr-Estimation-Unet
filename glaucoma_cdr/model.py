"""TensorFlow U-Net architecture used by the original training notebook."""

from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "model_weights.h5"


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required for model inference. Install requirements.txt."
        ) from exc
    return tf


def build_unet(input_shape: tuple[int, int, int] = (256, 256, 3)) -> Any:
    """Build the three-channel U-Net recorded in the project notebook."""

    if len(input_shape) != 3 or any(dimension <= 0 for dimension in input_shape):
        raise ValueError("input_shape must contain three positive dimensions")

    tf = _tensorflow()
    layers = tf.keras.layers

    def convolution_block(inputs: Any, filters: int) -> Any:
        x = layers.Conv2D(filters, 3, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    inputs = tf.keras.Input(shape=input_shape)

    encoder_1 = convolution_block(inputs, 16)
    pool_1 = layers.MaxPooling2D(pool_size=2)(encoder_1)
    encoder_2 = convolution_block(pool_1, 32)
    pool_2 = layers.MaxPooling2D(pool_size=2)(encoder_2)
    encoder_3 = convolution_block(pool_2, 64)
    pool_3 = layers.MaxPooling2D(pool_size=2)(encoder_3)
    encoder_4 = convolution_block(pool_3, 128)
    pool_4 = layers.MaxPooling2D(pool_size=2)(encoder_4)

    bottleneck = convolution_block(pool_4, 256)

    decoder_4 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(bottleneck)
    decoder_4 = layers.concatenate([encoder_4, decoder_4], axis=3)
    decoder_4 = convolution_block(decoder_4, 128)

    decoder_3 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(decoder_4)
    decoder_3 = layers.concatenate([encoder_3, decoder_3], axis=3)
    decoder_3 = convolution_block(decoder_3, 64)

    decoder_2 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(decoder_3)
    decoder_2 = layers.concatenate([encoder_2, decoder_2], axis=3)
    decoder_2 = convolution_block(decoder_2, 32)

    decoder_1 = layers.Conv2DTranspose(16, 2, strides=2, padding="same")(decoder_2)
    decoder_1 = layers.concatenate([encoder_1, decoder_1], axis=3)
    decoder_1 = convolution_block(decoder_1, 16)

    outputs = layers.Conv2D(3, 1, activation="sigmoid")(decoder_1)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="glaucoma_cdr_unet")


def load_trained_model(
    weights_path: str | Path = DEFAULT_WEIGHTS,
    input_shape: tuple[int, int, int] = (256, 256, 3),
) -> Any:
    """Build the U-Net and load the committed legacy HDF5 weights."""

    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights were not found: {path}")
    model = build_unet(input_shape)
    model.load_weights(path)
    return model
