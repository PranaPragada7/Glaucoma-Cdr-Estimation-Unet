"""Image preprocessing and U-Net inference helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .cdr import CDRMeasurement, estimate_vertical_cdr
from .model import DEFAULT_WEIGHTS, load_trained_model


@dataclass(frozen=True)
class InferenceResult:
    """Segmentation probabilities, masks, and calculated vertical CDR."""

    disc_probability: NDArray[np.float32]
    cup_probability: NDArray[np.float32]
    disc_mask: NDArray[np.bool_]
    cup_mask: NDArray[np.bool_]
    cdr: CDRMeasurement


def load_rgb_image(
    image_path: str | Path,
    *,
    target_size: tuple[int, int] = (256, 256),
    scale_01: bool = False,
) -> NDArray[np.float32]:
    """Load an RGB image and resize it to the model input dimensions."""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image was not found: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    if scale_01:
        array /= 255.0
    return array


def predict_array(
    model: Any,
    image: NDArray[np.float32],
    *,
    threshold: float = 0.5,
) -> InferenceResult:
    """Run a model on one preprocessed RGB image."""

    if not 0 <= threshold < 1:
        raise ValueError("threshold must be in the range [0, 1)")

    array = np.asarray(image, dtype=np.float32)
    if array.shape != (256, 256, 3):
        raise ValueError("image must have shape (256, 256, 3)")
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinite values")

    prediction = np.asarray(model.predict(array[None, ...], verbose=0))[0]
    if prediction.shape != (256, 256, 3):
        raise ValueError("model output must have shape (256, 256, 3)")

    disc_probability = prediction[:, :, 0].astype(np.float32, copy=False)
    cup_probability = prediction[:, :, 1].astype(np.float32, copy=False)
    disc_mask = disc_probability > threshold

    # Small segmentation errors can place cup pixels outside the disc. Restrict
    # the exported and measured cup mask while retaining raw probabilities.
    cup_mask = (cup_probability > threshold) & disc_mask
    cdr = estimate_vertical_cdr(disc_mask, cup_mask)
    return InferenceResult(
        disc_probability=disc_probability,
        cup_probability=cup_probability,
        disc_mask=disc_mask,
        cup_mask=cup_mask,
        cdr=cdr,
    )


def infer_image(
    image_path: str | Path,
    *,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    threshold: float = 0.5,
    scale_01: bool = False,
) -> InferenceResult:
    """Load the model and run end-to-end inference for one image."""

    model = load_trained_model(weights_path)
    image = load_rgb_image(image_path, scale_01=scale_01)
    return predict_array(model, image, threshold=threshold)
