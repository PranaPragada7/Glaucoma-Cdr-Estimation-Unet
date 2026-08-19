"""Cup-to-disc ratio measurements derived from segmentation masks."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CDRMeasurement:
    """Vertical cup and disc measurements in pixels."""

    cup_vertical_diameter: int
    disc_vertical_diameter: int
    ratio: float


def _binary_mask(mask: ArrayLike, threshold: float, name: str) -> NDArray[np.bool_]:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError(f"{name} must contain numeric or boolean values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array > threshold


def vertical_diameter(mask: ArrayLike, threshold: float = 0.5) -> int:
    """Return the inclusive vertical extent of the foreground in pixels.

    An empty mask has a diameter of zero.
    """

    if not 0 <= threshold < 1:
        raise ValueError("threshold must be in the range [0, 1)")

    binary = _binary_mask(mask, threshold, "mask")
    occupied_rows = np.flatnonzero(binary.any(axis=1))
    if occupied_rows.size == 0:
        return 0
    return int(occupied_rows[-1] - occupied_rows[0] + 1)


def estimate_vertical_cdr(
    disc_mask: ArrayLike,
    cup_mask: ArrayLike,
    *,
    threshold: float = 0.5,
    require_cup_within_disc: bool = True,
) -> CDRMeasurement:
    """Calculate vertical CDR from optic disc and cup masks.

    Args:
        disc_mask: Two-dimensional disc probability or binary mask.
        cup_mask: Two-dimensional cup probability or binary mask.
        threshold: Values above this threshold are foreground.
        require_cup_within_disc: Reject cup pixels outside the disc mask.

    Raises:
        ValueError: If shapes differ, the disc is empty, or containment fails.
    """

    if not 0 <= threshold < 1:
        raise ValueError("threshold must be in the range [0, 1)")

    disc = _binary_mask(disc_mask, threshold, "disc_mask")
    cup = _binary_mask(cup_mask, threshold, "cup_mask")
    if disc.shape != cup.shape:
        raise ValueError("disc_mask and cup_mask must have the same shape")

    if require_cup_within_disc and np.any(cup & ~disc):
        raise ValueError("cup_mask contains foreground outside disc_mask")

    disc_diameter = vertical_diameter(disc)
    if disc_diameter == 0:
        raise ValueError("disc_mask is empty; CDR is undefined")

    cup_diameter = vertical_diameter(cup)
    return CDRMeasurement(
        cup_vertical_diameter=cup_diameter,
        disc_vertical_diameter=disc_diameter,
        ratio=cup_diameter / disc_diameter,
    )
