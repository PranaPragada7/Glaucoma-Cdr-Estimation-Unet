import numpy as np
import pytest

from glaucoma_cdr.cdr import estimate_vertical_cdr, vertical_diameter


def test_vertical_diameter_uses_inclusive_foreground_extent():
    mask = np.zeros((12, 10), dtype=np.uint8)
    mask[2:9, 3:7] = 1

    assert vertical_diameter(mask) == 7


def test_vertical_cdr_for_nested_masks():
    disc = np.zeros((20, 20), dtype=np.uint8)
    cup = np.zeros_like(disc)
    disc[2:18, 4:16] = 1
    cup[6:14, 7:13] = 1

    result = estimate_vertical_cdr(disc, cup)

    assert result.disc_vertical_diameter == 16
    assert result.cup_vertical_diameter == 8
    assert result.ratio == pytest.approx(0.5)


def test_empty_cup_has_zero_cdr():
    disc = np.ones((8, 8), dtype=np.uint8)
    cup = np.zeros_like(disc)

    result = estimate_vertical_cdr(disc, cup)

    assert result.cup_vertical_diameter == 0
    assert result.ratio == 0


def test_empty_disc_is_rejected():
    empty = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="disc_mask is empty"):
        estimate_vertical_cdr(empty, empty)


def test_cup_outside_disc_is_rejected():
    disc = np.zeros((8, 8), dtype=np.uint8)
    cup = np.zeros_like(disc)
    disc[2:6, 2:6] = 1
    cup[1:4, 3:5] = 1

    with pytest.raises(ValueError, match="outside disc_mask"):
        estimate_vertical_cdr(disc, cup)


def test_mask_shapes_must_match():
    with pytest.raises(ValueError, match="same shape"):
        estimate_vertical_cdr(np.ones((4, 4)), np.ones((3, 4)))


def test_vertical_diameter_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="range"):
        vertical_diameter(np.ones((4, 4)), threshold=-0.1)
