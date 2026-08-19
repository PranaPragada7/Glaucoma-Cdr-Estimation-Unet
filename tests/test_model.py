from pathlib import Path

import pytest

from glaucoma_cdr.model import DEFAULT_WEIGHTS, build_unet, load_trained_model


@pytest.mark.integration
def test_unet_output_shape_and_committed_weights_are_compatible():
    assert Path(DEFAULT_WEIGHTS).is_file()

    model = load_trained_model()

    assert model.input_shape == (None, 256, 256, 3)
    assert model.output_shape == (None, 256, 256, 3)


def test_invalid_input_shape_is_rejected_before_tensorflow_build():
    with pytest.raises(ValueError, match="three positive dimensions"):
        build_unet((256, 256, 0))
