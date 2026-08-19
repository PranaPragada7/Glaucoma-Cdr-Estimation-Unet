import numpy as np
import pytest
from PIL import Image

from glaucoma_cdr import inference
from glaucoma_cdr.inference import load_rgb_image, predict_array


class FakeSegmentationModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 256, 256, 3)
        assert verbose == 0

        prediction = np.zeros((1, 256, 256, 3), dtype=np.float32)
        prediction[0, 40:200, 50:210, 0] = 0.9
        prediction[0, 80:160, 90:170, 1] = 0.8
        prediction[0, 20:30, 20:30, 1] = 0.8
        return prediction


def test_predict_array_returns_nested_masks_and_vertical_cdr():
    image = np.zeros((256, 256, 3), dtype=np.float32)

    result = predict_array(FakeSegmentationModel(), image)

    assert result.disc_mask.sum() == 160 * 160
    assert result.cup_mask.sum() == 80 * 80
    assert result.cdr.disc_vertical_diameter == 160
    assert result.cdr.cup_vertical_diameter == 80
    assert result.cdr.ratio == 0.5


def test_predict_array_rejects_wrong_image_shape():
    image = np.zeros((128, 128, 3), dtype=np.float32)

    try:
        predict_array(FakeSegmentationModel(), image)
    except ValueError as exc:
        assert "shape (256, 256, 3)" in str(exc)
    else:
        raise AssertionError("expected a shape validation error")


def test_load_rgb_image_resizes_and_optionally_scales(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), (255, 128, 0)).save(path)

    original_range = load_rgb_image(path)
    scaled = load_rgb_image(path, scale_01=True)

    assert original_range.shape == (256, 256, 3)
    assert original_range.dtype == np.float32
    assert original_range.max() == 255
    assert scaled.max() == 1
    assert scaled.min() == 0


def test_load_rgb_image_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_rgb_image(tmp_path / "missing.png")


@pytest.mark.parametrize(
    "image, message",
    [
        (np.full((256, 256, 3), np.nan), "NaN"),
        (np.zeros((256, 256), dtype=np.float32), "shape"),
    ],
)
def test_predict_array_validates_input(image, message):
    with pytest.raises(ValueError, match=message):
        predict_array(FakeSegmentationModel(), image)


def test_predict_array_validates_threshold_and_model_output():
    image = np.zeros((256, 256, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="range"):
        predict_array(FakeSegmentationModel(), image, threshold=1)

    class WrongOutputModel:
        def predict(self, batch, verbose=0):
            return np.zeros((1, 128, 128, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="model output"):
        predict_array(WrongOutputModel(), image)


def test_infer_image_composes_model_loading_and_prediction(tmp_path, monkeypatch):
    path = tmp_path / "sample.png"
    Image.new("RGB", (256, 256), "black").save(path)
    model = FakeSegmentationModel()
    monkeypatch.setattr(inference, "load_trained_model", lambda weights: model)

    result = inference.infer_image(path, weights_path="weights.h5", threshold=0.6)

    assert result.cdr.ratio == 0.5
