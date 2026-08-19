import numpy as np

from glaucoma_cdr.inference import predict_array


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
