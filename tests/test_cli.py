"""Tests for the command-line export workflow."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from glaucoma_cdr import cli
from glaucoma_cdr.cdr import CDRMeasurement


def test_cli_writes_masks_and_summary(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "retina.png"
    Image.new("RGB", (16, 16), "black").save(image_path)
    output_dir = tmp_path / "results"
    disc = np.zeros((256, 256), dtype=bool)
    cup = np.zeros_like(disc)
    disc[40:200, 40:200] = True
    cup[80:160, 80:160] = True
    result = SimpleNamespace(
        disc_mask=disc,
        cup_mask=cup,
        cdr=CDRMeasurement(
            cup_vertical_diameter=80,
            disc_vertical_diameter=160,
            ratio=0.5,
        ),
    )
    calls = {}

    def fake_infer(path, **kwargs):
        calls.update(path=path, **kwargs)
        return result

    monkeypatch.setattr(cli, "infer_image", fake_infer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "glaucoma-cdr",
            "--image",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--threshold",
            "0.6",
            "--scale-01",
        ],
    )

    cli.main()

    summary = json.loads((output_dir / "retina_cdr.json").read_text("utf-8"))
    assert summary["vertical_cdr"] == 0.5
    assert summary["disclaimer"].startswith("Research use only")
    assert (output_dir / "retina_disc_mask.png").is_file()
    assert (output_dir / "retina_cup_mask.png").is_file()
    assert calls["threshold"] == 0.6
    assert calls["scale_01"] is True
    assert "Vertical CDR: 0.500" in capsys.readouterr().out


@pytest.mark.parametrize("threshold", ["-0.1", "1"])
def test_cli_rejects_invalid_threshold(monkeypatch, threshold):
    monkeypatch.setattr(
        sys,
        "argv",
        ["glaucoma-cdr", "--image", "image.png", "--threshold", threshold],
    )
    with pytest.raises(SystemExit, match="threshold"):
        cli.main()
