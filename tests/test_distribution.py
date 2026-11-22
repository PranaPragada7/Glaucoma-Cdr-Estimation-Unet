"""Check that a non-editable distribution retains its default inference asset."""

import hashlib
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_built_wheel_contains_usable_default_weights(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("*.whl"))
    installed = tmp_path / "installed"
    with ZipFile(wheel) as archive:
        archive.extractall(installed)

    # An isolated interpreter and a separate working directory ensure the
    # editable source checkout cannot conceal a missing distribution asset.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import hashlib, sys; "
            "sys.path.insert(0, sys.argv[1]); "
            "from glaucoma_cdr.model import DEFAULT_WEIGHTS, load_trained_model; "
            "model = load_trained_model(); "
            "assert model.input_shape == (None, 256, 256, 3); "
            "assert model.output_shape == (None, 256, 256, 3); "
            "print(hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest())",
            str(installed),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    source_weights = project_root / "glaucoma_cdr" / "weights" / "model_weights.h5"
    expected = hashlib.sha256(source_weights.read_bytes()).hexdigest()
    assert result.stdout.strip() == expected
