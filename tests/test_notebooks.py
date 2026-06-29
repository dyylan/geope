"""Execute the example notebooks end-to-end to guard against example rot.

Every notebook under ``docs/examples/`` is re-run from a clean state with
nbconvert's :class:`~nbconvert.preprocessors.ExecutePreprocessor`. A test
fails if any cell raises, surfacing the offending cell's traceback via
nbclient's ``CellExecutionError``.
"""

from pathlib import Path

import pytest

# Notebook deps live in the [dev] extra and are absent from the standard test
# job (which installs only `pip install -e .`). The `notebooks` marker keeps
# these tests deselected there, but collection still imports this module, so
# skip cleanly rather than erroring when the deps are missing.
nbformat = pytest.importorskip("nbformat")
ExecutePreprocessor = pytest.importorskip("nbconvert.preprocessors").ExecutePreprocessor

# docs/examples relative to the repo root (mirrors the path idiom in conftest.py).
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "examples"
EXAMPLE_NOTEBOOKS = sorted(EXAMPLES_DIR.glob("*.ipynb"))

# Generous per-notebook ceiling so a hung cell fails rather than blocks forever.
NOTEBOOK_TIMEOUT = 900

# Heavy: excluded from the standard suite, run only in the release pipeline.
pytestmark = pytest.mark.notebooks


@pytest.mark.parametrize("notebook_path", EXAMPLE_NOTEBOOKS, ids=lambda p: p.name)
def test_example_notebook_runs(notebook_path):
    """Execute ``notebook_path`` and assert that no cell raises."""
    nb = nbformat.read(notebook_path, as_version=4)
    ep = ExecutePreprocessor(timeout=NOTEBOOK_TIMEOUT, kernel_name="python3")
    # Run with the notebook's own directory as cwd so relative paths resolve.
    ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
