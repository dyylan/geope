# Contributing

## Git Workflow

### Reporting Issues

Before opening a new issue, search the [issue tracker](../../issues) to see if it has already been reported. If it has, add a comment with any additional context. If it has not, open a new issue with a clear title and description, including steps to reproduce if applicable.

### Contributing Code

1. **Fork** the repository and clone your fork locally.
2. **Install the development tooling** and the [pre-commit](https://pre-commit.com) hooks. The hooks run [Black](https://black.readthedocs.io) on every commit so contributions are consistently formatted before they reach CI:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```
   You can format the whole repository on demand with `pre-commit run --all-files`.
3. **Create a branch** for your change:
   ```bash
   git checkout -b my-feature-or-fix
   ```
4. **Make your changes** and ensure existing tests still pass:
   ```bash
   pytest
   ```
5. Run black to format the code:
   ```bash
   black .
   ```
6. **Commit** your changes with a clear message:
   ```bash
   git commit -m "Short description of what changed"
   ```
7. **Push** to your fork and open a **pull request** against `main`. Reference the related issue (e.g. `Closes #42`) in the PR description.

## Benchmarks

Performance benchmarks for the hand-written JAX primitives (the manual
Jacobian and the `dexpm` block-exponential derivative) versus naive
`jax.jacobian` autodiff live in `benchmarks/`. They use
[pytest-benchmark](https://pytest-benchmark.readthedocs.io) and are **excluded
from the default `pytest` run and from CI** — run them deliberately:

```bash
pip install -e ".[dev]"
pytest benchmarks/ --benchmark-group-by=param --benchmark-columns=mean,median,rounds
```

Each file separates steady-state *execution* benchmarks from *compilation*
benchmarks. Grouping by `param` places the competing implementations for each
problem size side by side.

## Publish Package

Releases are automated by GitHub Actions ([.github/workflows/release.yml](.github/workflows/release.yml)): pushing a version tag builds the artifacts, runs the distribution-test matrix (Python 3.11–3.13, wheel + sdist), stages to TestPyPI and re-verifies the install, then publishes to PyPI.

### Automated release (recommended)

1. **Bump the version** in [pyproject.toml](pyproject.toml) — it must match the tag you push (`0.0.3` ↔ `v0.0.3`), or PyPI rejects the upload.
2. **Commit, tag, and push:**
   ```bash
   git commit -am "Release v0.0.3"
   git tag v0.0.3
   git push origin main --tags
   ```
3. The workflow then runs, in order: build + `twine check` → install-and-test across Python 3.11–3.13 (wheel and sdist) → publish to TestPyPI and verify a clean install → publish to PyPI.

For a dry run, trigger **Actions → Release → Run workflow**: it builds and runs the install-test matrix but skips every publish step (those are gated on a `v*` tag).

### Manual release (fallback)

Publish from a local machine when CI is unavailable.

#### Prerequisites

```bash
python -m pip install build twine
```

#### 1. Bump the version

Update `version` in [pyproject.toml](pyproject.toml) before each release:

```toml
[project]
version = "0.x.y"
```

#### 2. Build distributions

```bash
python -m build
```

This produces two artifacts in `dist/`:
- `geope-<version>-py3-none-any.whl` — the wheel
- `geope-<version>.tar.gz` — the source distribution

#### 3. Validate

```bash
python -m twine check dist/*
```

Fix any reported errors before uploading.

#### 4. Upload to TestPyPI

You need a [TestPyPI](https://test.pypi.org) account. Create an API token and either export it or add it to `~/.pypirc`.

```bash
python -m twine upload --repository testpypi dist/*
```

#### 5. Test the install from TestPyPI

Install the pinned dependencies from real PyPI **first**, then install only GEOPE from TestPyPI with `--no-deps`:

```bash
# dependencies (pinned in pyproject.toml) from real PyPI
pip install $(python -c "import tomllib;print(' '.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")
# then GEOPE alone from TestPyPI, without re-resolving dependencies
pip install --no-deps --index-url https://test.pypi.org/simple/ "geope==<version>"
```

#### 6. Upload to PyPI (when satisfied)

```bash
python -m twine upload dist/*
```
