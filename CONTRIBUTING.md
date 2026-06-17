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
5. **Commit** your changes with a clear message:
   ```bash
   git commit -m "Short description of what changed"
   ```
6. **Push** to your fork and open a **pull request** against `main`. Reference the related issue (e.g. `Closes #42`) in the PR description.

## Publish Package

### Prerequisites

```bash
python -m pip install build twine
```

### 1. Bump the version

Update `version` in [pyproject.toml](pyproject.toml) before each release:

```toml
[project]
version = "0.x.y"
```

### 2. Build distributions

```bash
python -m build
```

This produces two artifacts in `dist/`:
- `GEOPE-<version>-py3-none-any.whl` — the wheel
- `GEOPE-<version>.tar.gz` — the source distribution

### 3. Validate

```bash
python -m twine check dist/*
```

Fix any reported errors before uploading.

### 4. Upload to TestPyPI

You need a [TestPyPI](https://test.pypi.org) account. Create an API token and either export it or add it to `~/.pypirc`.

```bash
python -m twine upload --repository testpypi dist/*
```

### 5. Test the install from TestPyPI

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  "GEOPE==<version>"
```

The `--extra-index-url` flag lets pip resolve pinned dependencies (e.g. `jax`) from the real PyPI index.

### 6. Upload to PyPI (when satisfied)

```bash
python -m twine upload dist/*
```
