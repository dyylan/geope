# Contributing

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
