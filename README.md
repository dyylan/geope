# GEOPE

[![Tests](https://github.com/dyylan/geope/actions/workflows/tests.yml/badge.svg)](https://github.com/dyylan/geope/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/geope)](https://pypi.org/project/geope/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://dyylan.github.io/geope)

**GEOPE** (Geodesic Pulse Engineering) is a Python library for quantum gate synthesis using Lie algebraic and geodesic optimisation methods built on [JAX](https://github.com/jax-ml/jax).

## Installation

Install the latest release from [PyPI](https://pypi.org/project/geope/) (Python 3.11+):

```bash
pip install geope
```

### From source (development)

```bash
git clone https://github.com/dyylan/geope.git
cd geope
pip install -e ".[dev]"
```

## Documentation

Full API documentation is available at [dyylan.github.io/geope](https://dyylan.github.io/geope).

## References

The methods implemented in GEOPE are described in:

- D. Lewis, R. Wiersema, and S. Bose, *Quantum Optimal Control with Geodesic Pulse Engineering*, [arXiv:2508.16029](https://arxiv.org/abs/2508.16029) (2025) — the geodesic pulse engineering method.
- D. Lewis and R. Wiersema, *Pulse Quality Optimisation in Quantum Optimal Control*, [arXiv:2604.25768](https://arxiv.org/abs/2604.25768) (2026) — the Gecko pulse quality optimisation method.