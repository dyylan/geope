from __future__ import annotations

import numpy as np


def _default_log_fn(geope) -> dict:
    """Default per-step log row: with five standard data columns.

    Reads the running optimiser and returns a dict with ``parameters``
    (a full-basis snapshot of the current params), ``fidelities``,
    ``infidelities``, the transient ``step_sizes`` and an integer
    ``steps`` counter derived from the number of rows already recorded
    (so no persistent step counter is needed anywhere).

    ``fidelities`` is left as whatever the engine produced (a JAX scalar)
    rather than cast to ``float``, preserving today's values.
    """
    p = geope.params
    n = len(geope.history) if geope.history is not None else 0
    return {
        "parameters":   np.array(p.parameters),   # snapshot copy of current params (full-basis)
        "fidelities":   p.fidelity,
        "infidelities": 1 - p.fidelity,
        "step_sizes":   geope.step_size,           # transient: last line-search step size
        "steps":        n,                         # 0, 1, 2, ... derived from log length
    }


class History:
    """Opt-in, configurable run log for a `Geope` optimisation.

    By default records today's five columns (``parameters``,
    ``fidelities``, ``infidelities``, ``step_sizes``, ``steps``) every
    step. Pass ``log_fn`` to record arbitrary per-step values instead;
    it receives the running `Geope` and returns a dict of
    ``column -> value``.

    Columns are exposed as attributes and items, so ``history.fidelities``
    and ``history["fidelities"]`` are the same list. Best-over-trajectory
    helpers (``best_fidelity``, ``best_parameters``,
    ``best_basis_coefficients``, ``to_dict``) are available when the
    default ``fidelities``/``parameters`` columns are present and degrade
    to ``None``/``{}`` otherwise.
    """

    def __init__(self, log_fn=None) -> None:
        # ``logs`` must be set first so ``__getattr__`` can guard on it.
        self.logs: dict = {}
        self.log_fn = log_fn or _default_log_fn
        self.params = None   # back-ref to the Parameters, set by Geope

    def record(self, geope) -> dict:
        """Append one row by calling ``log_fn(geope)``; returns the row."""
        row = self.log_fn(geope)
        for k, v in row.items():
            self.logs.setdefault(k, []).append(v)
        return row

    def reset(self) -> None:
        """Drop all recorded rows."""
        self.logs = {}

    def __len__(self) -> int:
        for col in self.logs.values():
            return len(col)
        return 0

    def __getitem__(self, key):
        return self.logs[key]

    def __contains__(self, key) -> bool:
        return key in self.logs

    def keys(self):
        return self.logs.keys()

    def __getattr__(self, name):
        # Only reached when ``name`` is not a real attribute. Guard via
        # ``object.__getattribute__`` so that, before ``logs`` exists
        # (e.g. during unpickling/deepcopy), we raise a plain
        # ``AttributeError`` instead of recursing.
        logs = object.__getattribute__(self, "logs")
        if name in logs:
            return logs[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute or log column "
            f"{name!r}. Available columns: {sorted(logs)}"
        )

    def to_dataframe(self):
        """Return the recorded logs as a ``pandas.DataFrame``."""
        import pandas as pd
        return pd.DataFrame(self.logs)

    # --- Best over the trajectory -----------------------------------------
    @property
    def best_fidelity(self):
        """Maximum recorded fidelity, or ``None`` if none recorded."""
        fids = self.logs.get("fidelities")
        if not fids:
            return None
        return max(fids)

    @property
    def best_parameters(self):
        """Parameter snapshot at the step of maximum fidelity, or ``None``."""
        fids = self.logs.get("fidelities")
        params = self.logs.get("parameters")
        if not fids or not params:
            return None
        idx = int(np.argmax(fids))
        return params[idx]

    @property
    def best_basis_coefficients(self):
        """Best parameters mapped through ``param_transform`` if set.

        Returns ``None`` when there is no best parameter set. Requires the
        back-reference ``self.params`` (set automatically when a ``History``
        is passed to ``Geope``); raises ``ValueError`` if columns exist but
        the back-reference is missing.
        """
        bp = self.best_parameters
        if bp is None:
            return None
        if self.params is None:
            raise ValueError(
                "History.best_basis_coefficients needs a back-reference to a "
                "Parameters object (history.params); it is set automatically "
                "when a History is passed to Geope."
            )
        if self.params.param_transform is not None:
            import jax
            return np.array(jax.vmap(self.params.param_transform)(bp))
        return bp

    def to_dict(self) -> dict:
        """Export the best basis coefficients as a control-style dict.

        Mirrors ``Parameters.to_dict`` but over the best parameters seen
        across the trajectory. Returns ``{}`` when no best is available.
        """
        coeffs = self.best_basis_coefficients
        if coeffs is None:
            return {}
        params = self.params
        proj_indices = np.array(params.projected_basis.overlap(params.basis), dtype=bool)
        proj_coeffs = coeffs[0][proj_indices] if coeffs.ndim > 1 else coeffs[proj_indices]

        result: dict = {}
        for label, value in zip(params.projected_basis.labels, proj_coeffs):
            new_label = ""
            qubits = []
            for i, c in enumerate(label):
                if c != "I":
                    new_label += c.lower()
                    qubits.append(i + 1)
            key = tuple(qubits) if len(qubits) > 1 else qubits[0]
            if key not in result:
                result[key] = {}
            result[key][new_label] = float(np.real(value))
        return result
