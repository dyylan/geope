from __future__ import annotations

from typing import Callable


def normalize_callbacks(
    callbacks: Callable | list[Callable] | tuple[Callable, ...] | None,
) -> tuple[Callable, ...]:
    """Normalise the ``callbacks`` argument into a tuple of callables.

    Accepts the flexible forms allowed at the optimiser ``optimize()`` /
    ``Gecko`` public-method boundary: ``None`` (no callbacks), a single
    callable, or a list/tuple of callables.

    Args:
        callbacks: ``None``, a single callable, or a list/tuple of callables.
            Each callable must have the signature
            ``callback(step, history, optimizer) -> bool``.

    Returns:
        A (possibly empty) tuple of callables.

    Raises:
        TypeError: If ``callbacks`` (or any entry of it) is not callable.
    """
    if callbacks is None:
        return ()
    if callable(callbacks):
        return (callbacks,)
    if isinstance(callbacks, (list, tuple)):
        for cb in callbacks:
            if not callable(cb):
                raise TypeError(
                    f"Each callback must be callable, got {type(cb).__name__}."
                )
        return tuple(callbacks)
    raise TypeError(
        "callbacks must be None, a callable, or a list/tuple of callables, "
        f"got {type(callbacks).__name__}."
    )


def run_callbacks(
    callbacks: tuple[Callable, ...],
    step: int,
    history,
    optimizer,
) -> bool:
    """Invoke every callback and decide whether the loop should continue.

    All callbacks are always invoked (so their side-effects — logging,
    plotting, state updates — always run), regardless of what earlier ones
    return. The loop should continue only while **every** callback returns a
    truthy value; any falsy return (``False``, ``None``, ``0``, ``""``, …)
    requests a stop.

    Args:
        callbacks: Tuple of callables, as produced by `normalize_callbacks`.
        step: The step index just completed (1-based).
        history: The optimiser's `History` object, or ``None``.
        optimizer: The live optimiser instance (`Geope`, `Grape`, `Gecko`).

    Returns:
        ``True`` if the loop should continue, ``False`` if any callback
        requested a stop. With no callbacks, always ``True``.
    """
    should_continue = True
    for cb in callbacks:
        if not cb(step, history, optimizer):
            should_continue = False
    return should_continue
