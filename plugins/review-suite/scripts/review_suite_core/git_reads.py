"""Share Git reads during a command's evaluation of the checkout."""

from collections.abc import Callable
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from typing import Any


_reads: ContextVar[dict[tuple[Any, ...], Any] | None] = ContextVar(
    "review_suite_git_reads", default=None
)


def with_git_reads[**P, T](function: Callable[P, T]) -> Callable[P, T]:
    @wraps(function)
    def run(*args: P.args, **kwargs: P.kwargs) -> T:
        token = _reads.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _reads.reset(token)

    return run


def refresh_git_reads() -> None:
    # A reviewer can take minutes. Check the live checkout again after it returns.
    reads = _reads.get()
    if reads is not None:
        reads.clear()


def cached_git_read[**P, T](function: Callable[P, T]) -> Callable[P, T]:
    @wraps(function)
    def read(*args: P.args, **kwargs: P.kwargs) -> T:
        reads = _reads.get()
        if reads is None:
            return function(*args, **kwargs)
        key = (function, args, tuple(kwargs.items()))
        if key not in reads:
            reads[key] = function(*args, **kwargs)
        # Some callers enrich the returned facts for a particular review cycle.
        return deepcopy(reads[key])

    return read
