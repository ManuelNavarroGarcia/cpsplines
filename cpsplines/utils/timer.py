from contextlib import contextmanager
from timeit import default_timer
from typing import Optional


@contextmanager
def timer(tag: Optional[str] = None) -> None:

    """
    Computes the elapsed time that a task last.

    Parameters
    ----------
    tag : str, optional
        The name of the task. By default, None.
    """

    start = default_timer()
    try:
        yield
    finally:
        end = default_timer()
        header = "Elapsed time (s)" if tag is None else f"[{tag}] Elapsed time (s)"
        print(f"{header}: {end - start:.6f}")
