from contextlib import contextmanager
from timeit import default_timer


@contextmanager
def timer(tag=None):
    start = default_timer()
    try:
        yield
    finally:
        end = default_timer()
        header = "Elapsed time (s)" if tag is None else f"[{tag}] Elapsed time (s)"
        print(f"{header}: {end - start:.6f}")