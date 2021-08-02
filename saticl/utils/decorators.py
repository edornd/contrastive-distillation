from typing import Callable

from torch import distributed


def only_rank(rank: int = 0):

    def decorator(wrapped_fn: Callable):

        def wrapper(*args, **kwargs):
            if not distributed.is_initialized() or distributed.get_rank() == rank:
                return wrapped_fn(*args, **kwargs)

        return wrapper

    return decorator
