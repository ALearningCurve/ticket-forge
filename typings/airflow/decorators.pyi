from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

def task(*args: Any, **kwargs: Any) -> Callable[[F], F]: ...
def dag(*args: Any, **kwargs: Any) -> Callable[[F], F]: ...

