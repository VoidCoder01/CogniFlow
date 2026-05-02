# Python Best Practices for Production Services

This guide summarizes code style, typing, error handling, common design patterns, testing, async I/O, logging, and environment isolation for maintainable Python applications.

## Code style (PEP 8)

PEP 8 promotes readability: 4-space indentation, sensible line breaks, `snake_case` for functions and variables, `CapWords` for classes. Use `isort` + `black` or `ruff format` for consistency.

```bash
pip install ruff
ruff check .
ruff format .
```

Docstrings (Google or NumPy style) clarify public APIs; avoid noise on trivial helpers.

## Type hints

Type hints improve IDE support and catch defects early. Use `mypy` or `pyright` in CI.

```python
from collections.abc import Sequence

def total_amount(lines: Sequence[tuple[str, float]]) -> float:
    return sum(amount for _, amount in lines)
```

Prefer `list[str]` over `List[str]` on Python 3.9+; use `TypedDict` and `Protocol` for structured dicts and duck typing.

## Error handling

Raise specific exceptions; avoid bare `except:`. Use context managers for resources.

```python
class PaymentError(Exception):
    """Domain-specific failure."""

def charge(card_id: str, amount_cents: int) -> None:
    if amount_cents <= 0:
        raise ValueError("amount_cents must be positive")
    try:
        gateway.charge(card_id, amount_cents)
    except gateway.GatewayTimeout as exc:
        raise PaymentError("upstream timeout") from exc
```

Log exceptions with stack traces at service boundaries; return safe messages to clients.

## Design patterns

### Repository pattern

Encapsulate persistence behind a repository interface to keep domain logic storage-agnostic.

```python
class UserRepository:
    def get_by_id(self, user_id: int) -> User | None: ...
    def save(self, user: User) -> None: ...
```

### Factory pattern

Centralize construction of complex objects (LLM clients, DB engines) with configuration.

```python
def make_http_client(timeout: float = 5.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, headers={"User-Agent": "myapp/1.0"})
```

### Dependency injection

Inject collaborators via constructors or FastAPI `Depends` rather than global singletons in large apps (small projects may use pragmatic module-level instances).

## Testing with pytest

Structure tests as `test_*.py`, use fixtures for setup, and prefer parametrization.

```python
import pytest

@pytest.mark.parametrize("raw,expected", [(" hi ", "hi"), ("", "")])
def test_strip_optional(raw: str, expected: str) -> None:
    assert strip_optional(raw) == expected
```

Use `tmp_path` for filesystem isolation and `monkeypatch` for environment and time.

## Async programming

`asyncio` suits I/O-bound workloads. Avoid blocking calls inside async functions; use `asyncio.to_thread` for CPU-light blocking libraries when needed.

```python
import asyncio
import httpx

async def fetch_status(url: str) -> int:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10.0)
        return response.status_code

async def main() -> None:
    results = await asyncio.gather(
        fetch_status("https://example.com"),
        fetch_status("https://httpbin.org/get"),
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

## Logging

Use the `logging` module with structured context; configure once at startup.

```python
import logging

log = logging.getLogger(__name__)

def handle_request(request_id: str) -> None:
    log.info("start", extra={"request_id": request_id})
    try:
        do_work()
    except Exception:
        log.exception("work failed", extra={"request_id": request_id})
        raise
```

Avoid `print` in services; tune levels (`INFO` in prod, `DEBUG` behind flags).

## Virtual environments

Isolate dependencies per project:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Pin versions for reproducibility (`pip freeze` or tools like `uv pip compile`).

## Security and robustness

- Validate all external input (HTTP, CLI, files).
- Use `secrets` for tokens, not `random`.
- Keep dependencies updated; audit licenses and CVEs.

These practices compound: small consistency wins reduce defect rates as teams and codebases grow.

## Packaging and layout

Use `src/` layout to avoid import ambiguity:

```text
src/
  myapp/
    __init__.py
    api/
    services/
pyproject.toml
tests/
```

Define entry points in `pyproject.toml` for CLI tools.

## Data classes and immutability

`dataclasses` and `frozen=True` model value objects:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Money:
    currency: str
    amount_cents: int
```

## Concurrency vs parallelism

`threading` helps when libraries release the GIL (I/O). `multiprocessing` sidesteps the GIL for CPU-heavy work. `asyncio` excels when workloads are await-heavy.

```python
from concurrent.futures import ProcessPoolExecutor

def cpu_bound(n: int) -> int:
    return sum(i * i for i in range(n))

with ProcessPoolExecutor() as pool:
    result = pool.map(cpu_bound, range(4))
```

## Configuration management

Use Pydantic `BaseSettings` (v2: `pydantic-settings`) to load environment variables with validation.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    database_url: str
    log_level: str = "INFO"

settings = Settings()
```

## Observability

Export Prometheus metrics or OpenTelemetry traces at HTTP boundaries. Correlate logs with `request_id` propagated from ingress to workers.

These topics round out day-to-day engineering discipline beyond syntax and unit tests.
