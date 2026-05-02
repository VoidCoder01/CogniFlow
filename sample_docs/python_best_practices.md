# Python Best Practices for Services

## Type hints

Use type hints for public functions and Pydantic models for request/response schemas in APIs.

## Logging

Prefer structured logging keys (request id, user id) over printing. Configure log levels per environment.

## Testing

Isolate I/O with dependency injection and use temporary directories for SQLite and local vector stores in tests.

## Version

Guidance is framework-agnostic and stable across Python 3.10+.
