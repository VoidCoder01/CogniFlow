# FastAPI Application Guide

## Routing and dependencies

FastAPI maps HTTP paths to Python functions. Use `APIRouter` to group related endpoints.

```python
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/api/v1")

@router.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

## Dependency injection

`Depends()` injects shared logic such as database sessions, authentication, and configuration.

## Version

This guide reflects FastAPI patterns commonly used in production services (version notes are illustrative).
