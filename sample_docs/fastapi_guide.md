# FastAPI — Practical Tutorial

FastAPI is a modern Python web framework for building APIs with automatic OpenAPI docs, validation via Pydantic, and native `async` support. This tutorial walks through installation, routing, parameters, dependencies, OAuth2-style security, middleware, errors, and performance tips.

## Introduction

FastAPI leverages type annotations to generate request validation and JSON Schema. It sits on Starlette (ASGI) and integrates Pydantic v2 for models.

Key benefits:

- Interactive docs at `/docs` (Swagger UI) and `/redoc`.
- High performance comparable to Node/Go for many I/O-bound workloads.
- First-class `async def` endpoints.

## Installation

Create a virtual environment and install FastAPI with a production ASGI server:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "fastapi[standard]" uvicorn[standard]
```

Minimal application:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
```

Run locally:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Routing

Define routes with HTTP decorators. Path order matters: static paths before parameterized paths when they could conflict.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
def list_items():
    return []

@app.post("/items/")
def create_item():
    return {"ok": True}
```

APIRouter modules help organize large apps:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/v1")

@router.get("/health")
def health():
    return {"status": "ok"}
```

```python
from fastapi import FastAPI
from .routers import router as v1_router

app = FastAPI()
app.include_router(v1_router)
```

## Path and query parameters

Path parameters use `{name}` and function arguments with matching names. Query parameters default from function arguments.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
def read_user(user_id: int, verbose: bool = False):
    return {"user_id": user_id, "verbose": verbose}
```

Optional query parameters use `Optional` or defaults. Use `Annotated` with `Query` for validation:

```python
from typing import Annotated
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/search")
def search(
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
):
    return {"q": q, "limit": limit}
```

## Pydantic models

Request and response bodies use Pydantic models for validation and serialization.

```python
from pydantic import BaseModel, EmailStr, Field
from fastapi import FastAPI

class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=100)
    age: int | None = Field(default=None, ge=0, le=130)

class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: str

app = FastAPI()

@app.post("/users", response_model=UserOut)
def create_user(user: UserCreate) -> UserOut:
    # Persist user (illustrative)
    return UserOut(id=1, email=user.email, name=user.name)
```

Use `model_config` or `Field` aliases for external naming conventions.

## Dependency injection

`Depends` wires reusable logic: DB sessions, auth, settings.

```python
from typing import Annotated
from fastapi import Depends, FastAPI

class Settings:
    app_name: str = "My API"

def get_settings() -> Settings:
    return Settings()

app = FastAPI()

@app.get("/info")
def info(settings: Annotated[Settings, Depends(get_settings)]):
    return {"app_name": settings.app_name}
```

Dependencies can be async and nested:

```python
async def get_db():
    db = {"connected": True}
    try:
        yield db
    finally:
        db["connected"] = False

@app.get("/db-check")
async def db_check(db: dict = Depends(get_db)):
    return db
```

## OAuth2-style authentication

FastAPI provides utilities for OAuth2 password flow (often paired with JWT). Below is a minimal pattern using `OAuth2PasswordBearer` and token validation.

```python
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

FAKE_USERS = {"alice": {"password": "secret"}}

@app.post("/token")
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = FAKE_USERS.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=400, detail="Incorrect credentials")
    return {"access_token": form_data.username, "token_type": "bearer"}

def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    if token not in FAKE_USERS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

@app.get("/me")
def read_me(user: Annotated[str, Depends(get_current_user)]):
    return {"user": user}
```

In production, issue signed JWTs with short expirations and validate signatures.

## Middleware (CORS)

Middleware wraps requests and responses. CORS is common for browser clients.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Custom middleware example:

```python
import time
from starlette.requests import Request

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}"
    return response
```

## Error handling

Raise `HTTPException` for expected API errors.

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items_db: dict[int, str] = {}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"id": item_id, "name": items_db[item_id]}
```

Register exception handlers for uniform JSON errors:

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
```

## Performance tips

- Use `async def` for I/O-bound work; avoid blocking calls in async routes (offload to thread pool if needed).
- Enable **GZip** middleware for large JSON payloads when clients support it.
- Tune worker processes: `uvicorn main:app --workers 4` behind a load balancer for CPU-bound Python work.
- Cache expensive reads (Redis) and paginate list endpoints.
- Profile with `py-spy` or OpenTelemetry tracing to find hotspots.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Testing

FastAPI's `TestClient` uses Starlette to call your app in-process.

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
```

## Summary

FastAPI pairs ergonomic Python typing with production-ready ASGI features. Combine Pydantic models, dependency injection, and explicit error handling to build maintainable services, then harden security with OAuth2/JWT patterns appropriate to your deployment.
