# API Security Essentials

Securing HTTP APIs requires layered controls: strong authentication, least-privilege authorization, transport protection, abuse prevention, and disciplined input handling aligned with common vulnerability classes.

## Authentication

### JWT (JSON Web Tokens)

JWTs encode claims and are signed (JWS) or encrypted (JWE). Stateless verification suits horizontally scaled APIs when keys rotate properly.

```python
# Illustrative: verify signature with PyJWT (do not hardcode secrets)
import jwt

def decode_access_token(token: str, key: str, algorithms: list[str]) -> dict:
    return jwt.decode(token, key, algorithms=algorithms)
```

Best practices:

- Short access-token TTL; refresh tokens with rotation and revocation list where needed.
- Validate `iss`, `aud`, `exp`, and algorithm allow-list to prevent algorithm confusion attacks.

### OAuth2

OAuth2 delegates authorization to an identity provider. For first-party SPAs and mobile apps, prefer **Authorization Code with PKCE** over implicit flow.

```bash
# Example authorization URL (conceptual)
https://id.example.com/oauth/authorize?
  response_type=code&
  client_id=app&
  redirect_uri=https://app/callback&
  scope=openid%20profile&
  code_challenge=...&
  code_challenge_method=S256
```

### API keys

API keys identify projects or services. Store them server-side; never embed in public clients. Rotate keys periodically and scope keys to minimal permissions.

```http
GET /v1/resources HTTP/1.1
Host: api.example.com
X-API-Key: sk_live_...
```

## Authorization

### RBAC (Role-Based Access Control)

Roles aggregate permissions (e.g., `admin`, `editor`, `viewer`). Enforce checks at the handler layer and in data access paths.

```python
def require_role(user, allowed: set[str]):
    if user.role not in allowed:
        raise PermissionError("Forbidden")
```

### ABAC (Attribute-Based Access Control)

ABAC evaluates policies over subject, resource, action, and environment attributes. Useful for fine-grained rules (region, data classification).

Example policy idea: *Allow read if `resource.owner_org == user.org` and `resource.classification != "restricted"`.*

## HTTPS / TLS

Terminate TLS at the edge with modern cipher suites; enforce **HSTS** for browsers. Use **certificate pinning** only when you control clients end-to-end; otherwise rely on public PKI with automated renewal (ACME).

```nginx
# Nginx snippet (illustrative)
ssl_protocols TLSv1.2 TLSv1.3;
add_header Strict-Transport-Security "max-age=63072000" always;
```

## Rate limiting

Protect expensive endpoints and authentication routes with token buckets or sliding windows. Return `429 Too Many Requests` with `Retry-After`.

```python
# Pseudocode: fixed window per IP
from collections import defaultdict
import time

buckets = defaultdict(list)

def allow(ip: str, limit: int, window_sec: int) -> bool:
    now = time.time()
    buckets[ip] = [t for t in buckets[ip] if now - t < window_sec]
    if len(buckets[ip]) >= limit:
        return False
    buckets[ip].append(now)
    return True
```

## Input validation

Validate and parse early with explicit schemas (Pydantic, JSON Schema). Reject unknown fields when appropriate; normalize encodings; bound sizes.

```python
from pydantic import BaseModel, Field, field_validator

class CreateItem(BaseModel):
    name: str = Field(min_length=1, max_length=200)

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        return v.strip()
```

## CORS

Cross-Origin Resource Sharing is a browser mechanism. Servers send `Access-Control-Allow-Origin` (specific origins, not `*` when credentials are used). Preflight `OPTIONS` must mirror allowed methods and headers.

```python
# FastAPI-style (conceptual)
# allow_origins=["https://app.example.com"], allow_credentials=True
```

## OWASP API Security Top 10 (overview)

Representative categories include broken object level authorization, broken authentication, excessive data exposure, lack of rate limiting, and mass assignment. Map each risk to controls: policy tests, schema validation, logging, and least-privilege queries.

## Security headers (HTTP)

For browser-facing APIs or bundled UIs, set headers such as:

- `Content-Security-Policy`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY` or `SAMEORIGIN`
- `Referrer-Policy: strict-origin-when-cross-origin`

```http
Content-Security-Policy: default-src 'self'; frame-ancestors 'none'
X-Content-Type-Options: nosniff
```

## Operational practices

Centralize structured logging without sensitive payloads; monitor failed auth spikes; run dependency scanning (SCA) and static analysis (SAST) in CI; practice secret scanning on commits.

This guide complements framework-specific hardening checklists and threat modeling exercises for your services.

## Session fixation and cookies

For cookie-based sessions, regenerate session identifiers after authentication upgrades. Set `HttpOnly`, `Secure`, and `SameSite` attributes.

```http
Set-Cookie: sid=s%3A...; Path=/; HttpOnly; Secure; SameSite=Lax
```

## SSRF and outbound requests

When your API fetches user-supplied URLs, block link-local and metadata IP ranges, resolve DNS to IPs before connecting, and enforce allow-lists when possible.

## Mass assignment

Do not bind request bodies directly to ORM models with writable foreign keys. Use DTOs that whitelist fields.

```python
class UserUpdate(BaseModel):
    display_name: str | None = None
    # email omitted intentionally — admin-only change via separate endpoint
```

## Idempotency keys

For POST operations with side effects (payments), accept `Idempotency-Key` headers and store outcomes to prevent duplicate charges on retries.

## Secrets rotation

Automate API key and JWT signing key rotation with overlap periods; log verification failures during transitions to detect stragglers.

## Threat modeling (STRIDE snapshot)

| Threat | Example control |
|--------|-------------------|
| Spoofing | MFA, signed tokens |
| Tampering | TLS, signed webhooks |
| Repudiation | audit logs |
| Information disclosure | least-privilege queries |
| Denial of service | rate limits, autoscaling |
| Elevation of privilege | RBAC tests, SQL parameterization |

Layer controls proportional to asset sensitivity and compliance obligations.
