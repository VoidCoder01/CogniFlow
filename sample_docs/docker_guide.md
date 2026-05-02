# Docker and Containerization

Containers package applications with their dependencies for consistent execution across environments. This guide explains core concepts, Dockerfile authoring, multi-stage builds, Compose, orchestration, and troubleshooting.

## Images and containers

An **image** is an immutable template (layers). A **container** is a running instance of an image with writable overlay, isolated process namespace, and configured networking.

```bash
docker pull python:3.11-slim
docker run --rm -it python:3.11-slim python -c "import sys; print(sys.version)"
```

List and clean up:

```bash
docker ps
docker images
docker system df
docker container prune
```

## Volumes

Volumes persist data outside the container lifecycle. Bind mounts map host directories for development.

```bash
docker run --rm -v "$(pwd)/data:/data" alpine ls /data
```

Named volumes suit databases in production stacks:

```bash
docker volume create pgdata
docker run -e POSTGRES_PASSWORD=pass -v pgdata:/var/lib/postgresql/data -d postgres:16
```

## Networks

User-defined bridge networks provide DNS-based service discovery between containers.

```bash
docker network create appnet
docker run -d --name api --network appnet myapi:1.0
docker run --rm --network appnet curlimages/curl curl http://api:8080/health
```

## Dockerfile syntax

A Dockerfile describes build steps. Keep layers cache-friendly: install dependencies before copying app code.

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t myapp:1.0 .
docker run --rm -p 8000:8000 myapp:1.0
```

## Multi-stage builds

Use multi-stage builds to separate build tools from runtime images.

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /src
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*
COPY . .
CMD ["python", "main.py"]
```

## Docker Compose

Compose orchestrates multi-container apps with a YAML file.

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgres://user:pass@db:5432/app
    depends_on:
      - db
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata: {}
```

```bash
docker compose up --build
docker compose logs -f api
```

## Best practices

- Prefer slim or distroless base images; pin tags by digest for reproducibility.
- Run as non-root (`USER`) when possible.
- Use health checks (`HEALTHCHECK`) for orchestrators.
- Scan images (`docker scout`, `trivy`) in CI.

## Orchestration overview

Kubernetes schedules replicated workloads, rolling updates, service discovery, and autoscaling. Nomad and Docker Swarm offer lighter alternatives. For local Kubernetes development, **minikube** or **kind** are common.

```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl logs deploy/api
```

## Troubleshooting

| Symptom | Checks |
|---------|--------|
| Cannot connect to service | Port mapping, firewall, `depends_on` vs readiness |
| Out of disk | `docker system df`, prune unused images |
| Slow builds | Layer ordering, `.dockerignore`, build cache |
| Permission errors | File ownership in volumes, non-root user |

Inspect a container:

```bash
docker exec -it <container> sh
```

View resource usage:

```bash
docker stats
```

Containers are not VMs: share the host kernel, respect ulimits, and design stateless services with externalized persistence.

## Image layers and caching

Each Dockerfile instruction creates a layer. Put rarely changing steps first so rebuilds reuse cache.

```dockerfile
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY . /app
```

Use `.dockerignore` to exclude `.git`, `__pycache__`, and local virtualenvs from the build context:

```gitignore
.git
.venv
**/__pycache__
*.pyc
```

## Resource limits

Prevent noisy neighbors in shared hosts:

```bash
docker run --memory=512m --cpus=1.5 --pids-limit=256 myapp:1.0
```

In Compose:

```yaml
services:
  api:
    image: myapp:1.0
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
```

## Secrets management

Avoid `ENV` for production secrets. Prefer Docker secrets (Swarm), Kubernetes secrets, or runtime injection:

```bash
docker run --env-file ./prod.env myapp:1.0
```

For Compose, use `secrets:` with file providers and mount into `/run/secrets` for the app to read at startup.

## Registry workflow

Tag images with semantic versions and promote the same digest through environments:

```bash
docker tag myapp:1.4.0 registry.example.com/team/myapp:1.4.0
docker push registry.example.com/team/myapp:1.4.0
```

Enable content trust and vulnerability scanning in your registry when available.
