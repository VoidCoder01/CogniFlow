# Docker for Python Services

## Multi-stage builds

Use a builder stage to compile wheels, then copy artifacts into a slim runtime image to reduce attack surface and image size.

## Compose networking

Services on the same Docker Compose network resolve each other by service name (for example `http://api:8000`).

## Volumes

Persist databases and vector indices on named volumes so containers can be recreated without data loss.

## Version

Compose file format 3.8+ is widely supported by Docker Engine and Docker Desktop.
