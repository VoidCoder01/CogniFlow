# PostgreSQL Indexing Basics

## B-tree indexes

B-tree indexes accelerate equality and range predicates on common columns such as primary keys and timestamps.

## GIN indexes

GIN indexes help with full-text search and certain JSONB containment queries.

## Connection pooling

Use a pooler (for example PgBouncer) when many short-lived application workers connect to PostgreSQL.

## Version

PostgreSQL 15+ examples are typical for modern deployments.
