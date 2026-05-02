# PostgreSQL for Application Developers

PostgreSQL is a powerful open-source relational database. This guide covers installation, core SQL, indexing strategies, full-text search, JSONB, pooling, backups, and performance analysis.

## Installation

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl enable --now postgresql
```

macOS with Homebrew:

```bash
brew install postgresql@16
brew services start postgresql@16
```

Connect as the default superuser:

```bash
sudo -u postgres psql
```

Create a role and database:

```sql
CREATE ROLE appuser WITH LOGIN PASSWORD 'use-strong-secret';
CREATE DATABASE appdb OWNER appuser;
GRANT ALL PRIVILEGES ON DATABASE appdb TO appuser;
```

## Basic SQL

Create a table with constraints:

```sql
CREATE TABLE users (
    id          BIGSERIAL PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO users (email, display_name)
VALUES ('alice@example.com', 'Alice');
```

Select with filtering and ordering:

```sql
SELECT id, email
FROM users
WHERE email LIKE '%@example.com'
ORDER BY created_at DESC
LIMIT 10;
```

## Indexing

### B-tree (default)

B-tree indexes accelerate equality and range queries on scalar columns.

```sql
CREATE INDEX idx_users_email ON users (email);
```

### GIN for JSONB and full-text

GIN suits composite values and full-text vectors.

```sql
CREATE TABLE articles (
    id    BIGSERIAL PRIMARY KEY,
    body  TEXT,
    meta  JSONB
);

CREATE INDEX idx_articles_meta ON articles USING GIN (meta jsonb_path_ops);
```

### Partial indexes

Index a subset of rows to save space and speed hot paths.

```sql
CREATE INDEX idx_active_users
ON users (email)
WHERE deleted_at IS NULL;
```

## Full-text search

Use `to_tsvector` and `to_tsquery` with a GIN index on `tsvector`.

```sql
ALTER TABLE articles ADD COLUMN tsv TSVECTOR
GENERATED ALWAYS AS (to_tsvector('english', coalesce(body, ''))) STORED;

CREATE INDEX idx_articles_tsv ON articles USING GIN (tsv);

SELECT id, ts_rank(tsv, websearch_to_tsquery('english', 'postgres & indexing'))
FROM articles
WHERE tsv @@ websearch_to_tsquery('english', 'postgres & indexing')
ORDER BY ts_rank DESC
LIMIT 20;
```

## JSONB support

JSONB stores semi-structured payloads with binary efficiency and rich operators.

```sql
INSERT INTO articles (body, meta)
VALUES ('Hello', '{"tags": ["sql", "postgres"], "version": 1}'::jsonb);

SELECT meta->'tags' AS tags
FROM articles
WHERE meta @> '{"version": 1}';
```

Use JSONB when the schema evolves quickly; normalize critical relational data for integrity.

## Connection pooling

Avoid opening a new connection per HTTP request at scale. Deploy **PgBouncer** in transaction or session pooling mode, or use poolers built into drivers (e.g., SQLAlchemy `QueuePool`).

```ini
# pgbouncer.ini (illustrative)
[databases]
appdb = host=127.0.0.1 port=5432 dbname=appdb

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
```

Application settings: size pools to avoid exhausting PostgreSQL `max_connections`.

## Backup and recovery

Logical dumps with `pg_dump` for portability:

```bash
pg_dump -Fc -h localhost -U appuser appdb > appdb.dump
pg_restore -d appdb_restored appdb.dump
```

For large deployments, combine **WAL archiving** and base backups (e.g., `pg_basebackup`) for point-in-time recovery (PITR).

## Performance tuning with EXPLAIN ANALYZE

`EXPLAIN` shows the planner's strategy; `ANALYZE` executes and adds runtime stats.

```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT u.email
FROM users u
JOIN orders o ON o.user_id = u.id
WHERE o.placed_at > now() - interval '30 days';
```

Interpret:

- **Seq Scan** on large tables may indicate a missing index.
- **Nested Loop** vs **Hash Join** trade-offs depend on cardinality estimates.
- **Buffers: hit** vs **read** hints cache effectiveness.

Run `VACUUM (ANALYZE)` after bulk loads; keep statistics fresh with autovacuum defaults unless you have special batch workloads.

## Practical checklist

- Prefer `TIMESTAMPTZ` for instants.
- Use foreign keys and sensible `ON DELETE` behavior.
- Batch inserts with `COPY` for ingest pipelines.
- Monitor slow queries via `pg_stat_statements`.

PostgreSQL scales vertically well and horizontally with read replicas and careful sharding patterns when needed.

## Transactions and isolation

Use explicit transactions for multi-statement consistency:

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

Default isolation is `READ COMMITTED`. For stricter guarantees, consider `REPEATABLE READ` or `SERIALIZABLE` when anomalies appear.

## Common indexing mistakes

- Indexing low-cardinality columns alone (booleans) rarely helps selective queries.
- Over-indexing slows writes and vacuum work—measure with `pg_stat_user_indexes`.

## Window functions

Analytical queries without self-joins:

```sql
SELECT
    user_id,
    placed_at,
    sum(amount) OVER (PARTITION BY user_id ORDER BY placed_at) AS running_total
FROM orders;
```

## Extensions

Enable `pg_stat_statements` for query insights:

```sql
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
SELECT query, calls, mean_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

## Connection strings

Typical SQLAlchemy URL:

```text
postgresql+psycopg://user:pass@localhost:5432/appdb
```

Use SSL parameters in cloud-managed Postgres offerings (`sslmode=require`).

These advanced topics complement the fundamentals for building reliable data layers.
