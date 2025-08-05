# Migration Guide

This guide explains how llmemory handles database migrations in different deployment scenarios.

## Overview

llmemory uses async-db's migration system to manage its database schema. Key features:

- **Automatic execution**: Migrations run automatically during `initialize()`
- **Module-based tracking**: Migrations are tracked as `aword_memory` module
- **Schema isolation**: Tables use `{{tables.}}` syntax for schema awareness
- **Idempotent**: Safe to run multiple times

## Migration Behavior by Pattern

### Pattern 1: Standalone Mode

When llmemory creates its own connection:

```python
memory = AwordMemory(connection_string="postgresql://localhost/mydb")
await memory.initialize()  # Runs migrations
```

**What happens:**
1. Creates tables in the default schema (usually `public`)
2. Table names: `documents`, `document_chunks`, etc.
3. Migration history tracked in `public.schema_migrations`

### Pattern 2: Library Usage

When used by another library, behavior depends on initialization:

```python
# If library uses standalone mode
analyzer = DocumentAnalyzer(connection_string="...")
await analyzer.initialize()  # llmemory runs migrations

# If library uses shared pool mode
analyzer = DocumentAnalyzer(db_manager=db_manager)
await analyzer.initialize()  # llmemory runs migrations in assigned schema
```

### Pattern 3: Shared Pool Mode

When using a shared connection pool:

```python
# Application creates pool and schema-isolated manager
shared_pool = await AsyncDatabaseManager.create_shared_pool(config)
memory_db = AsyncDatabaseManager(pool=shared_pool, schema="llmemory")

# llmemory uses the manager
memory = AwordMemory.from_db_manager(memory_db)
await memory.initialize()  # Runs migrations in "llmemory" schema
```

**What happens:**
1. Creates tables in the assigned schema (e.g., `llmemory`)
2. Table names: `llmemory.documents`, `llmemory.document_chunks`, etc.
3. Migration history tracked in `public.schema_migrations` with module name

## The Migration File

llmemory's migration uses async-db's template syntax:

```sql
-- Uses {{tables.}} prefix for schema awareness
CREATE TABLE IF NOT EXISTS {{tables.documents}} (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ...
);

-- Foreign keys also use the syntax
CREATE TABLE IF NOT EXISTS {{tables.document_chunks}} (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES {{tables.documents}}(document_id),
    ...
);

-- Indexes use it too
CREATE INDEX IF NOT EXISTS idx_documents_owner_id
ON {{tables.documents}} (owner_id);
```

## How Schema Isolation Works

The `{{tables.}}` syntax is replaced at runtime:

- **Standalone**: `{{tables.documents}}` → `documents`
- **With schema**: `{{tables.documents}}` → `llmemory.documents`

This ensures:
- Tables are created in the correct schema
- Foreign keys reference the correct tables
- Indexes are created on the correct tables
- No conflicts with other services

## Migration Tracking

async-db tracks migrations in the `schema_migrations` table:

```sql
SELECT * FROM public.schema_migrations WHERE module_name = 'aword_memory';
```

Each migration is tracked by:
- `module_name`: Always `aword_memory` for llmemory
- `version`: Extracted from filename (e.g., "001")
- `filename`: Original migration filename
- `checksum`: SHA-256 hash to detect changes
- `applied_at`: When the migration was applied

## Multiple Instances

When multiple llmemory instances start:
1. First instance runs migrations
2. Other instances check migration table
3. Already-applied migrations are skipped
4. New migrations are applied if needed

## Permissions Required

The database user needs these permissions:

**For migrations:**
- `CREATE TABLE`
- `CREATE INDEX`
- `CREATE EXTENSION` (for pgvector)
- `INSERT` on `schema_migrations`

**For runtime:**
- Standard CRUD on llmemory tables
- `SELECT` on `schema_migrations`

**For schema isolation:**
- `CREATE SCHEMA` (if schema doesn't exist)
- `USAGE` on the schema

## Best Practices

1. **Let llmemory manage its schema**: Don't modify tables manually
2. **Use appropriate permissions**: Migration user vs runtime user
3. **Monitor first startup**: Check logs for migration success
4. **Test upgrades**: Always test new versions in staging first

## Troubleshooting

### Migration Fails with Permission Error

```
Error: permission denied to create extension "vector"
```

**Solution**: Run with a user that has SUPERUSER or create the extension manually:
```sql
-- As superuser
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL ON SCHEMA llmemory TO your_app_user;
```

### Tables Created in Wrong Schema

**Symptom**: Tables appear in `public` instead of assigned schema

**Cause**: Migration file not using `{{tables.}}` syntax

**Solution**: Ensure migration uses template syntax (this has been fixed in current version)

### Migration Already Applied Error

```
Error: Migration 001_complete_schema.sql has been modified after being applied
```

**Cause**: Migration file changed after deployment

**Solution**:
1. Never modify already-applied migrations
2. Create new migrations for schema changes
3. In development, reset: `DELETE FROM schema_migrations WHERE module_name = 'aword_memory'`

### Schema Not Found

```
Error: schema "llmemory" does not exist
```

**Solution**: Create schema before initializing:
```sql
CREATE SCHEMA IF NOT EXISTS llmemory;
```

Or ensure the database user has `CREATE SCHEMA` permission.
