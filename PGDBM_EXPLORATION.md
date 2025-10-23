# PGDBM Codebase Exploration: Complete Analysis

## Executive Summary

PGDBM is a production-grade async PostgreSQL database management library built on asyncpg with sophisticated template substitution, schema isolation, and connection pooling. This document provides a complete technical guide covering template substitution mechanisms, schema management, query execution patterns, and testing patterns based on actual source code inspection.

---

## 1. Template Substitution Mechanism

### 1.1 Core Algorithm: `prepare_query()` Method

**Location**: `/Users/juanre/prj/pgdbm/src/pgdbm/core.py:569-615`

The `prepare_query()` method is the heart of template substitution. It processes two template types:
- `{{schema}}` - Replaced with quoted schema name or "public"
- `{{tables.tablename}}` - Replaced with `"schema".tablename` or just `tablename`

#### Algorithm Flow:

```python
def prepare_query(self, query: str) -> str:
    # Step 1: Validate schema name against PostgreSQL identifier rules
    if self.schema:
        # Must start with letter/underscore, contain only alphanumeric+underscore, max 63 chars
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$", self.schema):
            raise SchemaError(...)
    
    # Step 2: Apply substitution based on schema presence
    if not self.schema:
        # No schema: use public schema
        query = query.replace("{{schema}}.", "")
        query = query.replace("{{schema}}", "public")
        query = re.sub(r"{{tables\.([a-zA-Z0-9_]+)}}", r"\1", query)
    else:
        # With schema: quote schema name and fully qualify tables
        quoted_schema = f'"{self.schema}"'
        query = query.replace("{{schema}}", quoted_schema)
        query = re.sub(r"{{tables\.([a-zA-Z0-9_]+)}}", f"{quoted_schema}.\\1", query)
    
    return query
```

### 1.2 Substitution Examples

**Example 1: With Schema**
```
Input:  CREATE TABLE {{tables.users}} (...)
Schema: "myschema"
Output: CREATE TABLE "myschema".users (...)

Input:  INSERT INTO {{tables.users}} (id) VALUES ($1)
Output: INSERT INTO "myschema".users (id) VALUES ($1)
```

**Example 2: Without Schema**
```
Input:  CREATE TABLE {{tables.users}} (...)
Schema: None
Output: CREATE TABLE users (...)

Input:  SELECT * FROM {{schema}}.products
Output: SELECT * FROM public.products
```

**Example 3: Function References**
```
Input:  CREATE FUNCTION {{schema}}.update_timestamp()
Schema: "inventory"
Output: CREATE FUNCTION "inventory".update_timestamp()
```

### 1.3 Where Template Substitution Happens Automatically

All methods that execute queries automatically call `prepare_query()`:

1. **Direct execution methods** (in `AsyncDatabaseManager`):
   - `execute()` - Line 691
   - `executemany()` - Line 710
   - `fetch_one()` - Line 728
   - `fetch_all()` - Line 748
   - `fetch_value()` - Line 768
   - `execute_and_return_id()` - Line 785

2. **Transaction methods** (in `TransactionManager`):
   - `execute()` - Line 58
   - `executemany()` - Line 65
   - `fetch_all()` - Line 72
   - `fetch_one()` - Line 83
   - `fetch_value()` - Line 94

3. **Migration execution**:
   - Migrations are processed through `prepare_query()` before execution
   - `module_name` parameter prevents interference between schemas

### 1.4 Security Validation

**Input validation** prevents SQL injection:
- Schema names: Must match `^[a-zA-Z_][a-zA-Z0-9_]{0,62}$` (PostgreSQL identifier rules)
- Table names in templates: Must match `[a-zA-Z0-9_]+`
- Schema names are always quoted: `"{schema_name}"`
- Parameterized queries for user data (uses `$1`, `$2`, etc.)

---

## 2. Schema Management Architecture

### 2.1 How Schemas Bind to AsyncDatabaseManager

**Binding happens at initialization** (core.py:314-365):

```python
class AsyncDatabaseManager:
    def __init__(
        self,
        config: Optional[DatabaseConfig] = None,
        pool: Optional[asyncpg.Pool] = None,
        schema: Optional[str] = None,
    ):
        # Option 1: Create with config (manages own pool)
        if config:
            self.schema = config.schema_name  # Binding from config
            # Later: connect() creates the pool
        
        # Option 2: Use external pool with schema override
        if pool:
            self.schema = schema or "public"  # Binding at init time
```

### 2.2 Schema Lifetime

Once bound, a manager's schema is **permanent for its lifetime**:

```python
# Schema binding is at initialization - NEVER change it
db1 = AsyncDatabaseManager(pool=shared_pool, schema="service1")
db2 = AsyncDatabaseManager(pool=shared_pool, schema="service2")

# This is WRONG - don't do this:
db1.schema = "service2"  # Would affect all db1 queries - bad pattern!
```

**Why permanent binding?**
- Eliminates race conditions
- Makes code easier to reason about
- Each manager has single, clear responsibility
- Prevents accidental cross-schema operations

### 2.3 Schema Application in Connection Initialization

When a connection is first acquired, the schema is set via `_connection_init()` (core.py:457-468):

```python
async def _connection_init(self, conn: asyncpg.Connection) -> None:
    if self.schema:
        # Create schema if needed
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
        # Set search_path so unqualified names default to this schema
        await conn.execute(f'SET search_path TO "{self.schema}", public')
```

**This means**:
1. Schemas are created automatically on first connection
2. `search_path` is set so unqualified table names use the schema
3. Template substitution provides explicit qualification for clarity

### 2.4 SchemaManager Utility Class

**Location**: core.py:895-915

```python
class SchemaManager:
    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db = db_manager
        self.schema_name = db_manager.schema
    
    async def ensure_schema_exists(self) -> None:
        # Helper to explicitly create schema
        await self.db.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema_name}"')
    
    def qualify_table_name(self, table_name: str) -> str:
        # Helper for manual qualification: "schema.table"
        if not self.schema_name:
            return table_name
        return f"{self.schema_name}.{table_name}"
```

---

## 3. Query Execution Patterns

### 3.1 Methods That Handle Template Substitution Automatically

All public query methods automatically call `prepare_query()`:

```python
async def execute(self, query: str, *args, timeout=None) -> str:
    query = self._prepare_query(query)  # ← Automatic substitution
    # ... execute with asyncpg

async def fetch_one(self, query: str, *args, timeout=None) -> Optional[dict]:
    query = self._prepare_query(query)  # ← Automatic substitution
    # ... fetch with asyncpg
```

### 3.2 When NOT to Use Template Syntax

**Don't use templates when**:

1. **Writing to raw connections**:
   ```python
   # WRONG - template not processed
   async with db.acquire() as conn:
       await conn.execute("INSERT INTO {{tables.users}} VALUES ($1)")
   
   # RIGHT - manually prepare the query
   async with db.acquire() as conn:
       query = db.prepare_query("INSERT INTO {{tables.users}} VALUES ($1)")
       await conn.execute(query)
   ```

2. **Using parameterized schema names** (use schema-isolated managers instead):
   ```python
   # WRONG - trying to parameterize schema
   query = "SELECT * FROM $1.users"  # $1 can't be a schema name
   
   # RIGHT - create separate manager for each schema
   tenant1_db = AsyncDatabaseManager(pool=shared_pool, schema="tenant_1")
   await tenant1_db.fetch_all("SELECT * FROM {{tables.users}}")
   ```

### 3.3 Transactions with Template Substitution

**TransactionManager** wraps connections and automatically applies templates:

```python
async with db.transaction() as tx:
    # Template substitution is AUTOMATIC
    await tx.execute("INSERT INTO {{tables.users}} (email) VALUES ($1)", email)
    user = await tx.fetch_one("SELECT * FROM {{tables.users}} WHERE email = $1", email)
    
    # Nested transactions (savepoints) also support templates
    async with tx.transaction() as nested_tx:
        await nested_tx.execute("UPDATE {{tables.users}} SET active = true")
```

**How it works** (core.py:30-122):

```python
class TransactionManager:
    def __init__(self, conn: asyncpg.Connection, db_manager: AsyncDatabaseManager):
        self._conn = conn
        self._db = db_manager
    
    async def execute(self, query: str, *args, timeout=None):
        query = self._db.prepare_query(query)  # ← Automatic substitution
        return await self._conn.execute(query, *args, timeout=timeout)
    
    @asynccontextmanager
    async def transaction(self):
        # Create savepoint for nested transaction
        async with self._conn.transaction():
            yield TransactionManager(self._conn, self._db)  # ← Yields new manager
```

### 3.4 Advanced: Raw Connection Access

For edge cases, you can access the raw connection:

```python
async with db.transaction() as tx:
    # Use TransactionManager's API (recommended)
    await tx.execute("INSERT INTO {{tables.users}} (email) VALUES ($1)", email)
    
    # OR access raw connection if needed
    raw_conn = tx.connection
    query = db.prepare_query("SELECT * FROM {{tables.users}}")
    result = await raw_conn.fetchrow(query)  # Manual substitution + raw query
```

---

## 4. Test Fixtures Architecture

### 4.1 Available Fixtures in `pgdbm.fixtures.conftest`

**Location**: `/Users/juanre/prj/pgdbm/src/pgdbm/fixtures/conftest.py`

#### 1. `test_db` (Most Common)
```python
@pytest_asyncio.fixture
async def test_db() -> AsyncDatabaseManager:
    """
    Basic test database with automatic cleanup.
    
    Usage:
        async def test_something(test_db):
            await test_db.execute("CREATE TABLE test (id INT)")
            await test_db.execute("INSERT INTO test VALUES (1)")
            result = await test_db.fetch_one("SELECT * FROM test")
            assert result["id"] == 1
    """
```

#### 2. `test_db_with_schema`
```python
@pytest_asyncio.fixture
async def test_db_with_schema() -> AsyncDatabaseManager:
    """
    Test database with schema isolation enabled (schema="test_schema").
    
    Usage:
        async def test_schema_isolation(test_db_with_schema):
            await test_db_with_schema.execute(
                "CREATE TABLE {{tables.products}} (id INT)"
            )
            # Table is created in test_schema.products
    """
```

#### 3. `test_db_factory`
```python
@pytest_asyncio.fixture
async def test_db_factory():
    """
    Factory for creating multiple isolated test databases in one test.
    
    Usage:
        async def test_multiple_dbs(test_db_factory):
            db1 = await test_db_factory.create_db("db1", schema="schema1")
            db2 = await test_db_factory.create_db("db2", schema="schema2")
            
            # Use both databases
            await db1.execute("CREATE TABLE {{tables.items}} (id INT)")
            await db2.execute("CREATE TABLE {{tables.items}} (id INT)")
            
            # Automatic cleanup
    """
```

#### 4. `test_db_with_tables`
```python
@pytest_asyncio.fixture
async def test_db_with_tables(test_db):
    """
    Pre-creates common tables: users, projects, agents.
    
    Tables:
    - users: id, email, full_name, is_active, created_at
    - projects: id, name, owner_id, description, created_at
    - agents: id, project_id, title, status, assigned_to, created_at
    """
```

#### 5. `test_db_with_data`
```python
@pytest_asyncio.fixture
async def test_db_with_data(test_db_with_tables):
    """
    Pre-creates tables and inserts sample data:
    - 3 users (alice, bob, charlie)
    - 2 projects
    - 5 agents
    """
```

#### 6. `db_test_utils`
```python
@pytest.fixture
def db_test_utils(test_db) -> DatabaseTestCase:
    """
    Provides helper methods for database testing:
    - create_test_user()
    - count_rows()
    - table_exists()
    - truncate_table()
    """
```

#### 7. `test_db_isolated`
```python
@pytest_asyncio.fixture
async def test_db_isolated(test_db) -> TransactionManager:
    """
    Wraps test in a transaction that auto-rolls back.
    
    Usage:
        async def test_isolated(test_db_isolated):
            # All changes automatically rolled back after test
            await test_db_isolated.execute("INSERT INTO users ...")
    """
```

### 4.2 How Fixtures Set Up Databases

The fixture setup process:

```python
# 1. Create unique test database
test_database = AsyncTestDatabase(config)
await test_database.create_test_database()  # Creates test_xxxxxxxx

# 2. Create database manager
config = test_database.get_test_db_config(schema="test_schema")
db_manager = AsyncDatabaseManager(config)
await db_manager.connect()  # Connects to test database

# 3. Create schema if needed
if schema:
    await db_manager.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

# 4. Yield to test
yield db_manager

# 5. Cleanup
await db_manager.disconnect()
await test_database.drop_test_database()  # Drops test_xxxxxxxx
```

### 4.3 Using Fixtures in Your Tests

#### Basic Pattern
```python
@pytest.mark.asyncio
async def test_user_creation(test_db):
    """Test user creation with basic fixture."""
    await test_db.execute("""
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL
        )
    """)
    
    await test_db.execute(
        "INSERT INTO users (email) VALUES ($1)",
        "alice@example.com"
    )
    
    user = await test_db.fetch_one("SELECT * FROM users WHERE email = $1", "alice@example.com")
    assert user["email"] == "alice@example.com"
```

#### Schema Isolation Pattern
```python
@pytest.mark.asyncio
async def test_schema_isolation(test_db_with_schema):
    """Test schema-isolated queries."""
    await test_db_with_schema.execute("""
        CREATE TABLE {{tables.items}} (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    
    await test_db_with_schema.execute(
        "INSERT INTO {{tables.items}} (name) VALUES ($1)",
        "Test Item"
    )
    
    items = await test_db_with_schema.fetch_all("SELECT * FROM {{tables.items}}")
    assert len(items) == 1
    assert items[0]["name"] == "Test Item"
```

#### Multiple Database Pattern
```python
@pytest.mark.asyncio
async def test_cross_database(test_db_factory):
    """Test with multiple isolated databases."""
    db1 = await test_db_factory.create_db("db1", schema="service1")
    db2 = await test_db_factory.create_db("db2", schema="service2")
    
    # Create same table in both databases
    for db in [db1, db2]:
        await db.execute("CREATE TABLE {{tables.config}} (key TEXT PRIMARY KEY, value TEXT)")
    
    # Insert different data
    await db1.execute("INSERT INTO {{tables.config}} VALUES ($1, $2)", "key1", "value1")
    await db2.execute("INSERT INTO {{tables.config}} VALUES ($1, $2)", "key2", "value2")
    
    # Verify isolation
    result1 = await db1.fetch_all("SELECT * FROM {{tables.config}}")
    result2 = await db2.fetch_all("SELECT * FROM {{tables.config}}")
    
    assert len(result1) == 1
    assert len(result2) == 1
    assert result1[0]["value"] == "value1"
    assert result2[0]["value"] == "value2"
```

#### Transaction Isolation Pattern
```python
@pytest.mark.asyncio
async def test_with_rollback(test_db_isolated):
    """Test with automatic rollback."""
    await test_db_isolated.execute("""
        CREATE TABLE test_data (id SERIAL PRIMARY KEY, value INT)
    """)
    
    await test_db_isolated.execute("INSERT INTO test_data (value) VALUES ($1)", 42)
    
    result = await test_db_isolated.fetch_one("SELECT * FROM test_data WHERE id = 1")
    assert result["value"] == 42
    
    # After test ends, all changes are rolled back automatically
```

### 4.4 Fixture Configuration from Environment

The default test config loads from environment variables:

```python
DatabaseTestConfig(
    host=os.environ.get("TEST_DB_HOST", "localhost"),
    port=int(os.environ.get("TEST_DB_PORT", "5432")),
    user=os.environ.get("TEST_DB_USER", "postgres"),
    password=os.environ.get("TEST_DB_PASSWORD", "postgres"),
    verbose=os.environ.get("TEST_DB_VERBOSE", "").lower() in ("1", "true", "yes"),
    log_sql=os.environ.get("TEST_DB_LOG_SQL", "").lower() in ("1", "true", "yes"),
)
```

To use custom values:

```bash
# Run tests with custom database
TEST_DB_HOST=db.example.com \
TEST_DB_PORT=5433 \
TEST_DB_USER=testuser \
TEST_DB_PASSWORD=testpass \
TEST_DB_VERBOSE=1 \
pytest tests/
```

### 4.5 Cleanup Behavior

Each fixture implements proper cleanup:

1. **`test_db`**: Drops entire test database after test
2. **`test_db_factory`**: Drops all created test databases in reverse order
3. **`test_db_with_schema`**: Drops test database (schema is part of it)
4. **`test_db_isolated`**: Rolls back to savepoint created at test start

---

## 5. Best Practices and Golden Rules

### 5.1 Connection Pool Pattern

**The Golden Rule**: Use ONE shared pool across your entire application.

```python
# ✅ CORRECT: One shared pool
async def main():
    config = DatabaseConfig(connection_string="postgresql://...")
    shared_pool = await AsyncDatabaseManager.create_shared_pool(config)
    
    # Create schema-specific managers
    server_db = AsyncDatabaseManager(pool=shared_pool, schema="server")
    api_db = AsyncDatabaseManager(pool=shared_pool, schema="api")
    analytics_db = AsyncDatabaseManager(pool=shared_pool, schema="analytics")
    
    return {
        'pool': shared_pool,
        'server': server_db,
        'api': api_db,
        'analytics': analytics_db,
    }

# ❌ WRONG: Multiple pools
server_db = AsyncDatabaseManager(DatabaseConfig(...))  # Creates own pool
api_db = AsyncDatabaseManager(DatabaseConfig(...))     # Another pool (wastes resources!)
```

### 5.2 Schema Isolation Pattern

**The Golden Rule**: Use schemas for logical separation, not multiple databases.

```python
# ✅ CORRECT: Schemas with shared pool
shared_pool = await AsyncDatabaseManager.create_shared_pool(config)
tenant1_db = AsyncDatabaseManager(pool=shared_pool, schema="tenant_1")
tenant2_db = AsyncDatabaseManager(pool=shared_pool, schema="tenant_2")

# ❌ WRONG: Multiple databases
tenant1_db = AsyncDatabaseManager(DatabaseConfig(database="tenant_1_db", ...))
tenant2_db = AsyncDatabaseManager(DatabaseConfig(database="tenant_2_db", ...))
```

### 5.3 Permanent Schema Binding

**The Golden Rule**: Set schema once at initialization, never change it.

```python
# ✅ CORRECT: Create manager per schema
db = AsyncDatabaseManager(pool=shared_pool, schema="myschema")
# Use db for all operations - schema is fixed

# ❌ WRONG: Don't switch schemas
db.schema = "otherschema"  # DON'T DO THIS!
```

### 5.4 Template Syntax Usage

**The Golden Rule**: Use `{{tables.tablename}}` for schema-aware code.

```python
# ✅ CORRECT: Use templates, works with any schema
async def create_user(db, email):
    await db.execute(
        "INSERT INTO {{tables.users}} (email) VALUES ($1)",
        email
    )

# ❌ WRONG: Hardcode schema (not portable)
async def create_user_bad(db, email):
    schema = db.schema or "public"
    await db.execute(
        f"INSERT INTO {schema}.users (email) VALUES ($1)",
        email
    )
```

### 5.5 Dynamic Table Names

**The Golden Rule**: Use schema-isolated managers for dynamic table scenarios.

```python
# ✅ CORRECT: Create manager per logical unit
managers = {}
for tenant_id in tenant_list:
    managers[tenant_id] = AsyncDatabaseManager(
        pool=shared_pool,
        schema=f"tenant_{tenant_id}"
    )

async def process_tenant(tenant_id, data):
    db = managers[tenant_id]  # Get pre-created manager
    await db.execute("INSERT INTO {{tables.data}} VALUES (...)", data)

# ❌ WRONG: Try to parameterize schema name
await db.execute("INSERT INTO $1.data VALUES (...)", f"tenant_{tenant_id}")
# $1 can't be a schema name in PostgreSQL
```

### 5.6 Transactions for Multi-Statement Operations

**The Golden Rule**: Use transactions for atomicity and template substitution support.

```python
# ✅ CORRECT: Use transaction context
async with db.transaction() as tx:
    user_id = await tx.fetch_value(
        "INSERT INTO {{tables.users}} (email) VALUES ($1) RETURNING id",
        "alice@example.com"
    )
    
    await tx.execute(
        "INSERT INTO {{tables.profiles}} (user_id, bio) VALUES ($1, $2)",
        user_id,
        "Alice's bio"
    )
    # Both commit or both rollback

# ❌ WRONG: Separate transactions
user_id = await db.fetch_value(
    "INSERT INTO {{tables.users}} (email) VALUES ($1) RETURNING id",
    "alice@example.com"
)
await db.execute(  # If this fails, user is already created!
    "INSERT INTO {{tables.profiles}} (user_id, bio) VALUES ($1, $2)",
    user_id,
    "Alice's bio"
)
```

### 5.7 Migration Template Usage

**The Golden Rule**: Use templates in migrations, schema is applied from manager.

```sql
-- migrations/001_create_users.sql
-- The schema is determined by AsyncMigrationManager's db_manager's schema
-- No need to specify schema in migration files

CREATE TABLE IF NOT EXISTS {{tables.users}} (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index
CREATE INDEX IF NOT EXISTS idx_users_email ON {{tables.users}}(email);

-- Function (uses {{schema}} placeholder)
CREATE OR REPLACE FUNCTION {{schema}}.hash_password(password TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN crypt(password, gen_salt('bf'));
END;
$$ LANGUAGE plpgsql;
```

**Migration execution**:

```python
async def run_migrations(db_manager, migrations_path, schema_name):
    # db_manager already bound to schema_name
    migrations = AsyncMigrationManager(
        db_manager,
        migrations_path=migrations_path,
        module_name=schema_name  # For tracking which schema's migrations ran
    )
    await migrations.apply_pending_migrations()
```

### 5.8 Testing Best Practices

**The Golden Rule**: Use fixtures for automatic cleanup and isolation.

```python
# ✅ CORRECT: Use fixtures for cleanup
async def test_user_creation(test_db):
    await test_db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
    await test_db.execute("INSERT INTO users VALUES (1)")
    # Fixture automatically drops database after test

# ❌ WRONG: Manual cleanup (error-prone)
async def test_user_creation():
    db = AsyncDatabaseManager(config)
    await db.connect()
    try:
        await db.execute("CREATE TABLE users (id INT PRIMARY KEY)")
        # ... test code ...
    finally:
        await db.disconnect()
    # Database still exists!
```

### 5.9 Schema Qualification Rules

When should you manually qualify vs. use templates?

**Use Templates** `{{tables.tablename}}` and `{{schema}}`:
- In application code (automatic substitution)
- In migration files (processed during migration)
- In transaction blocks (auto-substituted)
- When schema is bound to manager

**Manual Qualification** `"schema".tablename`:
- Querying information_schema or pg_* tables
- Cross-schema joins (when schema isn't bound manager)
- Raw SQL that needs explicit control
- When you know exact schema at dev time

**No Qualification** (implicit):
- When using `search_path` (set automatically on connection)
- For built-in functions and types
- For `public` schema tables when no schema is bound

### 5.10 Connection Pool Sizing Rules

Based on the patterns in examples:

```python
# Microservices shared pool
config = DatabaseConfig(
    min_connections=2,      # Per service minimum
    max_connections=10,     # Per service maximum
)
# Total connections = min/max * number_of_services
# For 4 services: 8-40 total connections

# Test database
config = DatabaseConfig(
    min_connections=2,      # Minimal for tests
    max_connections=5,      # Low to catch resource leaks
)

# Production single-service app
config = DatabaseConfig(
    min_connections=20,     # Maintain ready connections
    max_connections=50,     # Handle spikes
)
```

---

## 6. Complete Working Examples

### 6.1 Simple Single-Service App with Migrations

```python
# database.py
from pathlib import Path
from pgdbm import AsyncDatabaseManager, DatabaseConfig, AsyncMigrationManager

async def setup_database():
    # Create configuration
    config = DatabaseConfig(
        connection_string="postgresql://localhost/myapp",
        min_connections=10,
        max_connections=20,
        schema="myapp",  # App uses single schema
    )
    
    # Create and connect manager
    db = AsyncDatabaseManager(config)
    await db.connect()
    
    # Run migrations
    migrations = AsyncMigrationManager(
        db,
        migrations_path=str(Path(__file__).parent / "migrations"),
        module_name="myapp",
    )
    await migrations.apply_pending_migrations()
    
    return db

async def shutdown_database(db):
    await db.disconnect()
```

### 6.2 Multi-Service App with Shared Pool

```python
# database.py
from pgdbm import AsyncDatabaseManager, DatabaseConfig, AsyncMigrationManager

async def setup_services():
    # Create shared pool
    config = DatabaseConfig(
        connection_string="postgresql://localhost/myapp",
        min_connections=20,
        max_connections=50,
    )
    
    shared_pool = await AsyncDatabaseManager.create_shared_pool(config)
    
    # Create schema-specific managers
    databases = {
        'server': AsyncDatabaseManager(pool=shared_pool, schema="server"),
        'api': AsyncDatabaseManager(pool=shared_pool, schema="api"),
        'analytics': AsyncDatabaseManager(pool=shared_pool, schema="analytics"),
    }
    
    # Run migrations for each service
    migration_paths = {
        'server': './services/server/migrations',
        'api': './services/api/migrations',
        'analytics': './services/analytics/migrations',
    }
    
    for service_name, db in databases.items():
        migrations = AsyncMigrationManager(
            db,
            migrations_path=migration_paths[service_name],
            module_name=service_name,
        )
        applied = await migrations.apply_pending_migrations()
        print(f"{service_name}: Applied {len(applied['applied'])} migrations")
    
    return {
        'pool': shared_pool,
        'databases': databases,
    }

async def shutdown_services(infrastructure):
    await infrastructure['pool'].close()
```

### 6.3 Testing with Schema Isolation

```python
# tests/test_services.py
import pytest

@pytest.mark.asyncio
async def test_user_service(test_db_factory):
    """Test user service with isolated database and schema."""
    # Create isolated test database
    user_db = await test_db_factory.create_db(
        suffix="users",
        schema="user_service"
    )
    
    # Create tables
    await user_db.execute("""
        CREATE TABLE {{tables.users}} (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Test operations
    await user_db.execute(
        "INSERT INTO {{tables.users}} (email) VALUES ($1)",
        "alice@example.com"
    )
    
    users = await user_db.fetch_all("SELECT * FROM {{tables.users}}")
    assert len(users) == 1
    assert users[0]["email"] == "alice@example.com"

@pytest.mark.asyncio
async def test_with_rollback(test_db_isolated):
    """Test with automatic transaction rollback."""
    await test_db_isolated.execute("""
        CREATE TABLE test (id SERIAL PRIMARY KEY, value INT)
    """)
    
    await test_db_isolated.execute(
        "INSERT INTO test (value) VALUES ($1)",
        42
    )
    
    result = await test_db_isolated.fetch_one("SELECT * FROM test")
    assert result["value"] == 42
    
    # Automatically rolled back after test
```

---

## 7. Technical Reference Summary

### 7.1 Template Syntax Quick Reference

| Template | With Schema | Without Schema |
|----------|-------------|----------------|
| `{{schema}}` | `"myschema"` | `"public"` |
| `{{schema}}.table` | `"myschema".table` | `public.table` |
| `{{tables.users}}` | `"myschema".users` | `users` |

### 7.2 Methods That Auto-Apply Templates

**AsyncDatabaseManager**:
- `execute(query, *args)`
- `executemany(query, args_list)`
- `fetch_one(query, *args)`
- `fetch_all(query, *args)`
- `fetch_value(query, *args)`
- `execute_and_return_id(query, *args)`

**TransactionManager**:
- `execute(query, *args)`
- `executemany(query, args_list)`
- `fetch_one(query, *args)`
- `fetch_all(query, *args)`
- `fetch_value(query, *args)`

### 7.3 Schema Validation Rules

- **PostgreSQL identifiers**: `^[a-zA-Z_][a-zA-Z0-9_]{0,62}$`
- **Table names in templates**: `[a-zA-Z0-9_]+` only
- **All schema names are quoted**: `"{schema_name}"` to prevent reserved words
- **Parameters for data**: Use `$1`, `$2`, etc., never for identifiers

### 7.4 File Locations for Reference

| Component | Location |
|-----------|----------|
| Core implementation | `/Users/juanre/prj/pgdbm/src/pgdbm/core.py` |
| Migrations | `/Users/juanre/prj/pgdbm/src/pgdbm/migrations.py` |
| Test utilities | `/Users/juanre/prj/pgdbm/src/pgdbm/testing.py` |
| Fixtures | `/Users/juanre/prj/pgdbm/src/pgdbm/fixtures/conftest.py` |
| Patterns guide | `/Users/juanre/prj/pgdbm/PGDBM_PATTERNS_GUIDE.md` |
| Examples | `/Users/juanre/prj/pgdbm/examples/microservices/` |

---

## Summary: Golden Rules for pgdbm Usage

1. **One Pool**: Share a single connection pool across your entire application
2. **Schema Isolation**: Use schemas for logical separation, not multiple databases
3. **Permanent Binding**: Set schema once at initialization, never change it
4. **Template Syntax**: Use `{{tables.tablename}}` and `{{schema}}` for portability
5. **Automatic Substitution**: All query methods handle templates automatically
6. **Transaction Safety**: Use transactions for multi-statement atomicity
7. **Fixture Usage**: Use test fixtures for automatic cleanup and isolation
8. **Migration Management**: Let AsyncMigrationManager handle schema substitution
9. **Manual Qualification**: Only use explicit qualification when needed
10. **Pool Sizing**: Calculate total connections = (min/max per service) × number of services

