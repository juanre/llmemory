-- Migration: Add UNIQUE constraint for idempotent upsert on (owner_id, id_at_origin)
--
-- This enables the archive-protocol identity contract:
-- - owner_id = archive-protocol entity (e.g., jro, tsm, gsk)
-- - id_at_origin = archive-protocol document_id
--
-- With this constraint, re-indexing the same document updates the existing
-- record rather than creating duplicates.

-- Drop the existing non-unique index (if it exists) and replace with unique constraint
-- Use DO block to handle both cases (index exists or doesn't exist)
DO $$
BEGIN
    -- Try to drop the non-unique index if it exists
    DROP INDEX IF EXISTS idx_documents_owner_origin;
EXCEPTION WHEN OTHERS THEN
    -- Ignore errors (index might not exist or be in different schema)
    NULL;
END $$;

-- Add unique constraint for idempotent upsert behavior
-- Use DO block to make it idempotent (constraint might already exist)
DO $$
BEGIN
    ALTER TABLE {{tables.documents}}
    ADD CONSTRAINT uq_documents_owner_id_at_origin UNIQUE (owner_id, id_at_origin);
EXCEPTION WHEN duplicate_object THEN
    -- Constraint already exists, that's fine
    NULL;
END $$;
