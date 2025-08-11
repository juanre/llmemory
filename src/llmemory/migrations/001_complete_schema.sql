-- Complete schema for aword-memory with pgvector support
-- Designed for integration with agent-engine for document indexing and retrieval

-- Enable pgvector extension (requires superuser privileges)
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table (using pgdbm template syntax for schema isolation)
CREATE TABLE IF NOT EXISTS {{tables.documents}} (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id TEXT NOT NULL,       -- Owner identifier for filtering (e.g., workspace_id)
    id_at_origin TEXT NOT NULL,  -- User ID, thread ID, or other origin identifier within owner
    document_type TEXT NOT NULL,  -- pdf, markdown, code, text, html, docx, email, report, chat
    document_name TEXT NOT NULL,
    document_date TIMESTAMPTZ,    -- When the document was created/modified
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document chunks table with hierarchical support (no embeddings)
CREATE TABLE IF NOT EXISTS {{tables.document_chunks}} (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES {{tables.documents}}(document_id) ON DELETE CASCADE,
    parent_chunk_id UUID REFERENCES {{tables.document_chunks}}(chunk_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,  -- Order within parent or document
    chunk_level INTEGER NOT NULL DEFAULT 0,  -- 0 = leaf, 1 = section, 2 = chapter, etc.
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,  -- For deduplication
    token_count INTEGER NOT NULL,

    -- Metadata including chunking parameters and semantic info
    metadata JSONB DEFAULT '{}',

    -- Full-text search vector
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(document_id, content_hash)  -- Prevent duplicate chunks within a document
);

-- Embedding providers registry
CREATE TABLE IF NOT EXISTS {{tables.embedding_providers}} (
    provider_id TEXT PRIMARY KEY,  -- e.g., "openai-text-embedding-3-small"
    provider_type TEXT NOT NULL CHECK (provider_type IN ('openai', 'local')),
    model_name TEXT NOT NULL,      -- Actual model name
    dimension INTEGER NOT NULL,    -- Vector dimension
    table_name TEXT NOT NULL,      -- Name of the embedding table
    is_active BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- OpenAI embeddings table (1536 dimensions for text-embedding-3-small)
CREATE TABLE IF NOT EXISTS {{tables.embeddings_openai_3_small}} (
    chunk_id UUID PRIMARY KEY REFERENCES {{tables.document_chunks}}(chunk_id) ON DELETE CASCADE,
    embedding vector(1536) NOT NULL,
    model_version TEXT DEFAULT 'text-embedding-3-small',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default OpenAI provider
INSERT INTO {{tables.embedding_providers}} (provider_id, provider_type, model_name, dimension, table_name, is_default)
VALUES ('openai-text-embedding-3-small', 'openai', 'text-embedding-3-small', 1536, 'embeddings_openai_3_small', true)
ON CONFLICT (provider_id) DO NOTHING;

-- Embeddings queue for batch processing (now includes provider)
CREATE TABLE IF NOT EXISTS {{tables.embedding_queue}} (
    queue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID NOT NULL REFERENCES {{tables.document_chunks}}(chunk_id) ON DELETE CASCADE,
    provider_id TEXT NOT NULL REFERENCES {{tables.embedding_providers}}(provider_id),
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, processing, completed, failed
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    UNIQUE(chunk_id, provider_id)  -- One queue entry per chunk per provider
);

-- Search history for analytics and improvement (no embeddings stored)
CREATE TABLE IF NOT EXISTS {{tables.search_history}} (
    search_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id TEXT NOT NULL,      -- Owner identifier for filtering
    id_at_origin TEXT NOT NULL,
    query_text TEXT NOT NULL,
    provider_id TEXT REFERENCES {{tables.embedding_providers}}(provider_id),
    search_type TEXT NOT NULL,  -- vector, text, hybrid
    metadata_filter JSONB,
    result_count INTEGER,
    results JSONB,  -- Top results with scores
    feedback JSONB,  -- User feedback on relevance
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance Indexes

-- Vector similarity search index for OpenAI embeddings using HNSW
CREATE INDEX IF NOT EXISTS idx_embeddings_openai_3_small_embedding
ON {{tables.embeddings_openai_3_small}}
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector_gin
ON {{tables.document_chunks}}
USING gin(search_vector);

-- Multi-tenant filtering indexes
CREATE INDEX IF NOT EXISTS idx_documents_owner_id
ON {{tables.documents}} (owner_id);

CREATE INDEX IF NOT EXISTS idx_documents_owner_origin
ON {{tables.documents}} (owner_id, id_at_origin);

CREATE INDEX IF NOT EXISTS idx_documents_owner_date
ON {{tables.documents}} (owner_id, document_date DESC);

-- Document type filtering
CREATE INDEX IF NOT EXISTS idx_documents_document_type
ON {{tables.documents}} (document_type);

-- Metadata filtering (JSONB GIN indexes)
CREATE INDEX IF NOT EXISTS idx_documents_metadata
ON {{tables.documents}}
USING gin (metadata);

CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin
ON {{tables.document_chunks}}
USING gin(metadata);

-- Hierarchical navigation indexes
CREATE INDEX IF NOT EXISTS idx_chunks_hierarchy
ON {{tables.document_chunks}} (document_id, chunk_level, chunk_index);

CREATE INDEX IF NOT EXISTS idx_chunks_parent
ON {{tables.document_chunks}} (parent_chunk_id)
WHERE parent_chunk_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_document
ON {{tables.document_chunks}} (document_id, chunk_index);

-- Embedding queue processing
CREATE INDEX IF NOT EXISTS idx_embedding_queue_status
ON {{tables.embedding_queue}} (status, created_at)
WHERE status = 'pending';

-- Content deduplication
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash
ON {{tables.document_chunks}} (content_hash);

-- Index for finding which chunks have embeddings
CREATE INDEX IF NOT EXISTS idx_embeddings_openai_chunk_id
ON {{tables.embeddings_openai_3_small}} (chunk_id);

-- Index for embedding queue processing by provider
CREATE INDEX IF NOT EXISTS idx_embedding_queue_provider_status
ON {{tables.embedding_queue}} (provider_id, status, created_at)
WHERE status = 'pending';

-- Search history indexes
CREATE INDEX IF NOT EXISTS idx_search_history_owner_id
ON {{tables.search_history}} (owner_id);

CREATE INDEX IF NOT EXISTS idx_search_history_owner_created
ON {{tables.search_history}} (owner_id, created_at DESC);

-- Functions for common operations

-- Function to get chunk with parent context
CREATE OR REPLACE FUNCTION get_chunk_with_context(
    p_chunk_id UUID,
    p_context_window INTEGER DEFAULT 2
) RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    chunk_level INTEGER,
    chunk_index INTEGER,
    is_target BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE chunk_hierarchy AS (
        -- Get the target chunk
        SELECT c.chunk_id, c.parent_chunk_id, c.chunk_level, 0 as depth
        FROM {{tables.document_chunks}} c
        WHERE c.chunk_id = p_chunk_id

        UNION ALL

        -- Get parent chunks up to context window
        SELECT c.chunk_id, c.parent_chunk_id, c.chunk_level, ch.depth + 1
        FROM {{tables.document_chunks}} c
        JOIN chunk_hierarchy ch ON c.chunk_id = ch.parent_chunk_id
        WHERE ch.depth < p_context_window
    )
    SELECT
        c.chunk_id,
        c.content,
        c.chunk_level,
        c.chunk_index,
        c.chunk_id = p_chunk_id as is_target
    FROM {{tables.document_chunks}} c
    JOIN chunk_hierarchy ch ON c.chunk_id = ch.chunk_id
    ORDER BY c.chunk_level DESC, c.chunk_index;
END;
$$ LANGUAGE plpgsql;

-- Update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at
BEFORE UPDATE ON {{tables.documents}}
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update table statistics for query planner
ANALYZE {{tables.documents}};
ANALYZE {{tables.document_chunks}};
ANALYZE {{tables.embedding_queue}};
ANALYZE {{tables.search_history}};

-- Add comments explaining design choices
COMMENT ON TABLE {{tables.documents}} IS 'Document metadata with owner-based multi-tenancy and language detection';
COMMENT ON TABLE {{tables.document_chunks}} IS 'Hierarchical document chunks for semantic search with multi-language support';
COMMENT ON TABLE {{tables.embedding_providers}} IS 'Registry of available embedding providers and their configurations';
COMMENT ON TABLE {{tables.embeddings_openai_3_small}} IS 'OpenAI embeddings for document chunks (1536 dimensions)';
COMMENT ON TABLE {{tables.embedding_queue}} IS 'Queue for batch embedding generation with rate limiting';
COMMENT ON TABLE {{tables.search_history}} IS 'Search analytics for improving retrieval quality';
COMMENT ON INDEX idx_embeddings_openai_3_small_embedding IS 'HNSW index for fast k-NN search with OpenAI embeddings';
COMMENT ON INDEX idx_chunks_search_vector_gin IS 'GIN index for full-text search performance';
COMMENT ON INDEX idx_documents_owner_origin IS 'Composite index for multi-tenant filtering';
