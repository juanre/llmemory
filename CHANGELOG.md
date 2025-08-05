# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-01-15

### Added
- Document listing API with pagination and filtering
- Full document retrieval with chunks
- Enhanced search results with document metadata
- Statistics API for usage tracking
- Batch delete operations
- Detailed operation results
- Prometheus metrics support
- Health check endpoints
- Connection pool sharing
- Multi-language support (14+ languages)
- Local embedding provider support

### Changed
- Improved chunking strategies for different document types
- Enhanced search performance with HNSW indexing
- Better error handling and custom exceptions
- Optimized memory usage for large documents

### Fixed
- Connection pool exhaustion under high load
- Memory leaks in embedding generation
- Search result ordering consistency

## [0.1.0] - 2023-12-01

### Added
- Initial release
- Document storage with PostgreSQL and pgvector
- OpenAI embeddings integration
- Vector and hybrid search
- Basic chunking support
- Multi-tenant isolation
- Async/await API
- Basic validation