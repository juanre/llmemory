# Contributing to llmemory

We love your input! We want to make contributing to llmemory as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the style guidelines
6. Issue that pull request!

## Development Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector
- uv (recommended) or pip

### Setting Up Your Environment

1. Clone your fork:
```bash
git clone https://github.com/yourusername/llmemory.git
cd llmemory
```

2. Install dependencies:
```bash
# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev]"
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

4. Create a test database:
```bash
createdb memory_test
psql memory_test -c "CREATE EXTENSION vector"
```

5. Set environment variables:
```bash
export DATABASE_URL="postgresql://localhost/memory_test"
export LLMEMORY_OPENAI_API_KEY="sk-..." # Or use local embeddings
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llmemory --cov-report=html

# Run specific test file
pytest tests/test_search_quality.py

# Run tests in parallel
pytest -n auto
```

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **ruff** for linting
- **mypy** for type checking

Run all checks:
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Run linter
ruff check src tests

# Type checking
mypy src
```

Or use pre-commit to run all checks:
```bash
pre-commit run --all-files
```

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable
2. Update the CHANGELOG.md with your changes
3. Update any relevant documentation
4. The PR will be merged once you have the sign-off of at least one maintainer

### PR Title Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only changes
- `style:` Changes that don't affect code meaning
- `refactor:` Code change that neither fixes a bug nor adds a feature
- `perf:` Performance improvement
- `test:` Adding missing tests
- `chore:` Changes to build process or auxiliary tools

Examples:
- `feat: add support for custom embedding models`
- `fix: resolve connection pool exhaustion`
- `docs: update installation guide`

## Writing Tests

### Test Structure

```python
import pytest
from llmemory import AwordMemory, DocumentType

@pytest.mark.asyncio
async def test_document_search(test_memory):
    """Test that documents can be searched after adding."""
    # Arrange
    await test_memory.add_document(
        owner_id="test",
        id_at_origin="user1",
        document_name="test.txt",
        document_type=DocumentType.GENERAL,
        content="This is a test document about Python programming."
    )

    # Act
    results = await test_memory.search(
        owner_id="test",
        query_text="Python programming"
    )

    # Assert
    assert len(results) > 0
    assert "Python" in results[0].content
```

### Test Categories

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test component interactions
- **Performance tests**: Ensure operations meet performance requirements

## Reporting Bugs

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/juanreyero/llmemory/issues/new).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

We're always looking for suggestions to improve llmemory! If you have a feature request, please:

1. Check if the feature has already been requested
2. Open a new issue with the `enhancement` label
3. Describe the feature and why it would be useful
4. Provide examples of how it would be used

## Documentation

- Keep docstrings up to date
- Use Google-style docstrings
- Update the relevant .md files in `/docs`
- Include code examples where helpful

Example:
```python
async def search(
    self,
    owner_id: str,
    query_text: str,
    search_type: SearchType = SearchType.HYBRID,
    limit: int = 10
) -> List[SearchResult]:
    """Search for relevant document chunks.

    Args:
        owner_id: Workspace or tenant identifier
        query_text: The search query
        search_type: Type of search to perform
        limit: Maximum number of results

    Returns:
        List of search results ordered by relevance

    Raises:
        ValidationError: If inputs are invalid
        SearchError: If search operation fails

    Example:
        >>> results = await memory.search(
        ...     owner_id="workspace-123",
        ...     query_text="machine learning",
        ...     search_type=SearchType.HYBRID
        ... )
    """
```

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## Questions?

Feel free to contact the maintainers if you have any questions!
