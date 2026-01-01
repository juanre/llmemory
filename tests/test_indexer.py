"""Tests for archive indexer functionality."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmemory.archive import ArchiveItem
from llmemory.indexer import ArchiveIndexer, IndexResult
from llmemory.models import Document, DocumentChunk, DocumentType


class TestTextExtraction:
    """Test text extraction from different file types."""

    def test_extract_text_file(self, tmp_path: Path) -> None:
        """Test extracting text from a plain text file."""
        content = "This is plain text content."
        text_file = tmp_path / "test.txt"
        text_file.write_text(content)

        item = ArchiveItem(
            content_path=text_file,
            sidecar_path=tmp_path / "test.json",
            entity="jro",
            document_id="doc-001",
            relative_path="jro/docs/test.txt",
        )

        indexer = ArchiveIndexer(MagicMock())
        result = indexer._extract_text(item)

        assert result == content

    def test_extract_markdown(self, tmp_path: Path) -> None:
        """Test extracting text from a markdown file."""
        content = "# Title\n\nThis is **markdown** content."
        md_file = tmp_path / "test.md"
        md_file.write_text(content)

        item = ArchiveItem(
            content_path=md_file,
            sidecar_path=tmp_path / "test.json",
            entity="jro",
            document_id="doc-001",
            relative_path="jro/docs/test.md",
        )

        indexer = ArchiveIndexer(MagicMock())
        result = indexer._extract_text(item)

        assert result == content

    def test_extract_html(self, tmp_path: Path) -> None:
        """Test extracting text from an HTML file."""
        html = """
        <html>
        <head><title>Test</title></head>
        <body>
            <h1>Title</h1>
            <p>Paragraph with &amp; entity.</p>
            <script>alert('ignored');</script>
        </body>
        </html>
        """
        html_file = tmp_path / "test.html"
        html_file.write_text(html)

        item = ArchiveItem(
            content_path=html_file,
            sidecar_path=tmp_path / "test.json",
            entity="jro",
            document_id="doc-001",
            relative_path="jro/docs/test.html",
        )

        indexer = ArchiveIndexer(MagicMock())
        result = indexer._extract_text(item)

        assert "Title" in result
        assert "Paragraph with & entity" in result
        assert "alert" not in result  # Script should be removed

    def test_extract_unsupported_type(self, tmp_path: Path) -> None:
        """Test that unsupported file types return None."""
        bin_file = tmp_path / "test.bin"
        bin_file.write_bytes(b"\x00\x01\x02")

        item = ArchiveItem(
            content_path=bin_file,
            sidecar_path=tmp_path / "test.json",
            entity="jro",
            document_id="doc-001",
            relative_path="jro/docs/test.bin",
        )

        indexer = ArchiveIndexer(MagicMock())
        result = indexer._extract_text(item)

        assert result is None


class TestDocumentTypeMapping:
    """Test document type mapping from file extensions."""

    def test_txt_to_text(self) -> None:
        """Test .txt maps to TEXT."""
        item = ArchiveItem(
            content_path=Path("/test.txt"),
            sidecar_path=Path("/test.json"),
            entity="jro",
            document_id="doc-001",
            relative_path="test.txt",
        )

        indexer = ArchiveIndexer(MagicMock())
        assert indexer._get_document_type(item) == DocumentType.TEXT

    def test_md_to_markdown(self) -> None:
        """Test .md maps to MARKDOWN."""
        item = ArchiveItem(
            content_path=Path("/test.md"),
            sidecar_path=Path("/test.json"),
            entity="jro",
            document_id="doc-001",
            relative_path="test.md",
        )

        indexer = ArchiveIndexer(MagicMock())
        assert indexer._get_document_type(item) == DocumentType.MARKDOWN

    def test_html_to_html(self) -> None:
        """Test .html maps to HTML."""
        item = ArchiveItem(
            content_path=Path("/test.html"),
            sidecar_path=Path("/test.json"),
            entity="jro",
            document_id="doc-001",
            relative_path="test.html",
        )

        indexer = ArchiveIndexer(MagicMock())
        assert indexer._get_document_type(item) == DocumentType.HTML

    def test_pdf_to_pdf(self) -> None:
        """Test .pdf maps to PDF."""
        item = ArchiveItem(
            content_path=Path("/test.pdf"),
            sidecar_path=Path("/test.json"),
            entity="jro",
            document_id="doc-001",
            relative_path="test.pdf",
        )

        indexer = ArchiveIndexer(MagicMock())
        assert indexer._get_document_type(item) == DocumentType.PDF


class TestSidecarUpdate:
    """Test sidecar update functionality."""

    def test_update_sidecar_adds_llmemory_section(self, tmp_path: Path) -> None:
        """Test that update_sidecar adds llmemory section."""
        sidecar_path = tmp_path / "test.json"
        sidecar_path.write_text(json.dumps({"id": "doc-001", "source": "email"}))

        item = ArchiveItem(
            content_path=tmp_path / "test.txt",
            sidecar_path=sidecar_path,
            entity="jro",
            document_id="doc-001",
            relative_path="jro/docs/test.txt",
        )

        indexer = ArchiveIndexer(MagicMock())
        indexer._update_sidecar(item, "uuid-123", 5)

        # Read updated sidecar
        with open(sidecar_path) as f:
            updated = json.load(f)

        assert "llmemory" in updated
        assert updated["llmemory"]["document_id"] == "uuid-123"
        assert updated["llmemory"]["chunks_created"] == 5
        assert "indexed_at" in updated["llmemory"]

        # Original fields preserved
        assert updated["id"] == "doc-001"
        assert updated["source"] == "email"

    def test_update_sidecar_overwrites_existing_llmemory(self, tmp_path: Path) -> None:
        """Test that update_sidecar overwrites existing llmemory section."""
        sidecar_path = tmp_path / "test.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "id": "doc-001",
                    "llmemory": {
                        "indexed_at": "2024-01-01T00:00:00Z",
                        "document_id": "old-uuid",
                        "chunks_created": 2,
                    },
                }
            )
        )

        item = ArchiveItem(
            content_path=tmp_path / "test.txt",
            sidecar_path=sidecar_path,
            entity="jro",
            document_id="doc-001",
            relative_path="jro/docs/test.txt",
        )

        indexer = ArchiveIndexer(MagicMock())
        indexer._update_sidecar(item, "new-uuid", 10)

        with open(sidecar_path) as f:
            updated = json.load(f)

        assert updated["llmemory"]["document_id"] == "new-uuid"
        assert updated["llmemory"]["chunks_created"] == 10


@pytest.mark.asyncio
class TestIndexItem:
    """Test async indexing functionality."""

    async def test_index_item_success(self, tmp_path: Path) -> None:
        """Test successful indexing of an item."""
        # Create content and sidecar
        content_file = tmp_path / "test.txt"
        content_file.write_text("Test content for indexing")

        sidecar_file = tmp_path / "test.json"
        sidecar_file.write_text(json.dumps({"id": "doc-001", "source": "email"}))

        item = ArchiveItem(
            content_path=content_file,
            sidecar_path=sidecar_file,
            entity="jro",
            document_id="doc-001",
            relative_path="jro/docs/test.txt",
            source="email",
        )

        # Mock manager
        mock_doc = MagicMock(spec=Document)
        mock_doc.document_id = "uuid-123"
        mock_chunks = [MagicMock(spec=DocumentChunk) for _ in range(3)]

        mock_manager = AsyncMock()
        mock_manager.process_document.return_value = (mock_doc, mock_chunks)

        indexer = ArchiveIndexer(mock_manager)
        result = await indexer.index_item(item)

        assert result.success is True
        assert result.document_id == "uuid-123"
        assert result.chunks_created == 3
        assert result.error is None

        # Verify process_document was called correctly
        mock_manager.process_document.assert_called_once()
        call_args = mock_manager.process_document.call_args
        assert call_args.kwargs["owner_id"] == "jro"
        assert call_args.kwargs["id_at_origin"] == "doc-001"

    async def test_index_item_extraction_failure(self, tmp_path: Path) -> None:
        """Test handling of text extraction failure."""
        # Create unsupported file type
        content_file = tmp_path / "test.bin"
        content_file.write_bytes(b"\x00\x01\x02")

        sidecar_file = tmp_path / "test.json"
        sidecar_file.write_text(json.dumps({"id": "doc-001"}))

        item = ArchiveItem(
            content_path=content_file,
            sidecar_path=sidecar_file,
            entity="jro",
            document_id="doc-001",
            relative_path="jro/docs/test.bin",
        )

        indexer = ArchiveIndexer(AsyncMock())
        result = await indexer.index_item(item)

        assert result.success is False
        assert "Failed to extract text" in result.error

    async def test_index_items_multiple(self, tmp_path: Path) -> None:
        """Test indexing multiple items."""
        items = []
        for i in range(3):
            content_file = tmp_path / f"test{i}.txt"
            content_file.write_text(f"Content {i}")

            sidecar_file = tmp_path / f"test{i}.json"
            sidecar_file.write_text(json.dumps({"id": f"doc-{i}"}))

            items.append(
                ArchiveItem(
                    content_path=content_file,
                    sidecar_path=sidecar_file,
                    entity="jro",
                    document_id=f"doc-{i}",
                    relative_path=f"jro/docs/test{i}.txt",
                )
            )

        mock_doc = MagicMock(spec=Document)
        mock_doc.document_id = "uuid"
        mock_manager = AsyncMock()
        mock_manager.process_document.return_value = (mock_doc, [])

        indexer = ArchiveIndexer(mock_manager)
        results = await indexer.index_items(items)

        assert len(results) == 3
        assert all(r.success for r in results)
