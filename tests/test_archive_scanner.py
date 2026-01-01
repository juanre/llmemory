"""Tests for archive scanner functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from llmemory.archive import ArchiveItem, ArchiveScanner


class TestArchiveScanner:
    """Test archive scanning functionality."""

    def test_find_entities_empty_archive(self, tmp_path: Path) -> None:
        """Test finding entities in an empty archive."""
        scanner = ArchiveScanner(tmp_path)
        entities = scanner.find_entities()
        assert entities == []

    def test_find_entities_with_entities(self, tmp_path: Path) -> None:
        """Test finding entities in archive with entity directories."""
        # Create entity directories
        (tmp_path / "jro").mkdir()
        (tmp_path / "tsm").mkdir()
        (tmp_path / "gsk").mkdir()
        (tmp_path / ".hidden").mkdir()  # Should be ignored

        scanner = ArchiveScanner(tmp_path)
        entities = scanner.find_entities()
        assert entities == ["gsk", "jro", "tsm"]

    def test_find_items_empty_archive(self, tmp_path: Path) -> None:
        """Test finding items in an empty archive."""
        scanner = ArchiveScanner(tmp_path)
        items = scanner.find_items()
        assert items == []

    def test_find_items_with_sidecar(self, tmp_path: Path) -> None:
        """Test finding items with valid sidecar files."""
        # Create archive structure
        entity_dir = tmp_path / "jro" / "docs" / "2024"
        entity_dir.mkdir(parents=True)

        # Create content file
        content_file = entity_dir / "2024-01-15-invoice.pdf"
        content_file.write_bytes(b"PDF content")

        # Create sidecar
        sidecar_file = entity_dir / "2024-01-15-invoice.json"
        sidecar_file.write_text(
            json.dumps(
                {
                    "id": "doc-2024-001",
                    "entity": "jro",
                    "source": "email",
                    "workflow": "invoices",
                    "mimetype": "application/pdf",
                }
            )
        )

        scanner = ArchiveScanner(tmp_path)
        items = scanner.find_items()

        assert len(items) == 1
        item = items[0]
        assert item.entity == "jro"
        assert item.document_id == "doc-2024-001"
        assert item.id_at_origin == "doc-2024-001"
        assert item.source == "email"
        assert item.workflow == "invoices"
        assert item.is_indexed is False
        assert str(item.content_path) == str(content_file)

    def test_find_items_ignores_non_indexable(self, tmp_path: Path) -> None:
        """Test that non-indexable file types are ignored."""
        entity_dir = tmp_path / "jro" / "docs" / "2024"
        entity_dir.mkdir(parents=True)

        # Create image file (not indexable)
        content_file = entity_dir / "2024-01-15-photo.jpg"
        content_file.write_bytes(b"JPEG content")

        sidecar_file = entity_dir / "2024-01-15-photo.json"
        sidecar_file.write_text(json.dumps({"id": "doc-2024-002"}))

        scanner = ArchiveScanner(tmp_path)
        items = scanner.find_items()

        assert len(items) == 0  # JPG is not indexable

    def test_find_items_by_entity(self, tmp_path: Path) -> None:
        """Test filtering items by entity."""
        # Create items for two entities
        for entity in ["jro", "tsm"]:
            entity_dir = tmp_path / entity / "docs" / "2024"
            entity_dir.mkdir(parents=True)

            content_file = entity_dir / "2024-01-15-doc.txt"
            content_file.write_text("Content")

            sidecar_file = entity_dir / "2024-01-15-doc.json"
            sidecar_file.write_text(json.dumps({"id": f"doc-{entity}-001"}))

        scanner = ArchiveScanner(tmp_path)

        # Filter by jro
        jro_items = scanner.find_items(entity="jro")
        assert len(jro_items) == 1
        assert jro_items[0].entity == "jro"

        # Filter by tsm
        tsm_items = scanner.find_items(entity="tsm")
        assert len(tsm_items) == 1
        assert tsm_items[0].entity == "tsm"

    def test_find_unindexed_items(self, tmp_path: Path) -> None:
        """Test finding unindexed items."""
        entity_dir = tmp_path / "jro" / "docs" / "2024"
        entity_dir.mkdir(parents=True)

        # Create unindexed item
        (entity_dir / "unindexed.txt").write_text("Content 1")
        (entity_dir / "unindexed.json").write_text(json.dumps({"id": "doc-001"}))

        # Create indexed item
        (entity_dir / "indexed.txt").write_text("Content 2")
        (entity_dir / "indexed.json").write_text(
            json.dumps(
                {
                    "id": "doc-002",
                    "llmemory": {
                        "indexed_at": "2024-01-15T10:00:00Z",
                        "document_id": "uuid-123",
                    },
                }
            )
        )

        scanner = ArchiveScanner(tmp_path)
        unindexed = scanner.find_unindexed_items()

        assert len(unindexed) == 1
        assert unindexed[0].document_id == "doc-001"
        assert unindexed[0].is_indexed is False

    def test_find_items_invalid_sidecar(self, tmp_path: Path) -> None:
        """Test that invalid sidecars are skipped."""
        entity_dir = tmp_path / "jro" / "docs" / "2024"
        entity_dir.mkdir(parents=True)

        # Create content file
        content_file = entity_dir / "2024-01-15-doc.txt"
        content_file.write_text("Content")

        # Create invalid sidecar (not valid JSON)
        sidecar_file = entity_dir / "2024-01-15-doc.json"
        sidecar_file.write_text("not valid json")

        scanner = ArchiveScanner(tmp_path)
        items = scanner.find_items()

        assert len(items) == 0  # Invalid sidecar is skipped

    def test_find_items_missing_id(self, tmp_path: Path) -> None:
        """Test that sidecars without id field are skipped."""
        entity_dir = tmp_path / "jro" / "docs" / "2024"
        entity_dir.mkdir(parents=True)

        content_file = entity_dir / "2024-01-15-doc.txt"
        content_file.write_text("Content")

        # Sidecar without id field
        sidecar_file = entity_dir / "2024-01-15-doc.json"
        sidecar_file.write_text(json.dumps({"source": "email"}))

        scanner = ArchiveScanner(tmp_path)
        items = scanner.find_items()

        assert len(items) == 0

    def test_streams_directory(self, tmp_path: Path) -> None:
        """Test scanning streams directory structure."""
        stream_dir = tmp_path / "jro" / "streams" / "slack" / "general" / "2024"
        stream_dir.mkdir(parents=True)

        content_file = stream_dir / "2024-01-15-transcript.md"
        content_file.write_text("# Slack transcript\n\nContent here")

        sidecar_file = stream_dir / "2024-01-15-transcript.json"
        sidecar_file.write_text(
            json.dumps(
                {
                    "id": "slack-transcript-001",
                    "source": "slack",
                }
            )
        )

        scanner = ArchiveScanner(tmp_path)
        items = scanner.find_items()

        assert len(items) == 1
        item = items[0]
        assert item.document_id == "slack-transcript-001"
        assert item.source == "slack"
        assert "streams/slack/general" in item.relative_path


class TestArchiveItem:
    """Test ArchiveItem dataclass."""

    def test_id_at_origin(self) -> None:
        """Test id_at_origin property returns document_id."""
        item = ArchiveItem(
            content_path=Path("/archive/jro/docs/2024/doc.txt"),
            sidecar_path=Path("/archive/jro/docs/2024/doc.json"),
            entity="jro",
            document_id="doc-2024-001",
            relative_path="jro/docs/2024/doc.txt",
        )

        assert item.id_at_origin == "doc-2024-001"

    def test_owner_id(self) -> None:
        """Test owner_id property returns entity."""
        item = ArchiveItem(
            content_path=Path("/archive/jro/docs/2024/doc.txt"),
            sidecar_path=Path("/archive/jro/docs/2024/doc.json"),
            entity="jro",
            document_id="doc-2024-001",
            relative_path="jro/docs/2024/doc.txt",
        )

        assert item.owner_id == "jro"
