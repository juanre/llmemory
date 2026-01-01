"""Archive scanner for llmemory indexing.

Scans the filesystem archive (~/Archive) to find items that need indexing.
Follows the archive-protocol structure:
- Classified: {base_path}/{entity}/{subdirectory}/{YYYY}/{YYYY}-{MM}-{DD}-{normalized}.{ext}
- Streams: {base_path}/{entity}/streams/{stream_name...}/{YYYY}/{YYYY}-{MM}-{DD}-{normalized}.{ext}
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ArchiveItem:
    """Represents an item in the archive."""

    content_path: Path
    sidecar_path: Path
    entity: str
    document_id: str  # id field from sidecar
    relative_path: str

    # Parsed from sidecar
    source: Optional[str] = None
    workflow: Optional[str] = None
    mimetype: Optional[str] = None

    # llmemory indexing status
    is_indexed: bool = False
    indexed_at: Optional[str] = None

    @property
    def id_at_origin(self) -> str:
        """Archive-protocol document_id used as llmemory id_at_origin."""
        return self.document_id


class ArchiveScanner:
    """Scans the archive filesystem to find items for indexing."""

    # Extensions that have extractable text content
    INDEXABLE_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".pdf",
        ".json",  # Some JSON files contain text content
    }

    def __init__(self, archive_path: Path):
        """Initialize scanner with archive root path.

        Args:
            archive_path: Path to archive root (typically ~/Archive)
        """
        self.archive_path = archive_path

    def find_entities(self) -> list[str]:
        """Find all entity directories in the archive.

        Returns:
            List of entity names (e.g., ['jro', 'tsm', 'gsk'])
        """
        entities = []
        if not self.archive_path.exists():
            return entities

        for entry in self.archive_path.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                entities.append(entry.name)

        return sorted(entities)

    def find_items(self, entity: Optional[str] = None) -> list[ArchiveItem]:
        """Find all items in the archive.

        Args:
            entity: If provided, only scan this entity's directory

        Returns:
            List of ArchiveItem objects
        """
        items = []

        if entity:
            entities = [entity] if (self.archive_path / entity).exists() else []
        else:
            entities = self.find_entities()

        for ent in entities:
            entity_path = self.archive_path / ent
            items.extend(self._scan_entity(entity_path, ent))

        return items

    def find_unindexed_items(self, entity: Optional[str] = None) -> list[ArchiveItem]:
        """Find items that are not yet indexed in llmemory.

        An item is unindexed if its sidecar metadata doesn't have
        the 'llmemory' section with 'indexed_at' field.

        Args:
            entity: If provided, only scan this entity's directory

        Returns:
            List of unindexed ArchiveItem objects
        """
        all_items = self.find_items(entity=entity)
        return [item for item in all_items if not item.is_indexed]

    def _scan_entity(self, entity_path: Path, entity: str) -> list[ArchiveItem]:
        """Scan an entity directory for archive items.

        Args:
            entity_path: Path to entity directory
            entity: Entity name

        Returns:
            List of ArchiveItem objects
        """
        items = []

        # Walk through all subdirectories
        for sidecar_path in entity_path.rglob("*.json"):
            # Skip if this is not a sidecar (must have matching content file)
            content_path = self._find_content_file(sidecar_path)
            if content_path is None:
                continue

            # Skip non-indexable file types
            if content_path.suffix.lower() not in self.INDEXABLE_EXTENSIONS:
                continue

            item = self._parse_sidecar(
                sidecar_path=sidecar_path,
                content_path=content_path,
                entity=entity,
            )
            if item:
                items.append(item)

        return items

    def _find_content_file(self, sidecar_path: Path) -> Optional[Path]:
        """Find the content file for a sidecar.

        The content file has the same stem but different extension.

        Args:
            sidecar_path: Path to .json sidecar file

        Returns:
            Path to content file, or None if not found
        """
        stem = sidecar_path.stem
        parent = sidecar_path.parent

        # Look for files with same stem but different extension
        for candidate in parent.iterdir():
            if candidate.is_file() and candidate.stem == stem and candidate.suffix != ".json":
                return candidate

        return None

    def _parse_sidecar(
        self,
        sidecar_path: Path,
        content_path: Path,
        entity: str,
    ) -> Optional[ArchiveItem]:
        """Parse a sidecar file and create an ArchiveItem.

        Args:
            sidecar_path: Path to .json sidecar
            content_path: Path to content file
            entity: Entity name

        Returns:
            ArchiveItem, or None if sidecar is invalid
        """
        try:
            with open(sidecar_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        # Required fields
        document_id = metadata.get("id")
        if not document_id:
            return None

        # Check llmemory indexing status
        llmemory_info = metadata.get("llmemory", {})
        is_indexed = "indexed_at" in llmemory_info
        indexed_at = llmemory_info.get("indexed_at")

        # Relative path from archive root
        relative_path = str(content_path.relative_to(self.archive_path))

        return ArchiveItem(
            content_path=content_path,
            sidecar_path=sidecar_path,
            entity=entity,
            document_id=document_id,
            relative_path=relative_path,
            source=metadata.get("source"),
            workflow=metadata.get("workflow"),
            mimetype=metadata.get("mimetype"),
            is_indexed=is_indexed,
            indexed_at=indexed_at,
        )
