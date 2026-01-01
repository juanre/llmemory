"""Archive indexer for llmemory.

Handles text extraction and indexing of archive items into llmemory.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .archive import ArchiveItem
from .manager import MemoryManager
from .models import DocumentType

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of indexing an archive item."""

    item: ArchiveItem
    success: bool
    document_id: Optional[str] = None
    chunks_created: int = 0
    error: Optional[str] = None


class ArchiveIndexer:
    """Indexes archive items into llmemory."""

    # Map file extensions to DocumentType
    EXTENSION_TO_TYPE = {
        ".txt": DocumentType.TEXT,
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
        ".html": DocumentType.HTML,
        ".htm": DocumentType.HTML,
        ".pdf": DocumentType.PDF,
    }

    def __init__(self, manager: MemoryManager):
        """Initialize indexer with a MemoryManager instance.

        Args:
            manager: Initialized MemoryManager for database operations
        """
        self.manager = manager

    async def index_item(self, item: ArchiveItem) -> IndexResult:
        """Index a single archive item into llmemory.

        Extracts text content, indexes into llmemory, and updates the sidecar.

        Args:
            item: ArchiveItem to index

        Returns:
            IndexResult with success status and details
        """
        try:
            # Extract text content
            content = self._extract_text(item)
            if content is None:
                return IndexResult(
                    item=item,
                    success=False,
                    error=f"Failed to extract text from {item.content_path}",
                )

            # Determine document type
            doc_type = self._get_document_type(item)

            # Build metadata from sidecar
            metadata = {
                "source": item.source,
                "workflow": item.workflow,
                "mimetype": item.mimetype,
                "archive_path": item.relative_path,
            }
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

            # Index into llmemory using the identity contract
            doc, chunks = await self.manager.process_document(
                owner_id=item.owner_id,
                id_at_origin=item.id_at_origin,
                document_name=item.content_path.name,
                document_type=doc_type,
                content=content,
                metadata=metadata,
            )

            # Update sidecar with llmemory fields
            self._update_sidecar(item, str(doc.document_id), len(chunks))

            return IndexResult(
                item=item,
                success=True,
                document_id=str(doc.document_id),
                chunks_created=len(chunks),
            )

        except Exception as e:
            logger.exception(f"Error indexing {item.relative_path}: {e}")
            return IndexResult(
                item=item,
                success=False,
                error=str(e),
            )

    async def index_items(self, items: list[ArchiveItem]) -> list[IndexResult]:
        """Index multiple archive items.

        Args:
            items: List of ArchiveItems to index

        Returns:
            List of IndexResult objects
        """
        results = []
        for item in items:
            result = await self.index_item(item)
            results.append(result)
        return results

    def _extract_text(self, item: ArchiveItem) -> Optional[str]:
        """Extract text content from an archive item.

        Args:
            item: ArchiveItem to extract text from

        Returns:
            Extracted text content, or None if extraction fails
        """
        suffix = item.content_path.suffix.lower()

        try:
            if suffix in {".txt", ".md", ".markdown"}:
                return self._extract_text_file(item.content_path)
            elif suffix in {".html", ".htm"}:
                return self._extract_html(item.content_path)
            elif suffix == ".pdf":
                return self._extract_pdf(item.content_path)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return None
        except Exception as e:
            logger.exception(f"Error extracting text from {item.content_path}: {e}")
            return None

    def _extract_text_file(self, path: Path) -> Optional[str]:
        """Extract text from a plain text or markdown file."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with fallback encoding
            try:
                return path.read_text(encoding="latin-1")
            except Exception:
                logger.warning(f"Failed to decode {path} with utf-8 or latin-1")
                return None

    def _extract_html(self, path: Path) -> Optional[str]:
        """Extract text from an HTML file.

        Strips HTML tags and returns plain text.
        """
        import re

        try:
            html = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                html = path.read_text(encoding="latin-1")
            except Exception:
                logger.warning(f"Failed to decode HTML {path}")
                return None

        # Remove script and style elements
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", html)

        # Decode HTML entities
        import html as html_module

        text = html_module.unescape(text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _extract_pdf(self, path: Path) -> Optional[str]:
        """Extract text from a PDF file.

        Requires pypdf to be installed.
        """
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("pypdf not installed; cannot extract PDF text")
            return None
        except Exception as e:
            logger.exception(f"Error reading PDF {path}: {e}")
            return None

    def _get_document_type(self, item: ArchiveItem) -> DocumentType:
        """Determine DocumentType from file extension."""
        suffix = item.content_path.suffix.lower()
        return self.EXTENSION_TO_TYPE.get(suffix, DocumentType.TEXT)

    def _update_sidecar(
        self,
        item: ArchiveItem,
        document_id: str,
        chunks_created: int,
    ) -> None:
        """Update sidecar with llmemory indexing fields.

        Adds/updates the 'llmemory' section in the sidecar atomically.

        Args:
            item: ArchiveItem whose sidecar to update
            document_id: llmemory document_id
            chunks_created: Number of chunks created
        """
        try:
            # Read existing sidecar
            with open(item.sidecar_path) as f:
                metadata = json.load(f)

            # Update llmemory section
            metadata["llmemory"] = {
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "document_id": document_id,
                "chunks_created": chunks_created,
            }

            # Write atomically (temp file + rename)
            import tempfile

            temp_fd, temp_path = tempfile.mkstemp(
                dir=item.sidecar_path.parent,
                suffix=".json.tmp",
            )
            try:
                with open(temp_fd, "w") as f:
                    json.dump(metadata, f, indent=2)

                # Atomic rename
                Path(temp_path).rename(item.sidecar_path)
            except Exception:
                # Clean up temp file on error
                Path(temp_path).unlink(missing_ok=True)
                raise

        except Exception as e:
            logger.exception(f"Error updating sidecar {item.sidecar_path}: {e}")
            raise
