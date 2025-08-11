"""Advanced chunking strategies for document processing."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import tiktoken

from .config import get_config
from .exceptions import ChunkingError
from .models import ChunkingConfig, DocumentChunk, DocumentType


@dataclass
class ChunkMetadata:
    """Metadata for a chunk during processing."""

    start_pos: int
    end_pos: int
    token_count: int
    hierarchy_level: int
    parent_context: Optional[str] = None


class HierarchicalChunker:
    """Implements hierarchical chunking with parent-child relationships."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker with configuration.

        Args:
            config: Chunking configuration, uses defaults if not provided
        """
        self.config = config or ChunkingConfig()
        self.global_config = get_config().chunking
        try:
            self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
        except Exception as e:
            raise ChunkingError(
                f"Failed to initialize tokenizer: {str(e)}", strategy="hierarchical"
            )

    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_type: DocumentType,
        base_metadata: Optional[Dict] = None,
    ) -> List[DocumentChunk]:
        """
        Create hierarchical chunks with parent-child relationships.

        Args:
            text: Document text to chunk
            document_id: ID of the document
            document_type: Type of document for specific handling
            base_metadata: Base metadata to include in all chunks

        Returns:
            List of DocumentChunk objects with hierarchy
        """
        chunks = []
        base_metadata = base_metadata or {}

        # Get optimal chunk configuration for document type
        chunk_config = self._get_chunk_config(document_type, len(text))

        # Create parent chunks
        parent_chunks = self._create_chunks(
            text, chunk_config["parent_size"], chunk_config["overlap"]
        )

        for p_idx, parent_data in enumerate(parent_chunks):
            import hashlib

            content_hash = hashlib.sha256(parent_data["text"].encode()).hexdigest()

            parent_chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=p_idx,
                chunk_level=2,  # Parent level
                content=parent_data["text"],
                content_hash=content_hash,
                token_count=parent_data["token_count"],
                metadata={
                    **base_metadata,
                    "hierarchy_level": "parent",
                    "chunk_type": "parent",
                    "start_pos": parent_data["start"],
                    "end_pos": parent_data["end"],
                },
            )
            chunks.append(parent_chunk)

            # Create child chunks within parent
            child_chunks = self._create_chunks(
                parent_data["text"],
                chunk_config["child_size"],
                chunk_config["child_overlap"],
            )

            for c_idx, child_data in enumerate(child_chunks):
                # Create preview of parent content for context
                parent_preview = (
                    parent_data["text"][:150] + "..."
                    if len(parent_data["text"]) > 150
                    else parent_data["text"]
                )

                child_content_hash = hashlib.sha256(
                    child_data["text"].encode()
                ).hexdigest()

                child_chunk = DocumentChunk(
                    document_id=document_id,
                    parent_chunk_id=parent_chunk.chunk_id,
                    chunk_index=p_idx * 1000 + c_idx,  # Ensure unique ordering
                    chunk_level=1,  # Child level
                    content=child_data["text"],
                    content_hash=child_content_hash,
                    token_count=child_data["token_count"],
                    metadata={
                        **base_metadata,
                        "hierarchy_level": "child",
                        "chunk_type": "child",
                        "parent_context": parent_preview,
                        "start_pos": parent_data["start"] + child_data["start"],
                        "end_pos": parent_data["start"] + child_data["end"],
                    },
                )
                chunks.append(child_chunk)

        return chunks

    def _create_chunks(self, text: str, size: int, overlap: int) -> List[Dict]:
        """
        Create chunks with specified size and overlap.

        Args:
            text: Text to chunk
            size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens

        Returns:
            List of chunk data dictionaries
        """
        tokens = self.tokenizer.encode(text)
        chunks = []

        if len(tokens) <= size:
            # Text fits in a single chunk
            return [
                {"text": text, "start": 0, "end": len(text), "token_count": len(tokens)}
            ]

        i = 0
        start_char = 0  # Track character position efficiently

        while i < len(tokens):
            # Determine chunk boundaries
            end_token_idx = min(i + size, len(tokens))

            # Try to find sentence boundary near the end
            chunk_tokens = tokens[i:end_token_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Look for sentence boundary
            if end_token_idx < len(tokens):
                chunk_text = self._find_sentence_boundary(chunk_text)
                # Re-encode to get actual token count
                chunk_tokens = self.tokenizer.encode(chunk_text)

            # Calculate character positions
            end_char = start_char + len(chunk_text)

            chunks.append(
                {
                    "text": chunk_text,
                    "start": start_char,
                    "end": end_char,
                    "token_count": len(chunk_tokens),
                }
            )

            # Move forward with overlap
            tokens_to_advance = len(chunk_tokens) - overlap
            if tokens_to_advance > 0 and i + tokens_to_advance < len(tokens):
                # Decode the tokens we're skipping to update character position
                skipped_text = self.tokenizer.decode(tokens[i : i + tokens_to_advance])
                start_char += len(skipped_text)
                i += tokens_to_advance
            else:
                # Last chunk
                break

        return chunks

    def _find_sentence_boundary(self, text: str) -> str:
        """
        Find the last complete sentence in the text.

        Args:
            text: Text to find sentence boundary in

        Returns:
            Text up to the last complete sentence
        """
        # Look for sentence endings
        sentence_endings = [". ", "! ", "? ", "\n\n"]

        last_pos = -1
        for ending in sentence_endings:
            pos = text.rfind(ending)
            if pos > last_pos:
                last_pos = pos + len(ending) - 1

        if last_pos > len(text) * 0.8:  # Found ending near the end
            return text[: last_pos + 1]

        # If no good sentence boundary, look for other boundaries
        boundaries = ["\n", ", ", "; ", ": "]
        for boundary in boundaries:
            pos = text.rfind(boundary)
            if pos > len(text) * 0.8:
                return text[: pos + 1]

        return text

    def _get_chunk_config(self, doc_type: DocumentType, content_length: int) -> Dict:
        """
        Get optimal chunking parameters based on document type and length.

        Based on agent-engine integration requirements:
        - EMAIL: Small chunks for quick retrieval (300/150 tokens)
        - REPORT: Medium chunks for comprehensive content (600/300 tokens)
        - CODE: Larger chunks to preserve context (800/400 tokens)
        - CHAT: Very small chunks for conversation snippets (200/100 tokens)

        Args:
            doc_type: Type of document
            content_length: Length of content in characters

        Returns:
            Dictionary with chunking parameters
        """
        # Configurations from agent-engine integration plan and design document
        configs = {
            # Email: Quick retrieval of specific info
            DocumentType.EMAIL: {
                "parent_size": 300,
                "child_size": 150,
                "overlap": 25,
                "child_overlap": 15,
            },
            # Report: Comprehensive sections
            DocumentType.REPORT: {
                "parent_size": 600,
                "child_size": 300,
                "overlap": 50,
                "child_overlap": 25,
            },
            # Business Report: Larger chunks for financial/analytical content
            DocumentType.BUSINESS_REPORT: {
                "parent_size": 600,
                "child_size": 300,
                "overlap": 50,
                "child_overlap": 25,
            },
            # Code: Preserve function/class context
            DocumentType.CODE: {
                "parent_size": 800,
                "child_size": 400,
                "overlap": 60,
                "child_overlap": 30,
            },
            # Chat: Individual messages or exchanges
            DocumentType.CHAT: {
                "parent_size": 200,
                "child_size": 100,
                "overlap": 20,
                "child_overlap": 10,
            },
            # Presentation: Slide-based content
            DocumentType.PRESENTATION: {
                "parent_size": 400,
                "child_size": 200,
                "overlap": 30,
                "child_overlap": 15,
            },
            # Legal Document: Preserve clause/section context
            DocumentType.LEGAL_DOCUMENT: {
                "parent_size": 500,
                "child_size": 250,
                "overlap": 40,
                "child_overlap": 20,
            },
            # Technical Documentation: API docs, specifications
            DocumentType.TECHNICAL_DOC: {
                "parent_size": 800,
                "child_size": 400,
                "overlap": 60,
                "child_overlap": 30,
            },
            # PDF: Balanced for various content
            DocumentType.PDF: {
                "parent_size": 800,
                "child_size": 400,
                "overlap": 60,
                "child_overlap": 30,
            },
            # Markdown: Structured documents
            DocumentType.MARKDOWN: {
                "parent_size": 600,
                "child_size": 300,
                "overlap": 50,
                "child_overlap": 25,
            },
            # General text
            DocumentType.TEXT: {
                "parent_size": 600,
                "child_size": 300,
                "overlap": 50,
                "child_overlap": 25,
            },
            # HTML: Web content
            DocumentType.HTML: {
                "parent_size": 600,
                "child_size": 300,
                "overlap": 50,
                "child_overlap": 25,
            },
            # DOCX: Office documents
            DocumentType.DOCX: {
                "parent_size": 700,
                "child_size": 350,
                "overlap": 50,
                "child_overlap": 25,
            },
        }

        # Get base config or use default
        config = configs.get(
            doc_type,
            {"parent_size": 600, "child_size": 300, "overlap": 50, "child_overlap": 25},
        )

        # Adjust for document length as specified in integration plan
        if content_length < 1000:
            # Very short documents - halve the chunk sizes
            config = {k: v // 2 for k, v in config.items()}
        elif content_length > 50000:
            # Very long documents - increase parent chunks only
            config["parent_size"] = min(int(config["parent_size"] * 1.5), 1200)
            config["child_size"] = min(int(config["child_size"] * 1.2), 500)

        return config


class SemanticChunker:
    """Implements semantic chunking based on content structure."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

    def chunk_markdown(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk markdown based on semantic structure (headers, sections).

        Args:
            text: Markdown text to chunk
            document_id: ID of the document

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Split by headers
        header_pattern = r"^(#{1,6})\s+(.+)$"
        sections = []
        current_section = []
        current_level = 0

        for line in text.split("\n"):
            header_match = re.match(header_pattern, line)
            if header_match:
                # Save previous section
                if current_section:
                    sections.append(
                        {"level": current_level, "content": "\n".join(current_section)}
                    )

                # Start new section
                current_level = len(header_match.group(1))
                current_section = [line]
            else:
                current_section.append(line)

        # Add final section
        if current_section:
            sections.append(
                {"level": current_level, "content": "\n".join(current_section)}
            )

        # Create hierarchical chunks from sections
        parent_chunk = None
        for idx, section in enumerate(sections):
            content = section["content"]
            level = section["level"]

            # Create chunk
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=idx,
                chunk_level=max(0, 6 - level),  # Invert level (H1=5, H6=0)
                content=content,
                token_count=len(self.tokenizer.encode(content)),
                metadata={"section_level": level, "chunk_type": "section"},
            )

            # Set parent relationship
            if (
                level > 1
                and parent_chunk
                and parent_chunk.chunk_level > chunk.chunk_level
            ):
                chunk.parent_chunk_id = parent_chunk.chunk_id
            elif level == 1:
                parent_chunk = chunk

            chunks.append(chunk)

        return chunks

    def chunk_code(
        self, text: str, document_id: str, language: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Chunk code based on semantic structure (functions, classes).

        Args:
            text: Code text to chunk
            document_id: ID of the document
            language: Programming language (if known)

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Simple heuristic for now - can be enhanced with language-specific parsing
        # Look for function/class definitions
        function_pattern = r"^(?:def|function|func|fn)\s+(\w+)"
        class_pattern = r"^(?:class|struct|interface)\s+(\w+)"

        lines = text.split("\n")
        current_chunk = []
        current_type = None
        chunk_idx = 0

        for i, line in enumerate(lines):
            if re.match(function_pattern, line) or re.match(class_pattern, line):
                # Save previous chunk
                if current_chunk:
                    content = "\n".join(current_chunk)
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        chunk_level=1,
                        content=content,
                        token_count=len(self.tokenizer.encode(content)),
                        metadata={
                            "chunk_type": current_type or "code",
                            "language": language,
                        },
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

                # Start new chunk
                current_chunk = [line]
                current_type = (
                    "function" if "def" in line or "function" in line else "class"
                )
            else:
                current_chunk.append(line)

        # Add final chunk
        if current_chunk:
            content = "\n".join(current_chunk)
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk_idx,
                chunk_level=1,
                content=content,
                token_count=len(self.tokenizer.encode(content)),
                metadata={"chunk_type": current_type or "code", "language": language},
            )
            chunks.append(chunk)

        return chunks

    def chunk_email(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk email based on email structure (headers, body, signatures).

        Args:
            text: Email text to chunk
            document_id: ID of the document

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Email patterns
        header_pattern = r"^(From|To|Subject|Date|Cc|Bcc):\s*(.+)$"
        reply_pattern = r"^On .+ wrote:$"
        signature_pattern = r"^(--|Best regards|Sincerely|Thanks)"

        sections = {"headers": [], "body": [], "quoted": [], "signature": []}

        current_section = "headers"
        in_headers = True

        lines = text.split("\n")
        for i, line in enumerate(lines):
            # Check for headers
            if in_headers and re.match(header_pattern, line):
                sections["headers"].append(line)
            # Empty line after headers marks body start
            elif in_headers and line.strip() == "" and sections["headers"]:
                in_headers = False
                current_section = "body"
            # Check for quoted reply
            elif re.match(reply_pattern, line):
                current_section = "quoted"
                sections["quoted"].append(line)
            # Check for signature
            elif re.match(signature_pattern, line):
                current_section = "signature"
                sections["signature"].append(line)
            else:
                sections[current_section].append(line)

        # Create chunks for each section
        chunk_idx = 0

        # Headers chunk
        if sections["headers"]:
            header_content = "\n".join(sections["headers"])
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    chunk_level=2,  # Parent level
                    content=header_content,
                    token_count=len(self.tokenizer.encode(header_content)),
                    metadata={"chunk_type": "email_headers", "section": "headers"},
                )
            )
            chunk_idx += 1

        # Body chunks (may be multiple if long)
        if sections["body"]:
            body_content = "\n".join(sections["body"]).strip()
            # Split body into smaller chunks if needed
            body_chunks = self._create_text_chunks(body_content, max_tokens=150)

            for i, chunk_text in enumerate(body_chunks):
                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        chunk_level=1,  # Child level
                        content=chunk_text,
                        token_count=len(self.tokenizer.encode(chunk_text)),
                        metadata={
                            "chunk_type": "email_body",
                            "section": "body",
                            "part": i + 1,
                        },
                    )
                )
                chunk_idx += 1

        # Quoted text chunk
        if sections["quoted"]:
            quoted_content = "\n".join(sections["quoted"])
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    chunk_level=1,
                    content=quoted_content,
                    token_count=len(self.tokenizer.encode(quoted_content)),
                    metadata={"chunk_type": "email_quoted", "section": "quoted"},
                )
            )
            chunk_idx += 1

        # Signature chunk
        if sections["signature"]:
            sig_content = "\n".join(sections["signature"])
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    chunk_level=0,  # Lowest priority
                    content=sig_content,
                    token_count=len(self.tokenizer.encode(sig_content)),
                    metadata={"chunk_type": "email_signature", "section": "signature"},
                )
            )

        return chunks

    def chunk_chat(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk chat/conversation based on messages and turns.

        Args:
            text: Chat text to chunk
            document_id: ID of the document

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Common chat patterns
        timestamp_pattern = r"^\[?(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]?\s*"
        username_pattern = r"^([^:]+):\s*"
        system_msg_pattern = r"^\*\*\*\s*(.+)\s*\*\*\*$"

        messages = []
        current_message = {
            "timestamp": None,
            "user": None,
            "content": [],
            "type": "message",
        }

        lines = text.split("\n")
        for line in lines:
            # Check for timestamp
            timestamp_match = re.match(timestamp_pattern, line)
            if timestamp_match:
                line = line[timestamp_match.end() :]
                current_message["timestamp"] = timestamp_match.group(1)

            # Check for username
            username_match = re.match(username_pattern, line)
            if username_match:
                # Save previous message if exists
                if current_message["content"]:
                    messages.append(current_message)
                    current_message = {
                        "timestamp": current_message.get("timestamp"),
                        "user": None,
                        "content": [],
                        "type": "message",
                    }

                current_message["user"] = username_match.group(1)
                line = line[username_match.end() :]

            # Check for system message
            if re.match(system_msg_pattern, line):
                current_message["type"] = "system"

            # Add content
            if line.strip():
                current_message["content"].append(line.strip())

        # Add last message
        if current_message["content"]:
            messages.append(current_message)

        # Create chunks from messages
        for idx, msg in enumerate(messages):
            content = "\n".join(msg["content"])

            # Create metadata
            metadata = {
                "chunk_type": "chat_message",
                "message_type": msg["type"],
                "message_index": idx,
            }

            if msg["user"]:
                metadata["user"] = msg["user"]
            if msg["timestamp"]:
                metadata["timestamp"] = msg["timestamp"]

            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=idx,
                chunk_level=1,
                content=content,
                token_count=len(self.tokenizer.encode(content)),
                metadata=metadata,
            )
            chunks.append(chunk)

        # Group messages into conversation turns if many messages
        if len(chunks) > 10:
            # Create parent chunks for conversation turns
            turn_size = 5  # Group 5 messages per turn
            parent_chunks = []

            for i in range(0, len(messages), turn_size):
                turn_messages = messages[i : i + turn_size]
                turn_content = "\n".join(
                    [
                        f"{msg.get('user', 'System')}: {' '.join(msg['content'])}"
                        for msg in turn_messages
                    ]
                )

                parent_chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=len(chunks) + len(parent_chunks),
                    chunk_level=2,  # Parent level
                    content=turn_content[:500],  # Limit parent chunk size
                    token_count=len(self.tokenizer.encode(turn_content[:500])),
                    metadata={
                        "chunk_type": "chat_turn",
                        "turn_index": i // turn_size,
                        "message_count": len(turn_messages),
                    },
                )
                parent_chunks.append(parent_chunk)

                # Update child chunks with parent reference
                for j in range(i, min(i + turn_size, len(chunks))):
                    chunks[j].parent_chunk_id = parent_chunk.chunk_id

            chunks.extend(parent_chunks)

        return chunks

    def _create_text_chunks(self, text: str, max_tokens: int = 300) -> List[str]:
        """
        Create simple text chunks with a maximum token limit.

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        current_chunk = []
        current_count = 0

        sentences = text.split(". ")
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence + ".")

            if current_count + len(sentence_tokens) > max_tokens and current_chunk:
                # Save current chunk
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentence]
                current_count = len(sentence_tokens)
            else:
                current_chunk.append(sentence)
                current_count += len(sentence_tokens)

        # Add last chunk
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        return chunks


def get_chunker(
    strategy: str = "hierarchical", config: Optional[ChunkingConfig] = None
):
    """
    Factory function to get the appropriate chunker.

    Args:
        strategy: Chunking strategy to use
        config: Optional configuration

    Returns:
        Chunker instance
    """
    if strategy == "hierarchical":
        return HierarchicalChunker(config)
    elif strategy == "semantic":
        return SemanticChunker(config)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
