"""CLI for llmemory archive indexing and search.

Provides commands to:
- index: Scan ~/Archive and index unindexed items into llmemory
- search: Query the llmemory index
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import click


@click.group()
@click.version_option()
def cli() -> None:
    """llmemory - Archive indexing and search."""
    pass


async def _run_index(
    archive_path: Path,
    entity: Optional[str],
    dry_run: bool,
    verbose: bool,
) -> None:
    """Async implementation of the index command."""
    from llmemory.archive import ArchiveScanner
    from llmemory.indexer import ArchiveIndexer
    from llmemory.manager import MemoryManager

    scanner = ArchiveScanner(archive_path)
    unindexed = scanner.find_unindexed_items(entity=entity)

    if not unindexed:
        click.echo("No unindexed items found.")
        return

    if verbose or dry_run:
        click.echo(f"Found {len(unindexed)} unindexed items:")
        for item in unindexed:
            click.echo(f"  {item.relative_path}")

    if dry_run:
        click.echo("\nDry run - no changes made.")
        return

    # Initialize llmemory
    click.echo("\nConnecting to llmemory database...")
    manager = await MemoryManager.create()

    try:
        indexer = ArchiveIndexer(manager)

        click.echo(f"Indexing {len(unindexed)} items...")
        results = await indexer.index_items(unindexed)

        # Report results
        success_count = sum(1 for r in results if r.success)
        fail_count = sum(1 for r in results if not r.success)
        total_chunks = sum(r.chunks_created for r in results if r.success)

        click.echo(f"\nIndexed {success_count} items ({total_chunks} chunks)")
        if fail_count > 0:
            click.secho(f"Failed: {fail_count} items", fg="red")
            for r in results:
                if not r.success:
                    click.echo(f"  {r.item.relative_path}: {r.error}")
    finally:
        await manager.close()


@cli.command()
@click.option(
    "--archive-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path.home() / "Archive",
    help="Path to archive root (default: ~/Archive)",
)
@click.option(
    "--entity",
    type=str,
    default=None,
    help="Index only this entity (e.g., jro, tsm)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be indexed without actually indexing",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output",
)
def index(
    archive_path: Path,
    entity: Optional[str],
    dry_run: bool,
    verbose: bool,
) -> None:
    """Scan archive and index unindexed items into llmemory.

    Finds items in ~/Archive that are missing llmemory indexing fields
    in their metadata sidecars, extracts text, and indexes them.
    """
    asyncio.run(_run_index(archive_path, entity, dry_run, verbose))


async def _run_search(
    query_text: str,
    entity: Optional[str],
    source: Optional[str],
    document_type: Optional[str],
    date_from: Optional[datetime],
    date_to: Optional[datetime],
    limit: int,
    json_output: bool,
) -> None:
    """Async implementation of the search command."""
    import json

    from llmemory.manager import MemoryManager
    from llmemory.models import SearchQuery, SearchType

    if not entity:
        click.secho("Error: --entity is required", fg="red", err=True)
        raise SystemExit(1)

    # Initialize llmemory
    manager = await MemoryManager.create()

    try:
        # Build metadata filter
        metadata_filter = {}
        if source:
            metadata_filter["source"] = source
        if document_type:
            metadata_filter["document_type"] = document_type

        # Create search query (use text search for simplicity)
        search_query = SearchQuery(
            owner_id=entity,
            query_text=query_text,
            search_type=SearchType.TEXT,
            limit=limit,
            metadata_filter=metadata_filter if metadata_filter else None,
            date_from=date_from,
            date_to=date_to,
        )

        results = await manager.search(search_query)

        if json_output:
            output = {
                "query": query_text,
                "entity": entity,
                "count": len(results),
                "results": [
                    {
                        "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                        "score": r.score,
                        "metadata": r.metadata,
                        "archive_path": r.metadata.get("archive_path") if r.metadata else None,
                    }
                    for r in results
                ],
            }
            click.echo(json.dumps(output, indent=2, default=str))
        else:
            if not results:
                click.echo("No results found.")
            else:
                click.echo(f"Found {len(results)} results:\n")
                for i, r in enumerate(results, 1):
                    archive_path = r.metadata.get("archive_path") if r.metadata else "N/A"
                    excerpt = r.content[:100].replace("\n", " ")
                    if len(r.content) > 100:
                        excerpt += "..."

                    click.echo(f"{i}. [{r.score:.3f}] {archive_path}")
                    click.echo(f"   {excerpt}\n")
    finally:
        await manager.close()


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--entity",
    type=str,
    default=None,
    help="Search only this entity (e.g., jro, tsm) [required]",
)
@click.option(
    "--source",
    type=str,
    default=None,
    help="Filter by source (e.g., slack, gmail)",
)
@click.option(
    "--document-type",
    type=str,
    default=None,
    help="Filter by document type (e.g., invoice, receipt, email)",
)
@click.option(
    "--date-from",
    type=click.DateTime(),
    default=None,
    help="Filter by date (from)",
)
@click.option(
    "--date-to",
    type=click.DateTime(),
    default=None,
    help="Filter by date (to)",
)
@click.option(
    "--limit", "-n",
    type=int,
    default=10,
    help="Maximum number of results (default: 10)",
)
@click.option(
    "--json-output", "-j",
    is_flag=True,
    help="Output results as JSON",
)
def search(
    query: str,
    entity: Optional[str],
    source: Optional[str],
    document_type: Optional[str],
    date_from: Optional[datetime],
    date_to: Optional[datetime],
    limit: int,
    json_output: bool,
) -> None:
    """Search the llmemory index.

    QUERY is the search text. Requires --entity to specify which entity to search.
    """
    asyncio.run(
        _run_search(
            query, entity, source, document_type, date_from, date_to, limit, json_output
        )
    )


if __name__ == "__main__":
    cli()
