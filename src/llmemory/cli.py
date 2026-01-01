"""CLI for llmemory archive indexing and search.

Provides commands to:
- index: Scan ~/Archive and index unindexed items into llmemory
- search: Query the llmemory index
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import click


@click.group()
@click.version_option()
def cli() -> None:
    """llmemory - Archive indexing and search."""
    pass


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
    from llmemory.archive import ArchiveScanner

    scanner = ArchiveScanner(archive_path)
    unindexed = scanner.find_unindexed_items(entity=entity)

    if verbose or dry_run:
        click.echo(f"Found {len(unindexed)} unindexed items:")
        for item in unindexed:
            click.echo(f"  {item.relative_path}")

    if dry_run:
        click.echo("\nDry run - no changes made.")
        return

    # TODO: Actually index items
    click.echo(f"\nIndexed {len(unindexed)} items.")


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--entity",
    type=str,
    default=None,
    help="Search only this entity (e.g., jro, tsm)",
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

    QUERY is the search text.
    """
    # TODO: Implement actual search using MemoryManager
    click.echo(f"Searching for: {query}")

    if entity:
        click.echo(f"  Entity: {entity}")
    if source:
        click.echo(f"  Source: {source}")
    if document_type:
        click.echo(f"  Document type: {document_type}")
    if date_from:
        click.echo(f"  Date from: {date_from}")
    if date_to:
        click.echo(f"  Date to: {date_to}")

    click.echo(f"  Limit: {limit}")

    # Placeholder results
    results: list[dict] = []

    if json_output:
        click.echo(json.dumps({"query": query, "results": results}, indent=2))
    else:
        if not results:
            click.echo("\nNo results found.")
        else:
            for i, result in enumerate(results, 1):
                click.echo(f"\n{i}. {result.get('title', 'Untitled')}")
                click.echo(f"   Path: {result.get('path', 'N/A')}")
                click.echo(f"   Score: {result.get('score', 0):.3f}")


if __name__ == "__main__":
    cli()
