#!/usr/bin/env python3
"""
Migration script to compute and store content hashes for existing documents.

This enables incremental ingestion for documents that were ingested before
the content_hash field was added. Hashes are computed from the source files
in the documents folder to match what incremental ingestion will compute.

Usage:
    uv run python -m src.ingestion.migrate_hashes -d ./documents
    uv run python -m src.ingestion.migrate_hashes -d ./documents --dry-run
    uv run python -m src.ingestion.migrate_hashes -d ./documents --force
"""

import asyncio
import logging
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

from src.ingestion.ingest import compute_content_hash
from src.settings import load_settings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_document(file_path: str) -> Optional[str]:
    """
    Read document content from file - supports multiple formats via Docling.

    This mirrors the logic in DocumentIngestionPipeline._read_document but
    without requiring the full pipeline infrastructure.

    Args:
        file_path: Path to the document file

    Returns:
        Markdown content string, or None if reading fails
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    # Audio formats - transcribe with Whisper ASR
    audio_formats = ['.mp3', '.wav', '.m4a', '.flac']
    if file_ext in audio_formats:
        try:
            from docling.document_converter import (
                DocumentConverter,
                AudioFormatOption
            )
            from docling.datamodel.pipeline_options import AsrPipelineOptions
            from docling.datamodel import asr_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.pipeline.asr_pipeline import AsrPipeline

            audio_path = Path(file_path).resolve()
            logger.info(f"Transcribing audio: {audio_path.name}")

            pipeline_options = AsrPipelineOptions()
            pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

            converter = DocumentConverter(
                format_options={
                    InputFormat.AUDIO: AudioFormatOption(
                        pipeline_cls=AsrPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )

            result = converter.convert(audio_path)
            return result.document.export_to_markdown()

        except Exception as e:
            logger.error(f"Failed to transcribe {file_path}: {e}")
            return None

    # Docling-supported formats (convert to markdown)
    docling_formats = [
        '.pdf', '.docx', '.doc', '.pptx', '.ppt',
        '.xlsx', '.xls', '.html', '.htm',
        '.md', '.markdown'
    ]

    if file_ext in docling_formats:
        try:
            from docling.document_converter import DocumentConverter

            logger.debug(f"Converting {file_ext}: {os.path.basename(file_path)}")

            converter = DocumentConverter()
            result = converter.convert(file_path)
            return result.document.export_to_markdown()

        except Exception as e:
            logger.error(f"Failed to convert {file_path} with Docling: {e}")
            # Fall back to raw text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                return None

    # Plain text formats
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception:
            return None


async def migrate_hashes(
    documents_folder: str,
    dry_run: bool = False,
    force: bool = False
) -> None:
    """
    Compute and store content_hash for documents.

    Reads the source files from the documents folder and computes hashes
    from the converted content (same as ingestion does).

    Args:
        documents_folder: Path to the documents folder
        dry_run: If True, only report what would be updated without making changes
        force: If True, overwrite existing hashes with freshly computed ones
    """
    settings = load_settings()

    # Verify documents folder exists
    if not os.path.exists(documents_folder):
        logger.error(f"Documents folder not found: {documents_folder}")
        return

    logger.info(f"Documents folder: {documents_folder}")
    logger.info("Connecting to MongoDB...")

    client: AsyncMongoClient = AsyncMongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000
    )

    try:
        # Verify connection
        await client.admin.command("ping")
        logger.info(f"Connected to database: {settings.mongodb_database}")

        db = client[settings.mongodb_database]
        documents_collection = db[settings.mongodb_collection_documents]

        # Find documents to process
        if force:
            # All documents with a source path
            query = {"source": {"$exists": True}}
            total_docs = await documents_collection.count_documents(query)
            logger.info(f"FORCE mode: will recompute hashes for {total_docs} documents")
        else:
            # Only documents missing content_hash
            query = {"content_hash": {"$exists": False}}
            total_docs = await documents_collection.count_documents(query)

            if total_docs == 0:
                logger.info("All documents already have content_hash. Nothing to migrate.")
                logger.info("Use --force to recompute all hashes from source files.")
                return

            logger.info(f"Found {total_docs} documents without content_hash")

        if dry_run:
            logger.info("DRY RUN - no changes will be made")

        # Process documents
        updated = 0
        skipped = 0
        errors = 0
        cursor = documents_collection.find(query, {"_id": 1, "source": 1, "title": 1})

        async for doc in cursor:
            source = doc.get("source")
            title = doc.get("title", source)

            if not source:
                logger.warning(f"Document {doc['_id']} has no source path, skipping")
                skipped += 1
                continue

            # Construct full file path
            file_path = os.path.join(documents_folder, source)

            if not os.path.exists(file_path):
                logger.warning(f"Source file not found: {file_path} (doc: {title})")
                skipped += 1
                continue

            try:
                # Read and convert the file (same as ingestion)
                content = read_document(file_path)

                if not content:
                    logger.warning(f"Empty content for {file_path}, skipping")
                    skipped += 1
                    continue

                content_hash = compute_content_hash(content)

                if dry_run:
                    logger.info(f"  Would update: {title} -> {content_hash[:16]}...")
                else:
                    await documents_collection.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {
                            "content_hash": content_hash,
                            "hash_migrated_at": datetime.now()
                        }}
                    )
                    updated += 1

                    if updated % 10 == 0:
                        logger.info(f"Progress: {updated}/{total_docs} documents updated")

            except Exception as e:
                logger.error(f"Failed to process {title}: {e}")
                errors += 1

        if dry_run:
            logger.info(
                f"DRY RUN complete: {total_docs - skipped} would be updated, "
                f"{skipped} skipped, {errors} errors"
            )
        else:
            logger.info(
                f"Migration complete: {updated} updated, {skipped} skipped, {errors} errors"
            )

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    finally:
        await client.close()
        logger.info("Connection closed")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate existing documents to add content_hash field"
    )
    parser.add_argument(
        "--documents", "-d",
        required=True,
        help="Path to documents folder (required)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Recompute and overwrite all hashes, even if they already exist"
    )
    args = parser.parse_args()

    await migrate_hashes(
        documents_folder=args.documents,
        dry_run=args.dry_run,
        force=args.force
    )


if __name__ == "__main__":
    asyncio.run(main())
