#!/usr/bin/env python3
"""
Migration script to compute and store content hashes for existing documents.

This enables incremental ingestion for documents that were ingested before
the content_hash field was added. Hashes are computed from the raw source
files in the documents folder.

Usage:
    uv run python -m src.ingestion.migrate_hashes -d ./documents
    uv run python -m src.ingestion.migrate_hashes -d ./documents --dry-run
    uv run python -m src.ingestion.migrate_hashes -d ./documents --force
"""

import asyncio
import hashlib
import logging
import argparse
import os
from datetime import datetime

from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

from src.settings import load_settings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of a file's raw bytes.

    Args:
        file_path: Path to the file

    Returns:
        Hex digest of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


async def migrate_hashes(
    documents_folder: str,
    dry_run: bool = False,
    force: bool = False
) -> None:
    """
    Compute and store content_hash for documents from raw source files.

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
                content_hash = compute_file_hash(file_path)

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
