#!/usr/bin/env python3
"""
Migration script to compute and store content hashes for existing documents.

This enables incremental ingestion for documents that were ingested before
the content_hash field was added.

Usage:
    uv run python -m src.ingestion.migrate_hashes
    uv run python -m src.ingestion.migrate_hashes --dry-run
"""

import asyncio
import logging
import argparse
from datetime import datetime

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


async def migrate_hashes(dry_run: bool = False) -> None:
    """
    Compute and store content_hash for all documents missing it.

    Args:
        dry_run: If True, only report what would be updated without making changes
    """
    settings = load_settings()

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

        # Find documents without content_hash
        query = {"content_hash": {"$exists": False}}
        total_missing = await documents_collection.count_documents(query)

        if total_missing == 0:
            logger.info("All documents already have content_hash. Nothing to migrate.")
            return

        logger.info(f"Found {total_missing} documents without content_hash")

        if dry_run:
            logger.info("DRY RUN - no changes will be made")
            # Show sample of documents that would be updated
            cursor = documents_collection.find(query, {"_id": 1, "title": 1, "source": 1}).limit(10)
            async for doc in cursor:
                logger.info(f"  Would update: {doc.get('title', doc.get('source', doc['_id']))}")
            if total_missing > 10:
                logger.info(f"  ... and {total_missing - 10} more")
            return

        # Process documents in batches
        updated = 0
        errors = 0
        cursor = documents_collection.find(query, {"_id": 1, "content": 1, "title": 1})

        async for doc in cursor:
            try:
                content = doc.get("content", "")
                if not content:
                    logger.warning(f"Document {doc['_id']} has no content, skipping")
                    continue

                content_hash = compute_content_hash(content)

                await documents_collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "content_hash": content_hash,
                        "hash_migrated_at": datetime.now()
                    }}
                )

                updated += 1
                if updated % 100 == 0:
                    logger.info(f"Progress: {updated}/{total_missing} documents updated")

            except Exception as e:
                logger.error(f"Failed to update document {doc['_id']}: {e}")
                errors += 1

        logger.info(f"Migration complete: {updated} updated, {errors} errors")

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
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    args = parser.parse_args()

    await migrate_hashes(dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
