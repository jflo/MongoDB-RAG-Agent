"""Komga client for PDF deep linking."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import httpx

from src.settings import Settings

logger = logging.getLogger(__name__)


class KomgaClient:
    """Client for Komga API with local caching of book ID lookups."""

    def __init__(self, settings: Settings):
        """
        Initialize Komga client.

        Args:
            settings: Application settings with Komga configuration
        """
        self.base_url = settings.komga_base_url
        self.username = settings.komga_username
        self.password = settings.komga_password
        self.cache_file = Path(settings.komga_cache_file)
        self._cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached filename -> bookId mappings from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self._cache = json.load(f)
                logger.info(f"komga_cache_loaded: entries={len(self._cache)}")
            except Exception as e:
                logger.warning(f"komga_cache_load_failed: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save filename -> bookId mappings to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"komga_cache_saved: entries={len(self._cache)}")
        except Exception as e:
            logger.warning(f"komga_cache_save_failed: {e}")

    def is_configured(self) -> bool:
        """Check if Komga is properly configured."""
        return bool(self.base_url and self.username and self.password)

    async def test_connection(self) -> tuple[bool, str]:
        """
        Test connectivity to Komga server.

        Returns:
            Tuple of (success, message) where message contains server info or error
        """
        if not self.is_configured():
            return False, "Komga not configured (missing base_url, username, or password)"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/libraries",
                    auth=(self.username, self.password),
                    timeout=5.0
                )
                response.raise_for_status()

                libraries = response.json()
                lib_count = len(libraries)
                return True, f"Connected to {self.base_url} ({lib_count} libraries)"

        except httpx.ConnectError:
            return False, f"Cannot connect to Komga at {self.base_url}"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return False, "Komga authentication failed (invalid credentials)"
            return False, f"Komga HTTP error: {e.response.status_code}"
        except httpx.TimeoutException:
            return False, f"Komga connection timed out ({self.base_url})"
        except Exception as e:
            return False, f"Komga error: {e}"

    async def get_book_id(self, filename: str) -> Optional[str]:
        """
        Get Komga book ID for a filename.

        Checks cache first, then queries Komga API if not found.

        Args:
            filename: Source filename (e.g., "GRR6610_TheExpanse_TUE_Core.pdf")

        Returns:
            Komga book ID if found, None otherwise
        """
        if not self.is_configured():
            return None

        # Check cache first
        if filename in self._cache:
            return self._cache[filename]

        # Query Komga API
        try:
            async with httpx.AsyncClient() as client:
                # Search for books matching the filename
                response = await client.get(
                    f"{self.base_url}/api/v1/books",
                    params={"search": filename},
                    auth=(self.username, self.password),
                    timeout=10.0
                )
                response.raise_for_status()

                data = response.json()
                books = data.get("content", [])

                # Find exact or close match
                for book in books:
                    book_name = book.get("name", "")
                    book_url = book.get("url", "")

                    # Match by name or URL containing the filename
                    if filename in book_name or filename in book_url:
                        book_id = book.get("id")
                        if book_id:
                            # Cache the result
                            self._cache[filename] = book_id
                            self._save_cache()
                            logger.info(f"komga_book_found: filename={filename}, book_id={book_id}")
                            return book_id

                logger.debug(f"komga_book_not_found: filename={filename}")
                return None

        except httpx.HTTPError as e:
            logger.warning(f"komga_api_error: filename={filename}, error={e}")
            return None
        except Exception as e:
            logger.exception(f"komga_lookup_failed: filename={filename}, error={e}")
            return None

    def get_page_url(self, book_id: str, page_number: int) -> str:
        """
        Construct deep link URL to a specific page in Komga reader.

        Args:
            book_id: Komga book ID
            page_number: Page number (1-indexed)

        Returns:
            Full URL to the page in Komga web reader
        """
        # Komga uses 1-indexed page query parameter
        return f"{self.base_url}/book/{book_id}/read?page={page_number}"

    async def get_source_url(self, filename: str, page_number: Optional[int] = None) -> Optional[str]:
        """
        Get a deep link URL for a source document and optional page.

        Args:
            filename: Source filename
            page_number: Optional page number (1-indexed)

        Returns:
            Komga URL if book found, None otherwise
        """
        book_id = await self.get_book_id(filename)
        if not book_id:
            return None

        if page_number:
            return self.get_page_url(book_id, page_number)
        else:
            # Return URL to book viewer at page 1
            return f"{self.base_url}/book/{book_id}/read?page=1"

    def get_source_url_sync(self, filename: str, page_number: Optional[int] = None) -> Optional[str]:
        """
        Get a deep link URL using only the local cache (no API calls).

        This is a synchronous version that only checks the cache.
        Useful for post-processing citations without async context.

        Args:
            filename: Source filename (with or without .pdf extension)
            page_number: Optional page number (1-indexed)

        Returns:
            Komga URL if book found in cache, None otherwise
        """
        if not self.is_configured():
            return None

        # Try exact match first
        book_id = self._cache.get(filename)

        # Try with .pdf extension if not found
        if not book_id and not filename.endswith('.pdf'):
            book_id = self._cache.get(filename + '.pdf')

        if not book_id:
            return None

        if page_number:
            return self.get_page_url(book_id, page_number)
        else:
            return f"{self.base_url}/book/{book_id}/read?page=1"


# Singleton instance for reuse
_komga_client: Optional[KomgaClient] = None


def get_komga_client(settings: Settings) -> KomgaClient:
    """
    Get or create a Komga client instance.

    Args:
        settings: Application settings

    Returns:
        KomgaClient instance
    """
    global _komga_client
    if _komga_client is None:
        _komga_client = KomgaClient(settings)
    return _komga_client
