"""
PDF document ingestion module for institutional documents (SIPRI, CRS, ISS, NATO reports, etc.).

Extracts text from PDF files and converts to article dict structure compatible with
the existing NLP pipeline. Reuses process_nlp.py and load_to_database.py without changes.
"""

import os
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlparse

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PDFIngestor:
    """Ingests PDF documents and converts them to article dicts compatible with the pipeline."""

    def __init__(self, timeout: int = 30):
        """
        Initialize the PDF ingestor.

        Args:
            timeout: Download timeout in seconds
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("pymupdf is required. pip install pymupdf==1.25.3")

        self.timeout = timeout
        logger.info("PDFIngestor initialized with PyMuPDF")

    async def download_pdf(self, url: str) -> Optional[bytes]:
        """
        Download PDF from URL asynchronously.

        Args:
            url: URL to PDF file

        Returns:
            PDF content as bytes, or None if download fails
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                    if resp.status == 200 and 'pdf' in resp.content_type.lower():
                        return await resp.read()
                    else:
                        logger.warning(f"Failed to download PDF from {url}: HTTP {resp.status}")
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"Download timeout for PDF from {url}")
            return None
        except Exception as e:
            logger.warning(f"Failed to download PDF from {url}: {e}")
            return None

    def extract_text(self, pdf_bytes: bytes, max_pages: Optional[int] = None) -> Optional[str]:
        """
        Extract text from PDF bytes using PyMuPDF.

        Args:
            pdf_bytes: PDF content as bytes
            max_pages: Maximum pages to extract (None = all)

        Returns:
            Extracted text, or None if extraction fails
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []

            page_count = len(doc)
            if max_pages:
                page_count = min(page_count, max_pages)

            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text()
                if text:
                    text_parts.append(text)

            doc.close()

            full_text = "\n\n".join(text_parts).strip()

            if not full_text:
                logger.warning("PDF extraction returned empty text")
                return None

            return full_text

        except Exception as e:
            logger.warning(f"Failed to extract text from PDF: {e}")
            return None

    def _extract_first_page_text(self, pdf_bytes: bytes) -> str:
        """Extract text from first page only for summary."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(doc) > 0:
                text = doc[0].get_text()
                doc.close()
                return text[:500] if text else ""
            doc.close()
            return ""
        except Exception:
            return ""

    def build_article_dict(
        self,
        text: str,
        url: str,
        title: str,
        publisher: str,
        category: str,
        subcategory: str,
        published_date: Optional[datetime] = None
    ) -> Dict:
        """
        Build article dictionary from extracted PDF text.

        Args:
            text: Extracted PDF text
            url: PDF URL (used as unique identifier)
            title: Document title
            publisher: Publisher name (e.g., "SIPRI", "IISS")
            category: Article category
            subcategory: Article subcategory
            published_date: Publication date (defaults to today)

        Returns:
            Article dict compatible with NLP pipeline
        """
        if published_date is None:
            published_date = datetime.utcnow()

        return {
            'title': title,
            'link': url,
            'published': published_date.isoformat() if isinstance(published_date, datetime) else published_date,
            'summary': text[:300],  # First 300 chars
            'source': f"pdf:{publisher}",
            'category': category,
            'subcategory': subcategory,
            'full_content': {
                'text': text
            },
            'extraction_success': True,
            'extraction_method': 'pymupdf'
        }

    async def run_url(
        self,
        url: str,
        metadata: Dict
    ) -> Optional[Dict]:
        """
        Download and process a single PDF from URL.

        Args:
            url: PDF URL
            metadata: Dict with keys: title, publisher, category, subcategory, published (optional)

        Returns:
            Article dict, or None if processing fails
        """
        logger.info(f"Processing PDF from {url}")

        # Download
        pdf_bytes = await self.download_pdf(url)
        if not pdf_bytes:
            return None

        # Extract text
        text = self.extract_text(pdf_bytes)
        if not text or len(text.strip()) < 50:
            logger.warning(f"Insufficient text extracted from {url}")
            return None

        # Build article dict
        article = self.build_article_dict(
            text=text,
            url=url,
            title=metadata.get('title', 'Untitled PDF'),
            publisher=metadata.get('publisher', 'Unknown'),
            category=metadata.get('category', 'intelligence'),
            subcategory=metadata.get('subcategory', 'documents'),
            published_date=metadata.get('published')
        )

        logger.info(f"Successfully processed PDF: {article['title'][:50]}...")
        return article

    async def run_batch(self, pdf_configs: List[Dict]) -> List[Dict]:
        """
        Process multiple PDFs concurrently.

        Args:
            pdf_configs: List of dicts with keys: url, title, publisher, category, subcategory, published (optional)

        Returns:
            List of article dicts (failed PDFs skipped)
        """
        logger.info(f"Processing {len(pdf_configs)} PDFs concurrently...")

        tasks = [
            self.run_url(config['url'], config)
            for config in pdf_configs
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Filter out None results from failed downloads
        articles = [r for r in results if r is not None]

        logger.info(f"Successfully processed {len(articles)}/{len(pdf_configs)} PDFs")
        return articles

    def run_all(self, config_path: str = "config/pdf_sources.yaml") -> List[Dict]:
        """
        Load config and process all PDF sources.

        Args:
            config_path: Path to pdf_sources.yaml configuration

        Returns:
            List of article dicts
        """
        import yaml

        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}")
            return []

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not config or 'pdf_sources' not in config:
            logger.warning(f"No pdf_sources found in {config_path}")
            return []

        pdf_configs = config['pdf_sources']
        logger.info(f"Loaded {len(pdf_configs)} PDF sources from config")

        # Run async batch processing
        articles = asyncio.run(self.run_batch(pdf_configs))

        return articles

    def run_single_file(self, file_path: str, metadata: Dict) -> Optional[Dict]:
        """
        Process a single local PDF file.

        Args:
            file_path: Path to local PDF file
            metadata: Dict with keys: title, publisher, category, subcategory, published (optional)

        Returns:
            Article dict, or None if processing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        logger.info(f"Processing local PDF: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
        except Exception as e:
            logger.warning(f"Failed to read PDF file {file_path}: {e}")
            return None

        # Extract text
        text = self.extract_text(pdf_bytes)
        if not text or len(text.strip()) < 50:
            logger.warning(f"Insufficient text extracted from {file_path}")
            return None

        # Use file path as URL
        url = f"file://{file_path.absolute()}"

        # Build article dict
        article = self.build_article_dict(
            text=text,
            url=url,
            title=metadata.get('title', file_path.stem),
            publisher=metadata.get('publisher', 'Local Document'),
            category=metadata.get('category', 'intelligence'),
            subcategory=metadata.get('subcategory', 'documents'),
            published_date=metadata.get('published')
        )

        logger.info(f"Successfully processed local PDF: {article['title'][:50]}...")
        return article
