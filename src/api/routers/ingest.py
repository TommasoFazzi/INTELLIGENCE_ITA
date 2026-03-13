"""PDF ingestion API router."""
import logging
import json
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional

from ..auth import verify_api_key
from ...ingestion.pdf_ingestor import PDFIngestor
from ...nlp.processing import NLPProcessor
from ...nlp.bullet_generator import BulletGenerator
from ...storage.database import DatabaseManager
from ...utils.logger import get_logger

logger = get_logger(__name__)
api_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


class IngestResponse(BaseModel):
    """Response from PDF ingestion endpoint."""
    success: bool
    article_id: Optional[int] = None
    title: str
    chunks: int = 0
    bullet_points: List[str] = []
    message: str


def get_db() -> DatabaseManager:
    """Get database connection."""
    return DatabaseManager()


@router.post("/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
) -> IngestResponse:
    """
    Upload and ingest a PDF document.

    Automatically extracts text, processes through NLP pipeline, generates bullet points,
    and saves to database in a single transaction.

    - **file**: PDF file (multipart/form-data, max 50MB)
    - **Returns**: Article ID, chunk count, and AI-generated bullet points

    Rate limit: 5 requests per hour per API key
    """
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    file_content = await file.read()
    if len(file_content) > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    try:
        # Initialize ingestor
        ingestor = PDFIngestor()

        # Extract text from PDF
        text = ingestor.extract_text(file_content)
        if not text or len(text.strip()) < 50:
            raise HTTPException(status_code=400, detail="PDF contains insufficient text")

        # Build article dict
        article = ingestor.build_article_dict(
            text=text,
            url=f"upload://{file.filename}",
            title=file.filename.replace('.pdf', ''),
            publisher="Manual Upload",
            category="intelligence",
            subcategory="documents"
        )

        # Process through NLP pipeline
        logger.info(f"Processing uploaded PDF: {article['title']}")

        try:
            processor = NLPProcessor()
            processed_article = processor.process_article(article)

            if not processed_article.get('nlp_processing', {}).get('success'):
                raise HTTPException(
                    status_code=422,
                    detail="NLP processing failed"
                )

            # Extract chunks for response
            chunks_count = len(processed_article.get('nlp_data', {}).get('chunks', []))

            # Generate bullet points
            bullet_points = []
            try:
                bullet_gen = BulletGenerator()
                bullets = bullet_gen.generate_bullets(processed_article)
                if bullets:
                    processed_article['nlp_data']['bullet_points'] = bullets
                    bullet_points = bullets
                    logger.info(f"Generated {len(bullets)} bullet points")
            except Exception as e:
                logger.warning(f"Bullet generation failed: {e}, proceeding without bullets")

            # Save to database
            db = get_db()
            try:
                with db.get_connection() as conn:
                    with conn.cursor() as cur:
                        # Use the existing save_article logic
                        article_id = db.save_article(processed_article)

                        if article_id and bullet_points:
                            # Save bullet points to ai_analysis
                            try:
                                db.update_article_analysis(
                                    article_id,
                                    {'bullet_points': bullet_points}
                                )
                                logger.info(f"Saved bullet points for article {article_id}")
                            except Exception as e:
                                logger.warning(f"Failed to save bullet points: {e}")

                        return IngestResponse(
                            success=True,
                            article_id=article_id,
                            title=article['title'],
                            chunks=chunks_count,
                            bullet_points=bullet_points,
                            message=f"Successfully ingested PDF with {chunks_count} chunks"
                        )

            except Exception as e:
                api_logger.error(f"Database error: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save article to database: {str(e)}"
                )

        except HTTPException:
            raise
        except Exception as e:
            api_logger.error(f"NLP processing error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"NLP processing failed: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"PDF ingestion error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )
