"""
Document Processing Pipeline
=============================
Handles PDF manuals containing text, tables, diagrams, and images.
Extracts all content, performs OCR on images/diagrams, and produces
clean text chunks ready for embedding.

Pipeline:
  PDF → PyMuPDF (text + images) → pdfplumber (tables) → OCR (images) → chunks
"""

import os
import io
import re
import hashlib
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single chunk of extracted content from a manual."""
    text: str
    source_file: str
    page_number: int
    chunk_type: str  # "text", "table", "image_ocr", "diagram_ocr"
    equipment_id: str
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            content_hash = hashlib.md5(
                f"{self.source_file}:{self.page_number}:{self.text[:100]}".encode()
            ).hexdigest()[:12]
            self.chunk_id = f"{self.equipment_id}_{content_hash}"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# OCR helper
# ---------------------------------------------------------------------------

def ocr_image(image: Image.Image) -> str:
    """Extract text from an image using Tesseract OCR."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image, config="--psm 6")
        return text.strip()
    except ImportError:
        logger.warning("pytesseract not installed — skipping OCR")
        return ""
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# PDF Text Extraction (PyMuPDF)
# ---------------------------------------------------------------------------

def extract_text_pymupdf(pdf_path: str) -> list[dict]:
    """
    Extract text page-by-page using PyMuPDF.
    Returns list of {page: int, text: str}.
    """
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append({"page": page_num + 1, "text": text.strip()})
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF text extraction failed for {pdf_path}: {e}")
    return pages


# ---------------------------------------------------------------------------
# PDF Image Extraction + OCR (PyMuPDF)
# ---------------------------------------------------------------------------

def extract_images_pymupdf(pdf_path: str, min_size: int = 100) -> list[dict]:
    """
    Extract images from PDF using PyMuPDF, then OCR them.
    Returns list of {page: int, text: str, image_index: int}.
    Filters out tiny images (icons, bullets) with min_size.
    """
    results = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    # Skip tiny images
                    if image.width < min_size or image.height < min_size:
                        continue
                    # OCR the image
                    ocr_text = ocr_image(image)
                    if ocr_text and len(ocr_text) > 10:
                        results.append({
                            "page": page_num + 1,
                            "text": ocr_text,
                            "image_index": img_idx,
                        })
                except Exception as e:
                    logger.warning(f"Image extraction failed page {page_num+1} img {img_idx}: {e}")
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF image extraction failed for {pdf_path}: {e}")
    return results


# ---------------------------------------------------------------------------
# Table Extraction (pdfplumber)
# ---------------------------------------------------------------------------

def extract_tables_pdfplumber(pdf_path: str) -> list[dict]:
    """
    Extract tables from PDF using pdfplumber.
    Returns list of {page: int, text: str, table_index: int}.
    Tables are converted to readable markdown-style text.
    """
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for tbl_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    # Convert table to markdown
                    md_lines = []
                    for row_idx, row in enumerate(table):
                        clean_row = [str(cell).strip() if cell else "" for cell in row]
                        md_lines.append("| " + " | ".join(clean_row) + " |")
                        if row_idx == 0:
                            md_lines.append("|" + "|".join(["---"] * len(clean_row)) + "|")
                    table_text = "\n".join(md_lines)
                    if table_text.strip():
                        results.append({
                            "page": page_num + 1,
                            "text": table_text,
                            "table_index": tbl_idx,
                        })
    except Exception as e:
        logger.error(f"pdfplumber table extraction failed for {pdf_path}: {e}")
    return results


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks.
    Uses sentence-aware splitting to avoid breaking mid-sentence.
    """
    if not text or not text.strip():
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        # Fallback: simple chunking
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if end < len(text):
                # Try to break at sentence boundary
                last_period = chunk.rfind(". ")
                if last_period > chunk_size // 2:
                    end = start + last_period + 2
                    chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - chunk_overlap
        return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Main Processing Pipeline
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: str,
    equipment_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    progress_callback=None,
) -> list[DocumentChunk]:
    """
    Full processing pipeline for a single PDF manual.

    Steps:
      1. Extract text (PyMuPDF)
      2. Extract tables (pdfplumber)
      3. Extract images → OCR
      4. Chunk all content
      5. Return DocumentChunk list

    Args:
        pdf_path: Path to the PDF file
        equipment_id: Unique identifier for the equipment this manual belongs to
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        progress_callback: Optional callable(stage: str, progress: float)

    Returns:
        List of DocumentChunk objects ready for embedding
    """
    filename = os.path.basename(pdf_path)
    all_chunks: list[DocumentChunk] = []

    def _progress(stage, pct):
        if progress_callback:
            progress_callback(stage, pct)

    # --- Stage 1: Text extraction ---
    _progress("Extracting text...", 0.0)
    text_pages = extract_text_pymupdf(pdf_path)
    logger.info(f"Extracted text from {len(text_pages)} pages")

    for page_data in text_pages:
        page_chunks = chunk_text(page_data["text"], chunk_size, chunk_overlap)
        for chunk_text_str in page_chunks:
            all_chunks.append(DocumentChunk(
                text=chunk_text_str,
                source_file=filename,
                page_number=page_data["page"],
                chunk_type="text",
                equipment_id=equipment_id,
            ))

    # --- Stage 2: Table extraction ---
    _progress("Extracting tables...", 0.33)
    tables = extract_tables_pdfplumber(pdf_path)
    logger.info(f"Extracted {len(tables)} tables")

    for table_data in tables:
        # Tables are kept as single chunks (usually small enough)
        table_chunks = chunk_text(table_data["text"], chunk_size, chunk_overlap)
        for chunk_text_str in table_chunks:
            all_chunks.append(DocumentChunk(
                text=chunk_text_str,
                source_file=filename,
                page_number=table_data["page"],
                chunk_type="table",
                equipment_id=equipment_id,
                metadata={"table_index": table_data["table_index"]},
            ))

    # --- Stage 3: Image OCR ---
    _progress("Processing images (OCR)...", 0.66)
    images = extract_images_pymupdf(pdf_path)
    logger.info(f"OCR'd {len(images)} images")

    for img_data in images:
        img_chunks = chunk_text(img_data["text"], chunk_size, chunk_overlap)
        for chunk_text_str in img_chunks:
            all_chunks.append(DocumentChunk(
                text=chunk_text_str,
                source_file=filename,
                page_number=img_data["page"],
                chunk_type="image_ocr",
                equipment_id=equipment_id,
                metadata={"image_index": img_data["image_index"]},
            ))

    _progress("Done", 1.0)
    logger.info(f"Total chunks from {filename}: {len(all_chunks)}")
    return all_chunks


def process_directory(
    dir_path: str,
    equipment_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    progress_callback=None,
) -> list[DocumentChunk]:
    """Process all PDFs in a directory for a given equipment."""
    all_chunks = []
    pdf_files = sorted(Path(dir_path).glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {dir_path}")
        return all_chunks

    for i, pdf_file in enumerate(pdf_files):
        def _progress(stage, pct):
            if progress_callback:
                overall = (i + pct) / len(pdf_files)
                progress_callback(f"{pdf_file.name}: {stage}", overall)

        chunks = process_pdf(
            str(pdf_file),
            equipment_id,
            chunk_size,
            chunk_overlap,
            progress_callback=_progress,
        )
        all_chunks.extend(chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Processing stats
# ---------------------------------------------------------------------------

def get_processing_stats(chunks: list[DocumentChunk]) -> dict:
    """Return summary statistics for processed chunks."""
    if not chunks:
        return {"total_chunks": 0}

    type_counts = {}
    page_set = set()
    file_set = set()
    total_chars = 0

    for chunk in chunks:
        type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1
        page_set.add((chunk.source_file, chunk.page_number))
        file_set.add(chunk.source_file)
        total_chars += len(chunk.text)

    return {
        "total_chunks": len(chunks),
        "total_characters": total_chars,
        "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
        "files_processed": len(file_set),
        "pages_covered": len(page_set),
        "chunks_by_type": type_counts,
        "files": sorted(file_set),
    }
