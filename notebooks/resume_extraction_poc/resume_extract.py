"""
Step 1: Resume Extractor
------------------------
Extracts raw text from PDF/DOCX/DOC files.

Install:
    pip install pymupdf python-docx
"""

import os
from pathlib import Path
from typing import Dict

import fitz  # PyMuPDF
from docx import Document

# OCR (optional)
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# Windows .doc (optional)
try:
    import win32com.client
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


class ResumeExtractor:
    """Extract raw text from resume files."""
    
    SUPPORTED = {'.pdf', '.docx', '.doc'}
    
    def extract(self, file_path: str) -> str:
        """
        Extract text from a single resume file.
        
        Args:
            file_path: Path to PDF/DOCX/DOC file
            
        Returns:
            Extracted raw text
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in self.SUPPORTED:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.SUPPORTED}")
        
        if ext == '.pdf':
            return self._extract_pdf(file_path)
        elif ext == '.docx':
            return self._extract_docx(file_path)
        elif ext == '.doc':
            return self._extract_doc(file_path)
    
    def _extract_pdf(self, path: str) -> str:
        """Extract text from PDF, with OCR fallback."""
        text_parts = []
        
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        
        text = "\n".join(text_parts).strip()
        
        # If minimal text, try OCR (likely scanned PDF)
        if len(text) < 100:
            if HAS_OCR:
                print(f"    → Low text detected, using OCR...")
                return self._extract_pdf_ocr(path)
            else:
                print(f"    ⚠️ Low text and OCR not available")
        
        return text
    
    def _extract_pdf_ocr(self, path: str) -> str:
        """Extract text from scanned PDF using OCR."""
        text_parts = []
        
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text_parts.append(pytesseract.image_to_string(img))
        
        return "\n".join(text_parts).strip()
    
    def _extract_docx(self, path: str) -> str:
        """Extract text from DOCX."""
        doc = Document(path)
        parts = [p.text for p in doc.paragraphs]
        
        # Include table content
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        
        return "\n".join(parts).strip()
    
    def _extract_doc(self, path: str) -> str:
        """Extract text from DOC (Windows only)."""
        if not HAS_WIN32:
            raise RuntimeError(
                "DOC extraction requires pywin32 on Windows. "
                "Consider converting to DOCX."
            )
        
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        try:
            doc = word.Documents.Open(os.path.abspath(path))
            text = doc.Content.Text
            doc.Close()
            return text.strip()
        finally:
            word.Quit()
    
    def extract_directory(self, directory: str) -> Dict[str, str]:
        """
        Extract text from all resumes in a directory.
        
        Args:
            directory: Path to directory containing resume files
            
        Returns:
            Dict mapping engineer_id (filename stem) to raw text
        """
        dir_path = Path(directory)
        results = {}
        
        files = [f for f in dir_path.iterdir() if f.suffix.lower() in self.SUPPORTED]
        
        if not files:
            print(f"No supported files found in {directory}")
            return results
        
        print(f"Found {len(files)} resume(s)\n")
        
        for file_path in files:
            engineer_id = file_path.stem
            print(f"Extracting: {file_path.name}")
            
            try:
                text = self.extract(str(file_path))
                results[engineer_id] = text
                print(f"    ✓ {len(text.split())} words extracted")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        return results
