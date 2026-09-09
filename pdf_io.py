"""pdf_io.py - Paired PDF handle: pypdfium2 for text, pdfplumber for tables.

Text extraction uses PDFium (correct multi-column reading order, ~1 ms/page);
PDFium exposes no table API, so table structure comes from pdfplumber over the
same source. Both ship s390x-compatible distributions.
"""
from __future__ import annotations

import io
from typing import Union

import pdfplumber
import pypdfium2 as pdfium


class PdfBundle:
    """Two open handles to one PDF, closed together.

    Accepts a filesystem path or raw PDF bytes. Re-openable: construct a new
    instance per ``with`` block rather than reusing a closed one.
    """

    def __init__(self, source: Union[str, bytes]) -> None:
        self._source = source
        self.pdfium = None
        self.plumber = None

    def __enter__(self) -> "PdfBundle":
        src = self._source
        self.pdfium = pdfium.PdfDocument(src)
        # pdfplumber needs a file-like object; pdfium takes bytes directly.
        self.plumber = pdfplumber.open(io.BytesIO(src) if isinstance(src, bytes) else src)
        return self

    def __exit__(self, *exc_info) -> bool:
        try:
            if self.plumber is not None:
                self.plumber.close()
        finally:
            if self.pdfium is not None:
                self.pdfium.close()
        return False

    def __len__(self) -> int:
        return len(self.pdfium)

    def page_text(self, index: int) -> str:
        """Extract page text via PDFium, releasing the text page immediately."""
        textpage = self.pdfium[index].get_textpage()
        try:
            return (textpage.get_text_bounded() or "").replace("\r\n", "\n")
        finally:
            textpage.close()

    def page_tables(self, index: int) -> list:
        """Extract structured tables via pdfplumber, then release its page cache."""
        if index >= len(self.plumber.pages):
            return []
        page = self.plumber.pages[index]
        try:
            return [t for t in page.extract_tables() if t]
        finally:
            page.flush_cache()