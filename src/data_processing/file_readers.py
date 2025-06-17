import polars as pl
import pandas as pd
from docx import Document
from pypdf import PdfReader
import io
from PIL import Image
import pytesseract
import easyocr
import os

class FileReaders:
    def __init__(self):
        self.easy_ocr_reader = None
        tesseract_cmd = os.getenv("TESSERACT_CMD_PATH")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def read_csv(self, file_content: io.BytesIO):
        """Read CSV file with fallback encodings."""
        encodings = ["utf-8", "latin-1", "iso-8859-1"]
        for encoding in encodings:
            try:
                file_content.seek(0)
                return pl.read_csv(file_content, encoding=encoding)
            except Exception as e:
                if encoding == encodings[-1]:  # Last encoding tried
                    raise Exception(f"Failed to read CSV with tried encodings: {str(e)}")
                continue

    def read_xlsx(self, file_content: io.BytesIO, sheet_name=0):
        """Read Excel file."""
        try:
            file_content.seek(0)
            return pl.read_excel(file_content, sheet_name=sheet_name)
        except Exception as e:
            raise Exception(f"Failed to read XLSX: {str(e)}")

    def read_text(self, file_content: io.BytesIO):
        """Read text file."""
        try:
            file_content.seek(0)
            return file_content.read().decode("utf-8")
        except Exception as e:
            raise Exception(f"Failed to read text file: {str(e)}")

    def read_docx(self, file_content: io.BytesIO):
        """Read DOCX file."""
        try:
            file_content.seek(0)
            doc = Document(file_content)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Failed to read DOCX: {str(e)}")

    def read_pdf(self, file_content: io.BytesIO):
        """Read PDF file."""
        try:
            file_content.seek(0)
            pdf = PdfReader(file_content)
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")

    def read_image_text(self, file_content: io.BytesIO, ocr_engine="easyocr"):
        """Read text from image using specified OCR engine."""
        try:
            file_content.seek(0)
            image = Image.open(file_content).convert("RGB")
            if ocr_engine == "easyocr":
                if self.easy_ocr_reader is None:
                    self.easy_ocr_reader = easyocr.Reader(["en"])
                result = self.easy_ocr_reader.readtext(image)
                return "\n".join([text[1] for text in result])
            else:  # Default to pytesseract
                return pytesseract.image_to_string(image)
        except ImportError:
            print("pytesseract or Pillow not installed. Please install with: pip install pytesseract Pillow")
            return None
        except Exception as e:
            raise Exception(f"Failed to read image text: {str(e)}")