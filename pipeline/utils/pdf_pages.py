from pathlib import Path                                    # used to search data/guidelines for the real PDF file
import pdfplumber                                           # used to extract a specific page's real text (already a dependency — see drift_detector.py)
from pipeline.utils.logging_config import setup_logger      # imports our custom logger for consistent logging across the pipeline


logger = setup_logger()     # initialize the logger for this module

GUIDELINES_DIR = Path("data/guidelines")     # same relative path convention as upload_guidelines.py


# fetches the real, on-disk text of one page of an uploaded guideline PDF.
# used by the verifier to check whether a cited page actually contains what
# the answer attributes to it — PageIndex/Gemini occasionally return the right
# content on the wrong page number (see citation_formatter.py's TODO).
# returns None if the file can't be found or the page number is out of range,
# so callers can treat "couldn't check" distinctly from "checked and empty".
def get_page_text(filename: str, page: int) -> str | None:

    matches = list(GUIDELINES_DIR.rglob(f"{filename}.pdf"))
    if not matches:
        logger.warning(f"Page verifier — PDF not found for citation check: {filename}")
        return None

    try:
        with pdfplumber.open(matches[0]) as pdf:
            if page < 1 or page > len(pdf.pages):
                logger.warning(f"Page verifier — page {page} out of range for {filename} ({len(pdf.pages)} pages)")
                return None
            return pdf.pages[page - 1].extract_text() or ""

    except Exception as e:
        # a corrupt/unreadable PDF shouldn't crash verification — just skip the check for it
        logger.warning(f"Page verifier — could not read {filename} page {page}: {e}")
        return None
