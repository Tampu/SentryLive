import pdfplumber                                           # used to extract text from the first 3 pages of guideline PDFs
from pipeline.utils.logging_config import setup_logger      # imports our custom logger for consistent logging across the pipeline



logger = setup_logger()                                             # initialize the logger for the drift detector



# scans the first 3 pages of a PDF to detect if it is an update to a previous guideline
# returns True if update language is found, False if it appears to be an original guideline
def is_update_document(pdf_path: str) -> bool:

    logger.info(f"Drift detector — checking if document is an update: {pdf_path}")

    try:
        with pdfplumber.open(pdf_path) as f:

            # scan only the first 3 pages — ASCO guidelines always declare update status early
            for page in f.pages[:3]:
                text = page.extract_text()

                if text:
                    text_lower = text.lower()   # lowercase for case-insensitive matching

                    # check for common ASCO update phrases
                    # any one of these phrases appearing confirms the document is an update
                    if any(phrase in text_lower for phrase in [
                        "guideline update",             # e.g. "ASCO Guideline Update"
                        "rapid recommendation update",  # e.g. "ASCO Rapid Recommendation Update"
                        "recommendation update",        # catches other variations
                        "focused update",               # e.g. "Clinical Practice Guideline Focused Update"
                        "to update the"                 # e.g. abstract saying "To update the ASCO guideline"
                    ]):
                        

                        logger.info(f"Drift detector — update document detected: {pdf_path}")
                        return True


    except Exception as e:
        # if the PDF can't be read for any reason, log a warning and assume it's not an update
        # safer to miss an update than to crash the upload process
        logger.warning(f"Drift detector — could not read PDF {pdf_path}: {e}")


    return False     # no update language found — treat as original guideline
