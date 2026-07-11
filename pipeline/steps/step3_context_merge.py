from typing import Optional                                 # used to mark fields that may not always be present in the merged context
from pipeline.utils.logging_config import setup_logger      # imports our custom logger for consistent logging across the pipeline



logger = setup_logger()     # initialize the logger for this step


# takes the outputs of Steps 2A and 2C and combines them into one unified context object
# priority order: long-term memory corrections > retrieved guideline passages
def merge_context(
    retrieved_passages: Optional[str] = None,       # output from Step 2A — guideline text retrieved via PageIndex
    memory: Optional[dict] = None,                  # output from Step 2C — short and long term memory
) -> dict:

    logger.info("Step 3 — merging retrieval context")

    # default to empty structures if any stream has no data — prevents crashes when stubs return nothing
    retrieved_passages = retrieved_passages or ""
    memory = memory or {"short_term": [], "long_term": []}

    # combine both streams into one unified context object
    # priority ordering — clinician corrections first, then retrieved guideline passages
    merged = {
        "long_term_corrections": memory.get("long_term", []),   # highest priority — verified clinician corrections
        "retrieved_passages": retrieved_passages,               # second priority  — guideline text retrieved via PageIndex
        "short_term_history": memory.get("short_term", []),     # conversation history for multi-turn coherence
    }

    logger.info(f"Step 3 — context merge complete: {len(retrieved_passages)} passages retrieved")
    return merged