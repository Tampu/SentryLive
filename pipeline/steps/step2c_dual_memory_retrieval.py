from pipeline.utils.logging_config import setup_logger      # imports our custom logger for consistent logging across the pipeline


logger = setup_logger()     # initialize the logger for this step



# returns empty memory for both short-term and long-term — placeholder until Redis is implemented
def retrieve_memory(session_id: str) -> dict:

    logger.info("Step 2C — memory stub returning empty context")

    return {
        "short_term": [],       # short-term memory — last 5 dialogue turns (empty for now)
        "long_term": [],        # long-term memory — past clinician corrections and patient context (empty for now)
    }


