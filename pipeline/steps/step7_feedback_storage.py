import json                                                     # used to read and write feedback as JSON files
from pathlib import Path                                        # used to build the path to the feedback folder
from pipeline.utils.logging_config import setup_logger          # imports our custom logger for consistent logging across the pipeline
from datetime import datetime, timezone                         # used to generate timezone-aware UTC timestamps for each piece of feedback




logger = setup_logger()                                         # initialize the logger for this step

FEEDBACK_DIR = Path("data/feedback")                            # folder where clinician feedback is stored as JSON files
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)                 # create the folder if it doesn't exist yet




# stores a clinician correction in data/feedback/ as a JSON file
# indexed by timestamp so each correction is uniquely stored and retrievable
def store_feedback(query: str, original_answer: str, correction: str, clinician_id: str = "default") -> dict:

    logger.info(f"Step 7 — storing clinician feedback for query: {query}")

    # build the feedback object with all relevant information
    # each field is stored so future queries can retrieve and apply the correction in context
    feedback = {
        "timestamp": datetime.now(timezone.utc).isoformat(),    # when the correction was made — used for sorting and versioning
        "clinician_id": clinician_id,                           # who made the correction — enables personalization per clinician in the future
        "query": query,                                         # the original clinical question that triggered the incorrect answer
        "original_answer": original_answer,                     # what the pipeline generated — stored so we can see what was wrong
        "correction": correction,                               # what the clinician said was wrong or missing — the ground truth
    }

    # create a unique filename using the timestamp — colons replaced with dashes for filesystem compatibility
    filename = FEEDBACK_DIR / f"feedback_{feedback['timestamp'].replace(':', '-')}.json"

    # write the feedback to disk as a formatted JSON file
    with open(filename, "w") as f:
        json.dump(feedback, f, indent=4)                # indent=4 makes the file human-readable when opened directly

    logger.info(f"Step 7 — feedback stored at {filename}")
    return feedback                                     # return the feedback object so the caller can confirm it was saved





# loads all stored clinician corrections from data/feedback/
# called at the start of each pipeline run so relevant past corrections can be included in the context
def load_feedback() -> list:

    logger.info("Step 7 — loading stored clinician feedback")

    # start with an empty list — will be populated with all stored corrections
    feedback_list = []

    # loop through every JSON file in the feedback folder
    # each file is one clinician correction stored by store_feedback()
    for feedback_file in FEEDBACK_DIR.glob("*.json"):
        with open(feedback_file, "r") as f:
            feedback = json.load(f)             # parse the JSON file back into a Python dictionary
            feedback_list.append(feedback)      # add it to the list

    logger.info(f"Step 7 — loaded {len(feedback_list)} clinician corrections")
    return feedback_list        # returns the full list of corrections for the orchestrator to filter and apply



