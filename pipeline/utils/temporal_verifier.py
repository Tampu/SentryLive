# TODO: temporal verifier hallucination risk — Gemini may draw on training knowledge rather than passages
# current mitigation: requires exact passage quotes for all flags — prevents unsupported claims
# future improvement: run factual verifier on temporal agent's flagged claims as a cross-verification step
# this would add a fourth Gemini call per query — acceptable in production but too slow for current PoC
# additionally consider introducing POSSIBLY_OUTDATED verdict for low-confidence flags
# ------------------------------------------------------------------------------------------------------- #



import os                                                   # used to access GOOGLE_API_KEY from .env
from google import genai                                    # Google Gen AI client — used to check if the answer uses the most recent guidelines
from google.genai import types                              # used to pass generation config parameters like temperature
from dotenv import load_dotenv                              # loads environment variables from .env into os.getenv()
from pipeline.utils.logging_config import setup_logger      # imports our custom logger for consistent logging across the pipeline



load_dotenv()                                                       # loads .env file so os.getenv() can access the API key

logger = setup_logger()                                             # initialize the logger for the temporal verifier
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))          # initialize the Gemini client using the API key from .env




# checks whether the generated answer uses the most recent available guideline version
# runs after the factual verifier — focused solely on temporal accuracy, not factual accuracy
# returns a verdict (CURRENT / OUTDATED) and flags any outdated references
# TODO: in a future full-scale product, regenerate the answer if verdict is OUTDATED
def verify_temporal(query: str, answer: str, metadata: dict, guideline_passages: str) -> dict:

    logger.info("Temporal verifier — checking if answer uses most recent guidelines")


    # extract all available years from the metadata stored in doc_ids.json
    # filters out None values for documents where year extraction failed during upload
    years = [meta["year"] for meta in metadata.values() if meta.get("year")]


    # if no year metadata is available, skip temporal verification gracefully
    # this happens if doc_ids.json is missing year fields — don't crash, just flag as unknown
    if not years:
        logger.warning("Temporal verifier — no year metadata available, skipping")
        return {
            "verdict": "UNKNOWN",
            "verification_text": "Temporal verification could not be performed — no year metadata available.",
            "is_current": None
        }


    newest_year = max(years)        # the most recent guideline year — what the answer should prioritize





    # build the temporal verification prompt
    # passes both the answer and the actual guideline passages so Gemini can compare
    # year-stamped filenames in the passages help Gemini identify which version each claim comes from
    prompt = f"""You are a temporal accuracy checker for a clinical oncology assistant.

Your job is to verify whether the generated answer correctly uses the most recent available guidelines.

Available guideline years: {sorted(years)}
Most recent guideline year: {newest_year}

Query: {query}

Generated Answer:
{answer}

Guideline Passages (from multiple versions — filenames contain the year of each guideline):
{guideline_passages}

Using the guideline passages above as your reference, check whether the answer:
1. Correctly prioritizes recommendations from the most recent guidelines ({newest_year})
2. References any outdated recommendations that have been superseded by newer guidelines visible in the passages
3. Misses any important updates from the {newest_year} guidelines that are relevant to the query and present in the passages

Important: Only flag something as outdated if a newer passage explicitly supersedes it. Do not flag older guideline references that are still valid and not contradicted by newer passages.
Critical: Base your verdict ONLY on what is explicitly written in the guideline passages provided above. Do not use any external medical knowledge or training data to make judgments. If a recommendation is not addressed in the provided passages, do not flag it — only flag what you can directly point to in the text of the passages above.

Return your response in this exact format:
TEMPORAL_VERDICT: [CURRENT / OUTDATED]
CONFIDENCE: [HIGH / MEDIUM / LOW]
FLAGGED: [for each flagged claim, you MUST quote the exact passage that supports the flag in this format: "Claim: [what the answer said] | Superseded by: [exact quote from a newer passage]". If you cannot provide an exact quote from the passages, do not flag it.]
NOTES: [any additional context about the temporal accuracy of the answer]"""





    # read temperature from .env — defaults to 0.1 if not set
    # lower temperature = more deterministic and consistent clinical responses
    temperature = float(os.getenv("TEMPERATURE", "0.1"))


    # third independent Gemini call — separate from both answer generation and factual verification
    # focused solely on temporal accuracy — does not re-check factual claims
    response = client.models.generate_content(
        model=os.getenv("ANSWER_GENERATION_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature                 # low temperature for consistent temporal verification verdicts
        )
    )


    verification_text = response.text.strip()       # raw temporal verification response from Gemini


    # parse the verdict from the structured response
    # default to UNKNOWN if parsing fails
    verdict = "UNKNOWN"
    if "TEMPORAL_VERDICT: CURRENT" in verification_text:
        verdict = "CURRENT"


    logger.info(f"Temporal verifier — verdict: {verdict}")


    return {
        "verdict": verdict,                         # CURRENT or OUTDATED
        "verification_text": verification_text,     # full report with flagged claims and notes
        "is_current": verdict == "CURRENT",         # boolean for quick checks in the orchestrator
    }