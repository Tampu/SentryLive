import os                                                   # used to access GOOGLE_API_KEY from .env
import re                                                    # used to find citation tags in the generated answer
from google import genai                                    # Google Gen AI client — used to verify the generated answer against guideline passages
from google.genai import types                              # used to pass generation config parameters like temperature
from dotenv import load_dotenv                              # loads environment variables from .env into os.getenv()
from pipeline.utils.logging_config import setup_logger      # imports our custom logger for consistent logging across the pipeline
from pipeline.utils.pdf_pages import get_page_text           # fetches the real on-disk PDF text for a cited page




load_dotenv()                                                       # loads .env file so os.getenv() can access the API key

logger = setup_logger()                                             # initialize the logger for the verifier
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))          # initialize the Gemini client using the API key from .env





# matches raw <doc=filename;page=N> citation tags in the answer, BEFORE format_citations()
# has collapsed them into [Author et al. Year, p.N] — verify_answer() must run first in the
# orchestrator for this to see the real filename/page, not just an author surname.
_CITATION_TAG = re.compile(r'<doc=([^;>]+);page=(\d+)(?:;[^>]*)?>')


# builds the "ACTUAL PDF PAGE TEXT" section of the verification prompt: for every distinct
# (filename, page) cited in the answer, fetch what is really printed on that page — independent
# ground truth the verifier can check the citation against, rather than only the passages blob
# Gemini itself produced in Step 2A (which can carry the same page-number error forward).
def _build_page_evidence(answer: str) -> str:
    seen = set()
    blocks = []

    for match in _CITATION_TAG.finditer(answer):
        filename = re.sub(r'\.pdf$', '', match.group(1), flags=re.IGNORECASE)
        page = int(match.group(2))
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)

        page_text = get_page_text(filename, page)
        if page_text is None:
            blocks.append(f'<doc={filename};page={page}>: COULD NOT BE READ (file missing or page out of range — treat any claim citing this as unverifiable)')
        else:
            blocks.append(f'<doc={filename};page={page}> — actual text on this page:\n{page_text}')

    if not blocks:
        return "(No citation tags found in the answer to cross-check.)"
    return "\n\n".join(blocks)




# independently verifies the generated answer against the retrieved guideline passages
# uses a second Gemini call to check every factual claim in the answer
# returns an overall verdict, verified claims, and flagged claims
# TODO: in a future full-scale product, regenerate the answer if verdict is UNVERIFIED instead of flagging
# TODO: replace flagged claims with citation-based verification once citations are implemented
def verify_answer(query: str, answer: str, guideline_passages: str) -> dict:

    logger.info("Verifier — checking answer against guideline passages")

    page_evidence = _build_page_evidence(answer)


    # build the verification prompt
    # asks Gemini to return an overall verdict plus verified and flagged claims separately
    # this gives a clear picture of what is grounded in the guidelines and what is not
    prompt = f"""You are a clinical fact-checker verifying an answer against ASCO guideline passages.

Query: {query}

Generated Answer:
{answer}

Guideline Passages (retrieved for this query — may itself contain page-number errors):
{guideline_passages}

Actual PDF Page Text (ground truth — extracted directly from the real guideline PDFs on the
exact pages the answer cites; independent of what Step 2A retrieval returned):
{page_evidence}

For each factual claim in the generated answer, check whether it is explicitly supported by the guideline passages above.
Important: Ignore any citations in the generated answer when verifying claims — check each claim solely against the guideline passages provided, regardless of what citation the answer attributes to it. A citation does not verify a claim; only the actual guideline passage text does.

Separately, using the "Actual PDF Page Text" section: for each citation tag in the answer, check whether the
claim attributed to it is actually supported by the text that is really on that specific page — not merely
supported somewhere else in the guideline passages. A claim can be true and well-supported in general while
still citing the wrong page number; flag that case too, since it sends readers to the wrong page.

Return your response in this exact format:
VERDICT: [VERIFIED / UNVERIFIED / PARTIALLY_VERIFIED]
CONFIDENCE: [HIGH / MEDIUM / LOW]
FLAGGED_CLAIMS: [list each claim that is not supported by the passages, or "None"]
PAGE_ACCURACY_ISSUES: [list each citation whose specific page does not actually contain the claimed content, e.g. "Yu et al. 2025, p.6 — this content is not on page 6", or "None"]
NOTES: [any additional context about the verification]"""




    # read temperature from .env — defaults to 0.1 if not set
    # lower temperature = more deterministic and consistent clinical responses
    temperature = float(os.getenv("TEMPERATURE", "0.1"))


    # second independent Gemini call — separate from answer generation so verification is unbiased
    # the verifier has no knowledge of how the answer was generated — it only sees the query, answer, and passages
    response = client.models.generate_content(
        model=os.getenv("ANSWER_GENERATION_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature                 # low temperature for consistent temporal verification verdicts
        )
    )

    verification_text = response.text.strip()       # raw verification response from Gemini


    # parse the overall verdict from the structured response
    # default to UNVERIFIED if parsing fails — safer to flag than to silently pass in a clinical system
    verdict = "UNVERIFIED"
    if "VERDICT: VERIFIED" in verification_text:
        verdict = "VERIFIED"
    elif "VERDICT: PARTIALLY_VERIFIED" in verification_text:
        verdict = "PARTIALLY_VERIFIED"

    # a wrong page number is a real defect (breaks paragraph highlighting, sends the reader to the
    # wrong page) even when the underlying claim is otherwise fully content-verified — so it can't
    # leave the overall verdict at VERIFIED. Match the same "None" convention FLAGGED_CLAIMS uses.
    page_issues_match = re.search(r'PAGE_ACCURACY_ISSUES:\s*(.*?)(?:\nNOTES:|$)', verification_text, re.DOTALL)
    page_issues_text = page_issues_match.group(1).strip() if page_issues_match else ""
    has_page_issues = bool(page_issues_text) and page_issues_text.lower() not in ("none", "[none]", "none.")

    if has_page_issues and verdict == "VERIFIED":
        verdict = "PARTIALLY_VERIFIED"

    logger.info(f"Verifier — verdict: {verdict}" + (" (downgraded for page-accuracy issues)" if has_page_issues else ""))



    return {
        "verdict": verdict,                         # overall verdict — VERIFIED, PARTIALLY_VERIFIED, or UNVERIFIED
        "verification_text": verification_text,     # full report including verified claims, flagged claims, confidence, and notes
        "is_verified": verdict == "VERIFIED",       # boolean for quick checks in the orchestrator
        "has_page_accuracy_issues": has_page_issues, # True if any citation's page number doesn't actually contain the claimed content
    }