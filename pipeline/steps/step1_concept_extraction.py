import os                                               # used to access GOOGLE_API_KEY from the .env file at runtime
from google import genai                                # Google Gen AI client — used to make API calls to Gemini 2.5 Flash
from pydantic import BaseModel, field_validator          # field_validator normalizes Gemini's output before type-checking, since free-text JSON isn't schema-enforced on Gemini's side
from typing import Optional                             # used to mark fields that may not always be present in the extracted output
from pipeline.utils.logging_config import setup_logger  # imports our custom logger for consistent logging across the pipeline
from dotenv import load_dotenv                          # loads environment variables from .env into os.getenv()
import json                                             # used to parse Gemini's plain text JSON response into a Python object



load_dotenv()                                               # loads .env file so os.getenv() can access the API key

logger = setup_logger()                                     # initialize the logger for this step
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  # initialize the Gemini client using the API key from .env



# defines the exact structure Gemini must return after extracting concepts from a clinical query
# -> can be thought of as "whenever Gemini returns concept extraction results, they must look exactly like this"
# each field maps to a clinical entity — any field not mentioned in the query defaults to None instead of causing an error
# this structured output is passed directly to the orchestrator for downstream retrieval
class ConceptExtractionResult(BaseModel):
    treatment: Optional[str] = None                     # e.g. "pembrolizumab"
    cancer_type: Optional[str] = None                   # e.g. "metastatic castration-resistant prostate cancer"
    biomarker: Optional[str] = None                     # e.g. "PD-L1"
    treatment_line: Optional[str] = None                # e.g. "first-line"
    population: Optional[str] = None                    # e.g. "adult males with mCRPC"

    # Gemini's output here isn't schema-enforced (see extract_concepts) — it's free-text JSON,
    # so any field can come back as a list instead of a string whenever the query mentions more
    # than one value for that slot (e.g. multiple drugs, as happened with `treatment` in production).
    # Coercing before validation, for every field, means this can't resurface on a field we haven't
    # personally seen break yet.
    @field_validator("*", mode="before")
    @classmethod
    def _coerce_to_scalar_string(cls, value):
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            items = [str(v).strip() for v in value if v is not None and str(v).strip()]
            return ", ".join(items) if items else None
        return str(value)




# takes a raw clinical question as a string and returns a structured ConceptExtractionResult object
def extract_concepts(query: str) -> ConceptExtractionResult:

    # logs the incoming query so we can see it in the terminal when the pipeline runs
    logger.info(f"Step 1 — extracting concepts from query: {query}")


    # tells Gemini exactly what its job is and what format to return
    # Gemini doesn't support structured outputs so we instruct it via the prompt
    system_prompt = """You are a clinical NLP assistant. Extract the key clinical entities 
    from the given oncology query and return them as a JSON object with these exact fields:
    treatment, cancer_type, biomarker, treatment_line, population.
    If a field is not mentioned in the query, set it to null."""


    # sends the system prompt + clinical query to Gemini and gets a plain text response back
    response = client.models.generate_content(
        model=os.getenv("CONCEPT_EXTRACTION_MODEL", "gemini-2.5-flash"),  # uses model from .env, defaults to gemini-2.5-flash if not set
        contents=f"{system_prompt}\n\nQuery: {query}",                    # combines system prompt and query into one string
    )


    # Gemini sometimes wraps its JSON response in markdown code fences (```json ... ```)
    # we strip those out before parsing so json.loads() doesn't fail
    response_text = response.text.strip().replace("```json", "").replace("```", "")


    # parses the cleaned JSON string into a ConceptExtractionResult object
    # if Gemini's response doesn't match our expected fields, Pydantic will raise an error here
    result = ConceptExtractionResult(**json.loads(response_text))

    logger.info(f"Step 1 — extraction complete: {result}")  # logs the extracted concepts so we can verify them in the terminal
    return result                                           # passes the structured concepts to the orchestrator for downstream retrieval


