import os                                               # used to access PAGEINDEX_API_KEY and GOOGLE_API_KEY from .env
from pageindex import PageIndexClient                   # PageIndex client — used to upload guidelines and retrieve relevant passages
from google import genai                                # Google Gen AI client — used to make the agentic RAG call to Gemini 2.5 Flash
from dotenv import load_dotenv                          # loads environment variables from .env into os.getenv()
from pipeline.utils.logging_config import setup_logger  # imports our custom logger for consistent logging across the pipeline



load_dotenv()                                                       # loads .env file so os.getenv() can access the API keys

logger = setup_logger()                                             # initialize the logger for this step
pi = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY"))        # initialize the PageIndex client using the API key from .env
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))          # initialize the Gemini client using the API key from .env




# uploads a guideline PDF to PageIndex and returns the document ID
# this only needs to be run once per PDF — the document ID is reused for all future queries
def upload_guideline(pdf_path: str) -> str:

    logger.info(f"Step 2A — uploading guideline PDF: {pdf_path}")

    doc = pi.submit_document(pdf_path)      # uploads the PDF to PageIndex and starts building the tree index
    doc_id = doc["doc_id"]                  # extract the document ID from the response

    logger.info(f"Step 2A — guideline uploaded successfully, doc_id: {doc_id}")
    return doc_id                           # return the doc_id so it can be stored and reused for future queries





# takes a clinical query and a list of document IDs and returns relevant guideline passages
# uses PageIndex's built-in Chat API to reason through the guideline tree and retrieve relevant sections
def retrieve_guideline_passages(query: str, doc_ids: list) -> str:

    logger.info(f"Step 2A — retrieving passages for query: {query}")


    # call PageIndex's chat API directly — no need for Gemini Interactions API
    # passes all doc IDs so PageIndex searches across all uploaded guidelines
    response = pi.chat_completions(
        messages=[{"role": "user", "content": query}],
        doc_id=doc_ids,             # restricts retrieval to the uploaded guideline documents
        enable_citations=True       # includes page and section references in the response
    )


    # extract the response text from the API response
    passages = response["choices"][0]["message"]["content"]

    logger.info(f"Step 2A — retrieval complete")
    return passages                 # passes the retrieved passages to Step 3 for context merging