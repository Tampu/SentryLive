import os
import asyncio
from pageindex import PageIndexClient
from google import genai
from dotenv import load_dotenv
from pipeline.utils.logging_config import setup_logger

load_dotenv()

logger = setup_logger()

pi = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY"))
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# MCP retrieval requires a Gemini 2.5 model — Gemini 3.x does not support remote MCP
RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "gemini-2.5-flash")

_MCP_TOOL = {
    "type": "mcp_server",
    "name": "pageindex",
    "url": "https://api.pageindex.ai/mcp",
    "headers": {"Authorization": f"Bearer {os.getenv('PAGEINDEX_API_KEY')}"},
}

_SYSTEM_INSTRUCTION = """You are a clinical guideline retrieval assistant.
Use the available PageIndex MCP tools to search and retrieve relevant passages from the indexed ASCO clinical guidelines.

For every passage you retrieve, append a citation tag immediately after it in this exact format:
<doc=DOCUMENT_FILENAME;page=PAGE_NUMBER>

Where DOCUMENT_FILENAME is the EXACT document name or ID returned by the PageIndex tool call —
copy it verbatim, character for character. Never shorten, paraphrase, or guess a filename, and
never emit a citation tag for a document you did not actually retrieve via the tool. If the tool
returns no relevant passages, return nothing rather than inventing a plausible-looking one.

Return only the retrieved passages with their citation tags. Do not summarise, answer, or add commentary.

Example output format (note the filename is the full, exact identifier from the tool result,
not a shortened or invented version of it):
The recommended first-line treatment for HER2-positive metastatic breast cancer is trastuzumab plus pertuzumab plus a taxane. <doc=giordano-et-al-2022-systemic-therapy-for-advanced-human-epidermal-growth-factor-receptor-2-positive-breast-cancer-asco;page=12>
Patients who progress on CDK4/6 inhibitors may be considered for everolimus-based therapy. <doc=burstein-et-al-2021-endocrine-treatment-and-targeted-therapy-for-hormone-receptor-positive-human-epidermal-growth;page=8>
"""


async def _retrieve_async(query: str) -> str:
    collected = []

    stream = await client.aio.interactions.create(
        model=RETRIEVAL_MODEL,
        stream=True,
        system_instruction=_SYSTEM_INSTRUCTION,
        input=query,
        tools=[_MCP_TOOL],
    )

    async for event in stream:
        if event.event_type == "step.delta" and event.delta:
            if event.delta.type == "text":
                collected.append(event.delta.text or "")

    return "".join(collected)


def retrieve_guideline_passages(query: str, doc_ids: list) -> str:
    """Retrieve relevant guideline passages via Gemini + PageIndex MCP.

    doc_ids is accepted for API compatibility with the original step but is not
    passed to MCP — Gemini discovers and selects the relevant documents itself.
    """
    logger.info(f"Step 2A (MCP) — retrieving passages for query: {query}")

    try:
        # asyncio.run() works correctly inside a threadpool worker (no running loop there)
        passages = asyncio.run(_retrieve_async(query))
    except RuntimeError:
        # Fallback for environments that already have a running loop (e.g. Jupyter)
        loop = asyncio.new_event_loop()
        try:
            passages = loop.run_until_complete(_retrieve_async(query))
        finally:
            loop.close()

    logger.info("Step 2A (MCP) — retrieval complete")
    return passages


def upload_guideline(pdf_path: str) -> str:
    """Upload a guideline PDF to PageIndex (unchanged from original)."""
    logger.info(f"Step 2A — uploading guideline PDF: {pdf_path}")
    doc = pi.submit_document(pdf_path)
    doc_id = doc["doc_id"]
    logger.info(f"Step 2A — uploaded successfully, doc_id: {doc_id}")
    return doc_id
