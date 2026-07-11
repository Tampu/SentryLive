import json                                                                       # used to read doc_ids.json to get the document IDs for retrieval
from pathlib import Path                                                          # used to build the path to doc_ids.json
from pipeline.steps.step1_concept_extraction import extract_concepts              # Step 1  — extracts structured entities from the query
from pipeline.steps.step2a_guideline_retrieval_mcp import retrieve_guideline_passages # Step 2A — retrieves passages via Gemini + PageIndex MCP (no per-query PageIndex credits)
from pipeline.steps.step2c_dual_memory_retrieval import retrieve_memory           # Step 2C — retrieves short and long term memory
from pipeline.steps.step3_context_merge import merge_context                      # Step 3  — merges all retrieval streams into one unified context
from pipeline.steps.step5_role_router import route_prompt                         # Step 5  — selects the appropriate prompt template based on user role
from pipeline.steps.step6_answer_generation import generate_answer                # Step 6  — generates the final answer with confidence scoring
from pipeline.steps.step7_feedback_storage import store_feedback, load_feedback   # Step 7  — stores and retrieves clinician corrections
from pipeline.utils.verifier import verify_answer                                 # verifier — independently checks the generated answer against guideline passages
from pipeline.utils.logging_config import setup_logger                            # imports our custom logger for consistent logging across the pipeline
from pipeline.utils.citation_formatter import format_citations                    # formats raw PageIndex citation tags into readable citations
from pipeline.utils.conversation_summarizer import summarize_conversation         # summarizes conversation history every 3 turns to prevent context window overflow
from pipeline.utils.temporal_verifier import verify_temporal                      # temporal verifier — checks if the answer uses the most recent available guideline version
from pipeline.utils.evidence_extractor import extract_evidence                    # extracts structured evidence items from raw PageIndex passages for the web UI
from pipeline.utils.answer_cache import find_cached                               # semantic cache — returns a previous clinician-approved answer on similar questions




logger = setup_logger()     # initialize the logger for the orchestrator




# reads doc_ids.json and returns a flat list of all PageIndex document IDs
# called once at the start of each pipeline run so Step 2A knows which guidelines to search
def load_doc_ids() -> list:

    doc_ids_path = Path("data/guidelines/doc_ids.json")     # path to the doc IDs file


    # if the file doesn't exist, log a warning and return empty list rather than crashing
    # this happens if the upload script hasn't been run yet
    if not doc_ids_path.exists():
        logger.warning("doc_ids.json not found — no guidelines loaded, retrieval will be skipped")
        return [], {}

    with open(doc_ids_path, "r") as f:
        data = json.load(f)


    # extract structured metadata — each entry has a doc_id and year
    guidelines = data["guidelines"]

    # extract just the doc IDs as a flat list — this is what PageIndex needs for retrieval
    doc_ids = [v["doc_id"] for v in guidelines.values()]

    # keep the full metadata dictionary including year — this is what drift detection will use-
    # to determine which guideline is newer when comparing recommendations
    metadata = {k: v for k, v in guidelines.items()}

    logger.info(f"Loaded {len(doc_ids)} guideline document IDs")
    return doc_ids, metadata        # return both — doc_ids for retrieval, metadata for drift detection






# main pipeline function — runs the full SentryLive pipeline from query to answer
# takes a clinical query and user role and returns a grounded answer with confidence score
def run_pipeline(query: str, role: str = "patient", conversation_history: list = None, force_refresh: bool = False) -> dict:


    # use an empty list if no history is passed — avoids Python's mutable default argument gotcha-
    # where a shared list persists across calls if defined as = [] in the function signature
    conversation_history = conversation_history or []

    logger.info(f"Pipeline starting — query: {query} | role: {role} | history: {len(conversation_history)} turns")


    # Cache check — skip on force_refresh or when conversation context is active
    # (multi-turn follow-ups rely on history and shouldn't short-circuit to a cached standalone answer)
    if not force_refresh and not conversation_history:
        cached = find_cached(query, role)
        if cached:
            logger.info("Returning cached answer — skipping full pipeline")
            return {
                "answer":               cached["answer"],
                "evidence":             cached.get("evidence", []),
                "verification":         cached.get("verification"),
                "temporal_verification": cached.get("temporal_verification"),
                "cache_hit":            True,
                "cached_query":         cached["query"],
            }

    


    # summarize conversation history every 3 turns to prevent context window overflow
    # without summarization, the full history grows unboundedly and will eventually exceed Gemini's token limit
    # we trigger at every multiple of 3 (turn 3, 6, 9...) to keep the context window consistently small
    if len(conversation_history) > 0 and len(conversation_history) % 3 == 0:

        logger.info(f"Conversation summarizer — triggering summarization at {len(conversation_history)} turns")

        # call the summarizer to compress the full history into a compact paragraph
        summary = summarize_conversation(conversation_history, role=role)

        # preserve the last turn separately before replacing the history
        # this gives Gemini immediate conversational context for follow-up questions
        # e.g. "what about the side effects of that drug?" needs to know which drug was just discussed
        last_turn = conversation_history[-1]

        # replace the full history with just two entries:
        # 1. the compressed summary of everything discussed so far
        # 2. the most recent exchange for immediate context
        conversation_history = [
            {"query": "Previous conversation summary", "answer": summary},  # compressed history — replaces all prior turns
            last_turn                                                       # most recent exchange — preserved for follow-up handling
        ]

        logger.info("Conversation summarizer — history compressed to summary + last turn")




    # Step 7 — load any stored clinician corrections at the start of each run
    # these get passed into the context so Gemini can apply relevant past corrections
    feedback = load_feedback() if role == "clinician" else []    # only load feedback for clinicians — patients don't need it



    # Step 1 — extract structured clinical entities from the query
    concepts = extract_concepts(query)
    logger.info(f"Step 1 complete — concepts: {concepts}")




    # Step 2A — load guideline doc IDs and retrieve relevant passages via PageIndex
    # enrich the query with recent conversation context so PageIndex retrieves more relevantly on follow-ups
    doc_ids, metadata = load_doc_ids()
    if conversation_history:
        last_turn = conversation_history[-1]            # get the most recent exchange from the conversation history

        # combine the last question, a short summary of the last answer, and the current question
        # this gives PageIndex enough context to retrieve relevant guideline passages even for follow-up questions
        # [:200] limits the previous answer to 200 characters to keep the enriched query concise
        enriched_query = f"Previous question: {last_turn['query']}\nPrevious answer summary: {last_turn['answer'][:200]}\nCurrent question: {query}"
    else:
        enriched_query = query                          # first turn — use the raw query as is, no history to enrich with

    passages = retrieve_guideline_passages(enriched_query, doc_ids) if doc_ids else ""
    retrieval_score = 1.0 if passages else 0.0          # 1.0 if passages were retrieved, 0.0 if not
    logger.info(f"Step 2A complete — passages retrieved: {bool(passages)}")




    # Step 2C — retrieve short and long term memory (returns empty lists until Redis is implemented)
    memory = retrieve_memory(session_id="default")      # session ID hardcoded for now — will be unique per user in full implementation
    logger.info("Step 2C complete — memory retrieved")


    # Step 3 — merge all retrieval streams into one unified context object
    merged = merge_context(
        retrieved_passages=passages,
        memory=memory
    )
    logger.info("Step 3 complete — context merged")


    # convert the merged context dictionary into a plain string for the prompt template
    # formatted clearly so Gemini can identify guideline passages and clinician corrections separately
    context_str = ""
    if merged["retrieved_passages"]:
        context_str += f"Relevant guideline passages:\n{merged['retrieved_passages']}\n"




    # for clinicians only — instruct Gemini to prefer newer guidelines over older ones
    # dynamically sorts all loaded guidelines by year so this works regardless of how many documents are uploaded
    if role == "clinician" and metadata:

        # sort all guidelines by year ascending — oldest first, newest last
        sorted_by_year = sorted(metadata.items(), key=lambda x: x[1].get("year") or 0)

        # extract just the years as a list — filter out None values for documents without a year
        years = [meta["year"] for _, meta in sorted_by_year if meta.get("year")]

        if years:
            # tell Gemini the year range of available guidelines and which is the most recent
            # detailed drift detection instructions are in the prompt template in step5_role_router.py
            context_str += f"\nGuideline version note: Guidelines are available from {min(years)} to {max(years)}. Always prioritize information from the most recent guidelines ({max(years)}).\n"

        # ground-truth drift signal — upload_guidelines.py already scanned each PDF for ASCO
        # "update"/"focused update" language (pipeline/utils/drift_detector.py) and stored the
        # result in doc_ids.json. Passing it here gives Gemini a confirmed signal for which
        # documents are updates, instead of inferring purely from passage wording.
        update_docs = [name for name, meta in metadata.items() if meta.get("is_update")]
        if update_docs:
            context_str += f"\nDocuments confirmed as guideline updates (contain explicit 'update'/'focused update' language): {', '.join(update_docs)}.\n"




    # TODO: currently loads ALL stored corrections regardless of relevance to the current query
    # this is acceptable for the PoC where feedback volume is low, but creates two problems at scale:
    # 1. token limit — too many corrections will exceed Gemini's context window and cause failures
    # 2. noise — irrelevant corrections may confuse Gemini and degrade answer quality
    # future improvement — use embedding similarity to only retrieve the top 3-5 corrections
    # whose query is semantically closest to the current query, filtering out irrelevant ones
    if feedback:
        context_str += "\nPast clinician corrections:\n"
        for item in feedback:
            
            # include the original query and correction so Gemini understands the context of each correction
            context_str += f"- Query: {item['query']}\n"
            context_str += f"  Correction: {item['correction']}\n"



    # add any long term memory corrections from Redis (empty until Redis is implemented)
    if merged["long_term_corrections"]:
        context_str += "\nClinician corrections:\n"
        for correction in merged["long_term_corrections"]:
            context_str += f"- {correction}\n"


    # Step 5 — select the appropriate prompt template based on the user's declared role
    prompt = route_prompt(role=role, context=context_str, question=query)
    logger.info("Step 5 complete — prompt routed")


    # Step 6 — send the routed prompt to Gemini and get back an answer with confidence score
    result = generate_answer(prompt=prompt, retrieval_score=retrieval_score, conversation_history=conversation_history)
    logger.info("Step 6 complete — answer generated")




    # Verifier — independently verify the generated answer against the guideline passages
    # runs for both clinician and patient roles
    # TODO: in a future improvement, adapt the verification report format for patients-
    # so it is explained in simpler, more accessible language rather than the technical format used for clinicians
    if passages:
        verification = verify_answer(
            query=query,
            answer=result["answer"],
            guideline_passages=passages         # raw passages from Step 2A — the ground truth for verification
        )
        result["verification"] = verification   # attach verification report to the result
        logger.info(f"Verifier complete — verdict: {verification['verdict']}")
    else:
        result["verification"] = None           # no verification if no passages were retrieved




    # Temporal Verifier — checks if the answer uses the most recent available guideline version
    # runs for both clinician and patient roles — separate from factual verification
    # focused solely on temporal accuracy — does not re-check factual claims
    # runs after the factual verifier so both verdicts are available before citation formatting
    if passages and metadata:
        temporal_verification = verify_temporal(
            query=query,
            answer=result["answer"],            # the generated answer to check for temporal accuracy
            metadata=metadata,                  # year metadata from doc_ids.json — tells verifier which year is newest
            guideline_passages=passages         # raw passages from Step 2A — ground truth for temporal comparison
        )
        result["temporal_verification"] = temporal_verification     # attach temporal report to result
        logger.info(f"Temporal verifier complete — verdict: {temporal_verification['verdict']}")
    else:
        # skip temporal verification if passages or metadata are missing
        # passages missing means retrieval failed — nothing to compare against
        # metadata missing means doc_ids.json wasn't found — no year information available
        result["temporal_verification"] = None


    
    
    # extract structured evidence items from raw passages before citation tags are reformatted
    # metadata is passed through so each evidence item can carry its own is_update flag for the UI
    result["evidence"] = extract_evidence(passages, metadata) if passages else []

    # format raw PageIndex citation tags into clean readable citations e.g. [Burstein et al. 2024, p.2]
    # known_filenames = metadata.keys() lets the formatter flag any citation whose (author, year)
    # doesn't match a document we actually uploaded — retrieval (Step 2A) is not scoped to this
    # corpus at the tool-call level, so Gemini can otherwise surface or invent an out-of-corpus source
    result["answer"] = format_citations(result["answer"], known_filenames=metadata.keys())



    logger.info("Pipeline complete")
    return result