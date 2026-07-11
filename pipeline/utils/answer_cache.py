import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pipeline.utils.logging_config import setup_logger

logger = setup_logger()

CACHE_FILE = Path("data/cache/answer_cache.json")
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

SIMILARITY_THRESHOLD = 0.38   # Jaccard on content tokens; clinical questions rephrase often

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "could", "may", "might", "shall", "can", "need", "must",
    "i", "you", "he", "she", "it", "we", "they", "what", "which", "who",
    "whom", "this", "that", "these", "those", "am", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "up", "about", "into",
    "through", "during", "before", "after", "above", "below", "and",
    "but", "or", "nor", "so", "yet", "both", "either", "not", "no",
    "than", "then", "when", "where", "how", "why", "all", "any", "each",
    "my", "your", "his", "her", "our", "their", "its", "s", "t", "me",
    "him", "us", "them", "there", "here", "if", "as", "such", "also",
}


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalise(text: str) -> set[str]:
    """Lower-case, strip punctuation, drop stop words — keeps clinical content tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return {t for t in text.split() if t not in _STOP_WORDS and len(t) > 1}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Persistence ────────────────────────────────────────────────────────────────

def _load_cache() -> list:
    if not CACHE_FILE.exists():
        return []
    with open(CACHE_FILE) as f:
        return json.load(f)


def _save_cache(entries: list) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(entries, f, indent=2)


# ── Public API ─────────────────────────────────────────────────────────────────

def find_cached(query: str, role: str) -> Optional[dict]:
    """Return the best matching cache entry or None if no hit above threshold."""
    entries = _load_cache()
    if not entries:
        return None

    q_tokens = _normalise(query)
    best_score = 0.0
    best_entry = None

    for entry in entries:
        if entry.get("role") != role:
            continue
        # Always re-normalise from the stored query string so the
        # stop-word list and tokeniser are always applied consistently.
        stored_tokens = _normalise(entry.get("query", ""))
        score = _jaccard(q_tokens, stored_tokens)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= SIMILARITY_THRESHOLD:
        logger.info(f"Cache hit (score={best_score:.2f}) for query: {query[:60]}")
        # Increment hit counter in-place
        best_entry["hit_count"] = best_entry.get("hit_count", 0) + 1
        _save_cache(entries)
        return best_entry

    logger.info(f"Cache miss (best={best_score:.2f}) for query: {query[:60]}")
    return None


def save_to_cache(
    query: str,
    answer: str,
    role: str,
    evidence: list = None,
    verification: dict = None,
    temporal_verification: dict = None,
    saved_by: str = "clinician",
) -> dict:
    """Add or update a cache entry.  Returns the saved entry."""
    entries = _load_cache()
    tokens = list(_normalise(query))

    # Check if an entry for the same question already exists — update rather than duplicate
    q_tokens = set(tokens)
    for entry in entries:
        if entry.get("role") == role and _jaccard(q_tokens, set(entry.get("tokens", []))) >= 0.95:
            entry.update({
                "answer": answer,
                "evidence": evidence or [],
                "verification": verification,
                "temporal_verification": temporal_verification,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "saved_by": saved_by,
            })
            _save_cache(entries)
            logger.info(f"Cache updated for existing question: {query[:60]}")
            return entry

    entry = {
        "id": str(uuid.uuid4()),
        "query": query,
        "tokens": tokens,
        "answer": answer,
        "role": role,
        "evidence": evidence or [],
        "verification": verification,
        "temporal_verification": temporal_verification,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "saved_by": saved_by,
        "hit_count": 0,
    }
    entries.append(entry)
    _save_cache(entries)
    logger.info(f"Cache entry saved for query: {query[:60]}")
    return entry


def list_cache() -> list:
    """Return all cache entries (without internal token list)."""
    return [
        {k: v for k, v in e.items() if k != "tokens"}
        for e in _load_cache()
    ]


def delete_from_cache(entry_id: str) -> bool:
    """Delete a cache entry by ID. Returns True if found and deleted."""
    entries = _load_cache()
    new_entries = [e for e in entries if e.get("id") != entry_id]
    if len(new_entries) == len(entries):
        return False
    _save_cache(new_entries)
    return True
