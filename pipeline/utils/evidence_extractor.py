import re
from typing import List, Dict


# ── Recommendation scorer ────────────────────────────────────────────────────

def _rec_score(text: str) -> int:
    """Score a passage on how likely it is to be a clinical recommendation."""
    lower = text.lower()
    score = 0
    if re.search(r'\brecommendation\s+\d', lower):               score += 7
    if re.search(r'\bshould\b', lower):                          score += 3
    if 'evidence-based' in lower or 'evidence quality' in lower: score += 4
    if 'strength of recommendation' in lower:                    score += 4
    if re.search(r'\bclinician\b', lower):                       score += 2
    if re.search(r'\b(offer|advise|consider|recommend)\b', lower): score += 2
    # Penalise statistics / results paragraphs
    if 'hazard ratio' in lower or '95% ci' in lower:            score -= 3
    if re.search(r'\b\d+\.\d+%', lower):                        score -= 2
    if re.search(r'p\s*[=<>]\s*[01]?\.\d', lower):             score -= 2
    if re.search(r'\bhr\s*[=<>]', lower):                       score -= 2
    return score


# ── Evidence extractor ───────────────────────────────────────────────────────

def extract_evidence(passages: str, metadata: dict | None = None) -> List[Dict]:
    """
    Parse <doc=filename;page=N> citation tags from PageIndex passages.

    Multiple passages may map to the same (filename, page) — all are kept and
    sorted by recommendation score so the PDF viewer can highlight every
    paragraph on that page, ranked best-first.

    `metadata` is the doc_ids.json dict (filename stem -> {doc_id, year, is_update}) —
    passed through so each evidence item can flag itself as an "updated guideline"
    in the UI, using the same drift signal detected at upload time.
    """
    metadata = metadata or {}
    citation_pattern = re.compile(r'<doc=([^;>]+);page=(\d+)(?:;[^>]*)?>')
    matches = list(citation_pattern.finditer(passages))
    if not matches:
        return []

    # ── Pass 1: collect all "before" texts per (filename, page) ─────────────
    # Each citation tag is preceded by the passage Gemini intends to associate
    # with it.  We use only `before` (not before+after) so paragraphs stay
    # separate and don't bleed into each other.
    candidates: dict = {}   # (filename, page) -> list[str]
    seen_order: list = []

    for i, match in enumerate(matches):
        # Gemini doesn't always omit the extension despite the retrieval prompt asking for
        # "the document's name or ID (without .pdf)" — strip it here so pdf_url below never
        # ends up double-suffixed (...kinase.pdf.pdf), and so the same document always keys
        # to the same evidenceMap entry regardless of which form Gemini happened to emit.
        filename = re.sub(r'\.pdf$', '', match.group(1), flags=re.IGNORECASE)
        page     = int(match.group(2))
        key      = (filename, page)

        prev_end = matches[i - 1].end() if i > 0 else 0
        before   = passages[prev_end:match.start()].strip()
        before   = citation_pattern.sub("", before).strip()

        if before:
            if key not in candidates:
                candidates[key] = []
                seen_order.append(key)
            candidates[key].append(before)

    # ── Pass 2: build one evidence item per page, carrying all texts ─────────
    evidence_items: List[Dict] = []

    for (filename, page) in seen_order:
        texts = candidates[(filename, page)]

        # Sort all passages best-first by recommendation score
        ranked = sorted(texts, key=_rec_score, reverse=True)

        # Cap each text at 600 chars for display; keep up to 6 passages
        all_texts = [t[:597] + "…" if len(t) > 600 else t for t in ranked[:6]]
        display_text = all_texts[0]  # best passage shown in the evidence card

        # Build citation key and metadata
        parts      = filename.split("-")
        author     = parts[0].capitalize() if parts else "Unknown"
        year_match = re.search(r"\b(19|20)\d{2}\b", filename)
        year       = year_match.group() if year_match else "n.d."
        citation_key = f"[{author} et al. {year}, p.{page}]"

        year_idx = next(
            (j for j, p in enumerate(parts) if re.match(r"^(19|20)\d{2}$", p)),
            len(parts),
        )
        topic_parts = parts[year_idx + 1:]
        topic = " ".join(p.capitalize() for p in topic_parts) if topic_parts else "Clinical Guidelines"

        evidence_items.append({
            "filename":     filename,
            "page":         page,
            "text":         display_text,
            "all_texts":    all_texts,   # ranked list for multi-highlight in PDF viewer
            "citation_key": citation_key,
            "author":       f"{author} et al. {year}",
            "topic":        topic,
            "pdf_url":      f"/api/guidelines/{filename}.pdf",
            "is_update":    bool(metadata.get(filename, {}).get("is_update")),
        })

    return evidence_items
