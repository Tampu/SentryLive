# TODO: citation page accuracy has not been formally evaluated
# during the evaluation stage, verify that cited page numbers correspond to the actual
# content referenced in the answer — PageIndex may occasionally return slightly off page numbers
# this should be part of the 100 question eval set that Tampu and Gaurav are leading
# ---------------------------------------------------------------------------------------------- #


import re                                                       # used to find and replace citation tags in the answer text
from pipeline.utils.logging_config import setup_logger          # imports our custom logger for consistent logging across the pipeline



logger = setup_logger()     # initialize the logger for the citation formatter




# builds a set of (author, year) pairs from known guideline filenames (doc_ids.json keys),
# using the exact same derivation logic as the citation formatters below — so a citation is
# considered "grounded" if and only if it maps to a document we actually uploaded.
def _known_author_years(known_filenames) -> set:
    pairs = set()
    for filename in known_filenames or []:
        parts = filename.split('-')
        author = parts[0].lower()
        year_match = re.search(r'\b(19|20)\d{2}\b', filename)
        if year_match:
            pairs.add((author, year_match.group()))
    return pairs




# takes the raw answer text containing <doc=filename;page=N> citation tags
# and replaces them with clean, readable citations like [Burstein et al. 2024, p.2]
#
# known_filenames (optional) — the set of filenames actually uploaded to this corpus
# (doc_ids.json keys). Step 2A's retrieval is not scoped to this set at the tool-call
# level (see step2a_guideline_retrieval_mcp.py), so Gemini can still surface or invent a
# citation to a document outside it. When known_filenames is provided, any citation whose
# (author, year) doesn't match a real uploaded document is flagged inline as unverifiable
# instead of being formatted identically to a genuinely grounded one.
def format_citations(text: str, known_filenames=None) -> str:

    logger.info("Citation formatter — formatting citations in answer")

    known_pairs = _known_author_years(known_filenames)

    def _finalize(author: str, year: str, page: str) -> str:
        if known_pairs and (author.lower(), year) not in known_pairs:
            logger.warning(f"Citation formatter — unverifiable citation (not in known corpus): {author} et al. {year}, p.{page}")
            return f"[⚠ {author} et al. {year}, p.{page} — unverifiable source]"
        return f"[{author} et al. {year}, p.{page}]"


    # regex pattern to find all citation tags in the format <doc=filename;page=N>
    # captures two groups: the filename and the page number
    # Also matches extended tags like <doc=file;page=N;block=...> and <doc=file;page=N;quote="...">
    citation_pattern = re.compile(r'<doc=([^;>]+);page=(\d+)(?:;[^>]*)?>')




    def replace_citation(match):
        filename = match.group(1)   # the full filename e.g. "burstein-et-al-2024-endocrine-and-targeted..."
        page = match.group(2)       # the page number e.g. "2"

        # extract the author — take the first word of the filename and capitalize it
        # e.g. "burstein-et-al-2024-..." -> "Burstein"
        parts = filename.split('-')
        author = parts[0].capitalize()

        # extract the year using the same regex we use in upload_guidelines.py
        # looks for a 4-digit number starting with 19 or 20
        # falls back to "n.d." (no date) if no year is found in the filename
        year_match = re.search(r'\b(19|20)\d{2}\b', filename)
        year = year_match.group() if year_match else "n.d."

        # build the clean readable citation from the extracted parts
        return _finalize(author, year, page)




    # also handle citations that Gemini generates directly from filenames in the context
    # Gemini sometimes creates its own citation format: [filename.pdf;page=N] instead of <doc=filename;page=N>
    # this pattern catches both variations — with and without the .pdf extension
    gemini_citation_pattern = re.compile(r'\[([^\]]+\.pdf);page=(\d+)\]|\[([^\]]+);page=(\d+)\]')

    def replace_gemini_citation(match):
        # extract filename and page — two capture groups because the pattern has two alternatives
        filename = match.group(1) or match.group(3)     # group 1 matches .pdf version, group 3 matches without .pdf
        page = match.group(2) or match.group(4)         # group 2 and 4 are the corresponding page numbers

        # clean up the filename and extract author and year — same logic as the PageIndex formatter above
        parts = filename.replace('.pdf', '').split('-')  # remove .pdf extension before splitting
        author = parts[0].capitalize()                   # first word capitalized e.g. "Burstein"

        # extract year using the same regex as upload_guidelines.py
        year_match = re.search(r'\b(19|20)\d{2}\b', filename)
        year = year_match.group() if year_match else "n.d."

        # return the clean readable citation
        return _finalize(author, year, page)




    # handle citations Gemini generates in the format [filename, p.N] or [filename, p. N]
    # this is another variation Gemini uses when referencing filenames directly from the context
    # e.g. [burstein-et-al-2024, p.2] -> [Burstein et al. 2024, p.2]
    short_citation_pattern = re.compile(r'\[([a-z][a-zA-Z0-9\-]+),\s*p\.?\s*(\d+)\]')


    def replace_short_citation(match):

        filename = match.group(1)       # the truncated filename e.g. "burstein-et-al-2024"
        page = match.group(2)           # the page number e.g. "2"

        # extract author and year — same logic as the two formatters above
        parts = filename.split('-')
        author = parts[0].capitalize()  # first word capitalized e.g. "Burstein"

        # extract year — falls back to "n.d." if no year found
        year_match = re.search(r'\b(19|20)\d{2}\b', filename)
        year = year_match.group() if year_match else "n.d."

        # return the clean readable citation
        return _finalize(author, year, page)




    # handle citations already in near-final format but with lowercase author name
    # e.g. [burstein et al. 2021, p.8] -> [Burstein et al. 2021, p.8]
    lowercase_pattern = re.compile(r'\[([a-z][a-zA-Z]+) et al\. (\d{4}), p\.(\d+)\]')

    def replace_lowercase(match):
        author = match.group(1).capitalize()    # capitalize the author name
        year = match.group(2)                   # year e.g. "2021"
        page = match.group(3)                   # page number e.g. "8"
        return _finalize(author, year, page)




    # apply the replace_citation function to every citation tag found in the text
    # re.sub replaces each match with the result of replace_citation()
    formatted_text = citation_pattern.sub(replace_citation, text)

    # apply the Gemini citation formatter to the already-processed text
    formatted_text = gemini_citation_pattern.sub(replace_gemini_citation, formatted_text)

    # apply the short citation formatter to the already-processed text
    # runs last so it catches any remaining unformatted citations after the first two passes
    formatted_text = short_citation_pattern.sub(replace_short_citation, formatted_text)

    # apply lowercase author fix — runs last to catch any remaining uncapitalized citations
    formatted_text = lowercase_pattern.sub(replace_lowercase, formatted_text)


    logger.info("Citation formatter — formatting complete")
    return formatted_text