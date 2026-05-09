"""
parse.py
--------
Converts Adobe PDF Extract API raw output (structuredData.json) into a
structured JSON suitable for the LLM judge pipeline.

Reads from:   pdf_raw_output/<stem>_structuredData.json
Writes to:    pdf_output/<stem>.json

The Adobe Extract API tags every element with a Path that encodes its
structural role, e.g.:

    //Document/H1          ← top-level heading
    //Document/Sect/H2     ← sub-heading (any nesting depth)
    //Document/P           ← paragraph
    //Document/P[3]        ← indexed paragraph (same logical type)
    //Document/P/ParagraphSpan      ← paragraph that was split across pages;
    //Document/P/ParagraphSpan[2]   ← its continuation — must be merged
    //Document/Title       ← document title (treated as H0)

Sections are grouped by heading → following body text.  Each section is
fuzzy-matched against bucket anchor keywords (defined in config.py) to
assign it to one of four logical buckets:

    abstract          - introduction, background, study aims, summary
    data_description  - dataset, variables, provenance, schema
    methods           - methodology, preprocessing, analysis, results
    limitations       - limitations, discussion, conclusion, future work

Multiple sections may contribute to the same bucket; their text is
concatenated in reading order.  Sections that do not match any bucket
clearly enough are stored in ``misc`` for transparency.

Usage:
    python parse.py path/to/paper_structuredData.json
    python parse.py pdf_raw_output/          ← batch: every *_structuredData.json

Output per file:
    pdf_output/<stem>.json
"""

import json, logging, re, sys
from pathlib import Path

from config import BUCKET_ANCHORS, MATCH_THRESHOLD

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Path classification helpers

# Heading suffixes as encoded by Adobe's structure tree.
_HEADING_SUFFIXES = ("/H1", "/H2", "/H3", "/H4", "/H5", "/H6", "/Title")

# Element path endings that should be treated as body text.
# ParagraphSpan indicates a paragraph split across a page boundary — the
# continuation must be merged with the preceding paragraph body text.
_BODY_SUFFIXES = ("/P", "/LBody", "/ParagraphSpan")

# Paths containing these substrings are non-body structural elements that
# carry no analytical content and should be skipped entirely.
_SKIP_SUBSTRINGS = (
    "/Figure",   # image placeholder — no text
    "/Footnote", # footnote — peripheral to main argument
    "/Aside",    # sidebar, caption boxes, repository headers
    "/Lbl",      # list labels ("1.", "A.", bullet glyphs)
    "/TOC",      # table of contents entries
    "/Artifact", # page headers / footers / watermarks
)


def _path_tail(path: str) -> str:
    """Return the final structural component of an Adobe path, without index.

    Examples:
        '//Document/Sect/H2'        → '/H2'
        '//Document/P[3]'           → '/P'
        '//Document/P/ParagraphSpan[2]' → '/ParagraphSpan'
    """
    # Strip numeric index suffix: /H1[3] → /H1, /P[10] → /P
    tail = re.sub(r"\[\d+\]$", "", path)
    # Return everything from the last '/' onwards
    slash_pos = tail.rfind("/")
    return tail[slash_pos:] if slash_pos != -1 else tail


def is_heading(path: str) -> bool:
    """True if this Adobe element represents a heading of any level."""
    tail = _path_tail(path)
    return any(tail == suffix for suffix in _HEADING_SUFFIXES)


def is_body(path: str) -> bool:
    """True if this element contributes to body / paragraph text.

    Handles both plain paragraphs and ParagraphSpan continuations that Adobe
    emits when a paragraph is split across a page boundary.
    """
    # Reject anything that belongs to a skip category first.
    if any(skip in path for skip in _SKIP_SUBSTRINGS):
        return False
    tail = _path_tail(path)
    return any(tail == suffix for suffix in _BODY_SUFFIXES)


def heading_level(path: str) -> int:
    """Return numeric heading level (0 = Title, 1–6 = H1–H6, 99 = unknown)."""
    tail = _path_tail(path)
    if tail == "/Title":
        return 0
    for i in range(1, 7):
        if tail == f"/H{i}":
            return i
    return 99


# Text normalisation and fuzzy bucket matching

_STOP = {
    "and", "of", "the", "in", "to", "a", "an", "for", "with",
    "on", "at", "by", "from", "this", "that", "are", "is", "be",
}


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenise(text: str) -> set:
    return set(text.split())


def _similarity(heading_norm: str, anchor_norm: str) -> float:
    """Weighted similarity between a normalised heading and a normalised anchor.

    Three signals, weighted to favour substring containment (strongest signal
    for short section headings):
        Jaccard token overlap  40%
        Substring containment  45%
        Prefix bonus           15%
    """
    if heading_norm == anchor_norm:
        return 1.0

    h_tokens = _tokenise(heading_norm)
    a_tokens = _tokenise(anchor_norm)
    if not h_tokens or not a_tokens:
        return 0.0

    # Content-word sets (stopwords removed, fall back to full set if empty)
    h_cmp = (h_tokens - _STOP) or h_tokens
    a_cmp = (a_tokens - _STOP) or a_tokens

    intersection = len(h_cmp & a_cmp)
    union = len(h_cmp | a_cmp)
    jaccard = intersection / union if union else 0.0

    containment = 0.0
    if anchor_norm in heading_norm:
        containment = len(anchor_norm) / max(len(heading_norm), 1)
    elif heading_norm in anchor_norm:
        containment = len(heading_norm) / max(len(anchor_norm), 1)

    prefix = 0.2 if (
        heading_norm.startswith(anchor_norm) or anchor_norm.startswith(heading_norm)
    ) else 0.0

    return min(0.4 * jaccard + 0.45 * containment + 0.15 * prefix, 1.0)


def assign_bucket(heading: str, body: str = "") -> tuple[str, float]:
    """Fuzzy-match a section heading against all bucket anchors.

    If no bucket clears MATCH_THRESHOLD, a body-text frequency fallback
    is applied specifically for data_description — catching sections whose
    headings are dataset-specific (e.g. 'EZ-Link Smart Card in Singapore')
    but whose body is clearly describing data structure or provenance.

    Returns (bucket_name, score).  Falls through to ('misc', score) when
    neither the heading match nor the body fallback fires.
    """
    heading_norm = _normalise(heading)
    best_bucket = "misc"
    best_score = 0.0

    for bucket, anchors in BUCKET_ANCHORS.items():
        bucket_best = max(
            _similarity(heading_norm, _normalise(anchor)) for anchor in anchors
        )
        if bucket_best > best_score:
            best_score = bucket_best
            best_bucket = bucket

    if best_score >= MATCH_THRESHOLD:
        return best_bucket, best_score

    # Heading did not match any bucket clearly enough.
    # Apply body-frequency fallback for data_description only.
    if _body_suggests_data_description(body):
        return "data_description", best_score  # score reflects heading weakness

    return "misc", best_score

# Body-text frequency fallback for data_description which is harder to match
# Content words drawn from the data_description anchor list that are
# strong signals when they appear repeatedly in body text.
_DATA_DESCRIPTION_SIGNALS = {
    "data", "dataset", "datasets", "records", "variables", "variable",
    "schema", "attributes", "attribute", "provenance", "corpus",
    "cohort", "annotation", "labelling", "collection", "acquisition",
    "survey", "features", "field", "fields", "sample", "samples",
}

# Minimum normalised frequency (signal word occurrences / total words)
# required to trigger a data_description reclassification.
_DATA_BODY_FREQ_THRESHOLD = 0.04  # ~4 signal words per 100 words

def _body_suggests_data_description(body: str) -> bool:
    """Return True if body text contains enough data-description signal words
    to justify mapping a section to data_description, regardless of its heading.

    This is a fallback for sections whose headings are domain-specific
    (e.g. 'EZ-Link Smart Card in Singapore', 'Available Data Set') and
    therefore score poorly against generic anchor phrases, but whose body
    text is clearly describing a dataset's structure, provenance, or variables.
    """
    if not body:
        return False
    tokens = re.findall(r"[a-z]+", body.lower())
    if not tokens:
        return False
    signal_count = sum(1 for t in tokens if t in _DATA_DESCRIPTION_SIGNALS)
    return (signal_count / len(tokens)) >= _DATA_BODY_FREQ_THRESHOLD


# Core segmentation: elements → sections
def segment_elements(elements: list) -> tuple[list, list]:
    """Walk Adobe elements in reading order and group body text under headings.

    Adobe sometimes splits a single paragraph across a page boundary, emitting
    the first half as ``//Document/P/ParagraphSpan`` and the continuation as
    ``//Document/P/ParagraphSpan[2]``.  These must be concatenated rather than
    treated as separate paragraphs.

    Returns:
        preamble_lines  - text lines that appear before the first heading
                          (often contains the actual abstract)
        sections        - list of dicts, each with keys:
                            heading, level, body, bucket, match_score
    """
    sections: list[dict] = []
    preamble_lines: list[str] = []

    current_heading: str | None = None
    current_level: int | None = None
    current_body_parts: list[str] = []
    in_preamble = True

    def _flush_section():
        """Persist the current section to the sections list."""
        nonlocal current_heading, current_level, current_body_parts
        if current_heading is not None:
            bucket, score = assign_bucket(current_heading)
            sections.append({
                "heading": current_heading,
                "level": current_level,
                "body": " ".join(current_body_parts).strip(),
                "bucket": bucket,
                "match_score": round(score, 3),
            })
        elif current_body_parts:
            # Body text that preceded any heading → preamble
            preamble_lines.extend(current_body_parts)
        current_heading = None
        current_level = None
        current_body_parts = []

    for el in elements:
        path = el.get("Path", "")
        text = el.get("Text", "").strip()

        if not text:
            # Skip non-text structural elements (figures, rule lines, etc.)
            continue

        if is_heading(path):
            if in_preamble:
                # We have now encountered the first heading — preamble ends here.
                preamble_lines.extend(current_body_parts)
                current_body_parts = []
                in_preamble = False
            else:
                _flush_section()

            current_heading = text
            current_level = heading_level(path)

        elif is_body(path):
            if in_preamble:
                current_body_parts.append(text)
            else:
                current_body_parts.append(text)

        # All other element types (Figure placeholders, Footnotes, Asides, Lbl)
        # are silently skipped — they were filtered by is_body() returning False
        # and is_heading() returning False.

    # Flush the final section after iterating all elements.
    _flush_section()

    return preamble_lines, sections

# Bucket merging: sections → four-bucket output dict
def merge_into_buckets(preamble_lines: list, sections: list) -> dict:
    """Merge all sections into the four logical bucket text blobs.

    Multiple sections that map to the same bucket are concatenated in reading
    order, separated by a blank line, with a heading label prefix so the
    provenance of each contribution remains clear.

    Sections that do not match any bucket clearly enough are stored in
    ``misc`` (list of dicts) so nothing is silently discarded.

    The preamble (pre-heading text) is used as the abstract when no section
    was explicitly mapped there.
    """
    # Each bucket accumulates a list of labelled text entries.
    bucket_entries: dict[str, list[str]] = {
        "abstract": [],
        "data_description": [],
        "methods": [],
        "limitations": [],
    }
    misc: list[dict] = []

    for sec in sections:
        b = sec["bucket"]
        # Prefix each contribution with its heading so the reader can tell
        # which section each piece of text came from.
        entry = f"[{sec['heading']}]\n{sec['body']}" if sec["body"] else f"[{sec['heading']}]"

        if b in bucket_entries:
            bucket_entries[b].append(entry)
        else:
            misc.append({
                "heading": sec["heading"],
                "body": sec["body"],
                "match_score": sec["match_score"],
            })

    # Join multiple contributions per bucket with a blank line.
    result: dict = {k: "\n\n".join(v) for k, v in bucket_entries.items()}

    # Preamble fallback / supplement for abstract.
    preamble_text = " ".join(preamble_lines).strip()
    if not result["abstract"] and preamble_text:
        result["abstract"] = preamble_text
    elif preamble_text:
        # Prepend the preamble so it appears before any explicitly headed sections.
        result["abstract"] = preamble_text + "\n\n" + result["abstract"]

    result["misc"] = misc

    # Preserve a flat section index for debugging / manual inspection.
    result["raw_sections"] = [
        {
            "heading": s["heading"],
            "level": s["level"],
            "bucket": s["bucket"],
            "match_score": s["match_score"],
        }
        for s in sections
    ]

    return result


# File-level helpers
def output_name(raw_json_path: Path) -> str:
    """Derive a safe output filename from the raw JSON stem.

    The Adobe pipeline names raw files as ``<stem>_structuredData.json``.
    We strip that suffix and keep the first five alphanumeric tokens.

    Example:
        poverty_mapping_in_the_age_of_structuredData.json
        → poverty_mapping_in_the_age
    """
    stem = raw_json_path.stem  # e.g. "poverty_mapping_in_the_age_of_structuredData"
    # Remove trailing _structuredData or _raw_output suffixes if present.
    stem = re.sub(r"_structuredData$", "", stem)
    stem = re.sub(r"_raw_output$", "", stem)
    words = re.findall(r"[a-zA-Z0-9]+", stem.lower())
    return "_".join(words[:5])


# Main pipeline
def parse_raw_json(raw_json_path: Path) -> dict:
    """Full pipeline: raw Adobe structuredData.json → structured output dict.

    Reads the list of elements from ``data["elements"]``, runs segmentation
    and bucket assignment, then returns the final structured dict.
    """
    with open(raw_json_path, encoding="utf-8") as f:
        data = json.load(f)

    elements: list = data.get("elements", [])

    if not elements:
        return {
            "source_file": raw_json_path.name,
            "error": (
                "structuredData.json contained no elements — "
                "PDF may be scanned / image-based or the extraction failed."
            ),
            "abstract": "",
            "data_description": "",
            "methods": "",
            "limitations": "",
            "misc": [],
            "raw_sections": [],
        }

    preamble_lines, sections = segment_elements(elements)
    result = merge_into_buckets(preamble_lines, sections)
    logging.debug("Merged %d sections into buckets", len(sections))

    result["source_file"] = raw_json_path.name
    result["sections_detected"] = len(sections)

    # Warn about buckets that received no content — useful for diagnosing
    # PDFs whose headings are unusual or whose structure is atypical.
    empty = [
        k for k in ("abstract", "data_description", "methods", "limitations")
        if not result.get(k)
    ]
    if empty:
        result["parse_warnings"] = [
            f"No content mapped to bucket: {e}" for e in empty
        ]

    return result


def process_path(target: Path, output_dir: Path) -> None:
    """Process a single raw structuredData JSON or all such files in a directory."""
    if target.is_dir():
        raw_jsons = sorted(target.glob("*_structuredData.json"))
        if not raw_jsons:
            logging.warning("No *_structuredData.json files found in %s", target)
            return
        for rj in raw_jsons:
            process_single(rj, output_dir)
    elif target.suffix.lower() == ".json":
        process_single(target, output_dir)
    else:
        logging.error("Not a JSON file or directory: %s", target)
        sys.exit(1)


def process_single(raw_json_path: Path, output_dir: Path) -> None:
    """Parse one raw JSON file and write the structured output."""
    logging.info("Parsing: %s ...", raw_json_path.name)
    try:
        result = parse_raw_json(raw_json_path)
    except Exception:
        logging.exception("Failed to parse %s", raw_json_path.name)
        return

    name = output_name(raw_json_path)
    out_path = output_dir / f"{name}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    warnings = result.get("parse_warnings", [])
    n_sections = result.get("sections_detected", 0)
    # Summarise bucket fill for quick visual inspection in the terminal.
    filled = [
        k for k in ("abstract", "data_description", "methods", "limitations")
        if result.get(k)
    ]
    misc_count = len(result.get("misc", []))

    status = "⚠  " + "; ".join(warnings) if warnings else "✓"
    logging.info(
        "%s  (%d sections | buckets: %s | misc: %d → %s)",
        status,
        n_sections,
        ", ".join(filled) if filled else "none",
        misc_count,
        out_path.name,
    )

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python parse.py path/to/paper_structuredData.json\n"
            "  python parse.py pdf_raw_output/   ← batch mode\n"
        )
        sys.exit(1)

    target = Path(sys.argv[1])
    output_dir = Path(__file__).parent / "pdf_output"
    output_dir.mkdir(exist_ok=True)

    process_path(target, output_dir)