"""
parse.py
--------
Extracts structured text from a research paper PDF.

Usage:
    python parse.py path/to/paper.pdf
    python parse.py pdfs/          ← batch: processes every PDF in the folder

Output:
    pdf_output/<stem>.json  for each PDF

Strategy:
    1. Extract all text spans block-by-block, preserving reading order.
    2. Detect headings via keyword matching against a known list
        (case-insensitive, normalised whitespace) plus font-size signal.
    3. Assign each heading + its body text as a raw section.
    4. Map raw sections into four logical buckets using fuzzy similarity
        scoring against anchor phrases defined in config.py:
            abstract         - research aims, context, study overview
            data_description - dataset schema, provenance, variables
            methods          - cleaning, transformations, preprocessing, analysis
            limitations      - bias, gaps, constraints, transparency issues
    5. Preamble (text before first heading) is preserved separately.
    6. Headings that score below MATCH_THRESHOLD against all buckets go
        into misc[] so nothing is discarded.
"""

import json, re, sys
from pathlib import Path

import fitz  # pymupdf

from config import BUCKET_ANCHORS, MATCH_THRESHOLD, HEADING_KEYWORDS_SORTED

# Text normalisation

def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)  # keep hyphens, drop other punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Fuzzy bucket assignment

def _tokenise(text: str) -> set:
    """Split normalised text into a set of word tokens."""
    return set(text.split())


# Common stopwords that inflate token overlap without adding meaning
_STOP = {"and", "of", "the", "in", "to", "a", "an", "for", "with",
            "on", "at", "by", "from", "this", "that", "are", "is", "be"}


def _similarity(heading_norm: str, anchor_norm: str) -> float:
    """
    Compute a similarity score in [0, 1] between a heading and an anchor phrase.

    Three signals combined:
        1. Token overlap (Jaccard on content words): fraction of shared words.
        2. Containment: whether the anchor is fully contained in the heading
            or the heading is fully contained in the anchor (substring on
            normalised strings).
        3. Prefix bonus: heading starts with the anchor or vice versa.

    Calibration:
        - A single shared content word scores ~0.3-0.4
        - Full substring containment scores ~0.8+
        - Exact match scores 1.0
    """
    if heading_norm == anchor_norm:
        return 1.0

    h_tokens = _tokenise(heading_norm)
    a_tokens = _tokenise(anchor_norm)

    if not h_tokens or not a_tokens:
        return 0.0

    h_content = h_tokens - _STOP
    a_content = a_tokens - _STOP

    # Fall back to all tokens if content-word filtering empties a set
    h_cmp = h_content if h_content else h_tokens
    a_cmp = a_content if a_content else a_tokens

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

    score = 0.4 * jaccard + 0.45 * containment + 0.15 * prefix
    return min(score, 1.0)


def assign_bucket(heading: str) -> tuple:
    """
    Score a heading against all bucket anchors and return (bucket, score).
    Returns ("misc", 0.0) if no bucket clears MATCH_THRESHOLD.

    Takes the best-scoring anchor per bucket so a single strong match
    is sufficient — no averaging that would dilute specific matches.
    """
    heading_norm = normalise(heading)
    best_bucket = "misc"
    best_score = 0.0

    for bucket, anchors in BUCKET_ANCHORS.items():
        bucket_best = max(
            _similarity(heading_norm, normalise(anchor))
            for anchor in anchors
        )
        if bucket_best > best_score:
            best_score = bucket_best
            best_bucket = bucket

    if best_score < MATCH_THRESHOLD:
        return "misc", best_score

    return best_bucket, best_score


# Heading detection

def is_heading(line: str, font_size: float, median_font_size: float) -> bool:
    """
    Return True if this line looks like a section heading.
    Two independent signals — either is sufficient:
        1. Normalised line matches a keyword exactly or as a clean prefix.
        2. Font is dramatically larger (1.6x) AND the line is short.
    Hard 80-char cap keeps figure captions and body sentences out.
    """
    if not line.strip():
        return False

    norm = normalise(line)

    if len(norm) > 80:
        return False

    # Signal 1: keyword match
    for kw in HEADING_KEYWORDS_SORTED:
        kw_norm = normalise(kw)
        if norm == kw_norm or norm.startswith(kw_norm + " ") or norm.startswith(kw_norm + ":"):
            return True

    # Signal 2: clearly oversized font, short line
    if font_size >= median_font_size * 1.6 and len(norm) <= 50:
        return True

    return False


# PDF extraction

def extract_spans(pdf_path: Path) -> list:
    """
    Extract all text spans from the PDF preserving reading order.
    Returns a list of dicts: {text, size, page}.
    """
    doc = fitz.open(str(pdf_path))
    spans = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text_parts = []
                max_size = 0.0
                for span in line.get("spans", []):
                    t = span.get("text", "")
                    if t.strip():
                        line_text_parts.append(t)
                        max_size = max(max_size, span.get("size", 0.0))
                line_text = " ".join(line_text_parts).strip()
                if line_text:
                    spans.append({"text": line_text, "size": max_size, "page": page_num})
    doc.close()
    return spans


def compute_median_size(spans: list) -> float:
    """Compute the median font size across all spans (body text baseline)."""
    sizes = sorted(s["size"] for s in spans if s["size"] > 0)
    if not sizes:
        return 12.0
    mid = len(sizes) // 2
    return sizes[mid]


# Segmentation and bucketing

def segment_into_sections(spans: list) -> tuple:
    """
    Walk spans in order, detect heading boundaries, and group body text.
    Returns (preamble_lines, sections) where each section dict contains:
        heading, body, page_start, bucket, match_score
    """
    median_size = compute_median_size(spans)
    sections = []
    current_heading = None
    current_body_lines = []
    current_page = 1
    preamble_lines = []
    in_preamble = True

    for span in spans:
        text = span["text"].strip()
        size = span["size"]

        if is_heading(text, size, median_size):
            if in_preamble:
                in_preamble = False
            else:
                if current_heading is not None:
                    bucket, score = assign_bucket(current_heading)
                    sections.append({
                        "heading": current_heading,
                        "body": " ".join(current_body_lines).strip(),
                        "page_start": current_page,
                        "bucket": bucket,
                        "match_score": round(score, 3),
                    })
                elif current_body_lines:
                    preamble_lines.extend(current_body_lines)
            current_heading = text
            current_body_lines = []
            current_page = span["page"]
        else:
            if in_preamble:
                preamble_lines.append(text)
            else:
                current_body_lines.append(text)

    # Flush the final section
    if current_heading is not None:
        bucket, score = assign_bucket(current_heading)
        sections.append({
            "heading": current_heading,
            "body": " ".join(current_body_lines).strip(),
            "page_start": current_page,
            "bucket": bucket,
            "match_score": round(score, 3),
        })

    return preamble_lines, sections


def merge_into_buckets(preamble_lines: list, sections: list) -> dict:
    """
    Combine all sections sharing a bucket into one text blob per bucket.
    Preserves section order. Unmapped sections go into misc[].
    """
    buckets = {
        "abstract": [],
        "data_description": [],
        "methods": [],
        "limitations": [],
    }
    misc = []

    for sec in sections:
        b = sec["bucket"]
        entry = f"[{sec['heading']}]\n{sec['body']}"
        if b in buckets:
            buckets[b].append(entry)
        else:
            misc.append({
                "heading": sec["heading"],
                "body": sec["body"],
                "page_start": sec["page_start"],
                "match_score": sec["match_score"],
            })

    result = {k: "\n\n".join(v) for k, v in buckets.items()}

    preamble_text = " ".join(preamble_lines).strip()
    if not result["abstract"] and preamble_text:
        result["abstract"] = preamble_text
    elif preamble_text:
        result["preamble"] = preamble_text

    result["misc"] = misc
    result["raw_sections"] = [
        {
            "heading": s["heading"],
            "bucket": s["bucket"],
            "match_score": s["match_score"],
            "page_start": s["page_start"],
        }
        for s in sections
    ]

    return result


def parse_pdf(pdf_path: Path) -> dict:
    """Full pipeline: PDF → structured dict."""
    spans = extract_spans(pdf_path)
    if not spans:
        return {
            "source_file": pdf_path.name,
            "error": "No text extracted — PDF may be scanned/image-based.",
            "abstract": "",
            "data_description": "",
            "methods": "",
            "limitations": "",
            "misc": [],
            "raw_sections": [],
        }

    preamble_lines, sections = segment_into_sections(spans)
    result = merge_into_buckets(preamble_lines, sections)
    result["source_file"] = pdf_path.name
    result["page_count"] = fitz.open(str(pdf_path)).page_count
    result["sections_detected"] = len(sections)

    empty = [k for k in ("abstract", "data_description", "methods", "limitations")
             if not result.get(k)]
    if empty:
        result["parse_warnings"] = [f"No content mapped to bucket: {e}" for e in empty]

    return result


def process_path(target: Path, output_dir: Path):
    """Process a single PDF or all PDFs in a directory."""
    if target.is_dir():
        pdfs = sorted(target.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {target}")
            return
        for pdf in pdfs:
            process_single(pdf, output_dir)
    elif target.suffix.lower() == ".pdf":
        process_single(target, output_dir)
    else:
        print(f"Not a PDF or directory: {target}")
        sys.exit(1)


def process_single(pdf_path: Path, output_dir: Path):
    print(f"Parsing: {pdf_path.name} ...", end=" ", flush=True)
    result = parse_pdf(pdf_path)
    out_path = output_dir / (pdf_path.stem + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    warnings = result.get("parse_warnings", [])
    n_sections = result.get("sections_detected", 0)
    status = "⚠ " + "; ".join(warnings) if warnings else "✓"
    print(f"{status}  ({n_sections} sections → {out_path.name})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse.py <pdf_file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    output_dir = Path(__file__).parent / "pdf_output"
    output_dir.mkdir(exist_ok=True)

    process_path(target, output_dir)