"""
parse.py
--------
Extracts structured text from a research paper PDF using the Adobe PDF
Extract API, which uses a trained layout model to identify headings (H1–H6),
paragraphs, lists, and other structural elements with high accuracy.

Usage:
    python parse.py path/to/paper.pdf
    python parse.py pdfs/          ← batch: processes every PDF in the folder

Output:
    pdf_output/<stem>.json  for each PDF

Adobe returns a zip containing structuredData.json. Each element has:
    Path  - e.g. "//Document/Sect/H1", "//Document/Sect/P"
    Text  - the text content of that element

Heading detection is purely structural (Path ends with /H1–/H6 or /Title),
so no font-size heuristics or keyword matching are needed for detection.
Bucket assignment still uses the fuzzy scorer from config.py.
"""

import io, json, logging, os, re, sys, zipfile
from pathlib import Path

from dotenv import load_dotenv

from config import BUCKET_ANCHORS, MATCH_THRESHOLD

# Heading path detection
# Adobe encodes element type in the Path field, e.g.:
#   "//Document/Sect/H1"      ← top-level heading
#   "//Document/Sect/Sect/H2" ← sub-heading
#   "//Document/Sect/P"       ← paragraph
#   "//Document/Title"        ← document title

_HEADING_SUFFIXES = ("/H1", "/H2", "/H3", "/H4", "/H5", "/H6", "/Title")

def is_heading_element(path: str) -> bool:
    """Return True if this Adobe element path represents a heading."""
    return any(path.endswith(suffix) for suffix in _HEADING_SUFFIXES)

def heading_level(path: str) -> int:
    """Return heading level 1–6 (Title → 0) for sorting/display."""
    if path.endswith("/Title"):
        return 0
    for i in range(1, 7):
        if path.endswith(f"/H{i}"):
            return i
    return 99


# ── Text normalisation and fuzzy bucket assignment ────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

_STOP = {"and", "of", "the", "in", "to", "a", "an", "for", "with",
            "on", "at", "by", "from", "this", "that", "are", "is", "be"}

def _tokenise(text: str) -> set:
    return set(text.split())

def _similarity(heading_norm: str, anchor_norm: str) -> float:
    if heading_norm == anchor_norm:
        return 1.0
    h_tokens = _tokenise(heading_norm)
    a_tokens = _tokenise(anchor_norm)
    if not h_tokens or not a_tokens:
        return 0.0
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

def assign_bucket(heading: str) -> tuple:
    """
    Fuzzy-score a heading against all bucket anchors.
    Returns (bucket, score). Falls through to ("misc", score) if below threshold.
    """
    heading_norm = normalise(heading)
    best_bucket = "misc"
    best_score = 0.0
    for bucket, anchors in BUCKET_ANCHORS.items():
        bucket_best = max(
            _similarity(heading_norm, normalise(anchor)) for anchor in anchors
        )
        if bucket_best > best_score:
            best_score = bucket_best
            best_bucket = bucket
    if best_score < MATCH_THRESHOLD:
        return "misc", best_score
    return best_bucket, best_score

# Segmentation

def segment_elements(elements: list) -> tuple:
    """
    Walk Adobe elements in reading order, group body text under headings.
    Returns (preamble_lines, sections) where each section is:
      {heading, level, body, bucket, match_score}
    """
    sections = []
    preamble_lines = []
    current_heading = None
    current_level = None
    current_body = []
    in_preamble = True

    for el in elements:
        path = el.get("Path", "")
        text = el.get("Text", "").strip()
        if not text:
            continue

        if is_heading_element(path):
            if in_preamble:
                in_preamble = False
            else:
                if current_heading is not None:
                    bucket, score = assign_bucket(current_heading)
                    sections.append({
                        "heading": current_heading,
                        "level": current_level,
                        "body": " ".join(current_body).strip(),
                        "bucket": bucket,
                        "match_score": round(score, 3),
                    })
                elif current_body:
                    preamble_lines.extend(current_body)
            current_heading = text
            current_level = heading_level(path)
            current_body = []
        else:
            if in_preamble:
                preamble_lines.append(text)
            else:
                current_body.append(text)

    # Flush final section
    if current_heading is not None:
        bucket, score = assign_bucket(current_heading)
        sections.append({
            "heading": current_heading,
            "level": current_level,
            "body": " ".join(current_body).strip(),
            "bucket": bucket,
            "match_score": round(score, 3),
        })

    return preamble_lines, sections


def merge_into_buckets(preamble_lines: list, sections: list) -> dict:
    """
    Merge sections into four bucket text blobs. Unmapped → misc[].
    """
    buckets = {"abstract": [], "data_description": [], "methods": [], "limitations": []}
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
            "level": s["level"],
            "bucket": s["bucket"],
            "match_score": s["match_score"],
        }
        for s in sections
    ]
    return result


# Output naming

def output_name(pdf_path: Path) -> str:
    """Derive a safe output filename from the first 5 alphanumeric words of the stem."""
    words = re.findall(r"[a-zA-Z0-9]+", pdf_path.stem.lower())
    return "_".join(words[:5])


# Main pipeline

def parse_pdf(pdf_path: Path) -> dict:
    """Full pipeline: PDF → structured dict via Adobe Extract API."""
    elements = extract_structure_via_adobe(pdf_path)

    if not elements:
        return {
            "source_file": pdf_path.name,
            "error": "Adobe API returned no elements — PDF may be scanned/image-based.",
            "abstract": "",
            "data_description": "",
            "methods": "",
            "limitations": "",
            "misc": [],
            "raw_sections": [],
        }

    preamble_lines, sections = segment_elements(elements)
    result = merge_into_buckets(preamble_lines, sections)
    result["source_file"] = pdf_path.name
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
    try:
        result = parse_pdf(pdf_path)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    name = output_name(pdf_path)
    out_path = output_dir / f"{name}.json"
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