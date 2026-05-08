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
    2. Detect headings via fuzzy keyword matching against a known list
       (case-insensitive, normalised whitespace, partial-match friendly).
    3. Assign each heading + its body text as a raw section.
    4. Map raw sections into four logical buckets:
         abstract         – research aims, context, study overview
         data_description – dataset schema, provenance, variables
         methods          – cleaning, transformations, preprocessing, analysis
         limitations      – bias, gaps, constraints, transparency issues
    5. Preamble (text before first heading) is preserved separately.
    6. Unmapped sections go into misc[] so nothing is discarded.
"""

import json
import re
import sys
from pathlib import Path

import fitz  # pymupdf


# ── Heading keyword list ──────────────────────────────────────────────────────
# Each entry is a normalised string. Matching is done after normalising both
# the keyword and the candidate line (lowercase, collapse whitespace).

HEADING_KEYWORDS = [
    "abstract",
    "introduction",
    "results",
    "conclusion",
    "conclusions and recommendations",
    "data",
    "methods",
    "methodology",
    "challenges",
    "study aims",
    "objectives of this study",
    "key functionalities",
    "automated approaches to handling missing data",
    "outlier detection and removal",
    "noise reduction",
    "feature selection and dimensionality reduction",
    "natural language processing in healthcare data preprocessing",
    "limitations of current ai models",
    "challenges and limitations of ai in data preprocessing",
    "interpretability and transparency of ai algorithms",
    "quantitative analysis",
    "sensitive attribute labelling",
    "procurement of sensitive attributes",
    "data perturbation",
    "summary",
    "limitations of popular algorithmic fairness datasets",
    "transparency",
    "acquisition",
    "research design and analysis",
    "algorithms",
    "biases",
    "research results and interpretation",
    "construction of samples",
    "modifications",
    "augmenting",
    "model validation",
    "evaluation of r²",
    "data records",
    "description of data",
    "usage notes",
    "open peer review",
    "reviewer reports",
    "data and methods",
    "selection criteria",
    "basic data-cleaning process",
    "pre-processed before sentiment analysis",
    "evaluation results of classifiers",
    "variable definitions",
    "analysis at the bed level",
    "physician fixed effects",
    "table summary of robustness tests for alternative explanations",
    "datasets",
    "experiments",
    "data cleaning",
    "experimental setup",
    "issues",
    "correction strategies",
    "discussion",
    "limitations",
    "perspective",
    "data-processing pipeline",
    "detection",
    "inference",
    "data-quality issues",
    "data preparation",
    "sensitivity analysis",
]

# Sort longest-first so more specific phrases match before shorter substrings
HEADING_KEYWORDS_SORTED = sorted(HEADING_KEYWORDS, key=len, reverse=True)


# ── Bucket mapping ────────────────────────────────────────────────────────────
# Maps normalised heading text → logical bucket name.
# Uses startswith / substring matching (checked in order).

BUCKET_MAP = [
    # ── abstract / aims ──
    ("abstract",                        "abstract"),
    ("introduction",                    "abstract"),
    ("study aims",                      "abstract"),
    ("objectives of this study",        "abstract"),
    ("perspective",                     "abstract"),
    ("summary",                         "abstract"),
    ("open peer review",                "abstract"),
    ("reviewer reports",                "abstract"),

    # ── data description ──
    ("data records",                    "data_description"),
    ("description of data",             "data_description"),
    ("datasets",                        "data_description"),
    ("data and methods",                "data_description"),
    ("variable definitions",            "data_description"),
    ("construction of samples",         "data_description"),
    ("acquisition",                     "data_description"),
    ("usage notes",                     "data_description"),
    # NOTE: bare "data" catch-all must come AFTER all data-prefixed methods entries below

    # ── methods / transformations ──
    # data-prefixed entries listed before the bare "data" catch-all in data_description
    ("data cleaning",                   "methods"),
    ("data-processing pipeline",        "methods"),
    ("data preparation",                "methods"),
    ("data-quality issues",             "methods"),
    ("basic data-cleaning process",     "methods"),
    ("methods",                         "methods"),
    ("methodology",                     "methods"),
    ("research design and analysis",    "methods"),
    ("experimental setup",              "methods"),
    ("experiments",                     "methods"),
    ("automated approaches",            "methods"),
    ("outlier detection",               "methods"),
    ("noise reduction",                 "methods"),
    ("feature selection",               "methods"),
    ("natural language processing",     "methods"),
    ("model validation",                "methods"),
    ("evaluation of r",                 "methods"),
    ("evaluation results",              "methods"),
    ("algorithms",                      "methods"),
    ("detection",                       "methods"),
    ("inference",                       "methods"),
    ("modifications",                   "methods"),
    ("augmenting",                      "methods"),
    ("selection criteria",              "methods"),
    ("pre-processed",                   "methods"),
    ("sensitive attribute labelling",   "methods"),
    ("procurement of sensitive",        "methods"),
    ("data perturbation",               "methods"),
    ("quantitative analysis",           "methods"),
    ("analysis at the bed level",       "methods"),
    ("physician fixed effects",         "methods"),
    ("correction strategies",           "methods"),
    ("sensitivity analysis",            "methods"),
    ("research results",                "methods"),
    ("results",                         "methods"),

    # bare "data" catch-all — must come after all data-prefixed methods entries above
    ("data",                            "data_description"),

    # ── limitations ──
    ("limitations of current",         "limitations"),
    ("limitations of popular",         "limitations"),
    ("challenges and limitations",     "limitations"),
    ("limitations",                    "limitations"),
    ("challenges",                     "limitations"),
    ("biases",                         "limitations"),
    ("transparency",                   "limitations"),
    ("interpretability",               "limitations"),
    ("issues",                         "limitations"),
    ("discussion",                     "limitations"),
    ("conclusions and recommendations","limitations"),
    ("conclusion",                     "limitations"),
]


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text).strip().lower()


def is_heading(line: str, font_size: float, median_font_size: float) -> bool:
    """
    Return True if this line looks like a section heading.
    Uses two independent signals — either is sufficient:
      1. Font size is noticeably larger than the body median.
      2. Normalised line text matches (or starts with) a keyword.
    Also enforces a length cap: headings rarely exceed 120 characters.
    """
    if not line.strip():
        return False

    norm = normalise(line)

    # Hard length cap: real headings are short.
    # Allow slightly longer for known multi-word headings (e.g. "Table Summary of...")
    # but cap aggressively for font-size-only detection.
    if len(norm) > 80:
        return False

    # Signal 1: keyword match (any font size) — exact or clean prefix
    for kw in HEADING_KEYWORDS_SORTED:
        if norm == kw or norm.startswith(kw + " ") or norm.startswith(kw + ":"):
            return True

    # Signal 2: clearly oversized font with no keyword — very conservative
    # Only fire if font is dramatically larger (1.6x) AND line is short
    if font_size >= median_font_size * 1.6 and len(norm) <= 50:
        return True

    return False


def assign_bucket(heading_norm: str) -> str:
    """Map a normalised heading string to a logical bucket."""
    for prefix, bucket in BUCKET_MAP:
        if heading_norm.startswith(prefix) or prefix in heading_norm:
            return bucket
    return "misc"


def extract_spans(pdf_path: Path) -> list[dict]:
    """
    Extract all text spans from the PDF preserving reading order.
    Returns a list of dicts: {text, size, page}.
    """
    doc = fitz.open(str(pdf_path))
    spans = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # skip image blocks
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


def compute_median_size(spans: list[dict]) -> float:
    """Compute the median font size across all spans (body text baseline)."""
    sizes = sorted(s["size"] for s in spans if s["size"] > 0)
    if not sizes:
        return 12.0
    mid = len(sizes) // 2
    return sizes[mid]


def segment_into_sections(spans: list[dict]) -> list[dict]:
    """
    Walk spans in order, detect heading boundaries, and group body text.
    Returns a list of section dicts: {heading, body, page_start, bucket}.
    """
    median_size = compute_median_size(spans)
    sections = []
    current_heading = None
    current_body_lines = []
    current_page = 1
    preamble_lines = []  # text before any heading is found
    in_preamble = True

    for span in spans:
        text = span["text"].strip()
        size = span["size"]

        if is_heading(text, size, median_size):
            if in_preamble:
                # Save everything before the first heading
                in_preamble = False
            else:
                # Flush previous section
                if current_heading is not None:
                    sections.append({
                        "heading": current_heading,
                        "body": " ".join(current_body_lines).strip(),
                        "page_start": current_page,
                        "bucket": assign_bucket(normalise(current_heading)),
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

    # Flush final section
    if current_heading is not None:
        sections.append({
            "heading": current_heading,
            "body": " ".join(current_body_lines).strip(),
            "page_start": current_page,
            "bucket": assign_bucket(normalise(current_heading)),
        })

    return preamble_lines, sections


def merge_into_buckets(preamble_lines: list[str], sections: list[dict]) -> dict:
    """
    Combine all sections that share a bucket into one text blob per bucket.
    Preserves section order within each bucket.
    Also returns misc[] for unmapped sections and raw_sections[] for full detail.
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
            })

    # Join each bucket's sections with a separator
    result = {k: "\n\n".join(v) for k, v in buckets.items()}

    # Preamble heuristic: if abstract bucket is empty, use preamble text
    preamble_text = " ".join(preamble_lines).strip()
    if not result["abstract"] and preamble_text:
        result["abstract"] = preamble_text
    elif preamble_text:
        result["preamble"] = preamble_text

    result["misc"] = misc
    result["raw_sections"] = [
        {"heading": s["heading"], "bucket": s["bucket"], "page_start": s["page_start"]}
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

    # Warn if any bucket is empty
    empty = [k for k in ("abstract", "data_description", "methods", "limitations") if not result.get(k)]
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
