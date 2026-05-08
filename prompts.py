"""
prompts.py
----------
Rubric definitions and prompt builders for the LLM judge.

Four dimensions, each scored 0–2:
  1. documentation_quality   – how well cleaning/preprocessing decisions are explained
  2. contextual_transparency – how clearly the original data context is acknowledged
  3. limitation_awareness    – how explicitly limitations from repurposing are addressed
  4. metadata_sufficiency    – how adequately schema, provenance, and variables are described

Each dimension has:
  - a theoretical definition
  - four guiding questions
  - operationalised scoring anchors (0, 1, 2)
  - a mapping to which parsed bucket(s) are most relevant
"""

# ── Dimension definitions ─────────────────────────────────────────────────────

DIMENSIONS = {
    "documentation_quality": {
        "label": "Documentation Quality",
        "description": (
            "The degree to which cleaning, preprocessing, and transformation decisions "
            "are documented with explicit rationale — not just named, but explained."
        ),
        "guiding_questions": [
            "Does the text describe specific cleaning or preprocessing steps taken?",
            "Does it explain WHY those steps were taken, not just WHAT was done?",
            "Are decisions about handling missing data, outliers, or schema changes justified?",
            "Is there enough detail for another researcher to reproduce the preprocessing pipeline?",
        ],
        "anchors": {
            0: (
                "Absent. No cleaning or preprocessing decisions are described, "
                "or only a vague generic statement is present "
                "(e.g., 'data were cleaned prior to analysis')."
            ),
            1: (
                "Partial. At least one specific cleaning or preprocessing step is named "
                "(e.g., 'outliers were removed', 'missing values were imputed') "
                "but no rationale is provided for why that decision was made."
            ),
            2: (
                "Present. At least one cleaning or preprocessing decision is described "
                "with its explicit rationale "
                "(e.g., 'records missing more than 30% of fields were excluded because "
                "imputation at this level would introduce substantial uncertainty')."
            ),
        },
        "primary_buckets": ["methods", "data_description"],
        "question": (
            "Does the text explicitly state WHY a cleaning or preprocessing decision was made, "
            "not just what was done? For example, does it explain why missing values were imputed "
            "in a particular way, or why certain records were excluded?"
        ),
    },

    "contextual_transparency": {
        "label": "Contextual Transparency",
        "description": (
            "The degree to which the authors acknowledge that the data were collected for a "
            "different original purpose, and reflect on how that context shift affects validity."
        ),
        "guiding_questions": [
            "Does the text acknowledge that the dataset was originally created for a different purpose?",
            "Does it discuss how the original collection context differs from the new use case?",
            "Are assumptions made during the original data collection identified as potentially problematic?",
            "Does the text reflect on how context shift affects the validity or reliability of findings?",
        ],
        "anchors": {
            0: (
                "Absent. The text treats the dataset as if it were purpose-built for the current study. "
                "No acknowledgement of original intent or context shift."
            ),
            1: (
                "Partial. The original source of the data is mentioned or cited, "
                "but there is no substantive reflection on how the original collection context "
                "differs from or affects the current use."
            ),
            2: (
                "Present. The text explicitly acknowledges that the data were collected for a "
                "different purpose AND discusses how this affects the validity, interpretation, "
                "or reliability of the repurposed analysis."
            ),
        },
        "primary_buckets": ["abstract", "data_description", "limitations"],
        "question": (
            "Does the text explicitly acknowledge that this dataset was originally collected for "
            "a different purpose, AND reflect on how that context shift affects the validity "
            "or interpretation of the current analysis?"
        ),
    },

    "limitation_awareness": {
        "label": "Limitation Awareness",
        "description": (
            "The degree to which limitations arising specifically from data repurposing are "
            "explicitly identified — including bias, missing variables, temporal mismatches, "
            "and consent/ethical concerns."
        ),
        "guiding_questions": [
            "Are any limitations of using this dataset for this new purpose explicitly stated?",
            "Does the text identify biases introduced or amplified by the repurposing?",
            "Are missing variables, omitted context, or proxy measures acknowledged as limitations?",
            "Are ethical, consent, or privacy concerns from repurposing addressed?",
        ],
        "anchors": {
            0: (
                "Absent. No limitations specific to the repurposed use of the data are identified. "
                "Generic study limitations unrelated to data repurposing do not qualify."
            ),
            1: (
                "Partial. At least one limitation related to data repurposing is mentioned "
                "(e.g., 'the dataset may not be fully representative') but without specificity "
                "about what the limitation is, how it arose from repurposing, or its practical impact."
            ),
            2: (
                "Present. At least one limitation is described with specificity: "
                "what it is, how it arises from the repurposing context, and what its "
                "practical impact on the findings or generalisability might be."
            ),
        },
        "primary_buckets": ["limitations", "discussion"],
        "question": (
            "Does the text identify at least one specific limitation that arises from "
            "repurposing this dataset for a new purpose — and does it explain what the limitation "
            "is, how it arises, and what its practical impact might be?"
        ),
    },

    "metadata_sufficiency": {
        "label": "Metadata Sufficiency",
        "description": (
            "The degree to which the dataset's schema, provenance, variable definitions, "
            "and collection conditions are described in enough detail to assess suitability "
            "for the repurposed use."
        ),
        "guiding_questions": [
            "Does the text describe the structure or schema of the dataset (fields, types, units)?",
            "Is the provenance of the data clearly identified (who collected it, when, how)?",
            "Are key variables or constructs defined with enough precision to assess their meaning?",
            "Is there sufficient information to judge whether the dataset is appropriate for the new purpose?",
        ],
        "anchors": {
            0: (
                "Absent. The dataset is named or cited but its structure, provenance, "
                "and key variables are not described. A reader cannot assess what is in the data."
            ),
            1: (
                "Partial. Some structural or provenance information is given "
                "(e.g., a list of variable names, or a general description of the data source) "
                "but key details are missing — units are unstated, collection conditions are vague, "
                "or important variables are undefined."
            ),
            2: (
                "Present. The dataset's schema or key variables are described with enough "
                "specificity that a reader could assess whether the data is appropriate for "
                "the repurposed use — including provenance, collection conditions, "
                "and at least the most important field definitions."
            ),
        },
        "primary_buckets": ["data_description", "methods"],
        "question": (
            "Does the text describe the dataset's structure, key variables, and provenance "
            "in enough detail that a reader could assess whether the data is appropriate "
            "for the new purpose it is being repurposed for?"
        ),
    },
}


# ── Prompt builders ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a rigorous academic evaluator assessing research papers on data repurposing quality.

You are scoring one specific dimension: {label}

DEFINITION:
{description}

GUIDING QUESTIONS:
{guiding_questions}

SCORING ANCHORS:
  Score 0 — {anchor_0}
  Score 1 — {anchor_1}
  Score 2 — {anchor_2}

CRITICAL RULES:
- Score 0 if the text does not contain enough evidence to justify a higher score. Treat silence as a data point.
- Do not infer or assume compliance. Only score what is explicitly stated in the text.
- Generic statements (e.g. "limitations exist", "data were cleaned") score 0 or 1, never 2.
- Your evidence quote must be a direct excerpt or close paraphrase from the text. Do not fabricate.

Respond ONLY with a valid JSON object. No preamble, no explanation outside the JSON, no markdown fences.
"""

USER_PROMPT_TEMPLATE = """TEXT EXTRACTED FROM PAPER:
---
{text}
---

SCORING QUESTION:
{question}

Respond ONLY with this exact JSON structure:
{{
  "score": <0, 1, or 2>,
  "evidence": "<direct quote or close paraphrase from the text, max 60 words. Write null if absent>",
  "confidence": "<high, medium, or low>",
  "reasoning": "<one sentence explaining your score>"
}}"""


def build_prompts(dimension_key: str, parsed: dict) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for a given dimension and parsed PDF dict.
    Concatenates the most relevant bucket text(s) for that dimension.
    """
    dim = DIMENSIONS[dimension_key]

    # Gather relevant text from primary buckets, fallback to all buckets
    text_parts = []
    for bucket in dim["primary_buckets"]:
        content = parsed.get(bucket, "")
        if content and content.strip():
            text_parts.append(f"=== {bucket.upper()} ===\n{content.strip()}")

    # If primary buckets are mostly empty, include everything
    if len(text_parts) < 1:
        for bucket in ("abstract", "data_description", "methods", "limitations"):
            content = parsed.get(bucket, "")
            if content and content.strip():
                text_parts.append(f"=== {bucket.upper()} ===\n{content.strip()}")

    # Also include misc sections in case they contain relevant content
    for misc_sec in parsed.get("misc", []):
        body = misc_sec.get("body", "").strip()
        if body:
            text_parts.append(f"=== MISC: {misc_sec['heading'].upper()} ===\n{body}")

    combined_text = "\n\n".join(text_parts) if text_parts else "[No relevant text extracted from this paper.]"

    # Truncate to avoid exceeding context limits (~12k chars is safe for most models)
    if len(combined_text) > 12000:
        combined_text = combined_text[:12000] + "\n\n[... text truncated for length ...]"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        label=dim["label"],
        description=dim["description"],
        guiding_questions="\n".join(f"  {i+1}. {q}" for i, q in enumerate(dim["guiding_questions"])),
        anchor_0=dim["anchors"][0],
        anchor_1=dim["anchors"][1],
        anchor_2=dim["anchors"][2],
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        text=combined_text,
        question=dim["question"],
    )

    return system_prompt, user_prompt


def all_dimension_keys() -> list[str]:
    return list(DIMENSIONS.keys())
