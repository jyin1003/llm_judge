# LLM Judge

## Usage
```bash
pip install -r requirements.txt
python run.py parse  pdfs/    # extract text from all PDFs
python run.py score  pdf_output/  # score all extracted JSONs
# or both in one step:
python run.py all   pdfs/
```

## Files
`parse.py` — PDF → structured JSON. The heading detector uses two independent signals: keyword matching (your full list, longest-first) and font size (only fires at 1.6× median, conservatively). The key bug fixed was that data cleaning and data-processing pipeline were being caught by the bare data catch-all before reaching the methods entries — fixed by ordering specific entries before the catch-all in BUCKET_MAP.

`prompts.py` — All rubric logic lives here: four dimension definitions, scoring anchors, guiding questions, and the prompt builder that selects the most relevant bucket text per dimension.

`score.py` — Calls Gemini 1.5 Flash twice per dimension (temp 0 + temp 0.3), flags divergent scores, handles rate limiting with retry/backoff.

`run.py` — Entry point with three modes.

## Flow
```
run.py
  → ingest.py reads PDF, extracts sections into dict
  → prompts.py builds 4 dimension-specific prompts
  → score.py calls Gemini twice per dimension (temp 0 + 0.3)
  → flags divergent scores
  → writes output JSON
```

## Parsing Strategy
1. Extract all text block by block, preserving page order
2. Detect heading boundaries using fuzzy keyword matching against a list (normalised, case-insensitive, partial match)
3. Assign each detected heading + its following body text as a raw section
4.  map raw sections into the four logical buckets (abstract, data_description, methods, limitations) using a fixed mapping table
5. Any text before the first detected heading goes into a preamble field (often contains the actual abstract even without a heading)
6. Sections that don't map to any bucket are kept in a misc field so nothing is discarded

## Fuzzy Matching
To be able to detect relevant headings to extract sections, we use a weighted fuzzy scoring per bucket. - For each detected heading, we compute a similarity score against every keyword in each bucket's anchor list, take the best match per bucket and assign to whichever bucket wins. A configurable threshold can be lowered if if real sections are being lost, allowing us to control whether headings that don't resemble any anchor well enough will fall through to misc rather than being forced into the closest bucket.

The **similarity metric** will be a combination of:
- Jaccard token overlap (40%) — content words only (stopwords stripped), so "Study Design and Analytical Approach" matching "analytical approach" isn't diluted by "and/the/of"
- Substring containment (45%) — if the anchor appears inside the heading or vice versa, score by the proportional length coverage. This is the strongest signal, which is why exact anchors score 1.0
- Prefix bonus (15%) — small nudge when one starts with the other
