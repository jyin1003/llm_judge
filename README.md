# LLM Judge

## Usage
```bash
python -m venv .venv   # create virtual env
.venv\Scripts\Activate.ps1  # activate virutal env
pip install -r requirements.txt
python run.py parse  pdfs/    # extract text from all PDFs
python run.py score  pdf_output/  # score all extracted JSONs
# or both in one step:
python run.py all   pdfs/
```

## Repo Structure
```
llm_judge/
├── pdf_output/         # structured JSONs of relevant PDF sections
├── pdf_raw_output/     # Adobe Extract API output
├── pdfs/               # sourcePDFs
├── score_output/       # LLM Judge output
├── .env
├── config.py
├── extract_pdf.py
├── parse.py
├── prompts.py
├── README.md
├── requirements.txt
├── run.py
└── score.py
```

## Files
`extract_pdf.py` - PDF → PDF elements JSON. The Adobe Extract API extract text and PDF element structure. *Elements*: Ordered list of semantic elements (like headings, paragraphs, tables, figures) found in the document, on the basis of position in the structure tree of the document. *Path*: The Path describes the location of elements in the structure tree including the element type and the instance number. 

`parse.py` - PDF elements JSON → structured JSON. Consumes the Adobe Extract API output, identifies relevant sections (e.g., abstract, data_description, methods, limitations), and maps into the four logical buckets (abstract, data_description, methods, limitations).

`prompts.py` - All rubric logic lives here: four dimension definitions, scoring anchors, guiding questions, and the prompt builder that selects the most relevant bucket text per dimension.

`score.py` - Calls Gemini 1.5 Flash twice per dimension (temp 0 + temp 0.3), flags divergent scores, handles rate limiting with retry/backoff.

`run.py` - Entry point with three modes.

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
1. Extract all text and elements from PDF
2.  map raw sections into the four logical buckets (abstract, data_description, methods, limitations) using a fixed mapping table
3. Any text before the first detected heading goes into a preamble field (often contains the actual abstract even without a heading)
4. Sections that don't map to any bucket are kept in a misc field so nothing is discarded

## Fuzzy Matching
To be able to detect relevant headings to extract sections, we use a weighted fuzzy scoring per bucket. - For each detected heading, we compute a similarity score against every keyword in each bucket's anchor list, take the best match per bucket and assign to whichever bucket wins. A configurable threshold can be lowered if if real sections are being lost, allowing us to control whether headings that don't resemble any anchor well enough will fall through to misc rather than being forced into the closest bucket.

The **similarity metric** will be a combination of:
- Jaccard token overlap (40%) - content words only (stopwords stripped), so "Study Design and Analytical Approach" matching "analytical approach" isn't diluted by "and/the/of"
- Substring containment (45%) - if the anchor appears inside the heading or vice versa, score by the proportional length coverage. This is the strongest signal, which is why exact anchors score 1.0
- Prefix bonus (15%) - small nudge when one starts with the other
