# LLM Judge

## Usage
```bash
pip install -r requirements.txt
python run.py parse  pdfs/    # extract text from all PDFs
python run.py score  pdf_output/  # score all extracted JSONs
# or both in one step:
python run.py all   pdfs/
```

## Flow
```
run.py
  → ingest.py reads PDF, extracts sections into dict
  → prompts.py builds 4 dimension-specific prompts
  → score.py calls Gemini twice per dimension (temp 0 + 0.3)
  → flags divergent scores
  → writes output JSON
```