# LLM Judge

## Usage
```bash
pip install -r requirements.txt
python run.py parse  pdfs/    # extract text from all PDFs
python run.py score  pdf_output/  # score all extracted JSONs
# or both in one step:
python run.py all   pdfs/
```