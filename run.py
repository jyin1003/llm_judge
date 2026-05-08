"""
run.py
------
Entry point for the LLM judge pipeline.

Modes:
    python run.py parse  pdfs/               ← parse all PDFs in folder
    python run.py parse  pdfs/paper.pdf      ← parse a single PDF
    python run.py score  pdf_output/         ← score all parsed JSONs
    python run.py score  pdf_output/x.json   ← score a single parsed JSON
    python run.py all    pdfs/               ← parse then score in one step
"""

import sys
from pathlib import Path

from parse import process_path
from score import main as score_main


def usage():
    print(__doc__)
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        usage()

    mode = sys.argv[1].lower()
    target = Path(sys.argv[2])

    base = Path(__file__).parent
    pdf_output = base / "pdf_output"
    score_output = base / "score_output"
    pdf_output.mkdir(exist_ok=True)
    score_output.mkdir(exist_ok=True)

    if mode == "parse":
        process_path(target, pdf_output)

    elif mode == "score":
        sys.argv = [sys.argv[0], str(target)]
        score_main()

    elif mode == "all":
        print("=== Stage 1: Parsing PDFs ===\n")
        process_path(target, pdf_output)
        print("\n=== Stage 2: Scoring ===\n")
        sys.argv = [sys.argv[0], str(pdf_output)]
        score_main()

    else:
        usage()


if __name__ == "__main__":
    main()
