"""
score.py
--------
Sends prompts to the Gemini API and returns structured scores.

Each dimension is scored twice:
  - primary run:   temperature 0   (maximise determinism)
  - secondary run: temperature 0.3 (controlled variation)

Scores that diverge between runs are flagged for manual review.

Usage (standalone, for testing):
    python score.py pdf_output/some_paper.json
"""

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from prompts import build_prompts, all_dimension_keys, DIMENSIONS

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash"  # free tier model
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={{api_key}}"
)

REQUEST_TIMEOUT = 60  # seconds
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 5  # seconds between retries


# ── API call ──────────────────────────────────────────────────────────────────

def call_gemini(system_prompt: str, user_prompt: str, temperature: float) -> dict:
    """
    Call Gemini API with system + user prompts.
    Returns the parsed JSON response dict from the model, or raises on failure.

    Gemini uses a slightly different message format to OpenAI:
      - system instruction goes in systemInstruction
      - user content goes in contents[0]
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to judge/.env as: GEMINI_API_KEY=your_key_here"
        )

    url = GEMINI_URL.format(api_key=GEMINI_API_KEY)

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",  # ask Gemini to return JSON directly
        },
    }

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # Extract text from Gemini response structure
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError(f"No candidates in response: {data}")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise ValueError(f"No parts in candidate content: {content}")

            raw_text = parts[0].get("text", "").strip()

            # Strip markdown fences if present (belt and braces)
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)
                raw_text = raw_text.strip()

            return json.loads(raw_text)

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            if status == 429 and attempt < RETRY_ATTEMPTS:
                # Rate limited — back off and retry
                wait = RETRY_BACKOFF * attempt
                print(f"    Rate limited (429). Waiting {wait}s before retry {attempt+1}/{RETRY_ATTEMPTS}...")
                time.sleep(wait)
                continue
            raise
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF)
                continue
            raise RuntimeError(f"Failed to parse model response after {RETRY_ATTEMPTS} attempts: {e}")

    raise RuntimeError(f"All {RETRY_ATTEMPTS} attempts failed.")


# ── Score one dimension ───────────────────────────────────────────────────────

def score_dimension(dimension_key: str, parsed: dict) -> dict:
    """
    Score a single dimension with two temperature runs.
    Returns a dict with primary, secondary, divergent flag, and merged result.
    """
    system_prompt, user_prompt = build_prompts(dimension_key, parsed)

    # Primary run: temperature 0
    try:
        primary = call_gemini(system_prompt, user_prompt, temperature=0.0)
        primary["run"] = "primary (temp=0)"
    except Exception as e:
        primary = {"score": None, "evidence": None, "confidence": "low",
                   "reasoning": f"API error: {e}", "run": "primary (temp=0)", "error": True}

    # Small delay between calls to avoid rate limiting
    time.sleep(1.5)

    # Secondary run: temperature 0.3
    try:
        secondary = call_gemini(system_prompt, user_prompt, temperature=0.3)
        secondary["run"] = "secondary (temp=0.3)"
    except Exception as e:
        secondary = {"score": None, "evidence": None, "confidence": "low",
                     "reasoning": f"API error: {e}", "run": "secondary (temp=0.3)", "error": True}

    # Divergence check
    p_score = primary.get("score")
    s_score = secondary.get("score")
    divergent = (p_score is not None and s_score is not None and p_score != s_score)

    return {
        "dimension": dimension_key,
        "label": DIMENSIONS[dimension_key]["label"],
        "primary": primary,
        "secondary": secondary,
        "divergent": divergent,
        # Use primary score as the canonical score; flag if divergent
        "score": p_score,
        "needs_review": divergent or primary.get("confidence") == "low",
    }


# ── Score a full parsed PDF ───────────────────────────────────────────────────

def score_paper(parsed: dict, verbose: bool = True) -> dict:
    """
    Run all four dimensions for a parsed PDF dict.
    Returns the full scoring result dict.
    """
    source = parsed.get("source_file", "unknown")
    results = {}
    total_score = 0
    review_flags = []

    for dim_key in all_dimension_keys():
        if verbose:
            print(f"    Scoring [{dim_key}] ...", end=" ", flush=True)

        result = score_dimension(dim_key, parsed)
        results[dim_key] = result

        score = result.get("score")
        if score is not None:
            total_score += score

        if result.get("needs_review"):
            review_flags.append(dim_key)

        if verbose:
            score_str = str(score) if score is not None else "ERR"
            diverge_str = " ⚠ DIVERGENT" if result["divergent"] else ""
            review_str = " 🔍 REVIEW" if result.get("needs_review") and not result["divergent"] else ""
            print(f"score={score_str}/2{diverge_str}{review_str}")

        # Delay between dimensions to stay within free tier rate limits
        time.sleep(2)

    output = {
        "source_file": source,
        "total_score": total_score,
        "max_score": len(all_dimension_keys()) * 2,
        "dimensions": results,
        "review_flags": review_flags,
        "parse_warnings": parsed.get("parse_warnings", []),
    }

    return output


# ── Entry point ───────────────────────────────────────────────────────────────

import re  # imported here to avoid top-level order issue with retry logic above

def main():
    if len(sys.argv) < 2:
        print("Usage: python score.py <pdf_output_json_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    score_output_dir = Path(__file__).parent / "score_output"
    score_output_dir.mkdir(exist_ok=True)

    if target.is_dir():
        json_files = sorted(target.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {target}")
            sys.exit(1)
    elif target.suffix == ".json":
        json_files = [target]
    else:
        print(f"Expected a .json file or directory, got: {target}")
        sys.exit(1)

    for json_path in json_files:
        print(f"\nScoring: {json_path.name}")
        with open(json_path, encoding="utf-8") as f:
            parsed = json.load(f)

        result = score_paper(parsed, verbose=True)

        out_path = score_output_dir / (json_path.stem + "_scores.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        total = result["total_score"]
        max_s = result["max_score"]
        flags = result["review_flags"]
        print(f"  → Total: {total}/{max_s}  |  Saved: {out_path.name}")
        if flags:
            print(f"  ⚠ Manual review needed for: {', '.join(flags)}")


if __name__ == "__main__":
    main()
