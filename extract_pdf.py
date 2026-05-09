""" 
extract_pdf.py
--------
Extract API, which uses a trained layout model to identify headings (H1-H6),
paragraphs, lists, and other structural elements with high accuracy.

Usage:
    python extract_pdf.py path/to/paper.pdf
    python extract_pdf.py pdfs/          ← batch: processes every PDF in the folder

Output:
    pdf_raw_output/<stem>.json  for each PDF

Adobe returns a zip containing structuredData.json. Each element has:
    Path  - e.g. "//Document/Sect/H1", "//Document/Sect/P"
    Text  - the text content of that element
"""
import io, json, logging, os, re, sys, zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Suppress Adobe SDK's verbose logging
logging.getLogger("adobe.pdfservices").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Lazy Adobe SDK import
# Imported here so the error message is clear if the SDK isn't installed.

def _get_adobe_sdk():
    try:
        from adobe.pdfservices.operation.auth.service_principal_credentials import (
            ServicePrincipalCredentials,
        )
        from adobe.pdfservices.operation.exception.exceptions import (
            ServiceApiException,
            ServiceUsageException,
            SdkException,
        )
        from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
        from adobe.pdfservices.operation.io.stream_asset import StreamAsset
        from adobe.pdfservices.operation.pdf_services import PDFServices
        from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
        from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
        from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import (
            ExtractElementType,
        )
        from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import (
            ExtractPDFParams,
        )
        from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import (
            ExtractPDFResult,
        )
        return {
            "ServicePrincipalCredentials": ServicePrincipalCredentials,
            "ServiceApiException": ServiceApiException,
            "ServiceUsageException": ServiceUsageException,
            "SdkException": SdkException,
            "PDFServices": PDFServices,
            "PDFServicesMediaType": PDFServicesMediaType,
            "ExtractPDFJob": ExtractPDFJob,
            "ExtractElementType": ExtractElementType,
            "ExtractPDFParams": ExtractPDFParams,
            "ExtractPDFResult": ExtractPDFResult,
        }
    except ImportError:
        logging.exception(
            "\nAdobe PDF Services SDK not found.\n"
            "Install it with:  pip install pdfservices-sdk\n"
        )
        sys.exit(1)

# Output naming
def output_name(pdf_path: Path) -> str:
    """Derive a safe output filename from the first 5 alphanumeric words of the stem."""
    words = re.findall(r"[a-zA-Z0-9]+", pdf_path.stem.lower())
    return "_".join(words[:5])

# Adobe API call
def extract_structure_via_adobe(pdf_path: Path) -> list:
    """
    Upload PDF to Adobe Extract API and return the list of elements from
    structuredData.json. Each element is a dict with at minimum:
        Path (str), Text (str, may be absent for non-text elements)
    """
    client_id = os.getenv("PDF_SERVICES_CLIENT_ID")
    client_secret = os.getenv("PDF_SERVICES_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "PDF_SERVICES_CLIENT_ID and PDF_SERVICES_CLIENT_SECRET must be set in .env\n"
            "Get free credentials at: "
            "https://acrobatservices.adobe.com/dc-integration-creation-app-cdn/main.html"
        )

    sdk = _get_adobe_sdk()
    
    credentials = sdk["ServicePrincipalCredentials"](
        client_id=client_id,
        client_secret=client_secret,
    )
    pdf_services = sdk["PDFServices"](credentials=credentials)

    # Upload the PDF
    with open(pdf_path, "rb") as f:
        input_asset = pdf_services.upload(
            input_stream=f,
            mime_type=sdk["PDFServicesMediaType"].PDF.value,
        )

    # Configure extraction: text elements only (no table renditions needed)
    params = sdk["ExtractPDFParams"](
        elements_to_extract=[sdk["ExtractElementType"].TEXT],
    )
    
    logging.info("Submitting Adobe Extract PDF job for: %s", pdf_path.name)
    job = sdk["ExtractPDFJob"](input_asset=input_asset, extract_pdf_params=params)
    location = pdf_services.submit(job)
    logging.info("Adobe Extract PDF job submitted. Location: %s", location)

    response = pdf_services.get_job_result(location, sdk["ExtractPDFResult"])
    logging.debug(f"Response: {response}")
    logging.info("Adobe Extract PDF job completed successfully")
    
    result_asset = response.get_result().get_resource()
    stream_asset = pdf_services.get_content(result_asset)

    # Parse the zip in memory
    raw_zip_bytes = stream_asset.get_input_stream()

    # Save raw output locally
    raw_output_dir = Path("pdf_raw_output")
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    name = output_name(pdf_path)
    
    # Save raw ZIP
    zip_path = raw_output_dir / f"{name}_raw_output.zip"
    with open(zip_path, "wb") as f:
        f.write(raw_zip_bytes)
        
    logging.info(f"Saved ZIP to: {zip_path}")

    # Extract structuredData.json
    with zipfile.ZipFile(io.BytesIO(raw_zip_bytes)) as zf:
        with zf.open("structuredData.json") as jf:
            data = json.load(jf)

    json_path = raw_output_dir / f"{name}_structuredData.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved readable JSON to: {json_path}")

# Main pipeline
def process_path(target: Path):
    """Process a single PDF or all PDFs in a directory."""
    if target.is_dir():
        pdfs = sorted(target.glob("*.pdf"))
        if not pdfs:
            logging.error(f"No PDFs found in {target}")
            return
        for pdf in pdfs:
            process_single(pdf)
    elif target.suffix.lower() == ".pdf":
        process_single(target)
    else:
        logging.error(f"Not a PDF or directory: {target}")
        sys.exit(1)


def process_single(pdf_path: Path):
    # Checkpoint: skip if output already exists
    raw_output_dir = Path("pdf_raw_output")
    name = output_name(pdf_path)
    json_path = raw_output_dir / f"{name}_structuredData.json"
    if json_path.exists():
        logging.info("Skipping already processed PDF: %s", pdf_path.name)
        return

    # Process the file
    logging.info(f"Parsing: {pdf_path.name} ...")
    try:
        extract_structure_via_adobe(pdf_path)
    except Exception:
        logging.exception("Failed to extract PDF structure for: %s", pdf_path)
        return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf.py <pdf_file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])

    process_path(target)