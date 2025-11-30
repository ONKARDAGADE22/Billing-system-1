This project implements an AI‑powered bill extraction service for the HackRx Datathon.
The API receives a bill image URL and returns structured line‑item data with strong checks so that the AI‑extracted total closely matches the actual bill total.

Problem Statement
Given a URL pointing to a medical bill / invoice image, extract:

All line items from the main billing table

For each item: name, quantity, rate, and final amount

The total number of items across all pages

The solution should minimize:

Missing line items

Double counting

Mismatch between the sum of extracted item amounts and the final total printed on the bill.

The evaluation is based on the accuracy of line‑item extraction and how close the extracted totals are to the ground‑truth bill totals.

Tech Stack
Language: Python 3

Backend Framework: FastAPI + Uvicorn

Cloud / Hosting (example): Railway free tier or ngrok tunnel from Google Colab

AI Model: Google Gemini (via google-generativeai)

Image Processing: OpenCV (opencv-python-headless), NumPy

HTTP & Data Models: requests, Pydantic, typing

No database is used; all processing is done in memory per request.

API Specification
Endpoint
POST /extract-bill-data

Request
json
{
  "document": "https://.../sample_3.png"
}
document: Public URL of the bill/invoice image that can be downloaded directly.

Successful Response (Schema)
json
{
  "is_success": true,
  "token_usage": {
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0
  },
  "data": {
    "pagewise_line_items": [
      {
        "page_no": "1",
        "page_type": "Bill Detail | Final Bill | Pharmacy",
        "bill_items": [
          {
            "item_name": "string",
            "item_amount": 0.0,
            "item_rate": 0.0,
            "item_quantity": 0.0
          }
        ]
      }
    ],
    "total_item_count": 0
  }
}
In the internal implementation, data also contains:

reconciled_amount: Sum of all item_amount values

printed_bill_total: Total printed on the invoice, as read by the model

fraud_warnings: List of math/consistency flags

preprocessing_applied: List of preprocessing steps applied to the image

These extra fields are compatible with the required schema and give more transparency.

Approach
1. Input and Download
The API receives a JSON body with "document" as a URL.

The server downloads the image bytes using requests.

2. Image Preprocessing
To make OCR and vision more robust, the image goes through a preprocessing step (conceptually):

Convert to grayscale

Denoise and apply basic cleaning

Apply adaptive thresholding to enhance text/table contrast

The names of the applied techniques are stored in preprocessing_applied for debugging.

3. LLM‑Based Extraction (Gemini)
The core extraction logic uses Google Gemini:

The raw/preprocessed image is sent to a Gemini model (gemini‑flash‑latest, with gemini‑1.5‑flash as a fallback).

A structured prompt instructs the model to:

Extract all rows from the main item table

Ignore subtotal / tax / “category total” rows as separate items

Return strict JSON with:

invoice_total

pagewise_line_items containing bill_items

Optional fraud_flags

The response MIME type is forced to application/json so that the output can be parsed safely.

4. Post‑processing and Cleaning
The raw model output is normalized into Pydantic models:

For each item, numeric fields are cleaned by stripping commas, currency symbols, and spaces before casting to float.

A running total calculated_total is computed by summing item_amount over all pages.

total_item_count is computed as the number of extracted items across all pages.

This step tries to be robust to minor OCR formatting issues (e.g., “1,000.00” vs “1000.00”).

5. Mathematical Reconciliation
To avoid missing or double‑counted items:

The printed total is parsed from invoice_total returned by the model.

The absolute difference between calculated_total and printed_total is computed.

If the difference is greater than a small threshold (e.g., > 1.0) and printed_total > 0, a message is added to fraud_warnings such as:

“Mathematical Check Failed: AI Extracted Sum (...) != Bill Total (...)”

This acts as an automated quality/fraud flag and encourages the model + pipeline to keep the totals aligned.

6. Error Handling
On success, the API returns is_success = true and a fully populated data section.

Any unexpected exception (network error, model failure, parsing issue) results in is_success = false and an error message instead of crashing, so the caller can see what went wrong.

Running Locally
Prerequisites
Python 3.10+

A valid Google Gemini API key

Setup
bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/hackrx-bill-extraction.git
cd hackrx-bill-extraction

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
export GOOGLE_API_KEY="YOUR_GEMINI_KEY"   # Windows: set GOOGLE_API_KEY=...
uvicorn main:app --host 0.0.0.0 --port 8000
The API will be available at:

http://127.0.0.1:8000/extract-bill-data

Interactive docs: http://127.0.0.1:8000/docs

Example Request (curl)
bash
curl -X POST "http://127.0.0.1:8000/extract-bill-data" \
  -H "Content-Type: application/json" \
  -d '{
    "document": "https://hackrx.blob.core.windows.net/assets/datathon-IIT/sample_3.png?... "
  }'
Deployment (Example: Railway)
Push main.py and requirements.txt to a public GitHub repo.

Create a Railway project, connect the repo, and set:

Environment variable: GOOGLE_API_KEY

Start command: uvicorn main:app --host 0.0.0.0 --port $PORT

After deploy, Railway gives a URL like https://your-project.up.railway.app.

Final webhook for HackRx:

https://your-project.up.railway.app/extract-bill-data

Limitations and Future Work
Currently relies on a single LLM call per document; batch handling and caching could reduce latency and token cost.

No persistent storage is used; adding a database would allow auditing past bills and learning from manual corrections.

The reconciliation logic is simple; more advanced heuristics (e.g., matching rate × quantity vs amount, multi‑page bills, tax breakdown) can further improve accuracy.

You can shorten or delete sections depending on the word limit on the portal, but this version is safe to drop directly into README.md.
