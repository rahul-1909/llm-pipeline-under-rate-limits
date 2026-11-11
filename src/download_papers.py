import os
import requests
from pypdf import PdfReader
import json

PAPERS = [
    {
        "id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "filename": "1706.03762v7.pdf"
    },
    {
        "id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "filename": "2410.00037v1.pdf"
    },
    {
        "id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "filename": "2507.07955v2.pdf"
    }
]

PAPERS_DIR = "papers"

def download_papers():
    os.makedirs(PAPERS_DIR, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResearchPaperFetcher/1.0"}
    
    for paper in PAPERS:
        filepath = os.path.join(PAPERS_DIR, paper["filename"])
        url = f"https://arxiv.org/pdf/{paper['id']}.pdf"
        print(f"Downloading {paper['title']} ({paper['id']}) from {url}...")
        
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                print(f"Successfully downloaded {paper['filename']} ({len(resp.content)} bytes)")
            else:
                print(f"Failed {paper['id']}: status {resp.status_code}, len={len(resp.content)}")
                # Check alternative if 2507 was typo or if 2407 exists
                if "2507" in paper["id"]:
                    print("Testing alternate 2407.07955v2 if 2507 doesn't exist...")
                    alt_url = "https://arxiv.org/pdf/2407.07955v2.pdf"
                    alt_resp = requests.get(alt_url, headers=headers, timeout=30)
                    print(f"Alt url status: {alt_resp.status_code}, len={len(alt_resp.content)}")
        except Exception as e:
            print(f"Error downloading {paper['id']}: {e}")

def extract_text():
    extracted = {}
    for paper in PAPERS:
        filepath = os.path.join(PAPERS_DIR, paper["filename"])
        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist, skipping text extraction.")
            continue
        reader = PdfReader(filepath)
        pages_data = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_data.append({
                "page": idx + 1,
                "text": text
            })
        extracted[paper["id"]] = {
            "title": paper["title"],
            "num_pages": len(pages_data),
            "pages": pages_data
        }
        print(f"Extracted {len(pages_data)} pages from {paper['filename']}")

    out_file = os.path.join(PAPERS_DIR, "extracted_papers.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)
    print(f"Saved extracted text to {out_file}")

if __name__ == "__main__":
    download_papers()
    extract_text()
