"""
Deterministic Verifier for citations and factual grounding.
Enforces:
1. Valid paper ID and version.
2. Valid page number within document page boundaries.
3. Cited page was present in retrieved passages.
4. Lexical overlap between answer claims and retrieved passage text.
"""
import re
from typing import Dict, List, Any

VALID_PAPERS = {
    "1706.03762v7": {"max_page": 15, "short_name": "Attention"},
    "2410.00037v1": {"max_page": 66, "short_name": "Moshi"},
    "2507.07955v2": {"max_page": 39, "short_name": "Dynamic Chunking"}
}

# Regex to capture citations like [Paper 1706.03762v7, Page 5] or [1706.03762v7, p. 3] or [arXiv 2410.00037v1, page 12]
CITATION_PATTERN = re.compile(
    r"(?:\[|\()(?:Paper|arXiv\s*:?)?\s*(1706\.03762v?7?|2410\.00037v?1?|2507\.07955v?2?|Attention|Moshi|Dynamic\s+Chunking)[^,;\]\)]*[,;]?\s*(?:page|p\.?|sec\.?|section)?\s*(\d+)?(?:[\]\)])",
    re.IGNORECASE
)

REFUSAL_KEYWORDS = [
    "do not contain",
    "does not contain",
    "cannot be answered",
    "not mentioned in",
    "not covered in",
    "the library does not",
    "none of the provided papers",
    "neither paper",
    "no information"
]

def is_refusal(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in REFUSAL_KEYWORDS)

def extract_citations(text: str) -> List[Dict[str, Any]]:
    citations = []
    for match in CITATION_PATTERN.finditer(text):
        raw_paper = match.group(1).strip()
        raw_page = match.group(2)
        
        # Normalize paper id
        paper_id = None
        if "1706" in raw_paper or "attention" in raw_paper.lower():
            paper_id = "1706.03762v7"
        elif "2410" in raw_paper or "moshi" in raw_paper.lower():
            paper_id = "2410.00037v1"
        elif "2507" in raw_paper or "dynamic" in raw_paper.lower():
            paper_id = "2507.07955v2"

        page_num = int(raw_page) if raw_page and raw_page.isdigit() else None
        citations.append({
            "raw": match.group(0),
            "paper_id": paper_id,
            "page": page_num
        })
    return citations

def verify_answer(answer: str, retrieved_passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns:
      valid: bool
      reason: str
      citations: list
      is_refusal: bool
    """
    # 1. Check if the model explicitly stated refusal
    if is_refusal(answer):
        return {
            "valid": True,
            "reason": "Legitimate refusal: paper does not contain required facts.",
            "citations": [],
            "is_refusal": True
        }

    # 2. Extract citations
    citations = extract_citations(answer)
    if not citations:
        return {
            "valid": False,
            "reason": "Missing citation: every factual claim must cite paper ID and page number.",
            "citations": [],
            "is_refusal": False
        }

    retrieved_pages = {(p["paper_id"], p["page"]) for p in retrieved_passages}
    all_retrieved_text = " ".join([p["text"].lower() for p in retrieved_passages])

    for cit in citations:
        p_id = cit["paper_id"]
        p_num = cit["page"]

        if not p_id or p_id not in VALID_PAPERS:
            return {
                "valid": False,
                "reason": f"Invalid paper citation '{cit['raw']}': must be one of {list(VALID_PAPERS.keys())}",
                "citations": citations,
                "is_refusal": False
            }

        max_page = VALID_PAPERS[p_id]["max_page"]
        if p_num is None:
            return {
                "valid": False,
                "reason": f"Citation '{cit['raw']}' missing specific page number.",
                "citations": citations,
                "is_refusal": False
            }

        if p_num < 1 or p_num > max_page:
            return {
                "valid": False,
                "reason": f"Page out of bounds in '{cit['raw']}': Paper {p_id} only has {max_page} pages.",
                "citations": citations,
                "is_refusal": False
            }

        # Check if the cited page was actually retrieved
        if (p_id, p_num) not in retrieved_pages:
            return {
                "valid": False,
                "reason": f"Hallucinated citation: cited {p_id} page {p_num}, which was not in retrieved context.",
                "citations": citations,
                "is_refusal": False
            }

    # 3. Grounding check: ensure answer has high lexical overlap with retrieved context
    ans_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower()))
    context_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", all_retrieved_text))
    if ans_words:
        overlap = len(ans_words.intersection(context_words)) / len(ans_words)
        if overlap < 0.35:
            return {
                "valid": False,
                "reason": f"Low factual grounding: only {overlap*100:.1f}% lexical overlap with retrieved context.",
                "citations": citations,
                "is_refusal": False
            }

    return {
        "valid": True,
        "reason": "Verification passed: valid citations, page boundaries, and context grounding.",
        "citations": citations,
        "is_refusal": False
    }
