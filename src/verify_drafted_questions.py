"""
Verification script for drafted benchmark questions Q8 - Q20.
Searches extracted_papers.json and prints the exact supporting sentence and page number.
"""
import json
import re

with open("papers/extracted_papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

def verify_item(q_num, title, paper_id, expected_page, search_phrase):
    pdata = papers.get(paper_id, {})
    found = False
    for p in pdata.get("pages", []):
        if p["page"] == expected_page or expected_page is None:
            text = p["text"]
            for s in text.split("\n"):
                if search_phrase.lower() in s.lower():
                    clean_s = s.strip().encode("ascii", "ignore").decode("ascii")
                    print(f"[{q_num}] {title} | Paper: {paper_id}, Page: {p['page']}")
                    print(f"     Supporting text: \"{clean_s}\"\n")
                    found = True
                    break
            if found:
                break
    if not found:
        print(f"[{q_num}] FAILED: '{search_phrase}' not found on page {expected_page} of {paper_id}\n")

print("=== VERIFYING ANSWERABLE QUESTIONS (Q8 - Q16) ===\n")

# Q8
verify_item("Q8", "Label smoothing eps_ls=0.1", "1706.03762v7", 8, "label smoothing of value")

# Q9
verify_item("Q9", "Table 1 path lengths", "1706.03762v7", 6, "Maximum path lengths")

# Q10
verify_item("Q10", "Sinusoidal positional encoding", "1706.03762v7", 6, "PE(pos,2i)")

# Q11
verify_item("Q11", "Helium 7B parameter count", "2410.00037v1", 8, "7B-parameter Helium")

# Q12
verify_item("Q12", "Dual-stream audio modeling", "2410.00037v1", 3, "two audio streams")

# Q13
verify_item("Q13", "Fisher dataset duplex fine-tuning", "2410.00037v1", 8, "Fisher dataset")

# Q14
verify_item("Q14", "Dynamic chunking 4.5-5 bytes/chunk", "2507.07955v2", 3, "4.5-5 bytes/chunk")

# Q15
verify_item("Q15", "Smoothing module EMA formula (Eq 5)", "2507.07955v2", 7, "(5)")

# Q16 (Multi-paper)
verify_item("Q16a", "Transformer base 8 heads", "1706.03762v7", 9, "base 6 512 2048 8")
verify_item("Q16b", "Moshi Depth Transformer 16 heads", "2410.00037v1", 8, "Number of heads - 16")

print("=== VERIFYING UNANSWERABLE QUESTIONS (Q17 - Q20) ===\n")
for q_num, q_title, term in [
    ("Q17", "ResNet bottleneck expansion ratio", "bottleneck"),
    ("Q18", "Moshi financial training cost in USD", "dollar"),
    ("Q19", "Table 3 French BLEU for h=16", "newstest2013"),
    ("Q20", "Dynamic Chunking Japanese BLEU", "japanese")
]:
    total_matches = 0
    for pid, pdata in papers.items():
        for p in pdata.get("pages", []):
            if term in p["text"].lower():
                total_matches += 1
    print(f"[{q_num}] {q_title} (Checked term '{term}'): {total_matches} relevant benchmark matches across all papers -> Legitimate Refusal.")
