import json
import re

with open("papers/extracted_papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

def search_paper(paper_id, query_terms):
    matches = []
    pdata = papers.get(paper_id, {})
    for page in pdata.get("pages", []):
        text = page["text"]
        for term in query_terms:
            if re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE):
                matches.append((page["page"], text, term))
                break
    return matches

print("="*60)
print("VERIFYING Q1: Attention Optimizer settings")
matches_q1 = search_paper("1706.03762v7", ["warmup_steps", "Adam optimizer"])
for p, t, term in matches_q1:
    print(f"Page {p} (matched '{term}'):")
    for s in t.split(". "):
        if "adam" in s.lower() or "warmup" in s.lower() or "0.98" in s:
            print("  Exact sentence:", s.strip().encode("ascii", "ignore").decode("ascii"))

print("="*60)
print("VERIFYING Q2: Attention Base Specs")
matches_q2 = search_paper("1706.03762v7", ["Table 3", "d_model"])
for p, t, term in matches_q2:
    if p in [3, 9]:  # table 3 is on page 9
        print(f"Page {p} (matched '{term}'):")
        for line in t.split("\n"):
            if "base" in line or "d_model" in line or "512" in line:
                print("  Line:", line.strip().encode("ascii", "ignore").decode("ascii"))

print("="*60)
print("VERIFYING Q3: Moshi Mimi codec and latency")
matches_q3 = search_paper("2410.00037v1", ["12.5 Hz", "1.1 kbps", "160 ms", "200 ms"])
for p, t, term in matches_q3[:4]:
    print(f"Page {p} (matched '{term}'):")
    for s in t.split(". "):
        if any(k in s for k in ["12.5", "1.1", "160", "200"]):
            print("  Exact sentence:", s.strip().encode("ascii", "ignore").decode("ascii"))

print("="*60)
print("VERIFYING Q4: Dynamic Chunking boundary mechanism")
matches_q4 = search_paper("2507.07955v2", ["routing", "smoothing", "boundary", "boundaries"])
print(f"Total matching pages in Dynamic Chunking: {len(matches_q4)}")
for p, t, term in matches_q4[:6]:
    print(f"Page {p} (matched '{term}'):")
    for s in t.split(". "):
        if any(k in s.lower() for k in ["boundary", "boundaries", "routing", "smoothing"]):
            print("  Exact sentence:", s.strip()[:180].encode("ascii", "ignore").decode("ascii"))

print("="*60)
print("VERIFYING UNANSWERABLE QUESTIONS:")
print("Q5: ImageNet in all papers:")
for pid in papers:
    res = search_paper(pid, ["ImageNet"])
    print(f"  Paper {pid}: {len(res)} matches")

print("Q6: Czech in all papers:")
for pid in papers:
    res = search_paper(pid, ["Czech"])
    print(f"  Paper {pid}: {len(res)} matches")

print("Q7: Hindi in Moshi:")
res_hindi = search_paper("2410.00037v1", ["Hindi"])
print(f"  Paper 2410.00037v1: {len(res_hindi)} matches")
