"""
Smart Resume — Template Indexer
Run: python tools/index_templates.py
"""
import json, os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "resume_templates")
INDEX_PATH   = os.path.join(TEMPLATE_DIR, "index.json")

def extract_ats_patterns(text):
    lines = text.splitlines()
    section_order = []
    for line in lines:
        ll = line.strip().lower()
        for kw in ["summary","skills","experience","projects","education","certifications"]:
            if kw in ll and len(ll) < 30:
                section_order.append(kw); break
    bullet_lines = [l.strip() for l in lines if l.strip().startswith(("•","–","-","*"))]
    action_verbs = re.findall(r"^[•\-\*–]\s+([A-Z][a-z]+)", "\n".join(bullet_lines), re.MULTILINE)
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.]{2,}", text)
    stop  = {"the","and","for","with","you","are","have","will","this","that","from","your"}
    freq  = {}
    for w in words:
        wl = w.lower()
        if wl not in stop: freq[wl] = freq.get(wl,0)+1
    return {
        "section_order": list(dict.fromkeys(section_order)),
        "bullet_count": len(bullet_lines),
        "action_verbs": list(set(action_verbs))[:10],
        "top_keywords": sorted(freq, key=lambda k: freq[k], reverse=True)[:20],
        "avg_bullet_length": int(sum(len(l) for l in bullet_lines)/max(len(bullet_lines),1)),
    }

def detect_domain(fname, text):
    fname = fname.lower()
    for domain, kws in {
        "ml":["ml","machine_learning","ai"],"backend":["backend","api","server"],
        "frontend":["frontend","react","ui"],"devops":["devops","cloud","docker"],
        "data":["data_engineer","etl"],"embedded":["embedded","firmware","ros"],
    }.items():
        if any(k in fname for k in kws): return domain
    return "general"

def index_templates():
    files = [f for f in os.listdir(TEMPLATE_DIR) if f.lower().endswith((".pdf",".txt",".docx"))]
    if not files:
        print(f"No templates found in {TEMPLATE_DIR}")
        print("Add PDF/TXT/DOCX reference resumes with names like: ml_engineer_1.pdf")
        return
    index = []
    for fname in files:
        fpath = os.path.join(TEMPLATE_DIR, fname)
        try:
            if fname.endswith(".txt"):
                text = open(fpath, encoding="utf-8", errors="ignore").read()
            elif fname.endswith(".pdf"):
                from core.parser import extract_pdf_text
                text = extract_pdf_text(open(fpath,"rb").read())
            elif fname.endswith(".docx"):
                from core.parser import extract_docx_text
                text = extract_docx_text(open(fpath,"rb").read())
            domain = detect_domain(fname, text)
            patterns = extract_ats_patterns(text)
            index.append({"filename":fname,"domain":domain,"patterns":patterns,"preview":text[:500]})
            print(f"✓ {fname} → domain={domain}, {patterns['bullet_count']} bullets")
        except Exception as e:
            print(f"✗ {fname}: {e}")
    json.dump(index, open(INDEX_PATH,"w"), indent=2)
    print(f"\n✅ Indexed {len(index)} templates → {INDEX_PATH}")

if __name__ == "__main__": index_templates()