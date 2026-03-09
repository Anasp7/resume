"""
Smart Resume — Unified UI (Phase 1 + Phase 2 merged)
======================================================
Single page flow:
  1. Upload resume + paste JD + declare proficiencies
  2. Click Analyze → runs /analyze then /evaluate automatically
  3. Full results displayed on same page — no session ID needed

Run with: streamlit run ui/app.py
"""

import json
import os
import httpx
import streamlit as st

API_BASE = os.getenv("SMART_RESUME_API_URL", "http://localhost:8001")

def _fetch_downloads(session_id: str) -> None:
    """Pre-fetch DOCX and PDF bytes into session state after any evaluation."""
    try:
        dr = httpx.get(f"{API_BASE}/evaluate/{session_id}/download/docx", timeout=30)
        st.session_state.docx_bytes = dr.content if dr.status_code == 200 else None
    except Exception:
        st.session_state.docx_bytes = None

    try:
        pr = httpx.get(f"{API_BASE}/evaluate/{session_id}/download/pdf", timeout=30)
        if pr.status_code == 200 and len(pr.content) > 100:
            st.session_state.pdf_bytes = pr.content
            st.session_state.pop("pdf_error", None)
        else:
            st.session_state.pdf_bytes = None
            st.session_state.pdf_error = pr.text[:200]
    except Exception as e:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_error = str(e)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Resume",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
* { font-family: 'Inter', sans-serif; }

.main-title {
  font-size: 2.6rem; font-weight: 800;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 0;
}
.subtitle { color: #6b7280; font-size: 1rem; margin-bottom: 1.5rem; }

.section-head {
  font-size: 1rem; font-weight: 700; color: #1f2937;
  border-left: 4px solid #667eea; padding-left: 0.6rem;
  margin: 1.4rem 0 0.6rem;
}
.card {
  background: #f9fafb; border: 1px solid #e5e7eb;
  border-radius: 12px; padding: 1rem 1.2rem; margin: 0.4rem 0;
}
.tag {
  display: inline-block; background: #ede9fe; color: #5b21b6;
  border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; margin: 2px;
}
.miss-tag {
  display: inline-block; background: #fee2e2; color: #b91c1c;
  border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; margin: 2px;
}
.fix-box {
  background: #ecfdf5; border: 1px solid #6ee7b7;
  border-radius: 8px; padding: 0.6rem 1rem; margin: 0.3rem 0; font-size: 0.9rem;
}
.warn-box {
  background: #fef3c7; border: 1px solid #f59e0b;
  border-radius: 8px; padding: 0.6rem 1rem; margin: 0.3rem 0; font-size: 0.9rem;
}
.verdict-strong   { background: #d1fae5; border: 1px solid #10b981; border-radius: 8px; padding: 0.6rem 1rem; }
.verdict-moderate { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 0.6rem 1rem; }
.verdict-weak     { background: #fee2e2; border: 1px solid #ef4444; border-radius: 8px; padding: 0.6rem 1rem; }
.step-badge {
  display: inline-block; background: #667eea; color: white;
  border-radius: 50%; width: 24px; height: 24px; text-align: center;
  line-height: 24px; font-size: 0.75rem; font-weight: 700; margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🧠 Smart Resume</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Resume Optimization & Evaluation Engine for CS Graduates</div>', unsafe_allow_html=True)

# Show backend status badges
_health = st.session_state.get("health", {})
if _health:
    sim_mode = _health.get("similarity_mode", "")
    groq_ok  = _health.get("groq_configured", False)
    _badge_parts = []
    if sim_mode:
        _badge_color = "#10b981" if "sentence" in sim_mode else "#f59e0b"
        _badge_parts.append(f'<span style="background:{_badge_color};color:white;border-radius:4px;padding:2px 8px;font-size:0.72rem;margin-right:6px">🔮 {sim_mode}</span>')
    if groq_ok:
        _badge_parts.append('<span style="background:#3b82f6;color:white;border-radius:4px;padding:2px 8px;font-size:0.72rem;margin-right:6px">⚡ Groq / Llama-3.3-70b</span>')
    if _badge_parts:
        st.markdown(" ".join(_badge_parts), unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "analysis"         not in st.session_state: st.session_state["analysis"]         = None
if "evaluation"       not in st.session_state: st.session_state["evaluation"]       = None
if "docx_bytes"       not in st.session_state: st.session_state["docx_bytes"]       = None
if "pdf_bytes"        not in st.session_state: st.session_state["pdf_bytes"]        = None
if "score_history"    not in st.session_state: st.session_state["score_history"]    = []
if "proficiency_rows" not in st.session_state:
    st.session_state["proficiency_rows"] = [{"skill": "", "level": "Intermediate"}]
if "health" not in st.session_state:
    try:
        _h = httpx.get(f"{API_BASE}/health", timeout=5)
        st.session_state["health"] = _h.json() if _h.status_code == 200 else {}
    except Exception:
        st.session_state["health"] = {}

# ── Layout ────────────────────────────────────────────────────────────────────

left, right = st.columns([1, 1.4], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — INPUTS
# ══════════════════════════════════════════════════════════════════════════════

with left:
    st.markdown('<div class="section-head"><span class="step-badge">1</span>Upload Resume</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Resume (PDF, DOCX, TXT)", type=["pdf","docx","txt"], label_visibility="collapsed")

    st.markdown('<div class="section-head"><span class="step-badge">2</span>Target Role</div>', unsafe_allow_html=True)
    target_role = st.text_input("Target Role", placeholder="e.g. ML Engineer, Backend Developer", label_visibility="collapsed")

    st.markdown('<div class="section-head"><span class="step-badge">3</span>Job Description <span style="color:#9ca3af;font-size:0.8rem">(optional)</span></div>', unsafe_allow_html=True)
    jd_text = st.text_area("Job Description", height=180, label_visibility="collapsed",
                           placeholder="Paste the job description here (optional). Without it, Smart Resume will use its knowledge of the target role.")

    st.markdown('<div class="section-head"><span class="step-badge">4</span>Skill Proficiencies</div>', unsafe_allow_html=True)
    st.caption("Declare your level for key skills so we can verify evidence.")

    LEVELS = ["Beginner", "Intermediate", "Advanced", "Expert"]
    for i, row in enumerate(st.session_state.proficiency_rows):
        c1, c2, c3 = st.columns([3, 2, 0.6])
        with c1:
            st.session_state.proficiency_rows[i]["skill"] = st.text_input(
                f"s{i}", value=row["skill"], key=f"sk_{i}",
                placeholder=f"Skill {i+1}", label_visibility="collapsed")
        with c2:
            st.session_state.proficiency_rows[i]["level"] = st.selectbox(
                f"l{i}", LEVELS, index=LEVELS.index(row["level"]),
                key=f"lv_{i}", label_visibility="collapsed")
        with c3:
            if st.button("✕", key=f"rm_{i}") and len(st.session_state.proficiency_rows) > 1:
                st.session_state.proficiency_rows.pop(i)
                st.rerun()

    if st.button("＋ Add Skill"):
        st.session_state.proficiency_rows.append({"skill": "", "level": "Intermediate"})
        st.rerun()

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze & Evaluate Resume", type="primary", use_container_width=True)

# ── Submission ────────────────────────────────────────────────────────────────

if analyze_btn:
    errors = []
    if not resume_file:           errors.append("Upload a resume file.")
    if not target_role.strip():   errors.append("Enter the target role.")
    if errors:
        for e in errors: st.error(e)
    else:
        prof_list = [
            {"skill_name": r["skill"].strip(), "level": r["level"]}
            for r in st.session_state.proficiency_rows if r["skill"].strip()
        ]

        with right:
            # ── Step 1: Analyze ───────────────────────────────────────────
            with st.spinner("📄 Parsing resume..."):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/analyze",
                        files={"resume_file": (resume_file.name, resume_file.getvalue(), resume_file.type)},
                        data={
                            "job_description": jd_text,
                            "target_role":     target_role,
                            "proficiency_json": json.dumps(prof_list),
                        },
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        st.session_state.analysis   = resp.json()
                        st.session_state.evaluation = None
                    else:
                        st.error(f"Parse error {resp.status_code}: {resp.text[:200]}")
                        st.stop()
                except httpx.ConnectError:
                    st.error("Cannot connect to backend. Run `python main.py` first.")
                    st.stop()

            # ── Step 2: Evaluate ──────────────────────────────────────────
            session_id = st.session_state.analysis.get("session_id")
            with st.spinner("🧠 Smart Resume is evaluating... this takes 15–30 sec"):
                try:
                    resp2 = httpx.post(f"{API_BASE}/evaluate/{session_id}", timeout=90)
                    if resp2.status_code == 200:
                        st.session_state.evaluation = resp2.json()
                        # Save to score history
                        ev_data = st.session_state.evaluation
                        w_score = ev_data.get("job_match_analysis", {}).get("weighted_score")
                        if w_score is not None:
                            entry = {
                                "role":  target_role,
                                "score": w_score,
                                "verdict": ev_data.get("job_match_analysis", {}).get("verdict", ""),
                            }
                            st.session_state["score_history"] = (
                                st.session_state.get("score_history", [])[-4:] + [entry]
                            )
                        # Pre-fetch only if resume already generated (no clarification needed)
                        _ev = st.session_state.evaluation or {}
                        if _ev.get("final_resume") and not _ev.get("clarification_required"):
                            sid = st.session_state.analysis.get("session_id","")
                            _fetch_downloads(sid)
                    elif resp2.status_code == 400:
                        st.warning("⚠️ GROQ_API_KEY not set — showing parse results only.")
                    elif resp2.status_code == 429 or "429" in resp2.text:
                        st.warning("⚠️ Groq API rate limit hit. Wait 30 seconds, then click Analyze again.")
                    elif resp2.status_code == 500 and "429" in resp2.text:
                        st.warning("⚠️ Groq API rate limit hit. Wait 30 seconds, then click Analyze again.")
                    else:
                        st.error(f"Evaluation error {resp2.status_code}: {resp2.text[:300]}")
                except Exception as e:
                    st.warning(f"Evaluation unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with right:
    analysis   = st.session_state.analysis
    evaluation = st.session_state.evaluation

    if not analysis:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;color:#9ca3af;">
          <div style="font-size:3.5rem">🧠</div>
          <div style="font-size:1.1rem;margin-top:1rem;font-weight:600">Smart Resume</div>
          <div style="font-size:0.95rem;margin-top:0.5rem">
            Fill in the form and click Analyze to get your full evaluation.
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        pr  = analysis.get("parsed_resume", {})
        sim = analysis.get("semantic_similarity_score", 0)

        # ── Parse summary strip ───────────────────────────────────────────
        st.markdown('<div class="section-head">📋 Resume Parsed</div>', unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Similarity",  f"{sim:.1f}/100")
        m2.metric("Template",    analysis.get("selected_template","—").replace("TemplateType.",""))
        m3.metric("Projects",    pr.get("project_count", 0))
        m4.metric("Experience",  pr.get("experience_count", 0))
        st.progress(int(sim))

        # ── ATS Score ─────────────────────────────────────────────────────────
        ats_score   = analysis.get("ats_score")
        ats_verdict = analysis.get("ats_verdict", "")
        ats_flags   = analysis.get("ats_flags", [])
        if ats_score is not None:
            ats_color = "#10b981" if ats_score >= 80 else "#f59e0b" if ats_score >= 60 else "#ef4444"
            ats_icon  = "✅" if ats_score >= 80 else "⚠️" if ats_score >= 60 else "❌"
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                        padding:1rem 1.2rem;margin:0.6rem 0;border-left:5px solid {ats_color}">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:700;font-size:0.95rem">🤖 ATS Compatibility</span>
                <span style="font-size:1.3rem;font-weight:800;color:{ats_color}">{ats_icon} {ats_score}/100</span>
              </div>
              <div style="font-size:0.85rem;color:{ats_color};margin-top:4px;font-weight:600">{ats_verdict}</div>
            </div>""", unsafe_allow_html=True)
            if ats_flags:
                with st.expander(f"🔍 ATS Issues ({len(ats_flags)} found)", expanded=ats_score < 70):
                    for flag in ats_flags:
                        st.markdown(f'<div style="padding:4px 0 4px 10px;border-left:3px solid {ats_color};'
                                    f'font-size:0.85rem;margin:3px 0">⚠️ {flag}</div>',
                                    unsafe_allow_html=True)

        # ── Skills ────────────────────────────────────────────────────────
        if pr.get("skills"):
            st.markdown('<div class="section-head">🛠️ Extracted Skills</div>', unsafe_allow_html=True)
            st.markdown(" ".join(f'<span class="tag">{s}</span>' for s in pr["skills"]), unsafe_allow_html=True)

        if not evaluation:
            if st.session_state.get("groq_ok", True):
                st.info("Click **Analyze** to start evaluation.")
            else:
                st.warning("⚠️ GROQ_API_KEY not detected. Check your .env file and restart the server.")
        else:
            ev = evaluation

            # ── Mismatch corrections ──────────────────────────────────────
            fixes = ev.get("mismatch_corrections", [])
            if fixes:
                st.markdown('<div class="section-head">🔀 Mismatch Corrections</div>', unsafe_allow_html=True)
                for f in fixes:
                    st.markdown(f'<div class="fix-box">✅ {f}</div>', unsafe_allow_html=True)

            # ── Job match ─────────────────────────────────────────────────
            jma     = ev.get("job_match_analysis", {})
            verdict_raw = jma.get("verdict", "")
            # Normalize verdict — LLM sometimes returns a sentence
            if "strong" in verdict_raw.lower() and "not" not in verdict_raw.lower():
                verdict = "Strong"
            elif "moderate" in verdict_raw.lower() or "partial" in verdict_raw.lower():
                verdict = "Moderate"
            else:
                verdict = "Weak"
            css = {"Strong":"verdict-strong","Moderate":"verdict-moderate"}.get(verdict,"verdict-weak")

            st.markdown('<div class="section-head">📊 Job Match</div>', unsafe_allow_html=True)

            # Weighted score breakdown
            w_score   = jma.get("weighted_score")
            breakdown = jma.get("score_breakdown", {})
            if w_score is not None:
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Overall Match", f"{w_score:.0f}/100")
                sc2.metric("Skill Overlap", f"{breakdown.get('skill_overlap', 0):.0f}%")
                sc3.metric("Similarity",    f"{breakdown.get('similarity', 0):.0f}%")
                sc4.metric("Experience",    f"{breakdown.get('experience_match', 0):.0f}%")
                st.markdown("<br>", unsafe_allow_html=True)

            j1, j2 = st.columns(2)
            j1.metric("Domain",  jma.get("domain","—"))
            j2.metric("Verdict", verdict)
            st.markdown(f'<div class="{css}">{jma.get("alignment_score_reasoning","")}</div>', unsafe_allow_html=True)

            mc1, mc2 = st.columns(2)
            with mc1:
                st.caption("✅ Matched Skills")
                matched = jma.get("matched_skills", [])
                st.markdown(" ".join(f'<span class="tag">{s}</span>' for s in matched) or "—", unsafe_allow_html=True)
            with mc2:
                st.caption("❌ Missing Skills")
                missing = jma.get("missing_skills", [])
                st.markdown(" ".join(f'<span class="miss-tag">{s}</span>' for s in missing) or "None ✅", unsafe_allow_html=True)

            # ── JD mode notice ────────────────────────────────────────────
            jma = ev.get("job_match_analysis", {})
            jd_provided = bool(analysis.get("job_description_raw_text","").strip())
            if not jd_provided:
                pjd_skills = analysis.get("parsed_jd", {}).get("required_skills", [])
                skills_str = ", ".join(pjd_skills[:8]) if pjd_skills else "role-standard skills"
                st.markdown(f"""
                <div style="background:#1e3a5f;border:1px solid #3b82f6;border-radius:10px;
                            padding:0.8rem 1.2rem;margin-bottom:0.8rem;font-size:0.9rem;">
                  🔍 <strong>No JD provided</strong> — Smart Resume generated a virtual job profile
                  for <strong>{analysis.get("target_role","this role")}</strong> using industry knowledge.<br>
                  <span style="color:#93c5fd">Evaluated against: {skills_str}</span>
                </div>""", unsafe_allow_html=True)

            # ── Skill Gap Analysis (tiered) ───────────────────────────────────
            sga = ev.get("skill_gap_analysis", {})
            if any(sga.get(t) for t in ["tier_1_critical","tier_2_important","tier_3_nice_to_have"]):
                st.markdown('<div class="section-head">🎯 Skill Gap Analysis</div>', unsafe_allow_html=True)
                tier_cfg = [
                    ("tier_1_critical",      "🔴 Critical — Will disqualify if missing",  "#fee2e2", "#b91c1c"),
                    ("tier_2_important",     "🟡 Important — Differentiates candidates",  "#fef3c7", "#92400e"),
                    ("tier_3_nice_to_have",  "🟢 Nice to Have — Gives an edge",           "#d1fae5", "#065f46"),
                ]
                for tier_key, label, bg, color in tier_cfg:
                    items = sga.get(tier_key, [])
                    if not items:
                        continue
                    st.markdown(f'<div style="font-weight:700;color:{color};margin-top:0.8rem">{label}</div>', unsafe_allow_html=True)
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        skill    = item.get("skill","")
                        evidence = item.get("current_evidence","None")
                        what_to  = item.get("what_to_do","")
                        severity = item.get("gap_severity","")
                        st.markdown(f"""
                        <div style="background:{bg};border-radius:8px;padding:0.6rem 1rem;margin:0.3rem 0;">
                          <strong>{skill}</strong> &nbsp;
                          <span style="font-size:0.78rem;color:#6b7280">Current evidence: {evidence}</span><br>
                          <span style="font-size:0.88rem">→ {what_to}</span>
                        </div>""", unsafe_allow_html=True)

            # ── Skill classification ──────────────────────────────────────
            sc = ev.get("skill_classification", {})
            all_classified = [s for cat in sc.values() if isinstance(cat, list) for s in cat]
            if all_classified:
                st.markdown('<div class="section-head">🗂️ Skill Classification</div>', unsafe_allow_html=True)
                sc1, sc2 = st.columns(2)
                with sc1:
                    for cat, label, color in [
                        ("programming_languages", "Languages",  "#3b82f6"),
                        ("frameworks_libraries",  "Frameworks", "#8b5cf6"),
                        ("databases",             "Databases",  "#f59e0b"),
                    ]:
                        items = sc.get(cat, [])
                        if items:
                            st.markdown(
                                f'<div style="margin-bottom:8px">'
                                f'<span style="font-size:0.78rem;font-weight:600;color:{color};text-transform:uppercase;letter-spacing:0.05em">'
                                f'{label} ({len(items)})</span><br>'
                                + " ".join(f'<span class="tag">{s}</span>' for s in items)
                                + '</div>', unsafe_allow_html=True)
                with sc2:
                    for cat, label, color in [
                        ("tools_platforms",  "Tools & Platforms", "#10b981"),
                        ("core_cs_concepts", "CS Concepts",       "#6366f1"),
                    ]:
                        items = sc.get(cat, [])
                        if items:
                            st.markdown(
                                f'<div style="margin-bottom:8px">'
                                f'<span style="font-size:0.78rem;font-weight:600;color:{color};text-transform:uppercase;letter-spacing:0.05em">'
                                f'{label} ({len(items)})</span><br>'
                                + " ".join(f'<span class="tag">{s}</span>' for s in items)
                                + '</div>', unsafe_allow_html=True)

            # ── Proficiency consistency ───────────────────────────────────
            pc = ev.get("proficiency_consistency", {})
            if pc.get("analysis"):
                st.markdown('<div class="section-head">🧠 Proficiency Consistency</div>', unsafe_allow_html=True)
                for item in pc["analysis"]:
                    icon = "✅" if item.get("aligned") else "⚠️"
                    st.write(f"{icon} **{item.get('skill','').title()}** — Declared `{item.get('declared_level')}` | Evidence `{item.get('evidence_level')}/3` — {item.get('reasoning','')}")
                if pc.get("overall_assessment"):
                    st.info(pc["overall_assessment"])

            # ── Doubt detection ───────────────────────────────────────────
            dd     = ev.get("doubt_detection", {})
            issues = dd.get("issues", [])
            # Also show clarification questions from deterministic engine
            clarification_qs = ev.get("clarification_questions", [])
            if issues or clarification_qs:
                st.markdown('<div class="section-head">⚠️ Doubt Detection</div>', unsafe_allow_html=True)
                # Render LLM-returned issues (may be dicts or strings)
                for issue in issues:
                    if isinstance(issue, dict):
                        itype = issue.get("type", issue.get("issue_type", "Issue"))
                        if isinstance(itype, str):
                            itype = itype.replace("_", " ").title()
                        desc  = issue.get("description", issue.get("context", issue.get("question", str(issue))))
                        qs    = issue.get("questions", [])
                        if isinstance(qs, str):
                            qs = [qs]
                        with st.expander(f"🔍 {itype} — {str(desc)[:70]}"):
                            st.write(str(desc))
                            for q in qs:
                                st.markdown(f'<div class="warn-box">❓ {q}</div>', unsafe_allow_html=True)
                    elif isinstance(issue, str):
                        with st.expander(f"🔍 Issue — {issue[:70]}"):
                            st.write(issue)
                # Render clarification questions from doubt engine (always dicts/dataclasses serialized as strings)
                if clarification_qs:
                    for i, q in enumerate(clarification_qs, 1):
                        st.markdown(f'<div class="warn-box">❓ Q{i}: {q}</div>', unsafe_allow_html=True)

            # ── Factual evaluation ────────────────────────────────────────
            fe = ev.get("factual_evaluation", {})
            if fe:
                st.markdown('<div class="section-head">🔬 Factual Evaluation</div>', unsafe_allow_html=True)
                f1,f2,f3,f4 = st.columns(4)
                f1.metric("Tech Depth",    fe.get("technical_depth","—"))
                f2.metric("Consistency",   fe.get("logical_consistency","—"))
                f3.metric("Metrics",       fe.get("metric_realism","—"))
                f4.metric("Skill Align",   fe.get("skill_alignment","—"))
                if fe.get("confidence_narrative"):
                    st.caption(fe["confidence_narrative"])

            # ── Internal consistency ──────────────────────────────────────
            ic = ev.get("internal_consistency", {})
            ic_flags = ic.get("flags", [])
            if ic or ic_flags:
                st.markdown('<div class="section-head">🔍 Internal Consistency</div>', unsafe_allow_html=True)
                ic1, ic2, ic3, ic4 = st.columns(4)
                def ic_badge(val, good_vals, bad_vals):
                    v = str(val).lower()
                    if any(g in v for g in good_vals):  return f"✅ {val}"
                    if any(b in v for b in bad_vals):   return f"❌ {val}"
                    return f"⚠️ {val}"
                ic1.metric("Timeline",    ic.get("timeline_alignment",   "—"))
                ic2.metric("Skill Coverage", ic.get("skill_usage_coverage", "—"))
                ic3.metric("Claim Density",  ic.get("claim_density",        "—"))
                ic4.metric("Coherence",   ic.get("cross_section_coherence","—"))
                if ic_flags:
                    for flag in ic_flags:
                        st.markdown(
                            f'<div style="padding:6px 12px;background:#fef3c7;border-left:3px solid #f59e0b;'
                            f'border-radius:4px;margin:3px 0;font-size:0.85rem">⚠️ {flag}</div>',
                            unsafe_allow_html=True)

            # ── Score history ─────────────────────────────────────────
            history = st.session_state.get("score_history", [])
            if len(history) > 1:
                st.markdown('<div class="section-head">📈 Score History</div>', unsafe_allow_html=True)
                cols_h = st.columns(len(history))
                for col, entry in zip(cols_h, history):
                    clr = "#10b981" if entry["verdict"] == "Strong" else ("#f59e0b" if entry["verdict"] == "Moderate" else "#ef4444")
                    col.markdown(
                        f'<div style="text-align:center;padding:8px;background:#f8fafc;border-top:3px solid {clr};border-radius:8px">' +
                        f'<div style="font-size:1.2rem;font-weight:700;color:{clr}">{entry["score"]:.0f}</div>' +
                        f'<div style="font-size:0.72rem;color:#64748b">{entry["role"][:18]}</div></div>',
                        unsafe_allow_html=True)

            # ── Quality assessment ────────────────────────────────────────
            qa = ev.get("resume_quality_assessment", {})
            if qa:
                st.markdown('<div class="section-head">📋 Resume Quality Assessment</div>', unsafe_allow_html=True)

                def qa_color(val):
                    v = str(val).lower()
                    if v in ("high", "strong", "low redundancy", "low"):
                        return "#10b981", "✅"
                    if v in ("medium", "acceptable", "moderate"):
                        return "#f59e0b", "⚠️"
                    return "#ef4444", "❌"

                qa_metrics = [
                    ("Structure Clarity",  qa.get("structure_clarity",  "—")),
                    ("ATS Compatibility",  qa.get("ats_compatibility",   "—")),
                    ("Role Relevance",     qa.get("role_relevance",      "—")),
                    ("Redundancy",         qa.get("redundancy_level",    "—")),
                ]
                cols = st.columns(4)
                for col, (label, val) in zip(cols, qa_metrics):
                    color, icon = qa_color(val)
                    col.markdown(
                        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
                        f'padding:12px 8px;text-align:center;border-top:3px solid {color}">'
                        f'<div style="font-size:0.75rem;color:#64748b;margin-bottom:4px">{label}</div>'
                        f'<div style="font-size:1rem;font-weight:700;color:{color}">{icon} {val}</div>'
                        f'</div>', unsafe_allow_html=True)

                overall = qa.get("overall_quality", "")
                if overall:
                    oc, oi = qa_color(overall)
                    st.markdown(
                        f'<div style="margin-top:10px;padding:8px 14px;background:#f1f5f9;'
                        f'border-left:4px solid {oc};border-radius:4px">'
                        f'<strong>Overall:</strong> {oi} {overall}</div>',
                        unsafe_allow_html=True)

                imps = qa.get("improvements", [])
                if imps:
                    st.markdown('<div style="margin-top:10px;font-size:0.85rem;font-weight:600;color:#374151">Actionable Improvements</div>', unsafe_allow_html=True)
                    for imp in imps:
                        st.markdown(f'<div style="padding:4px 0 4px 12px;border-left:2px solid #6366f1;margin:3px 0;font-size:0.87rem">→ {imp}</div>', unsafe_allow_html=True)

            # ── Career plan ───────────────────────────────────────────────
            # ── ATS Flags ──────────────────────────────────────────────
            ats_flags = qa.get("ats_flags", []) if qa else []
            if ats_flags:
                st.markdown('<div class="section-head">🤖 ATS Compatibility Warnings</div>', unsafe_allow_html=True)
                for flag in ats_flags:
                    st.markdown(f'<div style="background:#fee2e2;border-left:3px solid #ef4444;border-radius:6px;padding:0.5rem 0.8rem;margin:0.3rem 0;font-size:0.87rem">⚠️ {flag}</div>', unsafe_allow_html=True)

            cp = ev.get("career_improvement_plan", {})
            if cp.get("missing_skills_to_learn") or cp.get("suggested_projects") or cp.get("learning_roadmap"):
                st.markdown('<div class="section-head">🚀 Career Improvement Plan</div>', unsafe_allow_html=True)
                st.write(f"**Target Domain:** {cp.get('target_domain','—')}")

                if cp.get("missing_skills_to_learn"):
                    st.caption("📚 Skills to Learn")
                    for s in cp["missing_skills_to_learn"]:
                        skill    = s.get("skill","")
                        reason   = s.get("reason","")
                        resource = s.get("resource","")
                        st.markdown(f"""
                        <div style="background:#f0fdf4;border-left:3px solid #22c55e;
                             border-radius:6px;padding:0.5rem 0.8rem;margin:0.3rem 0;">
                          <strong>{skill}</strong><br>
                          <span style="font-size:0.85rem;color:#374151">{reason}</span><br>
                          <span style="font-size:0.82rem;color:#6366f1">📖 {resource}</span>
                        </div>""", unsafe_allow_html=True)

                if cp.get("suggested_projects"):
                    st.caption("🛠️ Suggested Projects")
                    for i, p in enumerate(cp["suggested_projects"], 1):
                        st.markdown(f"""
                        <div style="background:#eff6ff;border-left:3px solid #3b82f6;
                             border-radius:6px;padding:0.5rem 0.8rem;margin:0.3rem 0;">
                          <strong>Project {i}: {p.get('title','')}</strong><br>
                          <span style="font-size:0.85rem">{p.get('description','')}</span>
                        </div>""", unsafe_allow_html=True)

                if cp.get("learning_roadmap"):
                    st.caption("🗓️ Week-by-Week Roadmap")
                    for r in cp["learning_roadmap"]:
                        week     = r.get("week","")
                        focus    = r.get("focus","")
                        task     = r.get("task","") or focus
                        resource = r.get("resource","")
                        st.markdown(f"""
                        <div style="background:#faf5ff;border-left:3px solid #a855f7;
                             border-radius:6px;padding:0.5rem 0.8rem;margin:0.25rem 0;">
                          <strong>{week}</strong> — {focus}<br>
                          <span style="font-size:0.83rem;color:#374151">Task: {task}</span>
                          {"<br><span style='font-size:0.81rem;color:#6366f1'>📖 " + resource + "</span>" if resource else ""}
                        </div>""", unsafe_allow_html=True)

            # ── Optimized resume ──────────────────────────────────────────
            if ev.get("final_resume"):
                st.markdown('<div class="section-head">📄 Optimized Resume</div>', unsafe_allow_html=True)

                session_id_dl = analysis.get("session_id","")
                dl1, dl2, dl3 = st.columns(3)

                # Plain text download
                with dl1:
                    st.download_button(
                        "⬇️ Download TXT",
                        data        = ev.get("final_resume") or "",
                        file_name   = "optimized_resume.txt",
                        mime        = "text/plain",
                        use_container_width = True,
                    )

                # DOCX download — 1-click (bytes pre-fetched at eval time)
                with dl2:
                    health = st.session_state.get("health", {})
                    docx_bytes = st.session_state.get("docx_bytes")
                    if docx_bytes:
                        st.download_button(
                            "⬇️ Download DOCX",
                            data      = docx_bytes,
                            file_name = "optimized_resume.docx",
                            mime      = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width = True,
                            key       = "dl_docx",
                        )
                    else:
                        st.caption("DOCX unavailable")

                # PDF download — 1-click (bytes pre-fetched at eval time)
                with dl3:
                    pdf_bytes = st.session_state.get("pdf_bytes")
                    if pdf_bytes:
                        st.download_button(
                            "⬇️ Download PDF",
                            data      = pdf_bytes,
                            file_name = "optimized_resume.pdf",
                            mime      = "application/pdf",
                            use_container_width = True,
                            key       = "dl_pdf",
                        )
                    else:
                        pdf_err = st.session_state.get("pdf_error", "")
                        if "reportlab" in pdf_err.lower():
                            st.caption("PDF: run `pip install reportlab`")
                        elif pdf_err:
                            st.caption(f"PDF error: {pdf_err[:80]}")
                        else:
                            st.caption("PDF unavailable")

                with st.expander("👁 Preview Resume", expanded=False):
                    st.text(ev.get("final_resume", ""))

            # ── Factual Integrity Gate ────────────────────────────────────
            clarif_qs = ev.get("clarification_questions", [])
            clarif_required = ev.get("clarification_required", False)

            if clarif_required and clarif_qs:
                # Clear any stale rule-based resume so LLM generates fresh after answers
                if ev.get("final_resume") and "rule-based" in str(ev.get("resume_verify_issues", [])):
                    ev["final_resume"] = None
                st.markdown('<div class="section-head">🔍 Factual Integrity Check</div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:10px;padding:1rem 1.2rem;margin-bottom:1rem;">
                  <strong>⚠️ Resume optimization is blocked until these questions are answered.</strong><br>
                  <small>Smart Resume never expands or decorates claims that have not been verified.
                  Answer honestly — your responses will be used to strengthen your resume with real facts only.</small>
                </div>
                """, unsafe_allow_html=True)

                session_id = analysis.get("session_id","")
                answers = []
                for i, q in enumerate(clarif_qs):
                    ans = st.text_area(
                        f"Q{i+1}: {q}",
                        key=f"clarify_{i}",
                        height=90,
                        placeholder='Be specific — or type "nil" if the question does not apply to you.'
                    )
                    if ans.strip():
                        answers.append({"question": q, "answer": ans})

                answered_count = len(answers)
                total_count = len(clarif_qs)
                st.caption(f"Answered {answered_count}/{total_count} questions")

                col_btn, col_skip = st.columns([2,1])
                with col_btn:
                    if st.button(
                        "✅ Submit Answers & Generate Optimized Resume",
                        type="primary",
                        use_container_width=True,
                        disabled=answered_count == 0
                    ):
                        with st.spinner("Verifying answers and generating optimized resume..."):
                            try:
                                r = httpx.post(
                                    f"{API_BASE}/evaluate/{session_id}/clarify",
                                    json=answers, timeout=90
                                )
                                if r.status_code == 200:
                                    ev_result = r.json()
                                    st.session_state.evaluation = ev_result
                                    # Always fetch downloads after clarification
                                    _sid = st.session_state.analysis.get("session_id","")
                                    _fetch_downloads(_sid)
                                    # Also clear docx/pdf cache to force re-fetch
                                    st.session_state.pop("docx_bytes", None)
                                    st.session_state.pop("pdf_bytes", None)
                                    _fetch_downloads(_sid)
                                    st.rerun()
                                else:
                                    st.error(f"Error: {r.text[:200]}")
                            except Exception as e:
                                if "429" in str(e):
                                    st.warning("⚠️ Groq rate limit — wait 30 seconds and try again.")
                                else:
                                    st.error(str(e))
                with col_skip:
                    if st.button("Skip & optimize anyway", use_container_width=True):
                        with st.spinner("Generating resume without clarification..."):
                            try:
                                r = httpx.post(f"{API_BASE}/evaluate/{session_id}/clarify", json=[], timeout=90)
                                if r.status_code == 200:
                                    st.session_state.evaluation = r.json()
                                    st.session_state.pop("docx_bytes", None)
                                    st.session_state.pop("pdf_bytes", None)
                                    _fetch_downloads(session_id)
                                    st.rerun()
                            except Exception as e:
                                st.error(str(e))


            # ── Raw JSON ──────────────────────────────────────────────────
            # ── Score History ─────────────────────────────────────────────
            if len(st.session_state.get("score_history", [])) > 1:
                st.markdown('<div class="section-head">📈 Score History</div>', unsafe_allow_html=True)
                hist = st.session_state.score_history
                cols = st.columns(len(hist))
                for col, entry in zip(cols, hist):
                    w = entry.get("weighted")
                    a = entry.get("ats")
                    col.markdown(
                        f'<div style="text-align:center;background:#f1f5f9;border-radius:8px;padding:8px 4px;">' +
                        f'<div style="font-size:0.7rem;color:#64748b">{entry["timestamp"]}</div>' +
                        f'<div style="font-weight:700;font-size:1.1rem">{w:.0f}' if w else '<div style="font-weight:700">—' +
                        f'</div><div style="font-size:0.75rem;color:#6366f1">ATS {a}</div>' if a else '</div>' +
                        '</div>',
                        unsafe_allow_html=True)

            with st.expander("🔩 Raw JSON"):
                tab1, tab2 = st.tabs(["Phase 1 — Parse", "Phase 2 — Evaluation"])
                with tab1:
                    st.json(analysis)
                with tab2:
                    st.json(evaluation)