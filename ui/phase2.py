"""
Smart Resume — Phase 2 Streamlit UI
=====================================
Shows full evaluation results from /evaluate/{session_id}.
Run with: streamlit run ui/phase2.py
"""

import os
import httpx
import streamlit as st

API_BASE = os.getenv("SMART_RESUME_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Smart Resume — Evaluation", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.big-title { font-size:2.2rem; font-weight:800;
  background:linear-gradient(135deg,#667eea,#764ba2);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.section-head { font-size:1.05rem; font-weight:700; color:#374151;
  border-left:4px solid #667eea; padding-left:0.6rem; margin:1.2rem 0 0.5rem; }
.verdict-strong  { background:#d1fae5; border:1px solid #10b981; border-radius:8px; padding:0.6rem 1rem; }
.verdict-moderate{ background:#fef3c7; border:1px solid #f59e0b; border-radius:8px; padding:0.6rem 1rem; }
.verdict-weak    { background:#fee2e2; border:1px solid #ef4444; border-radius:8px; padding:0.6rem 1rem; }
.tag { display:inline-block; background:#ede9fe; color:#5b21b6;
  border-radius:6px; padding:2px 10px; font-size:0.8rem; margin:2px; }
.miss-tag { display:inline-block; background:#fee2e2; color:#b91c1c;
  border-radius:6px; padding:2px 10px; font-size:0.8rem; margin:2px; }
.fix-box { background:#ecfdf5; border:1px solid #6ee7b7;
  border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🧠 Smart Resume — Phase 2 Evaluation</div>', unsafe_allow_html=True)
st.caption("Paste your session ID from Phase 1 to run the full AI evaluation.")

col_in, col_out = st.columns([1, 2], gap="large")

with col_in:
    session_id = st.text_input("Session ID from Phase 1", placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    run_btn    = st.button("🚀 Run Full Evaluation", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Need to add clarification answers first?")
    with st.expander("Add Clarification Answers"):
        q1 = st.text_input("Question 1")
        a1 = st.text_area("Answer 1", height=80)
        q2 = st.text_input("Question 2")
        a2 = st.text_area("Answer 2", height=80)
        clarify_btn = st.button("Re-evaluate with Answers")

if "eval_result" not in st.session_state:
    st.session_state.eval_result = None

# ── Run evaluation ────────────────────────────────────────────────────────────
if run_btn and session_id.strip():
    with col_out:
        with st.spinner("🔍 Claude is evaluating your resume..."):
            try:
                resp = httpx.post(f"{API_BASE}/evaluate/{session_id.strip()}", timeout=90)
                if resp.status_code == 200:
                    st.session_state.eval_result = resp.json()
                    st.success("✅ Evaluation complete!")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text[:300]}")
            except httpx.ConnectError:
                st.error("Cannot connect to backend. Make sure `python main.py` is running.")
            except Exception as e:
                st.error(f"Error: {e}")

if clarify_btn and session_id.strip():
    answers = []
    if q1.strip() and a1.strip():
        answers.append({"question": q1, "answer": a1})
    if q2.strip() and a2.strip():
        answers.append({"question": q2, "answer": a2})
    if answers:
        with col_out:
            with st.spinner("Re-evaluating with your answers..."):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/evaluate/{session_id.strip()}/clarify",
                        json=answers, timeout=90
                    )
                    if resp.status_code == 200:
                        st.session_state.eval_result = resp.json()
                        st.success("Re-evaluation complete!")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text[:200]}")
                except Exception as e:
                    st.error(str(e))

# ── Display results ───────────────────────────────────────────────────────────
result = st.session_state.eval_result

with col_out:
    if not result:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#9ca3af;">
          <div style="font-size:3rem">🧠</div>
          <div style="font-size:1.1rem;margin-top:1rem">
            Enter your session ID and click Run Full Evaluation.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # ── Mismatch corrections ──────────────────────────────────────────
        fixes = result.get("mismatch_corrections", [])
        if fixes:
            st.markdown('<div class="section-head">🔀 Section Mismatch Corrections</div>', unsafe_allow_html=True)
            for fix in fixes:
                st.markdown(f'<div class="fix-box">✅ {fix}</div>', unsafe_allow_html=True)

        # ── Job match ─────────────────────────────────────────────────────
        jma     = result.get("job_match_analysis", {})
        verdict = jma.get("verdict", "")
        css     = {"Strong": "verdict-strong", "Moderate": "verdict-moderate"}.get(verdict, "verdict-weak")

        st.markdown('<div class="section-head">📊 Job Match Analysis</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        v1.metric("Domain",  jma.get("domain", "—"))
        v2.metric("Verdict", verdict)
        st.markdown(f'<div class="{css}">{jma.get("alignment_score_reasoning","")}</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.caption("Matched Skills")
            tags = " ".join(f'<span class="tag">{s}</span>' for s in jma.get("matched_skills", []))
            st.markdown(tags or "—", unsafe_allow_html=True)
        with m2:
            st.caption("Missing Skills")
            tags = " ".join(f'<span class="miss-tag">{s}</span>' for s in jma.get("missing_skills", []))
            st.markdown(tags or "None ✅", unsafe_allow_html=True)

        # ── Skill classification ──────────────────────────────────────────
        sc = result.get("skill_classification", {})
        st.markdown('<div class="section-head">🛠️ Skill Classification</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            for cat, label in [("programming_languages","Languages"), ("frameworks_libraries","Frameworks"), ("databases","Databases")]:
                items = sc.get(cat, [])
                if items:
                    st.caption(label)
                    st.markdown(" ".join(f'<span class="tag">{s}</span>' for s in items), unsafe_allow_html=True)
        with sc2:
            for cat, label in [("tools_platforms","Tools & Platforms"), ("core_cs_concepts","CS Concepts")]:
                items = sc.get(cat, [])
                if items:
                    st.caption(label)
                    st.markdown(" ".join(f'<span class="tag">{s}</span>' for s in items), unsafe_allow_html=True)

        # ── Proficiency consistency ───────────────────────────────────────
        pc = result.get("proficiency_consistency", {})
        st.markdown('<div class="section-head">🧠 Proficiency Consistency</div>', unsafe_allow_html=True)
        for item in pc.get("analysis", []):
            icon = "✅" if item.get("aligned") else "⚠️"
            st.write(f"{icon} **{item.get('skill','').title()}** — Declared: `{item.get('declared_level')}` | Evidence: `{item.get('evidence_level')}/3` | {item.get('reasoning','')}")
        if pc.get("overall_assessment"):
            st.info(pc["overall_assessment"])

        # ── Doubt detection ───────────────────────────────────────────────
        dd = result.get("doubt_detection", {})
        issues = dd.get("issues", [])
        if issues:
            st.markdown('<div class="section-head">⚠️ Doubt Detection</div>', unsafe_allow_html=True)
            for issue in issues:
                with st.expander(f"🔍 {issue.get('type','').replace('_',' ').title()} — {issue.get('description','')[:60]}"):
                    st.write(issue.get("description",""))
                    st.caption("Clarification Questions:")
                    for q in issue.get("questions", []):
                        st.write(f"• {q}")

        # ── Quality assessment ────────────────────────────────────────────
        qa = result.get("resume_quality_assessment", {})
        st.markdown('<div class="section-head">📋 Resume Quality</div>', unsafe_allow_html=True)
        q1c, q2c, q3c, q4c = st.columns(4)
        q1c.metric("Structure",    qa.get("structure_clarity","—"))
        q2c.metric("ATS",          qa.get("ats_compatibility","—"))
        q3c.metric("Relevance",    qa.get("role_relevance","—"))
        q4c.metric("Overall",      qa.get("overall_quality","—"))
        for imp in qa.get("improvements", []):
            st.write(f"• {imp}")

        # ── Career plan ───────────────────────────────────────────────────
        cp = result.get("career_improvement_plan", {})
        st.markdown('<div class="section-head">🚀 Career Improvement Plan</div>', unsafe_allow_html=True)
        st.write(f"**Target Domain:** {cp.get('target_domain','—')}")

        if cp.get("missing_skills_to_learn"):
            st.caption("Skills to Learn")
            for s in cp["missing_skills_to_learn"]:
                st.write(f"• **{s.get('skill','')}** — {s.get('reason','')} → *{s.get('resource','')}*")

        if cp.get("suggested_projects"):
            st.caption("Suggested Projects")
            for p in cp["suggested_projects"]:
                st.write(f"• **{p.get('title','')}** — {p.get('description','')}")

        if cp.get("learning_roadmap"):
            st.caption("Learning Roadmap")
            for r in cp["learning_roadmap"]:
                st.write(f"• Week {r.get('week','')} → {r.get('focus','')}")

        # ── Final resume ──────────────────────────────────────────────────
        if result.get("final_resume"):
            st.markdown('<div class="section-head">📄 Optimized Resume</div>', unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Optimized Resume (.txt)",
                data=result["final_resume"],
                file_name="optimized_resume.txt",
                mime="text/plain",
            )
            with st.expander("Preview Optimized Resume"):
                st.text(result["final_resume"])

        # ── Full JSON ─────────────────────────────────────────────────────
        with st.expander("🔩 Full Evaluation JSON"):
            st.json(result)