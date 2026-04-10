import streamlit as st
import httpx
import os

API_BASE = os.getenv("SMART_RESUME_API_URL", "http://localhost:8001")

st.set_page_config(
    page_title="Career Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
* { font-family: 'DM Sans', sans-serif; }

.main-title {
  font-size: 2.4rem; font-weight: 800; font-family: 'Syne', sans-serif;
  background: linear-gradient(135deg, #10b981 0%, #047857 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 0;
}
.subtitle { color: #6b7280; font-size: 1rem; margin-bottom: 2rem; }

.job-card {
  background: #f9fafb; border: 1px solid #e5e7eb;
  border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 0.8rem;
}
.job-title { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #111827; }
.match-green { background: #d1fae5; color: #065f46; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
.match-amber { background: #fef3c7; color: #92400e; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
.match-red   { background: #fee2e2; color: #991b1b; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }

.pill-have { display: inline-block; background: #d1fae5; color: #065f46; border-radius: 6px; padding: 2px 9px; font-size: 0.75rem; font-weight: 500; margin: 2px 3px 2px 0; }
.pill-miss { display: inline-block; background: #fef3c7; color: #92400e; border-radius: 6px; padding: 2px 9px; font-size: 0.75rem; font-weight: 500; margin: 2px 3px 2px 0; }
.pill-red  { display: inline-block; background: #fee2e2; color: #991b1b; border-radius: 6px; padding: 2px 9px; font-size: 0.75rem; font-weight: 500; margin: 2px 3px 2px 0; }

.score-box { background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; text-align: center; }
.score-num { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; }
.score-lbl { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; }

.rm-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem; display: flex; gap: 12px; align-items: flex-start; }
.rm-num { background: #d1fae5; color: #065f46; border-radius: 8px; padding: 4px 10px; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.8rem; flex-shrink: 0; }
.rm-dur { background: #e0e7ff; color: #3730a3; border-radius: 20px; padding: 2px 9px; font-size: 0.7rem; font-weight: 600; flex-shrink: 0; }

.tl-item { display: flex; gap: 12px; padding: 8px 0; border-bottom: 1px solid #f3f4f6; }
.tl-period { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.8rem; color: #10b981; min-width: 80px; flex-shrink: 0; }

.daily-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; }
.daily-time { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #10b981; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎯 Career Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered career matching — upload your resume and get your top job matches, skill gaps, roadmap and daily plan.</div>', unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key in ["cr_analysis", "cr_detail", "cr_resume_text"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── STEP 0: Resume input ───────────────────────────────────────────────────────
st.markdown("### Upload your resume")

# Check if main page already processed a resume
has_main = st.session_state.get("analysis") is not None
resume_text = ""

if has_main:
    st.success("✅ Resume from main page detected — using it automatically.")
    payload = st.session_state.analysis
    resume_text = payload.get("resume_raw_text", "")

uploaded = st.file_uploader(
    "Upload a resume (PDF, DOCX) — or use the one from the main page above",
    type=["pdf", "docx", "txt"],
    label_visibility="collapsed"
)

if uploaded:
    with st.spinner("Parsing resume..."):
        try:
            from core.parser import parse_document
            file_bytes = uploaded.read()
            p_out = parse_document(file_bytes, uploaded.name)
            resume_text = p_out.get("raw_text", "")
            st.success(f"✅ Parsed: {uploaded.name}")
            # Reset analysis if new file
            if st.session_state.get("cr_last_file") != uploaded.name:
                st.session_state.cr_analysis = None
                st.session_state.cr_detail = None
                st.session_state["cr_last_file"] = uploaded.name
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            st.stop()

if not resume_text:
    st.info("💡 Please upload a resume above to begin.")
    st.stop()

# Store for detail calls
st.session_state.cr_resume_text = resume_text

# ── STEP 1: Analyse ────────────────────────────────────────────────────────────
if st.session_state.cr_analysis is None:
    if st.button("🔍 Find my career matches", type="primary"):
        with st.spinner("Analysing your resume with AI..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/career/recommend",
                    json={"resume_text": resume_text},
                    timeout=60
                )
                if resp.status_code == 200:
                    st.session_state.cr_analysis = resp.json()
                    st.session_state.cr_detail = None
                    st.rerun()
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
    st.stop()

# ── STEP 2: Show matches ───────────────────────────────────────────────────────
analysis = st.session_state.cr_analysis
name     = analysis.get("name", "")
level    = analysis.get("level", "Junior")
score    = analysis.get("readiness_score", 0)
reason   = analysis.get("readiness_reason", "")
jobs     = analysis.get("top_jobs", [])

# Profile header
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if name:
        st.markdown(f"### 👤 {name}")
    st.caption(reason)
with col2:
    st.markdown(f"""<div class="score-box">
        <div class="score-num" style="color:#10b981">{score}%</div>
        <div class="score-lbl">Career readiness</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="score-box">
        <div class="score-num" style="color:#6366f1">{level}</div>
        <div class="score-lbl">Candidate level</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🎯 Your top career matches")
st.caption("Click **View career plan** on any role to get your full roadmap, courses and daily plan.")

for i, job in enumerate(jobs):
    pct   = job.get("match_percent", 0)
    badge = "match-green" if pct >= 75 else ("match-amber" if pct >= 50 else "match-red")
    have  = job.get("skills_you_have", [])
    miss  = job.get("skills_missing", [])

    have_html = " ".join(f'<span class="pill-have">{s}</span>' for s in have)
    miss_html = " ".join(f'<span class="pill-miss">{s}</span>' for s in miss)

    with st.container():
        st.markdown(f"""<div class="job-card">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <span class="job-title">{job.get("title","")}</span>
                <span class="{badge}">{pct}% match</span>
            </div>
            <div style="margin-bottom:6px;">
                <span style="font-size:0.72rem;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.07em;margin-right:6px;">Have:</span>
                {have_html if have_html else '<span style="color:#9ca3af;font-size:0.8rem;">—</span>'}
            </div>
            <div>
                <span style="font-size:0.72rem;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.07em;margin-right:6px;">Missing:</span>
                {miss_html if miss_html else "<span style='color:#9ca3af;font-size:0.8rem;'>None — you're ready!</span>"}
            </div>
        </div>""", unsafe_allow_html=True)

        if st.button(f"View career plan →", key=f"plan_{i}"):
            with st.spinner(f"Building your {job.get('title','')} career plan..."):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/career/recommend/detail",
                        json={
                            "resume_text": st.session_state.cr_resume_text,
                            "job_title": job.get("title", "")
                        },
                        timeout=90
                    )
                    if resp.status_code == 200:
                        st.session_state.cr_detail = resp.json()
                        st.rerun()
                    else:
                        st.error(f"API error: {resp.text}")
                except Exception as e:
                    st.error(f"Failed: {e}")

# ── STEP 3: Detail view ────────────────────────────────────────────────────────
if st.session_state.cr_detail:
    d = st.session_state.cr_detail
    st.markdown("---")
    st.markdown(f"## 🗺️ Career plan: {d.get('role','')}")

    # ── Scores ──
    st.markdown("#### Readiness scores")
    sc = d.get("scores", {})
    s1, s2, s3, s4 = st.columns(4)
    for col, label, key, color in [
        (s1, "Overall",      "overall",       "#10b981"),
        (s2, "Skills",       "skills",        "#6366f1"),
        (s3, "Projects",     "projects",      "#f59e0b"),
        (s4, "Job ready",    "job_readiness", "#ef4444"),
    ]:
        v = sc.get(key, 0)
        col.markdown(f"""<div class="score-box">
            <div class="score-num" style="color:{color}">{v}</div>
            <div class="score-lbl">{label}</div>
            <div style="height:4px;background:#f3f4f6;border-radius:2px;margin-top:8px;overflow:hidden;">
              <div style="width:{v}%;height:100%;background:{color};border-radius:2px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Skill analysis ──
    st.markdown("#### Skill analysis")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**✅ Skills you have**")
        have_pills = " ".join(f'<span class="pill-have">{s}</span>' for s in d.get("skills_have", []))
        st.markdown(have_pills or "_None detected_", unsafe_allow_html=True)
    with cb:
        st.markdown("**❌ Skills to learn**")
        miss_pills = " ".join(f'<span class="pill-red">{s}</span>' for s in d.get("skills_missing", []))
        st.markdown(miss_pills or "_None — you're ready!_", unsafe_allow_html=True)

    st.markdown("---")

    # ── Courses ──
    st.markdown("#### 📚 Recommended courses")
    courses = d.get("courses", [])
    if courses:
        for c in courses:
            with st.expander(f"**{c.get('skill','')}** — {c.get('course','')} ({c.get('platform','')})"):
                url = c.get("url", "")
                if url:
                    st.markdown(f"[Open course →]({url})")
    else:
        st.write("No courses needed — your skills are sufficient!")

    st.markdown("---")

    # ── Roadmap ──
    st.markdown("#### 🧭 Step-by-step roadmap")
    for step in d.get("roadmap", []):
        st.markdown(f"""<div class="rm-card">
            <span class="rm-num">{step.get('step','')}</span>
            <div style="flex:1">
                <div style="font-weight:500;margin-bottom:3px;">{step.get('title','')}</div>
                <div style="color:#6b7280;font-size:0.85rem;">{step.get('description','')}</div>
            </div>
            <span class="rm-dur">{step.get('duration','')}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Timeline ──
    st.markdown("#### 📅 Timeline")
    for t in d.get("timeline", []):
        st.markdown(f"""<div class="tl-item">
            <span class="tl-period">{t.get('period','')}</span>
            <span style="color:#374151;font-size:0.88rem;">{t.get('goal','')}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Daily plan ──
    st.markdown("#### ⏰ Daily plan")
    dp = d.get("daily_plan", {})
    d1, d2, d3 = st.columns(3)
    for col, time_label, key in [
        (d1, "Morning",   "morning"),
        (d2, "Afternoon", "afternoon"),
        (d3, "Evening",   "evening"),
    ]:
        col.markdown(f"""<div class="daily-card">
            <div class="daily-time">{time_label}</div>
            <div style="font-size:0.87rem;color:#374151;line-height:1.5;">{dp.get(key,'—')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to matches"):
        st.session_state.cr_detail = None
        st.rerun()