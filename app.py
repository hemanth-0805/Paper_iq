import os
import re
import math  # noqa: F401 (used in some existing UI flows)
from typing import Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from config import APP_TITLE
from auth import login_user, signup_user, logout
from db import create_database, get_mongo_db, _get_analysis_col, _get_users_col
from nlp import _init_nltk
from engine import InsightEngine, extract_text, extract_keywords, compare_papers, PYMUPDF_AVAILABLE
from storage import save_analysis, build_insights_text, get_user_history, get_analysis_record
from plots import (
    plot_word_freq,
    plot_sentence_length_distribution,
    plot_keyword_importance,
    plot_readability_metrics,
)
from chat_ui import render_chat_with_paper


# ----------------------------
# Streamlit UI
# ----------------------------
def render_login() -> None:
    st.title(f"📚 {APP_TITLE}")
    st.caption("✨ Analyze research papers quickly with AI summaries, insights, and analytics.")

    tab_login, tab_signup, tab_admin = st.tabs(["🔐 User Login", "📝 User Signup", "🛠️ Admin Login"])

    with tab_login:
        st.subheader("👋 User Login")
        with st.form("user_login_form", clear_on_submit=False):
            username = st.text_input("Username 👤")
            password = st.text_input("Password 🔑", type="password")
            submitted = st.form_submit_button("Login ➡️")
        if submitted:
            ok, msg, auth = login_user(username, password, required_role="user")
            if ok:
                st.session_state["auth"] = auth
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        st.subheader("✨ Create Account")
        with st.form("signup_form", clear_on_submit=True):
            username = st.text_input("Choose a username 👤")
            password = st.text_input("Choose a password 🔑", type="password")
            password2 = st.text_input("Confirm password 🔑", type="password")
            submitted = st.form_submit_button("Sign up 🚀")
        if submitted:
            if password != password2:
                st.error("Passwords do not match.")
            else:
                ok, msg = signup_user(username, password, role="user")
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    with tab_admin:
        st.subheader("🛠️ Admin Login")
        with st.form("admin_login_form", clear_on_submit=False):
            username = st.text_input("Admin username 👤")
            password = st.text_input("Admin password 🔑", type="password")
            submitted = st.form_submit_button("Login as admin ➡️")
        if submitted:
            ok, msg, auth = login_user(username, password, required_role="admin")
            if ok:
                st.session_state["auth"] = auth
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


def render_user_sidebar() -> str:
    st.sidebar.title("🧠 PaperIQ")
    st.sidebar.caption(f"👤 Signed in as **{st.session_state['auth']['username']}**")
    options = {
        "🏠 Dashboard": "Dashboard",
        "📤 Upload Paper": "Upload Paper",
        "⚖️ Compare Papers": "Compare Papers",
        "📂 My Previous Analyses": "My Previous Analyses",
        "💬 Chat with Paper": "Chat with Paper",
        "📈 Analytics Dashboard": "Analytics Dashboard",
        "🚪 Logout": "Logout"
    }
    page = st.sidebar.radio("Navigation", list(options.keys()), index=0)
    return options.get(page, "Dashboard")


def render_admin_sidebar() -> str:
    st.sidebar.title("🛠️ PaperIQ (Admin)")
    st.sidebar.caption(f"👤 Signed in as **{st.session_state['auth']['username']}**")
    options = {
        "📊 Admin Dashboard": "Admin Dashboard",
        "👥 Users": "Users",
        "📂 All Analyses": "All Analyses",
        "⚙️ System Stats": "System Stats",
        "🚪 Logout": "Logout"
    }
    page = st.sidebar.radio("Navigation", list(options.keys()), index=0)
    return options.get(page, "Admin Dashboard")


def render_user_dashboard() -> None:
    st.subheader("🏠 Dashboard")
    st.write("Welcome! Upload papers for section-wise AI summaries, keyword extraction, and actionable insights. ✨")

    user_id = int(st.session_state["auth"]["user_id"])
    hist = get_user_history(user_id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Papers analyzed", int(len(hist)))
    c2.metric("Avg readability", float(hist["readability_score"].mean()) if len(hist) else 0.0)
    c3.metric("Avg coherence", float(hist["coherence_score"].mean()) if len(hist) else 0.0)

    st.divider()
    st.write("Recent analyses")
    st.dataframe(hist.head(10), width="stretch")

    if st.session_state.get("last_analysis_meta"):
        meta = st.session_state["last_analysis_meta"]
        st.success(f"Latest analysis saved: {meta['paper_name']} (analysis_id={meta['analysis_id']})")


def render_upload_paper() -> None:
    st.subheader("📤 Upload Paper")
    st.write("Upload a PDF or TXT file. The app will detect headings, split extracted text by section, and generate summaries. 📄✨")

    summary_len = st.select_slider(
        "📏 Summary detail level",
        options=["Short", "Medium", "Long"],
        value="Medium",
        help="If AI summarization isn't available, a heuristic summarizer is used.",
    )

    uploaded = st.file_uploader("Upload PDF/TXT 📥", type=["pdf", "txt"])
    if not uploaded:
        if "last_analysis" in st.session_state and "last_analysis_meta" in st.session_state:
            st.info("ℹ️ You have an active analysis in this session. Upload a new paper above to analyze another.")
            meta = st.session_state["last_analysis_meta"]
            last = st.session_state["last_analysis"]
            render_analysis_results(last["engine"], paper_name=meta["paper_name"], keywords_df=last["keywords"], summary_len=last.get("summary_len", "Medium"))
        return

    paper_name = uploaded.name
    ext = os.path.splitext(paper_name)[1].lower()
    file_bytes = uploaded.getvalue()

    engine = InsightEngine()
    if ext == ".pdf":
        engine.process_pdf_bytes(file_bytes)
        preview_text = engine.full_text
    else:
        try:
            preview_text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            preview_text = str(file_bytes)
        engine.process_text(preview_text)

    if not engine.full_text.strip():
        st.error("❌ Could not extract text from the file.")
        if ext == ".pdf" and not PYMUPDF_AVAILABLE:
            st.info("Tip: install PyMuPDF for better PDF extraction: `pip install pymupdf`")
        return

    st.success(f"✅ Extracted text from **{paper_name}**")

    core_sections = ["Abstract", "Introduction", "Methodology", "Results", "Conclusion"]
    core_rows = []
    for sec in core_sections:
        detected = bool(engine.section_detected_flag.get(sec, False))
        content = (engine.mandatory_map.get(sec) or "")
        core_rows.append(
            {
                "Section": sec,
                "Status": "✅ Detected" if detected else "🔍 Inferred",
                "Content (chars)": len(content),
            }
        )
    c1, c2, c3 = st.columns([1, 1, 2])
    st.markdown("### 📊 Extraction Overview")
    c1.metric("🌐 Domain", engine.domain or "General Research")
    c2.metric("📑 Sections found", len(engine.sections_detected))
    c3.table(pd.DataFrame(core_rows))

    with st.expander("👀 Preview extracted text"):
        st.text_area("Extracted text (first 4000 chars)", value=engine.full_text[:4000], height=220)

    if st.button("🚀 Run analysis", type="primary"):
        user_id = int(st.session_state["auth"]["user_id"])
        progress = st.progress(0, text="Initializing analysis…")

        progress.progress(10, text="Computing metrics…")
        progress.progress(25, text="Detecting sections & generating summaries…")

        with st.spinner("Analyzing paper (this may take a while on first run)…"):
            # We already processed the file into the engine; just derive keywords here.
            kw_df = extract_keywords(engine.full_text, top_k=20)

        progress.progress(80, text="Saving analysis to database…")
        analysis_id = save_analysis(user_id, paper_name, engine, keywords_df=kw_df)

        progress.progress(100, text="Done.")

        st.session_state["last_analysis"] = {"engine": engine, "keywords": kw_df, "summary_len": summary_len}
        st.session_state["last_analysis_meta"] = {"analysis_id": analysis_id, "paper_name": paper_name}

        st.success(f"🎉 Analysis complete and saved (ID:`{analysis_id}`).")
        render_analysis_results(engine, paper_name=paper_name, keywords_df=kw_df, summary_len=summary_len)


def render_analysis_results(engine: "InsightEngine", paper_name: str, keywords_df: Optional[pd.DataFrame] = None, summary_len: str = "Medium") -> None:
    st.subheader("📊 Analysis Results")
    st.caption(f"Paper: **{paper_name}**")

    # Quick TL;DR
    tldr = engine.paper_summaries.get(summary_len) or engine.paper_tldr
    if tldr:
        st.info(f"Summary (TL;DR): {tldr}")

    st.divider()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Composite", f"{engine.scores.get('Composite', 0.0):.2f}")
    c2.metric("Readability", f"{engine.scores.get('Readability', 0.0):.2f}")
    c3.metric("Coherence", f"{engine.scores.get('Coherence', 0.0):.2f}")
    c4.metric("Reasoning", f"{engine.scores.get('Reasoning', 0.0):.2f}")
    c5.metric("Language", f"{engine.scores.get('Language', 0.0):.2f}")
    c6.metric("Sophistication", f"{engine.scores.get('Sophistication', 0.0):.2f}")

    st.divider()

    st.subheader("🔑 Top 20 Keywords (TF-IDF)")
    kw_df = keywords_df if isinstance(keywords_df, pd.DataFrame) else pd.DataFrame(columns=["keyword", "score"])
    st.dataframe(kw_df, width="stretch", hide_index=True)

    st.divider()

    # Analysis tabs (Issues tab removed – not needed for this project)
    tab_sum, tab_detected, tab_missing, tab_suggest, tab_sent = st.tabs(
        ["📝 Section Summaries", "✅ Detected Sections", "⚠️ Missing Sections", "💡 Suggestions", "🎭 Sentiment"]
    )

    with tab_sum:
        st.subheader("Core section summaries")
        st.caption("Each section is either detected from headings or inferred from context.")
        for sec in ["Abstract", "Introduction", "Methodology", "Results", "Conclusion"]:
            detected = engine.section_detected_flag.get(sec, False)
            status = "Detected" if detected else "Inferred"
            summary = (engine.ai_summaries.get(sec, {}) or {}).get(summary_len, "N/A")
            with st.expander(f"{sec} — {status}"):
                st.write(summary or "N/A")

    with tab_detected:
        st.subheader("Validated detected sections")
        st.caption("Only headings with sufficient content are shown; each has an AI/heuristic summary.")
        search_q = st.text_input("Search heading", placeholder="e.g. INTRODUCTION, Methodology, Results")
        if engine.detected_section_summaries:
            for title, sums in engine.detected_section_summaries.items():
                if search_q and search_q.lower() not in title.lower():
                    continue
                with st.expander(title):
                    st.write(f"**Summary ({summary_len})**")
                    st.write(sums.get(summary_len, sums.get("Medium", "N/A")) or "N/A")
                    st.markdown("---")
                    st.caption("Extracted text")
                    st.text_area(
                        f"Extracted text – {title}",
                        value=engine.sections_detected.get(title, ""),
                        height=260,
                    )
        else:
            st.info("No valid headings were detected. (The core sections are still inferred.)")

    with tab_missing:
        st.subheader("Core sections not explicitly detected")
        missing = [s for s in ["Abstract", "Introduction", "Methodology", "Results", "Conclusion"] if not engine.section_detected_flag.get(s, False)]
        if missing:
            for s in missing:
                st.warning(f"**{s}** was not explicitly found in the document headings (it was inferred).")
        else:
            st.success("All core sections were detected from headings.")

    with tab_suggest:
        st.subheader("Vocabulary improvements")
        suggestions = engine._generate_suggestions()
        if suggestions:
            for s in suggestions:
                st.info(s)
        else:
            st.success("No obvious weak-phrase patterns found.")

    with tab_sent:
        st.subheader("Sentiment")
        sentiment = float(engine.sentiment or 0.0)
        if sentiment > 0.05:
            label = "Positive"
        elif sentiment < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        st.metric("Polarity", f"{sentiment:.2f} ({label})")

    st.divider()
    st.subheader("📥 Download")
    insights_text = build_insights_text(engine, paper_name=paper_name, keywords_df=kw_df)
    lines = insights_text.splitlines()
    df_out = pd.DataFrame({"insights": lines})
    st.download_button(
        "Download insights as CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name=f"paperiq_{re.sub(r'[^a-zA-Z0-9_-]+', '_', paper_name)}_insights.csv",
        mime="text/csv",
    )


def render_my_previous_analyses() -> None:
    st.subheader("📂 My Previous Analyses")
    user_id = int(st.session_state["auth"]["user_id"])
    hist = get_user_history(user_id)
    if hist.empty:
        st.info("No analyses yet. Upload a paper to get started.")
        return

    st.dataframe(hist, width="stretch", hide_index=True)
    analysis_id = st.selectbox("🔍 Select an analysis to view", options=hist["analysis_id"].tolist())
    if st.button("👀 View analysis"):
        rec = get_analysis_record(int(analysis_id), user_id=user_id)
        if not rec:
            st.error("Analysis not found.")
            return
        st.subheader(f"Insights for: {rec['paper_name']}")
        st.caption(f"Uploaded: {rec['upload_time']}")
        st.text_area("Saved insights", value=rec["insights_text"], height=420)

        df_out = pd.DataFrame({"insights": rec["insights_text"].splitlines()})
        st.download_button(
            "Download this analysis as CSV",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name=f"paperiq_analysis_{analysis_id}.csv",
            mime="text/csv",
        )


def render_compare_papers() -> None:
    st.subheader("⚖️ Compare Papers")
    st.write("Upload two papers to compute similarity, keyword overlap, and compare AI summaries.")

    c1, c2 = st.columns(2)
    with c1:
        up_a = st.file_uploader("Paper A (PDF/TXT)", type=["pdf", "txt"], key="cmp_a")
    with c2:
        up_b = st.file_uploader("Paper B (PDF/TXT)", type=["pdf", "txt"], key="cmp_b")

    if not up_a or not up_b:
        return

    name_a, text_a = extract_text(up_a)
    name_b, text_b = extract_text(up_b)
    if not text_a or not text_b:
        st.error("Could not extract text from one or both files.")
        return

    if st.button("🔍 Run comparison", type="primary"):
        with st.spinner("Comparing papers…"):
            comp = compare_papers(text_a, text_b)

        st.metric("Cosine similarity (TF-IDF)", f"{comp['similarity']:.3f}")

        st.subheader("🔑 Keyword overlap")
        overlap = comp["keyword_overlap"]
        if overlap:
            st.write(", ".join(overlap))
        else:
            st.info("No overlap detected in top keywords.")

        st.subheader("📝 Summary comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Paper A: **{name_a}**")
            st.write(comp["summary_a"] or "(no summary)")
        with col2:
            st.caption(f"Paper B: **{name_b}**")
            st.write(comp["summary_b"] or "(no summary)")

        st.subheader("📊 Score comparison")
        score_table = []
        for metric in ["Composite", "Readability", "Coherence", "Reasoning", "Language", "Sophistication"]:
            score_table.append({
                "Metric": metric,
                "Paper A": f"{comp['scores_a'].get(metric, 0.0):.2f}",
                "Paper B": f"{comp['scores_b'].get(metric, 0.0):.2f}",
                "Winner": comp["best_by"].get(metric, "-")
            })
        st.table(pd.DataFrame(score_table))

        st.subheader("🏆 Overall verdict")
        if comp["best_overall"] == "Tie":
            st.info("Both papers are equally strong across the computed metrics.")
        else:
            st.success(f"Best paper overall: {comp['best_overall']}")


def render_analytics_dashboard() -> None:
    st.subheader("📈 Analytics Dashboard")
    user_id = int(st.session_state["auth"]["user_id"])
    hist = get_user_history(user_id)
    if hist.empty:
        st.info("No analyses yet. Upload and analyze at least one paper.")
        return

    st.write("Your analysis trends")
    df = hist.copy()
    df["upload_time"] = pd.to_datetime(df["upload_time"].str.replace("Z", "", regex=False), errors="coerce")
    df = df.sort_values("upload_time")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total analyses", int(len(df)))
    c2.metric("Avg readability", float(df["readability_score"].mean()))
    c3.metric("Avg sophistication", float(df["sophistication_score"].mean()))

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df, x="upload_time", y="readability_score", marker="o", ax=ax, label="Readability")
    sns.lineplot(data=df, x="upload_time", y="coherence_score", marker="o", ax=ax, label="Coherence")
    sns.lineplot(data=df, x="upload_time", y="sophistication_score", marker="o", ax=ax, label="Sophistication")
    ax.set_title("Metrics over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.divider()
    st.subheader("📊 Visual analytics for latest analysis")
    pick = st.selectbox("🎯 Choose analysis for visuals", options=hist["analysis_id"].tolist(), index=0)
    rec = get_analysis_record(int(pick), user_id=user_id)
    if not rec:
        st.error("Analysis not found.")
        return

    insights_text = rec["insights_text"]
    text_for_plots = insights_text

    kw_lines = []
    in_kw = False
    for line in insights_text.splitlines():
        if line.strip() == "== Top Keywords ==":
            in_kw = True
            continue
        if in_kw and line.startswith("== "):
            break
        if in_kw and line.strip().startswith("- "):
            kw_lines.append(line.strip()[2:])

    parsed = []
    for l in kw_lines:
        m = re.match(r"^(.*)\s+\(([\d.]+)\)\s*$", l)
        if m:
            parsed.append({"keyword": m.group(1), "score": float(m.group(2))})
    kw_df = pd.DataFrame(parsed)

    st.pyplot(plot_word_freq(text_for_plots), clear_figure=True)
    st.pyplot(plot_sentence_length_distribution(text_for_plots), clear_figure=True)
    st.pyplot(plot_keyword_importance(kw_df), clear_figure=True)
    st.pyplot(
        plot_readability_metrics(
            {
                "readability_score": float(rec["readability_score"]),
                "coherence_score": float(rec["coherence_score"]),
                "sophistication_score": float(rec["sophistication_score"]),
            }
        ),
        clear_figure=True,
    )


def render_admin_dashboard() -> None:
    st.subheader("📊 Admin Dashboard")
    db = get_mongo_db()
    analysis_col = _get_analysis_col(db)
    users_col = _get_users_col(db)

    analysis_docs = list(
        analysis_col.find(
            {},
            {
                "_id": 0,
                "analysis_id": 1,
                "user_id": 1,
                "paper_name": 1,
                "upload_time": 1,
                "total_words": 1,
                "readability_score": 1,
                "coherence_score": 1,
                "sophistication_score": 1,
            },
        ).sort("analysis_id", -1)
    )
    analyses_df = pd.DataFrame(analysis_docs)
    if analyses_df.empty:
        analyses_df = pd.DataFrame(
            columns=[
                "analysis_id",
                "user_id",
                "paper_name",
                "upload_time",
                "total_words",
                "readability_score",
                "coherence_score",
                "sophistication_score",
                "username",
            ]
        )
    else:
        user_ids = analyses_df["user_id"].dropna().unique().tolist()
        users_docs = list(users_col.find({"user_id": {"$in": user_ids}}, {"_id": 0, "user_id": 1, "username": 1}))
        users_df = pd.DataFrame(users_docs)
        analyses_df = analyses_df.merge(users_df, on="user_id", how="left")

    c1, c2 = st.columns(2)
    c1.metric("Total analyses", int(len(analyses_df)))
    c2.metric("Unique active users", int(analyses_df["username"].nunique()) if not analyses_df.empty else 0)

    st.divider()
    st.write("Analyses by user")
    st.dataframe(
        analyses_df[["analysis_id", "username", "paper_name", "upload_time", "total_words", "readability_score", "coherence_score", "sophistication_score"]].head(50),
        width="stretch",
        hide_index=True,
    )


def render_admin_users() -> None:
    st.subheader("👥 Registered Users")
    db = get_mongo_db()
    users_col = _get_users_col(db)
    users_docs = list(users_col.find({}, {"_id": 0, "user_id": 1, "username": 1, "role": 1}).sort("user_id", 1))
    users_df = pd.DataFrame(users_docs)
    st.dataframe(users_df, width="stretch", hide_index=True)


def render_admin_analyses() -> None:
    st.subheader("📂 All Analyses")
    db = get_mongo_db()
    analysis_col = _get_analysis_col(db)
    users_col = _get_users_col(db)

    analysis_docs = list(
        analysis_col.find(
            {},
            {
                "_id": 0,
                "analysis_id": 1,
                "user_id": 1,
                "paper_name": 1,
                "upload_time": 1,
                "total_words": 1,
                "extracted_words": 1,
                "readability_score": 1,
                "coherence_score": 1,
                "sophistication_score": 1,
            },
        ).sort("analysis_id", -1)
    )
    analyses_df = pd.DataFrame(analysis_docs)
    if not analyses_df.empty:
        user_ids = analyses_df["user_id"].dropna().unique().tolist()
        users_docs = list(users_col.find({"user_id": {"$in": user_ids}}, {"_id": 0, "user_id": 1, "username": 1, "role": 1}))
        users_df = pd.DataFrame(users_docs)
        analyses_df = analyses_df.merge(users_df, on="user_id", how="left")

    if analyses_df.empty:
        analyses_df = pd.DataFrame(
            columns=[
                "analysis_id",
                "user_id",
                "username",
                "role",
                "paper_name",
                "upload_time",
                "total_words",
                "extracted_words",
                "readability_score",
                "coherence_score",
                "sophistication_score",
            ]
        )

    st.dataframe(analyses_df, width="stretch", hide_index=True)


def render_system_stats() -> None:
    st.subheader("⚙️ System Usage Statistics")
    db = get_mongo_db()
    analysis_col = _get_analysis_col(db)
    users_col = _get_users_col(db)

    analysis_docs = list(
        analysis_col.find(
            {},
            {
                "_id": 0,
                "user_id": 1,
                "upload_time": 1,
                "total_words": 1,
                "readability_score": 1,
                "coherence_score": 1,
                "sophistication_score": 1,
            },
        )
    )
    analyses_df = pd.DataFrame(analysis_docs)
    if analyses_df.empty:
        st.info("No system activity yet.")
        return

    user_ids = analyses_df["user_id"].dropna().unique().tolist()
    users_docs = list(users_col.find({"user_id": {"$in": user_ids}}, {"_id": 0, "user_id": 1, "username": 1}))
    users_df = pd.DataFrame(users_docs)
    analyses_df = analyses_df.merge(users_df, on="user_id", how="left")

    if analyses_df.empty:
        st.info("No system activity yet.")
        return

    analyses_df["upload_time"] = pd.to_datetime(analyses_df["upload_time"].str.replace("Z", "", regex=False), errors="coerce")

    st.write("Usage by user")
    usage = analyses_df.groupby("username").size().reset_index(name="analyses")
    st.dataframe(usage.sort_values("analyses", ascending=False), width="stretch", hide_index=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    daily = analyses_df.set_index("upload_time").resample("D").size().reset_index(name="analyses")
    sns.barplot(data=daily, x="upload_time", y="analyses", ax=ax, color="#4c72b0")
    ax.set_title("Analyses per day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.divider()
    st.write("Metric distributions")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(analyses_df["readability_score"], bins=20, ax=axes[0], color="#55a868")
    axes[0].set_title("Readability")
    sns.histplot(analyses_df["coherence_score"], bins=20, ax=axes[1], color="#c44e52")
    axes[1].set_title("Coherence")
    sns.histplot(analyses_df["sophistication_score"], bins=20, ax=axes[2], color="#8172b2")
    axes[2].set_title("Sophistication")
    for ax in axes:
        ax.set_xlabel("Score")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    sns.set_theme(style="whitegrid")

    _init_nltk()
    try:
        create_database()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return

    auth = st.session_state.get("auth")
    if not auth:
        render_login()
        return

    role = auth.get("role")
    if role == "admin":
        page = render_admin_sidebar()
        if page == "Logout":
            logout()
            st.rerun()
        elif page == "Admin Dashboard":
            render_admin_dashboard()
        elif page == "Users":
            render_admin_users()
        elif page == "All Analyses":
            render_admin_analyses()
        elif page == "System Stats":
            render_system_stats()
        return

    # User role
    page = render_user_sidebar()
    if page == "Logout":
        logout()
        st.rerun()
    elif page == "Dashboard":
        render_user_dashboard()
    elif page == "Upload Paper":
        render_upload_paper()
    elif page == "Compare Papers":
        render_compare_papers()
    elif page == "My Previous Analyses":
        render_my_previous_analyses()
    elif page == "Analytics Dashboard":
        render_analytics_dashboard()
    elif page == "Chat with Paper":
        render_chat_with_paper()


# Streamlit runs the script directly; call main() so the UI renders.
main()

