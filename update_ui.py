import os
import re

file_path = "app.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace Login
content = content.replace(
    'st.title(APP_TITLE)',
    'st.title(f"📚 {APP_TITLE}")'
).replace(
    'st.caption("Analyze research papers quickly with AI summaries, insights, and analytics.")',
    'st.caption("✨ Analyze research papers quickly with AI summaries, insights, and analytics.")'
).replace(
    'st.tabs(["User Login", "User Signup", "Admin Login"])',
    'st.tabs(["🔐 User Login", "📝 User Signup", "🛠️ Admin Login"])'
).replace(
    'st.subheader("User Login")',
    'st.subheader("👋 User Login")'
).replace(
    'st.text_input("Username")',
    'st.text_input("Username 👤")'
).replace(
    'st.text_input("Password", type="password")',
    'st.text_input("Password 🔑", type="password")'
).replace(
    'st.form_submit_button("Login")',
    'st.form_submit_button("Login ➡️")'
).replace(
    'st.subheader("Create Account")',
    'st.subheader("✨ Create Account")'
).replace(
    'st.text_input("Choose a username")',
    'st.text_input("Choose a username 👤")'
).replace(
    'st.text_input("Choose a password", type="password")',
    'st.text_input("Choose a password 🔑", type="password")'
).replace(
    'st.text_input("Confirm password", type="password")',
    'st.text_input("Confirm password 🔑", type="password")'
).replace(
    'st.form_submit_button("Sign up")',
    'st.form_submit_button("Sign up 🚀")'
).replace(
    'st.subheader("Admin Login")',
    'st.subheader("🛠️ Admin Login")'
).replace(
    'st.text_input("Admin username")',
    'st.text_input("Admin username 👤")'
).replace(
    'st.text_input("Admin password", type="password")',
    'st.text_input("Admin password 🔑", type="password")'
).replace(
    'st.form_submit_button("Login as admin")',
    'st.form_submit_button("Login as admin ➡️")'
)

# User Sidebar
content = content.replace(
    'st.sidebar.title("PaperIQ")',
    'st.sidebar.title("🧠 PaperIQ")'
).replace(
    'st.sidebar.caption(f"Signed in as **{st.session_state[\'auth\'][\'username\']}**")',
    'st.sidebar.caption(f"👤 Signed in as **{st.session_state[\'auth\'][\'username\']}**")'
)
old_options = '''    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Upload Paper", "Compare Papers", "My Previous Analyses", "Chat with Paper", "Analytics Dashboard", "Logout"],
        index=0,
    )
    return page'''
new_options = '''    options = {
        "🏠 Dashboard": "Dashboard",
        "📤 Upload Paper": "Upload Paper",
        "⚖️ Compare Papers": "Compare Papers",
        "📂 My Previous Analyses": "My Previous Analyses",
        "💬 Chat with Paper": "Chat with Paper",
        "📈 Analytics Dashboard": "Analytics Dashboard",
        "🚪 Logout": "Logout"
    }
    page = st.sidebar.radio("Navigation", list(options.keys()), index=0)
    return options.get(page, "Dashboard")'''
content = content.replace(old_options, new_options)

# Admin Sidebar
content = content.replace(
    'st.sidebar.title("PaperIQ (Admin)")',
    'st.sidebar.title("🛠️ PaperIQ (Admin)")'
)
old_admin_opt = '''    page = st.sidebar.radio(
        "Navigation",
        ["Admin Dashboard", "Users", "All Analyses", "System Stats", "Logout"],
        index=0,
    )
    return page'''
new_admin_opt = '''    options = {
        "📊 Admin Dashboard": "Admin Dashboard",
        "👥 Users": "Users",
        "📂 All Analyses": "All Analyses",
        "⚙️ System Stats": "System Stats",
        "🚪 Logout": "Logout"
    }
    page = st.sidebar.radio("Navigation", list(options.keys()), index=0)
    return options.get(page, "Admin Dashboard")'''
content = content.replace(old_admin_opt, new_admin_opt)

# Dashboard
content = content.replace(
    'st.subheader("Dashboard")',
    'st.subheader("🏠 Dashboard")'
).replace(
    'st.write("Upload papers for section-wise AI summaries, keyword extraction, and actionable insights.")',
    'st.write("Welcome! Upload papers for section-wise AI summaries, keyword extraction, and actionable insights. ✨")'
)

# Upload Paper
content = content.replace(
    'st.subheader("Upload Paper")',
    'st.subheader("📤 Upload Paper")'
).replace(
    'st.write("Upload a PDF or TXT file. The app will detect headings, split extracted text by section, and generate summaries.")',
    'st.write("Upload a PDF or TXT file. The app will detect headings, split extracted text by section, and generate summaries. 📄✨")'
).replace(
    '"Summary detail level"',
    '"📏 Summary detail level"'
).replace(
    '"Upload PDF/TXT"',
    '"Upload PDF/TXT 📥"'
).replace(
    'st.info(f"Extracted text from **{paper_name}**")',
    'st.success(f"✅ Extracted text from **{paper_name}**")'
).replace(
    'st.error("Could not extract text from the file.")',
    'st.error("❌ Could not extract text from the file.")'
).replace(
    '"Status": "Detected" if detected else "Inferred",',
    '"Status": "✅ Detected" if detected else "🔍 Inferred",'
).replace(
    'c1.metric("Domain", engine.domain or "General Research")',
    'st.markdown("### 📊 Extraction Overview")\\n    c1.metric("🌐 Domain", engine.domain or "General Research")'
).replace(
    'c2.metric("Sections found", len(engine.sections_detected))',
    'c2.metric("📑 Sections found", len(engine.sections_detected))'
).replace(
    '"Preview extracted text"',
    '"👀 Preview extracted text"'
).replace(
    'st.button("Run analysis", type="primary")',
    'st.button("🚀 Run analysis", type="primary")'
).replace(
    'st.success(f"Analysis complete and saved (analysis_id={analysis_id}).")',
    'st.success(f"🎉 Analysis complete and saved (ID:`{analysis_id}`).")'
)

# Analysis Results
content = content.replace(
    'st.subheader("Analysis Results")',
    'st.subheader("📊 Analysis Results")'
).replace(
    'st.subheader("Top 20 Keywords (TF-IDF)")',
    'st.subheader("🔑 Top 20 Keywords (TF-IDF)")'
).replace(
    '["Section Summaries", "Detected Sections", "Missing Sections", "Suggestions", "Sentiment"]',
    '["📝 Section Summaries", "✅ Detected Sections", "⚠️ Missing Sections", "💡 Suggestions", "🎭 Sentiment"]'
).replace(
    'st.subheader("Download")',
    'st.subheader("📥 Download")'
).replace(
    'st.download_button(\\n        "Download insights as CSV"',
    'st.download_button(\\n        "💾 Download insights as CSV"'
)

# Previous Analyses
content = content.replace(
    'st.subheader("My Previous Analyses")',
    'st.subheader("📂 My Previous Analyses")'
).replace(
    'st.selectbox("Select an analysis to view",',
    'st.selectbox("🔍 Select an analysis to view",'
).replace(
    'st.button("View analysis")',
    'st.button("👀 View analysis")'
).replace(
    'st.download_button(\\n            "Download this analysis as CSV"',
    'st.download_button(\\n            "💾 Download this analysis as CSV"'
)

# Compare papers
content = content.replace(
    'st.subheader("Compare Papers")',
    'st.subheader("⚖️ Compare Papers")'
).replace(
    'st.button("Run comparison", type="primary")',
    'st.button("🔍 Run comparison", type="primary")'
).replace(
    'st.subheader("Keyword overlap")',
    'st.subheader("🔑 Keyword overlap")'
).replace(
    'st.subheader("Summary comparison")',
    'st.subheader("📝 Summary comparison")'
).replace(
    'st.subheader("Score comparison")',
    'st.subheader("📊 Score comparison")'
).replace(
    'st.subheader("Overall verdict")',
    'st.subheader("🏆 Overall verdict")'
)

# Analytics dashboard
content = content.replace(
    'st.subheader("Analytics Dashboard")',
    'st.subheader("📈 Analytics Dashboard")'
).replace(
    'st.subheader("Visual analytics for latest analysis")',
    'st.subheader("📊 Visual analytics for latest analysis")'
).replace(
    'st.selectbox("Choose analysis for visuals"',
    'st.selectbox("🎯 Choose analysis for visuals"'
)

# Admin areas
content = content.replace(
    'st.subheader("Admin Dashboard")',
    'st.subheader("📊 Admin Dashboard")'
).replace(
    'st.subheader("Registered Users")',
    'st.subheader("👥 Registered Users")'
).replace(
    'st.subheader("All Analyses")',
    'st.subheader("📂 All Analyses")'
).replace(
    'st.subheader("System Usage Statistics")',
    'st.subheader("⚙️ System Usage Statistics")'
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated basic emojis in app.py!")
