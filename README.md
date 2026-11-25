# AI Resume Analyzer — Emerald Obsidian Edition

An intelligent, interpretable resume analysis, ATS optimization, and empirical benchmarking system that runs 100% locally.

---

## Quick Start

### Option 1: Web Interface (Recommended)
Start the local web server (auto-selects an open port starting at `8501` and opens your browser):
```bash
python3 app.py
```
Or open `index.html` directly in your browser.

### Option 2: Streamlit Interface
```bash
streamlit run app.py
```

### Option 3: CLI & Dataset Benchmark Modes
Analyze a single resume against a job description:
```bash
python3 app.py --cli sample_data/resume.txt sample_data/job_description.txt
python3 app.py --cli sample_data/resume.txt sample_data/job_description.txt --json
```
Benchmark all 270 resumes in `sample_data/sample_resumes.csv` against a job description:
```bash
python3 app.py --benchmark sample_data/job_description.txt
```

### Option 4: Run Unit & Integration Tests
```bash
python3 tests/test_analyzer.py
```

---

## Key Features

- **Emerald & Obsidian Editorial UI:** High-contrast dark obsidian-and-emerald theme with a persistent Sage Light Mode toggle (`localStorage`), Fraunces serif typography, and clean `@media print` PDF export styles.
- **Multi-Format Ingestion:** Client-side and Python parsing of **PDF**, **DOCX**, and **TXT** files, including drag-and-drop upload in the browser.
- **5-Dimensional Multi-Attribute Scoring:**
  - **Skill Fit (40%):** Recall over a 43-skill canonical taxonomy with regex alias normalization.
  - **Text Match (20%):** Smoothed TF-IDF cosine similarity.
  - **ATS Structure (15%):** Detection of 6 core sections (Contact, Education, Experience, Skills, Projects, Achievements).
  - **Impact (15%):** Word-boundary matching of 21 action verbs + quantified achievement metrics (`%`, `$`, `k`, `x`, `+`).
  - **Readability (10%):** Flesch Reading Ease index.
- **What-If Skill Simulator:** Click any missing skill chip in the UI to simulate adding it and preview the exact score boost.
- **Pin as Baseline (Before/After Comparison):** Pin a baseline score, edit your resume, and re-run to track exact point deltas.
- **Impact & Keyword Gap Audit:** Inspect detected action verbs, quantified metrics, and top missing JD keywords ranked by TF-IDF frequency.
- **Live 270-Resume CSV Benchmarker:** Evaluate `sample_data/sample_resumes.csv` (or any custom CSV) against the active JD, view domain-level averages, click top candidates to inspect them, and export results to `.csv`.
- **7-Role Affinity Projection:** Instant fit percentages across *Data Scientist, Data Analyst, ML Engineer, Data Engineer, Software Engineer, Web Developer,* and *Business Analyst*.
- **Exports:** Download `.txt` reports, `.json` structured analysis, or print to PDF.

---

## Project Deliverables & Documentation

- **Mini-Project Report (PDF):** [Mini_Project_Report_CSET312.pdf](Mini_Project_Report_CSET312.pdf)
- **Mini-Project Report (Word):** [Mini_Project_Report_CSET312.docx](Mini_Project_Report_CSET312.docx)
- **Presentation Slides:** [AI_Resume_Analyzer_Presentation.pptx](AI_Resume_Analyzer_Presentation.pptx)
- **Interactive Rubric:** Click **Evaluation Rubric · 12 Marks** in the web header of `index.html`.
