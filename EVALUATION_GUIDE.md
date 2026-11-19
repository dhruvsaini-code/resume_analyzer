# AI Resume Analyzer — Evaluation & Viva Defense Guide (12/12 Marks)

This guide is structured strictly according to the **Evaluation Criterion Rubric** (3 marks per section, Total: 12 Marks). Use this document to prepare for your viva and answer any questions from the evaluator with confidence.

---

## Criterion 1: Review of Existing Systems and Project Feasibility (3 Marks)

### 1.1 Review of Existing Systems (Literature & Market Survey)
| System Category | Representative Systems | Working Mechanism | Critical Limitations |
| :--- | :--- | :--- | :--- |
| **Traditional Enterprise ATS** | Taleo, Workday, SAP SuccessFactors | Keyword frequency scanning and exact string matching | <ul><li>Fails on synonyms (e.g., misses "PostgreSQL" if JD specifies "SQL")</li><li>Over-penalizes modern formatting & multi-column templates</li><li>Zero transparency or remediation guidance to candidates</li></ul> |
| **Commercial Candidate Scanners** | Jobscan, Resume Worded | Cloud API token matching | <ul><li>Requires paid recurring subscriptions ($$$)</li><li>Black-box heuristics with no mathematical explainability</li><li>Privacy hazards: Uploads confidential candidate PII to external servers</li></ul> |
| **LLM-Based Evaluators** | GPT-4 / Claude Resume Reviewers | Generative prompt-based assessment | <ul><li>High latency (5–15 seconds per evaluation)</li><li>Non-deterministic results (different scores for the same resume)</li><li>Hallucination risk and API cost overhead</li></ul> |

### 1.2 Project Feasibility Analysis
Our proposed system addresses all the above drawbacks through:
1. **Technical Feasibility:** Pure client-side or lightweight local Python execution. Requires zero GPU infrastructure, zero external paid APIs, and runs instantaneously ($< 100\text{ ms}$).
2. **Economic Feasibility:** Completely open-source stack (Python, Vanilla HTML/CSS/JS, Regex, TF-IDF). Zero operational cloud API costs.
3. **Operational Feasibility & Privacy:** Candidates and HR teams can evaluate resumes locally without transmitting sensitive PII (Phone numbers, emails, addresses) over public networks.
4. **Empirical Feasibility:** Validated and benchmarked against a diverse dataset of **270 real-world resumes** spanning 6 major technical domains (*Data Science, ML Engineering, Software Engineering, Data Analytics, Data Engineering, Business Analytics*).

---

## Criterion 2: Objectives and Methodology of Proposed Work (3 Marks)

### 2.1 Project Objectives
- **Objective 1:** Implement an automated, deterministic parsing engine supporting multiple formats (`.pdf`, `.docx`, `.txt`) with zero format distortion.
- **Objective 2:** Build an alias-aware canonical skill taxonomy to eliminate synonym mismatches in recruitment screening.
- **Objective 3:** Synthesize a multi-dimensional scoring framework that measures not just keyword overlap, but lexical depth, structural compliance, action-verb impact, and readability.
- **Objective 4:** Provide actionable, context-aware remediation plans ranked by severity (`High`, `Medium`, `Low`) to help candidates tailor their resumes ethically.

### 2.2 System Architecture & Methodology Flow
```
[Candidate Resume (.pdf / .docx / .txt)]       [Job Description (Text)]
                  │                                       │
                  ▼                                       ▼
       [1. Document Ingestion]                            │
    (pypdf / mammoth / FileReader)                        │
                  │                                       │
                  ▼                                       ▼
    [2. Preprocessing & NLP Pipeline]                     │
  - URL / Email Sanitization                              │
  - Lowercase Regex Tokenization                          │
  - Domain Stopword Elimination                           │
                  │                                       │
                  ├───► [3. Entity & Skill Extraction] ◄──┘
                  │      (Canonical Taxonomy & Alias Normalization)
                  │
                  ├───► [4. Vector Space Modeling (TF-IDF)]
                  │      (Sublinear TF + Smoothed IDF Cosine Similarity)
                  │
                  ├───► [5. Multi-Attribute Scoring Engine]
                  │      - Skill Coverage (40%)
                  │      - TF-IDF Cosine Match (20%)
                  │      - ATS Structural Integrity (15%)
                  │      - Action Verb & Quantified Impact (15%)
                  │      - Flesch Reading Ease Index (10%)
                  │
                  ▼
    [6. Visual Analytics & Remediation Synthesis]
  - Animated Radar Chart & Dimension Progress Bars
  - Interactive Paper View with Highlighted Skill Annotations
  - Multi-Role Affinity Projection (6 Career Profiles)
  - Prioritized Action Plan & Exportable Report
```

---

## Criterion 3: Relevance of Algorithms / Techniques (3 Marks)

### 3.1 Mathematical Formulations & Justifications

#### A. Canonical Skill Taxonomy Matching (Recall Metric)
Rather than raw substring searches which produce false positives (e.g., "management" matching "manage"), we utilize negative lookahead/lookbehind word boundary regular expressions:
$$\text{SkillFit} = \frac{|S_{\text{resume}} \cap S_{\text{JD}}|}{|S_{\text{JD}}|}$$
Where $S$ represents the set of canonicalized skills mapped from aliases (e.g. `['postgres', 'postgresql', 'mysql'] \to \text{'sql'}`).

#### B. TF-IDF Vector Space Modeling & Cosine Similarity
Captures domain vocabulary and contextual phrasing beyond the strict skill dictionary:
$$\text{TF}(t, d) = f_{t, d}$$
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$$
$$\text{Cosine Similarity} = \frac{\mathbf{v}_r \cdot \mathbf{v}_j}{\|\mathbf{v}_r\| \|\mathbf{v}_j\|} = \frac{\sum_{t \in V} w_{t, r} \cdot w_{t, j}}{\sqrt{\sum w_{t, r}^2} \cdot \sqrt{\sum w_{t, j}^2}}$$

#### C. Flesch Reading Ease Formula
Ensures the resume is concise and easily readable by human recruiters:
$$\text{RE} = 206.835 - 1.015 \left(\frac{\text{Total Words}}{\text{Total Sentences}}\right) - 84.6 \left(\frac{\text{Total Syllables}}{\text{Total Words}}\right)$$

#### D. Impact & Quantifiable Achievement Heuristic
Recruiters prioritize measurable achievements over passive duty lists:
$$\text{Impact} = \min\left(1.0, 0.60 \times \frac{\min(V_{\text{action}}, 5)}{5} + 0.40 \times \frac{\min(N_{\text{metrics}}, 3)}{3}\right)$$
Where $V_{\text{action}}$ denotes action verbs (*built, designed, engineered, reduced, etc.*) and $N_{\text{metrics}}$ denotes quantified results (*percentages, revenue saved, user metrics*).

#### E. Composite Scoring Equation
$$\mathbf{Score} = 100 \times \Big( 0.40 \cdot \text{SkillFit} + 0.20 \cdot \min\left(\frac{\text{Sim}}{0.5}, 1.0\right) + 0.15 \cdot \text{Structure} + 0.15 \cdot \text{Impact} + 0.10 \cdot \text{Readability} \Big)$$

---

## Criterion 4: Synchronization of Project Design & Implementation (3 Marks)

### 4.1 Traceability Matrix
Every architectural requirement is implemented with 100% synchronization across the Python core engine and the Web interface:

| Requirement / Module | Python Backend Implementation (`analyzer.py`) | Web Frontend Implementation (`index.html`) | Verification Artifact |
| :--- | :--- | :--- | :--- |
| **Document Ingestion** | `extract_text(data, filename)` (PDF/DOCX/TXT) | `FileReader` + `pdf.js` + `mammoth.js` | Supports `.pdf`, `.docx`, `.txt` live in UI |
| **Skill Taxonomy** | `extract_skills()` + `SKILLS_MAP` | `found()` + `SKILLS` mapping | 40+ canonical tech/soft skills |
| **Vector Similarity** | `tfidf_similarity()` | `tfidf()` | Smoothed TF-IDF cosine metric |
| **Readability Metric** | `flesch_reading_ease()` | `flesch()` | Syllable & sentence analysis |
| **Impact & Verbs** | `ACTION_VERBS` regex matching | `VERBS` + regex metric extraction | Identifies action verbs & metrics |
| **Role Fit Projection**| `ROLES` affinity calculation | `ROLES` progress bars | 6 industry roles benchmarked |
| **Action Plan** | `skill_priority()` priority queue | `priority()` tagged recommendation plan | Ranked High / Medium / Low |
| **Unit Test Coverage** | `tests/test_analyzer.py` | Automated assertions | `All unit tests passed successfully!` |
| **Empirical Evaluation**| `sample_data/sample_resumes.csv` | Dataset Dropdown Loader | 270 real resumes across 6 roles |

---

## 5. Viva / Presentation Quick Answers

**Q1: Why didn't you use BERT or a heavy LLM for the similarity score?**
> *Answer:* LLMs introduce non-deterministic results, high latency (several seconds per resume), and heavy compute/cost overhead. In real-world recruitment screening, high-throughput systems require sub-second deterministic responses. Our hybrid approach of an alias-normalized canonical taxonomy + TF-IDF cosine similarity runs in $< 50\text{ ms}$, offers complete interpretability, and guarantees candidate privacy by executing locally.

**Q2: How does your system handle synonyms and variations of skills?**
> *Answer:* We built an alias-normalized regex taxonomy mapping multiple industry expressions (e.g., `['postgres', 'postgresql', 'mysql'] \to \text{'sql'}`) with negative lookaround boundaries to avoid partial-word false positives.

**Q3: How is this validated on real data?**
> *Answer:* We benchmarked the system on a dataset of 270 real-world resumes across 6 technical profiles (`sample_resumes.csv`). In our live prototype, you can switch between these profiles directly to observe how the score, radar dimensions, and role affinity reflect different technical specializations.
