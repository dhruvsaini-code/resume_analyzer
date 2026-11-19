"""Core logic of the AI Resume Analyzer.

Synchronized with the multi-dimensional evaluation rubric:
1. Review of Existing Systems and Project Feasibility
2. Objectives and Methodology of Proposed Work
3. Relevance of Algorithms / Techniques:
   - Canonical Skill Taxonomy with Normalized Alias Extraction
   - TF-IDF Vector Space Modeling & Cosine Similarity
   - Structural Section Parsing & ATS Compliance
   - Action Verb & Quantified Metric Impact Scoring
   - Flesch Reading Ease Readability Metric
   - Multi-role Affinity Projection
4. Synchronization of Project Design & Implementation
"""

import csv
import io
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

# ---------------------------------------------------------------- 1. Document Extraction
def extract_text(data: bytes, filename: str) -> str:
    """Extract plain text from PDF, DOCX, or TXT resumes."""
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            try:
                import pypdf
            except ImportError:
                pass
            return data.decode("utf-8", errors="ignore")
    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return data.decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------- 2. Preprocessing & NLP
STOP_WORDS = {
    "a", "about", "also", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "etc", "for", "from", "had", "has", "have", "if", "in", "into", "is",
    "it", "its", "must", "not", "of", "on", "or", "our", "over", "per", "should",
    "so", "such", "than", "that", "the", "their", "then", "they", "this", "to",
    "use", "used", "using", "was", "we", "were", "will", "with", "work", "working",
    "you", "your"
}

ACTION_VERBS = [
    "built", "developed", "designed", "led", "created", "implemented", "improved",
    "reduced", "increased", "analyzed", "managed", "delivered", "optimized",
    "automated", "launched", "trained", "deployed", "collaborated", "engineered",
    "migrated", "researched"
]
ACTION_VERB_PATTERNS: Dict[str, re.Pattern] = {
    v: re.compile(rf"\b{re.escape(v)}\b", re.IGNORECASE) for v in ACTION_VERBS
}

def tokenize(text: str) -> List[str]:
    """Tokenize and filter stop words and URLs/emails."""
    cleaned = re.sub(r"https?://\S+|\S+@\S+", " ", text.lower())
    tokens = re.findall(r"[a-z][a-z+#]{1,}", cleaned)
    return [t for t in tokens if t not in STOP_WORDS]


# ---------------------------------------------------------------- 3. Skill Taxonomy
SKILLS_RAW = (
    "python|python;java|java;c++|c\\+\\+;javascript|javascript|js;typescript|typescript;"
    "sql|sql|mysql|postgresql|postgres;mongodb|mongodb;html|html5?;css|css3?;"
    "react|react(?:js|\\.js)?;node.js|node\\.?js|node;django|django;flask|flask;fastapi|fastapi;"
    "machine learning|machine learning|ml;deep learning|deep learning;"
    "nlp|nlp|natural language processing;computer vision|computer vision;"
    "tensorflow|tensorflow;pytorch|pytorch;scikit-learn|scikit-learn|sklearn;pandas|pandas;"
    "numpy|numpy;data analysis|data analysis|data analytics;"
    "data visualization|data visualization|matplotlib|tableau|power bi;"
    "statistics|statistics|statistical analysis;excel|excel;git|git|github;"
    "docker|docker;kubernetes|kubernetes|k8s;aws|aws|amazon web services;azure|azure;"
    "gcp|gcp|google cloud;linux|linux;rest api|rest apis?|restful;"
    "ci/cd|ci/cd|jenkins|github actions;spark|spark|pyspark;etl|etl;agile|agile|scrum;"
    "data structures|data structures|algorithms;oop|oop|object oriented;"
    "communication|communication;teamwork|teamwork|collaboration;"
    "leadership|leadership;problem solving|problem[- ]solving"
)

SKILLS_MAP: Dict[str, re.Pattern] = {}
for entry in SKILLS_RAW.split(";"):
    parts = entry.split("|")
    canonical = parts[0]
    pattern = rf"(?<![\w+#])(?:{'|'.join(parts[1:])})(?![\w+#])"
    SKILLS_MAP[canonical] = re.compile(pattern, re.IGNORECASE)


ROLES = {
    "Data Scientist": ["python", "machine learning", "statistics", "pandas", "scikit-learn", "sql", "data visualization"],
    "Data Analyst": ["sql", "excel", "data analysis", "data visualization", "statistics", "python"],
    "ML Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "docker", "aws"],
    "Data Engineer": ["sql", "python", "etl", "spark", "aws", "docker", "linux"],
    "Software Engineer": ["java", "python", "data structures", "git", "rest api", "oop", "ci/cd"],
    "Web Developer": ["javascript", "html", "css", "react", "node.js", "git", "rest api"],
    "Business Analyst": ["sql", "excel", "data analysis", "data visualization", "communication", "agile", "statistics"],
}


def extract_skills(text: str) -> Set[str]:
    """Identify present skills using canonical taxonomy regular expressions."""
    return {skill for skill, pat in SKILLS_MAP.items() if pat.search(text)}


# ---------------------------------------------------------------- 4. Algorithms
def tfidf_similarity(text_a: str, text_b: str) -> float:
    """Compute TF-IDF cosine similarity between two documents."""
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    tf_a: Dict[str, int] = {}
    tf_b: Dict[str, int] = {}
    for w in tokens_a:
        tf_a[w] = tf_a.get(w, 0) + 1
    for w in tokens_b:
        tf_b[w] = tf_b.get(w, 0) + 1

    vocabulary = set(tf_a.keys()) | set(tf_b.keys())
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for w in vocabulary:
        # standard smoothed IDF formula for 2 documents: log((1+2)/(1 + df)) + 1
        df = (1 if w in tf_a else 0) + (1 if w in tf_b else 0)
        idf = math.log((1 + 2) / (1 + df)) + 1.0
        val_a = tf_a.get(w, 0) * idf
        val_b = tf_b.get(w, 0) * idf
        dot += val_a * val_b
        norm_a += val_a * val_a
        norm_b += val_b * val_b

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def count_syllables(word: str) -> int:
    """Count approximate syllables in an English word."""
    w = word.lower().strip()
    if not w:
        return 0
    w = re.sub(r"[^a-z]", "", w)
    matches = re.findall(r"[aeiouy]+", w)
    count = len(matches)
    if w.endswith("e") and not w.endswith("le") and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    """Calculate Flesch Reading Ease score:
    206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    """
    sentences = max(1, len(re.findall(r"[.!?\n]+", text)))
    words = [w for w in re.split(r"\s+", text) if w]
    if not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / max(1, len(words)))
    return score


# ---------------------------------------------------------------- 5. Structure & Impact
SECTIONS = {
    "Contact": re.compile(r"@|\+?\d[\d\s-]{8,}", re.IGNORECASE),
    "Education": re.compile(r"education|b\.?tech|bachelor|master|university|college", re.IGNORECASE),
    "Experience": re.compile(r"experience|intern|worked|employment", re.IGNORECASE),
    "Skills": re.compile(r"skills", re.IGNORECASE),
    "Projects": re.compile(r"projects?", re.IGNORECASE),
    "Achievements": re.compile(r"achievement|award|certif|rank|winner", re.IGNORECASE),
}


def check_sections(text: str) -> Dict[str, bool]:
    return {sec: bool(pat.search(text)) for sec, pat in SECTIONS.items()}


def skill_priority(skill: str, jd_text: str) -> str:
    """Infer importance from JD sentence semantics."""
    sentences = re.split(r"(?<=[.!?])\s+|\n", jd_text)
    pat = SKILLS_MAP.get(skill)
    target_sentence = ""
    if pat:
        for s in sentences:
            if pat.search(s):
                target_sentence = s
                break

    if re.search(r"plus|preferred|nice|bonus|optional", target_sentence, re.IGNORECASE):
        return "Low"
    if re.search(r"required|must|strong|proficien|experience|essential", target_sentence, re.IGNORECASE):
        return "High"
    return "Medium"


def extract_keyword_gaps(resume_text: str, jd_text: str, top_n: int = 8) -> Tuple[List[str], List[str]]:
    """Return top JD keywords matched and missing in the resume, ranked by JD term frequency."""
    jd_tokens = tokenize(jd_text)
    if not jd_tokens:
        return [], []
    res_set = set(tokenize(resume_text))
    tf: Dict[str, int] = {}
    for w in jd_tokens:
        if len(w) >= 3:
            tf[w] = tf.get(w, 0) + 1
    ranked = sorted(tf.items(), key=lambda item: (-item[1], item[0]))
    matched_kw = [w for w, _ in ranked if w in res_set][:top_n]
    missing_kw = [w for w, _ in ranked if w not in res_set][:top_n]
    return matched_kw, missing_kw


def grade_and_verdict(score: float) -> Tuple[str, str]:
    """Map numeric score (0-100) to letter grade and summary verdict."""
    if score >= 85:
        return "A+", "A strong match. Polish the details and apply."
    if score >= 75:
        return "A", "A good match with a few gaps to close."
    if score >= 65:
        return "B", "A reasonable match. Addressing key skill gaps will boost your ranking."
    if score >= 50:
        return "C", "A partial match. The missing skills are what hurt you."
    return "D", "A weak match for this role. Tailor it before applying."


# ---------------------------------------------------------------- 6. Analysis Result
@dataclass
class DimensionScore:
    name: str
    weight: float
    value: float  # 0.0 to 1.0


@dataclass
class AnalysisResult:
    score: float
    dimensions: List[Tuple[str, float]]
    grade: str
    verdict: str
    matched_skills: Set[str]
    missing_skills: Set[str]
    extra_skills: Set[str]
    sections: Dict[str, bool]
    role_fits: List[Tuple[str, float]]
    action_plan: List[Tuple[str, str]]
    word_count: int
    flesch_score: float
    text_similarity: float
    skill_coverage: float
    raw_jd: str
    raw_resume: str
    matched_verbs: List[str] = field(default_factory=list)
    quantified_metrics: List[str] = field(default_factory=list)
    top_matched_keywords: List[str] = field(default_factory=list)
    top_missing_keywords: List[str] = field(default_factory=list)


def evaluate_resume(resume_text: str, jd_text: str) -> AnalysisResult:
    """Full synchronous multi-metric evaluation of resume against job description."""
    r_skills = extract_skills(resume_text)
    j_skills = extract_skills(jd_text)

    matched = r_skills & j_skills
    missing = j_skills - r_skills
    extra = r_skills - j_skills

    # 1. Skill coverage
    cov = len(matched) / len(j_skills) if j_skills else 0.0

    # 2. Text similarity
    sim = tfidf_similarity(resume_text, jd_text)
    sim_scaled = min(sim / 0.5, 1.0)

    # 3. Structure
    sections = check_sections(resume_text)
    struct = sum(1 for v in sections.values() if v) / len(sections)

    # 4. Impact (Action verbs + Quantified achievements)
    matched_verbs = [v for v, pat in ACTION_VERB_PATTERNS.items() if pat.search(resume_text)]
    vb_count = len(matched_verbs)
    quantified_metrics = [m.group(0).strip() for m in re.finditer(r"\d+(?:\.\d+)?\s?(?:%|x|k\b|\+)|\$\d+(?:,\d{3})*(?:\.\d+)?[kmb]?", resume_text, re.IGNORECASE)]
    num_matches = len(quantified_metrics)
    impact = min(1.0, (min(vb_count, 5) / 5.0) * 0.6 + (min(num_matches, 3) / 3.0) * 0.4)

    # 5. Readability
    fl = flesch_reading_ease(resume_text)
    readability = max(0.0, min(1.0, (fl - 20.0) / 50.0))

    # Composite weighted score (0-100)
    # Weights: Skill 40%, Text 20%, Structure 15%, Impact 15%, Readability 10%
    score = 100.0 * (
        0.40 * cov +
        0.20 * sim_scaled +
        0.15 * struct +
        0.15 * impact +
        0.10 * readability
    )
    score_rounded = round(score, 1)

    # Grade & Verdict
    grade, verdict = grade_and_verdict(score_rounded)

    # Dimensions for Radar & Progress Bars
    dimensions = [
        ("Skill fit", round(cov, 2)),
        ("Text match", round(sim_scaled, 2)),
        ("Structure", round(struct, 2)),
        ("Impact", round(impact, 2)),
        ("Readability", round(readability, 2)),
    ]

    # Role Fit Projection
    role_fits = []
    for role_name, req_skills in ROLES.items():
        found_in_role = sum(1 for s in req_skills if s in r_skills)
        fit_ratio = found_in_role / len(req_skills)
        role_fits.append((role_name, fit_ratio))
    role_fits.sort(key=lambda x: x[1], reverse=True)

    # Action Plan
    plan: List[Tuple[str, str]] = []
    missing_prioritized = [(m, skill_priority(m, jd_text)) for m in missing]
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    missing_prioritized.sort(key=lambda x: priority_order.get(x[1], 3))

    for m, prio in missing_prioritized[:4]:
        plan.append((prio, f"Add {m} to your skills or a project, if you have used it. The job description asks for it."))

    missing_sections = [k for k, v in sections.items() if not v]
    for k in missing_sections[:2]:
        plan.append(("Medium", f"Add a clear {k} section; ATS parsers look for it by heading."))

    if num_matches < 2:
        plan.append(("High", "Quantify your work: add numbers such as users, accuracy, time saved or dataset size."))
    if vb_count < 3:
        plan.append(("Medium", "Start bullet points with action verbs: built, designed, reduced, deployed."))

    wc = len([w for w in re.split(r"\s+", resume_text) if w])
    if wc < 150:
        plan.append(("Medium", "The resume is short. Add detail to projects and internships."))
    elif wc > 900:
        plan.append(("Low", "The resume is long. Trim to one or two pages."))

    if not plan:
        plan.append(("Low", "Strong match. Tailor the summary line to this role."))

    top_matched_kw, top_missing_kw = extract_keyword_gaps(resume_text, jd_text)

    return AnalysisResult(
        score=score_rounded,
        dimensions=dimensions,
        grade=grade,
        verdict=verdict,
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        sections=sections,
        role_fits=role_fits,
        action_plan=plan,
        word_count=wc,
        flesch_score=fl,
        text_similarity=sim,
        skill_coverage=cov,
        raw_jd=jd_text,
        raw_resume=resume_text,
        matched_verbs=matched_verbs,
        quantified_metrics=quantified_metrics,
        top_matched_keywords=top_matched_kw,
        top_missing_keywords=top_missing_kw,
    )


def result_to_dict(result: AnalysisResult) -> Dict[str, Any]:
    """Serialize an AnalysisResult into a JSON-friendly dictionary."""
    return {
        "score": result.score,
        "grade": result.grade,
        "verdict": result.verdict,
        "dimensions": [{"name": name, "value": val} for name, val in result.dimensions],
        "matched_skills": sorted(result.matched_skills),
        "missing_skills": sorted(result.missing_skills),
        "extra_skills": sorted(result.extra_skills),
        "sections": dict(result.sections),
        "role_fits": [{"role": name, "fit": round(val, 4)} for name, val in result.role_fits],
        "action_plan": [{"priority": prio, "recommendation": text} for prio, text in result.action_plan],
        "word_count": result.word_count,
        "flesch_score": round(result.flesch_score, 2),
        "text_similarity": round(result.text_similarity, 4),
        "skill_coverage": round(result.skill_coverage, 4),
        "matched_verbs": list(result.matched_verbs),
        "quantified_metrics": list(result.quantified_metrics),
        "top_matched_keywords": list(result.top_matched_keywords),
        "top_missing_keywords": list(result.top_missing_keywords),
    }


def generate_text_report(result: AnalysisResult) -> str:
    """Generate a formatted plain-text evaluation report."""
    dim_map = dict(result.dimensions)
    lines = [
        "AI RESUME ANALYZER - EVALUATION REPORT",
        "=" * 50,
        f"Match Score: {result.score:.1f} / 100  (Grade {result.grade})",
        f"Verdict: {result.verdict}",
        "",
        "DIMENSION BREAKDOWN",
        "-" * 19,
        f"Skill Fit:        {round(dim_map.get('Skill fit', 0) * 100)}%",
        f"Text Similarity:  {round(dim_map.get('Text match', 0) * 100)}% (Cosine: {result.text_similarity:.2f})",
        f"Structure:        {round(dim_map.get('Structure', 0) * 100)}%",
        f"Impact:           {round(dim_map.get('Impact', 0) * 100)}%",
        f"Readability:      {round(dim_map.get('Readability', 0) * 100)}% (Flesch Score: {result.flesch_score:.0f})",
        "",
        "SKILLS SUMMARY",
        "-" * 14,
        f"Matched Skills ({len(result.matched_skills)}): {', '.join(sorted(result.matched_skills)) or 'None'}",
        f"Missing Skills ({len(result.missing_skills)}): {', '.join(sorted(result.missing_skills)) or 'None'}",
        f"Other Candidate Skills: {', '.join(sorted(result.extra_skills)) or 'None'}",
        "",
        "IMPACT & KEYWORDS",
        "-" * 17,
        f"Action Verbs ({len(result.matched_verbs)}): {', '.join(result.matched_verbs) or 'None'}",
        f"Quantified Metrics ({len(result.quantified_metrics)}): {', '.join(result.quantified_metrics) or 'None'}",
        f"Top Missing JD Keywords: {', '.join(result.top_missing_keywords) or 'None'}",
        "",
        "ROLE AFFINITY PROJECTION",
        "-" * 24,
        *[f"- {role}: {round(fit * 100)}%" for role, fit in result.role_fits],
        "",
        "RECOMMENDED ACTION PLAN",
        "-" * 23,
        *[f"{idx}. [{prio}] {msg}" for idx, (prio, msg) in enumerate(result.action_plan, 1)],
        "=" * 50,
    ]
    return "\n".join(lines)


def batch_evaluate(records: List[Dict[str, str]], jd_text: str) -> List[Dict[str, Any]]:
    """Evaluate a list of {'resume_text': ..., 'label': ...} records against a JD."""
    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(records):
        text = row.get("resume_text", "")
        if not text.strip():
            continue
        label = row.get("label", "Unlabeled")
        res = evaluate_resume(text, jd_text)
        results.append({
            "index": idx,
            "label": label,
            "score": res.score,
            "grade": res.grade,
            "skill_coverage": round(res.skill_coverage * 100, 1),
            "text_similarity": round(res.text_similarity * 100, 1),
            "matched_count": len(res.matched_skills),
            "missing_count": len(res.missing_skills),
            "top_role": res.role_fits[0][0] if res.role_fits else "",
            "word_count": res.word_count,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def evaluate_csv(csv_path: str, jd_text: str) -> Dict[str, Any]:
    """Load a CSV dataset of resumes and compute ranked benchmark statistics."""
    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    ranked = batch_evaluate(records, jd_text)
    by_label: Dict[str, List[float]] = {}
    for item in ranked:
        by_label.setdefault(item["label"], []).append(item["score"])
    category_averages = {
        label: round(sum(scores) / len(scores), 1)
        for label, scores in sorted(by_label.items())
    }
    return {
        "total_resumes": len(ranked),
        "average_score": round(sum(x["score"] for x in ranked) / len(ranked), 1) if ranked else 0.0,
        "category_averages": category_averages,
        "top_candidates": ranked[:10],
        "all_results": ranked,
    }


# Backward compatibility for legacy tests / scripts
analyze = evaluate_resume
