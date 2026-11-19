"""Unit & integration tests for AI Resume Analyzer."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analyzer import (
    evaluate_csv,
    evaluate_resume,
    extract_keyword_gaps,
    extract_skills,
    flesch_reading_ease,
    generate_text_report,
    result_to_dict,
    tfidf_similarity,
    tokenize,
)

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
RESUME = open(os.path.join(BASE, "resume.txt"), encoding="utf-8").read()
JD = open(os.path.join(BASE, "job_description.txt"), encoding="utf-8").read()
CSV_PATH = os.path.join(BASE, "sample_resumes.csv")


class TestResumeAnalyzer(unittest.TestCase):
    def test_skill_extraction(self):
        s = extract_skills("Worked with Python, SQL and scikit-learn. Used C++ too.")
        self.assertTrue({"python", "sql", "scikit-learn", "c++"} <= s)

    def test_tokenization_and_stop_words(self):
        tokens = tokenize("The team is working at http://x.com on Python and SQL")
        self.assertIn("python", tokens)
        self.assertIn("sql", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("http", tokens)

    def test_tfidf_similarity(self):
        sim = tfidf_similarity("python data science machine learning", "python machine learning engineer")
        self.assertGreater(sim, 0.0)
        self.assertLessEqual(sim, 1.0)
        self.assertEqual(tfidf_similarity("", "python"), 0.0)

    def test_flesch_reading_ease(self):
        score = flesch_reading_ease("This is a simple sentence. It is very easy to read.")
        self.assertGreater(score, 60.0)
        self.assertEqual(flesch_reading_ease(""), 0.0)

    def test_action_verb_word_boundaries(self):
        # "scheduled" and "detailed" contain "led" as a substring and must not match
        res = evaluate_resume("Scheduled detailed meetings for the team.", JD)
        self.assertNotIn("led", res.matched_verbs)

        res_with_verb = evaluate_resume("Led a 5-person team and built 3 pipelines.", JD)
        self.assertIn("led", res_with_verb.matched_verbs)
        self.assertIn("built", res_with_verb.matched_verbs)

    def test_keyword_gaps(self):
        matched_kw, missing_kw = extract_keyword_gaps(
            "Experienced with Python and data pipelines.",
            "Looking for Python, Kubernetes, and Terraform pipelines."
        )
        self.assertIn("python", matched_kw)
        self.assertIn("kubernetes", missing_kw)
        self.assertIn("terraform", missing_kw)

    def test_full_evaluation_and_serialization(self):
        r = evaluate_resume(RESUME, JD)
        self.assertIn("python", r.matched_skills)
        self.assertIn("docker", r.missing_skills)
        self.assertTrue(0 <= r.score <= 100)
        self.assertEqual(len(r.dimensions), 5)
        self.assertEqual(len(r.role_fits), 7)
        self.assertTrue(any(role == "Business Analyst" for role, _ in r.role_fits))
        self.assertGreater(len(r.action_plan), 0)

        d = result_to_dict(r)
        self.assertEqual(d["score"], r.score)
        self.assertEqual(d["grade"], r.grade)
        self.assertIsInstance(d["matched_skills"], list)

        report = generate_text_report(r)
        self.assertIn("AI RESUME ANALYZER - EVALUATION REPORT", report)
        self.assertIn("IMPACT & KEYWORDS", report)

    def test_comparative_scoring(self):
        weak = evaluate_resume("Cooking and sports hobbyist. Education. Projects.", JD)
        strong = evaluate_resume(RESUME, JD)
        self.assertGreater(strong.score, weak.score)

    def test_batch_csv_benchmark(self):
        summary = evaluate_csv(CSV_PATH, JD)
        self.assertEqual(summary["total_resumes"], 270)
        self.assertEqual(len(summary["category_averages"]), 6)
        self.assertEqual(len(summary["top_candidates"]), 10)
        scores = [c["score"] for c in summary["top_candidates"]]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
