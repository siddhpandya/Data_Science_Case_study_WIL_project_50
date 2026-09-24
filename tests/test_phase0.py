"""Phase 0 Acceptance Tests.

Validates:
1. config/config.yaml loads cleanly and has all required sections.
2. The directory tree matches the required repository layout.
3. The four human-authored ground truth template files exist with exact schemas
   and contain at least one row (allowing questions and sources to be added
   without failing tests).
"""

from pathlib import Path
import csv
import unittest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPhase0Scaffold(unittest.TestCase):

    def test_config_yaml(self):
        config_path = REPO_ROOT / "config" / "config.yaml"
        self.assertTrue(config_path.exists(), "config/config.yaml does not exist")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.assertIsNotNone(config)
        self.assertEqual(config.get("seed"), 42)
        self.assertIn("paths", config)
        self.assertIn("chunking", config)
        self.assertIn("retrieval", config)
        self.assertIn("generation", config)
        self.assertIn("evaluation", config)

    def test_directory_tree_exists(self):
        required_dirs = [
            "config",
            "data/raw",
            "data/interim",
            "data/embeddings",
            "src/ingest",
            "src/retrieval",
            "src/generation",
            "src/evaluation",
            "src/experiments",
            "src/app",
            "runs",
            "results",
            "results/figures",
            "results/generations",
            "tests",
        ]
        for d in required_dirs:
            dir_path = REPO_ROOT / d
            self.assertTrue(dir_path.is_dir(), f"Required directory {d} does not exist")

    def test_sources_template(self):
        p = REPO_ROOT / "data" / "sources.csv"
        self.assertTrue(p.exists(), "data/sources.csv does not exist")
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_header = [
                "source_id",
                "url",
                "publisher",
                "title",
                "retrieved_at",
                "last_updated",
                "licence",
                "is_superseded",
                "superseded_by",
                "notes",
            ]
            self.assertEqual(header, expected_header)
            rows = list(reader)
            self.assertGreaterEqual(len(rows), 1, "data/sources.csv must have at least 1 row")

    def test_topics_template(self):
        p = REPO_ROOT / "data" / "topics.csv"
        self.assertTrue(p.exists(), "data/topics.csv does not exist")
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_header = [
                "topic_id",
                "topic",
                "question_id",
                "question",
                "question_type",
                "variant_of",
                "register",
                "currency_sensitive",
            ]
            self.assertEqual(header, expected_header)
            rows = list(reader)
            self.assertGreaterEqual(len(rows), 1, "data/topics.csv must have at least 1 row")

    def test_qrels_template(self):
        p = REPO_ROOT / "data" / "qrels.txt"
        self.assertTrue(p.exists(), "data/qrels.txt does not exist")
        with open(p, "r", encoding="utf-8") as f:
            lines = [line.strip().split("\t") for line in f if line.strip()]
            self.assertGreaterEqual(len(lines), 1, "data/qrels.txt must have at least 1 row")
            for row in lines:
                self.assertEqual(len(row), 4, f"TREC qrel line must have 4 tab-separated fields: {row}")
                self.assertEqual(row[1], "0", f"Second column must be '0' in TREC qrels: {row}")
                self.assertIn(row[3], ["1", "2"], f"Relevance must be 1 or 2: {row}")

    def test_gold_answers_template(self):
        p = REPO_ROOT / "data" / "gold_answers.csv"
        self.assertTrue(p.exists(), "data/gold_answers.csv does not exist")
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_header = ["question_id", "gold_answer", "passage_ids"]
            self.assertEqual(header, expected_header)
            rows = list(reader)
            self.assertGreaterEqual(len(rows), 1, "data/gold_answers.csv must have at least 1 row")


if __name__ == "__main__":
    unittest.main()
