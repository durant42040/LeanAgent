# import all the necessary modules
import datetime
import json
import math
import os
import random
import shutil
import unittest
from pathlib import Path
from typing import Tuple, Union
from unittest.mock import MagicMock, Mock, patch

import lean_dojo
from lean_dojo.data_extraction.lean import LeanGitRepo, Pos
from loguru import logger

import generate_benchmark_lean4
from dynamic_database import (
    AnnotatedTactic,
    Annotation,
    DynamicDatabase,
    Premise,
    PremiseFile,
    Repository,
    Theorem,
    parse_pos,
)
from prover.proof_search import SearchResult, Status

RAID_DIR = os.environ.get("RAID_DIR")
DATA_DIR = "datasets_new_unittest"
MERGED_DATA_DIR = "datasets_merged_unittest"
PROOF_LOG_FILE_NAME = "proof_logs_unittest/proof_log_unittest.log"


class TestDynamicDatabaseCore(unittest.TestCase):
    """
    Unit tests for the DynamicDatabase class and related functionality.
    This test suite covers the following aspects of the DynamicDatabase:
    - Repository operations (add, update, get)
    - Theorem operations (adding, retrieving, updating)
    - Premise file operations
    - JSON serialization and deserialization
    - Repository properties (total_theorems, etc.)
    - Special case handling (empty strings, None values)
    - Theorem difficulty rating calculation and updating
    - Converting sorry theorems to proven theorems
    - Handling of duplicate repositories
    - Repository equality
    The tests use a combination of simple and complex test cases to verify that
    all aspects of the database function correctly, including edge cases.
    """

    def setUp(self):
        self.db = DynamicDatabase()
        self.repo = Repository(
            url="https://github.com/test/repo",
            name="Test Repo",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )

    def assertDatetimeEqual(self, dt1, dt2):
        """
        Assert that two datetimes are equal, ignoring microseconds.
        This is important because serialization and deserialization of datetimes
        may lose microsecond precision.
        """
        self.assertEqual(dt1.replace(microsecond=0), dt2.replace(microsecond=0))

    def test_parse_pos_bad_position(self):
        with self.assertRaises(ValueError):
            parse_pos("invalid_pos")

    def test_get_update_theorem_in_repo(self):
        theorem = Theorem(
            full_name="test_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        self.repo.proven_theorems.append(theorem)
        self.db.add_repository(self.repo)

        retrieved_theorem = self.repo.get_theorem("test_theorem", "test.lean")
        self.assertEqual(retrieved_theorem, theorem)

        updated_theorem = Theorem(
            full_name="test_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            theorem_statement="Updated statement",
        )
        self.repo.update_theorem(updated_theorem)
        retrieved_theorem = self.repo.get_theorem("test_theorem", "test.lean")
        self.assertEqual(retrieved_theorem.theorem_statement, "Updated statement")

        non_existent_theorem = Theorem(
            full_name="non_existent",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        with self.assertRaises(ValueError):
            self.repo.update_theorem(non_existent_theorem)

    def test_get_premise_file_in_repo(self):
        premise_file = PremiseFile(path=Path("test.lean"), imports=[], premises=[])
        self.repo.premise_files.append(premise_file)
        self.db.add_repository(self.repo)

        retrieved_premise_file = self.repo.get_premise_file("test.lean")
        self.assertEqual(retrieved_premise_file, premise_file)

    def test_get_file_traced(self):
        self.repo.files_traced.append(Path("test.lean"))
        self.db.add_repository(self.repo)

        retrieved_file = self.repo.get_file_traced("test.lean")
        self.assertEqual(retrieved_file, Path("test.lean"))

    def test_update_pr_url(self):
        self.repo.pr_url = "https://github.com/test/repo/pull/1"
        self.db.add_repository(self.repo)

        self.repo.pr_url = "https://github.com/test/repo/pull/2"
        self.db.update_repository(self.repo)

        updated_repo = self.db.get_repository(self.repo.url, self.repo.commit)
        self.assertEqual(updated_repo.pr_url, "https://github.com/test/repo/pull/2")

    def test_difficulty_rating_in_theorem(self):
        theorem = Theorem(
            full_name="test_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            difficulty_rating=0.7,
        )
        self.repo.proven_theorems.append(theorem)
        self.db.add_repository(self.repo)

        retrieved_theorem = self.repo.get_theorem("test_theorem", "test.lean")
        self.assertEqual(retrieved_theorem.difficulty_rating, 0.7)

    def test_validation_in_from_dict(self):
        with self.assertRaises(ValueError):
            DynamicDatabase.from_dict({})

        with self.assertRaises(ValueError):
            Repository.from_dict({})

        with self.assertRaises(ValueError):
            Theorem.from_dict({}, "url", "commit")

        with self.assertRaises(ValueError):
            PremiseFile.from_dict({})

        with self.assertRaises(ValueError):
            Premise.from_dict({})

        with self.assertRaises(ValueError):
            Annotation.from_dict({})

        with self.assertRaises(ValueError):
            AnnotatedTactic.from_dict({})

    def test_empty_path_to_data_in_from_dict_repository(self):
        data = {
            "url": "https://github.com/test/repo",
            "name": "Test Repo",
            "commit": "abc123",
            "lean_version": "3.50.3",
            "lean_dojo_version": "1.8.4",
            "metadata": {"date_processed": datetime.datetime.now().isoformat()},
            "theorems_folder": "",
            "premise_files_corpus": "",
            "files_traced": "",
        }
        with self.assertRaises(ValueError):
            Repository.from_dict(data)

    def test_to_dict_for_all(self):
        theorem = Theorem(
            full_name="test_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        theorem_dict = theorem.to_dict()
        self.assertIsInstance(theorem_dict, dict)
        self.assertEqual(theorem_dict["full_name"], "test_theorem")

        premise_file = PremiseFile(path=Path("test.lean"), imports=[], premises=[])
        premise_file_dict = premise_file.to_dict()
        self.assertIsInstance(premise_file_dict, dict)
        self.assertEqual(premise_file_dict["path"], "test.lean")

        repo_dict = self.repo.to_dict()
        self.assertIsInstance(repo_dict, dict)
        self.assertEqual(repo_dict["url"], "https://github.com/test/repo")

        annotation = Annotation(
            full_name="test_annotation",
            def_path="test/path.lean",
            def_pos=Pos(1, 1),
            def_end_pos=Pos(2, 1),
        )
        annotation_dict = annotation.to_dict()
        self.assertIsInstance(annotation_dict, dict)
        self.assertEqual(annotation_dict["full_name"], "test_annotation")
        self.assertEqual(annotation_dict["def_path"], "test/path.lean")
        self.assertEqual(annotation_dict["def_pos"], "(1, 1)")
        self.assertEqual(annotation_dict["def_end_pos"], "(2, 1)")

        annotated_tactic = AnnotatedTactic(
            tactic="test_tactic",
            annotated_tactic=("test_tactic", [annotation]),
            state_before="test_state_before",
            state_after="test_state_after",
        )
        annotated_tactic_dict = annotated_tactic.to_dict()
        self.assertIsInstance(annotated_tactic_dict, dict)
        self.assertEqual(annotated_tactic_dict["tactic"], "test_tactic")
        self.assertEqual(annotated_tactic_dict["state_before"], "test_state_before")
        self.assertEqual(annotated_tactic_dict["state_after"], "test_state_after")

        premise = Premise(
            full_name="test_premise",
            code="test_code",
            start=Pos(1, 1),
            end=Pos(2, 1),
            kind="theorem",
        )
        premise_dict = premise.to_dict()
        self.assertIsInstance(premise_dict, dict)
        self.assertEqual(premise_dict["full_name"], "test_premise")
        self.assertEqual(premise_dict["code"], "test_code")
        self.assertEqual(premise_dict["start"], "(1, 1)")
        self.assertEqual(premise_dict["end"], "(2, 1)")
        self.assertEqual(premise_dict["kind"], "theorem")

    def test_empty_string_and_none_json_serialization(self):
        empty_theorem = Theorem(
            full_name="",
            file_path=Path(""),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            theorem_statement=None,
            difficulty_rating=None,
        )
        self.repo.proven_theorems.append(empty_theorem)
        self.db.add_repository(self.repo)

        json_file = "empty_none_test.json"
        self.db.to_json(json_file)

        loaded_db = DynamicDatabase.from_json(json_file)
        loaded_repo = loaded_db.get_repository(self.repo.url, self.repo.commit)
        loaded_theorem = loaded_repo.proven_theorems[-1]

        self.assertEqual(loaded_theorem.full_name, "")
        self.assertEqual(str(loaded_theorem.file_path), ".")
        self.assertEqual(loaded_theorem.url, "https://github.com/test/repo")
        self.assertEqual(loaded_theorem.commit, "abc123")
        self.assertIsNone(loaded_theorem.theorem_statement)
        self.assertIsNone(loaded_theorem.difficulty_rating)

    def test_complex_json_serialization(self):
        theorem1 = Theorem(
            full_name="theorem1",
            file_path=Path("test1.lean"),
            start=Pos(1, 1),
            end=Pos(10, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            theorem_statement="theorem1 : 2 + 2 = 4",
            traced_tactics=[
                AnnotatedTactic(
                    tactic="rw [add_comm]",
                    annotated_tactic=(
                        "rw [add_comm]",
                        [
                            Annotation(
                                full_name="add_comm",
                                def_path="src/add_comm.lean",
                                def_pos=Pos(5, 1),
                                def_end_pos=Pos(7, 1),
                            )
                        ],
                    ),
                    state_before="⊢ 2 + 2 = 4",
                    state_after="⊢ 2 + 2 = 4",
                )
            ],
            difficulty_rating=0.7,
        )

        theorem2 = Theorem(
            full_name="theorem2",
            file_path=Path("test2.lean"),
            start=Pos(1, 1),
            end=Pos(5, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            theorem_statement="theorem2 : ∀ x y : ℕ, x + y = y + x",
            traced_tactics=[],
            difficulty_rating=None,
        )

        premise_file = PremiseFile(
            path=Path("premise.lean"),
            imports=["import data.nat.basic"],
            premises=[
                Premise(
                    full_name="nat_add_comm",
                    code="theorem nat_add_comm : ∀ a b : ℕ, a + b = b + a := sorry",
                    start=Pos(1, 1),
                    end=Pos(1, 60),
                    kind="theorem",
                )
            ],
        )

        complex_repo = Repository(
            url="https://github.com/test/complex-repo",
            name="Complex Test Repo",
            commit="complex123",
            lean_version="4.0.0",
            lean_dojo_version="1.0.0",
            metadata={
                "date_processed": datetime.datetime.now(),
                "extra_info": {"key1": "value1", "key2": 2},
            },
            proven_theorems=[theorem1],
            sorry_theorems_unproved=[theorem2],
            premise_files=[premise_file],
            files_traced=[Path("test1.lean"), Path("test2.lean")],
            pr_url="https://github.com/test/complex-repo/pull/1",
        )

        self.db.add_repository(complex_repo)

        json_file = "complex_test_database.json"
        self.db.to_json(json_file)

        loaded_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(loaded_db.repositories), len(self.db.repositories))
        loaded_repo = loaded_db.get_repository(
            "https://github.com/test/complex-repo", "complex123"
        )
        self.assertIsNotNone(loaded_repo)

        self.assertEqual(loaded_repo.name, "Complex Test Repo")
        self.assertEqual(loaded_repo.lean_version, "4.0.0")
        self.assertEqual(
            loaded_repo.pr_url, "https://github.com/test/complex-repo/pull/1"
        )

        # Check theorems
        self.assertEqual(len(loaded_repo.proven_theorems), 1)
        self.assertEqual(len(loaded_repo.sorry_theorems_unproved), 1)

        loaded_theorem1 = loaded_repo.proven_theorems[0]
        self.assertEqual(loaded_theorem1.full_name, "theorem1")
        self.assertEqual(loaded_theorem1.theorem_statement, "theorem1 : 2 + 2 = 4")
        self.assertEqual(len(loaded_theorem1.traced_tactics), 1)
        self.assertEqual(loaded_theorem1.difficulty_rating, 0.7)

        loaded_theorem2 = loaded_repo.sorry_theorems_unproved[0]
        self.assertEqual(loaded_theorem2.full_name, "theorem2")
        self.assertIsNone(loaded_theorem2.difficulty_rating)

        # Check premise files
        self.assertEqual(len(loaded_repo.premise_files), 1)
        loaded_premise_file = loaded_repo.premise_files[0]
        self.assertEqual(str(loaded_premise_file.path), "premise.lean")
        self.assertEqual(len(loaded_premise_file.premises), 1)

        # Check metadata
        self.assertIn("extra_info", loaded_repo.metadata)
        self.assertEqual(loaded_repo.metadata["extra_info"]["key1"], "value1")
        self.assertEqual(loaded_repo.metadata["extra_info"]["key2"], 2)

        # Check files traced
        self.assertEqual(len(loaded_repo.files_traced), 2)
        self.assertIn(Path("test1.lean"), loaded_repo.files_traced)
        self.assertIn(Path("test2.lean"), loaded_repo.files_traced)

    def test_is_same_theorem(self):
        theorem1 = Theorem(
            full_name="test_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        theorem2 = Theorem(
            full_name="test_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        theorem3 = Theorem(
            full_name="other_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        self.assertTrue(theorem1.is_same_theorem(theorem2))
        self.assertFalse(theorem1.is_same_theorem(theorem3))

    def test_repository_properties(self):
        theorem1 = Theorem(
            full_name="test_theorem1",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        theorem2 = Theorem(
            full_name="test_theorem2",
            file_path=Path("test.lean"),
            start=Pos(3, 1),
            end=Pos(4, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        self.repo.proven_theorems.append(theorem1)
        self.repo.sorry_theorems_unproved.append(theorem2)

        self.assertEqual(self.repo.total_theorems, 2)
        self.assertEqual(self.repo.num_proven_theorems, 1)
        self.assertEqual(self.repo.num_sorry_theorems, 1)
        self.assertEqual(self.repo.num_sorry_theorems_unproved, 1)
        self.assertEqual(self.repo.num_sorry_theorems_proved, 0)

    def test_get_all_theorems(self):
        theorem1 = Theorem(
            full_name="test_theorem1",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        theorem2 = Theorem(
            full_name="test_theorem2",
            file_path=Path("test.lean"),
            start=Pos(3, 1),
            end=Pos(4, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        self.repo.proven_theorems.append(theorem1)
        self.repo.sorry_theorems_unproved.append(theorem2)

        all_theorems = self.repo.get_all_theorems
        self.assertEqual(len(all_theorems), 2)
        self.assertIn(theorem1, all_theorems)
        self.assertIn(theorem2, all_theorems)

    def test_empty_repository(self):
        empty_repo = Repository(
            url="https://github.com/empty/repo",
            name="Empty Repo",
            commit="empty123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )
        self.db.add_repository(empty_repo)

        self.assertEqual(empty_repo.total_theorems, 0)
        self.assertEqual(empty_repo.num_proven_theorems, 0)
        self.assertEqual(empty_repo.num_sorry_theorems, 0)
        self.assertEqual(len(empty_repo.premise_files), 0)
        self.assertEqual(len(empty_repo.files_traced), 0)

    def test_theorem_with_empty_traced_tactics(self):
        theorem = Theorem(
            full_name="term_style_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            traced_tactics=[],
        )
        self.repo.proven_theorems.append(theorem)
        self.db.add_repository(self.repo)

        retrieved_theorem = self.repo.get_theorem("term_style_theorem", "test.lean")
        self.assertEqual(len(retrieved_theorem.traced_tactics), 0)

    def test_none_values(self):
        theorem = Theorem(
            full_name="none_value_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            theorem_statement=None,
            difficulty_rating=None,
        )
        self.repo.proven_theorems.append(theorem)
        self.repo.pr_url = None
        self.db.add_repository(self.repo)

        retrieved_theorem = self.repo.get_theorem("none_value_theorem", "test.lean")
        self.assertIsNone(retrieved_theorem.theorem_statement)
        self.assertIsNone(retrieved_theorem.difficulty_rating)
        self.assertIsNone(self.repo.pr_url)

    def test_empty_strings(self):
        theorem = Theorem(
            full_name="",
            file_path=Path(""),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="",
            commit="",
        )
        self.repo.proven_theorems.append(theorem)
        self.db.add_repository(self.repo)

        retrieved_theorem = self.repo.get_theorem("", "")
        self.assertEqual(retrieved_theorem.full_name, "")
        self.assertEqual(str(retrieved_theorem.file_path), ".")
        self.assertEqual(retrieved_theorem.url, "")
        self.assertEqual(retrieved_theorem.commit, "")

        theorem2 = Theorem(
            full_name="",
            file_path=Path(""),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="new_url",
            commit="",
        )
        self.repo.update_theorem(theorem2)
        retrieved_theorem = self.repo.get_theorem("", "")  # Should be theorem2
        self.assertEqual(retrieved_theorem.full_name, "")
        self.assertEqual(str(retrieved_theorem.file_path), ".")
        self.assertEqual(retrieved_theorem.url, "new_url")
        self.assertEqual(retrieved_theorem.commit, "")

    def test_datetime_serialization(self):
        original_date = datetime.datetime.now()
        self.repo.metadata["date_processed"] = original_date
        self.db.add_repository(self.repo)

        json_file = "test_datetime.json"
        self.db.to_json(json_file)

        loaded_db = DynamicDatabase.from_json(json_file)
        loaded_repo = loaded_db.get_repository(self.repo.url, self.repo.commit)

        self.assertDatetimeEqual(original_date, loaded_repo.metadata["date_processed"])

    def test_duplicate_url_different_commit(self):
        repo1 = Repository(
            url="https://github.com/test/repo",
            name="Test Repo 1",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )
        repo2 = Repository(
            url="https://github.com/test/repo",
            name="Test Repo 2",
            commit="def456",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={
                "date_processed": datetime.datetime.now() + datetime.timedelta(days=1)
            },
        )

        # Add a theorem to both repositories
        common_theorem = Theorem(
            full_name="common_theorem",
            file_path=Path("common.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            theorem_statement="Old version",
        )
        repo1.proven_theorems.append(common_theorem)

        updated_common_theorem = Theorem(
            full_name="common_theorem",
            file_path=Path("common.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="def456",
            theorem_statement="New version",
        )
        repo2.proven_theorems.append(updated_common_theorem)

        # Add unique theorems to each repository
        repo1.proven_theorems.append(
            Theorem(
                full_name="unique_to_repo1",
                file_path=Path("repo1.lean"),
                start=Pos(1, 1),
                end=Pos(2, 1),
                url="https://github.com/test/repo",
                commit="abc123",
            )
        )
        repo2.proven_theorems.append(
            Theorem(
                full_name="unique_to_repo2",
                file_path=Path("repo2.lean"),
                start=Pos(1, 1),
                end=Pos(2, 1),
                url="https://github.com/test/repo",
                commit="def456",
            )
        )

        self.db.add_repository(repo1)
        self.db.add_repository(repo2)

        self.assertEqual(len(self.db.repositories), 2)

        dst_dir = Path(RAID_DIR) / DATA_DIR / "test_duplicate_url"
        self.db.generate_merged_dataset(dst_dir)
        with open(dst_dir / "random" / "train.json", "r") as f:
            data = json.load(f)

        # Check that both repositories are represented
        self.assertEqual(len(data), 3)

        # Check that the common theorem is from the most recent repository
        common_theorem_in_dataset = next(
            t for t in data if t["full_name"] == "common_theorem"
        )
        self.assertEqual(common_theorem_in_dataset["theorem_statement"], "New version")
        self.assertEqual(common_theorem_in_dataset["commit"], "def456")

        # Check that unique theorems from both repositories are present
        self.assertTrue(any(t["full_name"] == "unique_to_repo1" for t in data))
        self.assertTrue(any(t["full_name"] == "unique_to_repo2" for t in data))

        with open(dst_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.assertEqual(len(metadata["repositories"]), 2)
        self.assertTrue(
            any(repo["commit"] == "abc123" for repo in metadata["repositories"])
        )
        self.assertTrue(
            any(repo["commit"] == "def456" for repo in metadata["repositories"])
        )

    def test_change_sorry_to_proven(self):
        theorem = Theorem(
            full_name="sorry_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        self.repo.sorry_theorems_unproved.append(theorem)
        self.db.add_repository(self.repo)

        self.repo.change_sorry_to_proven(theorem, PROOF_LOG_FILE_NAME)
        self.assertEqual(len(self.repo.sorry_theorems_unproved), 0)
        self.assertEqual(len(self.repo.sorry_theorems_proved), 1)
        self.assertEqual(self.repo.sorry_theorems_proved[0].full_name, "sorry_theorem")

        not_found_theorem = Theorem(
            full_name="not_found_theorem",
            file_path=Path("test.lean"),
            start=Pos(3, 1),
            end=Pos(4, 1),
            url="https://github.com/test/repo",
            commit="abc123",
        )
        with self.assertRaises(ValueError):
            self.repo.change_sorry_to_proven(not_found_theorem, PROOF_LOG_FILE_NAME)

        with self.assertRaises(ValueError):
            self.repo.change_sorry_to_proven(theorem, PROOF_LOG_FILE_NAME)

    def test_add_repository_duplicate(self):
        repo = Repository(
            url="https://github.com/test/repo",
            name="Test Repo",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )

        # Add the repository for the first time
        self.db.add_repository(repo)
        self.assertEqual(len(self.db.repositories), 1)

        # Try to add the same repository again
        self.db.add_repository(repo)
        self.assertEqual(
            len(self.db.repositories), 1, "Repository should not be added twice"
        )

        # Verify that the repository details are unchanged
        added_repo = self.db.get_repository("https://github.com/test/repo", "abc123")
        self.assertIsNotNone(added_repo)
        self.assertEqual(added_repo.name, "Test Repo")
        self.assertEqual(added_repo.commit, "abc123")

    def test_repository_equality(self):
        repo1 = Repository(
            url="https://github.com/test/repo",
            name="Test Repo",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )
        repo2 = Repository(
            url="https://github.com/test/repo",
            name="Test Repo",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )
        repo3 = Repository(
            url="https://github.com/test/repo",
            name="Test Repo",
            commit="def456",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )
        self.assertEqual(repo1, repo2)
        self.assertNotEqual(repo1, repo3)

    def test_add_repository_duplicate(self):
        repo = Repository(
            url="https://github.com/test/repo",
            name="Test Repo",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )
        self.db.add_repository(repo)
        self.assertEqual(len(self.db.repositories), 1)
        self.db.add_repository(repo)
        self.assertEqual(len(self.db.repositories), 1)

    def test_update_repository_duplicate(self):
        repo = Repository(
            url="https://github.com/test/repo",
            name="Test Repo",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )
        self.db.add_repository(repo)
        self.assertEqual(len(self.db.repositories), 1)
        self.db.update_repository(repo)
        added_repo = self.db.get_repository("https://github.com/test/repo", "abc123")
        self.assertEqual(len(self.db.repositories), 1)
        self.assertEqual(added_repo.name, "Test Repo")
        self.assertEqual(added_repo.commit, "abc123")
        self.assertEqual(added_repo.lean_version, "3.50.3")
        added_repo.name = "Updated Repo"
        added_repo.lean_version = "3.50.4"
        self.db.update_repository(added_repo)
        updated_repo = self.db.get_repository("https://github.com/test/repo", "abc123")
        self.assertEqual(len(self.db.repositories), 1)
        self.assertEqual(updated_repo.name, "Updated Repo")
        self.assertEqual(updated_repo.commit, "abc123")
        self.assertEqual(added_repo.lean_version, "3.50.4")
        self.db.update_repository(added_repo)
        updated_repo = self.db.get_repository("https://github.com/test/repo", "abc123")
        self.assertEqual(len(self.db.repositories), 1)
        self.assertEqual(updated_repo.name, "Updated Repo")
        self.assertEqual(updated_repo.commit, "abc123")
        self.assertEqual(added_repo.lean_version, "3.50.4")

    def test_update_theorem_difficulty(self):
        theorem = Theorem(
            full_name="test_theorem",
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            difficulty_rating=None,
        )
        self.repo.proven_theorems.append(theorem)
        self.db.add_repository(self.repo)

        # Calculate and update difficulty
        difficulty = 10
        theorem.difficulty_rating = difficulty

        # Verify the difficulty has been updated
        updated_theorem = self.repo.get_theorem("test_theorem", "test.lean")
        self.assertIsNotNone(updated_theorem.difficulty_rating)
        self.assertEqual(updated_theorem.difficulty_rating, difficulty)

        # Test updating difficulty of an existing theorem
        new_difficulty = 0.8
        theorem.difficulty_rating = new_difficulty
        updated_theorem = self.repo.get_theorem("test_theorem", "test.lean")
        self.assertEqual(updated_theorem.difficulty_rating, new_difficulty)

        self.db.update_repository(self.repo)
        json_file = "theorem_difficulty_test.json"
        self.db.to_json(json_file)

        # Read the JSON file and verify its contents
        loaded_db = DynamicDatabase.from_json(json_file)
        loaded_repo = loaded_db.get_repository("https://github.com/test/repo", "abc123")
        self.assertIsNotNone(loaded_repo)

        loaded_theorem = loaded_repo.get_theorem("test_theorem", "test.lean")
        self.assertIsNotNone(loaded_theorem)

    def create_theorem(self, name, tactics):
        return Theorem(
            full_name=name,
            file_path=Path("test.lean"),
            start=Pos(1, 1),
            end=Pos(2, 1),
            url="https://github.com/test/repo",
            commit="abc123",
            traced_tactics=tactics,
        )

    def _calculate_difficulty(self, theorem: Theorem) -> Union[float, None]:
        proof_steps = theorem.traced_tactics
        if any("sorry" in step.tactic for step in proof_steps):
            return float("inf")  # Hard (no proof)
        if len(proof_steps) == 0:
            return None  # To be distributed later
        return math.exp(len(proof_steps))

    def test_calculate_and_update_difficulty(self):
        # Test case 1: Theorem with 'sorry'
        sorry_theorem = self.create_theorem(
            "sorry_theorem",
            [
                AnnotatedTactic(
                    tactic="sorry",
                    annotated_tactic=("sorry", []),
                    state_before="",
                    state_after="",
                )
            ],
        )
        self.repo.sorry_theorems_unproved.append(sorry_theorem)

        # Test case 2: Theorem with no tactics
        empty_theorem = self.create_theorem("empty_theorem", [])
        self.repo.proven_theorems.append(empty_theorem)

        # Test case 3: Theorem with proven sorry
        normal_theorem = self.create_theorem(
            "proven_sorry_theorem",
            [
                AnnotatedTactic(
                    tactic="tactic1",
                    annotated_tactic=("tactic1", []),
                    state_before="",
                    state_after="",
                ),
                AnnotatedTactic(
                    tactic="tactic2",
                    annotated_tactic=("tactic2", []),
                    state_before="",
                    state_after="",
                ),
            ],
        )
        self.repo.proven_theorems.append(normal_theorem)

        # Test case 4: Theorem with normal teactics
        normal_theorem = self.create_theorem(
            "normal_theorem",
            [
                AnnotatedTactic(
                    tactic="tactic1",
                    annotated_tactic=("tactic1", []),
                    state_before="before",
                    state_after="no goals",
                ),
                AnnotatedTactic(
                    tactic="tactic2",
                    annotated_tactic=("tactic2", []),
                    state_before="before2",
                    state_after="no goals",
                ),
            ],
        )
        self.repo.proven_theorems.append(normal_theorem)

        self.db.add_repository(self.repo)
        json_file = "theorem_difficulty_test.json"
        self.db.to_json(json_file)

        for theorem in self.repo.get_all_theorems:
            difficulty = self._calculate_difficulty(theorem)
            theorem.difficulty_rating = difficulty

        self.db.update_repository(self.repo)

        sorry_theorem = self.repo.get_theorem("sorry_theorem", "test.lean")
        self.assertEqual(sorry_theorem.difficulty_rating, float("inf"))

        empty_theorem = self.repo.get_theorem("empty_theorem", "test.lean")
        self.assertIsNone(empty_theorem.difficulty_rating)

        normal_theorem = self.repo.get_theorem("normal_theorem", "test.lean")
        self.assertEqual(normal_theorem.difficulty_rating, math.exp(2))

        # Test JSON serialization and deserialization
        self.db.to_json(json_file)

        loaded_db = DynamicDatabase.from_json(json_file)
        loaded_repo = loaded_db.get_repository("https://github.com/test/repo", "abc123")
        self.assertIsNotNone(loaded_repo)

        loaded_sorry_theorem = loaded_repo.get_theorem("sorry_theorem", "test.lean")
        self.assertEqual(loaded_sorry_theorem.difficulty_rating, float("inf"))

        loaded_empty_theorem = loaded_repo.get_theorem("empty_theorem", "test.lean")
        self.assertIsNone(loaded_empty_theorem.difficulty_rating)

        loaded_normal_theorem = loaded_repo.get_theorem("normal_theorem", "test.lean")
        self.assertEqual(loaded_normal_theorem.difficulty_rating, math.exp(2))


class TestDynamicDatabaseSimpleLean(unittest.TestCase):
    def setUp(self):
        self.db = DynamicDatabase()
        self.simple_lean_repo = self.create_simple_lean_repo()
        self.db.add_repository(self.simple_lean_repo)

    def create_simple_lean_repo(self):
        url = "https://github.com/Adarsh321123/SimpleLean"
        commit = "99a5078e1614e61f0d9cc234ca246c8744a4e660"
        lean_git_repo = LeanGitRepo(url, commit)
        dir_name = url.split("/")[-1].replace(".git", "") + "_" + commit
        dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + dir_name + "_updated"
        config = lean_git_repo.get_config("lean-toolchain")
        v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
        data = {
            "url": lean_git_repo.url,
            "name": "/".join(lean_git_repo.url.split("/")[-2:]),
            "commit": lean_git_repo.commit,
            "lean_version": v,
            "lean_dojo_version": lean_dojo.__version__,
            "metadata": {
                "date_processed": datetime.datetime.now(),
            },
            "theorems_folder": dst_dir + "/random",
            "premise_files_corpus": dst_dir + "/corpus.jsonl",
            "files_traced": dst_dir + "/traced_files.jsonl",
        }
        repo = Repository.from_dict(data)
        return repo

    def test_empty_repo(self):
        self.assertEqual(len(self.simple_lean_repo.proven_theorems), 0)
        self.assertEqual(len(self.simple_lean_repo.sorry_theorems_proved), 0)
        self.assertEqual(len(self.simple_lean_repo.sorry_theorems_unproved), 0)

    def test_generate_dataset_with_empty_repo(self):
        dst_dir = Path(RAID_DIR) / DATA_DIR / "simple_lean_generated"
        self.db.generate_merged_dataset(dst_dir)

        self.assertTrue(dst_dir.exists())
        self.assertTrue((dst_dir / "random").exists())
        self.assertTrue((dst_dir / "novel_premises").exists())

        for split in ["train", "val", "test"]:
            with open(dst_dir / "random" / f"{split}.json", "r") as f:
                data = json.load(f)
                self.assertEqual(len(data), 0)


class TestDynamicDatabaseUnicode(unittest.TestCase):
    """ "
    Unit test class for testing Unicode handling in the DynamicDatabase class.
    This test suite focuses on verifying that the DynamicDatabase correctly handles
    Unicode characters during serialization and deserialization operations, and
    that Unicode content can be properly manipulated within the database.
    The tests include:
    1. Serializing and deserializing a database with Unicode content
    2. Modifying a theorem with Unicode content and verifying the changes persist
    The test database includes:
    - A repository with a Unicode name
    - Theorems with Unicode mathematical symbols (∀, ℕ, ℝ, etc.)
    - Complex mathematical expressions using Unicode symbols
    This ensures the database can correctly handle international character sets
    and mathematical notation when saving to and loading from JSON files.
    """

    def setUp(self):
        self.db = DynamicDatabase()
        self.unicode_repo = self.create_unicode_sample_repo()
        self.db.add_repository(self.unicode_repo)

    def assertDatetimeEqual(self, dt1, dt2):
        """
        Assert that two datetimes are equal, ignoring microseconds.
        This is important because serialization and deserialization of datetimes
        may lose microsecond precision.
        """
        self.assertEqual(dt1.replace(microsecond=0), dt2.replace(microsecond=0))

    def create_unicode_sample_repo(self):
        repo = Repository(
            url="https://github.com/example/repo",
            name="Example Repo with Unicode ユニコード",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )

        theorem1 = Theorem(
            full_name="example.commutative_addition",
            theorem_statement="∀ x y : ℕ, x + y = y + x",
            file_path=Path("src/example.lean"),
            start=Pos(1, 1),
            end=Pos(5, 10),
            url="https://github.com/example/repo",
            commit="abc123",
            traced_tactics=[
                AnnotatedTactic(
                    tactic="induction x with n ih",
                    annotated_tactic=(
                        "induction x with n ih",
                        [
                            Annotation(
                                full_name="induction",
                                def_path="src/tactic/induction.lean",
                                def_pos=Pos(100, 1),
                                def_end_pos=Pos(100, 10),
                            )
                        ],
                    ),
                    state_before="⊢ ∀ x y : ℕ, x + y = y + x",
                    state_after="2 goals\ncase zero\n⊢ ∀ y : ℕ, 0 + y = y + 0\ncase succ\nn : ℕ\nih : ∀ y : ℕ, n + y = y + n\n⊢ ∀ y : ℕ, succ n + y = y + succ n",
                )
            ],
            difficulty_rating=0.7,
        )

        theorem2 = Theorem(
            full_name="example.quadratic_formula",
            theorem_statement="∀ a b c x : ℝ, a ≠ 0 → (a * x² + b * x + c = 0 ↔ x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a))",
            file_path=Path("src/sorry_example.lean"),
            start=Pos(10, 1),
            end=Pos(12, 10),
            url="https://github.com/example/repo",
            commit="abc123",
            traced_tactics=[],
            difficulty_rating=0.9,
        )

        repo.proven_theorems.append(theorem1)
        repo.sorry_theorems_unproved.append(theorem2)

        premise_file = PremiseFile(
            path=Path("src/premise.lean"),
            imports=["import data.real.basic"],
            premises=[
                Premise(
                    full_name="sqrt_squared",
                    code="theorem sqrt_squared (x : ℝ) (h : x ≥ 0) : √(x^2) = x := sorry",
                    start=Pos(1, 1),
                    end=Pos(1, 70),
                    kind="theorem",
                )
            ],
        )

        repo.premise_files.append(premise_file)
        repo.files_traced.append(Path("src/example.lean"))
        return repo

    def test_unicode_serialization_deserialization(self):
        json_file = "test_unicode_database.json"
        self.db.to_json(json_file)

        deserialized_db = DynamicDatabase.from_json(json_file)

        assert len(self.db.repositories) == len(deserialized_db.repositories)

        original_repo = self.db.repositories[0]
        deserialized_repo = deserialized_db.repositories[0]

        assert original_repo.name == deserialized_repo.name
        self.assertDatetimeEqual(
            original_repo.metadata["date_processed"],
            deserialized_repo.metadata["date_processed"],
        )

        original_theorem1 = original_repo.proven_theorems[0]
        deserialized_theorem1 = deserialized_repo.proven_theorems[0]

        assert (
            original_theorem1.theorem_statement
            == deserialized_theorem1.theorem_statement
        )
        assert (
            original_theorem1.traced_tactics[0].state_before
            == deserialized_theorem1.traced_tactics[0].state_before
        )
        assert (
            original_theorem1.traced_tactics[0].state_after
            == deserialized_theorem1.traced_tactics[0].state_after
        )

        original_theorem2 = original_repo.sorry_theorems_unproved[0]
        deserialized_theorem2 = deserialized_repo.sorry_theorems_unproved[0]

        assert (
            original_theorem2.theorem_statement
            == deserialized_theorem2.theorem_statement
        )

        original_premise = original_repo.premise_files[0].premises[0]
        deserialized_premise = deserialized_repo.premise_files[0].premises[0]

        assert original_premise.code == deserialized_premise.code

    def test_unicode_modification(self):
        json_file = "test_unicode_database.json"
        self.db.to_json(json_file)

        deserialized_db = DynamicDatabase.from_json(json_file)

        repo = deserialized_db.get_repository(
            "https://github.com/example/repo", "abc123"
        )
        assert repo is not None

        sorry_theorem = repo.sorry_theorems_unproved[0]

        sorry_theorem.traced_tactics = [
            AnnotatedTactic(
                tactic="intros a b c x h_a_nonzero",
                annotated_tactic=("intros a b c x h_a_nonzero", []),
                state_before="⊢ ∀ a b c x : ℝ, a ≠ 0 → (a * x² + b * x + c = 0 ↔ x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a))",
                state_after="a b c x : ℝ\nh_a_nonzero : a ≠ 0\n⊢ a * x² + b * x + c = 0 ↔ x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a)",
            ),
            AnnotatedTactic(
                tactic="apply iff.intro",
                annotated_tactic=("apply iff.intro", []),
                state_before="a b c x : ℝ\nh_a_nonzero : a ≠ 0\n⊢ a * x² + b * x + c = 0 ↔ x = (-b + √(b² - 4*a*c)) / (2*a) ∨ x = (-b - √(b² - 4*a*c)) / (2*a)",
                state_after="no goals",
            ),
        ]

        repo.change_sorry_to_proven(sorry_theorem, PROOF_LOG_FILE_NAME)
        deserialized_db.update_json(json_file)
        updated_db = DynamicDatabase.from_json(json_file)
        updated_repo = updated_db.get_repository(
            "https://github.com/example/repo", "abc123"
        )
        assert updated_repo is not None

        assert len(updated_repo.sorry_theorems_unproved) == 0
        assert len(updated_repo.sorry_theorems_proved) == 1

        updated_theorem = updated_repo.sorry_theorems_proved[0]
        assert updated_theorem.full_name == "example.quadratic_formula"
        assert len(updated_theorem.traced_tactics) == 2
        assert "√(b² - 4*a*c)" in updated_theorem.traced_tactics[0].state_before
        assert "↔" in updated_theorem.traced_tactics[1].state_before


class TestDynamicDatabase(unittest.TestCase):
    """
    Test suite for the DynamicDatabase class.

    This class contains unit tests that verify the functionality of the DynamicDatabase class,
    which is responsible for managing repository data. The tests cover basic CRUD operations
    (Create, Read, Update, Delete) for repositories as well as serialization and
    deserialization of the database to and from JSON.

    Tests:
        - test_add_repository: Verifies that repositories can be added to the database
        - test_get_repository: Ensures repositories can be retrieved by URL and commit
        - test_update_repository: Tests the ability to update existing repositories
        - test_delete_repository: Confirms repositories can be removed from the database
        - test_to_json_and_from_json: Validates JSON serialization and deserialization

    Each test method uses a fresh DynamicDatabase instance and a test Repository object
    created in the setUp method.
    """

    def setUp(self):
        self.db = DynamicDatabase()
        self.repo = Repository(
            url="https://github.com/test/repo",
            name="Test Repo",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={"date_processed": datetime.datetime.now()},
        )
        self.current_datetime = datetime.datetime.now()

    def assertDatetimeEqual(self, dt1, dt2):
        """
        Assert that two datetimes are equal, ignoring microseconds.
        This is important because serialization and deserialization of datetimes
        may lose microsecond precision.
        """
        self.assertEqual(dt1.replace(microsecond=0), dt2.replace(microsecond=0))

    def test_add_repository(self):
        self.db.add_repository(self.repo)
        self.assertEqual(len(self.db.repositories), 1)

    def test_get_repository(self):
        self.db.add_repository(self.repo)
        retrieved_repo = self.db.get_repository(
            "https://github.com/test/repo", "abc123"
        )
        self.assertEqual(retrieved_repo, self.repo)

    def test_update_repository(self):
        self.db.add_repository(self.repo)
        updated_repo = Repository(
            url="https://github.com/test/repo",
            name="Test Repo",
            commit="abc123",
            lean_version="3.50.3",
            lean_dojo_version="1.8.4",
            metadata={
                "date_processed": datetime.datetime.now() + datetime.timedelta(days=1)
            },
        )
        self.db.update_repository(updated_repo)
        retrieved_repo = self.db.get_repository(
            "https://github.com/test/repo", "abc123"
        )
        self.assertEqual(retrieved_repo.name, "Test Repo")
        self.assertNotEqual(
            retrieved_repo.metadata["date_processed"].replace(microsecond=0),
            self.current_datetime.replace(microsecond=0),
        )

    def test_delete_repository(self):
        self.db.add_repository(self.repo)
        self.db.delete_repository("https://github.com/test/repo", "abc123")
        self.assertEqual(len(self.db.repositories), 0)

    def test_to_json_and_from_json(self):
        self.db.add_repository(self.repo)
        json_file = "test_database.json"
        self.db.to_json(json_file)
        loaded_db = DynamicDatabase.from_json(json_file)
        self.assertEqual(len(loaded_db.repositories), 1)
        loaded_repo = loaded_db.get_repository("https://github.com/test/repo", "abc123")
        self.assertEqual(loaded_repo.name, "Test Repo")
        self.assertDatetimeEqual(
            loaded_repo.metadata["date_processed"], self.current_datetime
        )


class TestDynamicDatabasePFR(unittest.TestCase):
    def setUp(self):
        self.db = DynamicDatabase()
        self.sample_repo = self.create_sample_repo()
        self.db.add_repository(self.sample_repo)

    def create_sample_repo(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        lean_git_repo = LeanGitRepo(url, commit)
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + dir_name + "_updated"
        config = lean_git_repo.get_config("lean-toolchain")
        v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
        theorems_folder = dst_dir + "/random"
        premise_files_corpus = dst_dir + "/corpus.jsonl"
        files_traced = dst_dir + "/traced_files.jsonl"
        pr_url = None
        data = {
            "url": lean_git_repo.url,
            "name": "/".join(lean_git_repo.url.split("/")[-2:]),
            "commit": lean_git_repo.commit,
            "lean_version": v,
            "lean_dojo_version": lean_dojo.__version__,
            "metadata": {
                "key": "value",
                "unicode": "ユニコード ✨",
                "date_processed": datetime.datetime.now(),
            },
            "theorems_folder": theorems_folder,
            "premise_files_corpus": premise_files_corpus,
            "files_traced": files_traced,
            "pr_url": pr_url,
        }
        repo = Repository.from_dict(data)
        return repo

    def test_repository_creation(self):
        self.assertIsNotNone(self.sample_repo)
        self.assertEqual(self.sample_repo.url, "https://github.com/teorth/pfr")
        self.assertEqual(
            self.sample_repo.commit, "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        )

    def test_theorem_loading(self):
        self.assertGreater(len(self.sample_repo.proven_theorems), 0)
        self.assertGreater(len(self.sample_repo.sorry_theorems_unproved), 0)

        theorem = next(
            t
            for t in self.sample_repo.proven_theorems
            if t.full_name == "ContinuousLinearMap.opNorm_lsmul"
        )
        self.assertIsNotNone(theorem)
        self.assertEqual(
            theorem.file_path,
            Path(
                ".lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Mul.lean"
            ),
        )
        self.assertEqual(theorem.start, Pos(281, 1))
        self.assertEqual(theorem.end, Pos(290, 26))

    def test_traced_tactics(self):
        theorem = next(
            t
            for t in self.sample_repo.proven_theorems
            if t.full_name == "ContinuousLinearMap.opNorm_lsmul"
        )
        self.assertGreater(len(theorem.traced_tactics), 0)

        first_tactic = theorem.traced_tactics[0]
        self.assertEqual(
            first_tactic.tactic,
            "refine' ContinuousLinearMap.opNorm_eq_of_bounds zero_le_one (fun x => _) fun N _ h => _",
        )
        self.assertIn(
            "ContinuousLinearMap.opNorm_eq_of_bounds", first_tactic.annotated_tactic[0]
        )

    def test_premise_loading(self):
        self.assertGreater(len(self.sample_repo.premise_files), 0)

        premise_file = next(
            pf
            for pf in self.sample_repo.premise_files
            if pf.path == Path(".lake/packages/lean4/src/lean/Init/Prelude.lean")
        )
        self.assertIsNotNone(premise_file)
        self.assertGreater(len(premise_file.premises), 0)

        premise = next(p for p in premise_file.premises if p.full_name == "id")
        self.assertIsNotNone(premise)
        self.assertIn("def id", premise.code)

    def test_serialization_deserialization(self):
        json_file = "test_pfr_database.json"
        self.db.to_json(json_file)

        deserialized_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(self.db.repositories), len(deserialized_db.repositories))

        original_repo = self.db.repositories[0]
        deserialized_repo = deserialized_db.repositories[0]

        self.assertEqual(original_repo.name, deserialized_repo.name)
        self.assertEqual(original_repo.commit, deserialized_repo.commit)
        self.assertEqual(
            len(original_repo.proven_theorems), len(deserialized_repo.proven_theorems)
        )
        self.assertEqual(
            len(original_repo.premise_files), len(deserialized_repo.premise_files)
        )

    def test_generate_dataset_structure(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        self.db.generate_merged_dataset(dst_dir)

        self.assertTrue(dst_dir.exists())
        self.assertTrue((dst_dir / "random").exists())
        self.assertTrue((dst_dir / "novel_premises").exists())
        self.assertTrue((dst_dir / "random" / "train.json").exists())
        self.assertTrue((dst_dir / "random" / "val.json").exists())
        self.assertTrue((dst_dir / "random" / "test.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "train.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "val.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "test.json").exists())
        self.assertTrue((dst_dir / "corpus.jsonl").exists())
        self.assertTrue((dst_dir / "traced_files.jsonl").exists())
        self.assertTrue((dst_dir / "metadata.json").exists())

    def test_generated_dataset_content(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        self.db.generate_merged_dataset(dst_dir)

        with open(dst_dir / "random" / "train.json", "r") as f:
            train_data = json.load(f)
            self.assertIsInstance(train_data, list)
            self.assertGreater(len(train_data), 0)
            first_theorem = train_data[0]
            self.assertIn("url", first_theorem)
            self.assertIn("commit", first_theorem)
            self.assertIn("file_path", first_theorem)
            self.assertIn("full_name", first_theorem)
            self.assertIn("theorem_statement", first_theorem)
            self.assertIn("start", first_theorem)
            self.assertIn("end", first_theorem)
            self.assertIn("traced_tactics", first_theorem)

        with open(dst_dir / "corpus.jsonl", "r") as f:
            first_line = f.readline().strip()
            first_premise_file = json.loads(first_line)
            self.assertIn("path", first_premise_file)
            self.assertIn("imports", first_premise_file)
            self.assertIn("premises", first_premise_file)

        with open(dst_dir / "traced_files.jsonl", "r") as f:
            first_line = f.readline().strip()
            first_traced_file = json.loads(first_line)
            self.assertIn("traced_file_path", first_traced_file)

        with open(dst_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.assertIn("repositories", metadata)
            self.assertEqual(len(metadata["repositories"]), 1)
            self.assertIn("total_theorems", metadata)
            self.assertIn("num_proven_theorems", metadata)
            self.assertIn("num_sorry_theorems", metadata)
            self.assertIn("num_premise_files", metadata)
            self.assertIn("num_premises", metadata)
            self.assertIn("num_files_traced", metadata)

            for repo in metadata["repositories"]:
                self.assertIn("url", repo)
                self.assertIn("name", repo)
                self.assertIn("commit", repo)
                self.assertIn("lean_version", repo)
                self.assertIn("lean_dojo_version", repo)
                self.assertIn("metadata", repo)

    def test_dataset_splitting(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        self.db.generate_merged_dataset(dst_dir)

        for strategy in ["random", "novel_premises"]:
            train_set = set()
            val_set = set()
            test_set = set()

            with open(dst_dir / strategy / "train.json", "r") as f:
                train_data = json.load(f)
                train_set = set(
                    (
                        item["full_name"],
                        item["file_path"],
                        tuple(item["start"]),
                        tuple(item["end"]),
                    )
                    for item in train_data
                )

            with open(dst_dir / strategy / "val.json", "r") as f:
                val_data = json.load(f)
                val_set = set(
                    (
                        item["full_name"],
                        item["file_path"],
                        tuple(item["start"]),
                        tuple(item["end"]),
                    )
                    for item in val_data
                )

            with open(dst_dir / strategy / "test.json", "r") as f:
                test_data = json.load(f)
                test_set = set(
                    (
                        item["full_name"],
                        item["file_path"],
                        tuple(item["start"]),
                        tuple(item["end"]),
                    )
                    for item in test_data
                )

            self.assertGreater(len(train_set), 0)
            self.assertGreater(len(val_set), 0)
            self.assertGreater(len(test_set), 0)

            self.assertEqual(len(train_set.intersection(val_set)), 0)
            self.assertEqual(len(train_set.intersection(test_set)), 0)
            self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_dataset_consistency(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        self.db.generate_merged_dataset(dst_dir)

        # Check that all theorems in the dataset are from the original repository
        all_theorems = set(thm.full_name for thm in self.sample_repo.get_all_theorems)

        for strategy in ["random", "novel_premises"]:
            for split in ["train", "val", "test"]:
                with open(dst_dir / strategy / f"{split}.json", "r") as f:
                    data = json.load(f)
                    for item in data:
                        self.assertIn(item["full_name"], all_theorems)

    def test_compare_manual_and_dynamic_datasets(self):
        random.seed(3407)

        manual_dataset_path = (
            Path(RAID_DIR)
            / DATA_DIR
            / "pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_updated"
        )
        dynamic_dataset_path = (
            Path(RAID_DIR)
            / DATA_DIR
            / "pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_generated"
        )

        self.db.generate_merged_dataset(dynamic_dataset_path)

        for strategy in ["random", "novel_premises"]:
            logger.info(f"Comparing datasets for {strategy} strategy")
            manual_theorems = []
            dynamic_theorems = []

            for split in ["train", "val", "test"]:
                logger.info(f"Loading {split} split for {strategy} strategy")
                manual_file = manual_dataset_path / strategy / f"{split}.json"
                dynamic_file = dynamic_dataset_path / strategy / f"{split}.json"

                with open(manual_file, "r") as f:
                    manual_data = json.load(f)
                    manual_theorems.extend(manual_data)
                logger.info(
                    f"Loaded {len(manual_data)} theorems from manual {split} split"
                )

                with open(dynamic_file, "r") as f:
                    dynamic_data = json.load(f)
                    dynamic_theorems.extend(dynamic_data)
                logger.info(
                    f"Loaded {len(dynamic_data)} theorems from dynamic {split} split"
                )

            assert len(manual_theorems) == len(
                dynamic_theorems
            ), "Manual and dynamic datasets have different number of theorems"
            logger.info(
                f"Comparing {len(manual_theorems)} manual theorems with {len(dynamic_theorems)} dynamic theorems for {strategy} strategy"
            )
            self.assertTrue(
                self._fast_compare_theorems(manual_theorems, dynamic_theorems),
                f"Theorem content for {strategy} strategy does not match",
            )
            logger.info(f"Theorem content for {strategy} strategy matches")

        self.maxDiff = None
        logger.info("Comparing corpus and traced files")
        with open(manual_dataset_path / "corpus.jsonl", "r") as f:
            manual_corpus = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(manual_corpus)} items from manual corpus")

        with open(dynamic_dataset_path / "corpus.jsonl", "r") as f:
            dynamic_corpus = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(dynamic_corpus)} items from dynamic corpus")

        assert len(manual_corpus) == len(
            dynamic_corpus
        ), "Manual and dynamic datasets have different number of premise files"
        logger.info("Comparing corpus content")
        try:
            self.assertCountEqual(manual_corpus, dynamic_corpus)
            logger.info("Corpus content matches")
        except AssertionError as e:
            logger.info("Corpus content mismatch:")
            logger.info(str(e))
            raise

        with open(manual_dataset_path / "traced_files.jsonl", "r") as f:
            manual_traced = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(manual_traced)} items from manual traced files")

        with open(dynamic_dataset_path / "traced_files.jsonl", "r") as f:
            dynamic_traced = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(dynamic_traced)} items from dynamic traced files")

        assert len(manual_traced) == len(
            dynamic_traced
        ), "Manual and dynamic datasets have different number of traced files"
        logger.info("Comparing traced files content")
        try:
            self.assertCountEqual(manual_traced, dynamic_traced)
            logger.info("Traced files content matches")
        except AssertionError as e:
            logger.info("Traced files content mismatch:")
            logger.info(str(e))
            raise

    def _fast_compare_theorems(self, manual_theorems, dynamic_theorems):
        logger.info(
            f"Converting {len(manual_theorems)} manual theorems to hashable format"
        )
        manual_set = set(map(self._theorem_to_hashable, manual_theorems))
        assert len(manual_set) == len(
            manual_theorems
        ), "Manual theorems contain duplicates"
        logger.info(
            f"Converting {len(dynamic_theorems)} dynamic theorems to hashable format"
        )
        dynamic_set = set(map(self._theorem_to_hashable, dynamic_theorems))
        assert len(dynamic_set) == len(
            dynamic_theorems
        ), "Dynamic theorems contain duplicates"

        logger.info("Comparing theorem sets")
        only_in_manual = manual_set - dynamic_set
        only_in_dynamic = dynamic_set - manual_set

        if only_in_manual or only_in_dynamic:
            # Sort by file path and full name
            only_in_manual = sorted(only_in_manual, key=lambda x: (x[0], x[1]))
            only_in_dynamic = sorted(only_in_dynamic, key=lambda x: (x[0], x[1]))
            if only_in_manual:
                logger.info(f"{len(only_in_manual)} theorems only in manual dataset")
                for i, thm in enumerate(only_in_manual):
                    if i >= 10:
                        break
                    logger.info(f"Manual only: {thm[1]} in {thm[0]}")
                    logger.info(f"  Start: {thm[2]}, End: {thm[3]}")
            if only_in_dynamic:
                logger.info(f"{len(only_in_dynamic)} theorems only in dynamic dataset")
                for i, thm in enumerate(only_in_dynamic):
                    if i >= 10:
                        break
                    logger.info(f"Dynamic only: {thm[1]} in {thm[0]}")
                    logger.info(f"  Start: {thm[2]}, End: {thm[3]}")
            return False
        return True

    def _theorem_to_hashable(self, theorem):
        return (
            theorem["file_path"],
            theorem["full_name"],
            tuple(theorem["start"]),
            tuple(theorem["end"]),
        )

    def _tactic_to_hashable(self, tactic):
        return (
            tactic["tactic"],
            tactic["annotated_tactic"][0],
            tuple(
                (
                    a["full_name"],
                    a["def_path"],
                    tuple(a["def_pos"]),
                    tuple(a["def_end_pos"]),
                )
                for a in tactic["annotated_tactic"][1]
            ),
            tactic["state_before"],
            tactic["state_after"],
        )

    def test_unicode_handling_in_dataset(self):
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        self.db.generate_merged_dataset(dst_dir)

        with open(dst_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
            self.assertIn("repositories", metadata, "No 'repositories' key in metadata")
            self.assertGreater(
                len(metadata["repositories"]), 0, "No repositories in metadata"
            )
            repo = metadata["repositories"][0]
            self.assertIn("metadata", repo, "No 'metadata' key in repository")
            repo_metadata = repo["metadata"]
            self.assertIn(
                "unicode", repo_metadata, "No 'unicode' key in repository metadata"
            )
            self.assertIn(
                "ユニコード",
                repo_metadata["unicode"],
                "Unicode string not found in metadata",
            )
            self.assertIn(
                "ユニコード", metadata["repositories"][0]["metadata"]["unicode"]
            )

    def tearDown(self):
        # Clean up generated files after tests
        url = "https://github.com/teorth/pfr"
        commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = Path(RAID_DIR) / DATA_DIR / f"{dir_name}_generated"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)

    def test_theorem_statement(self):
        theorem = next(
            t
            for t in self.sample_repo.proven_theorems
            if t.full_name == "ContinuousLinearMap.opNorm_lsmul"
        )
        self.assertIsNotNone(theorem.theorem_statement)
        self.assertIn("opNorm_lsmul", theorem.theorem_statement)

    def test_unicode_handling(self):
        self.assertIsNotNone(self.sample_repo.metadata)
        self.assertIsNotNone(self.sample_repo.metadata["unicode"])
        self.assertIn("ユニコード", self.sample_repo.metadata["unicode"])

    def test_file_tracing(self):
        self.assertGreater(len(self.sample_repo.files_traced), 0)
        self.assertIn(
            Path("PFR/Mathlib/GroupTheory/Torsion.lean"), self.sample_repo.files_traced
        )
        self.assertIn(
            Path(".lake/packages/batteries/Batteries/Data/List/Lemmas.lean"),
            self.sample_repo.files_traced,
        )


class TestDynamicDatabasePFRNewVersion(unittest.TestCase):
    """
    Test suite for the DynamicDatabase class when working with multiple repositories.
    This test class validates the functionality of the DynamicDatabase class
    for managing multiple Lean repositories simultaneously, specifically the PFR
    repository and a new-version-test repository. It tests repository creation,
    theorem loading, tactic tracing, premise loading, serialization/deserialization,
    dataset generation, and comparison between manual and dynamic dataset generation.
    The test suite verifies:
    1. Repository creation and loading from different sources
    2. Correct loading of theorems, tactics, and premises
    3. JSON serialization and deserialization of the database
    4. Generation of merged datasets from multiple repositories
    5. Consistency and correctness of generated datasets
    6. Proper handling of unicode characters
    7. File tracing functionality
    8. Dataset splitting into train/val/test sets
    It also compares dynamically generated datasets with manually created ones
    to ensure compatibility and correctness.
    """

    def setUp(self):
        self.db = DynamicDatabase()
        self.sample_repo_PFR = self.create_sample_repo(
            "https://github.com/teorth/pfr", "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        )
        self.sample_repo_new_version = self.create_sample_repo(
            "https://github.com/Adarsh321123/new-version-test",
            "f465306be03ced999caa157a85558a6c41b3e3f5",
        )
        self.db.add_repository(self.sample_repo_PFR)
        self.db.add_repository(self.sample_repo_new_version)

    def create_sample_repo(self, url, commit):
        lean_git_repo = LeanGitRepo(url, commit)
        dir_name = url.split("/")[-1] + "_" + commit
        dst_dir = RAID_DIR + "/" + DATA_DIR + "/" + dir_name + "_updated"
        config = lean_git_repo.get_config("lean-toolchain")
        v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
        data = {
            "url": lean_git_repo.url,
            "name": "/".join(lean_git_repo.url.split("/")[-2:]),
            "commit": lean_git_repo.commit,
            "lean_version": v,
            "lean_dojo_version": lean_dojo.__version__,
            "metadata": {
                "key": "value",
                "unicode": "ユニコード ✨",
                "date_processed": datetime.datetime.now(),
            },
            "theorems_folder": dst_dir + "/random",
            "premise_files_corpus": dst_dir + "/corpus.jsonl",
            "files_traced": dst_dir + "/traced_files.jsonl",
        }
        repo = Repository.from_dict(data)
        return repo

    def test_multiple_json_serialization_deserialization(self):
        json_file1 = "test_multiple_1.json"
        json_file2 = "test_multiple_2.json"

        # First serialization
        self.db.to_json(json_file1)

        # Deserialize and modify
        loaded_db1 = DynamicDatabase.from_json(json_file1)
        new_repo = Repository(
            url="https://github.com/test/new-repo",
            name="New Test Repo",
            commit="newcommit123",
            lean_version="4.0.0",
            lean_dojo_version="1.0.0",
            metadata={"date_processed": datetime.datetime.now()},
        )
        loaded_db1.add_repository(new_repo)

        # Second serialization
        loaded_db1.to_json(json_file2)

        # Final deserialization
        loaded_db2 = DynamicDatabase.from_json(json_file2)

        self.assertEqual(
            len(loaded_db2.repositories), 3
        )  # PFR, new-version-test, and new-repo
        self.assertEqual(
            loaded_db2.repositories[2].url, "https://github.com/test/new-repo"
        )
        self.assertEqual(loaded_db2.repositories[2].commit, "newcommit123")

        # Check that the original repositories are still intact
        self.assertEqual(
            loaded_db2.repositories[0].url, "https://github.com/teorth/pfr"
        )
        self.assertEqual(
            loaded_db2.repositories[1].url,
            "https://github.com/Adarsh321123/new-version-test",
        )

        # Verify that the content of the repositories is preserved
        pfr_repo = loaded_db2.get_repository(
            "https://github.com/teorth/pfr", "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        )
        self.assertIsNotNone(pfr_repo)
        self.assertGreater(len(pfr_repo.proven_theorems), 0)
        self.assertGreater(len(pfr_repo.sorry_theorems_unproved), 0)

        new_version_repo = loaded_db2.get_repository(
            "https://github.com/Adarsh321123/new-version-test",
            "f465306be03ced999caa157a85558a6c41b3e3f5",
        )
        self.assertIsNotNone(new_version_repo)
        self.assertGreater(len(new_version_repo.proven_theorems), 0)
        self.assertGreater(len(new_version_repo.sorry_theorems_unproved), 0)

    def test_repository_creation(self):
        self.assertIsNotNone(self.sample_repo_PFR)
        self.assertEqual(self.sample_repo_PFR.url, "https://github.com/teorth/pfr")
        self.assertEqual(
            self.sample_repo_PFR.commit, "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        )
        self.assertIsNotNone(self.sample_repo_new_version)
        self.assertEqual(
            self.sample_repo_new_version.url,
            "https://github.com/Adarsh321123/new-version-test",
        )
        self.assertEqual(
            self.sample_repo_new_version.commit,
            "f465306be03ced999caa157a85558a6c41b3e3f5",
        )

    def test_theorem_loading(self):
        self.assertGreater(len(self.sample_repo_PFR.proven_theorems), 0)
        self.assertGreater(len(self.sample_repo_PFR.sorry_theorems_unproved), 0)

        theorem = next(
            t
            for t in self.sample_repo_PFR.proven_theorems
            if t.full_name == "ContinuousLinearMap.opNorm_lsmul"
        )
        self.assertIsNotNone(theorem)
        self.assertEqual(
            theorem.file_path,
            Path(
                ".lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Mul.lean"
            ),
        )
        self.assertEqual(theorem.start, Pos(281, 1))
        self.assertEqual(theorem.end, Pos(290, 26))

        self.assertGreater(len(self.sample_repo_new_version.proven_theorems), 0)
        self.assertGreater(len(self.sample_repo_new_version.sorry_theorems_unproved), 0)

        theorem = next(
            t
            for t in self.sample_repo_new_version.proven_theorems
            if t.full_name == "Ordinal.le_mul_right"
        )
        self.assertIsNotNone(theorem)
        self.assertEqual(
            theorem.file_path,
            Path(".lake/packages/mathlib/Mathlib/SetTheory/Ordinal/Arithmetic.lean"),
        )
        self.assertEqual(theorem.start, Pos(742, 1))
        self.assertEqual(theorem.end, Pos(744, 17))

    def test_traced_tactics(self):
        theorem = next(
            t
            for t in self.sample_repo_PFR.proven_theorems
            if t.full_name == "ContinuousLinearMap.opNorm_lsmul"
        )
        self.assertGreater(len(theorem.traced_tactics), 0)

        first_tactic = theorem.traced_tactics[0]
        self.assertEqual(
            first_tactic.tactic,
            "refine' ContinuousLinearMap.opNorm_eq_of_bounds zero_le_one (fun x => _) fun N _ h => _",
        )
        self.assertIn(
            "ContinuousLinearMap.opNorm_eq_of_bounds", first_tactic.annotated_tactic[0]
        )

        theorem = next(
            t
            for t in self.sample_repo_new_version.proven_theorems
            if t.full_name == "Ordinal.le_mul_right"
        )
        self.assertGreater(len(theorem.traced_tactics), 0)

        first_tactic = theorem.traced_tactics[0]
        self.assertEqual(
            first_tactic.tactic, "convert mul_le_mul_right' (one_le_iff_pos.2 hb) a"
        )
        self.assertIn("mul_le_mul_right'", first_tactic.annotated_tactic[0])

    def test_premise_loading(self):
        self.assertGreater(len(self.sample_repo_PFR.premise_files), 0)

        premise_file = next(
            pf
            for pf in self.sample_repo_PFR.premise_files
            if pf.path == Path(".lake/packages/lean4/src/lean/Init/Prelude.lean")
        )
        self.assertIsNotNone(premise_file)
        self.assertGreater(len(premise_file.premises), 0)

        premise = next(p for p in premise_file.premises if p.full_name == "id")
        self.assertIsNotNone(premise)
        self.assertIn("def id", premise.code)

        self.assertGreater(len(self.sample_repo_new_version.premise_files), 0)

        premise_file = next(
            pf
            for pf in self.sample_repo_new_version.premise_files
            if pf.path == Path(".lake/packages/lean4/src/lean/Init/Prelude.lean")
        )
        self.assertIsNotNone(premise_file)
        self.assertGreater(len(premise_file.premises), 0)

        premise = next(p for p in premise_file.premises if p.full_name == "id")
        self.assertIsNotNone(premise)
        self.assertIn("def id", premise.code)

    def test_serialization_deserialization(self):
        json_file = "test_pfr_new_version_database.json"
        self.db.to_json(json_file)

        deserialized_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(self.db.repositories), len(deserialized_db.repositories))

        original_repo_PFR = self.db.repositories[0]
        deserialized_repo_PFR = deserialized_db.repositories[0]

        self.assertEqual(original_repo_PFR.name, deserialized_repo_PFR.name)
        self.assertEqual(original_repo_PFR.commit, deserialized_repo_PFR.commit)
        self.assertEqual(
            len(original_repo_PFR.proven_theorems),
            len(deserialized_repo_PFR.proven_theorems),
        )
        self.assertEqual(
            len(original_repo_PFR.premise_files),
            len(deserialized_repo_PFR.premise_files),
        )

        original_repo_new_version = self.db.repositories[0]
        deserialized_repo_new_version = deserialized_db.repositories[0]

        self.assertEqual(
            original_repo_new_version.name, deserialized_repo_new_version.name
        )
        self.assertEqual(
            original_repo_new_version.commit, deserialized_repo_new_version.commit
        )
        self.assertEqual(
            len(original_repo_new_version.proven_theorems),
            len(deserialized_repo_new_version.proven_theorems),
        )
        self.assertEqual(
            len(original_repo_new_version.premise_files),
            len(deserialized_repo_new_version.premise_files),
        )

    def test_generate_dataset_structure(self):
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = (
            Path(RAID_DIR)
            / DATA_DIR
            / f"{dir_name_PFR}_{dir_name_new_version}_generated"
        )
        self.db.generate_merged_dataset(dst_dir)

        self.assertTrue(dst_dir.exists())
        self.assertTrue((dst_dir / "random").exists())
        self.assertTrue((dst_dir / "novel_premises").exists())
        self.assertTrue((dst_dir / "random" / "train.json").exists())
        self.assertTrue((dst_dir / "random" / "val.json").exists())
        self.assertTrue((dst_dir / "random" / "test.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "train.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "val.json").exists())
        self.assertTrue((dst_dir / "novel_premises" / "test.json").exists())
        self.assertTrue((dst_dir / "corpus.jsonl").exists())
        self.assertTrue((dst_dir / "traced_files.jsonl").exists())
        self.assertTrue((dst_dir / "metadata.json").exists())

    def test_generated_dataset_content(self):
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = (
            Path(RAID_DIR)
            / DATA_DIR
            / f"{dir_name_PFR}_{dir_name_new_version}_generated"
        )
        self.db.generate_merged_dataset(dst_dir)

        # Count sorry theorems in each repository
        repo_sorry_counts = {
            self.sample_repo_PFR.url: len(self.sample_repo_PFR.sorry_theorems_unproved),
            self.sample_repo_new_version.url: len(
                self.sample_repo_new_version.sorry_theorems_unproved
            ),
        }
        total_sorry_theorems = sum(repo_sorry_counts.values())

        # Count sorry theorems in the generated dataset
        dataset_sorry_count = 0
        for split in ["train", "val", "test"]:
            with open(dst_dir / "random" / f"{split}.json", "r") as f:
                data = json.load(f)
                for theorem in data:
                    if any(
                        tactic.get("tactic") == "sorry"
                        for tactic in theorem.get("traced_tactics", [])
                    ):
                        dataset_sorry_count += 1

        self.assertEqual(
            dataset_sorry_count,
            total_sorry_theorems,
            f"Number of sorry theorems in dataset ({dataset_sorry_count}) does not match "
            f"the sum from individual repositories ({total_sorry_theorems})",
        )

        with open(dst_dir / "random" / "train.json", "r") as f:
            train_data = json.load(f)
            self.assertIsInstance(train_data, list)
            self.assertGreater(len(train_data), 0)
            first_theorem = train_data[0]
            self.assertIn("url", first_theorem)
            self.assertIn("commit", first_theorem)
            self.assertIn("file_path", first_theorem)
            self.assertIn("full_name", first_theorem)
            self.assertIn("theorem_statement", first_theorem)
            self.assertIn("start", first_theorem)
            self.assertIn("end", first_theorem)
            self.assertIn("traced_tactics", first_theorem)

        with open(dst_dir / "corpus.jsonl", "r") as f:
            first_line = f.readline().strip()
            first_premise_file = json.loads(first_line)
            self.assertIn("path", first_premise_file)
            self.assertIn("imports", first_premise_file)
            self.assertIn("premises", first_premise_file)

        with open(dst_dir / "traced_files.jsonl", "r") as f:
            first_line = f.readline().strip()
            first_traced_file = json.loads(first_line)
            self.assertIn("traced_file_path", first_traced_file)

        with open(dst_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.assertIn("repositories", metadata)
            self.assertEqual(len(metadata["repositories"]), 2)
            self.assertIn("total_theorems", metadata)
            self.assertIn("num_proven_theorems", metadata)
            self.assertIn("num_sorry_theorems", metadata)
            self.assertIn("num_premise_files", metadata)
            self.assertIn("num_premises", metadata)
            self.assertIn("num_files_traced", metadata)

            # Check if the total number of sorry theorems in metadata matches our count
            self.assertEqual(
                metadata["num_sorry_theorems"],
                total_sorry_theorems,
                f"Number of sorry theorems in metadata ({metadata['num_sorry_theorems']}) "
                f"does not match the sum from individual repositories ({total_sorry_theorems})",
            )

            for repo in metadata["repositories"]:
                self.assertIn("url", repo)
                self.assertIn("name", repo)
                self.assertIn("commit", repo)
                self.assertIn("lean_version", repo)
                self.assertIn("lean_dojo_version", repo)
                self.assertIn("metadata", repo)

    def test_dataset_splitting(self):
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = (
            Path(RAID_DIR)
            / DATA_DIR
            / f"{dir_name_PFR}_{dir_name_new_version}_generated"
        )
        self.db.generate_merged_dataset(dst_dir)

        for strategy in ["random", "novel_premises"]:
            train_set = set()
            val_set = set()
            test_set = set()

            with open(dst_dir / strategy / "train.json", "r") as f:
                train_data = json.load(f)
                train_set = set(
                    (
                        item["full_name"],
                        item["file_path"],
                        tuple(item["start"]),
                        tuple(item["end"]),
                    )
                    for item in train_data
                )

            with open(dst_dir / strategy / "val.json", "r") as f:
                val_data = json.load(f)
                val_set = set(
                    (
                        item["full_name"],
                        item["file_path"],
                        tuple(item["start"]),
                        tuple(item["end"]),
                    )
                    for item in val_data
                )

            with open(dst_dir / strategy / "test.json", "r") as f:
                test_data = json.load(f)
                test_set = set(
                    (
                        item["full_name"],
                        item["file_path"],
                        tuple(item["start"]),
                        tuple(item["end"]),
                    )
                    for item in test_data
                )

            self.assertGreater(len(train_set), 0)
            self.assertGreater(len(val_set), 0)
            self.assertGreater(len(test_set), 0)

            self.assertEqual(len(train_set.intersection(val_set)), 0)
            self.assertEqual(len(train_set.intersection(test_set)), 0)
            self.assertEqual(len(val_set.intersection(test_set)), 0)

    def test_dataset_consistency(self):
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = (
            Path(RAID_DIR)
            / DATA_DIR
            / f"{dir_name_PFR}_{dir_name_new_version}_generated"
        )
        self.db.generate_merged_dataset(dst_dir)

        # Check that all theorems in the dataset are from the original repositories
        all_theorems_PFR = set(
            thm.full_name for thm in self.sample_repo_PFR.get_all_theorems
        )
        all_theorems_new_version = set(
            thm.full_name for thm in self.sample_repo_new_version.get_all_theorems
        )

        for strategy in ["random", "novel_premises"]:
            for split in ["train", "val", "test"]:
                with open(dst_dir / strategy / f"{split}.json", "r") as f:
                    data = json.load(f)
                    for item in data:
                        self.assertIn(
                            item["full_name"],
                            all_theorems_PFR | all_theorems_new_version,
                        )

    def test_generate_dataset_with_specific_repo(self):
        dynamic_dataset_path = Path(RAID_DIR) / DATA_DIR / "pfr_only_generated"
        self.db.generate_merged_dataset(
            dynamic_dataset_path,
            repos_to_include=[(self.sample_repo_PFR.url, self.sample_repo_PFR.commit)],
        )

        self.assertTrue(dynamic_dataset_path.exists())
        self.assertTrue((dynamic_dataset_path / "random").exists())
        self.assertTrue((dynamic_dataset_path / "novel_premises").exists())

        with open(dynamic_dataset_path / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.assertEqual(len(metadata["repositories"]), 1)
            self.assertEqual(
                metadata["repositories"][0]["url"], self.sample_repo_PFR.url
            )

        # Compare with the original PFR dataset
        manual_dataset_path = (
            Path(RAID_DIR)
            / DATA_DIR
            / "pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_updated"
        )

        for strategy in ["random", "novel_premises"]:
            logger.info(f"Comparing datasets for {strategy} strategy")
            manual_theorems = []
            dynamic_theorems = []

            for split in ["train", "val", "test"]:
                logger.info(f"Loading {split} split for {strategy} strategy")
                manual_file = manual_dataset_path / strategy / f"{split}.json"
                dynamic_file = dynamic_dataset_path / strategy / f"{split}.json"

                with open(manual_file, "r") as f:
                    manual_data = json.load(f)
                    manual_theorems.extend(manual_data)
                logger.info(
                    f"Loaded {len(manual_data)} theorems from manual {split} split"
                )

                with open(dynamic_file, "r") as f:
                    dynamic_data = json.load(f)
                    dynamic_theorems.extend(dynamic_data)
                logger.info(
                    f"Loaded {len(dynamic_data)} theorems from dynamic {split} split"
                )

            assert len(manual_theorems) == len(
                dynamic_theorems
            ), "Manual and dynamic datasets have different number of theorems"
            logger.info(
                f"Comparing {len(manual_theorems)} manual theorems with {len(dynamic_theorems)} dynamic theorems for {strategy} strategy"
            )
            self.assertTrue(
                self._fast_compare_theorems(manual_theorems, dynamic_theorems),
                f"Theorem content for {strategy} strategy does not match",
            )
            logger.info(f"Theorem content for {strategy} strategy matches")

        self.maxDiff = None
        logger.info("Comparing corpus and traced files")
        with open(manual_dataset_path / "corpus.jsonl", "r") as f:
            manual_corpus = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(manual_corpus)} items from manual corpus")

        with open(dynamic_dataset_path / "corpus.jsonl", "r") as f:
            dynamic_corpus = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(dynamic_corpus)} items from dynamic corpus")

        assert len(manual_corpus) == len(
            dynamic_corpus
        ), "Manual and dynamic datasets have different number of premise files"
        logger.info("Comparing corpus content")
        try:
            self.assertCountEqual(manual_corpus, dynamic_corpus)
            logger.info("Corpus content matches")
        except AssertionError as e:
            logger.info("Corpus content mismatch:")
            logger.info(str(e))
            raise

        with open(manual_dataset_path / "traced_files.jsonl", "r") as f:
            manual_traced = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(manual_traced)} items from manual traced files")

        with open(dynamic_dataset_path / "traced_files.jsonl", "r") as f:
            dynamic_traced = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(dynamic_traced)} items from dynamic traced files")

        assert len(manual_traced) == len(
            dynamic_traced
        ), "Manual and dynamic datasets have different number of traced files"
        logger.info("Comparing traced files content")
        try:
            self.assertCountEqual(manual_traced, dynamic_traced)
            logger.info("Traced files content matches")
        except AssertionError as e:
            logger.info("Traced files content mismatch:")
            logger.info(str(e))
            raise

    def test_compare_manual_and_dynamic_datasets(self):
        random.seed(3407)

        manual_dataset_path = (
            Path(RAID_DIR)
            / MERGED_DATA_DIR
            / "merged_pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_new-version-test_f465306be03ced999caa157a85558a6c41b3e3f5_updated"
        )
        dynamic_dataset_path = (
            Path(RAID_DIR)
            / MERGED_DATA_DIR
            / "merged_pfr_6a5082ee465f9e44cea479c7b741b3163162bb7e_new-version-test_f465306be03ced999caa157a85558a6c41b3e3f5_generated"
        )

        self.db.generate_merged_dataset(dynamic_dataset_path)

        for strategy in ["random", "novel_premises"]:
            logger.info(f"Comparing datasets for {strategy} strategy")
            manual_theorems = []
            dynamic_theorems = []

            for split in ["train", "val", "test"]:
                logger.info(f"Loading {split} split for {strategy} strategy")
                manual_file = manual_dataset_path / strategy / f"{split}.json"
                dynamic_file = dynamic_dataset_path / strategy / f"{split}.json"

                with open(manual_file, "r") as f:
                    manual_data = json.load(f)
                    manual_theorems.extend(manual_data)
                logger.info(
                    f"Loaded {len(manual_data)} theorems from manual {split} split"
                )

                with open(dynamic_file, "r") as f:
                    dynamic_data = json.load(f)
                    dynamic_theorems.extend(dynamic_data)
                logger.info(
                    f"Loaded {len(dynamic_data)} theorems from dynamic {split} split"
                )

            logger.info(
                f"Comparing {len(manual_theorems)} manual theorems with {len(dynamic_theorems)} dynamic theorems for {strategy} strategy"
            )

            # The manual code has a bug where it allows duplicate theorems as long as they exist in different repositories.
            # As such, we need to remove these duplicates.
            manual_dict = {}
            manual_duplicates = []
            for thm in manual_theorems:
                key = self._theorem_to_key(thm)
                if key in manual_dict:
                    manual_duplicates.append((key, thm, manual_dict[key]))
                else:
                    manual_dict[key] = thm
            deduplicated_manual_theorems = list(manual_dict.values())

            dynamic_dict = {self._theorem_to_key(t): t for t in dynamic_theorems}

            logger.info(
                f"After deduplication - Manual theorems: {len(deduplicated_manual_theorems)}, Dynamic theorems: {len(dynamic_theorems)}"
            )

            only_in_manual = set(manual_dict.keys()) - set(dynamic_dict.keys())
            only_in_dynamic = set(dynamic_dict.keys()) - set(manual_dict.keys())

            if only_in_manual:
                logger.error(
                    f"{len(only_in_manual)} theorems only in manual dataset for {strategy}"
                )
                for key in list(only_in_manual)[:1]:
                    manual_thm = manual_dict[key]
                    logger.error(
                        f"Manual only: {manual_thm['full_name']} in {manual_thm['file_path']}"
                    )
                    logger.error(
                        f"  URL: {manual_thm['url']}, Commit: {manual_thm['commit']}"
                    )
                    logger.error(
                        f"  Start: {manual_thm['start']}, End: {manual_thm['end']}"
                    )
                    logger.error(
                        f"  Theorem statement: {manual_thm['theorem_statement'][:100]}..."
                    )  # First 100 chars

            if only_in_dynamic:
                logger.error(
                    f"{len(only_in_dynamic)} theorems only in dynamic dataset for {strategy}"
                )
                for key in list(only_in_dynamic)[:1]:
                    dynamic_thm = dynamic_dict[key]
                    logger.error(
                        f"Dynamic only: {dynamic_thm['full_name']} in {dynamic_thm['file_path']}"
                    )
                    logger.error(
                        f"  URL: {dynamic_thm['url']}, Commit: {dynamic_thm['commit']}"
                    )
                    logger.error(
                        f"  Start: {dynamic_thm['start']}, End: {dynamic_thm['end']}"
                    )
                    logger.error(
                        f"  Theorem statement: {dynamic_thm['theorem_statement'][:100]}..."
                    )  # First 100 chars

            self.assertEqual(
                len(only_in_manual),
                0,
                f"Theorems found only in manual dataset for {strategy}",
            )
            self.assertEqual(
                len(only_in_dynamic),
                0,
                f"Theorems found only in dynamic dataset for {strategy}",
            )

            assert len(set(manual_dict.keys())) == len(
                set(dynamic_dict.keys())
            ), "Manual and dynamic datasets have different number of theorems"
            self.assertTrue(
                self._fast_compare_theorems(
                    deduplicated_manual_theorems, dynamic_theorems
                ),
                f"Theorem content for {strategy} strategy does not match",
            )
            logger.info(
                f"Theorem content for {strategy} strategy matches after deduplication"
            )

        self.maxDiff = None
        logger.info("Comparing corpus and traced files")
        with open(manual_dataset_path / "corpus.jsonl", "r") as f:
            manual_corpus = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(manual_corpus)} items from manual corpus")

        with open(dynamic_dataset_path / "corpus.jsonl", "r") as f:
            dynamic_corpus = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(dynamic_corpus)} items from dynamic corpus")

        manual_corpus_dict = {item["path"]: item for item in manual_corpus}
        deduplicated_manual_corpus = list(manual_corpus_dict.values())
        dynamic_corpus_dict = {item["path"]: item for item in dynamic_corpus}

        logger.info(
            f"Manual corpus: {len(manual_corpus)} items, {len(deduplicated_manual_corpus)} unique"
        )
        logger.info(f"Dynamic corpus: {len(dynamic_corpus)} items")

        only_in_manual_corpus = set(manual_corpus_dict.keys()) - set(
            dynamic_corpus_dict.keys()
        )
        only_in_dynamic_corpus = set(dynamic_corpus_dict.keys()) - set(
            manual_corpus_dict.keys()
        )

        self.assertEqual(
            len(only_in_manual_corpus), 0, "Corpus items found only in manual dataset"
        )
        self.assertEqual(
            len(only_in_dynamic_corpus), 0, "Corpus items found only in dynamic dataset"
        )

        assert len(set(dynamic_corpus_dict.keys())) == len(
            set(dynamic_corpus_dict.keys())
        ), "Manual and dynamic datasets have different number of premise files"
        # Since we choose the first processed premise file in the case of duplicates,
        # we can't compare the lines directly
        logger.info("Comparing corpus content")
        try:
            # Check that the paths are the same in both datasets
            manual_paths = set(manual_corpus_dict.keys())
            dynamic_paths = set(item["path"] for item in dynamic_corpus)
            self.assertEqual(
                manual_paths,
                dynamic_paths,
                "Paths in manual and dynamic corpus do not match",
            )
            logger.info("Corpus content matches after deduplication")
        except AssertionError as e:
            logger.info("Corpus content mismatch:")
            logger.info(str(e))
            raise

        with open(manual_dataset_path / "traced_files.jsonl", "r") as f:
            manual_traced = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(manual_traced)} items from manual traced files")

        with open(dynamic_dataset_path / "traced_files.jsonl", "r") as f:
            dynamic_traced = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(dynamic_traced)} items from dynamic traced files")

        manual_traced_dict = {item["traced_file_path"]: item for item in manual_traced}
        deduplicated_manual_traced = list(manual_traced_dict.values())
        logger.info(
            f"Manual traced files: {len(manual_traced)} items, {len(deduplicated_manual_traced)} unique"
        )
        logger.info(f"Dynamic traced files: {len(dynamic_traced)} items")

        logger.info("Comparing traced files content")
        try:
            self.assertCountEqual(deduplicated_manual_traced, dynamic_traced)
            logger.info("Traced files content matches after deduplication")
        except AssertionError as e:
            logger.info("Traced files content mismatch:")
            logger.info(str(e))
            raise

    def _theorem_to_key(self, theorem):
        return (
            theorem["file_path"],
            theorem["full_name"],
            tuple(theorem["start"]),
            tuple(theorem["end"]),
        )

    def _fast_compare_theorems(self, manual_theorems, dynamic_theorems):
        logger.info(
            f"Converting {len(manual_theorems)} manual theorems to hashable format"
        )
        manual_set = set(map(self._theorem_to_hashable, manual_theorems))
        assert len(manual_set) == len(
            manual_theorems
        ), "Manual theorems contain duplicates"
        logger.info(
            f"Converting {len(dynamic_theorems)} dynamic theorems to hashable format"
        )
        dynamic_set = set(map(self._theorem_to_hashable, dynamic_theorems))
        assert len(dynamic_set) == len(
            dynamic_theorems
        ), "Dynamic theorems contain duplicates"

        logger.info("Comparing theorem sets")
        only_in_manual = manual_set - dynamic_set
        only_in_dynamic = dynamic_set - manual_set

        if only_in_manual or only_in_dynamic:
            # Sort by file path and full name
            only_in_manual = sorted(only_in_manual, key=lambda x: (x[0], x[1]))
            only_in_dynamic = sorted(only_in_dynamic, key=lambda x: (x[0], x[1]))
            if only_in_manual:
                logger.info(f"{len(only_in_manual)} theorems only in manual dataset")
                for i, thm in enumerate(only_in_manual):
                    if i >= 10:
                        break
                    logger.info(f"Manual only: {thm[1]} in {thm[0]}")
                    logger.info(f"  Start: {thm[2]}, End: {thm[3]}")
            if only_in_dynamic:
                logger.info(f"{len(only_in_dynamic)} theorems only in dynamic dataset")
                for i, thm in enumerate(only_in_dynamic):
                    if i >= 10:
                        break
                    logger.info(f"Dynamic only: {thm[1]} in {thm[0]}")
                    logger.info(f"  Start: {thm[2]}, End: {thm[3]}")
            return False
        return True

    def _theorem_to_hashable(self, theorem):
        return (
            theorem["file_path"],
            theorem["full_name"],
            tuple(theorem["start"]),
            tuple(theorem["end"]),
        )

    def _tactic_to_hashable(self, tactic):
        return (
            tactic["tactic"],
            tactic["annotated_tactic"][0],
            tuple(
                (
                    a["full_name"],
                    a["def_path"],
                    tuple(a["def_pos"]),
                    tuple(a["def_end_pos"]),
                )
                for a in tactic["annotated_tactic"][1]
            ),
            tactic["state_before"],
            tactic["state_after"],
        )

    def test_unicode_handling_in_dataset(self):
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = (
            Path(RAID_DIR)
            / DATA_DIR
            / f"{dir_name_PFR}_{dir_name_new_version}_generated"
        )
        self.db.generate_merged_dataset(dst_dir)

        with open(dst_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
            self.assertIn("repositories", metadata, "No 'repositories' key in metadata")
            self.assertGreater(
                len(metadata["repositories"]), 0, "No repositories in metadata"
            )
            repo_PFR = metadata["repositories"][0]
            self.assertIn("metadata", repo_PFR, "No 'metadata' key in repository")
            repo_PFR_metadata = repo_PFR["metadata"]
            self.assertIn(
                "unicode", repo_PFR_metadata, "No 'unicode' key in repository metadata"
            )
            self.assertIn(
                "ユニコード",
                repo_PFR_metadata["unicode"],
                "Unicode string not found in metadata",
            )
            self.assertIn(
                "ユニコード", metadata["repositories"][0]["metadata"]["unicode"]
            )

            self.assertGreater(
                len(metadata["repositories"]), 1, "Only one repository in metadata"
            )
            repo_new_version = metadata["repositories"][1]
            self.assertIn(
                "metadata", repo_new_version, "No 'metadata' key in repository"
            )
            repo_new_version_metadata = repo_new_version["metadata"]
            self.assertIn(
                "unicode",
                repo_new_version_metadata,
                "No 'unicode' key in repository metadata",
            )
            self.assertIn(
                "ユニコード",
                repo_new_version_metadata["unicode"],
                "Unicode string not found in metadata",
            )
            self.assertIn(
                "ユニコード", metadata["repositories"][1]["metadata"]["unicode"]
            )

    def tearDown(self):
        # Clean up generated files after tests
        url_PFR = "https://github.com/teorth/pfr"
        commit_PFR = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
        dir_name_PFR = url_PFR.split("/")[-1] + "_" + commit_PFR
        url_new_version = "https://github.com/Adarsh321123/new-version-test"
        commit_new_version = "f465306be03ced999caa157a85558a6c41b3e3f5"
        dir_name_new_version = url_new_version.split("/")[-1] + "_" + commit_new_version
        dst_dir = (
            Path(RAID_DIR)
            / DATA_DIR
            / f"{dir_name_PFR}_{dir_name_new_version}_generated"
        )
        if dst_dir.exists():
            shutil.rmtree(dst_dir)

    def test_theorem_statement(self):
        theorem = next(
            t
            for t in self.sample_repo_PFR.proven_theorems
            if t.full_name == "ContinuousLinearMap.opNorm_lsmul"
        )
        self.assertIsNotNone(theorem.theorem_statement)
        self.assertIn("opNorm_lsmul", theorem.theorem_statement)

        theorem = next(
            t
            for t in self.sample_repo_new_version.proven_theorems
            if t.full_name == "Ordinal.le_mul_right"
        )
        self.assertIsNotNone(theorem.theorem_statement)
        self.assertIn("le_mul_right", theorem.theorem_statement)

    def test_unicode_handling(self):
        self.assertIsNotNone(self.sample_repo_PFR.metadata)
        self.assertIsNotNone(self.sample_repo_PFR.metadata["unicode"])
        self.assertIn("ユニコード", self.sample_repo_PFR.metadata["unicode"])

        self.assertIsNotNone(self.sample_repo_new_version.metadata)
        self.assertIsNotNone(self.sample_repo_new_version.metadata["unicode"])
        self.assertIn("ユニコード", self.sample_repo_new_version.metadata["unicode"])

    def test_file_tracing(self):
        self.assertGreater(len(self.sample_repo_PFR.files_traced), 0)
        self.assertIn(
            Path("PFR/Mathlib/GroupTheory/Torsion.lean"),
            self.sample_repo_PFR.files_traced,
        )
        self.assertIn(
            Path(".lake/packages/batteries/Batteries/Data/List/Lemmas.lean"),
            self.sample_repo_PFR.files_traced,
        )

        self.assertGreater(len(self.sample_repo_new_version.files_traced), 0)
        self.assertIn(
            Path("NewVersionTest/ExercisesOne.lean"),
            self.sample_repo_new_version.files_traced,
        )
        self.assertIn(
            Path(".lake/packages/batteries/Batteries/Data/List/Lemmas.lean"),
            self.sample_repo_new_version.files_traced,
        )


class TestDynamicDatabaseProver(unittest.TestCase):
    """
    Unit tests for DynamicDatabase class focusing on proof persistence.
    This test suite validates the DynamicDatabase class's capabilities to:
    1. Create and manage repositories of Lean theorems
    2. Track the state of theorems (unproved/proved)
    3. Update theorems with proofs
    4. Serialize to and deserialize from JSON
    5. Handle theorem proving workflows
    The tests verify:
    - Basic creation and manipulation of database objects
    - Annotated tactic creation and storage
    - Repository updates with proved theorems
    - JSON serialization and deserialization
    - Handling of duplicate theorems across repositories
    - Persistence during incremental proving
    Each test method uses a standard setup with a test repository containing
    a single unproved theorem, which can then be manipulated to test different
    aspects of the system's functionality.
    """

    def setUp(self):
        self.db = DynamicDatabase()
        self.repo = Repository(
            url="https://github.com/test/repo",
            name="test_repo",
            commit="abcdef1234567890",
            lean_version="4.0.0",
            lean_dojo_version="1.0.0",
            metadata={"date_processed": datetime.datetime.now()},
            sorry_theorems_unproved=[
                Theorem(
                    full_name="test_theorem",
                    file_path=Path("src/test.lean"),
                    start=Pos(1, 1),
                    end=Pos(10, 1),
                    url="https://github.com/test/repo",
                    commit="abcdef1234567890",
                    theorem_statement="theorem test_theorem : 2 + 2 = 4 := sorry",
                )
            ],
        )
        self.db.add_repository(self.repo)

    def test_create_annotated_tactic(self):
        tactic = "rw [add_comm]"
        annotated_tactic = AnnotatedTactic(
            tactic=tactic,
            annotated_tactic=(tactic, []),
            state_before="",
            state_after="",
        )
        self.assertEqual(annotated_tactic.tactic, tactic)
        self.assertEqual(annotated_tactic.annotated_tactic, (tactic, []))
        self.assertEqual(annotated_tactic.state_before, "")
        self.assertEqual(annotated_tactic.state_after, "")

    def test_update_theorem_with_proof(self):
        theorem = self.repo.sorry_theorems_unproved[0]
        traced_tactics = [
            AnnotatedTactic(
                tactic="rw [add_comm]",
                annotated_tactic=("rw [add_comm]", []),
                state_before="⊢ 2 + 2 = 4",
                state_after="⊢ 2 + 2 = 4",
            ),
            AnnotatedTactic(
                tactic="refl",
                annotated_tactic=("refl", []),
                state_before="⊢ 2 + 2 = 4",
                state_after="no goals",
            ),
        ]
        theorem.traced_tactics = traced_tactics
        self.repo.change_sorry_to_proven(theorem, PROOF_LOG_FILE_NAME)
        self.db.update_repository(self.repo)

        updated_repo = self.db.get_repository(self.repo.url, self.repo.commit)
        self.assertEqual(len(updated_repo.sorry_theorems_proved), 1)
        self.assertEqual(len(updated_repo.sorry_theorems_unproved), 0)
        self.assertEqual(
            updated_repo.sorry_theorems_proved[0].traced_tactics, traced_tactics
        )

    def test_json_serialization_with_proved_theorems(self):
        results = [
            SearchResult(
                theorem=self.repo.sorry_theorems_unproved[0],
                status=Status.PROVED,
                proof=["rw [add_comm]", "refl"],
                actor_time=1.0,
                environment_time=2.0,
                total_time=3.0,
                num_total_nodes=10,
                num_searched_nodes=5,
            )
        ]
        result = results[0]

        traced_tactics = [
            AnnotatedTactic(
                tactic=tactic,
                annotated_tactic=(tactic, []),
                state_before="",
                state_after="",
            )
            for tactic in result.proof
        ]
        self.repo.sorry_theorems_unproved[0].traced_tactics = traced_tactics
        self.repo.change_sorry_to_proven(
            self.repo.sorry_theorems_unproved[0], PROOF_LOG_FILE_NAME
        )

        # Serialize to JSON
        json_file = "proved_theorems_test.json"
        self.db.to_json(json_file)

        # Deserialize from JSON
        loaded_db = DynamicDatabase.from_json(json_file)

        # Verify the loaded data
        loaded_repo = loaded_db.get_repository(self.repo.url, self.repo.commit)
        self.assertIsNotNone(loaded_repo)

        self.assertEqual(len(loaded_repo.sorry_theorems_unproved), 0)
        self.assertEqual(len(loaded_repo.sorry_theorems_proved), 1)

        proved_theorem = loaded_repo.sorry_theorems_proved[0]
        self.assertEqual(proved_theorem.full_name, "test_theorem")
        self.assertEqual(len(proved_theorem.traced_tactics), 2)
        self.assertEqual(proved_theorem.traced_tactics[0].tactic, "rw [add_comm]")
        self.assertEqual(proved_theorem.traced_tactics[1].tactic, "refl")

        # Test updating the loaded database
        new_theorem = Theorem(
            full_name="new_theorem",
            file_path=Path("src/new_test.lean"),
            start=Pos(1, 1),
            end=Pos(5, 1),
            url="https://github.com/test/repo",
            commit="abcdef1234567890",
            theorem_statement="theorem new_theorem : 3 + 3 = 6 := sorry",
        )
        loaded_repo.sorry_theorems_unproved.append(new_theorem)

        # Serialize the updated database
        updated_json_file = "updated_proved_theorems_test.json"
        loaded_db.to_json(updated_json_file)

        # Deserialize and verify the update
        final_db = DynamicDatabase.from_json(updated_json_file)
        final_repo = final_db.get_repository(self.repo.url, self.repo.commit)

        self.assertEqual(len(final_repo.sorry_theorems_proved), 1)
        self.assertEqual(len(final_repo.sorry_theorems_unproved), 1)
        self.assertEqual(final_repo.sorry_theorems_unproved[0].full_name, "new_theorem")

    def test_update_theorem_with_proof_and_json(self):
        json_file = "temp_file.json"
        theorem = self.repo.sorry_theorems_unproved[0]

        traced_tactics = [
            AnnotatedTactic(
                tactic="rw [add_comm]",
                annotated_tactic=("rw [add_comm]", []),
                state_before="",
                state_after="",
            )
        ]

        theorem.traced_tactics = traced_tactics
        self.repo.change_sorry_to_proven(theorem, PROOF_LOG_FILE_NAME)
        self.db.update_repository(self.repo)
        self.db.to_json(json_file)

        loaded_db = DynamicDatabase.from_json(json_file)
        loaded_repo = loaded_db.get_repository(self.repo.url, self.repo.commit)

        self.assertEqual(len(loaded_repo.sorry_theorems_proved), 1)
        self.assertEqual(len(loaded_repo.sorry_theorems_unproved), 0)
        proved_theorem = loaded_repo.sorry_theorems_proved[0]
        self.assertEqual(proved_theorem.full_name, theorem.full_name)
        self.assertEqual(proved_theorem.file_path, theorem.file_path)
        self.assertEqual(proved_theorem.start, theorem.start)
        self.assertEqual(proved_theorem.end, theorem.end)

        self.assertEqual(len(proved_theorem.traced_tactics), 1)
        loaded_tactic = proved_theorem.traced_tactics[0]
        self.assertEqual(loaded_tactic.tactic, "rw [add_comm]")
        self.assertEqual(loaded_tactic.annotated_tactic, ("rw [add_comm]", []))
        self.assertEqual(loaded_tactic.state_before, "")
        self.assertEqual(loaded_tactic.state_after, "")

    def test_prove_sorry_theorems(self):
        results = [
            SearchResult(
                theorem=self.repo.sorry_theorems_unproved[0],
                status=Status.PROVED,
                proof=["rw [add_comm]", "refl"],
                actor_time=1.0,
                environment_time=2.0,
                total_time=3.0,
                num_total_nodes=10,
                num_searched_nodes=5,
            )
        ]
        result = results[0] if results else None

        if isinstance(result, SearchResult) and result.status == Status.PROVED:
            logger.info("Proving theorem")
            traced_tactics = []
            for tactic in result.proof:
                traced_tactics.append(
                    AnnotatedTactic(
                        tactic=tactic,
                        annotated_tactic=(tactic, []),
                        state_before="",
                        state_after="",
                    )
                )
            self.repo.sorry_theorems_unproved[0].traced_tactics = traced_tactics
            self.repo.change_sorry_to_proven(
                self.repo.sorry_theorems_unproved[0], PROOF_LOG_FILE_NAME
            )

        self.assertEqual(len(self.repo.sorry_theorems_unproved), 0)
        self.assertEqual(len(self.repo.sorry_theorems_proved), 1)
        proved_theorem = self.repo.sorry_theorems_proved[0]
        self.assertEqual(len(proved_theorem.traced_tactics), 2)
        self.assertEqual(proved_theorem.traced_tactics[0].tactic, "rw [add_comm]")
        self.assertEqual(proved_theorem.traced_tactics[1].tactic, "refl")

    def test_save_load_dynamic_database(self):
        json_file = "temp_file.json"

        self.db.to_json(json_file)
        loaded_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(self.db.repositories), len(loaded_db.repositories))
        self.assertEqual(self.db.repositories[0].url, loaded_db.repositories[0].url)
        self.assertEqual(
            self.db.repositories[0].commit, loaded_db.repositories[0].commit
        )
        self.assertEqual(
            len(self.db.repositories[0].sorry_theorems_unproved),
            len(loaded_db.repositories[0].sorry_theorems_unproved),
        )

    def test_add_repository_and_save(self):
        json_file = "temp_file.json"

        self.db.to_json(json_file)

        new_repo_data = {
            "url": "https://github.com/test/new-repo",
            "name": "new_test_repo",
            "commit": "1234567890abcdef",
            "lean_version": "4.0.0",
            "lean_dojo_version": "1.0.0",
            "metadata": {"date_processed": datetime.datetime.now().isoformat()},
            "sorry_theorems_unproved": [
                {
                    "full_name": "new_test_theorem",
                    "file_path": "src/new_test.lean",
                    "start": [1, 1],
                    "end": [10, 1],
                    "url": "https://github.com/test/new-repo",
                    "commit": "1234567890abcdef",
                    "theorem_statement": "theorem new_test_theorem : 3 + 3 = 6 := sorry",
                }
            ],
        }

        new_repo = Repository.from_dict(new_repo_data)
        self.db.add_repository(new_repo)
        self.db.to_json(json_file)
        loaded_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(loaded_db.repositories), 2)
        self.assertEqual(
            loaded_db.repositories[1].url, "https://github.com/test/new-repo"
        )
        self.assertEqual(len(loaded_db.repositories[1].sorry_theorems_unproved), 1)
        self.assertEqual(
            loaded_db.repositories[1].sorry_theorems_unproved[0].full_name,
            "new_test_theorem",
        )

    def test_prove_sorry_theorems_and_save(self):
        json_file = "temp_file.json"

        self.db.to_json(json_file)

        results = [
            SearchResult(
                theorem=self.repo.sorry_theorems_unproved[0],
                status=Status.PROVED,
                proof=["rw [add_comm]", "refl"],
                actor_time=1.0,
                environment_time=2.0,
                total_time=3.0,
                num_total_nodes=10,
                num_searched_nodes=5,
            )
        ]
        result = results[0] if results else None

        if isinstance(result, SearchResult) and result.status == Status.PROVED:
            logger.info("Proving theorem")
            traced_tactics = []
            for tactic in result.proof:
                traced_tactics.append(
                    AnnotatedTactic(
                        tactic=tactic,
                        annotated_tactic=(tactic, []),
                        state_before="",
                        state_after="",
                    )
                )
            self.repo.sorry_theorems_unproved[0].traced_tactics = traced_tactics
            self.repo.change_sorry_to_proven(
                self.repo.sorry_theorems_unproved[0], PROOF_LOG_FILE_NAME
            )

        self.db.to_json(json_file)
        loaded_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(loaded_db.repositories[0].sorry_theorems_unproved), 0)
        self.assertEqual(len(loaded_db.repositories[0].sorry_theorems_proved), 1)
        proved_theorem = loaded_db.repositories[0].sorry_theorems_proved[0]
        self.assertEqual(proved_theorem.full_name, "test_theorem")
        self.assertEqual(len(proved_theorem.traced_tactics), 2)
        self.assertEqual(proved_theorem.traced_tactics[0].tactic, "rw [add_comm]")
        self.assertEqual(proved_theorem.traced_tactics[1].tactic, "refl")

    def _theorem_identifier(
        self, theorem: Theorem
    ) -> Tuple[str, str, Tuple[int, int], Tuple[int, int]]:
        return (
            theorem.full_name,
            str(theorem.file_path),
            tuple(theorem.start),
            tuple(theorem.end),
        )

    def test_prove_sorry_theorems_with_duplicates(self):
        # Create two repositories with the same theorem but different commits
        repo1 = Repository(
            url="https://github.com/test/repo",
            name="test_repo_1",
            commit="commit1",
            lean_version="4.0.0",
            lean_dojo_version="1.0.0",
            metadata={"date_processed": datetime.datetime(2023, 1, 1)},
            sorry_theorems_unproved=[
                Theorem(
                    full_name="duplicate_theorem",
                    file_path=Path("src/test.lean"),
                    start=Pos(1, 1),
                    end=Pos(10, 1),
                    url="https://github.com/test/repo",
                    commit="commit1",
                    theorem_statement="theorem duplicate_theorem : 2 + 2 = 4 := sorry",
                )
            ],
        )

        repo2 = Repository(
            url="https://github.com/test/repo",
            name="test_repo_2",
            commit="commit2",
            lean_version="4.0.0",
            lean_dojo_version="1.0.0",
            metadata={"date_processed": datetime.datetime(2023, 1, 2)},
            sorry_theorems_unproved=[
                Theorem(
                    full_name="duplicate_theorem",
                    file_path=Path("src/test.lean"),
                    start=Pos(1, 1),
                    end=Pos(10, 1),
                    url="https://github.com/test/repo",
                    commit="commit2",
                    theorem_statement="theorem duplicate_theorem : 2 + 2 = 4 := sorry",
                )
            ],
        )

        # Create a test database with both repositories
        test_db = DynamicDatabase()
        test_db.add_repository(repo1)
        test_db.add_repository(repo2)

        json_file = "duplicate_theorems_test.json"
        test_db.to_json(json_file)

        loaded_db = DynamicDatabase.from_json(json_file)

        self.assertEqual(len(loaded_db.repositories), 2)
        loaded_repo1 = loaded_db.get_repository(repo1.url, repo1.commit)
        loaded_repo2 = loaded_db.get_repository(repo2.url, repo2.commit)
        self.assertIsNotNone(loaded_repo1)
        self.assertIsNotNone(loaded_repo2)

        results = [
            SearchResult(
                theorem=loaded_repo2.sorry_theorems_unproved[0],
                status=Status.PROVED,
                proof=["rw [add_comm]", "refl"],
                actor_time=1.0,
                environment_time=2.0,
                total_time=3.0,
                num_total_nodes=10,
                num_searched_nodes=5,
            )
        ]

        # Simulate the prove_sorry_theorems function
        processed_theorems = set()
        for repo in sorted(
            loaded_db.repositories,
            key=lambda r: r.metadata["date_processed"],
            reverse=True,
        ):
            for theorem in repo.sorry_theorems_unproved:
                theorem_id = self._theorem_identifier(theorem)
                if theorem_id in processed_theorems:
                    continue

                processed_theorems.add(theorem_id)

                # Apply the proof to the theorem
                result = results[
                    0
                ]  # In a real scenario, this would be the result of calling the prover
                traced_tactics = [
                    AnnotatedTactic(
                        tactic=tactic,
                        annotated_tactic=(tactic, []),
                        state_before="",
                        state_after="",
                    )
                    for tactic in result.proof
                ]
                theorem.traced_tactics = traced_tactics
                repo.change_sorry_to_proven(theorem, PROOF_LOG_FILE_NAME)
                loaded_db.update_repository(repo)
                break  # Simulate proving only one theorem

        updated_json_file = "updated_duplicate_theorems_test.json"
        loaded_db.to_json(updated_json_file)

        final_db = DynamicDatabase.from_json(updated_json_file)
        final_repo1 = final_db.get_repository(repo1.url, repo1.commit)
        final_repo2 = final_db.get_repository(repo2.url, repo2.commit)

        # Verify that only the more recent theorem was proved
        self.assertEqual(len(final_repo1.sorry_theorems_unproved), 1)
        self.assertEqual(len(final_repo1.sorry_theorems_proved), 0)
        self.assertEqual(len(final_repo2.sorry_theorems_unproved), 0)
        self.assertEqual(len(final_repo2.sorry_theorems_proved), 1)

        # Verify the proof of the more recent theorem
        proved_theorem = final_repo2.sorry_theorems_proved[0]
        self.assertEqual(proved_theorem.full_name, "duplicate_theorem")
        self.assertEqual(len(proved_theorem.traced_tactics), 2)
        self.assertEqual(proved_theorem.traced_tactics[0].tactic, "rw [add_comm]")
        self.assertEqual(proved_theorem.traced_tactics[1].tactic, "refl")

    def test_repeated_to_json_during_proving(self):
        json_file = "repeated_save_test.json"

        # Initial save
        self.db.to_json(json_file)

        results = [
            SearchResult(
                theorem=self.repo.sorry_theorems_unproved[0],
                status=Status.PROVED,
                proof=["rw [add_comm]", "refl"],
                actor_time=1.0,
                environment_time=2.0,
                total_time=3.0,
                num_total_nodes=10,
                num_searched_nodes=5,
            )
        ]

        # Simulate proving and saving after each theorem
        for result in results:
            if isinstance(result, SearchResult) and result.status == Status.PROVED:
                traced_tactics = [
                    AnnotatedTactic(
                        tactic=tactic,
                        annotated_tactic=(tactic, []),
                        state_before="",
                        state_after="",
                    )
                    for tactic in result.proof
                ]
                self.repo.sorry_theorems_unproved[0].traced_tactics = traced_tactics
                self.repo.change_sorry_to_proven(
                    self.repo.sorry_theorems_unproved[0], PROOF_LOG_FILE_NAME
                )
                self.db.update_repository(self.repo)
                self.db.to_json(json_file)  # Save after each theorem is proved

        # Final save
        self.db.to_json(json_file)

        # Load and verify
        loaded_db = DynamicDatabase.from_json(json_file)
        self.assertEqual(len(loaded_db.repositories[0].sorry_theorems_proved), 1)
        self.assertEqual(len(loaded_db.repositories[0].sorry_theorems_unproved), 0)


class TestDynamicDatabaseEmpty(unittest.TestCase):
    def setUp(self):
        self.empty_json_path = "empty_database.json"
        self.db = DynamicDatabase()

    def tearDown(self):
        if os.path.exists(self.empty_json_path):
            os.remove(self.empty_json_path)

    def test_from_json_empty_file(self):
        with open(self.empty_json_path, "w") as f:
            json.dump({"repositories": []}, f)

        db = DynamicDatabase.from_json(self.empty_json_path)
        self.assertEqual(len(db.repositories), 0)

    def test_to_json_empty_database(self):
        self.db.to_json(self.empty_json_path)

        self.assertTrue(os.path.exists(self.empty_json_path))
        with open(self.empty_json_path, "r") as f:
            content = json.load(f)
            self.assertEqual(content, {"repositories": []})

    def test_from_json_invalid_empty_object(self):
        with open(self.empty_json_path, "w") as f:
            json.dump({}, f)

        with self.assertRaises(ValueError):
            DynamicDatabase.from_json(self.empty_json_path)

    def test_from_json_nonexistent_file(self):
        if os.path.exists(self.empty_json_path):
            os.remove(self.empty_json_path)

        with self.assertRaises(FileNotFoundError):
            DynamicDatabase.from_json(self.empty_json_path)

    def test_add_repository_to_empty_database(self):
        with open(self.empty_json_path, "w") as f:
            json.dump({"repositories": []}, f)

        db = DynamicDatabase.from_json(self.empty_json_path)

        repo_data = {
            "url": "https://github.com/test/repo",
            "name": "Test Repo",
            "commit": "abc123",
            "lean_version": "3.50.3",
            "lean_dojo_version": "1.8.4",
            "metadata": {"date_processed": datetime.datetime.now().isoformat()},
        }
        repo = Repository.from_dict(repo_data)
        db.add_repository(repo)

        db.to_json(self.empty_json_path)

        loaded_db = DynamicDatabase.from_json(self.empty_json_path)
        self.assertEqual(len(loaded_db.repositories), 1)
        self.assertEqual(loaded_db.repositories[0].url, "https://github.com/test/repo")


def main():
    unittest.main()


if __name__ == "__main__":
    main()
