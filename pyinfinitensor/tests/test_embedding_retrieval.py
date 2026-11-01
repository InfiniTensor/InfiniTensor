import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext

import numpy as np
from pyinfinitensor.retrieval import build_index, search_index


class Session:
    max_length = 512
    model_dir = Path("model")
    model_path = Path("model/model.onnx")
    model_manifest = {"revision": "a" * 40, "sha256": {"model.onnx": "b" * 64}}

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        result = np.zeros((len(texts), 512), np.float32)
        result[:, 0] = 1
        return result


class TestOfflineRetrieval(unittest.TestCase):
    def test_persisted_artifacts_can_be_queried_with_stable_id_ties(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "input.jsonl"
            documents.write_text(
                "\n".join(
                    json.dumps(d)
                    for d in [{"id": "z", "text": "甲"}, {"id": "a", "text": "乙"}]
                )
                + "\n",
                encoding="utf-8",
            )
            session = Session()
            index = root / "index"
            manifest = build_index(session, documents, index, batch_size=1)
            self.assertEqual(manifest["count"], 2)
            results = search_index(session, index, "查询", 10)
            self.assertEqual([item["id"] for item in results], ["a", "z"])
            self.assertEqual(session.calls[-1][1], {"kind": "query", "batch_size": 1})
            with self.assertRaises(FileExistsError):
                build_index(session, documents, index)
            (index / "documents.jsonl").write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                search_index(session, index, "查询", 2)

    def test_index_and_query_respect_small_embedding_session(self):
        from test_embedding import TestEmbeddingSession

        for limit in (1, 4):
            with (
                self.subTest(limit=limit),
                tempfile.TemporaryDirectory() as folder,
                patch(
                    "pyinfinitensor.embedding.fp32_inference_policy",
                    side_effect=nullcontext,
                ),
            ):
                session = TestEmbeddingSession().make_session()
                session.max_batch_size = limit
                session.model_dir, session.model_path = (
                    Session.model_dir,
                    Session.model_path,
                )
                session.model_manifest = Session.model_manifest
                root = Path(folder)
                documents = root / "input.jsonl"
                documents.write_text(
                    "{" + '"id":"a","text":"test"' + "}", encoding="utf-8"
                )
                build_index(session, documents, root / "index")
                self.assertEqual(
                    search_index(session, root / "index", "query")[0]["id"], "a"
                )

    def test_selection_generator_sorts_string_ids_and_rejects_duplicates(self):
        import importlib.util

        path = (
            Path(__file__).resolve().parents[2]
            / "scripts/prepare_retrieval_selection.py"
        )
        spec = importlib.util.spec_from_file_location("selection", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [{"_id": i, "text": str(i)} for i in (2, 10, 1)]
        self.assertEqual([row["id"] for row in module.select(rows, 2)], ["1", "10"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.select(rows + rows[:1], 2)

    def test_duplicate_ids_and_mismatched_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "input.jsonl"
            documents.write_text('{"id":1,"text":"x"}\n{"id":"1","text":"y"}\n')
            session = Session()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build_index(session, documents, root / "index")
            self.assertEqual(session.calls, [])
            documents.write_text('{"id":"1","text":"x"}\n')
            build_index(session, documents, root / "index")
            other = Session()
            other.model_manifest = {
                "revision": "a" * 40,
                "sha256": {"model.onnx": "c" * 64},
            }
            with self.assertRaisesRegex(ValueError, "model"):
                search_index(other, root / "index", "查询", 1)
            self.assertEqual(other.calls, [])

    def test_same_onnx_with_different_tokenizer_cannot_query_old_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "input.jsonl"
            documents.write_text('{"id":"1","text":"x"}' + chr(10))
            original = Session()
            original.model_manifest = {
                "revision": "a" * 40,
                "sha256": {"model.onnx": "b" * 64, "tokenizer.json": "1" * 64},
            }
            build_index(original, documents, root / "index")
            changed = Session()
            changed.model_manifest = {
                "revision": "a" * 40,
                "sha256": {"model.onnx": "b" * 64, "tokenizer.json": "2" * 64},
            }
            with self.assertRaisesRegex(ValueError, "model"):
                search_index(changed, root / "index", "query", 1)
            self.assertEqual(changed.calls, [])

    def test_selected_onnx_is_part_of_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "input.jsonl"
            documents.write_text('{"id":"1","text":"x"}' + chr(10))
            original = Session()
            original.model_manifest = {
                "revision": "a" * 40,
                "sha256": {"model.onnx": "b" * 64, "alt.onnx": "c" * 64},
            }
            build_index(original, documents, root / "index")
            changed = Session()
            changed.model_manifest = original.model_manifest
            changed.model_path = Path("model/alt.onnx")
            with self.assertRaisesRegex(ValueError, "model"):
                search_index(changed, root / "index", "query", 1)


if __name__ == "__main__":
    unittest.main()
