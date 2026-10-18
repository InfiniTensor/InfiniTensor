import tempfile
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from pyinfinitensor.retrieval import KnowledgeIndex
from pyinfinitensor.reranking import position_ids
from pyinfinitensor.model_package import package_identity


class TestRetrievalIndex(unittest.TestCase):
    def test_self_contained_index_roundtrip(self):
        vectors = np.zeros((2, 512), np.float32)
        vectors[0, 0] = vectors[1, 1] = 1
        index = KnowledgeIndex(['甲', '乙'], ['中文文档', 'English document'], vectors,
                               {'model_onnx_sha256': 'example', 'rows': 2})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'index.npz'
            index.save(path)
            restored = KnowledgeIndex.load(path)
            self.assertEqual(restored.ids, index.ids)
            self.assertEqual(restored.texts, index.texts)
            np.testing.assert_array_equal(restored.vectors, index.vectors)
            self.assertFalse(path.with_name('index.npz.tmp').exists())

    def test_invalid_index_is_rejected(self):
        vectors = np.zeros((2, 512), np.float32)
        vectors[:, 0] = 1
        with self.assertRaises(ValueError):
            KnowledgeIndex(['x', 'x'], ['a', 'b'], vectors, {})
        with self.assertRaises(ValueError):
            KnowledgeIndex(['x', 'y'], ['a'], vectors, {})
        with self.assertRaises(ValueError):
            KnowledgeIndex(['x', 'y'], ['a', 'b'], vectors*2, {})

    def test_xlm_position_ids_keep_padding_positions(self):
        values = np.asarray([[0, 100, 2, 1, 1], [0, 5, 6, 7, 2]], np.int64)
        np.testing.assert_array_equal(position_ids(values), [[2, 3, 4, 1, 1], [2, 3, 4, 5, 6]])

    def test_changed_tokenizer_is_not_a_frozen_package(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'model.onnx').write_bytes(b'example model')
            (path/'tokenizer.json').write_text('{}')
            hashes = {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
                      for name in ('model.onnx','tokenizer.json')}
            (path/'manifest.json').write_text(json.dumps({'sha256':hashes}))
            identity = package_identity(path,path/'model.onnx')
            self.assertIn('fingerprint',identity)
            (path/'tokenizer.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'tokenizer.json'):
                package_identity(path,path/'model.onnx')


if __name__ == '__main__':
    unittest.main()
