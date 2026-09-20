import sys
import unittest
from pathlib import Path

# 定位本仓库的 Demo，不依赖启动命令所在目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from onnx_dynamic_shape_demo import validate_model


class TestDynamicShape(unittest.TestCase):
    def test_same_instance_matches_ort(self):
        for naive in (False, True):
            with self.subTest(naive=naive):
                records = validate_model(naive=naive, verbose=False)
                self.assertEqual([record[0] for record in records], [1, 2, 8, 3, 1])


if __name__ == "__main__":
    unittest.main()