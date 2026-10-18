import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / 'src/pyinfinitensor/batching.py'
spec = importlib.util.spec_from_file_location('batch_planner', path)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


class TestBatchPlanner(unittest.TestCase):
    def test_length_order_is_stable_and_complete(self):
        lengths = [512, 5, 128, 5, 500, 30, 129]
        batches = planner.plan_batches(lengths, 3, 600)
        indices = [index for batch in batches for index in batch]
        self.assertEqual(indices, [1, 3, 5, 2, 6, 4, 0])
        self.assertEqual(sorted(indices), list(range(len(lengths))))
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            self.assertLessEqual(len(batch)*max(lengths[i] for i in batch), 600)

    def test_mixed_lengths_reduce_actual_padding(self):
        lengths = [8, 512]*8
        cost = lambda batches: sum(len(b)*max(lengths[i] for i in b) for b in batches)
        baseline = planner.plan_batches(lengths, 4, 2048, 'input')
        optimized = planner.plan_batches(lengths, 4, 2048, 'length')
        self.assertEqual(cost(baseline), 8192)
        self.assertEqual(cost(optimized), 4160)

    def test_invalid_budget_and_integer_inputs(self):
        for args in [([0], 4, 100), ([1.5], 4, 100), ([1], True, 100),
                     ([101], 4, 100), ([1], 4, 0), ([True], 4, 100)]:
            with self.assertRaises(ValueError):
                planner.plan_batches(*args)
        with self.assertRaises(ValueError):
            planner.plan_batches([4], 2, 20, 'buckets')

    def test_empty_and_input_order(self):
        self.assertEqual(planner.plan_batches([], 4, 512), [])
        self.assertEqual(planner.plan_batches([5, 4, 3], 2, 512, 'input'), [[0, 1], [2]])


if __name__ == '__main__':
    unittest.main()
