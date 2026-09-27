import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from utils.task_registry import TaskRegistry

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from utils.metrics_logger import MetricsLogger
    from utils.probe_manager import extract_probe_outputs


class RegistryTests(unittest.TestCase):
    def test_round_trip_and_half_open_ranges(self):
        registry = TaskRegistry([(0, 0, 4), (1, 4, 7), (2, 7, 10)])
        restored = TaskRegistry(json.loads(json.dumps(registry.ranges)))
        self.assertEqual([restored.task_for(c) for c in (0, 3, 4, 6, 7, 9)], [0, 0, 1, 1, 2, 2])
        with self.assertRaises(KeyError):
            restored.task_for(10)

    def test_reject_overlap_or_gap(self):
        for row in ((1, 1, 4), (1, 3, 4), (2, 2, 4), (1, 2, 2)):
            registry = TaskRegistry([(0, 0, 2)])
            with self.assertRaises(ValueError):
                registry.append(*row)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for numerical monitoring tests")
class MonitoringTests(unittest.TestCase):
    def test_probe_preserves_rng_modes_and_batchnorm(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(2)
            def forward(self, x):
                return {"logits": self.bn(x)}
        model = Model().train()
        loader = DataLoader(TensorDataset(torch.arange(4), torch.ones(4, 2), torch.zeros(4, dtype=torch.long)), batch_size=2)
        rng = torch.random.get_rng_state().clone()
        mean = model.bn.running_mean.clone()
        extract_probe_outputs(model, loader)
        self.assertTrue(model.training and model.bn.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(mean, model.bn.running_mean))

    def test_confusion_and_boundary_distance(self):
        registry = TaskRegistry([(0, 0, 2), (1, 2, 3), (2, 3, 4)])
        labels = torch.tensor([0, 0, 0, 1, 2, 3])
        pred = torch.tensor([1, 2, 3, 1, 2, 0])
        logits = torch.zeros(6, 4).scatter_(1, pred[:, None], 5)
        features = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        with tempfile.TemporaryDirectory() as temp:
            logger = MetricsLogger(Path(temp) / "run", 1)
            logger.probe(2, 0, logits, labels, features, registry)
            logger.set_boundary()
            logger.probe(2, 1, logits, labels, features, registry)
            rows = [json.loads(line) for line in (logger.directory / "metrics.jsonl").read_text().splitlines()]
            first = next(row for row in rows if row["event"] == "class_probe" and row["class_id"] == 0)
            self.assertEqual(first["confusion_counts"]["same_old_task"], 1)
            self.assertEqual(first["confusion_counts"]["different_old_task"], 1)
            self.assertEqual(first["confusion_counts"]["old_to_current"], 1)
            pairs = [row for row in rows if row["event"] == "pair_probe" and row["round"] == 1]
            self.assertTrue(all(row["boundary_delta"] == 0 for row in pairs))


if __name__ == "__main__":
    unittest.main()
