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
    from utils.fedcbdr_interactions import ReplayExposureTracker, replacement_slots
    import numpy as np


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

    def test_directed_pair_confusion_and_boundary_margin(self):
        registry = TaskRegistry([(0, 0, 2), (1, 2, 3), (2, 3, 4)])
        labels = torch.tensor([0, 0, 0, 1, 2, 3])
        pred = torch.tensor([1, 2, 3, 1, 2, 0])
        logits = torch.zeros(6, 4).scatter_(1, pred[:, None], 5)
        features = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        with tempfile.TemporaryDirectory() as temp:
            logger = MetricsLogger(Path(temp) / "run", 1)
            logger.probe(2, 0, logits, labels, features, registry)
            logger.set_boundary(logits, labels)
            logger.probe(2, 1, logits, labels, features + 1, registry)
            rows = [json.loads(line) for line in (logger.directory / "metrics.jsonl").read_text().splitlines()]
            first = next(row for row in rows if row["event"] == "class_probe" and row["class_id"] == 0)
            self.assertEqual(first["confusion_counts"]["same_old_task"], 1)
            self.assertEqual(first["confusion_counts"]["different_old_task"], 1)
            self.assertEqual(first["confusion_counts"]["old_to_current"], 1)
            pair = torch.load(logger.directory / "pair_metrics_T2_R001.pt",
                              weights_only=False)
            self.assertEqual(pair["class_ids"], [0, 1, 2])
            self.assertAlmostEqual(float(pair["pair_confusion_rate"][0, 1]), 1/3)
            self.assertAlmostEqual(float(pair["pair_confusion_rate"][1, 0]), 0)
            self.assertAlmostEqual(float(pair["delta_pair_margin"][0, 1]), 0)
            self.assertAlmostEqual(float(pair["relative_feature_direction"][0, 1]), 0)
            self.assertGreater(abs(float(pair["feature_direction"][0, 1])), 0)
            self.assertEqual(int(pair["pair_type"][0, 1]), 0)
            self.assertEqual(int(pair["pair_type"][0, 2]), 1)

    def test_exposure_resets_history_by_epoch(self):
        tracker = ReplayExposureTracker(3)
        tracker.record(torch.tensor([0, 1]), torch.tensor([True, True]),
                       torch.tensor([2.0, 3.0]))
        tracker.record(torch.tensor([2]), torch.tensor([True]), torch.tensor([4.0]))
        tracker.begin_epoch()
        tracker.record(torch.tensor([1]), torch.tensor([True]), torch.tensor([5.0]))
        state = tracker.state_dict()
        self.assertEqual(int(state["co_batch_count"][0, 1]), 1)
        self.assertEqual(int(state["sequential_exposure"][0, 2]), 1)
        self.assertEqual(int(state["sequential_exposure"][0, 1]), 0)
        self.assertEqual(int(state["min_lag"][0, 2]), 1)
        self.assertEqual(int(state["lag_observations"][0, 2]), 1)
        self.assertEqual(int(state["draws"].sum()), 4)
        self.assertEqual(float(state["ce_mass"].sum()), 14.0)

    def test_replacement_preserves_slots_and_weights(self):
        def replay(labels):
            return dict(task_id=0, images=np.array(labels), labels=np.array(labels),
                        sampling_weights=np.arange(len(labels), dtype=float)+1)
        original = [[replay([0, 0, 2, 2])], [replay([0, 2])]]
        replaced, manifest = replacement_slots(original, 0, 2,
                                               [(0, 0, 3)], max_multiplicity=2)
        self.assertEqual(replaced[0][0]["labels"].tolist(), [2, 2, 2, 2])
        self.assertEqual(replaced[0][0]["sampling_weights"].tolist(),
                         original[0][0]["sampling_weights"].tolist())
        self.assertEqual(original[0][0]["labels"].tolist(), [0, 0, 2, 2])
        self.assertEqual(manifest[0]["max_C_multiplicity"], 2)
        with self.assertRaises(ValueError):
            replacement_slots(original, 0, 2, [(0, 0, 1), (1, 1, 3)])
        duplicate_c = [[replay([0, 0, 2, 2])]]
        duplicate_c[0][0]["source_dataset_indices"] = np.array([0, 1, 2, 2])
        with self.assertRaises(ValueError):
            replacement_slots(duplicate_c, 0, 2, [(0, 0, 3)],
                              max_multiplicity=1)


if __name__ == "__main__":
    unittest.main()
