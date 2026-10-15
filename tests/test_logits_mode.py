"""CPU checks for the FedAvg_Test logit objectives inside LANDER."""
import contextlib
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from logits_lab.objectives import cosine_targets, margin_loss, relational_kl
from logits_lab.runtime import PrototypeExperiment


class ToyDataset(Dataset):
    def __init__(self, features, labels, trsf=None):
        self.features = features
        self.labels = np.asarray(labels, dtype=np.int64)
        self.trsf = trsf or (lambda x: x)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return idx, self.trsf(self.features[idx]), int(self.labels[idx])


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 8)
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        features = self.encoder(x)
        return {'features': features, 'logits': self.fc(features)}


class ToyOwner:
    def __init__(self, current, replay, args):
        self.args = args
        self._cur_task = 1
        self._known_classes = 4
        self._total_classes = 6
        self.feature_dim = 8
        self.seed = 5
        self._network = ToyNet()
        self.test_loader = DataLoader(current, batch_size=4)
        self.retained_ds_all = replay
        self.repo_dual = self.exp5_distill_loss = self.exp5_single_distill_loss = False

    def _prepare_images(self, images):
        return images

    def _autocast(self):
        return contextlib.nullcontext()


class LogitsModeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.current = ToyDataset(torch.randn(6, 4), [4, 4, 4, 5, 5, 5])
        self.replay = {
            0: [ToyDataset(torch.randn(4, 4), [0, 1, 2, 3])],
            1: [ToyDataset(torch.randn(4, 4), [0, 1, 2, 3])],
        }
        self.groups = {0: [0, 1, 2], 1: [3, 4, 5]}
        self.args = {'dataset': 'toy', 'class_order': [5, 2, 4, 0, 3, 1],
                     'logits_warmup': 0, 'logits_refresh': 1,
                     'logits_probe_rounds': ''}

    def owner(self, mode):
        return ToyOwner(self.current, self.replay,
                        {**self.args, 'logits_mode': mode})

    def test_lte_margin_uses_shuffled_seen_class_order_and_full_objective(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            table = torch.randn(6, 8)
            path = Path(temp) / 'toy_le.pickle'
            with path.open('wb') as stream:
                pickle.dump(table, stream)
            owner = self.owner('LTE_margin')
            owner.args['logits_lte_path'] = str(path)
            lab = PrototypeExperiment(owner, self.current, self.groups)
            lab.start_round(1, owner._network)
            expected = cosine_targets(table[owner.args['class_order']])[1]
            self.assertTrue(torch.allclose(lab.qneg, expected))
            images = self.current.features[:3]
            labels = torch.tensor([4, 4, 4])
            logits = owner._network(images)['logits']
            ce = F.cross_entropy(logits, labels)
            actual = lab.augment_loss(ce, logits, labels, images, owner._network)
            wanted = (ce + 0.1 * relational_kl(logits, expected[labels], labels, 2.0)
                      + 0.5 * margin_loss(logits, labels, 1.0))
            self.assertTrue(torch.allclose(actual, wanted))
            actual.backward()
            self.assertTrue(torch.isfinite(owner._network.fc.weight.grad).all())

    def test_proto_margin_uses_only_current_and_retained_replay(self):
        owner = self.owner('proto_margin')
        lab = PrototypeExperiment(owner, self.current, self.groups)
        rng = torch.random.get_rng_state().clone()
        lab.start_round(1, owner._network)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(lab.proto_stats['counts'], [2, 2, 2, 2, 3, 3])
        images = torch.cat([self.current.features,
                            self.replay[0][0].features, self.replay[1][0].features])
        labels = torch.tensor([*self.current.labels, *self.replay[0][0].labels,
                               *self.replay[1][0].labels])
        with torch.no_grad():
            features = F.normalize(owner._network(images)['features'], dim=1)
            sums = torch.zeros(6, 8).index_add_(0, labels, features)
            centers = F.normalize(sums / torch.bincount(labels).float()[:, None], dim=1)
        self.assertTrue(torch.allclose(lab.sim, centers @ centers.T, atol=1e-6))

    def test_global_margin_uses_frozen_pre_round_teacher_and_warmup(self):
        owner = self.owner('global_margin')
        owner.args['logits_warmup'] = 1
        lab = PrototypeExperiment(owner, self.current, self.groups)
        images = self.current.features[:3]
        labels = torch.tensor([4, 4, 4])
        student = ToyNet()
        logits = student(images)['logits']
        ce = F.cross_entropy(logits, labels)
        lab.start_round(1, owner._network)
        self.assertIs(lab.augment_loss(ce, logits, labels, images, owner._network), ce)
        lab.start_round(2, owner._network)
        owner._network.train()
        teacher_before = [p.detach().clone() for p in owner._network.parameters()]
        loss = lab.augment_loss(ce, logits, labels, images, owner._network)
        self.assertGreater(float(loss.detach()), float(ce.detach()))
        loss.backward()
        self.assertTrue(owner._network.training)
        self.assertTrue(all(p.grad is None for p in owner._network.parameters()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(teacher_before, owner._network.parameters())))
        self.assertTrue(torch.isfinite(student.fc.weight.grad).all())

    def test_missing_old_class_prototype_fails(self):
        self.replay[0][0].labels[2] = 0
        self.replay[1][0].labels[2] = 1
        with self.assertRaisesRegex(RuntimeError, 'no current training or retained replay images'):
            PrototypeExperiment(self.owner('proto_margin'), self.current, self.groups)
        PrototypeExperiment(self.owner('global_margin'), self.current, self.groups)


if __name__ == '__main__':
    unittest.main()
