"""Frozen test-only probes; evaluation preserves model modes and training RNG."""
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset


@contextmanager
def preserve_rng():
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


class ProbeManager:
    def __init__(self, per_class=32, seed=0):
        if per_class < 1:
            raise ValueError("probe size must be positive")
        self.per_class, self.seed = int(per_class), int(seed)
        self.indices = {}
        self.datasets = {}

    def add_classes(self, data_manager, classes):
        for c in classes:
            c = int(c)
            if c in self.indices:
                continue
            dataset = data_manager.get_dataset(np.array([c]), source="test", mode="test")
            if not len(dataset):
                raise ValueError("no test probes for class {}".format(c))
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, c]))
            self.indices[c] = rng.choice(len(dataset), min(self.per_class, len(dataset)), replace=False).tolist()
            self.datasets[c] = Subset(dataset, self.indices[c])

    def loader(self):
        return DataLoader(ConcatDataset(list(self.datasets.values())), batch_size=256,
                          shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(self.seed))

    def state_dict(self):
        return {"seed": self.seed, "per_class": self.per_class, "indices": self.indices,
                "index_space": "within single-class test dataset in remapped label order"}


def extract_probe_outputs(model, loader, feature_extractor=None):
    modes = [(module, module.training) for module in model.modules()]
    logits, labels, features = [], [], []
    device = next(model.parameters()).device
    try:
        with preserve_rng(), torch.inference_mode():
            model.eval()
            for _, images, target in loader:
                output = model(images.to(device))
                logits.append(output["logits"].detach().cpu())
                labels.append(target.cpu())
                if feature_extractor is not None:
                    features.append(feature_extractor(output).detach().cpu())
    finally:
        for module, training in modes:
            module.training = training
    return torch.cat(logits), torch.cat(labels), torch.cat(features) if features else None
