"""Client-local replay exposure and paired replacement helpers for FedCBDR."""

from collections import Counter

import numpy as np
import torch


class ReplayExposureTracker:
    """Rows are earlier/aggressor A and columns are later/victim B.

    Histories are reset at each local epoch and never shared across clients.
    Only replay examples contribute to the matrices.
    """

    def __init__(self, old_classes):
        self.old_classes = int(old_classes)
        shape = (self.old_classes, self.old_classes)
        self.co_batch_count = torch.zeros(shape, dtype=torch.int64)
        self.sequential_exposure = torch.zeros(shape, dtype=torch.int64)
        self.lag_sum = torch.zeros(shape, dtype=torch.float64)
        self.lag_observations = torch.zeros(shape, dtype=torch.int64)
        self.min_lag = torch.full(shape, -1, dtype=torch.int64)
        self.draws = torch.zeros(self.old_classes, dtype=torch.int64)
        self.ce_mass = torch.zeros(self.old_classes, dtype=torch.float64)
        self.begin_epoch()

    def begin_epoch(self):
        self._step = 0
        self._previous = torch.zeros(self.old_classes, dtype=torch.int64)
        self._last_seen = torch.full((self.old_classes,), -1, dtype=torch.int64)

    def record(self, labels, is_replay, sample_ce):
        labels = labels.detach().cpu().long()
        mask = is_replay.detach().cpu().bool() & (labels < self.old_classes)
        active_labels = labels[mask]
        if active_labels.numel():
            counts = torch.bincount(active_labels, minlength=self.old_classes)
            present = torch.nonzero(counts, as_tuple=False).flatten()
            self.draws += counts
            self.ce_mass += torch.bincount(
                active_labels, weights=sample_ce.detach().cpu().double()[mask],
                minlength=self.old_classes,
            )
            for b in present.tolist():
                for a in present.tolist():
                    if a != b:
                        self.co_batch_count[a, b] += 1
                earlier = torch.nonzero(self._previous, as_tuple=False).flatten()
                for a in earlier.tolist():
                    if a == b:
                        continue
                    self.sequential_exposure[a, b] += self._previous[a]
                    lag = self._step - int(self._last_seen[a])
                    self.lag_sum[a, b] += lag
                    self.lag_observations[a, b] += 1
                    prior_min = int(self.min_lag[a, b])
                    self.min_lag[a, b] = lag if prior_min < 0 else min(prior_min, lag)
            self._previous[present] += 1
            self._last_seen[present] = self._step
        self._step += 1

    def state_dict(self):
        mean = torch.full_like(self.lag_sum, float("nan"))
        valid = self.lag_observations > 0
        mean[valid] = self.lag_sum[valid] / self.lag_observations[valid]
        return dict(
            co_batch_count=self.co_batch_count.clone(),
            sequential_exposure=self.sequential_exposure.clone(),
            mean_lag=mean,
            min_lag=self.min_lag.clone(),
            lag_sum=self.lag_sum.clone(),
            lag_observations=self.lag_observations.clone(),
            draws=self.draws.clone(), ce_mass=self.ce_mass.clone(),
        )


def replacement_slots(replay_by_client, aggressor, replacement, task_ranges,
                      max_multiplicity=3, materialize=True):
    """Copy replay buffers and replace only A slots with client-local C donors.

    A-slot sampling weights and positions stay fixed. The donor assignment is
    deterministic so both runs consume the same sampler and augmentation RNG.
    """
    if len({aggressor, replacement}) != 2:
        raise ValueError("aggressor and replacement must differ")
    task_for = {}
    for task, start, end in task_ranges:
        task_for.update({c: task for c in range(start, end)})
    if task_for.get(aggressor) != task_for.get(replacement):
        raise ValueError("replacement must belong to aggressor's historical task")

    copied = ([[dict(row, images=np.array(row["images"], copy=True),
                     labels=np.array(row["labels"], copy=True),
                     sampling_weights=np.array(row["sampling_weights"], copy=True))
                for row in client] for client in replay_by_client]
              if materialize else replay_by_client)
    manifest = []
    for client_id, client in enumerate(copied):
        slots = [(ds, i) for ds in client for i, label in enumerate(ds["labels"])
                 if int(label) == aggressor]
        donor_slots = [(ds, i) for ds in client for i, label in enumerate(ds["labels"])
                       if int(label) == replacement]
        donors = []
        donor_ids = set()
        donor_keys = []
        original_multiplicity = Counter()
        for ds, index in donor_slots:
            source = ds.get("source_dataset_indices")
            identity = (int(ds.get("task_id", -1)),
                        int(source[index]) if source is not None else id(ds),
                        0 if source is not None else index)
            original_multiplicity[identity] += 1
            if identity not in donor_ids:
                donor_ids.add(identity)
                donors.append((ds, index))
                donor_keys.append(identity)
        if slots and not donors:
            raise ValueError("client {} has A slots but no C donors".format(client_id))
        multiplicity = original_multiplicity.copy()
        for slot_number, (target, index) in enumerate(slots):
            donor_number = slot_number % len(donors)
            donor, donor_index = donors[donor_number]
            if materialize:
                target["images"][index] = donor["images"][donor_index]
                target["labels"][index] = replacement
            multiplicity[donor_keys[donor_number]] += 1
        maximum = max(multiplicity.values(), default=0)
        if maximum > max_multiplicity:
            raise ValueError("client {} exceeds C donor multiplicity limit".format(client_id))
        manifest.append(dict(
            client_id=client_id, n_A_slots=len(slots),
            n_C_slots=len(donor_slots), n_C_unique=len(donors),
            max_C_multiplicity=maximum,
            mean_C_multiplicity=((len(slots)+len(donor_slots)) / len(donors)
                                 if donors else 0.0),
        ))
    return copied, manifest
