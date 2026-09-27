"""Versioned JSONL monitoring events plus compact tensor feature artifacts."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F


class MetricsLogger:
    def __init__(self, directory, seed):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.seed = int(seed)
        self.previous = {}
        self.boundary = {}

    def write(self, event, task, round_id, **values):
        row = dict(schema_version=1, event=event, seed=self.seed, variant="baseline",
                   task=task, round=round_id, **values)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def accuracy(self, logits, labels):
        return float((logits.argmax(1) == labels).float().mean())

    def probe(self, task, round_id, logits, labels, features, registry):
        pred = logits.argmax(1)
        ce = F.cross_entropy(logits, labels, reduction="none")
        others = logits.clone()
        others[torch.arange(len(labels)), labels] = -torch.inf
        margins = logits[torch.arange(len(labels)), labels] - others.max(1).values
        stats = {}
        for c in labels.unique().tolist():
            mask = labels == c
            x = features[mask].double()
            mu = x.mean(0)
            centered = x - mu
            cov = centered.T @ centered / max(1, len(x) - 1)
            stats[c] = (mu, cov)
            ct = registry.task_for(c)
            wrong = pred[mask][pred[mask] != c].tolist()
            confusion = dict(same_old_task=0, different_old_task=0, old_to_current=0, current_to_old=0, same_current_task=0)
            for p in wrong:
                pt = registry.task_for(p)
                key = ("old_to_current" if pt == task else "same_old_task" if pt == ct else "different_old_task") if ct < task else ("current_to_old" if pt < task else "same_current_task")
                confusion[key] += 1
            prior = self.previous.get(c)
            self.write("class_probe", task, round_id, class_id=c, class_task_id=ct,
                       class_age=task-ct, count=int(mask.sum()), acc=self.accuracy(logits[mask], labels[mask]),
                       ce=float(ce[mask].mean()), margin=float(margins[mask].mean()) if logits.shape[1] > 1 else None,
                       confusion_counts=confusion,
                       mean_drift=None if prior is None else float((mu-prior[0]).norm()),
                       covariance_drift=None if prior is None else float((cov-prior[1]).norm()))
        groups = {"intra": [], "inter": []}
        for a in stats:
            for b in stats:
                if a >= b:
                    continue
                distance = float((stats[a][0]-stats[b][0]).norm())
                old_pair = registry.task_for(a) < task and registry.task_for(b) < task
                kind = "intra" if registry.task_for(a) == registry.task_for(b) else "inter"
                reference = self.boundary.get((a, b))
                delta = None if reference is None else distance-reference
                if old_pair and delta is not None:
                    groups[kind].append(delta)
                self.write("pair_probe", task, round_id, class_a=a, class_b=b, pair_type=kind,
                           old_pair=old_pair, distance=distance, boundary_delta=delta)
        self.write("drift_summary", task, round_id, **{
            "delta_d_"+kind: sum(values)/len(values) if values else None for kind, values in groups.items()})
        torch.save(stats, self.directory / "features_t{}_r{}.pt".format(task, round_id))
        self.previous = stats

    def set_boundary(self):
        self.boundary = {(a,b): float((self.previous[a][0]-self.previous[b][0]).norm())
                         for a in self.previous for b in self.previous if a < b}
