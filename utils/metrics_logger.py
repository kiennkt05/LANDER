"""FedCBDR class summaries and dense directed historical-pair artifacts."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F


def probe_by_class(logits, labels):
    """Functional probe metrics keyed by remapped class id."""
    pred = logits.argmax(1)
    ce = F.cross_entropy(logits, labels, reduction="none")
    others = logits.clone()
    others[torch.arange(len(labels)), labels] = -torch.inf
    margin = logits[torch.arange(len(labels)), labels] - others.max(1).values
    return {
        int(c): dict(acc=float((pred[labels == c] == c).float().mean()),
                     ce=float(ce[labels == c].mean()),
                     margin=float(margin[labels == c].mean()) if logits.shape[1] > 1 else None)
        for c in labels.unique().tolist()
    }


class MetricsLogger:
    def __init__(self, directory, seed, variant="baseline", pair_tasks=(2, 4)):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.seed = int(seed)
        self.variant = variant
        self.pair_tasks = set(pair_tasks)
        self.previous = {}
        self.boundary = {}
        self.last_class_metrics = {}
        self.boundary_class_metrics = {}

    def write(self, event, task, round_id, **values):
        row = dict(schema_version=2, event=event, seed=self.seed,
                   variant=self.variant, task=task, round=round_id, **values)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def accuracy(self, logits, labels):
        return float((logits.argmax(1) == labels).float().mean())

    def probe(self, task, round_id, logits, labels, features, registry, exposure=None):
        pred = logits.argmax(1)
        class_metrics = probe_by_class(logits, labels)
        self.last_class_metrics = class_metrics
        stats = {}
        old_classes = [c for c in sorted(class_metrics) if registry.task_for(c) < task]
        for c, values in class_metrics.items():
            mask = labels == c
            x = features[mask].double()
            mu = x.mean(0)
            centered = x - mu
            cov = centered.T @ centered / max(1, len(x) - 1)
            stats[c] = (mu, cov)
            ct = registry.task_for(c)
            wrong = pred[mask][pred[mask] != c].tolist()
            confusion = dict(same_old_task=0, different_old_task=0,
                             old_to_current=0, current_to_old=0,
                             same_current_task=0)
            for p in wrong:
                pt = registry.task_for(p)
                key = ("old_to_current" if pt == task else
                       "same_old_task" if pt == ct else "different_old_task") if ct < task else (
                       "current_to_old" if pt < task else "same_current_task")
                confusion[key] += 1
            prior = self.previous.get(c)
            self.write("class_probe", task, round_id, class_id=c,
                       class_task_id=ct, class_age=task-ct, count=int(mask.sum()),
                       **values, confusion_counts=confusion,
                       mean_drift=None if prior is None else float((mu-prior[0]).norm()),
                       covariance_drift=None if prior is None else float((cov-prior[1]).norm()))

        if task in self.pair_tasks and old_classes:
            self._save_pairs(task, round_id, logits, labels, pred, stats,
                             old_classes, registry, exposure)
            # Raw probe features are a compact, lossless input for rebuilding
            # every class mean and sample covariance used above.
            torch.save(dict(labels=labels, features=features),
                       self.directory / "features_t{}_r{}.pt".format(task, round_id))
        self.previous = stats

    def _save_pairs(self, task, round_id, logits, labels, pred, stats,
                    classes, registry, exposure):
        # Functional/feature matrices use [victim B, aggressor A]. Exposure
        # matrices use [aggressor A, victim B].
        n = len(classes)
        confusion = torch.zeros((n, n), dtype=torch.float32)
        margin = torch.zeros((n, n), dtype=torch.float32)
        means = torch.stack([stats[c][0] for c in classes])
        tasks = torch.tensor([registry.task_for(c) for c in classes])
        pair_type = (tasks[:, None] != tasks[None, :]).to(torch.int8)
        pair_type.fill_diagonal_(-1)
        for bi, b in enumerate(classes):
            mask = labels == b
            margin[bi] = logits[mask, b].mean() - logits[mask][:, classes].mean(0)
            counts = torch.bincount(pred[mask], minlength=logits.shape[1])
            confusion[bi] = counts[classes].float() / int(mask.sum())
        boundary_means = self.boundary.get("means", {})
        boundary_margin = self.boundary.get("margin", {})
        delta_margin = torch.full_like(margin, float("nan"))
        feature_direction = torch.full_like(margin, float("nan"))
        relative_direction = torch.full_like(margin, float("nan"))
        if all(c in boundary_means for c in classes):
            boundary = torch.stack([boundary_means[c] for c in classes])
            movement = means - boundary
            direction = boundary[None, :, :] - boundary[:, None, :]
            norm = direction.norm(dim=2, keepdim=True)
            valid = norm.squeeze(2) > 0
            direction = direction / norm.clamp_min(torch.finfo(direction.dtype).eps)
            absolute = torch.einsum("bd,bad->ba", movement, direction)
            relative = torch.einsum(
                "bad,bad->ba", movement[:, None, :] - movement[None, :, :], direction)
            feature_direction[valid] = absolute[valid].float()
            relative_direction[valid] = relative[valid].float()
        for bi, b in enumerate(classes):
            for ai, a in enumerate(classes):
                if a != b and (b, a) in boundary_margin:
                    delta_margin[bi, ai] = margin[bi, ai] - boundary_margin[b, a]
        artifact = dict(schema_version=2, task=task, round=round_id,
                        class_ids=classes, pair_type=pair_type,
                        pair_confusion_rate=confusion, pair_margin=margin,
                        delta_pair_margin=delta_margin,
                        feature_direction=feature_direction,
                        relative_feature_direction=relative_direction)
        if exposure is not None:
            artifact.update(exposure)
        filename = "pair_metrics_T{}_R{}.pt".format(task, str(round_id).zfill(3))
        torch.save(artifact, self.directory / filename)
        groups = {}
        for name, code in (("intra", 0), ("inter", 1)):
            mask = (pair_type == code) & torch.isfinite(delta_margin)
            values = delta_margin[mask]
            groups[name] = dict(count=int(mask.sum()),
                                mean_delta_margin=float(values.mean()) if len(values) else None,
                                mean_confusion=float(confusion[mask].mean()) if len(values) else None)
        self.write("pair_summary", task, round_id, **groups)

    def set_boundary(self, logits=None, labels=None):
        means = {c: values[0].clone() for c, values in self.previous.items()}
        margins = {}
        if logits is not None and labels is not None:
            classes = sorted(means)
            for b in classes:
                mask = labels == b
                for a in classes:
                    if a != b:
                        margins[b, a] = float((logits[mask, b] - logits[mask, a]).mean())
        self.boundary = dict(means=means, margin=margins)
        self.boundary_class_metrics = dict(self.last_class_metrics)
