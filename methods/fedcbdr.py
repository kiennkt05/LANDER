import copy
import logging
import math
import random
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from methods.base import BaseLearner
from utils.data_manager import (
    DatasetSplit,
    average_weights,
    partition_data,
    pil_loader,
    setup_seed,
)
from utils.inc_net import IncrementalNet
from utils.task_registry import TaskRegistry
from utils.probe_manager import ProbeManager, extract_probe_outputs
from utils.metrics_logger import MetricsLogger, probe_by_class
from utils.fedcbdr_interactions import ReplayExposureTracker


# -----------------------------------------------------------------------------
# Data / transform helpers
# -----------------------------------------------------------------------------

def _label_distribution(labels):
    values = np.asarray(labels, dtype=np.int64)
    if not len(values):
        return {}
    classes, counts = np.unique(values, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(classes, counts)
    }


class TaggedDataset(Dataset):
    """Attach replay provenance and an optional experimental replay weight."""

    def __init__(self, dataset, is_replay=False, task_id=None):
        self.dataset = dataset
        self.is_replay = bool(is_replay)
        self.task_id = int(getattr(dataset, "task_id", -1) if task_id is None else task_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample_id, image, label = self.dataset[index]
        if self.is_replay:
            weight = float(self.dataset.sampling_weights[index])
        else:
            weight = 1.0
        return sample_id, image, int(label), self.is_replay, weight, self.task_id


class ReplayDataset(Dataset):
    """Client-local replay cache.

    Duplicate GDR draws are deliberately retained as distinct replay samples.
    ``local_indices`` are positions inside the client's ``DatasetSplit``.
    """

    def __init__(
        self,
        dataset,
        local_indices,
        transform,
        use_path,
        sampling_weights=None,
        task_id=-1,
    ):
        indices = np.asarray(local_indices, dtype=np.int64)
        self.source_local_indices = indices
        self.task_id = int(task_id)
        self.source_dataset_indices = np.asarray(dataset.idxs, dtype=np.int64)[indices] if hasattr(dataset, "idxs") else indices.copy()

        if hasattr(dataset, "images") and dataset.images is not None:
            self.images = dataset.images[indices]
            self.labels = np.asarray(dataset.labels[indices], dtype=np.int64)
        elif hasattr(dataset, "dataset") and hasattr(dataset, "idxs"):
            global_indices = np.asarray(dataset.idxs)[indices]
            base_dataset = dataset.dataset
            base_images = base_dataset.images
            if not isinstance(base_images, np.ndarray):
                base_images = np.asarray(base_images)
            base_labels = base_dataset.labels
            if not isinstance(base_labels, np.ndarray):
                base_labels = np.asarray(base_labels)
            self.images = base_images[global_indices]
            self.labels = np.asarray(base_labels[global_indices], dtype=np.int64)
        else:
            raise AttributeError(
                f"Cannot extract images and labels from {type(dataset).__name__}"
            )

        self.transform = transform
        self.use_path = use_path

        if sampling_weights is None:
            sampling_weights = np.ones(len(indices), dtype=np.float32)
        self.sampling_weights = np.asarray(sampling_weights, dtype=np.float32)

        if len(self.sampling_weights) != len(self.labels):
            raise ValueError(
                "sampling_weights and local_indices must have equal length"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_value = self.images[index]

        if self.use_path:
            image = pil_loader(image_value)
        elif isinstance(image_value, Image.Image):
            image = image_value.copy()
        else:
            image = Image.fromarray(image_value)

        return (
            int(self.source_local_indices[index]),
            self.transform(image),
            int(self.labels[index]),
        )

    def snapshot(self):
        return dict(task_id=self.task_id,
                    source_local_indices=self.source_local_indices.copy(),
                    source_dataset_indices=self.source_dataset_indices.copy(),
                    sampling_weights=self.sampling_weights.copy(),
                    images=self.images.copy(), labels=self.labels.copy(),
                    use_path=self.use_path)

    @classmethod
    def from_snapshot(cls, state, transform):
        result = cls.__new__(cls)
        for name, value in state.items():
            setattr(result, name, value)
        result.transform = transform
        return result


# -----------------------------------------------------------------------------
# GDR selectors
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GDRSelection:
    local_index: int
    probability: float
    sampling_weight: float


@dataclass(frozen=True)
class GDRDiagnostics:
    draw_count: int
    unique_id_count: int
    multiplicities: Dict[Tuple[int, int], int]
    probability_sum: float
    sampling_matrix_frobenius_norm: Optional[float]


class _PairwiseMask(object):
    """Implicit 2x2-block orthogonal left mask."""

    def __init__(self, size, generator, dtype):
        order = torch.randperm(size, generator=generator)
        even_size = size - size % 2
        self.left = order[0:even_size:2]
        self.right = order[1:even_size:2]

        angle = 2 * math.pi * torch.rand(len(self.left), generator=generator)
        self.cos = torch.cos(angle).to(dtype)
        self.sin = torch.sin(angle).to(dtype)

    def apply(self, matrix, transpose=False):
        output = matrix.clone()
        if not len(self.left):
            return output

        left = self.left.to(matrix.device)
        right = self.right.to(matrix.device)
        cos = self.cos.to(matrix.device)
        sin = self.sin.to(matrix.device)

        x0 = matrix[left].clone()
        x1 = matrix[right].clone()

        if transpose:
            output[left] = cos[:, None] * x0 + sin[:, None] * x1
            output[right] = -sin[:, None] * x0 + cos[:, None] * x1
        else:
            output[left] = cos[:, None] * x0 - sin[:, None] * x1
            output[right] = sin[:, None] * x0 + cos[:, None] * x1

        return output


class GlobalPerspectiveReplaySelector(object):
    """Label-free client/server GDR with one global SVD.

    The client de-masking of the returned U blocks is the supplied
    reconstruction's auditable repair, not an explicitly stated FedCBDR
    equation.
    """

    def __init__(
        self,
        task_budget,
        seed,
        compute_device="cpu",
        mask_mode="dense_qr",
        leverage_mode="economy",
        rank=None,
        normalization_mode="global",
        replacement="with",
        correction_mode="none",
    ):
        if task_budget <= 0:
            raise ValueError("gdr_task_budget must be positive")
        if mask_mode not in {"dense_qr", "implicit_pairwise"}:
            raise ValueError("unknown gdr mask mode")
        if leverage_mode not in {"full", "economy", "truncated"}:
            raise ValueError("unknown gdr leverage mode")
        if leverage_mode == "truncated" and (rank is None or rank <= 0):
            raise ValueError("gdr_rank must be positive for truncated leverage")
        if normalization_mode not in {"global", "eq6_local"}:
            raise ValueError("invalid GDR normalization mode")
        if replacement not in {"with", "without"}:
            raise ValueError("invalid GDR replacement mode")
        if correction_mode not in {
            "none",
            "sampling_matrix",
            "replay_loss_experimental",
        }:
            raise ValueError("invalid GDR correction mode")

        self.task_budget = int(task_budget)
        self.compute_device = torch.device(compute_device)
        self.mask_mode = mask_mode
        self.leverage_mode = leverage_mode
        self.rank = rank
        self.normalization_mode = normalization_mode
        self.replacement = replacement
        self.correction_mode = correction_mode

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

        self.last_diagnostics = None
        self.last_probabilities = None
        self.last_leverage_scores = None

    def _orthogonal(self, size, dtype):
        matrix = torch.randn(
            size,
            size,
            generator=self.generator,
            dtype=torch.float32,
        )
        return torch.linalg.qr(matrix).Q.to(dtype)

    def _server_svd(self, pseudo_features):
        u, _, _ = torch.linalg.svd(
            pseudo_features,
            full_matrices=self.leverage_mode == "full",
        )

        if self.leverage_mode == "truncated":
            maximum = min(pseudo_features.shape)
            if self.rank is None or self.rank > maximum:
                raise ValueError("gdr_rank must be in [1, {}]".format(maximum))
            return u[:, : self.rank]

        return u

    def _normalize(self, blocks):
        epsilon = torch.finfo(blocks[0].dtype).eps
        safe = [block.clamp_min(epsilon) for block in blocks]

        if self.normalization_mode == "global":
            scores = torch.cat(safe)
            return scores / scores.sum()

        # Eq. 6-style client-local normalization, followed by a final global
        # normalization so torch.multinomial receives one valid distribution.
        scores = torch.cat([block / block.sum() for block in safe])
        return scores / scores.sum()

    def select(self, local_features):
        selection = {
            client_id: [] for client_id in range(len(local_features))
        }

        nonempty = [features for features in local_features if len(features)]
        if not nonempty:
            self.last_diagnostics = GDRDiagnostics(0, 0, {}, 0.0, None)
            return selection

        dim = int(nonempty[0].shape[1])
        dtype = nonempty[0].dtype

        if any(
            features.ndim != 2 or features.shape[1] != dim
            for features in local_features
            if len(features)
        ):
            raise ValueError(
                "all non-empty local feature tensors must have shape (n_client, d)"
            )

        q = self._orthogonal(dim, dtype)

        originals = []
        pseudo_blocks = []
        masks = []
        active_clients = []

        for client_id, block in enumerate(local_features):
            features = block.detach().cpu().to(dtype)
            if not len(features):
                continue

            if self.mask_mode == "dense_qr":
                mask = self._orthogonal(len(features), dtype)
                pseudo = mask @ features @ q
            else:
                mask = _PairwiseMask(len(features), self.generator, dtype)
                pseudo = mask.apply(features) @ q

            originals.append(features)
            pseudo_blocks.append(pseudo)
            masks.append(mask)
            active_clients.append(client_id)

        masked_u = self._server_svd(
            torch.cat(pseudo_blocks).to(self.compute_device)
        ).detach().cpu()

        leverage_blocks = []
        offset = 0

        for features, mask in zip(originals, masks):
            u_block = masked_u[offset : offset + len(features)]
            offset += len(features)

            if isinstance(mask, Tensor):
                unmasked = mask.t() @ u_block
            else:
                unmasked = mask.apply(u_block, transpose=True)

            leverage_blocks.append(unmasked.square().sum(dim=1))

        probabilities = self._normalize(leverage_blocks)
        self.last_probabilities = probabilities.detach().clone()
        self.last_leverage_scores = torch.cat(leverage_blocks).detach().clone()

        if (
            self.replacement == "without"
            and self.task_budget > len(probabilities)
        ):
            raise ValueError(
                "without-replacement GDR budget exceeds global task pool"
            )

        sampled = torch.multinomial(
            probabilities,
            self.task_budget,
            replacement=self.replacement == "with",
            generator=self.generator,
        )
        corrections = torch.rsqrt(
            self.task_budget * probabilities[sampled]
        )

        ends = np.cumsum([len(block) for block in originals]).tolist()
        diagnostic_rows = []

        for row, correction in zip(
            sampled.tolist(), corrections.tolist()
        ):
            position = next(
                index for index, end in enumerate(ends) if row < end
            )
            start = 0 if position == 0 else ends[position - 1]
            client_id = active_clients[position]
            local_id = row - start

            value = GDRSelection(
                local_index=int(local_id),
                probability=float(probabilities[row]),
                sampling_weight=float(correction),
            )
            selection[client_id].append(value)

            if self.correction_mode == "sampling_matrix":
                diagnostic_rows.append(
                    originals[position][local_id] * value.sampling_weight
                )

        pairs = [
            (client_id, value.local_index)
            for client_id, values in selection.items()
            for value in values
        ]

        sampling_norm = None
        if diagnostic_rows:
            sampling_norm = float(
                torch.linalg.vector_norm(torch.stack(diagnostic_rows))
            )

        self.last_diagnostics = GDRDiagnostics(
            draw_count=len(pairs),
            unique_id_count=len(set(pairs)),
            multiplicities=dict(Counter(pairs)),
            probability_sum=float(probabilities.sum()),
            sampling_matrix_frobenius_norm=sampling_norm,
        )
        return selection


class RepoStyleReplaySelector(object):
    """Public-repository-style FedCBDR/FEAT GDR selector.

    Retained behavior:
      1. dense P_k and shared Q masking,
      2. one SVD independently per masked client,
      3. per-client leverage normalization,
      4. equal 1/K probability mass per client,
      5. floor(M/K) with-replacement draws per client,
      6. remainder draws from the concatenated distribution.
    """

    def __init__(self, task_budget, seed):
        if task_budget <= 0:
            raise ValueError("gdr_task_budget must be positive")

        self.task_budget = int(task_budget)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

        self.last_diagnostics = None
        self.last_probabilities = None
        self.last_leverage_scores = None

    def _orthogonal(self, size, dtype):
        matrix = torch.randn(
            size,
            size,
            generator=self.generator,
            dtype=torch.float32,
        )
        return torch.linalg.qr(matrix).Q.to(dtype)

    def select(self, local_features):
        num_clients = len(local_features)
        if num_clients == 0:
            raise ValueError("local_features must be non-empty")
        if any(len(features) == 0 for features in local_features):
            raise ValueError(
                "public-repo GDR assumes every client has at least one sample"
            )

        features = [block.detach().cpu() for block in local_features]
        dim = int(features[0].shape[1])

        if any(
            block.ndim != 2 or block.shape[1] != dim
            for block in features
        ):
            raise ValueError(
                "all local feature tensors must have shape (n_client, d)"
            )

        # Match the supplied public-code reconstruction: construct all P_k
        # masks before constructing shared Q.
        p_masks = [
            self._orthogonal(len(block), block.dtype)
            for block in features
        ]
        q = self._orthogonal(dim, features[0].dtype)

        leverage_blocks = []
        for p_mask, block in zip(p_masks, features):
            masked = p_mask @ block @ q
            u, _, _ = torch.svd(masked)
            tau = u.square().sum(dim=1)
            tau = tau / tau.sum()
            leverage_blocks.append(tau)

        probabilities = torch.cat(
            [tau / num_clients for tau in leverage_blocks]
        )
        probabilities = probabilities / probabilities.sum()

        self.last_probabilities = probabilities.detach().clone()
        self.last_leverage_scores = torch.cat(leverage_blocks).detach().clone()

        selection = {i: [] for i in range(num_clients)}
        quota = self.task_budget // num_clients
        offsets = np.cumsum(
            [0] + [len(block) for block in features]
        ).tolist()

        for client_id in range(num_clients):
            start = offsets[client_id]
            end = offsets[client_id + 1]
            client_p = probabilities[start:end]

            sampled = torch.multinomial(
                client_p,
                quota,
                replacement=True,
                generator=self.generator,
            )

            for local_id in sampled.tolist():
                global_id = start + local_id
                selection[client_id].append(
                    GDRSelection(
                        local_index=int(local_id),
                        probability=float(probabilities[global_id]),
                        sampling_weight=1.0,
                    )
                )

        remaining = self.task_budget - quota * num_clients
        if remaining:
            sampled = torch.multinomial(
                probabilities,
                remaining,
                replacement=True,
                generator=self.generator,
            )

            for global_id in sampled.tolist():
                client_id = next(
                    i
                    for i in range(num_clients)
                    if offsets[i] <= global_id < offsets[i + 1]
                )
                local_id = global_id - offsets[client_id]
                selection[client_id].append(
                    GDRSelection(
                        local_index=int(local_id),
                        probability=float(probabilities[global_id]),
                        sampling_weight=1.0,
                    )
                )

        pairs = [
            (client_id, value.local_index)
            for client_id, values in selection.items()
            for value in values
        ]
        self.last_diagnostics = GDRDiagnostics(
            draw_count=len(pairs),
            unique_id_count=len(set(pairs)),
            multiplicities=dict(Counter(pairs)),
            probability_sum=float(probabilities.sum()),
            sampling_matrix_frobenius_norm=None,
        )
        return selection


# -----------------------------------------------------------------------------
# FedCBDR/TTS loss
# -----------------------------------------------------------------------------


class TaskAwareTemperatureScalingLoss(nn.Module):
    """FedCBDR task-aware temperature scaling audit modes."""

    def __init__(
        self,
        num_old_classes,
        tau_old,
        tau_new,
        weight_old,
        weight_new,
        mode="paper_eq",
        correction_mode="none",
    ):
        super().__init__()

        if tau_old <= 0 or tau_new <= 0:
            raise ValueError("TTS temperatures must be positive")
        if mode not in {"paper_eq", "repo_dual"}:
            raise ValueError("invalid TTS mode")

        self.num_old_classes = int(num_old_classes)
        self.tau_old = float(tau_old)
        self.tau_new = float(tau_new)
        self.weight_old = float(weight_old)
        self.weight_new = float(weight_new)
        self.mode = mode
        self.correction_mode = correction_mode

    def forward(
        self,
        logits,
        labels,
        is_replay=None,
        replay_weights=None,
    ):
        if self.num_old_classes <= 0:
            return F.cross_entropy(logits, labels)
        if self.num_old_classes >= logits.shape[1]:
            raise ValueError(
                "num_old_classes must be smaller than output dimension"
            )

        scaled = torch.cat(
            (
                logits[:, : self.num_old_classes] / self.tau_old,
                logits[:, self.num_old_classes :] / self.tau_new,
            ),
            dim=1,
        )
        old = labels < self.num_old_classes

        if self.mode == "repo_dual":
            reference = logits[:, 0]
            sample_temperatures = torch.where(
                old,
                torch.full_like(reference, self.tau_old),
                torch.full_like(reference, self.tau_new),
            )
            scaled = scaled / sample_temperatures[:, None]

        losses = F.cross_entropy(scaled, labels, reduction="none")
        self.last_sample_ce = losses.detach()

        if (
            self.correction_mode == "replay_loss_experimental"
            and is_replay is not None
            and replay_weights is not None
        ):
            losses = torch.where(
                is_replay.bool(),
                losses * replay_weights.to(losses.device),
                losses,
            )

        if self.mode == "repo_dual":
            task_weights = torch.where(
                old,
                torch.full_like(losses, self.weight_old),
                torch.full_like(losses, self.weight_new),
            )
            return (losses * task_weights).mean()

        output = logits.sum() * 0.0
        if old.any():
            output = output + self.weight_old * losses[old].mean()
        if (~old).any():
            output = output + self.weight_new * losses[~old].mean()
        return output


def uniform_average_state_dicts(state_dicts):
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    return average_weights(list(state_dicts))


# -----------------------------------------------------------------------------
# Learner -- organized like verified LANDER
# -----------------------------------------------------------------------------


class FedCBDR(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

        self.retained_ds_all = [
            [] for _ in range(int(args["num_users"]))
        ]
        self.user_groups = {}
        self.train_dataset = None
        self.logger = logging.getLogger(__name__)
        self.task_registry = TaskRegistry()
        self.task_class_ranges: List[Tuple[int, int, int]] = self.task_registry.ranges
        self.probes = ProbeManager(args.get("fedcbdr_probe_per_class", 32), self.seed)
        monitor_dir = args.get("fedcbdr_monitor_dir")
        self.metrics = MetricsLogger(Path(monitor_dir) / "seed_{}".format(self.seed), self.seed) if monitor_dir else None
        self._mass = {}
        self._active_tracker = None
        self._ablation_mode = False

        self._validate_options()

    def _validate_options(self):
        if self.args.get("gdr_task_budget") is None:
            if not self.args.get("fedcbdr_legacy_mem_size", False):
                raise ValueError(
                    "FedCBDR requires --gdr-task-budget; legacy mem_size "
                    "requires --fedcbdr-legacy-mem-size"
                )
            self.args["gdr_task_budget"] = (
                int(self.args["mem_size"])
                * int(self.args["increment"])
            )

        if (
            self.args.get("gdr_leverage_mode", "economy") == "truncated"
            and self.args.get("gdr_rank") is None
        ):
            raise ValueError(
                "--gdr-rank is required with truncated leverage"
            )

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        test_acc = self._compute_accuracy(
            self._old_network,
            self.test_loader,
        )
        print("After Test Acc: %s" % test_acc)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = (
            self._known_classes
            + data_manager.get_task_size(self._cur_task)
        )

        self._network.update_fc(self._total_classes)
        self.task_registry.append(self._cur_task, self._known_classes, self._total_classes)
        self.probes.add_classes(data_manager, range(self._known_classes, self._total_classes))
        self._prepare_model(self._network)
        print(
            "Learning on {}-{}".format(
                self._known_classes,
                self._total_classes,
            )
        )

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=self.args["num_worker"],
            pin_memory=True,
            multiprocessing_context=self.args["mulc"] if self.args["num_worker"] > 0 else None,
            persistent_workers=self.args["num_worker"] > 0,
        )
        print(self.test_loader)

        if self._cur_task > 0:
            old_dataset = data_manager.get_dataset(
                np.arange(0, self._known_classes),
                source="test",
                mode="test",
            )
            self.old_loader = DataLoader(
                old_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=self.args["num_worker"],
                pin_memory=True,
                multiprocessing_context=self.args["mulc"] if self.args["num_worker"] > 0 else None,
                persistent_workers=self.args["num_worker"] > 0,
            )

            new_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="test",
                mode="test",
            )
            self.new_loader = DataLoader(
                new_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=self.args["num_worker"],
                pin_memory=True,
                multiprocessing_context=self.args["mulc"] if self.args["num_worker"] > 0 else None,
                persistent_workers=self.args["num_worker"] > 0,
            )

        setup_seed(self.seed, fast_cuda=self.args.get("fast_cuda", False))
        self._fl_train(self.train_dataset, self.test_loader)

    def _selected_clients(self):
        # Preserve the supplied FedCBDR participation semantics exactly.
        users = int(self.args["num_users"])
        fraction = float(self.args.get("frac", 1.0))
        count = min(users, max(1, math.ceil(users * fraction)))

        if count == users:
            return np.arange(users)
        return np.random.choice(users, count, replace=False)

    def _learning_rate(self, round_id):
        base = float(self.args["local_lr"])
        schedule = self.args.get("fedcbdr_lr_schedule", "constant")

        if schedule == "constant":
            return base

        # Preserve the supplied implementation: every non-constant mode uses
        # the cosine schedule below.
        eta_min = min(1e-3, base)
        return eta_min + 0.5 * (base - eta_min) * (
            1
            + math.cos(
                math.pi
                * round_id
                / max(1, int(self.args["com_round"]))
            )
        )

    def _compute_fedcbdr_accuracy(self, model, loader):
        # LANDER's verified BaseLearner contract is the two-argument form.
        # Retain compatibility with the supplied FedCBDR's optional scaled
        # evaluation only when the local BaseLearner actually supports it.
        use_scale = bool(self.args.get("scale", False)) and self._cur_task > 0
        if use_scale:
            try:
                return self._compute_accuracy(
                    model,
                    loader,
                    scale=True,
                    old_classes=self._known_classes,
                )
            except TypeError:
                pass
        return self._compute_accuracy(model, loader)

    def _client_dataset(self, client_id):
        current = TaggedDataset(
            DatasetSplit(
                self.train_dataset,
                self.user_groups[client_id],
            ),
            is_replay=False,
            task_id=self._cur_task,
        )
        replay = [
            TaggedDataset(dataset, is_replay=True)
            for dataset in self.retained_ds_all[client_id]
        ]

        if not replay:
            return current
        repeat = int(self.args.get("fedcbdr_replay_repeat", 1))
        if repeat < 1:
            raise ValueError("fedcbdr_replay_repeat must be positive")
        return ConcatDataset([current] + replay * repeat)

    def _make_client_loader(self, client_id):
        return DataLoader(
            self._client_dataset(client_id),
            batch_size=self.args["local_bs"],
            shuffle=True,
            num_workers=self.args["num_worker"],
            pin_memory=True,
            multiprocessing_context=self.args["mulc"] if self.args["num_worker"] > 0 else None,
            persistent_workers=self.args["num_worker"] > 0,
        )

    def _fl_train(self, train_dataset, test_loader, frozen_partition=False):
        self._prepare_model(self._network)
        self.best_model = None  # Best model using the lowest training loss
        self.lowest_loss = np.inf
        self.best_round = None

        if not frozen_partition:
            user_groups, _ = partition_data(
                train_dataset.labels,
                beta=self.args["beta"],
                n_parties=self.args["num_users"],
            )
            self.user_groups = {
                int(client_id): indices
                for client_id, indices in user_groups.items()
            }

        print(
            "FedCBDR audit: protocol={} aggregation=uniform_repo "
            "mask={} leverage={} normalization={} replacement={} "
            "correction={} tts={} budget={}".format(
                self.args.get("gdr_protocol", "paper_global"),
                self.args.get("gdr_mask_mode", "dense_qr"),
                self.args.get("gdr_leverage_mode", "economy"),
                self.args.get("gdr_normalization_mode", "global"),
                self.args.get("gdr_replacement", "with"),
                self.args.get("gdr_correction_mode", "none"),
                self.args.get("tts_mode", "paper_eq"),
                self.args["gdr_task_budget"],
            )
        )

        prog_bar = tqdm(range(self.args["com_round"]))
        client_loaders = {
            idx: self._make_client_loader(idx)
            for idx in range(self.args["num_users"])
        }
        client_model = copy.deepcopy(self._network)
        if not self._ablation_mode and self._cur_task in (2, 4):
            self.save_transition_start()

        for _, com in enumerate(prog_bar):
            local_weights = []
            loss_weight = []
            self._mass = {}
            client_exposure = {}
            if self.metrics:
                before = self._probe_by_class(self._network)
                local_by_class = {}

            idxs_users = self._selected_clients()
            lr = self._learning_rate(com)

            for idx in idxs_users:
                client_model.load_state_dict(self._network.state_dict())
                local_train_loader = client_loaders[int(idx)]
                self._active_tracker = (ReplayExposureTracker(self._known_classes)
                                        if self.metrics and self._known_classes else None)

                if self._cur_task == 0:
                    w, total_loss = self._local_update(
                        client_model,
                        local_train_loader,
                        lr,
                    )
                else:
                    w, total_loss = self._local_finetune(
                        client_model,
                        local_train_loader,
                        lr,
                    )

                local_weights.append({key: value.detach().clone() for key, value in w.items()})
                loss_weight.append(total_loss)
                if self.metrics:
                    local_by_class[int(idx)] = self._probe_by_class(client_model)
                    if self._active_tracker is not None:
                        client_exposure[int(idx)] = self._active_tracker.state_dict()
                self._active_tracker = None

            global_weights = uniform_average_state_dicts(local_weights)
            self._network.load_state_dict(global_weights)
            del local_weights, global_weights
            if self.metrics:
                exposure = self._pack_exposure(client_exposure)
                self._log_probe(com, exposure=exposure)
                if exposure is not None:
                    for client_id in range(self.args["num_users"]):
                        self.metrics.write("exposure_summary", self._cur_task, com,
                                           scope="client", client_id=client_id,
                                           replay_draws=int(exposure["draws"][client_id].sum()),
                                           co_batch_count=int(exposure["co_batch_count"][client_id].sum()),
                                           sequential_exposure=int(exposure["sequential_exposure"][client_id].sum()))
                    self.metrics.write("exposure_summary", self._cur_task, com,
                                       scope="round", client_id=None,
                                       replay_draws=int(exposure["round_draws"].sum()),
                                       co_batch_count=int(exposure["round_co_batch_count"].sum()),
                                       sequential_exposure=int(exposure["round_sequential_exposure"].sum()))
                after = self._probe_by_class(self._network)
                weighted_local = {
                    c: {metric: sum(local[c][metric] for local in local_by_class.values()) /
                        len(local_by_class) for metric in ("acc", "ce", "margin")
                        if before[c][metric] is not None}
                    for c in before
                }
                local_delta = {
                    client: {c: {metric: value[metric] - before[c][metric]
                                 for metric in weighted_local[c]}
                             for c, value in classes.items()}
                    for client, classes in local_by_class.items()
                }
                fedavg_delta = {
                    c: {metric: after[c][metric] - weighted_local[c][metric]
                        for metric in weighted_local[c]} for c in before
                }
                self.metrics.write("fedavg_by_class", self._cur_task, com,
                                   global_before=before, local_after=local_by_class,
                                   global_after=after, local_delta=local_delta,
                                   fedavg_delta=fedavg_delta)
                for (c, replay), (count, mass) in self._mass.items():
                    self.metrics.write("update_mass", self._cur_task, com, class_id=c,
                                       class_task_id=self.task_registry.task_for(c), is_replay=replay,
                                       draws=count, mean_ce=mass/count, ce_mass_proxy=mass)

            sum_loss = sum(loss_weight)
            if sum_loss < self.lowest_loss:
                self.lowest_loss = sum_loss
                self.best_model = copy.deepcopy(self._network.state_dict())
                self.best_round = com

            if self._should_evaluate(com):
                test_acc = self._compute_fedcbdr_accuracy(
                    self._network,
                    test_loader,
                )

                if self._cur_task > 0:
                    test_old_acc = self._compute_fedcbdr_accuracy(
                        self._network,
                        self.old_loader,
                    )
                    test_new_acc = self._compute_fedcbdr_accuracy(
                        self._network,
                        self.new_loader,
                    )
                    print(
                        "Task {}, Test_accy {:.2f} O {} N {}".format(
                            self._cur_task,
                            test_acc,
                            test_old_acc,
                            test_new_acc,
                        )
                    )

                mean_local_loss = (
                    float(np.mean(loss_weight))
                    if loss_weight
                    else 0.0
                )
                if self.metrics:
                    self.metrics.write("global_accuracy", self._cur_task, com, test_acc=test_acc,
                                       test_old_acc=test_old_acc if self._cur_task > 0 else None,
                                       test_new_acc=test_new_acc if self._cur_task > 0 else test_acc)
                info = (
                    "Task {}, Epoch {}/{} =>  Test_accy {:.2f}, "
                    "Local_loss {:.4f}"
                ).format(
                    self._cur_task,
                    com + 1,
                    self.args["com_round"],
                    test_acc,
                    mean_local_loss,
                )
                prog_bar.set_description(info)

                if getattr(self, "wandb", 0) == 1:
                    try:
                        import wandb

                        wandb.log(
                            {
                                "Task_{}, accuracy".format(
                                    self._cur_task
                                ): test_acc
                            }
                        )
                    except ImportError:
                        pass

        self._network.load_state_dict(self.best_model)  # Best model using the lowest training loss
        del self.best_model
        del client_model, client_loaders

        if self.metrics:
            self._log_probe("selected")
            self.metrics.write("selected_model", self._cur_task, "selected",
                               source_round=self.best_round,
                               selection_loss=float(self.lowest_loss))
            boundary_logits, boundary_labels, _ = extract_probe_outputs(
                self._network, self.probes.loader())
            self.metrics.set_boundary(boundary_logits, boundary_labels)
        if self._ablation_mode:
            return
        # FedCBDR's next task needs this task's selected replay buffer.
        self._construct_replay_for_current_task()
        checkpoint_dir = self.metrics.directory if self.metrics else Path(self.save_dir or "store") / "fedcbdr_seed_{}".format(self.seed)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_checkpoint(str(checkpoint_dir / "checkpoint"))

    def _local_update(self, model, train_data_loader, lr):
        print(lr)
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )

        losses = []
        for local_epoch in range(self.args["local_ep"]):
            for batch_idx, (
                _,
                images,
                labels,
                _,
                _,
                _,
            ) in enumerate(train_data_loader):
                images = self._prepare_images(images)
                labels = labels.cuda(non_blocking=True)

                with self._autocast():
                    output = model(images)["logits"]
                    sample_ce = F.cross_entropy(output, labels, reduction="none")
                    loss = sample_ce.mean()
                self._record_mass(labels, torch.zeros_like(labels, dtype=torch.bool), sample_ce)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())

        loss_values = torch.stack(losses).float().cpu().numpy() if losses else np.array([])
        total_loss = float(np.mean(loss_values)) if losses else 0.0
        print(
            "---task {} => CE: {}, T: {}".format(
                self._cur_task,
                float(loss_values[-1]) if losses else 0.0,
                total_loss,
            )
        )
        return model.state_dict(), total_loss

    def _local_finetune(self, model, train_data_loader, lr):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )

        criterion = TaskAwareTemperatureScalingLoss(
            self._known_classes,
            self.args["tau_old"],
            self.args["tau_new"],
            self.args["w_old"],
            self.args["w_new"],
            self.args.get("tts_mode", "paper_eq"),
            self.args.get("gdr_correction_mode", "none"),
        )

        losses = []
        loss_values = np.array([])
        for local_epoch in range(self.args["local_ep"]):
            if self._active_tracker is not None:
                self._active_tracker.begin_epoch()
            for batch_idx, (
                _,
                images,
                labels,
                is_replay,
                replay_weights,
                task_ids,
            ) in enumerate(train_data_loader):
                images = self._prepare_images(images)
                labels = labels.cuda(non_blocking=True)
                is_replay = is_replay.cuda(non_blocking=True)
                replay_weights = replay_weights.cuda(non_blocking=True)

                with self._autocast():
                    logits = model(images)["logits"]
                    loss = criterion(
                        logits,
                        labels,
                        is_replay,
                        replay_weights,
                    )
                self._record_mass(labels, is_replay, criterion.last_sample_ce)
                if self._active_tracker is not None:
                    self._active_tracker.record(labels, is_replay, criterion.last_sample_ce)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())

            loss_values = torch.stack(losses).float().cpu().numpy() if losses else np.array([])
            print(
                "---task {}, ep {}/{} => TTS: {}, T: {}".format(
                    self._cur_task,
                    local_epoch + 1,
                    self.args["local_ep"],
                    float(loss_values[-1]) if losses else 0.0,
                    float(np.mean(loss_values)) if losses else 0.0,
                )
            )

        total_loss = float(np.mean(loss_values)) if losses else 0.0
        return model.state_dict(), total_loss

    def _record_mass(self, labels, replay, losses):
        if not self.metrics:
            return
        with torch.no_grad():
            for c in labels.unique().tolist():
                for is_replay in (False, True):
                    mask = (labels == c) & (replay.bool() == is_replay)
                    count = int(mask.sum())
                    if count:
                        old_count, old_mass = self._mass.get((c, is_replay), (0, 0.0))
                        self._mass[c, is_replay] = (old_count+count, old_mass+float(losses[mask].detach().sum()))

    def _extract_probe_features(self, model, probe_loader):
        return extract_probe_outputs(model, probe_loader, self._extract_feature_tensor)[2]

    def _probe_accuracy(self, model):
        logits, labels, _ = extract_probe_outputs(model, self.probes.loader())
        return self.metrics.accuracy(logits, labels)

    def _probe_by_class(self, model):
        logits, labels, _ = extract_probe_outputs(model, self.probes.loader())
        return probe_by_class(logits, labels)

    def _pack_exposure(self, client_states):
        if not self._known_classes:
            return None
        empty = ReplayExposureTracker(self._known_classes).state_dict()
        ordered = [client_states.get(k, empty) for k in range(self.args["num_users"])]
        result = {name: torch.stack([state[name] for state in ordered])
                  for name in empty}
        for name in ("co_batch_count", "sequential_exposure", "draws", "ce_mass",
                     "lag_sum", "lag_observations"):
            result["round_" + name] = result[name].sum(0)
        observations = result["round_lag_observations"]
        mean = torch.full_like(result["round_lag_sum"], float("nan"))
        valid = observations > 0
        mean[valid] = result["round_lag_sum"][valid] / observations[valid]
        result["round_mean_lag"] = mean
        minimum = result["min_lag"].clone()
        minimum[minimum < 0] = torch.iinfo(minimum.dtype).max
        result["round_min_lag"] = minimum.min(0).values
        result["round_min_lag"][result["round_min_lag"] ==
                                torch.iinfo(minimum.dtype).max] = -1
        mass_a = result["ce_mass"][:, :, None]
        mass_b = result["ce_mass"][:, None, :]
        result["mass_a_over_b"] = torch.where(
            mass_b > 0, mass_a / mass_b.clamp_min(torch.finfo(mass_b.dtype).eps),
            torch.full_like(mass_a / mass_b.clamp_min(1), float("nan")))
        round_mass = result["round_ce_mass"]
        result["round_mass_a_over_b"] = torch.where(
            round_mass[None, :] > 0,
            round_mass[:, None] / round_mass[None, :].clamp_min(torch.finfo(round_mass.dtype).eps),
            torch.full((self._known_classes, self._known_classes), float("nan"),
                       dtype=round_mass.dtype))
        return result

    def _log_probe(self, round_id, exposure=None):
        logits, labels, features = extract_probe_outputs(self._network, self.probes.loader(), self._extract_feature_tensor)
        self.metrics.probe(self._cur_task, round_id, logits, labels, features,
                           self.task_registry, exposure=exposure)

    def save_transition_start(self):
        directory = self.metrics.directory if self.metrics else (
            Path(self.save_dir or "store") / "fedcbdr_seed_{}".format(self.seed))
        directory.mkdir(parents=True, exist_ok=True)
        state = dict(
            schema_version=1, task=self._cur_task,
            known_classes=self._known_classes, total_classes=self._total_classes,
            model_state_dict={k: v.detach().cpu().clone()
                              for k, v in self._network.state_dict().items()},
            task_class_ranges=list(self.task_class_ranges),
            class_order=list(self.args["class_order"]),
            retained_ds_all=[[ds.snapshot() for ds in client]
                             for client in self.retained_ds_all],
            probes=self.probes, train_dataset=self.train_dataset,
            test_dataset=self.test_loader.dataset,
            old_dataset=self.old_loader.dataset if self._cur_task > 0 else None,
            new_dataset=self.new_loader.dataset if self._cur_task > 0 else None,
            user_groups=copy.deepcopy(self.user_groups),
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.random.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            torch_backend_state=dict(
                deterministic=torch.backends.cudnn.deterministic,
                benchmark=torch.backends.cudnn.benchmark,
                enabled=torch.backends.cudnn.enabled,
                allow_tf32=torch.backends.cudnn.allow_tf32,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            ),
            metrics_previous=copy.deepcopy(self.metrics.previous) if self.metrics else {},
            metrics_boundary=copy.deepcopy(self.metrics.boundary) if self.metrics else {},
            boundary_class_metrics=copy.deepcopy(self.metrics.boundary_class_metrics)
                if self.metrics else {},
            args=copy.deepcopy(self.args),
        )
        path = directory / "transition_start_T{}.pkl".format(self._cur_task)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def save_checkpoint(self, filename):
        replay = [[ds.snapshot() for ds in client] for client in self.retained_ds_all]
        state = dict(schema_version=1, tasks=self._cur_task,
                     model_state_dict={k: v.detach().cpu() for k,v in self._network.state_dict().items()},
                     task_class_ranges=list(self.task_class_ranges), probes=self.probes.state_dict(),
                     retained_ds_all=replay, args=self.args, user_groups=self.user_groups)
        path = Path("{}_{}.pkl".format(filename, self._cur_task))
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def _extract_feature_tensor(self, output):
        # The supplied FedCBDR implementation expects ``features``.  LANDER's
        # verified IncrementalNet usage relies on ``att``.  Prefer the original
        # FedCBDR key, but accept LANDER's verified key for drop-in compatibility.
        if "features" in output:
            features = output["features"]
        elif "att" in output:
            features = output["att"]
        else:
            raise KeyError(
                "IncrementalNet output must contain 'features' or 'att' "
                "for FedCBDR GDR selection"
            )

        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features

    def _extract_client_features(self, client_id):
        local_dataset = DatasetSplit(
            self.train_dataset,
            self.user_groups[client_id],
        )
        local_loader = DataLoader(
            local_dataset,
            batch_size=self.args["local_bs"],
            shuffle=False,
            num_workers=self.args["num_worker"],
            pin_memory=True,
            multiprocessing_context=self.args["mulc"] if self.args["num_worker"] > 0 else None,
            persistent_workers=self.args["num_worker"] > 0,
        )

        self._network.eval()
        features = []

        with torch.no_grad():
            for _, images, _ in local_loader:
                # GDR feature geometry and downstream SVD remain in FP32.
                output = self._network(self._prepare_images(images))
                features.append(
                    self._extract_feature_tensor(output).detach().cpu()
                )

        if features:
            return torch.cat(features, dim=0)

        feature_dim = int(getattr(self._network, "feature_dim", 0))
        return torch.empty((0, feature_dim), dtype=torch.float32)

    def _construct_replay_for_current_task(self):
        selector_seed = int(self.seed) + 10000 * (self._cur_task + 1)
        protocol = self.args.get("gdr_protocol", "paper_global")

        if protocol == "repo_local":
            if self.args.get("gdr_mask_mode", "dense_qr") != "dense_qr":
                raise ValueError(
                    "repo_local GDR requires --gdr-mask-mode dense_qr"
                )
            if self.args.get("gdr_replacement", "with") != "with":
                raise ValueError(
                    "repo_local GDR requires --gdr-replacement with"
                )
            if self.args.get("gdr_correction_mode", "none") != "none":
                raise ValueError(
                    "controlled repo_local GDR requires "
                    "--gdr-correction-mode none"
                )

            selector = RepoStyleReplaySelector(
                int(self.args["gdr_task_budget"]),
                selector_seed,
            )

        elif protocol == "paper_global":
            compute_device = "cuda" if torch.cuda.is_available() else "cpu"
            selector = GlobalPerspectiveReplaySelector(
                int(self.args["gdr_task_budget"]),
                selector_seed,
                compute_device,
                mask_mode=self.args.get("gdr_mask_mode", "dense_qr"),
                leverage_mode=self.args.get(
                    "gdr_leverage_mode", "economy"
                ),
                rank=self.args.get("gdr_rank"),
                normalization_mode=self.args.get(
                    "gdr_normalization_mode", "global"
                ),
                replacement=self.args.get("gdr_replacement", "with"),
                correction_mode=self.args.get(
                    "gdr_correction_mode", "none"
                ),
            )
        else:
            raise ValueError(
                "unknown gdr_protocol: {!r}".format(protocol)
            )

        local_features = [
            self._extract_client_features(client_id)
            for client_id in range(self.args["num_users"])
        ]
        selected = selector.select(local_features)
        diagnostic = selector.last_diagnostics
        if self.metrics:
            self.metrics.write("gdr", self._cur_task, "selected", draw_count=diagnostic.draw_count,
                               unique_id_count=diagnostic.unique_id_count, probability_sum=diagnostic.probability_sum,
                               sampling_matrix_frobenius_norm=diagnostic.sampling_matrix_frobenius_norm,
                               multiplicities=[dict(client_id=k[0], local_index=k[1], count=v)
                                               for k,v in diagnostic.multiplicities.items()])

        print(
            "Task {} GDR draws={} unique={} p_sum={:.6f} "
            "multiplicities={} sampling_matrix_norm={}".format(
                self._cur_task,
                diagnostic.draw_count,
                diagnostic.unique_id_count,
                diagnostic.probability_sum,
                diagnostic.multiplicities,
                diagnostic.sampling_matrix_frobenius_norm,
            )
        )

        for client_id, values in selected.items():
            if not values:
                continue

            if (
                self.args.get("gdr_correction_mode", "none")
                == "replay_loss_experimental"
            ):
                weights = [value.sampling_weight for value in values]
            else:
                weights = [1.0 for _ in values]

            local_dataset = DatasetSplit(
                self.train_dataset,
                self.user_groups[client_id],
            )
            replay = ReplayDataset(
                local_dataset,
                [value.local_index for value in values],
                self.train_dataset.trsf,
                self.train_dataset.use_path,
                weights,
                task_id=self._cur_task,
            )
            self.retained_ds_all[client_id].append(replay)

            print(
                "Task {} client {} local replay histogram={}".format(
                    self._cur_task,
                    client_id,
                    _label_distribution(replay.labels),
                )
            )


Learner = FedCBDR

__all__ = [
    "FedCBDR",
    "GDRDiagnostics",
    "GDRSelection",
    "GlobalPerspectiveReplaySelector",
    "RepoStyleReplaySelector",
    "Learner",
    "ReplayDataset",
    "TaggedDataset",
    "TaskAwareTemperatureScalingLoss",
    "uniform_average_state_dicts",
]
