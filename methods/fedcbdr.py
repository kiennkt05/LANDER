import copy
import logging
import math
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


# -----------------------------------------------------------------------------
# Data / transform helpers
# -----------------------------------------------------------------------------


def get_norm_and_transform(dataset):
    """FedCBDR preprocessing retained from the supplied FedCBDR implementation."""
    if dataset == "cifar100":
        data_normalize = dict(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(**data_normalize),
            ]
        )
    elif dataset == "tiny_imagenet":
        data_normalize = dict(
            mean=(0.4802, 0.4481, 0.3975),
            std=(0.2302, 0.2265, 0.2262),
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**data_normalize),
            ]
        )
    else:
        raise ValueError("FedCBDR has no transform for dataset {!r}".format(dataset))

    normalizer = Normalizer(**data_normalize)
    return train_transform, normalizer


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    return (tensor - _mean[None, :, None, None]) / _std[None, :, None, None]


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


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

    def __init__(self, dataset, is_replay=False):
        self.dataset = dataset
        self.is_replay = bool(is_replay)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample_id, image, label = self.dataset[index]
        if self.is_replay:
            weight = float(self.dataset.sampling_weights[index])
        else:
            weight = 1.0
        return sample_id, image, int(label), self.is_replay, weight


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
    ):
        indices = np.asarray(local_indices, dtype=np.int64)
        self.source_local_indices = indices
        self.images = dataset.images[indices]
        self.labels = np.asarray(dataset.labels[indices], dtype=np.int64)
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

        self.transform, self.normalizer = get_norm_and_transform(
            self.args["dataset"]
        )
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
        self._network.cuda()
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
            multiprocessing_context=self.args["mulc"],
            persistent_workers=True,
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
                multiprocessing_context=self.args["mulc"],
                persistent_workers=True,
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
                multiprocessing_context=self.args["mulc"],
                persistent_workers=True,
            )

        setup_seed(self.seed)
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
        )
        replay = [
            TaggedDataset(dataset, is_replay=True)
            for dataset in self.retained_ds_all[client_id]
        ]

        if not replay:
            return current
        return ConcatDataset([current] + replay)

    def _make_client_loader(self, client_id):
        return DataLoader(
            self._client_dataset(client_id),
            batch_size=self.args["local_bs"],
            shuffle=True,
            num_workers=self.args["num_worker"],
            pin_memory=True,
            multiprocessing_context=self.args["mulc"],
            persistent_workers=True,
        )

    def _fl_train(self, train_dataset, test_loader):
        self._network.cuda()

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

        for _, com in enumerate(prog_bar):
            local_weights = []
            loss_weight = []

            idxs_users = self._selected_clients()
            lr = self._learning_rate(com)

            for idx in idxs_users:
                local_train_loader = self._make_client_loader(int(idx))

                if self._cur_task == 0:
                    w, total_loss = self._local_update(
                        copy.deepcopy(self._network),
                        local_train_loader,
                        lr,
                    )
                else:
                    w, total_loss = self._local_finetune(
                        copy.deepcopy(self._network),
                        local_train_loader,
                        lr,
                    )

                local_weights.append(copy.deepcopy(w))
                loss_weight.append(total_loss)

                del local_train_loader, w
                torch.cuda.empty_cache()

            global_weights = uniform_average_state_dicts(local_weights)
            self._network.load_state_dict(global_weights)

            if com % 1 == 0 and com < self.args["com_round"]:
                test_acc = self._compute_fedcbdr_accuracy(
                    self._network,
                    test_loader,
                )

                if self._cur_task > 0:
                    test_old_acc = self._compute_fedcbdr_accuracy(
                        copy.deepcopy(self._network),
                        self.old_loader,
                    )
                    test_new_acc = self._compute_fedcbdr_accuracy(
                        copy.deepcopy(self._network),
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

        # FedCBDR's next task needs this task's selected replay buffer.
        self._construct_replay_for_current_task()

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
            ) in enumerate(train_data_loader):
                images = images.cuda()
                labels = labels.cuda()

                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

        total_loss = float(np.mean(losses)) if losses else 0.0
        print(
            "---task {} => CE: {}, T: {}".format(
                self._cur_task,
                losses[-1] if losses else 0.0,
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
        for local_epoch in range(self.args["local_ep"]):
            for batch_idx, (
                _,
                images,
                labels,
                is_replay,
                replay_weights,
            ) in enumerate(train_data_loader):
                images = images.cuda()
                labels = labels.cuda()
                is_replay = is_replay.cuda()
                replay_weights = replay_weights.cuda()

                logits = model(images)["logits"]
                loss = criterion(
                    logits,
                    labels,
                    is_replay,
                    replay_weights,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            print(
                "---task {}, ep {}/{} => TTS: {}, T: {}".format(
                    self._cur_task,
                    local_epoch + 1,
                    self.args["local_ep"],
                    losses[-1] if losses else 0.0,
                    float(np.mean(losses)) if losses else 0.0,
                )
            )

        total_loss = float(np.mean(losses)) if losses else 0.0
        return model.state_dict(), total_loss

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
            multiprocessing_context=self.args["mulc"],
            persistent_workers=True,
        )

        self._network.eval()
        features = []

        with torch.no_grad():
            for _, images, _ in local_loader:
                output = self._network(images.cuda())
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
                self.transform,
                self.args["dataset"] == "tiny_imagenet",
                weights,
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
