import copy
import json
import logging
from numbers import Integral
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from methods.base import BaseLearner
from methods.fedcbdr import TaskAwareTemperatureScalingLoss
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed, pil_loader

try:
    import wandb
except ImportError:
    wandb = None


class _ImplicitOrthogonalMask:
    """Composes multiple layers of random pairwise Givens rotations.

    Given dimension `dim` and `num_layers` (default 12):
    For each layer l:
        Generates a deterministic random permutation of dim indices.
        Pairs up (p_{2j}, p_{2j+1}).
        Generates random angles theta_j in [0, 2*pi).
        Applies Givens rotation:
            [x[p_{2j}]']   = [cos(theta_j)  -sin(theta_j)] [x[p_{2j}]]
            [x[p_{2j+1}]']   [sin(theta_j)   cos(theta_j)] [x[p_{2j+1}]]

    This operator is strictly orthogonal (preserves Euclidean norms and inner products)
    without materializing an explicit dense dim x dim matrix.
    """
    def __init__(self, dim, num_layers=12, seed=42):
        self.dim = dim
        self.num_layers = num_layers
        self.layers = []

        if dim > 1:
            rng = np.random.RandomState(seed)
            num_pairs = dim // 2
            for l in range(num_layers):
                perm = rng.permutation(dim)
                idx1 = torch.tensor(perm[0::2][:num_pairs], dtype=torch.long)
                idx2 = torch.tensor(perm[1::2][:num_pairs], dtype=torch.long)
                angles = rng.uniform(0.0, 2.0 * np.pi, size=num_pairs).astype(np.float32)
                cos = torch.tensor(np.cos(angles), dtype=torch.float32)
                sin = torch.tensor(np.sin(angles), dtype=torch.float32)
                self.layers.append((idx1, idx2, cos, sin))

    def apply_right(self, X):
        """Applies X @ Q where X has shape (M, dim). Modifies a clone of X."""
        if self.dim <= 1 or not self.layers:
            return X.clone()
        device = X.device
        X_out = X.clone()
        for idx1, idx2, cos, sin in self.layers:
            i1 = idx1.to(device)
            i2 = idx2.to(device)
            c = cos.to(device)
            s = sin.to(device)
            u = X_out[:, i1]
            w = X_out[:, i2]
            X_out[:, i1] = u * c - w * s
            X_out[:, i2] = u * s + w * c
        return X_out

    def apply_left(self, X):
        """Applies P @ X where X has shape (dim, M). Modifies a clone of X."""
        if self.dim <= 1 or not self.layers:
            return X.clone()
        device = X.device
        X_out = X.clone()
        for idx1, idx2, cos, sin in self.layers:
            i1 = idx1.to(device)
            i2 = idx2.to(device)
            c = cos.to(device).unsqueeze(1)
            s = sin.to(device).unsqueeze(1)
            u = X_out[i1, :]
            w = X_out[i2, :]
            X_out[i1, :] = u * c - w * s
            X_out[i2, :] = u * s + w * c
        return X_out


class ReplayDataset(Dataset):
    """Stores retained replay samples for a client."""
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Images and labels length mismatch!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.logits = None  # Detached CPU targets from a best task model.

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label


class ClientTaskDataset(Dataset):
    """Combines a client's current-task data with all previously retained replay datasets.

    Current items appear once; replay items have `repeat_rate` virtual copies.
    Images and cached logits remain stored only in the original replay datasets.

    Returns:
        (idx, image, label, is_current, candidate_id)
        where `is_current` is True only for current-task samples, and `candidate_id` is the
        row index in the current task trajectory matrix V (or -1 for replay).
    """
    def __init__(self, current_dataset, candidate_ids, replay_datasets=None, repeat_rate=1):
        if isinstance(repeat_rate, bool) or not isinstance(repeat_rate, Integral) or repeat_rate < 1:
            raise ValueError("repeat_rate must be a positive integer")
        self.repeat_rate = int(repeat_rate)
        self.current_dataset = current_dataset
        self.candidate_ids = candidate_ids
        self.num_current = len(current_dataset)

        self.replay_datasets = replay_datasets or []
        self.replay_offsets = []
        cur_offset = self.num_current
        for ds in self.replay_datasets:
            self.replay_offsets.append(cur_offset)
            cur_offset += self.repeat_rate * len(ds)
        self.total_len = cur_offset

    def __len__(self):
        return self.total_len

    def _replay_item(self, idx):
        """Map a virtual replay index to its single stored item."""
        for ds, offset in zip(self.replay_datasets, self.replay_offsets):
            if offset <= idx < offset + self.repeat_rate * len(ds):
                return ds, (idx - offset) % len(ds)
        raise IndexError(f"Invalid replay index: {idx}")

    def replay_logits(self, indices):
        """Return padded targets and their class counts in replay batch order."""
        targets = []
        for idx in indices.tolist():
            if idx < self.num_current:
                continue
            ds, item_idx = self._replay_item(idx)
            if ds.logits is None:
                raise RuntimeError("Replay distillation targets have not been captured.")
            targets.append(ds.logits[item_idx])
        widths = torch.tensor([target.numel() for target in targets], dtype=torch.long)
        return torch.nn.utils.rnn.pad_sequence(targets, batch_first=True), widths

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range for ClientTaskDataset of length {self.total_len}")
        if idx < self.num_current:
            _, image, label = self.current_dataset[idx]
            cand_id = self.candidate_ids[idx]
            return idx, image, label, True, cand_id
        else:
            ds, item_idx = self._replay_item(idx)
            _, image, label = ds[item_idx]
            return idx, image, label, False, -1


class Exp5Base(BaseLearner):
    """Base implementation for Exp5 Trajectory-Subspace Replay Selection.

    Implements:
      Stage A: Federated training setup & exact candidate indexing.
      Stage B: Exact per-sample trajectory accumulation with SGD momentum & weight decay.
      Stage C: Attribution invariant verification gate (round-level and task-level).
      Stage D: Greedy projected selection, evaluation, and replay memory updates.
    """
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.tau = args.get("exp5a_tau", 0.05)
        self.gdr_task_budget = args.get("gdr_task_budget", 400)
        self.exp5a_rank = args.get("exp5a_rank", 64)
        self.exp5a_svd_oversampling = args.get("exp5a_svd_oversampling", 16)
        self.target_mode = args.get("exp5a_target_mode", "budget_scaled_sum")
        self.repeat_rate = args.get("repeat_rate", 1)
        if isinstance(self.repeat_rate, bool) or not isinstance(self.repeat_rate, Integral) or self.repeat_rate < 1:
            raise ValueError("repeat_rate must be a positive integer")
        self.exp5_distill_loss = args.get("exp5_distill_loss", False)
        self.exp5_single_distill_loss = args.get("exp5_single_distill_loss", False)
        self.repo_dual = args.get("repo_dual", False)
        if sum(bool(mode) for mode in (self.exp5_distill_loss, self.exp5_single_distill_loss, self.repo_dual)) > 1:
            raise ValueError("Choose only one Exp5 loss variant.")
        self.sample_weighted_fedavg = args.get("sample_weighted_fedavg", False)
        self._repo_dual_loss = None
        if self.repo_dual:
            self._repo_dual_loss = TaskAwareTemperatureScalingLoss(
                self._known_classes,
                args.get("tau_old", 0.9),
                args.get("tau_new", 1.1),
                args.get("w_old", 1.1),
                args.get("w_new", 0.9),
                mode="repo_dual",
            )

        # Exp5 greedy-selection diagnostics. Instrumentation only: these values do not
        # participate in candidate scoring or alter the selected subset.
        diag_name = f"exp5_greedy_diag_{args.get('exp_name', args.get('method', 'exp5'))}_seed{args.get('seed', 0)}.jsonl"
        self.exp5_diag_path = os.path.abspath(diag_name)
        self._exp5_diag_initialized = False

        self.num_users = args.get("num_users", 5)
        # Client-retained replay datasets across tasks: dict {client_id: [ReplayDataset_t0, ...]}
        self.retained_ds_all = {k: [] for k in range(self.num_users)}
        self.trajectory_matrix = None

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        print(f"--- Task {self._cur_task}: Learning on classes {self._known_classes} - {self._total_classes} ---")

        # 1. Fetch current task dataset
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.test_loader = self._test_data_loader(test_dataset)

        setup_seed(self.seed, fast_cuda=(self.args.get("fast_cuda", False) or
                                        self.args.get("t4_parralel", False)))
        partition_res = partition_data(
            train_dataset.labels, beta=self.args["beta"], n_parties=self.num_users
        )
        if isinstance(partition_res, tuple):
            user_groups = partition_res[0]
        else:
            user_groups = partition_res

        # 2. Dimensions and Candidate Table
        d = self.feature_dim
        C_t = self._total_classes
        D = C_t * d + C_t
        print(f"Task {self._cur_task}: feature_dim d={d}, classes C_t={C_t}, trajectory dim D={D}")

        cand_rows = []
        offsets = []
        cur_offset = 0
        for k in range(self.num_users):
            offsets.append(cur_offset)
            Nk = len(user_groups[k])
            for l in range(Nk):
                global_id = cur_offset + l
                dataset_idx = user_groups[k][l]
                raw_label = train_dataset.labels[dataset_idx]
                if isinstance(raw_label, (np.ndarray, list)):
                    arr = np.array(raw_label)
                    label = int(arr.item()) if arr.size == 1 else int(arr[0])
                else:
                    label = int(raw_label)
                cand_rows.append({
                    "client_id": k,
                    "local_id": l,
                    "global_id": global_id,
                    "dataset_idx": dataset_idx,
                    "label": label,
                })
            cur_offset += Nk
        N = cur_offset
        print(f"Task {self._cur_task}: Total current-task candidates N={N} across {self.num_users} clients.")

        # Allocate trajectory matrix V in CPU memory
        self.trajectory_matrix = torch.zeros((N, D), dtype=torch.float32)

        # 3. Federated Training with exact trajectory accumulation
        self._prepare_model(self._network)
        self._fl_train(train_dataset, user_groups, offsets, D)

        # 4. Trajectory projection into rank-r subspace (implemented by subclasses)
        print(f"Task {self._cur_task}: Computing rank-{self.exp5a_rank} projected trajectory embeddings...")
        Z_dict, R_dict, energies = self._compute_projected_trajectories(user_groups, offsets, cand_rows)

        # 5. Greedy Replay Selection
        M = self.gdr_task_budget
        print(f"Task {self._cur_task}: Performing replay selection for budget M={M} (target_mode={self.target_mode})...")
        selected_by_client = self._greedy_selection(Z_dict, user_groups, M, offsets, cand_rows)

        # 6. Evaluate Replay Selection
        self._evaluate_selection(selected_by_client, Z_dict, user_groups, offsets, D, cand_rows)

        # 7. Convert selected candidate rows to ReplayDataset & update retained memory
        self._commit_replay_memory(selected_by_client, user_groups, train_dataset)

        # 8. Release trajectory matrix
        del self.trajectory_matrix
        self.trajectory_matrix = None
        print(f"Task {self._cur_task}: Replay selection completed. Trajectory matrix released.")

    def _fl_train(self, train_dataset, user_groups, offsets, D):
        """Runs federated training rounds, accumulating exact sample trajectories."""
        com_round = self.args["com_round"]
        prog_bar = tqdm(range(com_round), desc=f"Task {self._cur_task} FL Training")

        local_lr = self.args.get("local_lr", 0.01)
        momentum = 0.9
        weight_decay = self.args.get("weight_decay", 5e-4)
        local_ep = self.args["local_ep"]
        local_bs = self.args["local_bs"]
        frac = self.args["frac"]

        round_errors = []
        task_pred_delta_theta = torch.zeros(D, dtype=torch.float32)
        task_actual_delta_theta = torch.zeros(D, dtype=torch.float32)

        optimizer_head = torch.optim.SGD(self._network.parameters(), lr=local_lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = self._init_scheduler(optimizer_head)

        for com in prog_bar:
            current_lr = scheduler.get_last_lr()[0]
            # Snapshot head parameters before local training
            head_before = torch.cat([
                self._network.fc.weight.data.flatten(),
                self._network.fc.bias.data.flatten()
            ]).detach().cpu()

            # Client selection
            idxs_users = range(self.args["num_users"])
            if self.sample_weighted_fedavg:
                sample_counts = [len(user_groups[k]) + self.repeat_rate * sum(map(len, self.retained_ds_all[k])) for k in idxs_users]
                client_weights = [n / sum(sample_counts) for n in sample_counts]
            else:
                p_k = 1.0 / self.args["num_users"]

            round_delta_cur = torch.zeros(D, dtype=torch.float32)
            round_delta_rep = torch.zeros(D, dtype=torch.float32)
            round_delta_wd = torch.zeros(D, dtype=torch.float32)

            local_weights = []

            for k in idxs_users:
                if self.sample_weighted_fedavg:
                    p_k = client_weights[k]
                local_model = copy.deepcopy(self._network)
                local_model.train()
                self._prepare_model(local_model)
                forward_model = self._training_model(local_model, ("logits", "features"))

                Nk = len(user_groups[k])
                cand_ids = np.arange(offsets[k], offsets[k] + Nk)
                cur_ds = DatasetSplit(train_dataset, user_groups[k])
                client_ds = ClientTaskDataset(
                    cur_ds, cand_ids, self.retained_ds_all[k], repeat_rate=self.repeat_rate
                )

                local_loader = DataLoader(
                    client_ds, batch_size=local_bs, shuffle=True,
                    num_workers=self.args["num_worker"], pin_memory=True
                )

                optimizer = torch.optim.SGD(
                    local_model.parameters(),
                    lr=current_lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    dampening=0.0,
                    nesterov=False,
                )

                num_steps_per_epoch = len(local_loader)
                S = local_ep * num_steps_per_epoch
                step = 0

                for ep in range(local_ep):
                    for batch_idx, (b_idxs, images, labels, is_cur, cands) in enumerate(local_loader):
                        step += 1
                        # Exact momentum multiplier: W_s = (1 - mu^(S - s + 1)) / (1 - mu)
                        W_s = (1.0 - (momentum ** (S - step + 1))) / (1.0 - momentum)

                        # Keep exact trajectory derivatives in FP32 for the invariant gate.
                        images = self._prepare_images(images)
                        labels = labels.cuda(non_blocking=True)

                        # Capture head parameters before update step for weight decay attribution
                        theta_head = torch.cat([
                            local_model.fc.weight.data.flatten(),
                            local_model.fc.bias.data.flatten()
                        ]).detach().cpu()

                        outputs = forward_model(images)
                        features = outputs["features"]
                        logits = outputs["logits"]

                        loss = self._training_loss(logits, labels, is_cur, b_idxs, client_ds)

                        # Exact derivative of optimized scalar loss with respect to logits
                        q = torch.autograd.grad(loss, logits, retain_graph=True)[0]

                        # Current-task sample gradients & trajectory contribution
                        is_cur_bool = is_cur.bool()
                        if is_cur_bool.any():
                            cur_q = q[is_cur_bool]
                            cur_h = features[is_cur_bool]
                            cur_cands = cands[is_cur_bool].numpy()

                            # Batch outer product vec(q_i h_i^T)
                            cur_grad_W = torch.bmm(cur_q.unsqueeze(2), cur_h.unsqueeze(1)).flatten(start_dim=1)
                            cur_g = torch.cat([cur_grad_W, cur_q], dim=1)

                            # delta v_i,s = -eta_a * p_k * W_s * g_i,s
                            step_coeff = -current_lr * p_k * W_s
                            delta_v = step_coeff * cur_g
                            self.trajectory_matrix[cur_cands] += delta_v.detach().cpu()

                            g_cur_sum = cur_g.sum(dim=0).detach().cpu()
                        else:
                            g_cur_sum = torch.zeros(D, dtype=torch.float32)

                        # Replay sample gradients (participate in training & invariant check)
                        rep_bool = ~is_cur_bool
                        if rep_bool.any():
                            rep_q = q[rep_bool]
                            rep_h = features[rep_bool]
                            rep_grad_W = torch.bmm(rep_q.unsqueeze(2), rep_h.unsqueeze(1)).flatten(start_dim=1)
                            rep_g = torch.cat([rep_grad_W, rep_q], dim=1)
                            g_rep_sum = rep_g.sum(dim=0).detach().cpu()
                        else:
                            g_rep_sum = torch.zeros(D, dtype=torch.float32)

                        step_coeff = -current_lr * p_k * W_s
                        round_delta_cur += step_coeff * g_cur_sum
                        round_delta_rep += step_coeff * g_rep_sum
                        round_delta_wd += (step_coeff * weight_decay) * theta_head

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                local_weights.append(copy.deepcopy(local_model.state_dict()))
                del local_loader, forward_model, local_model

            # FedAvg aggregation
            if self.sample_weighted_fedavg:
                global_weights = copy.deepcopy(local_weights[0])
                for key in global_weights:
                    if global_weights[key].is_floating_point():
                        global_weights[key] = sum((local_weights[i][key] * client_weights[i] for i in range(len(local_weights))), torch.zeros_like(global_weights[key]))
                    else:
                        global_weights[key] = sum((local_weights[i][key].float() * client_weights[i] for i in range(len(local_weights))), torch.zeros_like(global_weights[key], dtype=torch.float32)).round().to(global_weights[key].dtype)
            else:
                global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            scheduler.step()

            # Measure actual FedAvg classifier head delta
            head_after = torch.cat([
                self._network.fc.weight.data.flatten(),
                self._network.fc.bias.data.flatten()
            ]).detach().cpu()

            actual_delta_a = head_after - head_before
            pred_delta_a = round_delta_cur + round_delta_rep + round_delta_wd

            # Normalized round invariant error: E_a = ||pred - actual|| / (||actual|| + eps)
            err_a = (torch.norm(pred_delta_a - actual_delta_a) / (torch.norm(actual_delta_a) + 1e-8)).item()
            round_errors.append(err_a)

            task_pred_delta_theta += pred_delta_a
            task_actual_delta_theta += actual_delta_a

            if self._should_evaluate(com):
                test_acc = self._compute_accuracy(self._network, self.test_loader)
                info = f"Task {self._cur_task}, Round {com + 1}/{com_round} => Test_accy {test_acc:.2f} | E_a: {err_a:.2e}"
                prog_bar.set_description(info)
                if self.wandb == 1 and wandb is not None:
                    wandb.log({
                        f"Task_{self._cur_task}/accuracy": test_acc,
                        f"Task_{self._cur_task}/E_a": err_a,
                    })

        # Check Attribution Invariant Validation Gate
        task_err = (torch.norm(task_pred_delta_theta - task_actual_delta_theta) /
                    (torch.norm(task_actual_delta_theta) + 1e-8)).item()
        max_round_err = max(round_errors) if round_errors else 0.0

        print(f"\n[Attribution Invariant Gate] Task {self._cur_task}: E_task = {task_err:.2e}, max E_a = {max_round_err:.2e} (tau = {self.tau})")
        if task_err > self.tau or max_round_err > self.tau:
            raise RuntimeError(
                f"Trajectory attribution invariant gate FAILED! "
                f"E_task={task_err:.4e} > tau={self.tau} or max_E_a={max_round_err:.4e} > tau={self.tau}. "
                "Coreset selection is aborted as mandated by Exp5 specification."
            )
        print("[Attribution Invariant Gate] PASSED successfully.")

    def _training_loss(self, logits, labels, is_cur, batch_indices, client_ds):
        """Select CE, FedCBDR repo_dual TTS, or all-sample CE plus replay logit MSE."""
        if self.repo_dual:
            self._repo_dual_loss.num_old_classes = self._known_classes
            return self._repo_dual_loss(logits, labels)
        if not (self.exp5_distill_loss or self.exp5_single_distill_loss):
            return F.cross_entropy(logits, labels)

        current = is_cur.to(device=logits.device, dtype=torch.bool)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        if (~current).any():
            targets, widths = client_ds.replay_logits(batch_indices)
            targets = targets.to(logits)
            widths = widths.to(logits.device)
            # Each item matches only classes seen by its own teacher snapshot.
            predicted = logits[~current, :targets.shape[1]]
            valid = torch.arange(targets.shape[1], device=logits.device)[None, :] < widths[:, None]
            errors = F.mse_loss(predicted, targets, reduction="none").masked_fill(~valid, 0.0)
            loss = loss + (errors.sum(dim=1) / widths).sum()
        return loss / logits.shape[0]

    @torch.no_grad()
    def _refresh_replay_logits(self, datasets=None):
        """Capture supplied datasets (or all replay) using the trained model."""
        if datasets is None:
            datasets = [ds for client_datasets in self.retained_ds_all.values() for ds in client_datasets]
        model = self._network
        was_training = model.training
        device = next(model.parameters()).device
        model.eval()
        try:
            for ds in datasets:
                ds.logits = torch.empty((len(ds), self._total_classes), dtype=torch.float32)
                loader = DataLoader(ds, batch_size=self.args["local_bs"], shuffle=False,
                                    num_workers=self.args.get("num_worker", 4))
                for indices, images, _ in loader:
                    ds.logits[indices] = model(self._prepare_images(images))["logits"].detach().float().cpu()
        finally:
            model.train(was_training)

    def _compute_projected_trajectories(self, user_groups, offsets, cand_rows):
        raise NotImplementedError("Subclasses must implement _compute_projected_trajectories.")

    def _write_exp5_diag_jsonl(self, records):
        """Append machine-readable greedy diagnostics without affecting selection."""
        if not records:
            return
        mode = "w" if not self._exp5_diag_initialized else "a"
        try:
            with open(self.exp5_diag_path, mode, encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")
            if not self._exp5_diag_initialized:
                print(f"[EXP5_GREEDY_DIAG_FILE] {self.exp5_diag_path}")
            self._exp5_diag_initialized = True
        except Exception as exc:
            # Diagnostics must never abort or alter the experiment itself.
            print(f"[EXP5_GREEDY_DIAG_WARNING] Could not write JSONL diagnostics: {exc}")

    @staticmethod
    def _safe_percentile_leq(pool, value):
        """Empirical percentile of value in a 1-D tensor: fraction(pool <= value)."""
        if pool.numel() == 0:
            return 0.0
        return torch.mean((pool <= value).float()).item()

    def _greedy_selection(self, Z_dict, user_groups, M, offsets, cand_rows):
        """
        Per-client FedCBDR-style greedy selection on projected trajectory vectors,
        with exhaustive instrumentation of the residual-matching dynamics.

        IMPORTANT: the diagnostics below DO NOT change the original selection rule.
        At a residual r, the chosen item minimizes ||r-z_j||^2, equivalently maximizes

            Delta_j = ||r||^2 - ||r-z_j||^2
                    = 2 z_j^T r - ||z_j||^2.

        We log both:
          (1) marginal residual gain Delta_j -- what the greedy rule truly optimizes, and
          (2) raw trajectory magnitude ||v_j|| -- how large that sample's accumulated
              model-update contribution is independent of the current residual.
        """
        K = self.num_users
        B = M // K
        R = M - K * B
        eps = 1e-8

        selected_by_client = {k: [] for k in range(K)}
        residual_by_client = {}
        target_by_client = {}
        v_target_by_client = {}
        v_residual_by_client = {}
        diag_by_client = {k: [] for k in range(K)}
        jsonl_records = []

        # 1. Construct reconstruction target and initial residual exactly as before.
        for k in range(K):
            Z_k = Z_dict[k]
            Nk = Z_k.shape[0]
            if self.target_mode == "full_sum":
                T_k = torch.sum(Z_k, dim=0)
            else:  # budget_scaled_sum
                T_k = (float(B) / float(Nk)) * torch.sum(Z_k, dim=0)
            target_by_client[k] = T_k
            residual_by_client[k] = T_k.clone()

        available_by_client = {k: np.ones(Z_dict[k].shape[0], dtype=bool) for k in range(K)}
        norms_sq_by_client = {k: torch.sum(Z_dict[k] ** 2, dim=1) for k in range(K)}

        # Original-space trajectory targets/norms diagnose raw model-update contribution
        # and whether projected-space greedy choices also improve the true D-space target.
        v_norms_by_client = {}
        v_blocks_by_client = {}
        for k in range(K):
            Nk = len(user_groups[k])
            start_idx = offsets[k]
            V_k = self.trajectory_matrix[start_idx:start_idx + Nk]
            v_blocks_by_client[k] = V_k
            v_norms_by_client[k] = torch.linalg.vector_norm(V_k, dim=1)
            if self.target_mode == "full_sum":
                T_v = torch.sum(V_k, dim=0)
            else:
                T_v = (float(B) / float(Nk)) * torch.sum(V_k, dim=0)
            v_target_by_client[k] = T_v
            v_residual_by_client[k] = T_v.clone()

        def record_and_apply(k, best_j, phase):
            """Record diagnostics for a selected item, then perform the original r <- r-z update."""
            r_before = residual_by_client[k]
            T_k = target_by_client[k]
            z_j = Z_dict[k][best_j]

            target_sq = torch.sum(T_k ** 2).item()
            denom = target_sq + eps
            r_before_sq = torch.sum(r_before ** 2).item()
            z_sq = torch.sum(z_j ** 2).item()
            dot = torch.dot(z_j, r_before).item()
            r_after_vec = r_before - z_j
            r_after_sq = torch.sum(r_after_vec ** 2).item()

            delta_raw = r_before_sq - r_after_sq
            delta_norm = delta_raw / denom
            delta_rel_resid = delta_raw / (r_before_sq + eps)
            resid_before = np.sqrt(max(r_before_sq, 0.0) / denom)
            resid_after = np.sqrt(max(r_after_sq, 0.0) / denom)
            resid_sq_before = r_before_sq / denom
            resid_sq_after = r_after_sq / denom
            explained_sq_after = 1.0 - resid_sq_after

            z_norm = np.sqrt(max(z_sq, 0.0))
            target_norm = np.sqrt(max(target_sq, 0.0))
            r_norm = np.sqrt(max(r_before_sq, 0.0))
            cosine = dot / (z_norm * r_norm + eps)

            # Apply the SAME selected item to an independently tracked original-D-space
            # residual. This does not influence selection; it reveals projection mismatch.
            v_r_before = v_residual_by_client[k]
            v_T = v_target_by_client[k]
            v_j_vec = v_blocks_by_client[k][best_j]
            v_target_sq = torch.sum(v_T ** 2).item()
            v_denom = v_target_sq + eps
            v_r_before_sq = torch.sum(v_r_before ** 2).item()
            v_r_after_vec = v_r_before - v_j_vec
            v_r_after_sq = torch.sum(v_r_after_vec ** 2).item()
            v_delta_raw = v_r_before_sq - v_r_after_sq
            v_delta_norm = v_delta_raw / v_denom
            v_delta_rel_resid = v_delta_raw / (v_r_before_sq + eps)
            v_resid_before = np.sqrt(max(v_r_before_sq, 0.0) / v_denom)
            v_resid_after = np.sqrt(max(v_r_after_sq, 0.0) / v_denom)
            v_explained_sq_after = 1.0 - (v_r_after_sq / v_denom)

            z_norm_pool = torch.sqrt(torch.clamp(norms_sq_by_client[k], min=0.0))
            v_norm_pool = v_norms_by_client[k]
            z_norm_t = torch.tensor(z_norm, dtype=z_norm_pool.dtype)
            v_norm = v_norm_pool[best_j].item()
            v_norm_t = torch.tensor(v_norm, dtype=v_norm_pool.dtype)
            z_pct = self._safe_percentile_leq(z_norm_pool, z_norm_t)
            v_pct = self._safe_percentile_leq(v_norm_pool, v_norm_t)
            z_median = torch.median(z_norm_pool).item() if z_norm_pool.numel() else 0.0
            v_median = torch.median(v_norm_pool).item() if v_norm_pool.numel() else 0.0
            z_max = torch.max(z_norm_pool).item() if z_norm_pool.numel() else 0.0
            v_max = torch.max(v_norm_pool).item() if v_norm_pool.numel() else 0.0

            local_rank = len(selected_by_client[k]) + 1
            global_row = offsets[k] + best_j
            label = int(cand_rows[global_row]["label"])

            rec = {
                "type": "step",
                "task": int(self._cur_task),
                "client": int(k),
                "phase": phase,
                "step": int(local_rank),
                "budget_client_base": int(B),
                "local_candidate": int(best_j),
                "global_candidate": int(global_row),
                "label": label,
                "target_norm": float(target_norm),
                "resid_before": float(resid_before),
                "resid_after": float(resid_after),
                "resid_sq_before": float(resid_sq_before),
                "resid_sq_after": float(resid_sq_after),
                "explained_sq_after": float(explained_sq_after),
                "delta_raw": float(delta_raw),
                "delta_norm": float(delta_norm),
                "delta_rel_resid": float(delta_rel_resid),
                "v_resid_before": float(v_resid_before),
                "v_resid_after": float(v_resid_after),
                "v_explained_sq_after": float(v_explained_sq_after),
                "v_delta_raw": float(v_delta_raw),
                "v_delta_norm": float(v_delta_norm),
                "v_delta_rel_resid": float(v_delta_rel_resid),
                "cosine_to_residual": float(cosine),
                "dot_to_residual_norm": float(dot / denom),
                "z_norm": float(z_norm),
                "z_norm_over_target": float(z_norm / (target_norm + eps)),
                "z_norm_percentile": float(z_pct),
                "z_norm_over_median": float(z_norm / (z_median + eps)),
                "z_norm_over_max": float(z_norm / (z_max + eps)),
                "v_norm": float(v_norm),
                "v_norm_percentile": float(v_pct),
                "v_norm_over_median": float(v_norm / (v_median + eps)),
                "v_norm_over_max": float(v_norm / (v_max + eps)),
                "negative_gain": bool(delta_norm < -1e-10),
                "nonpositive_gain": bool(delta_norm <= 0.0),
                "tiny_gain_1e4": bool(0.0 < delta_norm < 1e-4),
                "tiny_gain_1e5": bool(0.0 < delta_norm < 1e-5),
                "v_negative_gain": bool(v_delta_norm < -1e-10),
                "v_nonpositive_gain": bool(v_delta_norm <= 0.0),
            }
            diag_by_client[k].append(rec)
            jsonl_records.append(rec)


            # Original algorithmic state transition.
            selected_by_client[k].append(best_j)
            available_by_client[k][best_j] = False
            residual_by_client[k] = r_after_vec
            v_residual_by_client[k] = v_r_after_vec

        # 2. Hard base quota: unchanged selection behavior.
        for step in range(B):
            for k in range(K):
                r_k = residual_by_client[k]
                T_k = target_by_client[k]
                denom = torch.sum(T_k ** 2).item() + eps

                avail_indices = np.where(available_by_client[k])[0]
                if len(avail_indices) == 0:
                    continue

                Z_avail = Z_dict[k][avail_indices]
                norms_avail = norms_sq_by_client[k][avail_indices]
                r_norm_sq = torch.sum(r_k ** 2)
                dots = torch.mv(Z_avail, r_k)
                costs = (r_norm_sq + norms_avail - 2.0 * dots) / denom

                best_idx_in_avail = torch.argmin(costs).item()
                best_j = avail_indices[best_idx_in_avail]
                record_and_apply(k, best_j, "base")

        # 3. Remainder slots: unchanged global cheapest-candidate behavior.
        for rem in range(R):
            best_cost = float("inf")
            best_client = None
            best_j = None

            for k in range(K):
                r_k = residual_by_client[k]
                T_k = target_by_client[k]
                denom = torch.sum(T_k ** 2).item() + eps

                avail_indices = np.where(available_by_client[k])[0]
                if len(avail_indices) == 0:
                    continue

                Z_avail = Z_dict[k][avail_indices]
                norms_avail = norms_sq_by_client[k][avail_indices]
                r_norm_sq = torch.sum(r_k ** 2)
                dots = torch.mv(Z_avail, r_k)
                costs = (r_norm_sq + norms_avail - 2.0 * dots) / denom

                min_c, min_pos = torch.min(costs, dim=0)
                if min_c.item() < best_cost:
                    best_cost = min_c.item()
                    best_client = k
                    best_j = avail_indices[min_pos.item()]

            if best_client is not None and best_j is not None:
                record_and_apply(best_client, best_j, "remainder")

        total_selected = sum(len(v) for v in selected_by_client.values())
        assert total_selected == M, f"Expected {M} selected samples, but got {total_selected}!"
        for k in range(K):
            assert len(selected_by_client[k]) == len(set(selected_by_client[k])), f"Client {k} has duplicate selections!"

        # 4. Diagnostic summaries. These make saturation visible without parsing every line.
        all_steps = []
        for k in range(K):
            recs = diag_by_client[k]
            all_steps.extend(recs)
            deltas = np.asarray([r["delta_norm"] for r in recs], dtype=np.float64)
            v_pcts = np.asarray([r["v_norm_percentile"] for r in recs], dtype=np.float64)
            v_deltas = np.asarray([r["v_delta_norm"] for r in recs], dtype=np.float64)
            positive = np.maximum(deltas, 0.0)
            negative = np.minimum(deltas, 0.0)
            positive_total = float(positive.sum())
            negative_damage = float(-negative.sum())
            net_gain = float(deltas.sum())

            def checkpoint(step_idx):
                if not recs:
                    return float("nan")
                idx = min(max(step_idx, 1), len(recs)) - 1
                return recs[idx]["resid_after"]

            def earliest_positive_capture(frac):
                if positive_total <= 0.0:
                    return None
                c = np.cumsum(positive)
                idx = np.searchsorted(c, frac * positive_total, side="left")
                return int(min(idx + 1, len(recs)))

            summary = {
                "type": "client_summary",
                "task": int(self._cur_task),
                "client": int(k),
                "selected": int(len(recs)),
                "positive_steps": int(np.sum(deltas > 0.0)),
                "nonpositive_steps": int(np.sum(deltas <= 0.0)),
                "negative_steps": int(np.sum(deltas < -1e-10)),
                "tiny_positive_1e4": int(np.sum((deltas > 0.0) & (deltas < 1e-4))),
                "tiny_positive_1e5": int(np.sum((deltas > 0.0) & (deltas < 1e-5))),
                "first_nonpositive_step": next((r["step"] for r in recs if r["nonpositive_gain"]), None),
                "first_negative_step": next((r["step"] for r in recs if r["negative_gain"]), None),
                "v_nonpositive_steps": int(np.sum(v_deltas <= 0.0)),
                "v_negative_steps": int(np.sum(v_deltas < -1e-10)),
                "first_v_nonpositive_step": next((r["step"] for r in recs if r["v_nonpositive_gain"]), None),
                "first_v_negative_step": next((r["step"] for r in recs if r["v_negative_gain"]), None),
                "net_gain_norm": net_gain,
                "positive_gain_total": positive_total,
                "negative_damage_total": negative_damage,
                "final_resid": float(recs[-1]["resid_after"]) if recs else None,
                "final_explained_sq": float(recs[-1]["explained_sq_after"]) if recs else None,
                "final_v_resid": float(recs[-1]["v_resid_after"]) if recs else None,
                "final_v_explained_sq": float(recs[-1]["v_explained_sq_after"]) if recs else None,
                "median_delta_first10": float(np.median(deltas[:10])) if len(deltas) else None,
                "median_delta_last10": float(np.median(deltas[-10:])) if len(deltas) else None,
                "late10_over_early10": float(np.median(deltas[-10:]) / (abs(np.median(deltas[:10])) + 1e-12)) if len(deltas) else None,
                "median_v_percentile_first10": float(np.median(v_pcts[:10])) if len(v_pcts) else None,
                "median_v_percentile_last10": float(np.median(v_pcts[-10:])) if len(v_pcts) else None,
                "positive_capture_step_90": earliest_positive_capture(0.90),
                "positive_capture_step_95": earliest_positive_capture(0.95),
                "positive_capture_step_99": earliest_positive_capture(0.99),
                "resid_after_1": checkpoint(1),
                "resid_after_5": checkpoint(5),
                "resid_after_10": checkpoint(10),
                "resid_after_20": checkpoint(20),
                "resid_after_30": checkpoint(30),
                "resid_after_45": checkpoint(45),
                "resid_after_60": checkpoint(60),
                "resid_after_75": checkpoint(75),
                "resid_after_90": checkpoint(90),
            }
            jsonl_records.append(summary)
            print(
                f"[EXP5_GREEDY_CLIENT_SUMMARY] task={self._cur_task} client={k} n={len(recs)} "
                f"pos={summary['positive_steps']} nonpos={summary['nonpositive_steps']} neg={summary['negative_steps']} "
                f"tiny1e-4={summary['tiny_positive_1e4']} tiny1e-5={summary['tiny_positive_1e5']} "
                f"firstNonPos={summary['first_nonpositive_step']} firstNeg={summary['first_negative_step']} "
                f"vNonPos={summary['v_nonpositive_steps']} vNeg={summary['v_negative_steps']} "
                f"firstVNonPos={summary['first_v_nonpositive_step']} firstVNeg={summary['first_v_negative_step']} "
                f"netGain={net_gain:+.6e} negDamage={negative_damage:.6e} "
                f"finalR={summary['final_resid']:.6e} finalVR={summary['final_v_resid']:.6e} "
                f"medDeltaFirst10={summary['median_delta_first10']:+.6e} "
                f"medDeltaLast10={summary['median_delta_last10']:+.6e} "
                f"late/early={summary['late10_over_early10']:+.6e} "
                f"cap90={summary['positive_capture_step_90']} cap95={summary['positive_capture_step_95']} cap99={summary['positive_capture_step_99']} "
                f"medVPctFirst10={summary['median_v_percentile_first10']:.4f} "
                f"medVPctLast10={summary['median_v_percentile_last10']:.4f}"
            )
            print(
                f"[EXP5_GREEDY_RESID_CHECKPOINTS] task={self._cur_task} client={k} "
                f"r1={summary['resid_after_1']:.6e} r5={summary['resid_after_5']:.6e} "
                f"r10={summary['resid_after_10']:.6e} r20={summary['resid_after_20']:.6e} "
                f"r30={summary['resid_after_30']:.6e} r45={summary['resid_after_45']:.6e} "
                f"r60={summary['resid_after_60']:.6e} r75={summary['resid_after_75']:.6e} "
                f"r90={summary['resid_after_90']:.6e}"
            )

        # Aggregate step-rank bins across clients.
        max_rank = max((r["step"] for r in all_steps), default=0)
        requested_bins = [(1, 10), (11, 30), (31, 60), (61, max_rank)]
        task_bin_records = []
        for lo, hi in requested_bins:
            hi = min(hi, max_rank)
            if lo > hi:
                continue
            rows = [r for r in all_steps if lo <= r["step"] <= hi]
            if not rows:
                continue
            d = np.asarray([r["delta_norm"] for r in rows], dtype=np.float64)
            vp = np.asarray([r["v_norm_percentile"] for r in rows], dtype=np.float64)
            zp = np.asarray([r["z_norm_percentile"] for r in rows], dtype=np.float64)
            cs = np.asarray([r["cosine_to_residual"] for r in rows], dtype=np.float64)
            vd = np.asarray([r["v_delta_norm"] for r in rows], dtype=np.float64)
            b = {
                "type": "task_bin",
                "task": int(self._cur_task),
                "step_lo": int(lo),
                "step_hi": int(hi),
                "count": int(len(rows)),
                "delta_mean": float(np.mean(d)),
                "delta_median": float(np.median(d)),
                "delta_min": float(np.min(d)),
                "delta_max": float(np.max(d)),
                "negative_count": int(np.sum(d < -1e-10)),
                "nonpositive_count": int(np.sum(d <= 0.0)),
                "v_delta_mean": float(np.mean(vd)),
                "v_delta_median": float(np.median(vd)),
                "v_negative_count": int(np.sum(vd < -1e-10)),
                "v_nonpositive_count": int(np.sum(vd <= 0.0)),
                "tiny_positive_1e4": int(np.sum((d > 0.0) & (d < 1e-4))),
                "v_percentile_median": float(np.median(vp)),
                "z_percentile_median": float(np.median(zp)),
                "cosine_median": float(np.median(cs)),
            }
            task_bin_records.append(b)
            jsonl_records.append(b)
            print(
                f"[EXP5_GREEDY_TASK_BIN] task={self._cur_task} steps={lo:02d}-{hi:02d} n={len(rows)} "
                f"deltaMean={b['delta_mean']:+.6e} deltaMed={b['delta_median']:+.6e} "
                f"deltaMin={b['delta_min']:+.6e} deltaMax={b['delta_max']:+.6e} "
                f"vDeltaMed={b['v_delta_median']:+.6e} "
                f"nonpos={b['nonpositive_count']} neg={b['negative_count']} "
                f"vNonpos={b['v_nonpositive_count']} vNeg={b['v_negative_count']} tiny1e-4={b['tiny_positive_1e4']} "
                f"medVPct={b['v_percentile_median']:.4f} medZPct={b['z_percentile_median']:.4f} "
                f"medCos={b['cosine_median']:+.5f}"
            )

        if all_steps:
            all_d = np.asarray([r["delta_norm"] for r in all_steps], dtype=np.float64)
            all_vd = np.asarray([r["v_delta_norm"] for r in all_steps], dtype=np.float64)
            first10 = np.asarray([r["delta_norm"] for r in all_steps if r["step"] <= 10], dtype=np.float64)
            last10 = np.asarray([r["delta_norm"] for r in all_steps if r["step"] > max_rank - 10], dtype=np.float64)
            task_summary = {
                "type": "task_summary",
                "task": int(self._cur_task),
                "selected": int(len(all_steps)),
                "max_rank": int(max_rank),
                "positive_steps": int(np.sum(all_d > 0.0)),
                "nonpositive_steps": int(np.sum(all_d <= 0.0)),
                "negative_steps": int(np.sum(all_d < -1e-10)),
                "v_nonpositive_steps": int(np.sum(all_vd <= 0.0)),
                "v_negative_steps": int(np.sum(all_vd < -1e-10)),
                "tiny_positive_1e4": int(np.sum((all_d > 0.0) & (all_d < 1e-4))),
                "tiny_positive_1e5": int(np.sum((all_d > 0.0) & (all_d < 1e-5))),
                "median_delta_first10": float(np.median(first10)) if first10.size else None,
                "median_delta_last10": float(np.median(last10)) if last10.size else None,
                "late10_over_early10": float(np.median(last10) / (abs(np.median(first10)) + 1e-12)) if first10.size and last10.size else None,
            }
            jsonl_records.append(task_summary)
            print(
                f"[EXP5_GREEDY_TASK_SUMMARY] task={self._cur_task} selected={len(all_steps)} "
                f"pos={task_summary['positive_steps']} nonpos={task_summary['nonpositive_steps']} neg={task_summary['negative_steps']} "
                f"vNonpos={task_summary['v_nonpositive_steps']} vNeg={task_summary['v_negative_steps']} "
                f"tiny1e-4={task_summary['tiny_positive_1e4']} tiny1e-5={task_summary['tiny_positive_1e5']} "
                f"medDeltaFirst10={task_summary['median_delta_first10']:+.6e} "
                f"medDeltaLast10={task_summary['median_delta_last10']:+.6e} "
                f"late/early={task_summary['late10_over_early10']:+.6e}"
            )

        self._write_exp5_diag_jsonl(jsonl_records)
        return selected_by_client

    def _evaluate_selection(self, selected_by_client, Z_dict, user_groups, offsets, D, cand_rows):
        K = self.num_users
        per_client_z_err = []
        per_client_v_err = []

        sum_hat_v = torch.zeros(D, dtype=torch.float32)
        sum_target_v = torch.zeros(D, dtype=torch.float32)

        client_counts = []
        class_hist = {}

        for k in range(K):
            Nk = len(user_groups[k])
            Bk = len(selected_by_client[k])
            client_counts.append(Bk)

            # Projected-space reconstruction
            Z_k = Z_dict[k]
            T_k_z_tilde = (float(Bk) / float(Nk)) * torch.sum(Z_k, dim=0)
            hat_T_k_z = torch.sum(Z_k[selected_by_client[k]], dim=0)
            err_z = (torch.norm(hat_T_k_z - T_k_z_tilde) / (torch.norm(T_k_z_tilde) + 1e-8)).item()
            per_client_z_err.append(err_z)

            # Original D-space reconstruction
            start_idx = offsets[k]
            end_idx = start_idx + Nk
            V_k = self.trajectory_matrix[start_idx:end_idx]
            T_k_v = (float(Bk) / float(Nk)) * torch.sum(V_k, dim=0)
            hat_T_k_v = torch.sum(V_k[selected_by_client[k]], dim=0)
            err_v = (torch.norm(hat_T_k_v - T_k_v) / (torch.norm(T_k_v) + 1e-8)).item()
            per_client_v_err.append(err_v)

            sum_hat_v += hat_T_k_v
            sum_target_v += T_k_v

            # Record class labels of selected samples
            for j in selected_by_client[k]:
                lbl = cand_rows[start_idx + j]["label"]
                class_hist[lbl] = class_hist.get(lbl, 0) + 1

        E_global_v = (torch.norm(sum_hat_v - sum_target_v) / (torch.norm(sum_target_v) + 1e-8)).item()
        mean_z_err = float(np.mean(per_client_z_err))
        mean_v_err = float(np.mean(per_client_v_err))

        total_M = sum(client_counts)
        shares = [c / total_M for c in client_counts]
        max_share = max(shares)
        hhi = sum(s ** 2 for s in shares)

        print(f"\n--- Exp5 Replay Evaluation (Task {self._cur_task}) ---")
        print(f"Client allocations: {client_counts} (Max share: {max_share:.3f}, HHI: {hhi:.4f})")
        print(f"Mean client projected z-error: {mean_z_err:.4e}")
        print(f"Mean client original v-error:  {mean_v_err:.4e}")
        print(f"Global original v-error E_v:   {E_global_v:.4e}")
        print(f"Class histogram (unique classes={len(class_hist)}): {sorted(class_hist.items())}")

        if hasattr(self, "_log_global_z_error"):
            self._log_global_z_error(selected_by_client, Z_dict, user_groups)

        if self.wandb == 1 and wandb is not None:
            wandb.log({
                f"Task_{self._cur_task}/mean_z_recon_error": mean_z_err,
                f"Task_{self._cur_task}/global_v_recon_error": E_global_v,
                f"Task_{self._cur_task}/client_hhi": hhi,
                f"Task_{self._cur_task}/max_client_share": max_share,
            })

    def _commit_replay_memory(self, selected_by_client, user_groups, train_dataset):
        """Converts selected candidate indices into persistent ReplayDataset instances."""
        new_datasets = []
        for k in range(self.num_users):
            local_ids = sorted(selected_by_client[k])
            dataset_indices = [user_groups[k][l] for l in local_ids]

            selected_images = train_dataset.images[dataset_indices]
            selected_labels = train_dataset.labels[dataset_indices]

            replay_ds = ReplayDataset(
                selected_images,
                selected_labels,
                self.test_loader.dataset.trsf
                if (self.exp5_distill_loss or self.exp5_single_distill_loss) else train_dataset.trsf,
                use_path=train_dataset.use_path,
            )
            self.retained_ds_all[k].append(replay_ds)
            new_datasets.append(replay_ds)
            total_client_replay = sum(len(ds) for ds in self.retained_ds_all[k])
            print(f"Client {k}: Retained {len(local_ids)} new replay samples (cumulative: {total_client_replay}).")

        if self.exp5_distill_loss:
            self._refresh_replay_logits()
        elif self.exp5_single_distill_loss:
            self._refresh_replay_logits(new_datasets)


class Exp5aLocal(Exp5Base):
    """Exp5a-Local: Each client learns its own rank-r trajectory basis independently."""
    def __init__(self, args):
        super().__init__(args)

    def _compute_projected_trajectories(self, user_groups, offsets, cand_rows):
        K = self.num_users
        r_target = self.exp5a_rank
        p = self.exp5a_svd_oversampling

        Z_dict = {}
        R_dict = {}
        energies = []

        for k in range(K):
            Nk = len(user_groups[k])
            start_idx = offsets[k]
            end_idx = start_idx + Nk
            X_k = self.trajectory_matrix[start_idx:end_idx].clone()
            D = X_k.shape[1]

            rk = min(r_target, Nk, D)
            qk = min(rk + p, Nk, D)

            setup_seed(self.seed + self._cur_task * 1000 + k,
                       fast_cuda=(self.args.get("fast_cuda", False) or
                                  self.args.get("t4_parralel", False)))
            # Uncentered randomized PCA: X_k \approx U_k \Sigma_k V_k^T
            U, S, V = torch.pca_lowrank(X_k, q=qk, center=False)
            R_k = V[:, :rk]

            fro_sq = torch.sum(X_k ** 2) + 1e-8
            rho_k = (torch.sum(S[:rk] ** 2) / fro_sq).item()
            energies.append(rho_k)

            # Project client trajectory vectors: Z_k = X_k R_k
            Z_k = X_k @ R_k
            Z_dict[k] = Z_k
            R_dict[k] = R_k

        mean_rho = float(np.mean(energies))
        print(f"[Exp5a-Local] Retained energy per client: {[round(e, 4) for e in energies]} (Mean: {mean_rho:.4f})")
        if self.wandb == 1 and wandb is not None:
            wandb.log({f"Task_{self._cur_task}/mean_retained_energy": mean_rho})

        return Z_dict, R_dict, energies


class Exp5aGlobal(Exp5Base):
    """Exp5a-Global: All clients share one globally coordinated rank-r basis."""
    def __init__(self, args):
        super().__init__(args)
        # Exp6 inherits this setting for its separate masked projection path.
        self.exp5a_mask_layers = args.get("exp5a_mask_layers", 12)

    def _compute_projected_trajectories(self, user_groups, offsets, cand_rows):
        K = self.num_users
        r_target = self.exp5a_rank
        p = self.exp5a_svd_oversampling
        # Client trajectory blocks already occupy consecutive rows of this matrix.
        X = self.trajectory_matrix
        N, D = X.shape

        # Learn one shared subspace from the unmasked pooled trajectories.
        r_glob = min(r_target, N, D)
        q = min(r_glob + p, N, D)

        setup_seed(self.seed + self._cur_task * 1000,
                   fast_cuda=(self.args.get("fast_cuda", False) or
                              self.args.get("t4_parralel", False)))
        _, S, V = torch.pca_lowrank(X, q=q, center=False)
        R_r = V[:, :r_glob]

        fro_sq = torch.sum(X ** 2) + 1e-8
        rho_global = (torch.sum(S[:r_glob] ** 2) / fro_sq).item()
        print(f"[Exp5a-Global] Global retained trajectory energy rho_global: {rho_global:.4f}")

        if self.wandb == 1 and wandb is not None:
            wandb.log({f"Task_{self._cur_task}/global_retained_energy": rho_global})

        # Project each client's original trajectories into the shared basis.
        Z_dict = {}
        R_dict = {}
        for k in range(K):
            Nk = len(user_groups[k])
            start_idx = offsets[k]
            end_idx = start_idx + Nk
            X_k = X[start_idx:end_idx]
            Z_k = X_k @ R_r
            Z_dict[k] = Z_k
            R_dict[k] = R_r

        return Z_dict, R_dict, [rho_global]

    def _log_global_z_error(self, selected_by_client, Z_dict, user_groups):
        """Global z-space error is geometrically meaningful because all clients share R_r."""
        K = self.num_users
        r_dim = Z_dict[0].shape[1]
        sum_hat_z = torch.zeros(r_dim, dtype=torch.float32)
        sum_target_z = torch.zeros(r_dim, dtype=torch.float32)

        for k in range(K):
            Nk = len(user_groups[k])
            Bk = len(selected_by_client[k])
            Z_k = Z_dict[k]

            T_k_z_tilde = (float(Bk) / float(Nk)) * torch.sum(Z_k, dim=0)
            hat_T_k_z = torch.sum(Z_k[selected_by_client[k]], dim=0)

            sum_hat_z += hat_T_k_z
            sum_target_z += T_k_z_tilde

        E_global_z = (torch.norm(sum_hat_z - sum_target_z) / (torch.norm(sum_target_z) + 1e-8)).item()
        print(f"[Exp5a-Global] Global projected z-error E_global_z: {E_global_z:.4e}")
        if self.wandb == 1 and wandb is not None:
            wandb.log({f"Task_{self._cur_task}/global_z_recon_error": E_global_z})

class Exp5bTopK(Exp5aGlobal):
    """
    Exp5b-TopK:
    Same trajectory construction and global projected subspace as Exp5aGlobal.

    Selection ablation:
      1. For each client, construct the same budget-scaled target T_k.
      2. Score every candidate ONCE against the initial target:

             gain_j = ||T_k||^2 - ||T_k - z_j||^2
                    = 2 z_j^T T_k - ||z_j||^2

      3. Keep only candidates with strictly positive gain.
      4. Rank them by gain descending.
      5. Select the highest-positive items.

    Unlike Exp5aGlobal, the residual is NOT updated after each selection.
    Therefore this isolates the ranking quality from iterative residual matching.

    The nominal task budget M is preserved as an upper bound. If fewer than M
    positive candidates exist, the unused budget is intentionally left empty.
    """

    def __init__(self, args):
        super().__init__(args)

    def _greedy_selection(self, Z_dict, user_groups, M, offsets, cand_rows):
        K = self.num_users

        # Same base/remainder budget convention as Exp5aGlobal.
        B = M // K
        R = M - K * B

        selected_by_client = {k: [] for k in range(K)}
        ranked_by_client = {}
        jsonl_records = []

        # -------------------------------------------------------------
        # 1. Compute one-shot positive-gain ranking independently
        #    inside every client.
        # -------------------------------------------------------------
        for k in range(K):
            Z_k = Z_dict[k]
            Nk = Z_k.shape[0]

            if Nk == 0:
                ranked_by_client[k] = []
                continue

            # Same target definition used by Exp5aGlobal.
            if self.target_mode == "full_sum":
                T_k = torch.sum(Z_k, dim=0)
            else:
                T_k = (float(B) / float(Nk)) * torch.sum(Z_k, dim=0)

            # Initial marginal gain:
            #
            #   ||T||^2 - ||T-z_j||^2
            # = 2 z_j^T T - ||z_j||^2
            #
            z_norm_sq = torch.sum(Z_k ** 2, dim=1)
            dots = torch.mv(Z_k, T_k)
            gains = 2.0 * dots - z_norm_sq

            # Strict positive-only filtering.
            positive_idx = torch.nonzero(
                gains > 0.0, as_tuple=False
            ).flatten()

            if positive_idx.numel() == 0:
                ranked_by_client[k] = []
                print(
                    f"[EXP5B_TOPK_RANKING] task={self._cur_task} "
                    f"client={k} candidates={Nk} positive=0"
                )
                continue

            positive_gains = gains[positive_idx]

            # Highest positive gain first.
            order = torch.argsort(
                positive_gains,
                descending=True,
            )

            ranked_idx = positive_idx[order].cpu().tolist()
            ranked_by_client[k] = ranked_idx

            sorted_gains = positive_gains[order]

            print(
                f"[EXP5B_TOPK_RANKING] task={self._cur_task} "
                f"client={k} candidates={Nk} "
                f"positive={len(ranked_idx)} "
                f"best_gain={sorted_gains[0].item():+.6e} "
                f"worst_positive_gain={sorted_gains[-1].item():+.6e}"
            )

        # -------------------------------------------------------------
        # 2. Give every client its normal base quota B.
        # -------------------------------------------------------------
        for k in range(K):
            take = min(B, len(ranked_by_client[k]))
            selected_by_client[k] = ranked_by_client[k][:take]

        # -------------------------------------------------------------
        # 3. Allocate the R ordinary remainder slots globally according
        #    to the next-highest positive gain.
        #
        #    In addition, if some client cannot fill B because it has
        #    too few positive candidates, those unused slots are also
        #    allowed to move globally. This keeps M as the task-level
        #    upper bound while never admitting a non-positive item.
        # -------------------------------------------------------------
        used = sum(len(v) for v in selected_by_client.values())
        remaining_slots = M - used

        if remaining_slots > 0:
            global_remaining = []

            for k in range(K):
                Z_k = Z_dict[k]
                Nk = Z_k.shape[0]

                if Nk == 0:
                    continue

                if self.target_mode == "full_sum":
                    T_k = torch.sum(Z_k, dim=0)
                else:
                    T_k = (float(B) / float(Nk)) * torch.sum(Z_k, dim=0)

                z_norm_sq = torch.sum(Z_k ** 2, dim=1)
                gains = 2.0 * torch.mv(Z_k, T_k) - z_norm_sq

                already = set(selected_by_client[k])

                for j in ranked_by_client[k]:
                    if j not in already:
                        global_remaining.append(
                            (float(gains[j].item()), k, j)
                        )

            global_remaining.sort(key=lambda x: x[0], reverse=True)

            for gain, k, j in global_remaining[:remaining_slots]:
                selected_by_client[k].append(j)

        # -------------------------------------------------------------
        # 4. Diagnostics.
        # -------------------------------------------------------------
        total_selected = sum(
            len(v) for v in selected_by_client.values()
        )

        client_counts = [
            len(selected_by_client[k]) for k in range(K)
        ]

        print(
            f"[EXP5B_TOPK_SUMMARY] task={self._cur_task} "
            f"budget={M} selected={total_selected} "
            f"unused={M - total_selected} "
            f"allocations={client_counts}"
        )

        for k in range(K):
            Z_k = Z_dict[k]
            Nk = Z_k.shape[0]

            if Nk == 0:
                continue

            if self.target_mode == "full_sum":
                T_k = torch.sum(Z_k, dim=0)
            else:
                T_k = (float(B) / float(Nk)) * torch.sum(Z_k, dim=0)

            z_norm_sq = torch.sum(Z_k ** 2, dim=1)
            gains = 2.0 * torch.mv(Z_k, T_k) - z_norm_sq

            for rank, j in enumerate(selected_by_client[k], start=1):
                global_row = offsets[k] + j

                rec = {
                    "type": "exp5b_topk_step",
                    "task": int(self._cur_task),
                    "client": int(k),
                    "rank": int(rank),
                    "local_candidate": int(j),
                    "global_candidate": int(global_row),
                    "label": int(cand_rows[global_row]["label"]),
                    "initial_gain_raw": float(gains[j].item()),
                    "positive_gain": bool(gains[j].item() > 0.0),
                }
                jsonl_records.append(rec)

            assert (
                len(selected_by_client[k])
                == len(set(selected_by_client[k]))
            ), f"Client {k} has duplicate selections!"

        jsonl_records.append({
            "type": "exp5b_topk_task_summary",
            "task": int(self._cur_task),
            "budget": int(M),
            "selected": int(total_selected),
            "unused_budget": int(M - total_selected),
            "client_allocations": client_counts,
        })

        self._write_exp5_diag_jsonl(jsonl_records)

        return selected_by_client


class Exp5bFedCBDR(Exp5aGlobal):
    """
    Exp5b-FedCBDR:
    Same trajectory matrix and Exp5aGlobal projected representation Z_k,
    but replace residual matching by FedCBDR's leverage-score mechanism.

    For each client:
        Z_k = U_k S_k V_k^T
        tau_j = ||U_k[j, :]||_2^2
        tau <- tau / sum(tau)

    Following FedCBDR:
      - each client receives floor(M/K) draws;
      - leverage scores determine sampling probability;
      - sampling uses torch.multinomial(..., replacement=True).

    NOTE:
    ReplayDataset ultimately represents retained dataset items rather than
    repeated sampling weights. Therefore duplicate FedCBDR draws are
    collapsed to unique local sample indices before memory is committed.
    """

    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def _fedcbdr_leverage_scores(Z_k):
        """
        FedCBDR-style row leverage scores:

            tau_j = ||e_j^T U||_2^2

        where U comes from the SVD of the client matrix.
        """
        Nk = Z_k.shape[0]

        if Nk == 0:
            return torch.empty(0, dtype=torch.float32)

        # torch.linalg.svd is the modern equivalent of the torch.svd
        # operation used in the released FedCBDR implementation.
        U, _, _ = torch.linalg.svd(
            Z_k,
            full_matrices=False,
        )

        tau = torch.sum(U ** 2, dim=1)

        tau_sum = tau.sum()
        if tau_sum <= 0.0 or not torch.isfinite(tau_sum):
            # Defensive fallback only for a degenerate matrix.
            tau = torch.ones(
                Nk,
                dtype=Z_k.dtype,
                device=Z_k.device,
            )
            tau_sum = tau.sum()

        tau = tau / tau_sum
        return tau

    def _greedy_selection(self, Z_dict, user_groups, M, offsets, cand_rows):
        K = self.num_users
        B = M // K
        R = M - K * B

        selected_by_client = {k: [] for k in range(K)}
        leverage_by_client = {}
        jsonl_records = []

        # -------------------------------------------------------------
        # 1. FedCBDR local leverage scores.
        # -------------------------------------------------------------
        for k in range(K):
            tau_k = self._fedcbdr_leverage_scores(Z_dict[k])
            leverage_by_client[k] = tau_k

            if tau_k.numel() > 0:
                print(
                    f"[EXP5B_FEDCBDR_LEVERAGE] "
                    f"task={self._cur_task} client={k} "
                    f"N={tau_k.numel()} "
                    f"sum={tau_k.sum().item():.6f} "
                    f"min={tau_k.min().item():.6e} "
                    f"max={tau_k.max().item():.6e}"
                )

        # -------------------------------------------------------------
        # 2. Construct FedCBDR global probability.
        #
        # FedCBDR multiplies each client's internally normalized tau by
        # 1/K, concatenates them, and normalizes globally.
        # -------------------------------------------------------------
        global_parts = []

        for k in range(K):
            tau_k = leverage_by_client[k]
            if tau_k.numel() > 0:
                global_parts.append(tau_k / float(K))
            else:
                global_parts.append(tau_k)

        p_global = torch.cat(global_parts)

        p_sum = p_global.sum()
        if p_sum <= 0.0 or not torch.isfinite(p_sum):
            raise RuntimeError(
                "Exp5bFedCBDR obtained invalid global leverage probabilities."
            )

        p_global = p_global / p_sum

        # -------------------------------------------------------------
        # 3. FedCBDR base sampling:
        #
        #       B = floor(M / K)
        #
        # draws PER CLIENT, WITH replacement.
        # -------------------------------------------------------------
        sampled_draws_by_client = {k: [] for k in range(K)}

        for k in range(K):
            tau_k = leverage_by_client[k]
            Nk = tau_k.numel()

            if Nk == 0 or B == 0:
                continue

            # FedCBDR uses the client's slice of the globally aggregated
            # probability vector directly. torch.multinomial accepts
            # unnormalized non-negative weights.
            draws = torch.multinomial(
                tau_k,
                B,
                replacement=True,
            )

            sampled_draws_by_client[k].extend(
                draws.cpu().tolist()
            )

        # -------------------------------------------------------------
        # 4. FedCBDR remainder sampling globally.
        # -------------------------------------------------------------
        if R > 0:
            extra = torch.multinomial(
                p_global,
                R,
                replacement=True,
            ).cpu().tolist()

            sizes = [Z_dict[k].shape[0] for k in range(K)]
            boundaries = np.cumsum([0] + sizes)

            for global_idx in extra:
                for k in range(K):
                    if (
                        boundaries[k]
                        <= global_idx
                        < boundaries[k + 1]
                    ):
                        local_j = global_idx - boundaries[k]
                        sampled_draws_by_client[k].append(
                            int(local_j)
                        )
                        break

        # -------------------------------------------------------------
        # 5. Convert draws to retained dataset items.
        #
        # FedCBDR samples with replacement, so duplicate draws are
        # possible. Exp5 replay memory stores dataset items, therefore
        # collapse duplicates while preserving first-draw order.
        # -------------------------------------------------------------
        total_draws = 0
        total_unique = 0

        for k in range(K):
            draws = sampled_draws_by_client[k]
            total_draws += len(draws)

            seen = set()
            unique = []

            for j in draws:
                if j not in seen:
                    seen.add(j)
                    unique.append(j)

            selected_by_client[k] = unique
            total_unique += len(unique)

            duplicate_count = len(draws) - len(unique)

            print(
                f"[EXP5B_FEDCBDR_CLIENT] "
                f"task={self._cur_task} client={k} "
                f"draws={len(draws)} unique={len(unique)} "
                f"duplicates={duplicate_count}"
            )

            tau_k = leverage_by_client[k]

            for draw_rank, j in enumerate(draws, start=1):
                global_row = offsets[k] + j

                jsonl_records.append({
                    "type": "exp5b_fedcbdr_draw",
                    "task": int(self._cur_task),
                    "client": int(k),
                    "draw": int(draw_rank),
                    "local_candidate": int(j),
                    "global_candidate": int(global_row),
                    "label": int(cand_rows[global_row]["label"]),
                    "leverage_score": float(tau_k[j].item()),
                })

        assert total_draws == M, (
            f"Expected {M} FedCBDR draws, got {total_draws}."
        )

        print(
            f"[EXP5B_FEDCBDR_SUMMARY] "
            f"task={self._cur_task} budget={M} "
            f"draws={total_draws} unique_retained={total_unique} "
            f"duplicate_draws={total_draws - total_unique} "
            f"allocations="
            f"{[len(selected_by_client[k]) for k in range(K)]}"
        )

        jsonl_records.append({
            "type": "exp5b_fedcbdr_task_summary",
            "task": int(self._cur_task),
            "budget": int(M),
            "draws": int(total_draws),
            "unique_retained": int(total_unique),
            "duplicate_draws": int(total_draws - total_unique),
            "client_allocations": [
                len(selected_by_client[k]) for k in range(K)
            ],
        })

        self._write_exp5_diag_jsonl(jsonl_records)

        return selected_by_client
