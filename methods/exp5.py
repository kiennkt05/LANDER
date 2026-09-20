import copy
import logging
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

    Returns:
        (idx, image, label, is_current, candidate_id)
        where `is_current` is True only for current-task samples, and `candidate_id` is the
        row index in the current task trajectory matrix V (or -1 for replay).
    """
    def __init__(self, current_dataset, candidate_ids, replay_datasets=None):
        self.current_dataset = current_dataset
        self.candidate_ids = candidate_ids
        self.num_current = len(current_dataset)

        self.replay_datasets = replay_datasets or []
        self.replay_offsets = []
        cur_offset = self.num_current
        for ds in self.replay_datasets:
            self.replay_offsets.append(cur_offset)
            cur_offset += len(ds)
        self.total_len = cur_offset

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.num_current:
            _, image, label = self.current_dataset[idx]
            cand_id = self.candidate_ids[idx]
            return idx, image, label, True, cand_id
        else:
            for ds_idx, offset in enumerate(self.replay_offsets):
                ds = self.replay_datasets[ds_idx]
                if idx < offset + len(ds):
                    item_idx = idx - offset
                    _, image, label = ds[item_idx]
                    return idx, image, label, False, -1
            raise IndexError(f"Index {idx} out of range for ClientTaskDataset of length {self.total_len}")


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
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )

        setup_seed(self.seed)
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
        self._network.cuda()
        self._fl_train(train_dataset, user_groups, offsets, D)

        # 4. Trajectory projection into rank-r subspace (implemented by subclasses)
        print(f"Task {self._cur_task}: Computing rank-{self.exp5a_rank} projected trajectory embeddings...")
        Z_dict, R_dict, energies = self._compute_projected_trajectories(user_groups, offsets, cand_rows)

        # 5. Greedy Replay Selection
        M = self.gdr_task_budget
        print(f"Task {self._cur_task}: Performing replay selection for budget M={M} (target_mode={self.target_mode})...")
        selected_by_client = self._greedy_selection(Z_dict, user_groups, M)

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

        self.best_model = None  # Best model using the lowest training loss
        self.lowest_loss = np.inf

        local_lr = self.args.get("local_lr", 0.01)
        momentum = 0.9
        weight_decay = self.args.get("weight_decay", 5e-4)
        local_ep = self.args["local_ep"]
        local_bs = self.args["local_bs"]
        frac = self.args["frac"]

        round_errors = []
        task_pred_delta_theta = torch.zeros(D, dtype=torch.float32)
        task_actual_delta_theta = torch.zeros(D, dtype=torch.float32)

        for com in prog_bar:
            # Snapshot head parameters before local training
            head_before = torch.cat([
                self._network.fc.weight.data.flatten(),
                self._network.fc.bias.data.flatten()
            ]).detach().cpu()

            # Client selection
            idxs_users = range(self.args["num_users"])
            p_k = 1.0 / self.args["num_users"]

            round_delta_cur = torch.zeros(D, dtype=torch.float32)
            round_delta_rep = torch.zeros(D, dtype=torch.float32)
            round_delta_wd = torch.zeros(D, dtype=torch.float32)

            local_weights = []
            loss_weight = []

            for k in idxs_users:
                local_model = copy.deepcopy(self._network)
                local_model.train()
                local_model.cuda()

                Nk = len(user_groups[k])
                cand_ids = np.arange(offsets[k], offsets[k] + Nk)
                cur_ds = DatasetSplit(train_dataset, user_groups[k])
                client_ds = ClientTaskDataset(cur_ds, cand_ids, self.retained_ds_all[k])

                local_loader = DataLoader(
                    client_ds, batch_size=local_bs, shuffle=True, num_workers=4
                )

                optimizer = torch.optim.SGD(
                    local_model.parameters(),
                    lr=local_lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    dampening=0.0,
                    nesterov=False,
                )

                num_steps_per_epoch = len(local_loader)
                S = local_ep * num_steps_per_epoch
                step = 0

                client_loss = 0.0
                for ep in range(local_ep):
                    for batch_idx, (b_idxs, images, labels, is_cur, cands) in enumerate(local_loader):
                        step += 1
                        # Exact momentum multiplier: W_s = (1 - mu^(S - s + 1)) / (1 - mu)
                        W_s = (1.0 - (momentum ** (S - step + 1))) / (1.0 - momentum)

                        images = images.cuda()
                        labels = labels.cuda()

                        # Capture head parameters before update step for weight decay attribution
                        theta_head = torch.cat([
                            local_model.fc.weight.data.flatten(),
                            local_model.fc.bias.data.flatten()
                        ]).detach().cpu()

                        outputs = local_model(images)
                        features = outputs["features"]
                        logits = outputs["logits"]

                        loss = F.cross_entropy(logits, labels)
                        if ep == 0:
                            client_loss += loss.detach()

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
                            step_coeff = -local_lr * p_k * W_s
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

                        step_coeff = -local_lr * p_k * W_s
                        round_delta_cur += step_coeff * g_cur_sum
                        round_delta_rep += step_coeff * g_rep_sum
                        round_delta_wd += (step_coeff * weight_decay) * theta_head

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                local_weights.append(copy.deepcopy(local_model.state_dict()))
                loss_weight.append(client_loss)
                del local_loader, local_model
                torch.cuda.empty_cache()

            # FedAvg aggregation
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            sum_loss = sum(loss_weight)
            if sum_loss < self.lowest_loss:
                self.lowest_loss = sum_loss
                self.best_model = copy.deepcopy(self._network.state_dict())

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

            if (com + 1) % 1 == 0 or com == com_round - 1:
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

        self._network.load_state_dict(self.best_model)  # Best model using the lowest training loss
        del self.best_model
        torch.cuda.empty_cache()

    def _compute_projected_trajectories(self, user_groups, offsets, cand_rows):
        raise NotImplementedError("Subclasses must implement _compute_projected_trajectories.")

    def _greedy_selection(self, Z_dict, user_groups, M):
        """Per-client FedCBDR-style greedy selection on projected trajectory vectors."""
        K = self.num_users
        B = M // K
        R = M - K * B

        selected_by_client = {k: [] for k in range(K)}
        residual_by_client = {}
        target_by_client = {}

        # 1. Construct reconstruction target and initial residual
        for k in range(K):
            Z_k = Z_dict[k]
            Nk = Z_k.shape[0]
            if self.target_mode == "full_sum":
                T_k = torch.sum(Z_k, dim=0)
            else: # budget_scaled_sum
                T_k = (float(B) / float(Nk)) * torch.sum(Z_k, dim=0)
            target_by_client[k] = T_k
            residual_by_client[k] = T_k.clone()

        available_by_client = {k: np.ones(Z_dict[k].shape[0], dtype=bool) for k in range(K)}
        norms_sq_by_client = {k: torch.sum(Z_dict[k] ** 2, dim=1) for k in range(K)}

        # 2. Hard base quota: B samples per client in round-robin order
        for step in range(B):
            for k in range(K):
                r_k = residual_by_client[k]
                T_k = target_by_client[k]
                denom = torch.sum(T_k ** 2).item() + 1e-8

                avail_indices = np.where(available_by_client[k])[0]
                if len(avail_indices) == 0:
                    continue

                Z_avail = Z_dict[k][avail_indices]
                norms_avail = norms_sq_by_client[k][avail_indices]

                # ||r_k - z_j||^2 = ||r_k||^2 + ||z_j||^2 - 2 z_j^T r_k
                r_norm_sq = torch.sum(r_k ** 2)
                dots = torch.mv(Z_avail, r_k)
                costs = (r_norm_sq + norms_avail - 2.0 * dots) / denom

                best_idx_in_avail = torch.argmin(costs).item()
                best_j = avail_indices[best_idx_in_avail]

                selected_by_client[k].append(best_j)
                available_by_client[k][best_j] = False
                residual_by_client[k] -= Z_dict[k][best_j]

        # 3. Remainder slots: R slots globally allocated to cheapest candidate
        for rem in range(R):
            best_cost = float("inf")
            best_client = None
            best_j = None

            for k in range(K):
                r_k = residual_by_client[k]
                T_k = target_by_client[k]
                denom = torch.sum(T_k ** 2).item() + 1e-8

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
                selected_by_client[best_client].append(best_j)
                available_by_client[best_client][best_j] = False
                residual_by_client[best_client] -= Z_dict[best_client][best_j]

        total_selected = sum(len(v) for v in selected_by_client.values())
        assert total_selected == M, f"Expected {M} selected samples, but got {total_selected}!"
        for k in range(K):
            assert len(selected_by_client[k]) == len(set(selected_by_client[k])), f"Client {k} has duplicate selections!"

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
        for k in range(self.num_users):
            local_ids = sorted(selected_by_client[k])
            dataset_indices = [user_groups[k][l] for l in local_ids]

            selected_images = train_dataset.images[dataset_indices]
            selected_labels = train_dataset.labels[dataset_indices]

            replay_ds = ReplayDataset(
                selected_images,
                selected_labels,
                train_dataset.trsf,
                use_path=train_dataset.use_path,
            )
            self.retained_ds_all[k].append(replay_ds)
            total_client_replay = sum(len(ds) for ds in self.retained_ds_all[k])
            print(f"Client {k}: Retained {len(local_ids)} new replay samples (cumulative: {total_client_replay}).")


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

            setup_seed(self.seed + self._cur_task * 1000 + k)
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
        self.exp5a_mask_layers = args.get("exp5a_mask_layers", 12)

    def _compute_projected_trajectories(self, user_groups, offsets, cand_rows):
        K = self.num_users
        r_target = self.exp5a_rank
        p = self.exp5a_svd_oversampling
        layers = self.exp5a_mask_layers

        N = self.trajectory_matrix.shape[0]
        D = self.trajectory_matrix.shape[1]

        # 1. Common D-dimensional feature-space orthogonal mask Q
        mask_seed = self.seed + self._cur_task * 5000
        Q = _ImplicitOrthogonalMask(D, num_layers=layers, seed=mask_seed)

        # 2. Pool masked blocks X'_k = P_k X_k Q vertically into X' in R^(N x D)
        X_prime = torch.zeros((N, D), dtype=torch.float32)

        for k in range(K):
            Nk = len(user_groups[k])
            start_idx = offsets[k]
            end_idx = start_idx + Nk
            X_k = self.trajectory_matrix[start_idx:end_idx].clone()

            # Client sample-space orthogonal row mask P_k
            row_seed = mask_seed + 100 + k
            P_k = _ImplicitOrthogonalMask(Nk, num_layers=layers, seed=row_seed)

            X_k_masked_rows = P_k.apply_left(X_k)
            X_k_prime = Q.apply_right(X_k_masked_rows)
            X_prime[start_idx:end_idx] = X_k_prime

        # 3. Learn global trajectory subspace on pooled X'
        r_glob = min(r_target, N, D)
        q = min(r_glob + p, N, D)

        setup_seed(self.seed + self._cur_task * 1000)
        U_prime, S_prime, V_prime = torch.pca_lowrank(X_prime, q=q, center=False)
        R_prime_r = V_prime[:, :r_glob]

        fro_sq = torch.sum(X_prime ** 2) + 1e-8
        rho_global = (torch.sum(S_prime[:r_glob] ** 2) / fro_sq).item()
        print(f"[Exp5a-Global] Global retained trajectory energy rho_global: {rho_global:.4f}")

        if self.wandb == 1 and wandb is not None:
            wandb.log({f"Task_{self._cur_task}/global_retained_energy": rho_global})

        del X_prime

        # 4. Project un-row-mixed client vectors: Z_k = (X_k Q) R'_r
        Z_dict = {}
        R_dict = {}
        for k in range(K):
            Nk = len(user_groups[k])
            start_idx = offsets[k]
            end_idx = start_idx + Nk
            X_k = self.trajectory_matrix[start_idx:end_idx]

            X_k_Q = Q.apply_right(X_k)
            Z_k = X_k_Q @ R_prime_r
            Z_dict[k] = Z_k
            R_dict[k] = R_prime_r

        return Z_dict, R_dict, [rho_global]

    def _log_global_z_error(self, selected_by_client, Z_dict, user_groups):
        """Global z-space error is geometrically meaningful because all clients share R'_r."""
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


