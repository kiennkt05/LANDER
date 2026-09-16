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
from methods.exp5 import (
    _ImplicitOrthogonalMask,
    ReplayDataset,
    ClientTaskDataset,
    Exp5Base,
    Exp5aGlobal,
)
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed, pil_loader

try:
    import wandb
except ImportError:
    wandb = None


def _exp5a_global_projection(
    vectors,
    rank,
    user_groups=None,
    offsets=None,
    svd_oversampling=16,
    mask_layers=12,
    seed=42,
    cur_task=0,
    mask_seed_offset=5000,
):
    """Shared global trajectory subspace projection via orthogonal masking and pooled randomized SVD.

    Learns a shared global trajectory subspace via orthogonal masking and pooled randomized SVD,
    then projects each client's un-row-mixed vectors into the shared rank-r subspace.

    Args:
        vectors: Either a 2D Tensor of shape (N, D) or a dict {k: X_k} of 2D Tensors.
        rank: int, target projection rank r.
        user_groups: List of client candidate indices or None.
        offsets: List of client start row indices or None.
        svd_oversampling: int, oversampling parameter p for randomized SVD (default: 16).
        mask_layers: int, number of Givens rotation layers for orthogonal mask (default: 12).
        seed: int, base random seed.
        cur_task: int, current continual learning task ID.
        mask_seed_offset: int, seed stride for task masking (default: 5000).

    Returns:
        Z_dict: dict {k: Z_k} where Z_k is (Nk, r)
        diagnostics: dict with R_prime_r, rho_global, R_dict, energies, S_prime
    """
    if isinstance(vectors, dict):
        K = len(vectors)
        client_sizes = [vectors[k].shape[0] for k in range(K)]
        D = vectors[0].shape[1]
        offsets = [0] + [int(x) for x in np.cumsum(client_sizes[:-1])]
        N = sum(client_sizes)
        X_mat = torch.cat([vectors[k] for k in range(K)], dim=0)
    else:
        assert isinstance(vectors, torch.Tensor), "vectors must be a Tensor or dict of Tensors"
        N, D = vectors.shape
        if user_groups is not None:
            K = len(user_groups)
            if isinstance(user_groups, dict):
                client_sizes = [len(user_groups[k]) for k in range(K)]
            else:
                client_sizes = [len(g) for g in user_groups]
            if offsets is None:
                offsets = [0] + [int(x) for x in np.cumsum(client_sizes[:-1])]
        elif offsets is not None:
            K = len(offsets)
            client_sizes = []
            for k in range(K):
                nxt = offsets[k + 1] if k + 1 < K else N
                client_sizes.append(nxt - offsets[k])
        else:
            raise ValueError("user_groups or offsets must be provided when vectors is a Tensor")
        X_mat = vectors

    # 1. Common D-dimensional feature-space orthogonal mask Q
    mask_seed = seed + cur_task * mask_seed_offset
    Q = _ImplicitOrthogonalMask(D, num_layers=mask_layers, seed=mask_seed)

    # 2. Pool masked blocks X'_k = P_k X_k Q vertically into X' in R^(N x D)
    X_prime = torch.zeros((N, D), dtype=torch.float32)

    for k in range(K):
        Nk = client_sizes[k]
        start_idx = offsets[k]
        end_idx = start_idx + Nk
        X_k = X_mat[start_idx:end_idx].clone()

        # Client sample-space orthogonal row mask P_k
        row_seed = mask_seed + 100 + k
        P_k = _ImplicitOrthogonalMask(Nk, num_layers=mask_layers, seed=row_seed)

        X_k_masked_rows = P_k.apply_left(X_k)
        X_k_prime = Q.apply_right(X_k_masked_rows)
        X_prime[start_idx:end_idx] = X_k_prime

    # 3. Learn global trajectory subspace on pooled X'
    r_glob = min(rank, N, D)
    q = min(r_glob + svd_oversampling, N, D)

    setup_seed(seed + cur_task * 1000)
    U_prime, S_prime, V_prime = torch.pca_lowrank(X_prime, q=q, center=False)
    R_prime_r = V_prime[:, :r_glob]

    fro_sq = torch.sum(X_prime ** 2) + 1e-8
    rho_global = (torch.sum(S_prime[:r_glob] ** 2) / fro_sq).item()

    del X_prime

    # 4. Project un-row-mixed client vectors: Z_k = (X_k Q) R'_r
    Z_dict = {}
    R_dict = {}
    for k in range(K):
        Nk = client_sizes[k]
        start_idx = offsets[k]
        end_idx = start_idx + Nk
        X_k = X_mat[start_idx:end_idx]

        X_k_Q = Q.apply_right(X_k)
        Z_k = X_k_Q @ R_prime_r
        Z_dict[k] = Z_k
        R_dict[k] = R_prime_r

    diagnostics = {
        "R_prime_r": R_prime_r,
        "rho_global": rho_global,
        "R_dict": R_dict,
        "energies": [rho_global],
        "S_prime": S_prime,
    }
    return Z_dict, diagnostics


def _greedy_fixed_quota_to_target(vectors, target, quota):
    """Greedily selects `quota` vectors from `vectors` without replacement to minimize:
        ||target - sum_{j in selected} vectors[j]||^2

    Args:
        vectors: Tensor of shape (Nk, r) representing candidate projected vectors.
        target: Tensor of shape (r,) representing target to reconstruct.
        quota: int, number of items to select.

    Returns:
        selected_positions: List[int] of indices into `vectors` of length `quota`.
    """
    Nk = vectors.shape[0]
    B = min(int(quota), Nk)
    res = target.clone()
    denom = torch.sum(target ** 2).item() + 1e-8

    available = np.ones(Nk, dtype=bool)
    norms_sq = torch.sum(vectors ** 2, dim=1)
    selected_positions = []

    for step in range(B):
        avail_indices = np.where(available)[0]
        if len(avail_indices) == 0:
            break

        Z_avail = vectors[avail_indices]
        norms_avail = norms_sq[avail_indices]
        r_norm_sq = torch.sum(res ** 2)
        dots = torch.mv(Z_avail, res)
        costs = (r_norm_sq + norms_avail - 2.0 * dots) / denom

        best_idx_in_avail = torch.argmin(costs).item()
        best_j = avail_indices[best_idx_in_avail]

        selected_positions.append(best_j)
        available[best_j] = False
        res -= vectors[best_j]

    return selected_positions


def _joint_coordinate_match_on(
    vectors,
    initial_rows,
    target=None,
    quota_per_client=None,
    max_passes=5,
    improvement_tol=1e-8,
):
    r"""Baseline-initialized joint coordinate optimization.

    Minimizes:
        J(S) = ||T - \sum_k \sum_{i \in S_k} z_i||^2
    initialized from `initial_rows` (Exp5a-global per-client selection).

    Args:
        vectors: dict {k: Z_k} of candidate vectors per client, where Z_k is (Nk, r).
        initial_rows: dict {k: S_k} of initial selected indices per client.
        target: Tensor of shape (r,) representing global target T.
                If None, defaults to T = sum_k (B_k / Nk) * sum(Z_k).
        quota_per_client: dict or int or None for quota per client.
        max_passes: int, maximum coordinate passes (default: 5).
        improvement_tol: float, threshold tolerance for coordinate improvement (default: 1e-8).

    Returns:
        selected_rows: dict {k: S_k} of final selected indices.
        metrics: dict of summary metrics.
    """
    K = len(vectors)
    quotas = {}
    for k in range(K):
        if quota_per_client is None:
            quotas[k] = len(initial_rows[k])
        elif isinstance(quota_per_client, int):
            quotas[k] = quota_per_client
        else:
            quotas[k] = quota_per_client[k]

    if target is None:
        target_k = {}
        for k in range(K):
            Nk = vectors[k].shape[0]
            Bk = quotas[k]
            target_k[k] = (float(Bk) / float(Nk)) * torch.sum(vectors[k], dim=0)
        T = torch.stack(list(target_k.values())).sum(dim=0)
    else:
        T = target.clone()

    T_norm = torch.norm(T).item() + 1e-8

    # S <- Exp5a-global per-client selection
    S = {k: list(initial_rows[k]) for k in range(K)}
    G = {}
    for k in range(K):
        if len(S[k]) > 0:
            G[k] = torch.sum(vectors[k][S[k]], dim=0)
        else:
            G[k] = torch.zeros_like(T)

    Sigma = torch.stack(list(G.values())).sum(dim=0)
    J = torch.sum((T - Sigma) ** 2).item()
    initial_J = J
    E_ind = (torch.norm(T - Sigma).item()) / T_norm

    passes_completed = 0
    for pass_idx in range(max_passes):
        improved = False
        passes_completed += 1

        for k in range(K):
            # R_k = T - \sum_{l \neq k} G_l
            Sigma_not_k = Sigma - G[k]
            R_k = T - Sigma_not_k

            S_prime_k = _greedy_fixed_quota_to_target(
                vectors=vectors[k],
                target=R_k,
                quota=quotas[k],
            )

            G_prime_k = torch.sum(vectors[k][S_prime_k], dim=0)
            J_prime = torch.sum((R_k - G_prime_k) ** 2).item()

            if J_prime < J - improvement_tol:
                S[k] = S_prime_k
                Sigma = Sigma_not_k + G_prime_k
                G[k] = G_prime_k
                J = J_prime
                improved = True

        if not improved:
            break

    final_Sigma = torch.stack(list(G.values())).sum(dim=0)
    final_J = torch.sum((T - final_Sigma) ** 2).item()
    E_joint = (torch.norm(T - final_Sigma).item()) / T_norm
    rel_improvement = (E_ind - E_joint) / (E_ind + 1e-8)

    total_overlap = sum(len(set(initial_rows[k]) & set(S[k])) for k in range(K))
    total_selected = sum(len(S[k]) for k in range(K))
    overlap_ratio = total_overlap / total_selected if total_selected > 0 else 1.0

    client_hist = {k: len(S[k]) for k in range(K)}
    shares = [len(S[k]) / total_selected for k in range(K)] if total_selected > 0 else [0] * K
    hhi = sum(s ** 2 for s in shares)
    unique_count = sum(len(set(S[k])) for k in range(K))

    metrics = {
        "passes": passes_completed,
        "initial_J": initial_J,
        "final_J": final_J,
        "independent_z_recon": E_ind,
        "joint_z_recon": E_joint,
        "relative_improvement": rel_improvement,
        "selection_overlap": f"{total_overlap}/{total_selected} ({overlap_ratio:.2%})",
        "overlap_count": total_overlap,
        "overlap_ratio": overlap_ratio,
        "client_hist": client_hist,
        "HHI": hhi,
        "unique": unique_count,
        "total_selected": total_selected,
        "target": T,
        "G_joint": final_Sigma,
    }

    return S, metrics


class Exp6Global(Exp5aGlobal):
    r"""Exp6: Joint FedAvg-Aware Replay Selection via Baseline-Initialized Coordinate Descent.

    Keeps representation and FedCBDR allocation (|S_k| = M/K = 90, no duplicates) identical to Exp5a-global,
    but replaces independent per-client selection with joint global trajectory reconstruction:
        J(S) = ||T - \sum_k \sum_{i \in S_k} z_i||^2
    initialized from Exp5a-global per-client selection.
    """
    def __init__(self, args):
        super().__init__(args)
        self.exp6_max_passes = args.get("exp6_max_passes", 5)
        self.exp6_improvement_tol = args.get("exp6_improvement_tol", 1e-8)
        self.exp5a_mask_seed_offset = args.get("exp5a_mask_seed_offset", 5000)

    def _compute_projected_trajectories(self, user_groups, offsets, cand_rows):
        Z_dict, diagnostics = _exp5a_global_projection(
            vectors=self.trajectory_matrix,
            rank=self.exp5a_rank,
            user_groups=user_groups,
            offsets=offsets,
            svd_oversampling=self.exp5a_svd_oversampling,
            mask_layers=self.exp5a_mask_layers,
            seed=self.seed,
            cur_task=self._cur_task,
            mask_seed_offset=self.exp5a_mask_seed_offset,
        )
        rho_global = diagnostics["rho_global"]
        print(f"[Exp6-Global] Global retained trajectory energy rho_global: {rho_global:.4f}")

        if self.wandb == 1 and wandb is not None:
            wandb.log({f"Task_{self._cur_task}/global_retained_energy": rho_global})

        return Z_dict, diagnostics["R_dict"], diagnostics["energies"]

    def _build_replay_from_rows(self, selected_by_client, user_groups, train_dataset):
        """Converts selected candidate rows to ReplayDataset & updates retained memory."""
        return self._commit_replay_memory(selected_by_client, user_groups, train_dataset)

    def _construct_exp6_coreset(self, Z_dict, user_groups, M, cand_rows, offsets, D, train_dataset):
        K = self.num_users
        B = M // K
        assert M % K == 0, f"Replay budget M={M} must be divisible by num_users K={K}"

        # 1. Baseline Exp5a-global per-client greedy selection
        S_initial = self._greedy_selection(Z_dict, user_groups, M)

        # 2. Joint coordinate optimization conditioned on selections of all other clients
        S_exp6, metrics = _joint_coordinate_match_on(
            vectors=Z_dict,
            initial_rows=S_initial,
            target=None,
            quota_per_client=B,
            max_passes=self.exp6_max_passes,
            improvement_tol=self.exp6_improvement_tol,
        )

        # 3. Compute original D-space reconstruction on raw trajectory matrix
        sum_hat_v = torch.zeros(D, dtype=torch.float32)
        sum_target_v = torch.zeros(D, dtype=torch.float32)
        for k in range(K):
            Nk = len(user_groups[k])
            Bk = len(S_exp6[k])
            start_idx = offsets[k]
            end_idx = start_idx + Nk
            V_k = self.trajectory_matrix[start_idx:end_idx]
            T_k_v = (float(Bk) / float(Nk)) * torch.sum(V_k, dim=0)
            hat_T_k_v = torch.sum(V_k[S_exp6[k]], dim=0)
            sum_hat_v += hat_T_k_v
            sum_target_v += T_k_v

        orig_D_recon = (torch.norm(sum_hat_v - sum_target_v) / (torch.norm(sum_target_v) + 1e-8)).item()

        # 4. Mandatory Logging: EXP6_CORESET_SUMMARY
        summary_str = f"""
==================== EXP6_CORESET_SUMMARY ====================
task={self._cur_task}
rank={self.exp5a_rank}
passes={metrics['passes']}
independent_z_recon={metrics['independent_z_recon']:.6e}
joint_z_recon={metrics['joint_z_recon']:.6e}
relative_improvement={metrics['relative_improvement']:.4%}
orig_D_recon={orig_D_recon:.6e}
selection_overlap={metrics['selection_overlap']}
client_hist={metrics['client_hist']}
HHI={metrics['HHI']:.4f}
unique={metrics['unique']}
==============================================================
"""
        print(summary_str)

        if self.wandb == 1 and wandb is not None:
            wandb.log({
                f"Task_{self._cur_task}/exp6_independent_z_recon": metrics['independent_z_recon'],
                f"Task_{self._cur_task}/exp6_joint_z_recon": metrics['joint_z_recon'],
                f"Task_{self._cur_task}/exp6_relative_improvement": metrics['relative_improvement'],
                f"Task_{self._cur_task}/exp6_orig_D_recon": orig_D_recon,
                f"Task_{self._cur_task}/exp6_passes": metrics['passes'],
                f"Task_{self._cur_task}/exp6_overlap_ratio": metrics['overlap_ratio'],
            })

        # 5. Invariants Verification Gate
        expected_client_hist = {k: B for k in range(K)}
        assert metrics['client_hist'] == expected_client_hist, (
            f"Client quota invariant failed! Expected {expected_client_hist}, got {metrics['client_hist']}"
        )
        expected_hhi = 1.0 / float(K)
        assert abs(metrics['HHI'] - expected_hhi) < 1e-4, (
            f"HHI invariant failed! Expected {expected_hhi:.4f}, got {metrics['HHI']:.4f}"
        )
        assert metrics['unique'] == M, f"Unique samples invariant failed! Expected {M}, got {metrics['unique']}"
        assert metrics['total_selected'] == M, f"Total samples invariant failed! Expected {M}, got {metrics['total_selected']}"
        assert metrics['joint_z_recon'] <= metrics['independent_z_recon'] + 1e-7, (
            f"Monotonicity invariant failed! Joint recon {metrics['joint_z_recon']:.6e} > Independent {metrics['independent_z_recon']:.6e}"
        )

        # 6. Replay memory commit via _build_replay_from_rows
        self._build_replay_from_rows(S_exp6, user_groups, train_dataset)

        # 7. Also run standard evaluation metrics
        self._evaluate_selection(S_exp6, Z_dict, user_groups, offsets, D, cand_rows)

        return S_exp6

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

        # 4. Trajectory projection into rank-r subspace
        print(f"Task {self._cur_task}: Computing rank-{self.exp5a_rank} projected trajectory embeddings...")
        Z_dict, R_dict, energies = self._compute_projected_trajectories(user_groups, offsets, cand_rows)

        # 5. Exp6 Joint Replay Selection
        M = self.gdr_task_budget
        print(f"Task {self._cur_task}: Performing Exp6 Joint Replay Selection for budget M={M}...")
        selected_by_client = self._construct_exp6_coreset(
            Z_dict=Z_dict,
            user_groups=user_groups,
            M=M,
            cand_rows=cand_rows,
            offsets=offsets,
            D=D,
            train_dataset=train_dataset,
        )

        # 6. Release trajectory matrix
        del self.trajectory_matrix
        self.trajectory_matrix = None
        print(f"Task {self._cur_task}: Replay selection completed. Trajectory matrix released.")
