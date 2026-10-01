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
    fast_cuda=False,
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

    setup_seed(seed + cur_task * 1000, fast_cuda=fast_cuda)
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


def _build_target_by_client(vectors, quotas, target_mode="budget_scaled_sum"):
    """Builds the per-client reconstruction targets used by replay selection.

    For the default ``budget_scaled_sum`` target, client k approximates the
    expected sum of ``B_k`` uniformly sampled trajectories:
        T_k = (B_k / N_k) * sum_i z_i.

    ``full_sum`` is retained for compatibility with Exp5's target_mode.
    """
    target_by_client = {}
    for k, Z_k in vectors.items():
        Nk = Z_k.shape[0]
        Bk = int(quotas[k])
        if target_mode == "full_sum":
            T_k = torch.sum(Z_k, dim=0)
        elif target_mode == "budget_scaled_sum":
            T_k = (float(Bk) / float(Nk)) * torch.sum(Z_k, dim=0)
        else:
            raise ValueError(
                f"Unsupported target_mode={target_mode!r}; expected "
                "'budget_scaled_sum' or 'full_sum'."
            )
        target_by_client[k] = T_k
    return target_by_client


def _projected_reconstruction_metrics(vectors, selected_rows, target_by_client):
    """Computes global/local reconstruction and cross-client cancellation diagnostics."""
    K = len(vectors)
    G = {}
    residuals = {}
    client_rel_sq = {}
    client_rel = {}

    for k in range(K):
        T_k = target_by_client[k]
        rows = selected_rows[k]
        if len(rows) > 0:
            G_k = torch.sum(vectors[k][rows], dim=0)
        else:
            G_k = torch.zeros_like(T_k)
        G[k] = G_k

        e_k = T_k - G_k
        residuals[k] = e_k
        denom_k = torch.sum(T_k ** 2).item() + 1e-8
        rel_sq = torch.sum(e_k ** 2).item() / denom_k
        client_rel_sq[k] = rel_sq
        client_rel[k] = float(np.sqrt(max(rel_sq, 0.0)))

    T = torch.stack([target_by_client[k] for k in range(K)]).sum(dim=0)
    global_residual = torch.stack([residuals[k] for k in range(K)]).sum(dim=0)
    global_rel_sq = torch.sum(global_residual ** 2).item() / (torch.sum(T ** 2).item() + 1e-8)

    mean_client_rel_sq = float(np.mean(list(client_rel_sq.values())))
    mean_client_rel = float(np.mean(list(client_rel.values())))
    rms_client_rel = float(np.sqrt(max(mean_client_rel_sq, 0.0)))

    sum_residual_norms = sum(torch.norm(residuals[k]).item() for k in range(K))
    cancellation_ratio = torch.norm(global_residual).item() / (sum_residual_norms + 1e-8)

    return {
        "G": G,
        "residuals": residuals,
        "target": T,
        "global_rel_sq": global_rel_sq,
        "global_recon": float(np.sqrt(max(global_rel_sq, 0.0))),
        "client_rel_sq": client_rel_sq,
        "client_recon": client_rel,
        "mean_client_rel_sq": mean_client_rel_sq,
        "mean_client_recon": mean_client_rel,
        "rms_client_recon": rms_client_rel,
        # 0 => strong residual cancellation; 1 => residuals point in similar directions.
        "cancellation_ratio": cancellation_ratio,
    }


def _greedy_fixed_quota_hybrid(
    vectors,
    global_target_conditioned,
    local_target,
    quota,
    global_denom_sq,
    local_denom_sq,
    local_weight,
):
    r"""Greedily solves one Exp6b client-coordinate subproblem.

    With all other client selections fixed, selecting client k changes only:

        global term = ||R_global,k - G_k||^2 / ||T||^2
        local term  = w_local * ||T_k - G_k||^2 / ||T_k||^2

    where ``R_global,k = T - sum_{l != k} G_l`` and
    ``w_local = lambda_local / K``.
    """
    Nk = vectors.shape[0]
    B = min(int(quota), Nk)
    available = np.ones(Nk, dtype=bool)
    norms_sq = torch.sum(vectors ** 2, dim=1)
    selected_positions = []

    # Residuals after the partial client subset selected so far.
    global_res = global_target_conditioned.clone()
    local_res = local_target.clone()

    for _ in range(B):
        avail_indices = np.where(available)[0]
        if len(avail_indices) == 0:
            break

        Z_avail = vectors[avail_indices]
        norms_avail = norms_sq[avail_indices]

        global_costs = (
            torch.sum(global_res ** 2)
            + norms_avail
            - 2.0 * torch.mv(Z_avail, global_res)
        ) / global_denom_sq

        local_costs = (
            torch.sum(local_res ** 2)
            + norms_avail
            - 2.0 * torch.mv(Z_avail, local_res)
        ) / local_denom_sq

        costs = global_costs + float(local_weight) * local_costs
        best_idx_in_avail = torch.argmin(costs).item()
        best_j = avail_indices[best_idx_in_avail]

        selected_positions.append(best_j)
        available[best_j] = False
        global_res -= vectors[best_j]
        local_res -= vectors[best_j]

    return selected_positions


def _hybrid_coordinate_match_on(
    vectors,
    initial_rows,
    target_by_client=None,
    quota_per_client=None,
    lambda_local=1.0,
    max_passes=5,
    improvement_tol=1e-8,
    target_mode="budget_scaled_sum",
):
    r"""Exp6b baseline-initialized global+local coordinate optimization.

    Minimizes the normalized hybrid objective

        J_b(S) = ||sum_k e_k||^2 / (||T||^2 + eps)
                 + lambda_local / K * sum_k ||e_k||^2 / (||T_k||^2 + eps),

    where ``e_k = T_k - sum_{i in S_k} z_i``.

    The first term is Exp6's FedAvg-aware global reconstruction objective.
    The second term prevents a low global error from being obtained purely by
    large, mutually-cancelling per-client residuals.

    ``lambda_local=0`` delegates to the original Exp6 optimizer so that the
    zero-lambda endpoint is an exact implementation-level control (up to the
    supplied targets/tolerance), rather than merely a similar objective.
    """
    if lambda_local < 0:
        raise ValueError(f"lambda_local must be >= 0, got {lambda_local}")

    K = len(vectors)
    quotas = {}
    for k in range(K):
        if quota_per_client is None:
            quotas[k] = len(initial_rows[k])
        elif isinstance(quota_per_client, int):
            quotas[k] = int(quota_per_client)
        else:
            quotas[k] = int(quota_per_client[k])

    if target_by_client is None:
        target_by_client = _build_target_by_client(
            vectors=vectors,
            quotas=quotas,
            target_mode=target_mode,
        )

    T = torch.stack([target_by_client[k] for k in range(K)]).sum(dim=0)

    # Exact Exp6 endpoint for lambda=0.
    if float(lambda_local) == 0.0:
        S, exp6_metrics = _joint_coordinate_match_on(
            vectors=vectors,
            initial_rows=initial_rows,
            target=T,
            quota_per_client=quotas,
            max_passes=max_passes,
            improvement_tol=improvement_tol,
        )
        baseline_metrics = _projected_reconstruction_metrics(
            vectors, initial_rows, target_by_client
        )
        final_metrics = _projected_reconstruction_metrics(
            vectors, S, target_by_client
        )
        return S, {
            "passes": exp6_metrics["passes"],
            "initial_objective": baseline_metrics["global_rel_sq"],
            "final_objective": final_metrics["global_rel_sq"],
            "relative_objective_improvement": (
                baseline_metrics["global_rel_sq"] - final_metrics["global_rel_sq"]
            ) / (baseline_metrics["global_rel_sq"] + 1e-8),
            "baseline": baseline_metrics,
            "final": final_metrics,
            "selection_overlap": exp6_metrics["selection_overlap"],
            "overlap_count": exp6_metrics["overlap_count"],
            "overlap_ratio": exp6_metrics["overlap_ratio"],
            "client_hist": exp6_metrics["client_hist"],
            "HHI": exp6_metrics["HHI"],
            "unique": exp6_metrics["unique"],
            "total_selected": exp6_metrics["total_selected"],
            "lambda_local": float(lambda_local),
        }

    # S <- Exp5a-global independent per-client selection.
    S = {k: list(initial_rows[k]) for k in range(K)}
    current_metrics = _projected_reconstruction_metrics(vectors, S, target_by_client)
    baseline_metrics = current_metrics
    J = (
        current_metrics["global_rel_sq"]
        + float(lambda_local) * current_metrics["mean_client_rel_sq"]
    )
    initial_J = J

    G = {k: current_metrics["G"][k].clone() for k in range(K)}
    local_rel_sq = dict(current_metrics["client_rel_sq"])
    Sigma = torch.stack([G[k] for k in range(K)]).sum(dim=0)

    global_denom_sq = torch.sum(T ** 2).item() + 1e-8
    local_denom_sq = {
        k: torch.sum(target_by_client[k] ** 2).item() + 1e-8
        for k in range(K)
    }
    local_weight = float(lambda_local) / float(K)

    passes_completed = 0
    for _ in range(max_passes):
        improved = False
        passes_completed += 1

        for k in range(K):
            Sigma_not_k = Sigma - G[k]
            R_global_k = T - Sigma_not_k

            S_prime_k = _greedy_fixed_quota_hybrid(
                vectors=vectors[k],
                global_target_conditioned=R_global_k,
                local_target=target_by_client[k],
                quota=quotas[k],
                global_denom_sq=global_denom_sq,
                local_denom_sq=local_denom_sq[k],
                local_weight=local_weight,
            )

            if len(S_prime_k) > 0:
                G_prime_k = torch.sum(vectors[k][S_prime_k], dim=0)
            else:
                G_prime_k = torch.zeros_like(target_by_client[k])

            global_residual_prime = R_global_k - G_prime_k
            global_rel_sq_prime = (
                torch.sum(global_residual_prime ** 2).item() / global_denom_sq
            )

            local_residual_prime = target_by_client[k] - G_prime_k
            local_rel_sq_prime = (
                torch.sum(local_residual_prime ** 2).item() / local_denom_sq[k]
            )

            # Other clients' local reconstruction terms are constant for this coordinate.
            sum_local_rel_sq_prime = (
                sum(local_rel_sq.values()) - local_rel_sq[k] + local_rel_sq_prime
            )
            J_prime = (
                global_rel_sq_prime
                + float(lambda_local) * (sum_local_rel_sq_prime / float(K))
            )

            if J_prime < J - improvement_tol:
                S[k] = S_prime_k
                G[k] = G_prime_k
                Sigma = Sigma_not_k + G_prime_k
                local_rel_sq[k] = local_rel_sq_prime
                J = J_prime
                improved = True

        if not improved:
            break

    final_metrics = _projected_reconstruction_metrics(vectors, S, target_by_client)
    final_J = (
        final_metrics["global_rel_sq"]
        + float(lambda_local) * final_metrics["mean_client_rel_sq"]
    )

    total_overlap = sum(len(set(initial_rows[k]) & set(S[k])) for k in range(K))
    total_selected = sum(len(S[k]) for k in range(K))
    overlap_ratio = total_overlap / total_selected if total_selected > 0 else 1.0
    client_hist = {k: len(S[k]) for k in range(K)}
    shares = [len(S[k]) / total_selected for k in range(K)] if total_selected > 0 else [0] * K
    hhi = sum(s ** 2 for s in shares)
    unique_count = sum(len(set(S[k])) for k in range(K))

    return S, {
        "passes": passes_completed,
        "initial_objective": initial_J,
        "final_objective": final_J,
        "relative_objective_improvement": (initial_J - final_J) / (initial_J + 1e-8),
        "baseline": baseline_metrics,
        "final": final_metrics,
        "selection_overlap": f"{total_overlap}/{total_selected} ({overlap_ratio:.2%})",
        "overlap_count": total_overlap,
        "overlap_ratio": overlap_ratio,
        "client_hist": client_hist,
        "HHI": hhi,
        "unique": unique_count,
        "total_selected": total_selected,
        "lambda_local": float(lambda_local),
    }


def _original_space_reconstruction_metrics(
    trajectory_matrix,
    selected_rows,
    user_groups,
    offsets,
    target_mode="budget_scaled_sum",
):
    """Original-D analogue of Exp6b reconstruction diagnostics."""
    K = len(user_groups)
    targets = {}
    residuals = {}
    client_rel = {}
    client_rel_sq = {}

    for k in range(K):
        Nk = len(user_groups[k])
        Bk = len(selected_rows[k])
        start_idx = offsets[k]
        end_idx = start_idx + Nk
        V_k = trajectory_matrix[start_idx:end_idx]

        if target_mode == "full_sum":
            T_k = torch.sum(V_k, dim=0)
        elif target_mode == "budget_scaled_sum":
            T_k = (float(Bk) / float(Nk)) * torch.sum(V_k, dim=0)
        else:
            raise ValueError(f"Unsupported target_mode={target_mode!r}")

        if Bk > 0:
            G_k = torch.sum(V_k[selected_rows[k]], dim=0)
        else:
            G_k = torch.zeros_like(T_k)

        e_k = T_k - G_k
        denom_k = torch.sum(T_k ** 2).item() + 1e-8
        rel_sq = torch.sum(e_k ** 2).item() / denom_k

        targets[k] = T_k
        residuals[k] = e_k
        client_rel_sq[k] = rel_sq
        client_rel[k] = float(np.sqrt(max(rel_sq, 0.0)))

    T = torch.stack([targets[k] for k in range(K)]).sum(dim=0)
    global_residual = torch.stack([residuals[k] for k in range(K)]).sum(dim=0)
    global_rel_sq = torch.sum(global_residual ** 2).item() / (torch.sum(T ** 2).item() + 1e-8)
    mean_client_rel_sq = float(np.mean(list(client_rel_sq.values())))

    sum_residual_norms = sum(torch.norm(residuals[k]).item() for k in range(K))
    cancellation_ratio = torch.norm(global_residual).item() / (sum_residual_norms + 1e-8)

    return {
        "global_recon": float(np.sqrt(max(global_rel_sq, 0.0))),
        "mean_client_recon": float(np.mean(list(client_rel.values()))),
        "rms_client_recon": float(np.sqrt(max(mean_client_rel_sq, 0.0))),
        "client_recon": client_rel,
        "cancellation_ratio": cancellation_ratio,
    }


class Exp6Global(Exp5aGlobal):
    r"""Exp6: Joint FedAvg-Aware Replay Selection via Baseline-Initialized Coordinate Descent.

    Keeps representation and FedCBDR allocation (|S_k| = B = M/K, no duplicates) identical to Exp5a-global,
    but replaces independent per-client selection with joint global trajectory reconstruction:
        J(S) = ||T - \sum_k \sum_{i \in S_k} z_i||^2
    initialized from Exp5a-global per-client selection.
    """
    def __init__(self, args):
        super().__init__(args)
        self.exp6_max_passes = args.get("exp6_max_passes", 5)
        self.exp6_improvement_tol = args.get("exp6_improvement_tol", 1e-8)
        self.exp5a_mask_seed_offset = args.get("exp5a_mask_seed_offset", 5000)
        self.exp6_variant_name = "Exp6"
        self.exp6_selection_label = "Exp6 Joint Replay Selection"

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
            fast_cuda=self.args.get("fast_cuda", False),
        )
        rho_global = diagnostics["rho_global"]
        print(f"[{self.exp6_variant_name}-Global] Global retained trajectory energy rho_global: {rho_global:.4f}")

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
        self.test_loader = self._test_data_loader(test_dataset)

        setup_seed(self.seed, fast_cuda=self.args.get("fast_cuda", False))
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

        # 4. Trajectory projection into rank-r subspace
        print(f"Task {self._cur_task}: Computing rank-{self.exp5a_rank} projected trajectory embeddings...")
        Z_dict, R_dict, energies = self._compute_projected_trajectories(user_groups, offsets, cand_rows)

        # 5. Exp6 Joint Replay Selection
        M = self.gdr_task_budget
        print(f"Task {self._cur_task}: Performing {self.exp6_selection_label} for budget M={M}...")
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



class Exp6bGlobal(Exp6Global):
    r"""Exp6b: Hybrid Global+Local Trajectory Replay Selection.

    Exp6b keeps Exp6's representation, Exp5a-global initialization, replay
    budget, hard per-client quotas, and coordinate-descent structure.  The
    only intended experimental change is the selection objective:

        J_b = E_global^2 + lambda_local * mean_k(E_local,k^2)

    where errors are normalized by their corresponding target norms.

    This directly tests the failure mode exposed by Exp6: a very small global
    FedAvg reconstruction error can be achieved while individual client
    reconstructions get worse because their residuals cancel after summation.
    """
    def __init__(self, args):
        super().__init__(args)
        self.exp6b_lambda = float(args.get("exp6b_lambda", 1.0))
        self.exp6b_max_passes = int(args.get("exp6b_max_passes", self.exp6_max_passes))
        self.exp6b_improvement_tol = float(
            args.get("exp6b_improvement_tol", self.exp6_improvement_tol)
        )
        if self.exp6b_lambda < 0:
            raise ValueError(f"exp6b_lambda must be >= 0, got {self.exp6b_lambda}")

        self.exp6_variant_name = "Exp6b"
        self.exp6_selection_label = (
            f"Exp6b Hybrid Global+Local Replay Selection (lambda={self.exp6b_lambda:g})"
        )

    def _construct_exp6_coreset(self, Z_dict, user_groups, M, cand_rows, offsets, D, train_dataset):
        K = self.num_users
        B = M // K
        assert M % K == 0, f"Replay budget M={M} must be divisible by num_users K={K}"

        # 1. Exact same initialization as Exp5a-global / Exp6.
        S_initial = self._greedy_selection(Z_dict, user_groups, M)
        quotas = {k: B for k in range(K)}
        target_by_client = _build_target_by_client(
            vectors=Z_dict,
            quotas=quotas,
            target_mode=self.target_mode,
        )

        # 2. Hybrid coordinate optimization. Only the objective differs from Exp6.
        S_exp6b, metrics = _hybrid_coordinate_match_on(
            vectors=Z_dict,
            initial_rows=S_initial,
            target_by_client=target_by_client,
            quota_per_client=quotas,
            lambda_local=self.exp6b_lambda,
            max_passes=self.exp6b_max_passes,
            improvement_tol=self.exp6b_improvement_tol,
            target_mode=self.target_mode,
        )

        # 3. Verify whether projected-space behavior survives in original D-space.
        baseline_D = _original_space_reconstruction_metrics(
            trajectory_matrix=self.trajectory_matrix,
            selected_rows=S_initial,
            user_groups=user_groups,
            offsets=offsets,
            target_mode=self.target_mode,
        )
        final_D = _original_space_reconstruction_metrics(
            trajectory_matrix=self.trajectory_matrix,
            selected_rows=S_exp6b,
            user_groups=user_groups,
            offsets=offsets,
            target_mode=self.target_mode,
        )

        b = metrics["baseline"]
        f = metrics["final"]

        # 4. Mandatory diagnostic block.  The cancellation ratio is especially
        # important: lower values mean stronger cross-client residual cancellation.
        summary_str = f"""
=================== EXP6B_CORESET_SUMMARY ===================
task={self._cur_task}
rank={self.exp5a_rank}
lambda_local={self.exp6b_lambda:.6g}
passes={metrics['passes']}
initial_hybrid_objective={metrics['initial_objective']:.6e}
final_hybrid_objective={metrics['final_objective']:.6e}
hybrid_relative_improvement={metrics['relative_objective_improvement']:.4%}
baseline_global_z_recon={b['global_recon']:.6e}
final_global_z_recon={f['global_recon']:.6e}
baseline_mean_client_z_recon={b['mean_client_recon']:.6e}
final_mean_client_z_recon={f['mean_client_recon']:.6e}
baseline_rms_client_z_recon={b['rms_client_recon']:.6e}
final_rms_client_z_recon={f['rms_client_recon']:.6e}
baseline_z_cancellation_ratio={b['cancellation_ratio']:.6e}
final_z_cancellation_ratio={f['cancellation_ratio']:.6e}
baseline_orig_D_global_recon={baseline_D['global_recon']:.6e}
final_orig_D_global_recon={final_D['global_recon']:.6e}
baseline_orig_D_mean_client_recon={baseline_D['mean_client_recon']:.6e}
final_orig_D_mean_client_recon={final_D['mean_client_recon']:.6e}
baseline_orig_D_cancellation_ratio={baseline_D['cancellation_ratio']:.6e}
final_orig_D_cancellation_ratio={final_D['cancellation_ratio']:.6e}
baseline_client_z_recon={{{', '.join(f'{k}: {b["client_recon"][k]:.6e}' for k in range(K))}}}
final_client_z_recon={{{', '.join(f'{k}: {f["client_recon"][k]:.6e}' for k in range(K))}}}
selection_overlap={metrics['selection_overlap']}
client_hist={metrics['client_hist']}
HHI={metrics['HHI']:.4f}
unique={metrics['unique']}
==============================================================
"""
        print(summary_str)

        if self.wandb == 1 and wandb is not None:
            wandb.log({
                f"Task_{self._cur_task}/exp6b_lambda": self.exp6b_lambda,
                f"Task_{self._cur_task}/exp6b_initial_objective": metrics["initial_objective"],
                f"Task_{self._cur_task}/exp6b_final_objective": metrics["final_objective"],
                f"Task_{self._cur_task}/exp6b_objective_improvement": metrics["relative_objective_improvement"],
                f"Task_{self._cur_task}/exp6b_baseline_global_z_recon": b["global_recon"],
                f"Task_{self._cur_task}/exp6b_final_global_z_recon": f["global_recon"],
                f"Task_{self._cur_task}/exp6b_baseline_mean_client_z_recon": b["mean_client_recon"],
                f"Task_{self._cur_task}/exp6b_final_mean_client_z_recon": f["mean_client_recon"],
                f"Task_{self._cur_task}/exp6b_baseline_z_cancellation_ratio": b["cancellation_ratio"],
                f"Task_{self._cur_task}/exp6b_final_z_cancellation_ratio": f["cancellation_ratio"],
                f"Task_{self._cur_task}/exp6b_baseline_orig_D_global_recon": baseline_D["global_recon"],
                f"Task_{self._cur_task}/exp6b_final_orig_D_global_recon": final_D["global_recon"],
                f"Task_{self._cur_task}/exp6b_baseline_orig_D_mean_client_recon": baseline_D["mean_client_recon"],
                f"Task_{self._cur_task}/exp6b_final_orig_D_mean_client_recon": final_D["mean_client_recon"],
                f"Task_{self._cur_task}/exp6b_overlap_ratio": metrics["overlap_ratio"],
                f"Task_{self._cur_task}/exp6b_passes": metrics["passes"],
            })

        # 5. Invariants.  For lambda>0, global reconstruction alone is allowed
        # to trade off against local reconstruction; only the hybrid objective
        # must be monotone relative to the Exp5a initialization.
        expected_client_hist = {k: B for k in range(K)}
        assert metrics["client_hist"] == expected_client_hist, (
            f"Client quota invariant failed! Expected {expected_client_hist}, "
            f"got {metrics['client_hist']}"
        )
        expected_hhi = 1.0 / float(K)
        assert abs(metrics["HHI"] - expected_hhi) < 1e-4, (
            f"HHI invariant failed! Expected {expected_hhi:.4f}, got {metrics['HHI']:.4f}"
        )
        assert metrics["unique"] == M, (
            f"Unique samples invariant failed! Expected {M}, got {metrics['unique']}"
        )
        assert metrics["total_selected"] == M, (
            f"Total samples invariant failed! Expected {M}, got {metrics['total_selected']}"
        )
        assert metrics["final_objective"] <= metrics["initial_objective"] + 1e-7, (
            "Hybrid monotonicity invariant failed! "
            f"final={metrics['final_objective']:.6e} > "
            f"initial={metrics['initial_objective']:.6e}"
        )

        # Strong endpoint check: lambda=0 should preserve Exp6's global monotonicity.
        if self.exp6b_lambda == 0.0:
            assert f["global_recon"] <= b["global_recon"] + 1e-7, (
                "lambda=0 endpoint failed Exp6 global monotonicity: "
                f"final={f['global_recon']:.6e} > baseline={b['global_recon']:.6e}"
            )

        # 6. Commit and run the same standard Exp5 evaluation used by Exp6.
        self._build_replay_from_rows(S_exp6b, user_groups, train_dataset)
        self._evaluate_selection(S_exp6b, Z_dict, user_groups, offsets, D, cand_rows)

        return S_exp6b
