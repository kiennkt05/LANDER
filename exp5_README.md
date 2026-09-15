# Exp5a: Trajectory-Subspace Replay Selection

## 1. Problem setup and notation

At incremental task \(t\):

* \(K\): number of federated clients.
* \(M\): replay budget for the **current task**, given by `gdr_task_budget`.
* \(\mathcal D_{t,k}\): current-task training samples assigned to client \(k\).
* \(N_k = |\mathcal D_{t,k}|\).
* \(N=\sum_k N_k\).
* \(C_t\): total number of classes seen through task \(t\).
* \(d\): backbone feature dimension.
* \(D=C_t d+C_t\): dimension of the flattened classifier-head weight+bias update.
* \(r\): Exp5a projection rank (`exp5a_rank`).
* \(p\): randomized-PCA oversampling (`exp5a_svd_oversampling`).
* \(v_i\in\mathbb R^D\): exact accumulated trajectory-contribution vector of current-task sample \(i\).
* \(z_i\in\mathbb R^r\): projected trajectory vector used by Exp5a for replay selection.

The implementation defaults to \(r=64\), oversampling \(p=16\), and 12 implicit orthogonal-mask layers for the global variant. Both Exp5a phases require the trajectory-attribution validation gate to have passed before selection is allowed.
The essential distinction is:

$$
\boxed{\text{Exp5a-Local: each client learns its own rank-}r\text{ basis}}
$$

versus

$$
\boxed{\text{Exp5a-Global: all clients share one globally coordinated rank-}r\text{ basis}.}
$$

Everything before the projection and everything after the projection is otherwise essentially shared.

---

# 2. Common Stage A — Federated training and exact trajectory construction

## 2.1 Initialize the task

At the beginning of task \(t\):

1. Partition the current-task dataset among the \(K\) clients.
2. Expand the classifier head from the previous number of classes to \(C_t\).
3. Identify its linear weight

$$
W\in\mathbb R^{C_t\times d}
$$

and bias

$$
b\in\mathbb R^{C_t}.
$$

4. Define the trajectory-vector dimension

$$
D=C_t d+C_t.
$$

5. Give every current-task sample a stable global candidate row containing:

   * client ID,
   * client-local sample ID,
   * class label.
6. Allocate

$$
V=
\begin{bmatrix}
v_1^\top\\
\vdots\\
v_N^\top
\end{bmatrix}
\in\mathbb R^{N\times D}
$$

and initialize \(V=0\).

The code explicitly constructs this current-task candidate table and allocates the trajectory matrix only for trajectory-based experimental phases such as Exp5a.
Previous replay samples are **not candidates for the new coreset**. They remain part of local training, while the persistent \(v_i\) matrix is constructed only for current-task samples.

---

## 2.2 Construct each client's training dataset

For client \(k\),

$$
\mathcal B_{t,k}
=
\mathcal D_{t,k}
\cup
\mathcal R_{0,k}
\cup\cdots\cup
\mathcal R_{t-1,k},
$$

where \(\mathcal R_{\tau,k}\) is the replay dataset retained from previous task \(\tau\).

Thus replay accumulates across tasks. The next task trains on the current data plus **all previously retained client-local replay datasets**.

---

## 2.3 Federated communication round

For communication round \(a=1,\ldots,A\):

1. Select participating clients \(\mathcal K_a\).
2. Use uniform FedAvg weight

$$
p_{k,a}=\frac{1}{|\mathcal K_a|}.
$$

3. Copy the current global model to every selected client.
4. Train each selected client locally.
5. Uniformly average the resulting local state dictionaries to obtain the next global model.

The implementation performs this loop explicitly and computes the global classifier-head update after every communication round.

---

# 3. Common Stage B — Per-sample trajectory contribution

## 3.1 Local SGD dynamics

For one selected client in one communication round, suppose there are \(S\) local SGD steps.

The optimizer is ordinary SGD with:

* learning rate \(\eta_a\),
* momentum \(\mu\),
* weight decay \(\lambda_{\mathrm{wd}}\),
* no dampening,
* no Nesterov.

For step \(s\),

$$
d_s=g_s+\lambda_{\mathrm{wd}}\theta_{s-1},
$$

$$
m_s=\mu m_{s-1}+d_s,
$$

$$
\theta_s=\theta_{s-1}-\eta_a m_s.
$$

Consequently, the gradient at step \(s\) influences every subsequent momentum update. Its exact accumulated momentum multiplier is

$$
W_s
=
\frac{1-\mu^{S-s+1}}{1-\mu}.
$$

The implementation computes this coefficient for every local step.

---

## 3.2 Exact per-sample classifier-head gradient

For current sample \(i\) occurring at SGD step \(s\), let

$$
h_{i,s}\in\mathbb R^d
$$

be its backbone feature and

$$
q_{i,s}
=
\frac{\partial L_s}{\partial \ell_{i,s}}
\in\mathbb R^{C_t}
$$

be the derivative of the **actual optimized scalar loss** with respect to that sample's logits.

Then

$$
\nabla_W L_{i,s}
=
q_{i,s}h_{i,s}^{\top}
$$

and

$$
\nabla_b L_{i,s}=q_{i,s}.
$$

Flatten them into

$$
g_{i,s}
=
\begin{bmatrix}
\operatorname{vec}(q_{i,s}h_{i,s}^{\top})\\
q_{i,s}
\end{bmatrix}
\in\mathbb R^D.
$$

## The code obtains \(q_{i,s}\) using `autograd.grad(loss, logits)` and constructs exactly this weight-plus-bias gradient vector.

## 3.3 Add the sample's contribution to its trajectory vector

For current sample \(i\) belonging to client \(k\),

$$
\Delta v_{i,s}
=
-\eta_a\,p_{k,a}\,W_s\,g_{i,s}.
$$

Therefore,

$$
\boxed{
v_i
=
\sum_{\substack{\text{rounds }a\\
\text{local occurrences }s\text{ of }i}}
-\eta_a p_{k,a}W_sg_{i,s}
}
$$

over the entire training of task \(t\).

The code implements this accumulation as

$$
v_i\leftarrow
v_i-\eta_a p_{k,a}W_sg_{i,s}.
$$

Important consequences:

* a sample appearing during multiple local epochs contributes multiple times;
* a sample contributes only in communication rounds in which its client participates;
* \(v_i\) already incorporates the actual learning rate, FedAvg coefficient, momentum propagation, model state through \(h_i\), and actual task-aware loss through \(q_i\);
* replay observations participate in training and invariant checking but are deliberately excluded from the persistent current-task \(v_i\) store.

---

# 4. Common Stage C — Attribution invariant

Before Exp5a is permitted to construct a coreset, the code verifies that its trajectory decomposition reproduces the actual classifier-head update.

For each communication round it reconstructs

$$
\widehat{\Delta\theta}_a
=
\Delta\theta^{\text{current}}_a
+
\Delta\theta^{\text{replay}}_a
+
\Delta\theta^{\text{wd}}_a
$$

and compares it with the actual FedAvg head delta

$$
\Delta\theta_a.
$$

The normalized error is

$$
E_a
=
\frac{
\|\widehat{\Delta\theta}_a-\Delta\theta_a\|_2
}{
\|\Delta\theta_a\|_2+\epsilon
}.
$$

It also aggregates the reconstruction across the complete task and computes

$$
E_{\text{task}}
=
\frac{
\|\widehat{\Delta\theta}_{\text{task}}
-\Delta\theta_{\text{task}}\|_2
}{
\|\Delta\theta_{\text{task}}\|_2+\epsilon
}.
$$

The task passes only when both

$$
E_{\text{task}}\le \tau
$$

and

$$
\max_a E_a\le\tau.
$$

If the invariant fails, both `exp5a_local_r` and `exp5a_global_r` refuse to run the coreset constructor.

After this point, the two methods diverge.

---

# 5. Algorithm 1 — `exp5a_local_r`

## Goal

Compress each client's \(D\)-dimensional trajectory geometry independently into a rank-\(r\) space and select samples that reconstruct that client's trajectory target in its own local subspace.

---

## 5.1 Construct the local trajectory matrix

For client \(k\), gather its trajectory vectors:

$$
X_k
=
\begin{bmatrix}
v_{k,1}^{\top}\\
\vdots\\
v_{k,N_k}^{\top}
\end{bmatrix}
\in\mathbb R^{N_k\times D}.
$$

Require

$$
r\le \min(N_k,D).
$$

---

## 5.2 Learn a client-specific trajectory basis

Set randomized-PCA working rank

$$
q_k
=
\min(r+p,N_k,D).
$$

Run **uncentered randomized PCA**

$$
X_k
\approx
U_k\Sigma_kV_k^\top
$$

using `torch.pca_lowrank(..., center=False)`.

Retain

$$
R_k=V_k[:,1:r]
\in\mathbb R^{D\times r}.
$$

The implementation uses a deterministic task/client-dependent random seed for this PCA.

---

## 5.3 Project local trajectories

For every current-task sample on client \(k\),

$$
z_{k,i}
=
v_{k,i}R_k.
$$

Equivalently,

$$
Z_k=X_kR_k
\in\mathbb R^{N_k\times r}.
$$

The global projected matrix \(Z\in\mathbb R^{N\times r}\) merely stores these vectors in candidate-row order.

Crucially,

$$
R_k\neq R_j
$$

in general for \(k\neq j\).

Therefore the coordinates of two clients live in **different rank-\(r\) coordinate systems**. A global sum such as

$$
\sum_k z_k
$$

has no valid geometric interpretation. The implementation explicitly recognizes this and reports only mean per-client projected reconstruction for `local_r`.

---

## 5.4 Measure retained trajectory energy

For client \(k\),

$$
\rho_k
=
\frac{
\sum_{j=1}^{r}\sigma_{k,j}^{2}
}{
\|X_k\|_F^2+\epsilon
}.
$$

The algorithm logs every \(\rho_k\) and their mean as diagnostics.

---

# 6. Local projected replay selection

Let

$$
M=\texttt{gdr\_task\_budget}.
$$

Use the same FedCBDR-style per-client allocation as Exp4.5:

$$
B=\left\lfloor\frac{M}{K}\right\rfloor
$$

base samples per client and

$$
R=M-KB
$$

remainder slots.

There is **no class quota**. Selection is unique and without replacement in Exp5a.

---

## 6.1 Construct each client's reconstruction target

For client \(k\), using its projected vectors \(Z_k\), the default `budget_scaled_sum` target is

$$
T_k^{(z)}
=
\frac{B}{N_k}
\sum_{i=1}^{N_k}z_{k,i}.
$$

If `full_sum` mode is explicitly selected instead,

$$
T_k^{(z)}
=
\sum_i z_{k,i}.
$$

The budget-scaled target is the implementation's default target convention.

Initialize residual

$$
r_k^{(0)}=T_k^{(z)}.
$$

---

## 6.2 Fill the hard base quota

For selection step

$$
m=1,\ldots,B,
$$

visit all clients in round-robin order.

For every still-unselected candidate \(j\) of client \(k\), compute

$$
c_{k,j}
=
\frac{
\|r_k-z_{k,j}\|_2^2
}{
\|T_k^{(z)}\|_2^2+\epsilon
}.
$$

The code evaluates this efficiently as

$$
c_{k,j}
=
\frac{
\|r_k\|^2+\|z_{k,j}\|^2
-2z_{k,j}^{\top}r_k
}{
\|T_k^{(z)}\|^2+\epsilon
}.
$$

Select

$$
j^\star
=
\arg\min_{j\notin S_k}c_{k,j}
$$

and update

$$
S_k\leftarrow S_k\cup\{j^\star\},
$$

$$
r_k\leftarrow r_k-z_{k,j^\star}.
$$

After this stage,

$$
|S_k|=B
$$

for every client.

---

## 6.3 Allocate remainder slots

If \(R>0\), repeat \(R\) times:

1. For each client, find its currently cheapest remaining candidate

$$
j_k^\star
=
\arg\min_{j\notin S_k}
\frac{\|r_k-z_{k,j}\|^2}
{\|T_k^{(z)}\|^2+\epsilon}.
$$

2. Compare those best costs across all clients.
3. Pick the globally smallest `(cost, client_id, position)` tuple.
4. Give that client one additional replay slot.
5. Update its residual.

Thus final client quota is

$$
B_k=B+e_k,
\qquad
\sum_k e_k=R.
$$

**Implementation detail:** during remainder allocation, the residual still corresponds to the original base-quota target \(B/N_k\sum_i z_i\). The target is not rescaled before choosing extra items. Only the final diagnostic target is recomputed using the actual final \(B_k\). This does not matter when \(M\) is divisible by \(K\).

---

# 7. Evaluate Exp5a-Local selection

For each client, after final quota \(B_k\) is known, define

$$
\widetilde T_k^{(z)}
=
\frac{B_k}{N_k}
\sum_i z_{k,i}.
$$

Projected-space reconstruction error is

$$
E_k^{(z)}
=
\frac{
\left\|
\sum_{i\in S_k}z_{k,i}
-
\widetilde T_k^{(z)}
\right\|_2
}{
\|\widetilde T_k^{(z)}\|_2+\epsilon
}.
$$

The local variant reports

$$
\boxed{
E_{\text{local-z}}
=
\frac1K\sum_kE_k^{(z)}
}
$$

rather than a global \(z\)-space error because the bases differ across clients.

## The common greedy selector also records client counts, class histogram, HHI, uniqueness and reconstruction diagnostics.

## 7.1 Evaluate in the original trajectory space

Selection is performed in \(r\)-dimensional projected space, but the algorithm also asks whether those selected items reconstruct the original \(D\)-dimensional trajectory.

For client \(k\),

$$
T_k^{(v)}
=
\frac{B_k}{N_k}
\sum_i v_{k,i},
$$

and

$$
\widehat T_k^{(v)}
=
\sum_{i\in S_k}v_{k,i}.
$$

Client original-space error:

$$
E_k^{(v)}
=
\frac{
\|\widehat T_k^{(v)}-T_k^{(v)}\|_2
}{
\|T_k^{(v)}\|_2+\epsilon
}.
$$

Global original-space reconstruction is

$$
E_{\mathrm{global}}^{(v)}
=
\frac{
\left\|
\sum_k\widehat T_k^{(v)}
-
\sum_kT_k^{(v)}
\right\|_2
}{
\left\|\sum_kT_k^{(v)}\right\|_2+\epsilon
}.
$$

This is explicitly computed after projected-space selection.

---

# 8. Algorithm 2 — `exp5a_global_r`

The federated training, trajectory-vector construction, attribution invariant, replay budget, target construction and greedy selection are identical.

Only the construction of \(z_i\) changes.

The purpose is to make all clients describe their trajectories in **one shared global rank-\(r\) coordinate system**.

---

## 8.1 Gather client trajectory matrices conceptually

Again,

$$
X_k\in\mathbb R^{N_k\times D}.
$$

Conceptually, without masking,

$$
X=
\begin{bmatrix}
X_1\\
X_2\\
\vdots\\
X_K
\end{bmatrix}
\in\mathbb R^{N\times D}.
$$

Instead of running PCA directly on \(X\), the implementation constructs a masked version.

---

# 9. Global orthogonal coordination

## 9.1 Construct common feature-space orthogonal mask

Generate

$$
Q\in\mathbb R^{D\times D},
$$

with

$$
Q^\top Q=QQ^\top=I.
$$

`Mine.py` implements this without explicitly materializing a dense \(D\times D\) matrix. `_ImplicitOrthogonalMask` composes multiple random pairwise Givens rotations.

The same \(Q\) is used by every client.

---

## 9.2 Construct client-specific sample-space masks

For each client \(k\), generate another orthogonal operator

$$
P_k\in\mathbb R^{N_k\times N_k},
$$

satisfying

$$
P_k^\top P_k=P_kP_k^\top=I.
$$

---

## 9.3 Mask each trajectory matrix

Client \(k\)'s masked contribution matrix is

$$
X'_k
=
P_k X_k Q.
$$

The code then vertically pools all masked client blocks:

$$
X'
=
\begin{bmatrix}
X'_1\\
X'_2\\
\vdots\\
X'_K
\end{bmatrix}
\in\mathbb R^{N\times D}.
$$

This is implemented client-by-client so the entire set of separate client matrices need not coexist as additional copies in memory.

---

## 9.4 Why the two orthogonal masks preserve the desired geometry

Define

$$
P
=
\operatorname{blockdiag}(P_1,\ldots,P_K).
$$

Then

$$
X'=PXQ.
$$

Because \(P\) is orthogonal,

$$
X'^\top X'
=
Q^\top X^\top P^\top PXQ
=
Q^\top X^\top XQ.
$$

Therefore:

* \(P_k\) mixes rows but does not alter the global singular spectrum;
* \(Q\) merely rotates the feature/trajectory coordinates;
* global low-rank structure is preserved up to an orthogonal change of basis.

This is why PCA can be performed on \(X'\) while retaining the global trajectory subspace.

The current implementation describes this masking as a **simulation** of FedCBDR-style orthogonally masked coordination, not as a formal privacy mechanism: masks and raw vectors coexist in the same process.

---

# 10. Learn the global trajectory subspace

Require

$$
r\le \min(N,D).
$$

Choose

$$
q=\min(r+p,N,D).
$$

Run uncentered randomized PCA:

$$
X'
\approx
U'\Sigma'V'^\top.
$$

Take

$$
R'_r
=
V'[:,1:r]
\in\mathbb R^{D\times r}.
$$

The code logs retained global energy

$$
\rho_{\mathrm{global}}
=
\frac{
\sum_{j=1}^{r}{\sigma'_j}^2
}{
\|X'\|_F^2+\epsilon
}.
$$

---

# 11. Project every client into the shared global subspace

The row mask \(P_k\) is needed only for constructing the pooled masked SVD.

For actual individual candidate embeddings, client \(k\) computes

$$
\boxed{
Z_k=(X_kQ)R'_r
}
$$

so each sample receives

$$
\boxed{
z_{k,i}
=
(v_{k,i}Q)R'_r.
}
$$

The implementation therefore returns to the un-row-mixed client matrix, applies the common \(Q\), and projects onto \(R'_r\).

All clients now share the same coordinate system.

In exact arithmetic, if \(R_r\) denotes the leading right singular vectors of the unmasked global matrix \(X\), the orthogonal transformation implies approximately

$$
R'_r\approx Q^\top R_r,
$$

and hence

$$
(X_kQ)R'_r
\approx
X_kQQ^\top R_r
=
X_kR_r.
$$

Thus `exp5a_global_r` is effectively performing replay selection in a **global trajectory PCA subspace**, while the masking transforms the representation used for cross-client coordination.

---

# 12. Global projected replay selection

Once \(Z\) has been produced, the selection algorithm is **the exact same FedCBDR-style per-client greedy solver used by Exp5a-Local**:

$$
B=\left\lfloor\frac{M}{K}\right\rfloor,
\qquad
R=M-KB.
$$

For every client:

$$
T_k^{(z)}
=
\frac{B}{N_k}\sum_i z_{k,i},
$$

and initialize

$$
r_k=T_k^{(z)}.
$$

For each base selection:

$$
j^\star
=
\arg\min_{j\notin S_k}
\frac{
\|r_k-z_{k,j}\|^2
}{
\|T_k^{(z)}\|^2+\epsilon
},
$$

then

$$
r_k\leftarrow r_k-z_{k,j^\star}.
$$

## Fill \(B\) items per client, then allocate the \(R\) remainder slots globally according to the smallest normalized residual cost. Selection is without replacement and imposes no class constraint.

# 13. Evaluate Exp5a-Global

Because every \(z_i\) now uses one shared coordinate system, both client-level and global projected reconstruction have geometric meaning.

For final client quota \(B_k\),

$$
\widetilde T_k^{(z)}
=
\frac{B_k}{N_k}\sum_i z_{k,i},
$$

$$
\widehat T_k^{(z)}
=
\sum_{i\in S_k}z_{k,i}.
$$

The global projected-space error is

$$
\boxed{
E_{\mathrm{global}}^{(z)}
=
\frac{
\left\|
\sum_k\widehat T_k^{(z)}
-
\sum_k\widetilde T_k^{(z)}
\right\|_2
}{
\left\|
\sum_k\widetilde T_k^{(z)}
\right\|_2+\epsilon
}.
}
$$

Unlike Local-r, this global quantity is meaningful because every client uses \(R'_r\).

The code therefore logs `global_z_recon_error` for Global-r, whereas Local-r logs `mean_z_recon_error`.

The same selected indices are also evaluated in the original \(D\)-dimensional trajectory space to produce

$$
E_{\mathrm{global}}^{(v)}.
$$

---

# 14. Common Stage D — Convert selected candidate rows into replay memory

After either Local-r or Global-r returns the final set

$$
S=\bigcup_kS_k,
\qquad |S|=M,
$$

the selected global candidate rows are mapped back to

$$
(\text{client ID},\text{client-local sample ID},\text{label}).
$$

For every client:

1. collect its selected local IDs;
2. verify uniqueness;
3. sort the IDs;
4. instantiate a `ReplayDataset`;
5. use sampling weight \(1\) for every replay item;
6. append that dataset to

$$
\texttt{retained\_ds\_all[k]}.
$$

Consequently, at task \(t+1\), client \(k\) trains on

$$
\mathcal D_{t+1,k}
\cup
\mathcal R_{0,k}
\cup\cdots\cup
\mathcal R_{t,k}.
$$

After constructing the replay dataset, the current task's \(N\times D\) trajectory matrix is released.

---

# 15. Complete pseudocode — Exp5a-Local

```text
ALGORITHM EXP5A_LOCAL_R

INPUT:
    current global model θ
    task-t current dataset
    previous replay buffers R[1...K]
    K clients
    task replay budget M
    projection rank r
    PCA oversampling p
    communication rounds A
    local epochs E
    SGD momentum μ

FOR task t:

    1. Partition current task among K clients.
    2. Expand classifier to all classes seen through task t.
    3. Let D = C_t * d + C_t.
    4. Create one candidate row for every current-task sample.
    5. Initialize v_i = 0 ∈ R^D for every current-task candidate i.

    FOR communication round a = 1 ... A:

        select clients K_a
        FedAvg coefficient p_k = 1 / |K_a|

        FOR each selected client k:

            local_dataset =
                current_task_data[k]
                ∪ all_previous_replay[k]

            clone global model
            S = number of local SGD steps

            FOR local SGD step s = 1 ... S:

                W_s = (1 - μ^(S-s+1)) / (1-μ)

                run forward pass
                compute actual training loss
                q_i = ∂L / ∂logits_i
                h_i = backbone feature

                FOR each CURRENT-TASK sample i in batch:

                    g_i =
                        concat(
                            vec(q_i h_i^T),
                            q_i
                        )

                    v_i += -η_a * p_k * W_s * g_i

                record full current+replay trace for invariant
                perform normal SGD optimizer step

        FedAvg all selected local models
        verify reconstructed head delta against actual head delta

    verify task-level attribution invariant
    ABORT selection if invariant fails

    ------------------------------------------------
    LOCAL SUBSPACE
    ------------------------------------------------

    FOR client k:

        X_k = matrix of client-k current trajectory vectors

        require r <= min(N_k, D)
        q = min(r + p, N_k, D)

        [U_k, S_k, V_k] =
            randomized_uncentered_PCA(X_k, q)

        R_k = first r columns of V_k

        Z_k = X_k R_k

        log retained energy

    ------------------------------------------------
    FEDCBDR-STYLE PER-CLIENT GREEDY SELECTION
    ------------------------------------------------

    B = floor(M / K)
    R = M - B*K

    FOR client k:

        T_k = (B/N_k) Σ_i z_ki
        residual_k = T_k
        selected_k = ∅

    FOR selection_step = 1 ... B:

        FOR client k:

            j* = argmin over unselected j of
                 ||residual_k - z_kj||² /
                 (||T_k||² + ε)

            selected_k += j*
            residual_k -= z_kj*

    FOR each of R remainder slots:

        FOR every client k:
            find its cheapest currently available candidate

        choose globally cheapest
        add it to that client
        update that client's residual

    verify exactly M unique samples selected

    ------------------------------------------------
    DIAGNOSTICS
    ------------------------------------------------

    compute per-client projected reconstruction
    mean these values across clients

    DO NOT compute a global projected-vector reconstruction
    as a geometric metric, because R_k differs by client

    evaluate same selected samples in original D-space
    compute global original-space reconstruction

    compute client histogram, class histogram, max share, HHI

    ------------------------------------------------
    REPLAY UPDATE
    ------------------------------------------------

    map selected candidate rows to client-local IDs

    FOR client k:
        create ReplayDataset from selected current-task samples
        append it to retained_ds_all[k]

    discard current task trajectory matrix
```

---

# 16. Complete pseudocode — Exp5a-Global

```text
ALGORITHM EXP5A_GLOBAL_R

INPUT:
    same inputs as EXP5A_LOCAL_R
    number of orthogonal mask layers L
    mask seed

FOR task t:

    ------------------------------------------------
    IDENTICAL FEDERATED TRAINING / ATTRIBUTION STAGE
    ------------------------------------------------

    construct current-task v_i ∈ R^D exactly as in EXP5A_LOCAL_R

    verify round-level and task-level trajectory invariant
    ABORT selection if invariant fails

    ------------------------------------------------
    GLOBAL MASKED SUBSPACE COORDINATION
    ------------------------------------------------

    construct common D-dimensional orthogonal mask Q

    allocate pooled matrix X' ∈ R^(N×D)

    FOR client k:

        X_k = client-k trajectory matrix

        construct N_k-dimensional orthogonal row mask P_k

        X'_k = P_k X_k Q

        place X'_k into corresponding block of X'

    require r <= min(N, D)

    q = min(r + p, N, D)

    [U', S', V'] =
        randomized_uncentered_PCA(X', q)

    R'_r = first r columns of V'

    log global retained energy

    discard pooled X'

    ------------------------------------------------
    PROJECT ORIGINAL CLIENT VECTORS
    ------------------------------------------------

    FOR client k:

        X_k = original trajectory matrix

        Z_k = (X_k Q) R'_r

    Now every z_i belongs to the SAME R^r coordinate system.

    ------------------------------------------------
    FEDCBDR-STYLE PER-CLIENT GREEDY SELECTION
    ------------------------------------------------

    B = floor(M / K)
    R = M - B*K

    FOR client k:

        T_k = (B/N_k) Σ_i z_ki
        residual_k = T_k
        selected_k = ∅

    FOR selection_step = 1 ... B:

        FOR client k:

            j* = argmin over unselected j of
                 ||residual_k - z_kj||² /
                 (||T_k||² + ε)

            selected_k += j*
            residual_k -= z_kj*

    FOR each remainder slot:

        find the cheapest available candidate for every client
        select the globally cheapest candidate
        assign that slot to its client
        update that client's residual

    verify exactly M unique samples selected

    ------------------------------------------------
    DIAGNOSTICS
    ------------------------------------------------

    compute per-client z-space reconstruction

    because all clients share the same basis:
        compute GLOBAL z-space reconstruction

    evaluate selected samples again in original D-space:
        compute global original-space reconstruction

    compute client histogram, class histogram,
    max client share and HHI

    ------------------------------------------------
    REPLAY UPDATE
    ------------------------------------------------

    FOR client k:
        map selected rows to local sample IDs
        create ReplayDataset
        append to retained_ds_all[k]

    discard current task trajectory matrix
```

---

# 17. Exact conceptual difference

| Component                                     | Exp5a-Local-r                       | Exp5a-Global-r                     |
| --------------------------------------------- | ----------------------------------- | ---------------------------------- |
| Original signal                               | Exact trajectory vector \(v_i\)     | Exact trajectory vector \(v_i\)    |
| Training procedure                            | Same                                | Same                               |
| Attribution invariant                         | Same                                | Same                               |
| PCA input                                     | Each \(X_k\) independently          | Pooled masked \(X'=PXQ\)           |
| Basis                                         | \(R_k\), different for every client | One \(R'_r\) shared by all clients |
| Projected vector                              | \(z_i=v_iR_k\)                      | \(z_i=(v_iQ)R'_r\)                 |
| Cross-client structure used to learn basis    | **No**                              | **Yes**                            |
| Per-client replay allocation                  | Same                                | Same                               |
| Greedy reconstruction solver                  | Same                                | Same                               |
| Hard class balancing                          | No                                  | No                                 |
| Duplicate selection                           | No                                  | No                                 |
| Global \(z\)-space reconstruction meaningful? | **No**                              | **Yes**                            |
| Original \(D\)-space reconstruction measured  | Yes                                 | Yes                                |
| Replay training afterward                     | Same                                | Same                               |

Hence the clean experimental interpretation is

$$
\boxed{
\text{Exp5a-Local}
=
\text{Mine trajectory selection}
+
\text{client-local low-rank compression}
}
$$

whereas

$$
\boxed{
\text{Exp5a-Global}
=
\text{Mine trajectory selection}
+
\text{globally coordinated low-rank trajectory geometry}.
}
$$

The experiment therefore isolates a very specific question:

> **Does Mine underperform because each client's trajectory vectors lack a shared estimate of the globally important contribution directions?**

`exp5a_local_r` controls for the benefit of simple dimensionality reduction. `exp5a_global_r` adds the missing cross-client subspace coordination while keeping the underlying trajectory signal, replay allocation, target construction, and greedy selector essentially unchanged.