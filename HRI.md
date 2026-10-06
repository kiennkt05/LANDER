# HRI Experiment Algorithm

1. **Train baseline FedCBDR**
   - CIFAR100, 5 tasks, 5 clients, $\beta = 0.1$, seed=2023.
   - Save transition checkpoints before **T2** and **T4**.
   - Build fixed probe set for every class.

2. **Measure observational HRI**
   For every historical pair $A \rightarrow B$:
   - Pair margin:
     $$
     M_{B,A} = \mathbb{E}[z_B - z_A]
     $$
   - Margin change:
     $$
     \Delta M_{B,A} = M_{B,A}^{\mathrm{after}} - M_{B,A}^{\mathrm{before}}
     $$
   - $B \rightarrow A$ confusion rate.
   - Feature drift of $B$ toward $A$.
   - Replay exposure statistics.

3. **Select candidate pairs**
   - Prioritize large negative $\Delta M_{B,A}$.
   - Use confusion/drift/exposure as supporting evidence.
   - Require feasible same-task replacement class $C$.

4. **Construct causal intervention**
   For candidate $A \rightarrow B$:
   - Baseline: replay original $A$.
   - Intervention: replace replay slots of $A$ with $C$.
   - Keep fixed:
     - checkpoint
     - replay budget/weights
     - clients
     - batches/order
     - optimizer steps
     - RNG/augmentations.

5. **Rerun the transition**
   Run baseline and replacement branches deterministically from the same checkpoint.

6. **Measure causal effect on victim $B$**
   $$
   \begin{aligned}
   \Delta \mathrm{Acc}_B &= \mathrm{Acc}_B^{\mathrm{rep}} - \mathrm{Acc}_B^{\mathrm{base}} \\
   \Delta \mathrm{CE}_B &= \mathrm{CE}_B^{\mathrm{rep}} - \mathrm{CE}_B^{\mathrm{base}} \\
   \Delta \mathrm{Margin}_B &= \mathrm{Margin}_B^{\mathrm{rep}} - \mathrm{Margin}_B^{\mathrm{base}}
   \end{aligned}
   $$

7. **Decide HRI**
   Causal HRI if replacement gives:
   $$
   \Delta \mathrm{Acc}_B > 0, \quad
   \Delta \mathrm{CE}_B < 0, \quad
   \Delta \mathrm{Margin}_B > 0
   $$

8. **Replay-strength control**
   Repeat intervention with stronger replay ($2\times$) to test whether HRI is stable or trajectory-dependent.

9. **Final interpretation**
   - T2: causal positives **$12 \rightarrow 21$, $14 \rightarrow 21$**.
   - **$14 \rightarrow 27$**: observational false positive.
   - T4 **$23 \rightarrow 44$**: negative/control case.