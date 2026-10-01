# FedCBDR historical-class interaction study

This pipeline is specific to `--method=fedcbdr`. It studies directed old-old
class interactions while keeping old-to-current errors visible as a confound.
Use CIFAR-100, five clients and five 20-class tasks for the main experiment.

## Baseline and exact transition snapshots

Run the normal five-task FedCBDR baseline once with
`--fedcbdr_monitor_dir store/fedcbdr_interactions --fedcbdr_probe_per_class 32`.
Each seed writes a fresh `seed_<seed>` directory; an existing directory is
rejected. The main run saves `transition_start_T2.pkl` and
`transition_start_T4.pkl` immediately before communication round 0, after
classifier expansion and the actual client partition. Each snapshot contains
the expanded model state, class order and task ranges, current training and
test datasets, frozen probes, replay exemplars and weights, client partition,
arguments, and Python/NumPy/Torch/CUDA RNG states. It also carries the prior
selected-model boundary metrics, feature statistics and Torch backend settings.
For snapshots made before backend settings were saved, the runner reapplies
`setup_seed` from the saved arguments before restoring RNG tensors. The experiment runner
restores these states rather than repartitioning or rerunning GDR. It never
continues the baseline sequence from an intervention model.

The main run still saves `checkpoint_<T>.pkl` after selecting the best model and
constructing the next replay buffer. These are task-end artifacts. Snapshot
files contain pickled Python objects; load only files produced by a trusted run.
Image paths in path-backed datasets must remain available.

## Observational artifacts

`metrics.jsonl` stores one `class_probe` record per class and round. A
`selected_model` event records which communication round supplied the selected
model. Accuracy is
a fraction, CE uses ordinary logits, and global margin is true-class logit
minus the largest other logit. Its confusion counts separate same-old-task,
different-old-task and old-to-current errors. `fedavg_by_class` contains
`global_before`, `local_after`, `global_after`, per-client local deltas, and
the difference between the post-FedAvg class metric and the uniform mean of
client-local class metrics. The latter matches this implementation's uniform
model aggregation, although model averaging and metric averaging are distinct
operations. `update_mass` and `exposure_summary` retain descriptive training
signal and exposure summaries.

At T2 and T4, `pair_metrics_T<T>_R<R>.pt` contains **every directed old-old
pair**. For functional/feature matrices, row B is the victim and column A is
the possible aggressor. For exposure matrices, row A is the earlier class and
column B is the later class. Off-diagonal `pair_type` is 0 for intra-task and
1 for inter-task. `pair_confusion_rate[B,A]` is P(pred=A | true=B),
`pair_margin[B,A]` is mean(logit_B-logit_A), and `delta_pair_margin` is
relative to the previous selected task boundary. `feature_direction` projects
B's centroid movement toward A; `relative_feature_direction` projects B's
movement minus A's movement onto the same boundary direction. Missing
boundary comparisons are NaN in tensors and null in JSON, never zero.

Exposure arrays have shape `[num_clients, old_classes, old_classes]` or
`[num_clients, old_classes]`, with summed `round_*` arrays. A client contributes
only its own local update history. Histories reset each local epoch. Co-batch
counts are optimizer steps with both replay classes. Sequential exposure
counts earlier A-bearing updates before later B-bearing updates; lag uses the
last A-bearing update. Zero-exposure pairs are retained. CE mass is the sum of
unreduced TTS-scaled sample CE before replay/task weighting. It is a loss
magnitude proxy, not a gradient norm or exact optimizer contribution.

`features_t<T>_r<R>.pt` stores raw probe features and labels compactly; means
and full sample covariances can be reconstructed from them. The logger uses
full covariance internally for drift metrics. Pair summaries in JSON are
descriptive; the tensor files remain the complete source.

## All-pair export and predefined candidate selection

After completing the full baseline sequence, run:

```text
python analyze_fedcbdr_interactions.py \
  store/fedcbdr_interactions/seed_2023 \
  store/fedcbdr_interactions/seed_2023/transition_start_T4.pkl \
  store/fedcbdr_T4_analysis --replay-strength-control
```

Repeat for T2. The output contains `pairs_T4.jsonl` with one record for every
directed old-old pair, `summary.json` with separate intra/inter counts and
descriptive means, and `ablation_spec.json`. Exposure is summed only through
the selected model's source round; functional outcomes use that same model.
`delta_P_B_to_A` compares
the selected model with the first communication round because a boundary
confusion matrix is not available. It must not be described as boundary change.
The pair table includes task age, distance, replay draws, CE mass, clients
exposing each class, co-batch/sequential exposure, lag, victim accuracy/CE,
margin, confusion and feature direction.

The deterministic candidate rule selects high-signal, random stratified and
exposed negative-control pairs separately within intra/inter categories. The
random strata use task distance, A/B age, exposure and boundary margin. High
signal requires actual sequential exposure, a negative boundary-relative
pair-margin change, and increased post-round-0 confusion or positive relative
feature movement. Candidates are fixed before intervention training. The
spec lists each A/C arm once even when several selected victims use it.
Controls C must be in A's historical task, differ from A and B, be available
on A-bearing clients, and pass count, boundary accuracy, CE and donor reuse
limits. Multiple controls are selected for high-signal pairs when available.
An empty `C_controls` entry means no matched control passed; it is visible in
the spec and produces no causal arm.

## Paired causal transition suite

Run the generated spec from the matching snapshot:

```text
python run_transition_ablation.py \
  store/fedcbdr_interactions/seed_2023/transition_start_T4.pkl \
  store/fedcbdr_T4_analysis/ablation_spec.json \
  store/fedcbdr_T4_causal
```

The runner preflights all arms, trains one baseline per replay-repeat setting,
and trains one A-to-C arm per unique pair. Each arm starts from the same model,
partition, replay snapshot and RNG. A slots retain position and replay weight;
their image/label is replaced by a deterministic, client-local C donor from
A's task. Minibatch order, client schedule, augmentation draws, optimizer and
FedAvg remain paired. C donor counts and maximum/mean reuse are recorded in
`manifest.json`; multiplicity includes existing C replay draws and the new A
slots. Excessive reuse rejects the arm. No GDR reselection occurs.

Every arm evaluates all historical B other than A and C. The shared baseline
is used for all A/C comparisons within one replay-repeat setting.
`causal_effects.jsonl` contains delta victim accuracy, CE, global margin,
B-to-A pair margin/confusion and relative feature direction at the same final
communication round and at each arm's selected model. The fixed-round
comparison avoids a difference caused solely by selecting different best
rounds. A positive
accuracy and negative CE change are the primary functional evidence; geometric
changes explain the mechanism. One outcome need not have every metric improve.
For repeat 1, `reproduction_audit.json` checks the replayed baseline against
the original selected-model class metrics and selected source round. It stops
the suite if the source round differs or the maximum accuracy or CE gap exceeds
`--max-baseline-drift` (default 0.01).

The optional `replay_repeats: [1, 2]` runs a replay-strength control from the
same frozen transition. It repeats the same historical replay slots uniformly
without selecting new exemplars. Compare A/C within each strength; the two
strengths have different dataset lengths and update counts, so their absolute
outcomes are not a paired single-factor A/C effect.

For discovery, report effect sizes and distributions separately for T2 and T4.
Thousands of pair rows share classes, tasks and seed and are not independent
replicates. Confirm selected A/C arms across additional baseline seeds before
making population claims. Full gradient matrices, MIR updates, gradient
projection and minibatch isolation are outside this study.

## Verification and execution limits

`python -m unittest discover -s tests -p test_fedcbdr_monitoring.py` exercises
pair directions, exposure locality and donor replacement when PyTorch is
installed. Training requires the repository's normal CUDA, TorchVision and
dataset dependencies; syntax checks alone do not establish runtime performance
or scientific effects.
