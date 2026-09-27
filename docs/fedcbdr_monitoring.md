# FedCBDR baseline monitoring (Phases 0, 1, 2a)

Add `--fedcbdr_monitor_dir store/probe_experiment --fedcbdr_probe_per_class 32`
to an existing FedCBDR training command. Each seed writes a fresh `seed_<seed>`
directory. Existing directories are rejected to prevent mixing experiments.
Omitting the monitoring directory disables round monitoring. Task registry,
test probe selection, task tags and task-end checkpoints are still enabled.

Outputs:

- `metrics.jsonl`: versioned events for class probes, class pairs, drift summaries,
  before/local/after FedAvg accuracy, actual training draws and CE mass, global
  accuracy, and GDR diagnostics. All events include seed, task, round and variant.
- `features_t<T>_r<R>.pt`: per-class mean and full sample covariance tensors.
- `checkpoint_<T>.pkl`: selected best model, task ranges, probe indices, arguments
  including class order, current client partition, and all historical replay
  buffers. Replay snapshots include duplicates, local and task-dataset indices,
  weights, labels and image data (or source paths for path-backed datasets).
  Without monitoring these go under `<save_dir>/fedcbdr_seed_<seed>`.

Indices for probes refer to the single-class test dataset, in remapped class
order, with the same dataset version and deterministic test transform. Probes
never come from replay. Checkpoints are task-end snapshots, not a mid-round
training resume interface. Path-backed image files must remain available.

Metric definitions:

- Probe accuracy is a fraction; existing global accuracies retain their percent
  units. CE uses ordinary logits; margin is true-class logit minus the largest
  other logit. Confusion fields are error counts with `count` as denominator.
- Pair distance is Euclidean distance between feature centroids. Covariance is
  stored separately; it is not part of this distance. Mean/covariance drift is
  relative to the preceding probe event. Boundary deltas compare against the
  preceding task's selected best model, only averaging historical pairs for
  intra/inter summaries. Missing comparisons are null, never zero.
- `draws` counts actual exposures across participating clients and local epochs,
  including duplicate replay draws. `ce_mass_proxy = draws * mean_ce` uses the
  unreduced TTS-scaled CE before replay/task weighting; task 0 uses ordinary CE.
  It is a loss-magnitude proxy, not a gradient norm or exact optimizer influence.
  Group by `class_task_id` to obtain task mass.
- `selected` round events describe the restored best model, which can differ
  from the final communication round. GDR diagnostics describe the buffer
  selected after that task, not the preceding round's training exposures.

Evaluation preserves RNG state and module training flags. Full covariance
storage and per-client forward passes have nonzero overhead; measure it on a
short run before launching all seeds. No selector, sampler or objective
reweighting is added in this phase.

Validation: `python -m unittest discover -s tests -p test_fedcbdr_monitoring.py`.
Numerical tests require PyTorch; the existing learner also requires CUDA and
its normal torchvision/data dependencies.

Next gate: inspect baseline class degradation, pair distance deltas and mass
across seeds, then choose transitions for causal ablations. Replay variants,
matched pseudo-task permutation analysis, and gradient probes remain deferred
until this evidence exists, following the requested sequencing.
