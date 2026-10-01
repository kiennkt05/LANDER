"""Paired FedCBDR A->C transition interventions from an exact start snapshot.

Run only on trusted transition snapshots produced by this repository: they
contain pickled datasets and transforms.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from methods.fedcbdr import FedCBDR, ReplayDataset
from utils.data_manager import setup_seed
from utils.fedcbdr_interactions import replacement_slots
from utils.metrics_logger import MetricsLogger
from utils.task_registry import TaskRegistry


def load_snapshot(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"task", "known_classes", "total_classes", "model_state_dict",
                "task_class_ranges", "class_order", "retained_ds_all", "probes",
                "train_dataset", "test_dataset", "old_dataset", "new_dataset",
                "user_groups", "python_rng", "numpy_rng", "torch_rng",
                "cuda_rng", "args"}
    missing = required - state.keys()
    if missing:
        raise ValueError("incomplete transition snapshot: {}".format(sorted(missing)))
    if list(state["args"]["class_order"]) != list(state["class_order"]):
        raise ValueError("snapshot class order does not match arguments")
    return state


def restore_rng(state):
    # The main entry point calls setup_seed before every task. Besides RNGs it
    # configures cuDNN's algorithm selection; restoring RNG tensors alone
    # leaves a fresh ablation process with different convolution behavior.
    args = state["args"]
    setup_seed(int(args["seed"]), fast_cuda=bool(args.get("fast_cuda", False)))
    backends = state.get("torch_backend_state", {})
    if "deterministic_algorithms" in backends:
        torch.use_deterministic_algorithms(backends["deterministic_algorithms"])
    if "float32_matmul_precision" in backends:
        torch.set_float32_matmul_precision(backends["float32_matmul_precision"])
    for name in ("deterministic", "benchmark", "enabled", "allow_tf32"):
        if name in backends:
            setattr(torch.backends.cudnn, name, backends[name])
    if "cuda_matmul_allow_tf32" in backends:
        torch.backends.cuda.matmul.allow_tf32 = backends["cuda_matmul_allow_tf32"]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.random.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available():
            raise RuntimeError("snapshot has CUDA RNG state but CUDA is unavailable")
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA device visibility differs from the transition snapshot")
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def make_loader(dataset, args):
    workers = int(args["num_worker"])
    return DataLoader(dataset, batch_size=256, shuffle=False,
                      num_workers=workers, pin_memory=True,
                      multiprocessing_context=args["mulc"] if workers else None,
                      persistent_workers=workers > 0)


def run_arm(state, replay, directory, name, repeat):
    args = copy.deepcopy(state["args"])
    args["fedcbdr_monitor_dir"] = None
    args["fedcbdr_replay_repeat"] = int(repeat)
    args.setdefault("fedcbdr_lr_scheduler", args.get("fedcbdr_lr_schedule", "constant"))
    learner = FedCBDR(args)
    learner._cur_task = int(state["task"])
    learner._known_classes = int(state["known_classes"])
    learner._total_classes = int(state["total_classes"])
    learner._network.update_fc(learner._total_classes)
    learner._network.load_state_dict(state["model_state_dict"])
    learner.task_registry = TaskRegistry(state["task_class_ranges"])
    learner.task_class_ranges = learner.task_registry.ranges
    learner.probes = state["probes"]
    learner.train_dataset = state["train_dataset"]
    learner.user_groups = copy.deepcopy(state["user_groups"])
    learner.retained_ds_all = [
        [ReplayDataset.from_snapshot(row, learner.train_dataset.trsf)
         for row in client] for client in replay
    ]
    learner.test_loader = make_loader(state["test_dataset"], args)
    learner.old_loader = make_loader(state["old_dataset"], args)
    learner.new_loader = make_loader(state["new_dataset"], args)
    learner.metrics = MetricsLogger(directory, learner.seed, variant=name,
                                    pair_tasks=(learner._cur_task,))
    learner.metrics.previous = copy.deepcopy(state.get("metrics_previous", {}))
    learner.metrics.boundary = copy.deepcopy(state.get("metrics_boundary", {}))
    learner.metrics.boundary_class_metrics = copy.deepcopy(
        state.get("boundary_class_metrics", {}))
    learner._ablation_mode = True
    restore_rng(state)
    learner._fl_train(learner.train_dataset, learner.test_loader,
                      frozen_partition=True)
    return learner.metrics.directory


def probe_classes(directory, task, round_id="selected"):
    with (directory / "metrics.jsonl").open(encoding="utf-8") as stream:
        return {row["class_id"]: row for line in stream
                if (row := json.loads(line))["event"] == "class_probe"
                and row["task"] == task
                and row["round"] == round_id}


def probe_pairs(directory, task, round_id="selected"):
    return torch.load(directory / "pair_metrics_T{}_R{}.pt".format(
                          task, str(round_id).zfill(3)),
                      map_location="cpu", weights_only=False)


def effects(baseline_dir, arm_dir, task, aggressor, replacement, registry,
            repeat, round_id):
    baseline = probe_classes(baseline_dir, task, round_id)
    arm = probe_classes(arm_dir, task, round_id)
    base_pairs = probe_pairs(baseline_dir, task, round_id)
    arm_pairs = probe_pairs(arm_dir, task, round_id)
    class_ids = base_pairs["class_ids"]
    if class_ids != arm_pairs["class_ids"]:
        raise ValueError("baseline and arm use different historical class orders")
    ai = class_ids.index(aggressor)
    rows = []
    for bi, victim in enumerate(class_ids):
        if victim in (aggressor, replacement):
            continue
        before, after = baseline[victim], arm[victim]
        row = dict(
            transition=task, evaluation_round=round_id,
            replay_repeat=repeat, A=aggressor, B=victim,
            C=replacement, task_A=registry.task_for(aggressor),
            task_B=registry.task_for(victim),
            pair_type="intra" if registry.task_for(aggressor) == registry.task_for(victim) else "inter",
            delta_acc_B=after["acc"]-before["acc"],
            delta_ce_B=after["ce"]-before["ce"],
            delta_global_margin_B=(after["margin"]-before["margin"])
                if before["margin"] is not None else None,
        )
        for field, output in (("pair_margin", "delta_pair_margin_BA"),
                              ("pair_confusion_rate", "delta_confusion_B_to_A"),
                              ("relative_feature_direction", "delta_relative_feature_direction_BA")):
            value = arm_pairs[field][bi, ai] - base_pairs[field][bi, ai]
            row[output] = float(value) if torch.isfinite(value) else None
        rows.append(row)
    return rows


def reproduction_audit(original_dir, replay_dir, task):
    if not (original_dir / "metrics.jsonl").exists():
        return None
    original = probe_classes(original_dir, task)
    replayed = probe_classes(replay_dir, task)
    if not original:
        raise ValueError("original baseline has no selected class probes for this transition")
    if original.keys() != replayed.keys():
        raise ValueError("replayed baseline uses different probe classes")
    def source_round(directory):
        with (directory / "metrics.jsonl").open(encoding="utf-8") as stream:
            rows = [row for line in stream if (row := json.loads(line))["event"] ==
                    "selected_model" and row["task"] == task]
        if len(rows) != 1:
            raise ValueError("expected one selected-model event")
        return rows[0]["source_round"]
    return dict(original_source_round=source_round(original_dir),
                replayed_source_round=source_round(replay_dir),
                max_acc_gap=max(abs(original[c]["acc"]-replayed[c]["acc"])
                                for c in original),
                max_ce_gap=max(abs(original[c]["ce"]-replayed[c]["ce"])
                               for c in original))


def validate_arm(state, arm, max_multiplicity, count_ratio,
                 max_acc_gap, max_ce_gap, materialize=True):
    a, c = int(arm["aggressor"]), int(arm["replacement"])
    if not (0 <= a < state["known_classes"] and 0 <= c < state["known_classes"]):
        raise ValueError("A and C must both be historical classes")
    replaced, manifest = replacement_slots(
        state["retained_ds_all"], a, c, state["task_class_ranges"],
        max_multiplicity=max_multiplicity, materialize=materialize)
    count_a = sum(row["n_A_slots"] for row in manifest)
    count_c = sum(row["n_C_slots"] for row in manifest)
    if not count_a or not count_c:
        raise ValueError("A and C must both have replay exemplars")
    ratio = count_c / count_a
    if not 1 / count_ratio <= ratio <= count_ratio:
        raise ValueError("A/C replay count ratio is outside matching limit")
    for row in manifest:
        if row["n_A_slots"]:
            local_ratio = row["n_C_slots"] / row["n_A_slots"]
            if not 1 / count_ratio <= local_ratio <= count_ratio:
                raise ValueError("client {} A/C replay count ratio exceeds matching limit".format(
                    row["client_id"]))
    boundary = state.get("boundary_class_metrics", {})
    if a not in boundary or c not in boundary:
        raise ValueError("missing selected task-boundary class metrics for A/C")
    if abs(boundary[a]["acc"] - boundary[c]["acc"]) > max_acc_gap:
        raise ValueError("A/C boundary accuracy gap exceeds matching limit")
    if abs(boundary[a]["ce"] - boundary[c]["ce"]) > max_ce_gap:
        raise ValueError("A/C boundary CE gap exceeds matching limit")
    return replaced, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-c-multiplicity", type=int, default=3)
    parser.add_argument("--max-replay-count-ratio", type=float, default=2.0)
    parser.add_argument("--max-boundary-acc-gap", type=float, default=0.25)
    parser.add_argument("--max-boundary-ce-gap", type=float, default=2.0)
    parser.add_argument("--max-baseline-drift", type=float, default=0.01,
                        help="largest allowed selected-model accuracy/CE gap on replaying repeat-1 baseline")
    cli = parser.parse_args()
    if cli.max_c_multiplicity < 1 or cli.max_replay_count_ratio < 1:
        parser.error("donor multiplicity must be positive and count ratio must be at least 1")
    state = load_snapshot(cli.snapshot)
    spec = json.loads(cli.spec.read_text(encoding="utf-8"))
    if spec["transition"] != state["task"]:
        raise ValueError("spec transition does not match snapshot")
    if state["task"] not in (2, 4):
        raise ValueError("causal suite is defined for transitions T2 and T4")
    if not spec["arms"]:
        raise ValueError("spec contains no matched A/C arms")
    repeats = spec.get("replay_repeats", [1])
    if any(int(value) < 1 for value in repeats) or len(set(repeats)) != len(repeats):
        raise ValueError("replay_repeats must contain distinct positive integers")
    prepared = []
    for arm in spec["arms"]:
        _, manifest = validate_arm(
            state, arm, cli.max_c_multiplicity,
            cli.max_replay_count_ratio, cli.max_boundary_acc_gap,
            cli.max_boundary_ce_gap, materialize=False)
        prepared.append((int(arm["aggressor"]), int(arm["replacement"]),
                         manifest))
    if len({(a, c) for a, c, _ in prepared}) != len(prepared):
        raise ValueError("duplicate A/C training arms")
    cli.output.mkdir(parents=True, exist_ok=False)
    registry = TaskRegistry(state["task_class_ranges"])
    all_effects = []
    audits = {}
    for repeat in repeats:
        root = cli.output / "repeat_{}".format(repeat)
        baseline_dir = run_arm(state, state["retained_ds_all"],
                               root / "baseline", "baseline", repeat)
        if repeat == 1:
            audit = reproduction_audit(cli.snapshot.parent, baseline_dir,
                                       state["task"])
            audits[str(repeat)] = audit
            (cli.output / "reproduction_audit.json").write_text(
                json.dumps(audits, indent=2), encoding="utf-8")
            if audit is not None and (
                    audit["original_source_round"] != audit["replayed_source_round"]
                    or max(audit["max_acc_gap"], audit["max_ce_gap"]) >
                    cli.max_baseline_drift):
                raise RuntimeError("replayed baseline differs from original run: {}".format(audit))
        for a, c, manifest in prepared:
            replay, actual_manifest = replacement_slots(
                state["retained_ds_all"], a, c, state["task_class_ranges"],
                max_multiplicity=cli.max_c_multiplicity)
            if actual_manifest != manifest:
                raise RuntimeError("preflight donor manifest changed")
            name = "A{}_to_C{}".format(a, c)
            arm_dir = run_arm(state, replay, root / name, name, repeat)
            for round_id in (int(state["args"]["com_round"])-1, "selected"):
                all_effects.extend(effects(baseline_dir, arm_dir, state["task"],
                                           a, c, registry, repeat, round_id))
            del replay
    with (cli.output / "causal_effects.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_effects:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    (cli.output / "manifest.json").write_text(json.dumps(dict(
        snapshot=str(cli.snapshot), spec=spec, matching_limits=dict(
            max_c_multiplicity=cli.max_c_multiplicity,
            max_replay_count_ratio=cli.max_replay_count_ratio,
            max_boundary_acc_gap=cli.max_boundary_acc_gap,
            max_boundary_ce_gap=cli.max_boundary_ce_gap),
        reproduction_audits=audits,
        arms=[dict(A=a, C=c, clients=manifest) for a, c, manifest in prepared]),
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
