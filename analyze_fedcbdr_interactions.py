"""Export every directed old-old FedCBDR pair and predefine causal arms."""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from run_transition_ablation import load_snapshot, validate_arm
from utils.task_registry import TaskRegistry


def scalar(value):
    number = float(value)
    return number if math.isfinite(number) else None


def load_events(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_pairs(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def export_pairs(monitor_dir, state):
    task = state["task"]
    registry = TaskRegistry(state["task_class_ranges"])
    events = load_events(monitor_dir / "metrics.jsonl")
    selected_classes = {row["class_id"]: row for row in events
                        if row["event"] == "class_probe" and row["task"] == task
                        and row["round"] == "selected"}
    selected_events = [row for row in events if row["event"] == "selected_model"
                       and row["task"] == task]
    if len(selected_events) != 1:
        raise ValueError("expected one selected-model event for the transition")
    selected_round = int(selected_events[0]["source_round"])
    previous_classes = {row["class_id"]: row for row in events
                        if row["event"] == "class_probe" and row["task"] == task-1
                        and row["round"] == "selected"}
    pair_files = sorted(monitor_dir.glob("pair_metrics_T{}_R*.pt".format(task)))
    selected_file = monitor_dir / "pair_metrics_T{}_Rselected.pt".format(task)
    if selected_file not in pair_files:
        raise FileNotFoundError(selected_file)
    final = load_pairs(selected_file)
    rounds = [load_pairs(path) for path in pair_files if path != selected_file]
    rounds = [row for row in rounds if isinstance(row["round"], int)
              and row["round"] <= selected_round]
    if not rounds:
        raise ValueError("no communication-round pair artifacts")
    rounds.sort(key=lambda row: row["round"])
    classes = final["class_ids"]
    n = len(classes)
    clients = int(state["args"]["num_users"])
    co_batch = torch.zeros((clients, n, n), dtype=torch.int64)
    sequential = torch.zeros_like(co_batch)
    lag_sum = torch.zeros((clients, n, n), dtype=torch.float64)
    lag_obs = torch.zeros_like(co_batch)
    draws = torch.zeros((clients, n), dtype=torch.int64)
    mass = torch.zeros((clients, n), dtype=torch.float64)
    for artifact in rounds:
        if artifact["class_ids"] != classes:
            raise ValueError("class ids changed within transition")
        for key, target in (("co_batch_count", co_batch),
                            ("sequential_exposure", sequential),
                            ("lag_sum", lag_sum),
                            ("lag_observations", lag_obs),
                            ("draws", draws), ("ce_mass", mass)):
            if key not in artifact:
                raise ValueError("missing exposure tensor: {}".format(key))
            target += artifact[key]
    first = rounds[0]
    rows = []
    for ai, a in enumerate(classes):
        for bi, b in enumerate(classes):
            if a == b:
                continue
            boundary_margin = final["pair_margin"][bi, ai] - final["delta_pair_margin"][bi, ai]
            row = dict(
                transition=task, round="selected", selected_source_round=selected_round,
                A=a, B=b,
                task_A=registry.task_for(a), task_B=registry.task_for(b),
                pair_type="intra" if registry.task_for(a) == registry.task_for(b) else "inter",
                age_A=task-registry.task_for(a), age_B=task-registry.task_for(b),
                task_distance=abs(registry.task_for(a)-registry.task_for(b)),
                replay_draws_A=int(draws[:, ai].sum()),
                replay_draws_B=int(draws[:, bi].sum()),
                CE_mass_A=float(mass[:, ai].sum()), CE_mass_B=float(mass[:, bi].sum()),
                clients_exposing_A=int((draws[:, ai] > 0).sum()),
                clients_exposing_B=int((draws[:, bi] > 0).sum()),
                clients_exposing_A_and_B=int(((draws[:, ai] > 0) & (draws[:, bi] > 0)).sum()),
                co_batch_A_B=int(co_batch[:, ai, bi].sum()),
                sequential_A_to_B=int(sequential[:, ai, bi].sum()),
                lag_A_to_B=scalar(lag_sum[:, ai, bi].sum() / lag_obs[:, ai, bi].sum())
                    if int(lag_obs[:, ai, bi].sum()) else None,
                Acc_B=selected_classes[b]["acc"],
                delta_Acc_B=selected_classes[b]["acc"]-previous_classes[b]["acc"],
                CE_B=selected_classes[b]["ce"],
                P_B_to_A=scalar(final["pair_confusion_rate"][bi, ai]),
                delta_P_B_to_A=scalar(final["pair_confusion_rate"][bi, ai] -
                                      first["pair_confusion_rate"][bi, ai]),
                pair_margin_BA=scalar(final["pair_margin"][bi, ai]),
                boundary_pair_margin_BA=scalar(boundary_margin),
                delta_pair_margin_BA=scalar(final["delta_pair_margin"][bi, ai]),
                feature_direction_BA=scalar(final["feature_direction"][bi, ai]),
                relative_feature_direction_BA=scalar(final["relative_feature_direction"][bi, ai]),
            )
            rows.append(row)
    return rows


def choose_pairs(rows, seed, per_group):
    rng = random.Random(seed)
    chosen = []
    used = set()
    for kind in ("intra", "inter"):
        pool = [r for r in rows if r["pair_type"] == kind]
        exposed = [r for r in pool if r["sequential_A_to_B"] > 0]
        high = [r for r in exposed if r["delta_pair_margin_BA"] is not None
                and r["delta_pair_margin_BA"] < 0
                and ((r["delta_P_B_to_A"] or 0) > 0 or
                     (r["relative_feature_direction_BA"] or 0) > 0)]
        high.sort(key=lambda r: (r["delta_pair_margin_BA"], -r["sequential_A_to_B"]))
        for row in high[:per_group]:
            chosen.append(("high_signal", row))
            used.add((row["A"], row["B"]))
        negative = [r for r in exposed if r["delta_pair_margin_BA"] is not None
                    and (r["A"], r["B"]) not in used
                    and (r["delta_P_B_to_A"] or 0) <= 0
                    and (r["relative_feature_direction_BA"] or 0) <= 0]
        negative.sort(key=lambda r: (abs(r["delta_pair_margin_BA"]),
                                     -r["sequential_A_to_B"]))
        for row in negative[:per_group]:
            chosen.append(("negative_control", row))
            used.add((row["A"], row["B"]))
        # Stratify by task separation, exposure, boundary similarity and age.
        median_exposure = sorted(r["sequential_A_to_B"] for r in pool)[len(pool)//2]
        similarities = sorted(abs(r["boundary_pair_margin_BA"] or 0) for r in pool)
        median_similarity = similarities[len(similarities)//2]
        strata = defaultdict(list)
        for row in pool:
            if (row["A"], row["B"]) in used:
                continue
            key = (row["task_distance"], row["age_A"], row["age_B"],
                   row["sequential_A_to_B"] > median_exposure,
                   abs(row["boundary_pair_margin_BA"] or 0) > median_similarity)
            strata[key].append(row)
        keys = sorted(strata)
        rng.shuffle(keys)
        for key in keys[:per_group]:
            row = rng.choice(strata[key])
            chosen.append(("random_stratified", row))
            used.add((row["A"], row["B"]))
    return chosen


def choose_controls(state, pairs, controls_per_high):
    registry = TaskRegistry(state["task_class_ranges"])
    controls = defaultdict(list)
    for stratum, row in pairs:
        a, b = row["A"], row["B"]
        boundary = state["boundary_class_metrics"]
        candidates = [c for c in range(state["known_classes"])
                      if c not in (a, b) and registry.task_for(c) == registry.task_for(a)]
        candidates.sort(key=lambda c: (abs(boundary[a]["acc"]-boundary[c]["acc"]),
                                       abs(boundary[a]["ce"]-boundary[c]["ce"]), c))
        wanted = controls_per_high if stratum == "high_signal" else 1
        for c in candidates:
            if len(controls[a, b]) >= wanted:
                break
            try:
                validate_arm(state, dict(aggressor=a, replacement=c), 3, 2.0,
                             0.25, 2.0, materialize=False)
            except ValueError:
                continue
            controls[a, b].append(c)
    return controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("monitor_dir", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--controls-per-high", type=int, default=2)
    parser.add_argument("--replay-strength-control", action="store_true")
    cli = parser.parse_args()
    if cli.per_group < 1 or cli.controls_per_high not in (2, 3):
        parser.error("per-group must be positive and controls-per-high must be 2 or 3")
    state = load_snapshot(cli.snapshot)
    if state["task"] not in (2, 4):
        raise ValueError("analysis is defined for T2 and T4")
    cli.output.mkdir(parents=True, exist_ok=False)
    rows = export_pairs(cli.monitor_dir, state)
    with (cli.output / "pairs_T{}.jsonl".format(state["task"])).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    chosen = choose_pairs(rows, cli.seed, cli.per_group)
    controls = choose_controls(state, chosen, cli.controls_per_high)
    selected = [dict(stratum=stratum, A=row["A"], B=row["B"],
                     C_controls=controls[row["A"], row["B"]])
                for stratum, row in chosen]
    arms = sorted({(row["A"], c) for row in selected for c in row["C_controls"]})
    spec = dict(transition=state["task"],
                replay_repeats=[1, 2] if cli.replay_strength_control else [1],
                selection_seed=cli.seed, selected_pairs=selected,
                arms=[dict(aggressor=a, replacement=c) for a, c in arms])
    (cli.output / "ablation_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8")
    summaries = {}
    for kind in ("intra", "inter"):
        subset = [r for r in rows if r["pair_type"] == kind]
        summaries[kind] = dict(count=len(subset),
                               mean_delta_pair_margin=sum(r["delta_pair_margin_BA"]
                                   for r in subset if r["delta_pair_margin_BA"] is not None) /
                                   max(1, sum(r["delta_pair_margin_BA"] is not None for r in subset)),
                               exposed_pairs=sum(r["sequential_A_to_B"] > 0 for r in subset))
    (cli.output / "summary.json").write_text(json.dumps(summaries, indent=2),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
