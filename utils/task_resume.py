"""Task-boundary checkpoints for continuing an incremental-learning run."""

import hashlib
from importlib import metadata
import logging
import os
from pathlib import Path
import platform
import random
import tempfile

import numpy as np
import torch


FORMAT_VERSION = 1
# DataLoaders own worker processes and are reconstructed by incremental_train.
TRANSIENT_FIELDS = {
    "test_loader", "old_loader", "new_loader", "syn_data_loader",
    "pre_loader", "train_dataset", "logger", "_active_tracker", "_t4_scaler",
}
RUNTIME_ARGS = {"num_tasks_to_run", "resume", "checkpoint_path", "class_order"}


def source_fingerprints():
    root = Path(__file__).resolve().parent.parent
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if (path.name == "main.py" or
            path.relative_to(root).parts[0] in {"methods", "utils", "convs"})
    }


def runtime_fingerprint():
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "Pillow", "kornia"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
    }


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(state["cuda"]):
            raise ValueError("Resume requires the same number of visible CUDA GPUs")
        torch.cuda.set_rng_state_all(state["cuda"])
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.set_float32_matmul_precision(state["float32_matmul_precision"])
    torch.backends.cudnn.allow_tf32 = state["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = state["matmul_allow_tf32"]


def _lander_images(learner, completed):
    if type(learner).__name__ != "LANDER" or completed >= learner.tasks:
        return None
    folder = Path(learner.save_dir) / "task_{}".format(completed - 1)
    if not folder.is_dir():
        raise FileNotFoundError("LANDER replay images missing: {}".format(folder))
    files = {str(path.relative_to(folder)).replace("\\", "/"): path.read_bytes()
             for path in sorted(folder.rglob("*")) if path.is_file()}
    if not files:
        raise ValueError("LANDER replay image directory is empty: {}".format(folder))
    return files


def _restore_lander_images(files, save_dir, completed):
    if files is None:
        return
    folder = Path(save_dir) / "task_{}".format(completed - 1)
    folder.mkdir(parents=True, exist_ok=True)
    existing = {str(path.relative_to(folder)).replace("\\", "/")
                for path in folder.rglob("*") if path.is_file()}
    if existing - files.keys():
        raise ValueError("LANDER replay directory has extra files: {}".format(folder))
    for relative, contents in files.items():
        path = folder / relative
        if path.is_file():
            if path.read_bytes() != contents:
                raise ValueError("LANDER replay image differs from checkpoint: {}".format(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)


def _sidecar_paths(learner):
    paths = {}
    metrics = getattr(learner, "metrics", None)
    if metrics is not None:
        paths["fedcbdr_metrics"] = Path(metrics.directory) / "metrics.jsonl"
    diag_path = getattr(learner, "exp5_diag_path", None)
    if diag_path is not None:
        paths["exp5_diagnostics"] = Path(diag_path)
    return paths


def _sidecar_files(learner):
    return {name: path.read_bytes() if path.is_file() else None
            for name, path in _sidecar_paths(learner).items()}


def _restore_sidecar_files(files, learner):
    paths = _sidecar_paths(learner)
    if files.keys() != paths.keys():
        raise ValueError("Checkpoint diagnostic files are inconsistent")
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = files[name]
        if contents is None:
            if path.is_file():
                raise ValueError("Unexpected diagnostic file at {}".format(path))
        elif path.is_file():
            if path.read_bytes() != contents:
                raise ValueError("Diagnostic file differs from checkpoint: {}".format(path))
        else:
            path.write_bytes(contents)


def save_task_checkpoint(path, learner, args, completed, total_tasks, cnn_curve):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scaler = getattr(learner, "_t4_scaler", None)
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "learner_type": type(learner).__name__,
        "args": {key: value for key, value in args.items() if key not in RUNTIME_ARGS},
        "class_order": list(args["class_order"]),
        "source_fingerprints": source_fingerprints(),
        "runtime_fingerprint": runtime_fingerprint(),
        "completed_tasks": completed,
        "total_tasks": total_tasks,
        "cnn_curve": cnn_curve,
        "learner_state": {key: value for key, value in vars(learner).items()
                          if key not in TRANSIENT_FIELDS},
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "lander_images": _lander_images(learner, completed),
        "sidecar_files": _sidecar_files(learner),
        "rng": rng_state(),
    }
    # An interrupted write must leave the previous complete checkpoint usable.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_task_checkpoint(path, args, data_manager, learner_type):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Resume checkpoint not found: {}".format(path))
    # Whole learner objects are stored; only load checkpoints you created yourself.
    state = torch.load(path, weights_only=False)
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported task checkpoint format")
    if state["learner_type"] != learner_type.__name__:
        raise ValueError("Checkpoint learner differs from --method")
    current_args = {key: value for key, value in args.items() if key not in RUNTIME_ARGS}
    if current_args != state["args"]:
        differences = sorted(key for key in current_args.keys() | state["args"].keys()
                             if current_args.get(key) != state["args"].get(key))
        raise ValueError("Resume arguments differ: {}".format(", ".join(differences)))
    if state["class_order"] != list(data_manager.get_class_order()):
        raise ValueError("Dataset class order differs from checkpoint")
    if state["source_fingerprints"] != source_fingerprints():
        raise ValueError("Python source changed since checkpoint; exact continuation is unavailable")
    if state["runtime_fingerprint"] != runtime_fingerprint():
        raise ValueError("Python, dependency, CUDA, or GPU environment differs from checkpoint")
    if state["total_tasks"] != data_manager.nb_tasks:
        raise ValueError("Dataset task count differs from checkpoint")
    completed = state["completed_tasks"]
    if not 0 < completed <= data_manager.nb_tasks:
        raise ValueError("Invalid completed task count in checkpoint")
    if len(state["cnn_curve"]["top1"]) != completed:
        raise ValueError("Checkpoint accuracy curve length is inconsistent")
    if state["learner_state"]["_cur_task"] != completed - 1:
        raise ValueError("Checkpoint learner task index is inconsistent")
    _restore_lander_images(state["lander_images"], args["save_dir"], completed)
    learner = learner_type.__new__(learner_type)
    learner.__dict__.update(state["learner_state"])
    _restore_sidecar_files(state["sidecar_files"], learner)
    learner.logger = logging.getLogger(learner_type.__module__)
    learner._active_tracker = None
    learner._t4_scaler = None
    if state["scaler_state"] is not None:
        learner._t4_scaler = torch.cuda.amp.GradScaler()
        learner._t4_scaler.load_state_dict(state["scaler_state"])
    # DataManager construction and deserialization may consume random numbers.
    restore_rng(state["rng"])
    return learner, completed, state["cnn_curve"]
