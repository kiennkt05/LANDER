"""Read LANDER's fixed label-text embeddings; match remapped FCIL class IDs.

The pickle must come from a trusted source (e.g., original LANDER repo).
"""
import hashlib
import io
import pickle
from pathlib import Path

import torch
from .objectives import cosine_targets


class _CPUTensorUnpickler(pickle.Unpickler):
    """Remap serialized torch storage to CPU (including GPU-saved tensors)."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=True)
        return super().find_class(module, name)


def load_label_embedding_teacher(args, seen_classes, temperature, teacher_margin, device):
    dataset = args.get('dataset')
    if not dataset:
        raise RuntimeError('cosine_margin requires args[dataset]')
    path = Path(args.get('logits_lte_path') or
                Path(__file__).resolve().parent.parent / 'label_embedding' / f'{dataset}_le.pickle')
    if not path.is_file():
        raise FileNotFoundError('Missing LTE artifact: '+str(path))
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open('rb') as f:
        # Python pickle may execute code. Load ONLY trusted LANDER repository data.
        raw = _CPUTensorUnpickler(f).load()
    if not isinstance(raw, torch.Tensor):
        raise RuntimeError('LTE artifact must contain one torch.Tensor')
    if raw.ndim != 2 or raw.shape[0] < 3 or raw.shape[1] < 2:
        raise RuntimeError('LTE must be [num_classes, embedding_dim]')
    if not raw.is_floating_point() or raw.is_complex():
        raise RuntimeError('LTE must be real floating point')
    matrix = raw.detach().to('cpu', dtype=torch.float32)
    if not bool(torch.isfinite(matrix).all()):
        raise RuntimeError('LTE contains NaN/Inf')
    if bool((torch.linalg.vector_norm(matrix, dim=1) <= 1.e-12).any()):
        raise RuntimeError('LTE contains a zero-norm class embedding')

    # LANDER methods/lander.py uses exactly: label_emb=label_emb[class_order]
    # DataManager remaps original class IDs to those positions. Only seen classes.
    original_order = args.get('class_order')
    if original_order is None:
        raise RuntimeError('class_order is required for FCIL LTE alignment')
    original_order = [int(x) for x in original_order]
    if (len(original_order) != matrix.shape[0]
        or sorted(original_order) != list(range(matrix.shape[0]))):
        raise RuntimeError('class_order must permute every original LTE class')
    if not (3 <= int(seen_classes) <= len(original_order)):
        raise RuntimeError('Invalid count of seen FCIL classes')
    labels_file = path.with_name(path.name.replace('_le.pickle','_label.txt'))
    if labels_file.exists():
        names = [line.strip() for line in labels_file.read_text().splitlines() if line.strip()]
        if len(names) != matrix.shape[0]:
            raise RuntimeError('LTE labels text file count does not match tensor')

    ids = original_order[:int(seen_classes)]
    selected = matrix.index_select(0, torch.tensor(ids, dtype=torch.long)).to(device)
    # Crucial: exactly proto_margin's cosine_targets, not a different softmax.
    sim,qneg,qfull,zscore=cosine_targets(selected,temperature,teacher_margin)
    info = {'asset':str(path),'sha256':sha256,'embedding_dim':int(matrix.shape[1]),
            'original_num_classes':int(matrix.shape[0]),
            'seen_classes':int(seen_classes),'original_class_ids':ids,
            'teacher_type':'frozen_label_text_embedding',
            'postprocessing':'l2_cosine_offdiag_zscore_softmax',
            'temperature':float(temperature)}
    return sim,qneg,qfull,zscore,info,selected.detach()
