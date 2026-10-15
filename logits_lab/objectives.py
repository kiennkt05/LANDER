"""Pure objectives for class-prototype relational distillation (torch only)."""
import torch
import torch.nn.functional as F

def cosine_targets(prototypes, temperature=2.0, teacher_margin=1.0, eps=1e-6):
    """Build off-diagonal z-scored targets and full-margin targets.

    prototypes: [C,D], already from a *common* global feature extractor.
    Returns (sim[C,C], offdiag_target[C,C], full_target[C,C], standardized[C,C]).
    """
    if prototypes.ndim != 2 or prototypes.shape[0] < 3:
        raise ValueError('At least 3 class prototypes required for off-diagonal standardization')
    if temperature <= 0 or eps <= 0 or teacher_margin <= 0:
        raise ValueError('temperature, teacher_margin and eps must be positive')
    C = prototypes.shape[0]
    p = F.normalize(prototypes.float(), dim=1, eps=eps)
    s = p @ p.T
    diag = torch.eye(C, device=s.device, dtype=torch.bool)
    off = s.masked_fill(diag, 0.)
    mu = off.sum(1, keepdim=True) / (C - 1)
    centered = (s - mu).masked_fill(diag, 0.)
    std = ((centered.square().sum(1, keepdim=True) / (C - 1)).sqrt()).clamp_min(eps)
    z = (s - mu) / std
    neg_logits = z.masked_fill(diag, float('-inf'))
    qneg = F.softmax(neg_logits / temperature, dim=-1)
    full = z.clone()
    full.diagonal().copy_(neg_logits.max(dim=1).values + teacher_margin)
    qfull = F.softmax(full / temperature, dim=-1)
    if not (torch.isfinite(qneg).all() and torch.isfinite(qfull).all()):
        raise RuntimeError('Non-finite relational targets')
    return s, qneg.detach(), qfull.detach(), z.detach()


def relational_kl(student_logits, target, labels, temperature):
    """KL(q_negative || p_negative) with correct class strictly excluded."""
    if temperature <= 0:
        raise ValueError('temperature must be >0')
    C = student_logits.shape[1]
    if C < 3:
        return student_logits.sum() * 0.
    mask = F.one_hot(labels.long(), C).bool()
    neg = (student_logits.float() / temperature).masked_fill(mask, float('-inf'))
    log_p = F.log_softmax(neg, dim=-1)
    q = target.float()  # [B,C] with q_y=0, or row-indexed externally
    if q.shape != log_p.shape:
        raise ValueError('Bad target shape: %s vs %s' % (tuple(q.shape), tuple(log_p.shape)))
    # Avoid 0 * (-inf) and preserve the exact negative-only normalization.
    safe_log_p = log_p.masked_fill(mask, 0.)
    kls = (q * (q.clamp_min(1e-12).log() - safe_log_p)).sum(dim=-1)
    return kls.mean() * temperature**2


def full_kl(student_logits, targets, labels, temperature):
    q = targets.index_select(0, labels.long()).float()
    logp = F.log_softmax(student_logits.float() / temperature, dim=1)
    return (q * (q.clamp_min(1e-12).log() - logp)).sum(1).mean() * temperature**2


def margin_loss(logits, labels, margin):
    C = logits.shape[1]
    if C < 2:
        return logits.sum() * 0.
    mask = F.one_hot(labels.long(), C).bool()
    z = logits.float()
    strongest = z.masked_fill(mask, float('-inf')).max(dim=1).values
    true = z.gather(1, labels.long()[:,None]).squeeze(1)
    return F.relu(margin + strongest - true).mean()
