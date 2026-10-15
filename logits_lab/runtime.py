"""FedAvg_Test relational logit objectives for Exp5-family local training.

Visual prototypes use current train samples plus retained replay only.
"""
import contextlib
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_manager import DatasetSplit
from .objectives import cosine_targets, relational_kl, margin_loss
from .lte import load_label_embedding_teacher  # PROTOREL_COSINE_MARGIN_V3


@contextlib.contextmanager
def preserve_rng():
    """Probes/feature extraction must not advance the training RNG streams."""
    py = random.getstate()
    np_state = np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class PrototypeExperiment:
    def __init__(self, learner, train_dataset, user_groups):
        self.owner = learner
        requested = learner.args.get('logits_mode', 'none')
        self.mode = {'LTE_margin': 'cosine_margin',
                     'proto_margin': 'proto_margin',
                     'global_margin': 'global_margin'}.get(requested)
        if self.mode is None:
            raise ValueError('Unsupported --logits_mode: %s' % requested)
        self.task = learner._cur_task
        out_root = learner.args.get('logits_output_dir', '').strip()
        # Per-task directory: do not overwrite earlier configurations or centers.
        self.out = str(Path(out_root) / ('task_%02d' % self.task)) if out_root else ''
        self.enabled = True
        self.lambda_ = learner.args.get('logits_lambda', 0.1)
        self.gamma = learner.args.get('logits_gamma', 0.5)
        self.temperature = learner.args.get('logits_temperature', 2.0)
        self.student_margin = learner.args.get('logits_margin', 1.0)
        # The exposed modes use only negative-class targets; this value affects
        # the exported, unused full-class target from FedAvg_Test.
        self.teacher_margin = 1.0
        self.warmup = learner.args.get('logits_warmup', 10)
        self.refresh = learner.args.get('logits_refresh', 5)
        self.batch_size = learner.args.get('logits_proto_bs', 256)
        probe = learner.args.get('logits_probe_rounds', '')
        self.probe_rounds = {int(s.strip()) for s in probe.split(',') if s.strip()}
        self.source = None
        self.sums = None
        self.qneg = None
        self.qfull = None
        self.sim = None
        self.zscore = None
        self.lte_info = None
        self.pre_stats = None
        self.local_stats = {}
        self.proto_stats = {}
        self.round_id = None
        self.dataset = None
        self.groups = ({int(k): [int(i) for i in user_groups[k]] for k in sorted(user_groups)}
                       if self.enabled else {})
        # A client's current task data live in train_dataset/user_groups, while
        # previous tasks live ONLY in retained_ds_all[k] (replay memory).
        self.replay_datasets = ({k: tuple(getattr(learner, 'retained_ds_all', {}).get(k, ()))
                                 for k in self.groups} if self.enabled else {})
        self.current_counts = []
        self.replay_counts = []
        self.counts = []
        if sorted(self.groups) != list(range(len(self.groups))):
            raise RuntimeError('Client IDs must be contiguous 0..K-1')
        C = learner._total_classes
        for k in sorted(self.groups):
            cur_y = np.asarray(train_dataset.labels)[self.groups[k]].astype(np.int64)
            if (cur_y < 0).any() or (cur_y >= C).any():
                raise RuntimeError('Current labels outside expanded classifier range')
            cur = np.bincount(cur_y, minlength=C).astype(np.int64)
            rep = np.zeros(C, dtype=np.int64)
            for ds in self.replay_datasets[k]:
                ys = np.asarray(ds.labels).astype(np.int64)
                if (ys < 0).any() or (ys >= learner._known_classes).any():
                    raise RuntimeError('Replay labels must belong to strictly earlier tasks')
                rep += np.bincount(ys, minlength=C)
            self.current_counts.append(cur.tolist())
            self.replay_counts.append(rep.tolist())
            self.counts.append((cur + rep).tolist())
        if self.wants_proto:
            coverage = np.sum(np.asarray(self.counts, dtype=np.int64), axis=0)
            missing = np.where(coverage == 0)[0].tolist()
            if missing:
                raise RuntimeError(
                    'Task %d: cannot compute ALL seen-class prototypes. Classes %s '
                    'have no current training or retained replay images. '
                    'LANDER replay selection is not class-balanced; do not use '
                    'future data or invent centers. Check retained_ds_all coverage.'
                    % (self.task, missing))
        if learner.repo_dual or learner.exp5_distill_loss or learner.exp5_single_distill_loss:
            raise RuntimeError('Prototype experiment requires original Exp5 CE loss, no replay-specific loss switches.')
        if learner._total_classes < 3:
            raise ValueError('Requires >=3 classes')
        if self.lambda_ < 0 or self.gamma < 0 or self.student_margin < 0:
            raise ValueError('Invalid loss coefficients or student margin')
        if self.temperature <= 0 or self.teacher_margin <= 0 or self.refresh < 1 or self.batch_size < 1 or self.warmup < 0:
            raise ValueError('Invalid prototype schedule or temperature')
        # Use exactly the same training examples/order, but deterministic evaluation
        # transforms from the test dataset. No test examples participate in prototypes.
        self.dataset = copy.copy(train_dataset)
        self.dataset.trsf = learner.test_loader.dataset.trsf
        if self.mode == 'cosine_margin':
            # Fixed label-text embedding teacher, shared for current and replay.
            (self.sim, self.qneg, self.qfull, self.zscore,
             self.lte_info, lte_vectors) = load_label_embedding_teacher(
                learner.args, learner._total_classes, self.temperature,
                self.teacher_margin, self._device(learner._network))
            if self.out:
                Path(self.out).mkdir(parents=True, exist_ok=True)
                torch.save({'embeddings':lte_vectors.cpu(),
                            'similarity':self.sim.cpu(), 'zscore':self.zscore.cpu(),
                            'q_negative':self.qneg.cpu(), 'q_full':self.qfull.cpu(),
                            'metadata':self.lte_info},
                           Path(self.out)/'lte_targets.pt')
        if self.out:
            path = Path(self.out)
            path.mkdir(parents=True, exist_ok=True)
            (path / 'config.json').write_text(json.dumps({
                'mode':self.mode, 'lambda':self.lambda_, 'gamma':self.gamma,
                'temperature':self.temperature, 'student_margin':self.student_margin,
                'teacher_margin':self.teacher_margin,
                'warmup':self.warmup, 'refresh':self.refresh,
                'probe_rounds':sorted(self.probe_rounds),
                'client_class_counts':self.counts,
                'client_current_class_counts':self.current_counts,
                'client_replay_class_counts':self.replay_counts,
                'prototype_source':('precomputed_label_text_embedding' if self.mode=='cosine_margin'
                                    else 'all_current_train_plus_retained_replay' if self.wants_proto
                                    else 'pre_round_global_logits'),
                'lte_info':self.lte_info,
                'class_order':learner.args['class_order'],
                'seed':learner.seed, 'task':self.task, 'num_classes':learner._total_classes,
            }, indent=2, default=str))

    @property
    def wants_proto(self):
        return self.mode in ('proto', 'proto_margin', 'proto_full')

    def _device(self, model):
        return next(model.parameters()).device

    def _prototype_sources(self, client_id):
        """Current source plus *distinct retained samples*, with eval transforms.

        Never add old full-training images, test images, or replay virtual copies.
        The original replay datasets and their augmentation transforms are untouched.
        """
        yield DatasetSplit(self.dataset, self.groups[client_id])
        for original_ds in self.replay_datasets[client_id]:
            if len(original_ds) == 0:
                continue
            source = copy.copy(original_ds)
            source.trsf = self.dataset.trsf
            yield source

    @torch.no_grad()
    def _collect_prototypes(self, model):
        """Compute ALL seen-class means with ONE pre-round global encoder."""
        C = self.owner._total_classes
        D = self.owner.feature_dim
        device = self._device(model)
        # Accumulate on CPU: CUDA index_add_ can be nondeterministic under
        # torch.use_deterministic_algorithms(True), as enforced by LANDER.
        sums = torch.zeros((C, D), dtype=torch.float64)
        counts = torch.zeros((C,), dtype=torch.float64)
        was_training = model.training
        model.eval()
        try:
            with preserve_rng():
                # Simulates each client's local feature-sum/count contribution:
                # the server sees only aggregated sufficient statistics. This
                # code runs in one process because LANDER simulates all clients.
                for k in sorted(self.groups):
                    local_sum = torch.zeros_like(sums)
                    local_count = torch.zeros_like(counts)
                    for source in self._prototype_sources(k):
                        loader = DataLoader(source, batch_size=self.batch_size,
                                            shuffle=False, num_workers=0)
                        for _, images, labels in loader:
                            images = self.owner._prepare_images(images)
                            labels = labels.to(device=device, dtype=torch.long)
                            with self.owner._autocast():
                                features = model(images)['features']
                            feat = F.normalize(features.float(), dim=1).detach().cpu().double()
                            cpu_labels = labels.cpu()
                            local_sum.index_add_(0, cpu_labels, feat)
                            local_count.index_add_(0, cpu_labels, torch.ones_like(cpu_labels, dtype=torch.float64))
                    sums += local_sum
                    counts += local_count
        finally:
            model.train(was_training)
        if (counts <= 0).any():
            raise RuntimeError('Cannot form all class prototypes: counts=%s' % counts.tolist())
        expected = torch.tensor(np.sum(np.asarray(self.counts, dtype=np.int64), axis=0),
                                dtype=torch.float64)
        if not torch.equal(counts, expected):
            raise RuntimeError('Prototype data count mismatch: got=%s expected=%s'
                               % (counts.tolist(), expected.tolist()))
        centers = F.normalize((sums / counts[:,None]).float(), dim=1).to(device)
        self.sim, self.qneg, self.qfull, self.zscore = cosine_targets(
            centers, self.temperature, self.teacher_margin)
        self.proto_stats = {
            'counts':counts.int().tolist(),
            'offdiag_mean':float(self.sim[~torch.eye(C, device=device, dtype=torch.bool)].mean().item()),
            'offdiag_std':float(self.sim[~torch.eye(C, device=device, dtype=torch.bool)].std(unbiased=False).item()),
            'mean_max_negative_target':float(self.qneg.max(1).values.mean().item()),
        }
        if self.out:
            # CPU-only export, no raw samples or private example activations.
            torch.save({'centers':centers.cpu(), 'similarity':self.sim.cpu(),
                        'zscore':self.zscore.cpu(), 'q_negative':self.qneg.cpu(),
                        'q_full':self.qfull.cpu(), 'counts':counts.int().cpu()},
                       Path(self.out)/('prototypes_round_%03d.pt' % self.round_id))

    def start_round(self, round_id, global_model):
        if not self.enabled:
            return
        self.round_id = round_id
        self.local_stats = {}
        self.pre_stats = None
        self.proto_stats = {}
        if self.wants_proto and round_id > self.warmup and (self.qneg is None or (round_id-self.warmup-1) % self.refresh == 0):
            self._collect_prototypes(global_model)
        if round_id in self.probe_rounds:
            self.pre_stats = self._test_class_stats(global_model)

    def augment_loss(self, original_loss, logits, labels, images, global_model):
        if not self.enabled or self.round_id <= self.warmup:
            return original_loss
        if self.mode in ('proto_margin','cosine_margin'):
            if self.qneg is None:
                raise RuntimeError('Missing prototypes')
            distill = relational_kl(logits, self.qneg.index_select(0,labels.long()),labels,self.temperature)
        elif self.mode == 'global_margin':
            # Global model remains at pre-round weights throughout ALL local clients.
            # Detached negative-class teacher predictions, no teacher parameter updates.
            with torch.no_grad():
                was = global_model.training
                global_model.eval()
                try:
                    with self.owner._autocast():
                        teacher_logits = global_model(images)['logits'].float()
                finally:
                    global_model.train(was)
                mask = F.one_hot(labels.long(), logits.shape[1]).bool()
                teacher_neg = (teacher_logits/self.temperature).masked_fill(mask,float('-inf'))
                target = F.softmax(teacher_neg,dim=1)
            distill = relational_kl(logits,target,labels,self.temperature)
        else:
            raise AssertionError(self.mode)
        loss = original_loss + self.lambda_*distill
        loss = loss + self.gamma*margin_loss(logits,labels,self.student_margin)
        return loss

    @torch.no_grad()
    def _test_class_stats(self, model):
        C = self.owner._total_classes
        device = self._device(model)
        # CPU accumulation avoids nondeterministic CUDA scatter/index_add_.
        correct = torch.zeros(C, dtype=torch.long)
        totals = torch.zeros(C, dtype=torch.long)
        confusion = torch.zeros((C,C), dtype=torch.long)
        gap_sum = torch.zeros(C, dtype=torch.float64)
        violations = torch.zeros(C, dtype=torch.long)
        was_training = model.training
        model.eval()
        try:
            with preserve_rng():
                for _,images,labels in self.owner.test_loader:
                    images = self.owner._prepare_images(images)
                    labels = labels.to(device=device,dtype=torch.long)
                    with self.owner._autocast():
                        logits = model(images)['logits'].float()
                        pred = logits.argmax(1)
                        true_logits = logits.gather(1,labels[:,None]).squeeze(1)
                        negative_logits = logits.masked_fill(
                            F.one_hot(labels,C).bool(),float('-inf')).max(1).values
                        gap = (true_logits-negative_logits).detach().cpu().double()
                    cpu_y = labels.cpu()
                    cpu_pred = pred.cpu()
                    gap_sum.index_add_(0,cpu_y,gap)
                    violations.index_add_(0,cpu_y,(gap<self.student_margin).long())
                    totals.index_add_(0,cpu_y,torch.ones_like(cpu_y))
                    correct.index_add_(0,cpu_y,(cpu_pred==cpu_y).long())
                    confusion.view(-1).index_add_(0,cpu_y*C+cpu_pred,
                                                   torch.ones_like(cpu_y))
        finally:
            model.train(was_training)
        n=int(totals.sum().item())
        return {'accuracy':round(100*int(correct.sum().item())/max(1,n),5),
                'correct':correct.cpu().tolist(),'total':totals.cpu().tolist(),
                'per_class':[(round(100*a/b,5) if b else None)
                             for a,b in zip(correct.cpu().tolist(),totals.cpu().tolist())],
                'margin_gap_mean':[(float(a/b) if b else None)
                                   for a,b in zip(gap_sum.tolist(),totals.tolist())],
                'margin_violation_rate':[(float(a/b) if b else None)
                                         for a,b in zip(violations.tolist(),totals.tolist())],
                'confusion':confusion.cpu().tolist()}

    def after_client(self, client_id, local_model):
        if self.enabled and self.round_id in self.probe_rounds:
            stats = self._test_class_stats(local_model)
            seen = [i for i,n in enumerate(self.counts[client_id]) if n>0]
            num=sum(stats['total'][i] for i in seen)
            cor=sum(stats['correct'][i] for i in seen)
            self.local_stats[str(client_id)]={'seen_classes':seen,
                'counts':self.counts[client_id],
                'accuracy_seen_classes':(100*cor/num if num else None),
                'global_test_accuracy':stats['accuracy'],
                'per_class':stats['per_class'],
                'per_class_margin_gap':stats['margin_gap_mean'],
                'per_class_margin_violations':stats['margin_violation_rate']}

    def after_aggregation(self, global_model):
        if not self.enabled or self.round_id not in self.probe_rounds:
            return
        post = self._test_class_stats(global_model)
        pre = self.pre_stats
        rec={'type':'probe','task':self.task,'round':self.round_id,
             'mode':self.mode,'pre':pre,'post':post,'clients':self.local_stats,
             'prototype':self.proto_stats}
        if self.out:
            with (Path(self.out)/'probes.jsonl').open('a') as f:
                f.write(json.dumps(rec,allow_nan=False)+'\n')

    def after_evaluation(self, accuracy, error):
        if not self.enabled or not self.out:
            return
        with (Path(self.out)/'rounds.jsonl').open('a') as f:
            f.write(json.dumps({'round':self.round_id,'task':self.task,'mode':self.mode,
                                'accuracy':float(accuracy),'attribution_error':float(error),
                                'proto_refreshed':bool(self.proto_stats),
                                'proto':self.proto_stats})+'\n')
