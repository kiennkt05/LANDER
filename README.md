## The source code for "Text-Enhanced Data-free Approach for Federated Class-Incremental Learning" accepted by CVPR 2024.
## Paper link: https://arxiv.org/abs/2403.14101

# Method
In this paper, we introduce LANDER (Label Text Centered Data-Free Knowledge Transfer) to address this issue by utilizing label text embeddings (LTE) produced by pretrained language models. Specifically, during the model training phase, our approach treats LTE as anchor points and constrains the feature embeddings of corresponding training samples around them, enriching the surrounding area with more meaningful information. In the DFKT phase, by using these LTE anchors, LANDER can synthesize more meaningful samples, thereby effectively addressing the forgetting problem. Additionally, instead of tightly constraining embeddings toward the anchor, the Bounding Loss is introduced to encourage sample embeddings to remain flexible within a defined radius. This approach preserves the natural differences in sample embeddings and mitigates the embedding overlap caused by heterogeneous federated settings. Extensive experiments conducted on CIFAR100, Tiny-ImageNet, and ImageNet demonstrate that LANDER significantly outperforms previous methods and achieves state-of-the-art performance in FCIL. 
 
![alt text](https://github.com/tmtuan1307/LANDER/blob/main/cvpr2024_lander_thumb.png)

# Reproducing
We test the code on RTX 4090 GPU with pytorch: 
```
torch==2.0.1
torchvision==0.15.2
```

### B200 / Blackwell

The 4090 versions above predate Blackwell. Use a Blackwell-capable PyTorch build
(PyTorch 2.7 or newer with CUDA 12.8 or a supported newer CUDA build) and a matching
`torchvision` version. See the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
Install the remaining imports used by this repository, including `kornia`,
`scipy`, `tqdm`, and `wandb`.

One supported CUDA 12.8 install is:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'numpy==1.26.4' 'kornia<0.7.2' scipy tqdm wandb
```

Check the runtime before training:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
```

For any method on one B200, add `--fast_cuda` to the normal command. For example,
the FedCBDR configuration can be launched as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --group=c100t5 --exp_name=fedcbdr \
  --dataset=cifar100 --method=fedcbdr --fedcbdr_lr_scheduler=cosine \
  --tasks=5 --num_users=5 --beta=0.1 --seed=2023 \
  --gdr_protocol=repo_local --tts_mode=repo_dual --fast_cuda
```

The FedCBDR T2/T4 historical-class monitoring and paired A-to-C intervention
workflow is documented in [docs/fedcbdr_monitoring.md](docs/fedcbdr_monitoring.md).

This mode uses BF16 autocast, channels-last convolutions, and pinned transfers.
It changes floating-point rounding, so compare accuracy with a standard run
before relying on its results. All runs use deterministic PyTorch algorithms and
disable cuDNN benchmarking so full and resumed task sequences use the same kernels.
BF16 covers neural-network training in LANDER, TARGET, FedCBDR, and the
Finetune, LwF, iCaRL, and EWC baselines. Exp5 and Exp6 keep their exact
trajectory training in FP32, while using channels-last layout and pinned input
transfers. FedCBDR replay feature extraction and QR/SVD also stay in FP32.
`--eval_interval=5` reduces repeated reporting evaluations while retaining the
final-round evaluation. Finetune still measures per-class accuracy each round for
its best-per-class summary. Most learners reuse data-loader workers across communication rounds; tune
`--num_worker` for the host CPU and `--local_bs` for B200 memory and throughput.
For a fair speed comparison, use the same dataset, batch size, and task settings
with and without `--fast_cuda`, and synchronize CUDA before measuring elapsed time.

`--fedcbdr_lr_scheduler` applies to every method despite its historical name.
`constant` (the default) keeps `--local_lr` unchanged across communication rounds;
`cosine` anneals it toward `min(0.001, local_lr)` over `--com_round` rounds. This
also replaces the previous implicit 50/75-round step schedule on Tiny ImageNet,
so older runs made with either flag are not directly comparable to new runs.

### Kaggle Tesla T4 x2

The previously working PyTorch 2.2 CUDA 11.8 environment can be used on T4;
the CUDA 12.8 install above is intended for Blackwell.
Expose both GPUs and use `--t4_parralel` (the spelling `--t4_parallel` is also accepted):

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py --group=c100t5 --exp_name=fedcbdr \
  --dataset=cifar100 --method=fedcbdr --fedcbdr_lr_scheduler=cosine \
  --tasks=5 --num_users=5 --beta=0.1 --seed=2023 \
  --gdr_protocol=repo_local --tts_mode=repo_dual \
  --num_worker=2 --t4_parralel
```

This mode splits each local client training batch across the two GPUs with
`DataParallel`, uses FP16 autocast with gradient scaling for Finetune, LANDER,
and FedCBDR, and enables channels-last convolutions. Exp5 and Exp6 also split
local model forwards but keep trajectory gradients in FP32. Global aggregation,
evaluation, replay selection, and synthetic-data generation still use GPU 0.
`--fast_cuda` and `--t4_parralel` cannot be combined. The flag requires two
visible CUDA GPUs and does not change the local batch size or client sampling.
Benchmark full communication rounds against the single-T4 run: `DataParallel`
can be slower for small batches because it copies model replicas each forward.
Keep `--local_bs=128` for a like-for-like comparison; increasing it changes the
number of optimizer steps per local epoch. With five persistent client loaders,
`--num_worker=2` starts ten client workers, so tune it for Kaggle's CPU allocation.

### Stop and resume at a task boundary

`--num_tasks_to_run` counts tasks in this invocation. To train the first 10 of 20
CIFAR-100 tasks and then the remaining 10, run the same training options both
times (including the GPU flags, if used):

```bash
python main.py --dataset=cifar100 --method=fedcbdr --tasks=20 \
  --group=c100t20 --exp_name=split --num_users=5 --beta=0.1 --seed=2023 \
  --num_tasks_to_run=10

python main.py --dataset=cifar100 --method=fedcbdr --tasks=20 \
  --group=c100t20 --exp_name=split --num_users=5 --beta=0.1 --seed=2023 \
  --resume --num_tasks_to_run=10
```

The default checkpoint is `run/<group>_<beta>_<method>_<exp_name><spec>/task_resume.pt`.
Use `--checkpoint_path=PATH` to choose another output file; `--resume=PATH` loads
a specific file. A run with `--resume` and no `--num_tasks_to_run` finishes all
remaining tasks. The checkpoint contains the learner, replay state, random-number
generator states, accuracy curve, and the LANDER images needed for the next task.
Copy the checkpoint to a new session if needed; LANDER replay images are restored
from it. FedCBDR's `metrics.jsonl` and Exp5's diagnostics log are restored too.
Retain other monitoring artifacts separately if you use them for analysis.

Without `--num_tasks_to_run`, a fresh run completes all tasks in one invocation.
Full and split runs use the same deterministic PyTorch settings automatically.
Use the same dataset files, Python and library versions, GPU model/count, and code
for both segments. Resume checks the code, environment, task count, class order,
and training arguments before loading. Deterministic kernels may run more slowly;
PyTorch can raise an error if an operation has no deterministic implementation.

## FedAvg_Test logit objectives for Exp5

The Exp5-family methods accept `--logits_mode` with `none` (default),
`LTE_margin`, `proto_margin`, or `global_margin`. `none` runs the existing Exp5
loss and training path. The other modes add the FedAvg_Test negative-class
relational KL and student logit margin to the original cross-entropy loss.
They apply to both current-task and retained replay batches. The modes are
available for `exp5a_local`, `exp5a_global`, `exp5b_topk`, `exp5b_fedcbdr`,
`exp6_global`, and `exp6b_global`. Other methods reject a non-`none` mode.

| Mode | Teacher for the negative-class KL |
| --- | --- |
| `LTE_margin` | Fixed label-text embeddings, reordered by the FCIL class order. This corresponds to FedAvg_Test's `cosine_margin`. |
| `proto_margin` | Class means from the pre-round global encoder, current training images, and retained replay images. |
| `global_margin` | Detached pre-round global model logits on each local batch. |

The prototype mode requires at least one current or retained image for every
seen class and raises an error if coverage is missing. The first 10 rounds of
each task use only the original CE loss by default. `--logits_lambda` (0.1),
`--logits_gamma` (0.5), `--logits_temperature` (2), and `--logits_margin` (1)
control the objective. `--logits_warmup` (10), `--logits_refresh` (5), and
`--logits_proto_bs` (256) control the prototype schedule. `--logits_lte_path`
can select a trusted LTE pickle; otherwise the repository's
`label_embedding/<dataset>_le.pickle` is used. These modes require the
original Exp5 CE loss, so they cannot be combined with `--repo_dual`,
`--exp5_distill_loss`, or `--exp5_single_distill_loss`.

For example:

```bash
python main.py --dataset=cifar100 --method=exp5a_global --tasks=10 \
  --logits_mode=LTE_margin --fedcbdr_lr_scheduler=cosine
```

Pass `--logits_output_dir=<directory>` to save task-specific targets and
diagnostics. `--logits_probe_rounds=20,50,100` additionally computes the
FedAvg_Test class-wise test probes at those communication rounds. Both are
disabled by default; the probes add evaluation time but do not enter training.

## Baseline
Here, we provide a simple example for different methods. 
For example, for `cifar100-5tasks`, please run the following commands to test the model performance with non-IID (`$\beta=0.5$`) data.

```
#!/bin/bash
# method= ["finetune", "lwf", "ewc", "icarl", "target"]

CUDA_VISIBLE_DEVICES=0 python main.py --group=c100t5 --exp_name=$method_b05 --dataset cifar100 --method=$method --tasks=5 --num_users 5 --beta=0.5 --fedcbdr_lr_scheduler='cosine'
```

### Ours
```
CUDA_VISIBLE_DEVICES=0 python main.py --group=c100t5 --exp_name=lander_b05 --dataset cifar100 --method=lander --tasks=5 --num_users 5 --beta=0.5 --fedcbdr_lr_scheduler='cosine'
```

### Forgetting Calculator
You can put the log file into \forgetting_calculator\log and use the \forgetting_calculator\forgetting_cal.py to calculate the forgetting score.


## Citation:
  ```
@inproceedings{lander,
  title={Text-enhanced data-free approach for federated class-incremental learning},
  author={Tran, Minh-Tuan and Le, Trung and Le, Xuan-May and Harandi, Mehrtash and Phung, Dinh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23870--23880},
  year={2024}
}
  ```
