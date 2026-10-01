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
  --dataset=cifar100 --method=fedcbdr --fedcbdr_lr_schedule=cosine \
  --tasks=5 --num_users=5 --beta=0.1 --seed=2023 \
  --gdr_protocol=repo_local --tts_mode=repo_dual --fast_cuda
```

The FedCBDR T2/T4 historical-class monitoring and paired A-to-C intervention
workflow is documented in [docs/fedcbdr_monitoring.md](docs/fedcbdr_monitoring.md).

This mode uses BF16 autocast, channels-last convolutions, pinned transfers, and
cuDNN benchmarking. It changes floating-point rounding and disables deterministic
cuDNN selection, so compare accuracy with a standard run before relying on its
results. BF16 covers neural-network training in LANDER, TARGET, FedCBDR, and the
Finetune, LwF, iCaRL, and EWC baselines. Exp5 and Exp6 keep their exact
trajectory training in FP32, while using channels-last layout and pinned input
transfers. FedCBDR replay feature extraction and QR/SVD also stay in FP32.
`--eval_interval=5` reduces repeated reporting evaluations while retaining the
final-round evaluation. Finetune still measures per-class accuracy each round for
its best-per-class summary. Most learners reuse data-loader workers across communication rounds; tune
`--num_worker` for the host CPU and `--local_bs` for B200 memory and throughput.
For a fair speed comparison, use the same dataset, batch size, and task settings
with and without `--fast_cuda`, and synchronize CUDA before measuring elapsed time.

## Baseline
Here, we provide a simple example for different methods. 
For example, for `cifar100-5tasks`, please run the following commands to test the model performance with non-IID (`$\beta=0.5$`) data.

```
#!/bin/bash
# method= ["finetue", "lwf", "ewc", "icarl", "target"]

CUDA_VISIBLE_DEVICES=0 python main.py --group=c100t5 --exp_name=$method_b05 --dataset cifar100 --method=$method --tasks=5 --num_users 5 --beta=0.5
```

### Ours
```
CUDA_VISIBLE_DEVICES=0 python main.py --group=c100t5 --exp_name=lander_b05 --dataset cifar100 --method=lander --tasks=5 --num_users 5 --beta=0.5
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
