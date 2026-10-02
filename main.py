import argparse
import os
import torch
try:
    import wandb
except ImportError:
    wandb = None
from utils.data_manager import DataManager, setup_seed
from utils.toolkit import count_parameters
from methods.finetune import Finetune
from methods.lander import LANDER
from methods.exp5 import Exp5aLocal, Exp5aGlobal, Exp5bTopK, Exp5bFedCBDR
from methods.exp6 import Exp6Global, Exp6bGlobal
from methods.fedcbdr import FedCBDR
from utils.task_resume import load_task_checkpoint, save_task_checkpoint
import warnings

warnings.filterwarnings('ignore')


LEARNERS = {
    "exp5a_local": Exp5aLocal, "exp5a_global": Exp5aGlobal,
    "exp5b_topk": Exp5bTopK, "exp5b_fedcbdr": Exp5bFedCBDR,
    "exp6_global": Exp6Global, "exp6b_global": Exp6bGlobal,
    "finetune": Finetune, "lander": LANDER, "fedcbdr": FedCBDR,
}


def get_learner_type(model_name):
    try:
        return LEARNERS[model_name.lower()]
    except KeyError as exc:
        raise ValueError("Unknown learner: {}".format(model_name)) from exc


def get_learner(model_name, args):
    return get_learner_type(model_name)(args)


def train(args):
    checkpointing = args.get("num_tasks_to_run") is not None or args.get("resume") is not None
    if checkpointing and args["method"].lower() == "lander" and args.get("type", -1) != -1:
        raise ValueError("LANDER task resume requires --type=-1")
    # Full and resumed runs must use the same kernels for task-boundary parity.
    # Set this before CUDA initializes a cuBLAS handle.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    if args["fast_cuda"] and args["t4_parralel"]:
        raise ValueError("--fast_cuda and --t4_parralel cannot be combined")
    if args["fast_cuda"]:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("--fast_cuda requires a CUDA GPU with BF16 support")
    if args["t4_parralel"]:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("--t4_parralel requires two visible CUDA GPUs; use CUDA_VISIBLE_DEVICES=0,1")
        torch.cuda.set_device(0)
        print("T4 parallel mode: local batches use cuda:0 and cuda:1")
    setup_seed(args["seed"], fast_cuda=args["fast_cuda"] or args["t4_parralel"])
    # setup the dataset and labels
    data_manager = DataManager(
        args["dataset"],
        args["class_shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args
    )
    args["class_order"] = data_manager.get_class_order()
    run_config = dict(args)
    total_tasks = data_manager.nb_tasks
    checkpoint_path = args.get("checkpoint_path") or os.path.join(args["save_dir"], "task_resume.pt")
    resume = args.get("resume")
    if resume is not None:
        resume_path = checkpoint_path if resume == "auto" else resume
        learner, start_task, cnn_curve = load_task_checkpoint(
            resume_path, run_config, data_manager, get_learner_type(args["method"]))
        print("Resumed after task {} of {} from {}".format(start_task, total_tasks, resume_path))
    else:
        learner = get_learner(args["method"], args)
        start_task, cnn_curve = 0, {"top1": []}
    num_to_run = args.get("num_tasks_to_run")
    stop_task = total_tasks if num_to_run is None else min(total_tasks, start_task + num_to_run)
    if start_task >= total_tasks:
        print("All {} tasks are already complete".format(total_tasks))
        return cnn_curve

    # train for each task
    for task in range(start_task, stop_task):
        print("All params: {}, Trainable params: {}".format(count_parameters(learner._network),
                                                            count_parameters(learner._network, True)))
        learner.incremental_train(data_manager)  # train for one task
        cnn_accy, nme_accy = learner.eval_task()
        learner.after_task()

        print("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        print("CNN top1 curve: {}".format(cnn_curve["top1"]))

    if checkpointing:
        saved = save_task_checkpoint(checkpoint_path, learner, run_config,
                                     stop_task, total_tasks, cnn_curve)
        print("Saved task checkpoint: {} ({} of {} tasks complete)".format(
            saved, stop_task, total_tasks))
    return cnn_curve


def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # Exp settings
    parser.add_argument('--exp_name', type=str, default='lander_b0', help='name of this experiment')
    parser.add_argument('--wandb', type=int, default=0, help='1 for using wandb')
    parser.add_argument('--save_dir', type=str, default="", help='save the syn data')
    parser.add_argument('--project', type=str, default="LANDER", help='wandb project')
    parser.add_argument('--group', type=str, default="c100", help='wandb group')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--spec', type=str, default="t1", help='choose a model')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--fast_cuda', action='store_true',
                        help='enable B200 CUDA optimizations across all learners')
    parser.add_argument('--t4_parralel', '--t4_parallel', dest='t4_parralel', action='store_true',
                        help='use two visible T4 GPUs for local training with FP16 where supported')
    parser.add_argument('--num_tasks_to_run', type=int, default=None,
                        help='run this many tasks now, then save a task-boundary checkpoint')
    parser.add_argument('--resume', nargs='?', const='auto', default=None, metavar='CHECKPOINT',
                        help='continue from a task checkpoint (default: save_dir/task_resume.pt)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='task-boundary checkpoint output path')

    # federated continual learning settings
    parser.add_argument('--dataset', type=str, default="cifar100", help='which dataset')
    parser.add_argument('--tasks', type=int, default=5, help='num of tasks')
    parser.add_argument('--method', type=str, default="lander", help='choose a learner')
    parser.add_argument('--net', type=str, default="resnet18", help='choose a model')
    parser.add_argument('--com_round', type=int, default=100, help='communication rounds')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluate every N communication rounds, and at the final round')
    parser.add_argument('--local_ep', type=int, default=2, help='local training epochs')
    parser.add_argument('--num_users', type=int, default=5, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
    parser.add_argument('--beta', type=float, default=0.0, help='control the degree of label skew')
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of selected clients')
    parser.add_argument('--class_shuffle', type=int, default=1, help='class shuffle')

    # Data-free Generation
    parser.add_argument('--lr_g', default=2e-3, type=float, help='learning rate of generator')
    parser.add_argument('--synthesis_batch_size', default=256, type=int, help='synthetic data batch size')
    parser.add_argument('--bn', default=1.0, type=float, help='parameter for batchnorm regularization')
    parser.add_argument('--oh', default=0.5, type=float, help='parameter for similarity')
    parser.add_argument('--adv', default=1.0, type=float, help='parameter for diversity')
    parser.add_argument('--nz', default=256, type=int, help='output size of noisy nayer')
    parser.add_argument('--nums', type=int, default=10000, help='the num of synthetic data')
    parser.add_argument('--warmup', default=10, type=int, help='number of epoches generator only warmups not stores images')
    parser.add_argument('--syn_round', default=40, type=int, help='number of synthetize round.')
    parser.add_argument('--g_steps', default=40, type=int, help='number of generation steps.')
    parser.add_argument('--synthesis_log_interval', default=10, type=int,
                        help='print one synthesis loss line every N steps')
    parser.add_argument('--synthesis_eval_interval', default=1, type=int,
                        help='evaluate generated student/teacher every N synthesis rounds')

    # Client Training
    parser.add_argument('--num_worker', type=int, default=4, help='number of worker for dataloader')
    parser.add_argument('--mulc', type=str, default="fork", help='type of multi process for dataloader')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay for optimizer')
    parser.add_argument('--syn_bs', default=1, type=int, help='number of old synthetic data in training, 1 for similar to local_bs')
    parser.add_argument('--local_lr', default=4e-2, type=float, help='learning rate for optimizer')
    parser.add_argument('--kd', default=1.0, type=float,
                        help='TARGET local distillation loss weight')

    # LANDER
    parser.add_argument('--r', default=0.015, type=float, help='LTE center radius')
    parser.add_argument('--ltc', default=5, type=float, help='lamda_ltc parameter for LTE center')
    parser.add_argument('--pre', type=float, default=0.4, help='alpha_pre for distilling from previous task')
    parser.add_argument('--cur', type=float, default=0.2, help='alpha_cur for current task training')

    parser.add_argument('--type', default=-1, type=int,
                        help='seed for initializing training.') # 0 for train forward, 1 pretrain stage 1, 2 pretrain stage 2
    parser.add_argument('--syn', default=1, type=int,
                        help='seed for initializing training.')  # 0 for train forward, 1 pretrain stage 1, 2 pretrain stage 2

    # FedCBDR
    parser.add_argument('--fedcbdr_monitor_dir', default=None, help='enable baseline probe monitoring in a fresh output directory')
    parser.add_argument('--fedcbdr_probe_per_class', type=int, default=32, help='frozen test samples per class')
    parser.add_argument('--fedcbdr_replay_repeat', type=int, default=1, help='FedCBDR replay-strength control: uniformly repeat fixed replay slots')
    parser.add_argument('--tau_old', type=float, default=0.9, help='temperature for old classes in TTS')
    parser.add_argument('--tau_new', type=float, default=1.1, help='temperature for new classes in TTS')
    parser.add_argument('--w_old', type=float, default=1.1, help='old-class loss weight in TTS')
    parser.add_argument('--w_new', type=float, default=0.9, help='new-class loss weight in TTS')
    parser.add_argument('--scale', action='store_true', help='enable FedCBDR scaled evaluation if supported by BaseLearner')
    parser.add_argument('--mem_size', type=int, default=50, help='legacy FedCBDR memory size; only used with fedcbdr_legacy_mem_size')
    parser.add_argument('--fedcbdr_legacy_mem_size', action='store_true', help='derive gdr_task_budget from mem_size * increment')
    parser.add_argument('--gdr_protocol', type=str, default='paper_global', choices=['paper_global', 'repo_local'], help='FedCBDR GDR protocol')
    parser.add_argument('--gdr_mask_mode', type=str, default='dense_qr', choices=['dense_qr', 'implicit_pairwise'], help='orthogonal masking strategy')
    parser.add_argument('--gdr_leverage_mode', type=str, default='economy', choices=['full', 'economy', 'truncated'], help='SVD/leverage mode')
    parser.add_argument('--gdr_rank', type=int, default=None, help='rank used when gdr_leverage_mode=truncated')
    parser.add_argument('--gdr_normalization_mode', type=str, default='global', choices=['global', 'eq6_local'], help='GDR leverage-score normalization')
    parser.add_argument('--gdr_replacement', type=str, default='with', choices=['with', 'without'], help='whether replay sampling uses replacement')
    parser.add_argument('--gdr_correction_mode', type=str, default='none', choices=['none', 'sampling_matrix', 'replay_loss_experimental'], help='GDR sampling correction')
    parser.add_argument('--tts_mode', type=str, default='paper_eq', choices=['paper_eq', 'repo_dual'], help='task-aware temperature-scaling implementation')
    parser.add_argument('--joint_loss', type=str, default='tts', choices=['tts'], help='FedCBDR joint-training loss')
    parser.add_argument('--fedcbdr_lr_scheduler', type=str, default='constant', choices=['constant', 'cosine'], help='FedCBDR local learning-rate scheduler')

    # Exp5 Trajectory Subspace Replay
    parser.add_argument('--repeat_rate', default=1, type=int, help='number of appearances per retained replay item in each Exp5 local epoch (positive integer; current items appear once)')
    exp5_loss = parser.add_mutually_exclusive_group()
    exp5_loss.add_argument('--repo_dual', action='store_true', help='use FedCBDR repo_dual TTS loss in Exp5 with tau_old/tau_new and w_old/w_new')
    exp5_loss.add_argument('--exp5_distill_loss', action='store_true', help='use current-sample CE and replay MSE, refreshing all replay logits after each task')
    exp5_loss.add_argument('--exp5_single_distill_loss', action='store_true', help='use current-sample CE and replay MSE, capturing logits only once from each replay item\'s original task best model')
    parser.add_argument('--gdr_task_budget', default=None, type=int, help='replay budget M per task')
    parser.add_argument('--exp5a_rank', default=128, type=int, help='projection rank r')
    parser.add_argument('--exp5a_svd_oversampling', default=16, type=int, help='randomized-PCA oversampling p')
    parser.add_argument('--exp5a_mask_layers', default=12, type=int, help='number of orthogonal mask layers')
    parser.add_argument('--exp5a_mask_seed_offset', default=5000, type=int, help='mask seed stride offset across tasks')
    parser.add_argument('--exp5a_target_mode', default='budget_scaled_sum', type=str, choices=['budget_scaled_sum', 'full_sum'], help='reconstruction target mode')
    parser.add_argument('--exp5a_tau', default=0.05, type=float, help='attribution invariant tolerance tau')

    # Exp6 Joint Reconstruction
    parser.add_argument('--exp6_max_passes', default=5, type=int, help='maximum coordinate passes for Exp6')
    parser.add_argument('--exp6_improvement_tol', default=1e-8, type=float, help='minimum improvement tolerance for Exp6')

    # Exp6b Hybrid Global + Local Reconstruction
    parser.add_argument('--exp6b_lambda', default=1.0, type=float, help='tradeoff lambda between global and local reconstruction in Exp6b (0=Exp6)')
    parser.add_argument('--exp6b_max_passes', default=5, type=int, help='maximum coordinate passes for Exp6b')
    parser.add_argument('--exp6b_improvement_tol', default=1e-8, type=float, help='minimum improvement tolerance for Exp6b')

    args = parser.parse_args()
    if args.repeat_rate < 1:
        parser.error('--repeat_rate must be a positive integer')
    if args.synthesis_log_interval < 1:
        parser.error('--synthesis_log_interval must be a positive integer')
    if args.eval_interval < 1:
        parser.error('--eval_interval must be a positive integer')
    if args.synthesis_eval_interval < 1:
        parser.error('--synthesis_eval_interval must be a positive integer')
    if args.num_tasks_to_run is not None and args.num_tasks_to_run < 1:
        parser.error('--num_tasks_to_run must be a positive integer')

    return args


if __name__ == '__main__':

    args = args_parser()
    if args.dataset == "tiny_imagenet":
        args.num_class = 200
    elif args.dataset == "cifar10":
        args.num_class = 10
    elif args.dataset == "cifar100":
        args.num_class = 100
    elif args.dataset == "imagenet":
        args.num_class = 1000
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")

    # Match and validate gdr_task_budget based on dataset name and number of tasks
    budget = {
        "cifar10": {3: 450, 5: 300},
        "cifar100": {5: 1000, 10: 500, 20: 250},
        "tiny_imagenet": {10: 2000, 20: 1000},
    }
    
    if args.gdr_task_budget is None:
        try:
            args.gdr_task_budget = budget[args.dataset][args.tasks]
        except KeyError as exc:
            raise ValueError("Specify --gdr_task_budget for this dataset/task count") from exc

    args.init_cls = int(args.num_class / args.tasks)
    args.increment = args.init_cls

    print(args)

    args.exp_name = f"{args.beta}_{args.method}_{args.exp_name}"

    dir = "run"
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists("store"):
        os.makedirs("store")
    if not args.save_dir:
        args.save_dir = os.path.join(dir, args.group + "_" + args.exp_name + "" + args.spec)

    if args.wandb == 1:
        wandb.init(config=args, project=args.project, group=args.group, name=args.exp_name)
        wandb.run.log_code(".")
    args = vars(args)

    train(args)
