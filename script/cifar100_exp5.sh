#!/bin/bash

# Exp5a-Local: Trajectory-Subspace Replay with client-local rank-r basis
# NIID (beta=0.5)
python main.py --group=c100t5 --method=exp5a_local --dataset cifar100 --tasks=5 --num_users 5 --beta=0.5 --gdr_task_budget 1000 --exp5a_rank 128

# Exp5a-Global: Trajectory-Subspace Replay with shared globally coordinated rank-r basis
# NIID (beta=0.5)
python main.py --group=c100t5 --method=exp5a_global --dataset cifar100 --tasks=5 --num_users 5 --beta=0.5 --gdr_task_budget 1000 --exp5a_rank 128
