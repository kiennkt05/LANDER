import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb



# init_epoch = 200
# self.args["com_round"] = 100
# self.args["num_users"] = 5
# self.args["frac"] = 1
# self.args["local_bs"] = 128
# self.args["local_ep"] = 5
# batch_size = 128
# num_workers = 4

lamda = 1000
fishermax = 0.0001


class EWC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # self.fisher = None
        self.fisher_list = {}
        self.mean_list = {}
        self._network = IncrementalNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = self._test_data_loader(test_dataset)
        setup_seed(self.seed, fast_cuda=self.args.get("fast_cuda", False))

        self._fl_train(train_dataset, self.test_loader)

        # if self.fisher is None:
        #     self.fisher = self.getFisherDiagonal(self.train_loader)
        # else:
        #     alpha = self._known_classes / self._total_classes
        #     new_finsher = self.getFisherDiagonal(self.train_loader)
        #     for n, p in new_finsher.items():
        #         new_finsher[n][: len(self.fisher[n])] = (
        #             alpha * self.fisher[n]
        #             + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
        #         )
        #     self.fisher = new_finsher
        # self.mean_list[idx] = {
        #     n: p.clone().detach()
        #     for n, p in self._network.named_parameters()
        #     if p.requires_grad
        # }

    def _local_update(self, model, train_data_loader, idx):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = self._prepare_images(images), labels.cuda(non_blocking=True)
                with self._autocast():
                    output = model(images)["logits"]
                    loss = F.cross_entropy(output, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        

        self.fisher_list[idx] = self.getFisherDiagonal(train_data_loader, model)
        self.mean_list[idx] = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        return model.state_dict()

    def _local_finetune(self, model, train_data_loader, idx):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = self._prepare_images(images), labels.cuda(non_blocking=True)
                fake_targets = labels - self._known_classes
                with self._autocast():
                    output = model(images)["logits"]
                    loss_clf = F.cross_entropy(output[:, self._known_classes :], fake_targets)
                    loss_ewc = self.compute_ewc(idx)
                    loss = loss_clf + lamda * loss_ewc
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        alpha = self._known_classes / self._total_classes
        new_finsher = self.getFisherDiagonal(train_data_loader, model)
        for n, p in new_finsher.items():
            new_finsher[n][: len(self.fisher_list[idx][n])] = (
                alpha * self.fisher_list[idx][n]
                + (1 - alpha) * new_finsher[n][: len(self.fisher_list[idx][n])]
            )
        self.fisher_list[idx] = new_finsher
        self.mean_list[idx] = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        return model.state_dict()


    def _fl_train(self, train_dataset, test_loader):
        self._prepare_model(self._network)
        user_groups, _ = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        prog_bar = tqdm(range(self.args["com_round"]))
        local_loaders = self._client_loaders(train_dataset, user_groups)
        client_model = copy.deepcopy(self._network)
        for _, com in enumerate(prog_bar):
            local_weights = []
            idxs_users = range(self.args["num_users"])
            for idx in idxs_users:
                client_model.load_state_dict(self._network.state_dict())
                local_train_loader = local_loaders[idx]
                if self._cur_task == 0:
                    w = self._local_update(client_model, local_train_loader, idx)
                else:
                    w = self._local_finetune(client_model, local_train_loader, idx)
                local_weights.append({key: value.detach().clone() for key, value in w.items()})
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            if self._should_evaluate(com):
                test_acc = self._compute_accuracy(self._network, test_loader)
                info=("Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc,))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})
        del client_model, local_loaders


    def compute_ewc(self, idx):
        loss = 0
        for n, p in self._network.named_parameters():
            if n in self.fisher_list[idx].keys():
                loss += (
                    torch.sum(
                        (self.fisher_list[idx][n])
                        * (p[: len(self.mean_list[idx][n])] - self.mean_list[idx][n]).pow(2)
                    )
                    / 2
                )
        return loss

    def getFisherDiagonal(self, train_loader, model):
        fisher = {
            n: torch.zeros(p.shape).cuda()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = self._prepare_images(inputs), targets.cuda(non_blocking=True)
            logits = model(inputs)["logits"]
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher
