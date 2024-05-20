# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml

from time import time
from typing import Any
from torch import Tensor

import numpy as np
import torch
from torch import nn

import backbone
from methods.meta_template import MetaTemplate


class MAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, n_query, params, approx=False):
        super().__init__(model_func, n_way, n_support, n_query, change_way=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)

        self.maml_adapt_classifier = params.maml_adapt_classifier

        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx.

    # @override MetaTemplate
    def forward(self, x) -> Tensor:
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    # @override MetaTemplate

    def set_forward_loss(self, x):
        x_support, x_query = self._split_set(x)

        if self.maml_adapt_classifier:
            fast_parameters = list(self.classifier.parameters())
            for weight in self.classifier.parameters():
                weight.fast = None
        else:
            fast_parameters = list(
                self.parameters()
            )  # the first gradient calcuated in line 45 is based on original weight
            for weight in self.parameters():
                weight.fast = None

        self.zero_grad()
        self._maml_adapt(x_support)

        scores = self.forward(x_query)
        query_data_labels = self.query_labels()
        loss = self.loss_fn(scores, query_data_labels)

        return loss, scores

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        acc_all = []

        # train
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"

            optimizer.zero_grad()

            loss, scores = self.set_forward_loss(x)
            task_accuracy = self._task_accuracy(scores, self.query_labels())
            avg_loss += loss.item()  # .data[0]
            loss_all.append(loss)
            acc_all.append(task_accuracy)

            task_count += 1

            # each iteration is one task
            # after n_task iterations we aggregate the independent losses together and update target loss
            # this should be a nested loop instead of the way its done here though...
            # and I don't approve of multiplying num_epochs for MAML exclusively, it gets confusing
            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            if i % print_freq == 0:
                print(
                    "Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(
                        epoch, i, len(train_loader), avg_loss / float(i + 1)
                    )
                )

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)

        metrics = {"accuracy/train": acc_mean}

        return metrics

    # shouldn't test also perform adaptation? It's definitely missing here
    def test_loop(
        self, test_loader, return_std=False, return_time: bool = False
    ):  # overwrite parrent function
        _correct = 0
        _count = 0
        acc_all = []
        eval_time = 0
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"
            s = time()
            scores = self.set_forward(x)
            t = time()
            eval_time += t - s
            task_accuracy = self._task_accuracy(scores, self.query_labels())
            acc_all.append(task_accuracy)

        num_tasks = len(acc_all)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print(
            "%d Test Acc = %4.2f%% +- %4.2f%%"
            % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        )
        print("Num tasks", num_tasks)

        ret: list[Any] = [acc_mean]
        if return_std:
            ret.append(acc_std)
        if return_time:
            ret.append(eval_time)
        ret.append({})

        return ret

    def _task_accuracy(self, out: Tensor, y_true: Tensor) -> float:
        _max_scores, max_labels = torch.max(out, dim=1)
        max_labels = max_labels.flatten()
        correct_preds_count = torch.sum(max_labels == y_true)
        task_accuracy = (correct_preds_count / len(y_true)) * 100
        return task_accuracy.item()

    def _split_set(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_support = (
            x[:, : self.n_support, :, :, :]
            .contiguous()
            .view(self.n_way * self.n_support, *x.size()[2:])
        )
        x_query = (
            x[:, self.n_support:, :, :, :]
            .contiguous()
            .view(self.n_way * self.n_query, *x.size()[2:])
        )

        return x_support, x_query

    def _maml_adapt(self, x_support: Tensor):
        y_support = self.support_labels()

        for _task_step in list(range(self.task_update_num)):
            scores = self.forward(x_support)
            set_loss = self.loss_fn(scores, y_support)
            grad = torch.autograd.grad(
                set_loss, fast_parameters, create_graph=True
            )  # build full graph support gradient of gradient
            if self.approx:
                grad = [
                    g.detach() for g in grad
                ]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            parameters = (
                self.classifier.parameters()
                if self.maml_adapt_classifier
                else self.parameters()
            )
            for k, weight in enumerate(parameters):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * \
                        grad[k]  # create weight.fast
                else:
                    weight.fast = (
                        weight.fast - self.train_lr * grad[k]
                    )  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(
                    weight.fast
                )  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
