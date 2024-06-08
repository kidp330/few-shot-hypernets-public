# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml
from copy import deepcopy
from time import time
from typing import Any, Callable, Generator, Iterable, Iterator
from torch import Tensor
from torch.nn.parameter import Parameter

import numpy as np
import torch
from torch import nn

from methods.meta_template import MetaTemplate
from modules.linear import MetaLinear


class MAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, n_query, params, approx=False):
        super().__init__(model_func, n_way, n_support, n_query, change_way=False)

        self.loss_fn: Callable[[Tensor, Tensor],
                               Tensor] = nn.CrossEntropyLoss()
        self.classifier = MetaLinear(self.feat_dim, n_way)

        # initializes bias to 0 but doesn't touch weights, why?
        self.classifier.bias.data.fill_(0)

        self.maml_only_adapt_classifier = params.maml_only_adapt_classifier

        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx.

    def _hyperparameters(self):
        return {
            'n_task': 4,
            'outer_lr': 0.01,
            'gradient_steps': 5,
        }

    # @override MetaTemplate
    def forward(self, x, params=None) -> Tensor:
        out = self.feature(x, params=self.get_subdict(params, 'feature'))
        scores = self.classifier(
            out, params=self.get_subdict(params, 'classifier'))
        return scores

    # @override MetaTemplate
    def set_forward_loss(self, x):
        x_support, x_query = self._split_set(x)

        self.zero_grad()
        params_adapted = self._maml_adapt(x_support)

        scores = self.forward(x_query, params=params_adapted)
        query_data_labels = self.query_labels()
        loss = self.loss_fn(scores, query_data_labels)

        return loss, scores

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10

        batches = len(train_loader)
        n_task = self._hyperparameters()['n_task']
        assert batches % n_task == 0

        loss_all = torch.empty(n_task)
        acc_all = torch.empty(len(train_loader))

        task_count = 0
        total_loss = 0

        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"

            optimizer.zero_grad()

            loss, scores = self.set_forward_loss(x)
            task_accuracy = self._task_accuracy(scores, self.query_labels())
            total_loss += loss.item()
            loss_all[task_count] = loss
            acc_all[i] = task_accuracy

            task_count += 1

            # 1. Sample a number of tasks \tau_i from p(\tau)
            # 2. For each task, obtain \phi_i = U_{\tau_i}(\theta), by minimizing \L_{\tau_i,\textbf{train}}(\theta) on n_support training samples
            # 3. Update \theta by gradient descent such that it minimizes \L(\theta) := \sum_i \L_{\tau_i,\textbf{test}}(\phi_i) on n_query test samples

            # this is the boundary where the algorithm has to be translated into our few-shot setting.
            # the "number of tasks" from step 1 is equivalent to n_task, the number of batches
            # that need processing before updating the universal parameters, and is a hyperparameter.

            # __jm__ I don't approve of multiplying num_epochs for MAML exclusively, it gets confusing
            if task_count == self._hyperparameters()['n_task']:
                loss_q = loss_all.sum()
                loss_q.backward()

                optimizer.step()
                task_count = 0
            if i % print_freq == 0:
                running_avg_loss = total_loss / float(i+1)
                print(
                    "Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(
                        epoch, i, len(train_loader), running_avg_loss
                    )
                )

        # for i in range(batches // n_task):
        #     optimizer.zero_grad()
        #     losses = torch.empty(n_task)
        #     for k in range(n_task):
        #         x, _ = next(train_loader)
        #         self.n_query = x.size(1) - self.n_support
        #         assert self.n_way == x.size(
        #             0), "MAML does not support way change"

        #         loss, scores = self.set_forward_loss(x)
        #         task_accuracy = self._task_accuracy(
        #             scores, self.query_labels())
        #         total_loss += loss.item()
        #         losses[k] = loss
        #         acc_all.append(task_accuracy)

        #     outer_loss = losses.sum()
        #     assert outer_loss.dim() == 1
        #     outer_loss.backward()
        #     optimizer.step()

        metrics = {"accuracy/train": acc_all.mean()}

        return metrics

    # @override MetaTemplate
    def test_loop(
        self, test_loader, return_std=False, return_time: bool = False
    ):
        _correct = 0
        _count = 0

        batches = len(test_loader)
        acc_all = torch.empty(batches)
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
            acc_all[i] = task_accuracy

        num_tasks = len(acc_all)
        acc_mean, acc_std = torch.std_mean(acc_all)
        print(
            "%d Test Acc = %4.2f%% +- %4.2f%%"
            % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        )
        print("Num tasks", num_tasks)

        ret: list[Any] = [acc_mean, acc_std]
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

    def _maml_adapt(self, x_support: Tensor) -> dict[str, Tensor]:
        y_support = self.support_labels()
        names, params = zip(*self.meta_named_parameters())
        params = torch.tensor(params)

        for _task_step in range(self.task_update_num):
            scores = self.forward(x_support, params=dict(zip(names, params)))
            set_loss = self.loss_fn(scores, y_support)

            grad: tuple[Tensor, ...] = torch.autograd.grad(
                set_loss, params, create_graph=(not self.approx)
            )  # build full graph support gradient of gradient

            assert params.dim() == 1
            assert len(params) == len(grad)
            params -= self.train_lr * torch.tensor(grad)

        return dict(zip(names, params))
