# This code is modified from
# https://github.com/dragen1860/MAML-Pytorch
# https://github.com/katerakelly/pytorch-maml
# https://github.com/tristandeleu/pytorch-meta/blob/master/examples/maml/train.py
from collections import OrderedDict
from time import time
from typing import Callable, Optional
from attr import dataclass
from torch import Tensor

import math
import torch
from torch import nn

from torch.utils.data import DataLoader

from methods.meta_template import MetaTemplate, TestResults
from modules.linear import MetaLinear
from modules.module import MetaParamDict
from parsers.parsers import ParamHolder
import backbone


class MAML(MetaTemplate):
    @dataclass
    class Hyperparameters:
        n_task: int = 4
        inner_lr: float = 0.01
        gradient_steps: int = 5

    def __init__(self, model_func, n_way, n_support, n_query, params: ParamHolder, hyperparameters: Hyperparameters, approx=False):
        super().__init__(model_func, n_way, n_support, n_query, change_way=False)

        self.loss_fn: Callable[[Tensor, Tensor],
                               Tensor] = nn.CrossEntropyLoss()
        self._init_classifier()
        self.maml_only_adapt_classifier = params.maml_only_adapt_classifier
        self.log = False
        self.hyperparameters = hyperparameters
        self.approx = approx  # first order approx.

    # @override MetaTemplate
    def forward(self, x: Tensor, params: Optional[OrderedDict[str, Tensor]] = None) -> Tensor:
        x = self.feature(
            x, params=None if self.maml_only_adapt_classifier else self.get_subdict(params, 'feature'))
        x = self.classifier(
            x, params=self.get_subdict(params, 'classifier'))
        return x

    # @override MetaTemplate
    def set_forward(self, x: Tensor):
        x_support, x_query = self._split_set(x)

        self.zero_grad()
        params_adapted = self._maml_adapt(x_support)

        scores = self.forward(x_query, params=params_adapted)
        return scores

    # @override MetaTemplate
    def set_forward_loss(self, x: Tensor):
        scores = self.set_forward(x)
        query_labels = self.query_labels()

        # assert len(scores) == len(query_labels)
        loss = self.loss_fn(scores, query_labels)
        return loss, scores

    # @override MetaTemplate
    # overwrite parrent function
    def train_loop(self, epoch: int, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
        print_freq = 10

        n_task = self.hyperparameters.n_task
        batches = len(train_loader)
        # assert batches % n_task == 0

        loss_all: list[Tensor] = [torch.scalar_tensor(0)] * n_task
        acc_all = torch.empty(batches)

        task_count = 0
        total_loss = 0

        for i, (x, _) in enumerate(train_loader):
            self._batch_guard(x)

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
            if task_count == n_task:
                loss_q = sum(loss_all, torch.scalar_tensor(0))
                loss_q.backward()

                optimizer.step()
                task_count = 0
            if i % print_freq == 0:
                running_avg_loss = total_loss / float(i+1)
                print(f"Epoch {epoch} | Batch {i}/{batches} | Loss {running_avg_loss}")  # nopep8
        # for i in range(batches // n_task):
        #     optimizer.zero_grad()
        #     losses = torch.empty(n_task)
        #     for k in range(n_task):
        #         x, _ = next(train_loader)
        #         self._batch_guard(x)

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

        metrics = {"accuracy/train": acc_all.mean().item()}

        return metrics

    # @override MetaTemplate
    def test_loop(
        self, test_loader: DataLoader
    ) -> TestResults:
        batches = len(test_loader)
        acc_all = torch.empty(batches)
        eval_time = 0
        for i, (x, _) in enumerate(test_loader):
            self._batch_guard(x)

            s = time()
            if i < 2:
                self.log = True
            else:
                self.log = False
            scores = self.set_forward(x)
            t = time()
            eval_time += t - s
            task_accuracy = self._task_accuracy(scores, self.query_labels())
            acc_all[i] = task_accuracy

        acc_std, acc_mean = torch.std_mean(acc_all)

        print(f"Num tasks = {batches=}")
        print(f"Test Acc = {acc_mean:4.2f} +- {1.96 * acc_std / math.sqrt(batches):4.2f}")  # nopep8

        return TestResults(acc_mean.item(), acc_std.item(), eval_time, {})

    def _maml_adapt(self, x_support: Tensor) -> MetaParamDict:
        y_support = self.support_labels()
        lr = self.hyperparameters.inner_lr
        params: MetaParamDict = self._get_adaptable_params()
        gradient_steps = self.hyperparameters.gradient_steps

        for _task_step in range(gradient_steps):
            scores = self.forward(
                x_support, params=self._parameters_meld_adaptable(params))
            set_loss = self.loss_fn(scores, y_support)
            # build full graph support gradient of gradient unless first-order MAML
            # no graph is equivalent to detaching afterwards
            grad: tuple[Tensor, ...] = torch.autograd.grad(
                set_loss, list(params.values()),
                create_graph=(not self.approx)
            )

            assert len(params) == len(grad)
            for name, g in zip(params.keys(), grad):
                # warning: this does NOT have the same behaviour as -=
                # -= modifies the parameter in place, changing the model's original parameters,
                # and it also requires torch.no_grad()
                params[name] = params[name] - lr * g

        return self._parameters_meld_adaptable(params)

    def _get_adaptable_params(self) -> MetaParamDict:
        pdict = OrderedDict(self.meta_named_parameters())
        if self.maml_only_adapt_classifier:
            pdict = self.get_subdict(pdict, 'classifier')
        assert isinstance(pdict, OrderedDict)  # __jm__ HACK shuts up pylance
        return pdict

    def _parameters_meld_adaptable(self, adaptable: MetaParamDict) -> MetaParamDict:
        # assert adaptable.keys() == self._get_adaptable_params().keys()
        melded = self._get_adaptable_params()
        for name in melded.keys():
            ad_name = name
            if self.maml_only_adapt_classifier:
                prefix = 'classifier.'
                if not str.startswith(name, prefix):
                    continue
                ad_name = str.removeprefix(name, prefix)

            melded[name] = adaptable[ad_name]
        return melded

    def _init_classifier(self):
        self.classifier = MetaLinear(self.feat_dim, self.n_way)
        with torch.no_grad():
            self.classifier.bias.fill_(0)

# MAMLTemplate candidates:
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

    def _batch_guard(self, x: Tensor):
        self.n_query = x.size(1) - self.n_support
        assert self.n_way == x.size(0), \
            f"MAML does not support way change, {self.n_way=}, {x.size(0)=}"
