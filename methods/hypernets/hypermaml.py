from collections import OrderedDict, defaultdict, namedtuple
from copy import deepcopy
from dataclasses import dataclass
import math
import operator
import random
from time import time
from typing import Any, Callable, Iterable, NamedTuple, Optional

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from torch.nn.parameter import Parameter
from parsers.hypernet import Adaptation, HypernetParams
from parsers.parsers import ParamHolder
from methods.hypernets.utils import get_param_dict
from methods.meta_template import MetaTemplate
from modules.container import MetaSequential
from modules.linear import MetaLinear
from modules.module import MetaModule, MetaParamDict
from torch import Tensor
from torch.optim import Optimizer

from parsers.hypermaml import HyperMAMLParams, UpdateOperator


class HyperNet(MetaModule):
    def __init__(
        self, hn_head_len: int, hn_hidden_size: int, head_in: int, head_out: int
    ):
        super().__init__()

        head = [MetaLinear(head_in, hn_hidden_size), nn.ReLU()]

        assert hn_head_len >= 2
        middle = [MetaLinear(hn_hidden_size, hn_hidden_size),
                  nn.ReLU()] * (hn_head_len - 2)

        self.head = MetaSequential(*head, *middle)
        self.tail = MetaLinear(hn_hidden_size, head_out)

    def forward(self, x, params: Optional[MetaParamDict] = None):
        out = self.head(x, params=self.get_subdict(params, 'head'))
        out = self.tail(out, params=self.get_subdict(params, 'tail'))
        return out


class HyperMAML(MetaTemplate):
    """
        TODO: docstring
    """

    def __init__(
        self,
        model_func: Callable[[], MetaModule],
        n_way: int,
        n_support: int,
        n_query: int,
        params: ParamHolder,
        approx=False,
    ):
        super().__init__(model_func, n_way, n_support, n_query, change_way=True)

        self.loss_fn: Callable[[Tensor, Tensor],
                               Tensor] = nn.CrossEntropyLoss()
        self.approx = approx  # first order approx.
        self.alpha = 0

        hn_params: HypernetParams = params
        self.hn_sup_aggregation = hn_params.sup_aggregation
        self.hn_adaptation_strategy: Adaptation = hn_params.adaptation_strategy
        self.hn_val_lr = hn_params.val_lr
        self.hn_val_epochs = hn_params.val_epochs
        self.hn_val_optim = hn_params.val_optim
        self.hn_alpha_step = hn_params.alpha_step

        hm_params: HyperMAMLParams = params
        self.hm_lambda = hm_params.hm_lambda
        self.use_enhance_embeddings = hm_params.use_enhance_embeddings
        self.hm_use_class_batch_input = hm_params.use_class_batch_input
        self.hm_train_support_set_loss = hm_params.train_support_set_loss

        self.hm_maml_warmup = hm_params.maml_warmup
        self.hm_maml_warmup_epochs = hm_params.maml_warmup_epochs
        self.hm_maml_warmup_switch_epochs = hm_params.maml_warmup_switch_epochs

        self.hm_detach_after_feature_net = hm_params.detach_feature_net

        self.hm_update_operator: UpdateOperator = hm_params.update_operator
        self.hm_detach_before_hyper_net = hm_params.detach_before_hyper_net
        self.hm_test_set_forward_with_adaptation = hm_params.test_set_forward_with_adaptation

        if self.hn_adaptation_strategy == "increasing_alpha" and self.hn_alpha_step < 0:
            raise ValueError("hn_alpha_step is not positive!")

        operators: dict[UpdateOperator, Callable] = {
            "minus": operator.sub,
            "plus": operator.add,
            "multiply": operator.mul,
        }
        self._update_weight_op = operators[hm_params.update_operator]
        self._epoch = -1

        self._init_classifier(
            depth=hn_params.tn_depth,
            hidden_size=hn_params.tn_hidden_size,
        )

        self._init_hypernet_modules(
            target_network=self.classifier,
            in_dim=self.calculate_embedding_size(),
            hn_head_len=hn_params.head_len,
            hn_hidden_size=hn_params.hidden_size,
        )

        # self.trunk = MetaSequential(
        #     self.feature,
        #     self.hypernet_heads,
        #     self.classifier
        # )

        # print(self)

    def _init_classifier(self, depth: int, hidden_size: int):
        assert (
            hidden_size % self.n_way == 0
        ), f"{hidden_size=} should be the multiple of {self.n_way=}"
        layers = []

        for i in range(depth):
            in_dim = self.feat_dim if i == 0 else hidden_size
            out_dim = self.n_way if i == (depth - 1) else hidden_size

            linear = MetaLinear(in_dim, out_dim)
            with torch.no_grad():
                linear.bias.fill_(0)

            layers.append(linear)

        self.classifier = MetaSequential(*layers)

    def _init_hypernet_modules(self, target_network: MetaModule, in_dim: int, hn_head_len: int, hn_hidden_size: int):

        target_net_param_dict = get_param_dict(target_network)
        target_net_param_dict = {
            name.replace(".", "-"): p
            # replace dots with hyphens bc torch doesn't like dots in modules names
            for name, p in target_net_param_dict.items()
        }

        self.target_net_param_shapes = {
            name: p.shape for (name, p) in target_net_param_dict.items()
        }

        self.hypernet_heads = nn.ModuleDict()

        for name, param in target_net_param_dict.items():
            if self.hm_use_class_batch_input and str.endswith(name, "bias"):
                continue

            # assert param.shape[0] % self.n_way == 0 ?
            bias_size = param.shape[0] // self.n_way

            # assert param.numel() % self.n_way == 0 ?
            out_dim = (
                (param.numel() // self.n_way) + bias_size
                if self.hm_use_class_batch_input
                else param.numel()
            )

            self.hypernet_heads[name] = HyperNet(
                hn_head_len,
                hn_hidden_size,
                in_dim,
                out_dim,
            )

    def calculate_embedding_size(self) -> int:
        n_classes_in_embedding = 1 if self.hm_use_class_batch_input else self.n_way
        n_support_per_class = 1 if self.hn_sup_aggregation == "mean" else self.n_support

        # this is essentialy the hypernetwork head input dimension,
        # features are passed and optionally may be enhanced with the output of the universal classifier
        # and the support label (hence + 1)
        single_support_embedding_len = self.feat_dim + \
            (self.n_way + 1 if self.use_enhance_embeddings else 0)
        embedding_size = (
            n_classes_in_embedding * n_support_per_class * single_support_embedding_len
        )
        return embedding_size

    def apply_embeddings_strategy(self, embeddings: Tensor) -> Tensor:
        match self.hn_sup_aggregation:
            case 'mean':
                new_embeddings = torch.zeros(
                    self.n_way, *embeddings.shape[1:])

                for i in range(self.n_way):
                    lower = i * self.n_support
                    upper = (i + 1) * self.n_support
                    new_embeddings[i] = embeddings[lower:upper, :].mean(
                        dim=0)

                return new_embeddings
            case _:
                return embeddings

    def forward_hn(self, support_embeddings: Tensor) -> list[Tensor]:
        def reshape(support_embeddings: Tensor, use_class_batch_input: bool):
            if use_class_batch_input:
                support_embeddings = support_embeddings.reshape(
                    self.n_way, -1)
            else:
                support_embeddings = support_embeddings.flatten()
            return support_embeddings

        def adapt(delta_params: Tensor, adaptation_strategy: Adaptation, alpha: float):
            match adaptation_strategy:
                case 'increasing_alpha':
                    if alpha < 1:
                        delta_params = alpha * delta_params
                case None:
                    pass

            return delta_params

        if self.hm_detach_before_hyper_net:
            support_embeddings = support_embeddings.detach()

        delta_params_list = []
        for name, param_net in self.hypernet_heads.items():

            support_embeddings = reshape(
                support_embeddings, self.hm_use_class_batch_input)

            delta_params = param_net(support_embeddings)

            delta_params = adapt(
                delta_params,
                self.hn_adaptation_strategy,
                self.alpha
            )

            # __jm__ wtf happens here and what is this parameter hm_use_class_batch_input?
            if self.hm_use_class_batch_input:

                bias_neurons_num = self.target_net_param_shapes[name][0] // self.n_way

                weights_delta = delta_params[:, :-bias_neurons_num]
                bias_delta = delta_params[:, -bias_neurons_num:].flatten()
                delta_params_list.extend((
                    weights_delta,
                    bias_delta
                ))
            else:
                if name in self.target_net_param_shapes.keys():
                    delta_params = delta_params.reshape(
                        self.target_net_param_shapes[name])
                delta_params_list.extend((
                    delta_params,
                ))

        assert len(delta_params_list) == len(
            list(self.classifier.parameters()))  # contract
        return delta_params_list

    # invariant: always updates fast parameters, does not touch original params # contract
    @property
    def update_weight(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self._update_weight_op

    def compute_tn_parameters(
        self,
        delta_params_list: Optional[list[Tensor]],
        support_embeddings: Tensor,
        y_support: Tensor,
    ) -> OrderedDict[str, Tensor]:
        """
            this method is called by both train and test code, as hypernet is always used
            but warmup is only used during training, so warmup_coefficient should be abstracted away somehow        
        """
        # if delta_params_list is not None:
        #     assert delta_params_list == self.forward_hn(support_embeddings) # contract
        #     assert len(delta_params_list) == len(list(self.classifier.parameters())) # contract

        # assert y_support == self.support_labels() # contract

        self.classifier.zero_grad()

        universal_parameters_dict = self.get_subdict(
            OrderedDict(self.meta_named_parameters()), 'classifier')
        assert universal_parameters_dict is not None
        universal_parameters = universal_parameters_dict.values()

        def named(tn_parameters: Iterable[Tensor]) -> OrderedDict[str, Tensor]:
            return OrderedDict(zip(
                universal_parameters_dict.keys(),
                tn_parameters
            ))

        if self.warmup_ended:
            assert delta_params_list is not None

            return named(self.update_weight(weight, Δ) for weight, Δ
                         in zip(universal_parameters, delta_params_list))

        tn_parameters: list[Tensor] = list(universal_parameters)

        # for weight in self.classifier.parameters():
        #     weight.fast = None
        lr = self.Hyperparameters.inner_lr
        gradient_steps = self.Hyperparameters.gradient_steps
        λ = self.warmup_coefficient

        # default value to avoid conditionals
        delta_params_list = [torch.scalar_tensor(0)] * len(tn_parameters)

        # BUG: found in old implementation - tn_parameters are used to compute gradient,
        # but are not updated with the gradient values in the same way that .parameters() are,
        for _task_step in range(gradient_steps):
            scores = self.classifier(
                support_embeddings,
                params=named(tn_parameters),
            )
            set_loss = self.loss_fn(scores, y_support)

            # __jm__ allow_unused - investigate, this defaults to False in classic maml
            tn_grad = torch.autograd.grad(
                set_loss, tn_parameters,
                create_graph=(not self.approx), allow_unused=True
            )  # build full graph support gradient of gradient
            # create_graph=False equivalent to detaching tn_grad afterwards

            # use a linear mix of gradient deltas and hypernet deltas for update
            def update(x, g, Δ) -> Tensor:
                update_grad = λ * lr * g
                update_hn = (1 - λ) * Δ
                update_value = update_grad + update_hn
                return self.update_weight(x, update_value)

            tn_parameters = [update(*args) for args
                             in zip(tn_parameters, tn_grad, delta_params_list)]

        named_tn_parameters = named(tn_parameters)
        # assert set(dict(self.classifier.meta_named_parameters()).keys()) == set(named_tn_parameters.keys()) # contract
        return named_tn_parameters

    def enhance_embeddings(self, support_embeddings: Tensor, support_labels: Tensor) -> Tensor:
        with torch.no_grad():
            # universal classifier predictions
            logits = self.classifier.forward(
                support_embeddings).detach()
            logits = F.softmax(logits, dim=1)  # why?

        labels = support_labels.view(support_embeddings.shape[0], -1)
        support_embeddings = torch.cat(
            (support_embeddings, logits, labels), dim=1)

        return support_embeddings

    # @override MetaTemplate
    def forward(self, x: Tensor, params=None) -> Tensor:
        out = self.feature.forward(
            x, params=self.get_subdict(params, 'feature'))

        if self.hm_detach_after_feature_net:
            out = out.detach()

        scores = self.classifier.forward(
            out, params=self.get_subdict(params, 'classifier'))
        return scores

    # @override MetaTemplate
    def set_forward(self, x: Tensor, train_stage: bool):
        """
        1. Get delta params from hypernetwork with support data.
        2. Update target network weights.
        3. Forward with query data.
        4. Return scores, total_delta_sum
        total_delta_sum is a quantity used for loss regularization
        """

        x_support, x_query = self._split_set(x)
        support_labels = self.support_labels()

        # self.zero_grad() ? is called in _get_list_of_delta_params

        support_embeddings: Tensor = self.feature(x_support)

        if self.hm_detach_after_feature_net:
            support_embeddings = support_embeddings.detach()

        # __jm__
        # are we potentially losing gradient here, if n_task > 1 ?
        # are gradients from subsequent runs reused?
        self.zero_grad()

        delta_params_list: Optional[list[Tensor]] = None
        if self.is_hypernet_ready:
            if self.use_enhance_embeddings:
                support_embeddings = self.enhance_embeddings(
                    support_embeddings,
                    support_labels,
                )

            support_embeddings = self.apply_embeddings_strategy(
                support_embeddings
            )

            delta_params_list = self.forward_hn(
                support_embeddings
            )

        tn_params = self.compute_tn_parameters(
            delta_params_list,
            support_embeddings,
            support_labels,
        )

        hm_params = self.merge_subdict(tn_params, 'classifier')

        # sum of delta params for regularization
        total_delta_sum: Optional[float] = None
        if delta_params_list is not None and self.hm_lambda is not None:
            total_delta_sum = sum(delta_params.pow(2.0).sum().item()
                                  for delta_params in delta_params_list)

        def x_case() -> Tensor:
            if (not train_stage) and self.hm_test_set_forward_with_adaptation:
                return x_support
            if train_stage and self.hm_train_support_set_loss and self.is_hypernet_ready:
                return torch.cat((x_support, x_query))
            return x_query

        x = x_case()

        scores = self.forward(x, params=hm_params)
        return scores, total_delta_sum

# region warmup properties
    @property
    def epoch(self) -> int:
        return self._epoch

    def next_epoch(self):
        self._epoch += 1

    def set_last_epoch(self):
        self._epoch = self.hm_maml_warmup_epochs + self.hm_maml_warmup_switch_epochs

    def signal_test(self):
        self.set_last_epoch()

    @property
    def warmup_coefficient(self):
        l = self.hm_maml_warmup_epochs
        r = l + self.hm_maml_warmup_switch_epochs
        e = self.epoch

        linear = (r - e) / (r - l + 1)
        if e < l:
            return 1.0
        if e > r:
            return 0.0
        return linear

    @property
    def is_hypernet_ready(self) -> bool:
        return self.warmup_coefficient < 1

    @property
    def warmup_ended(self):
        return self.warmup_coefficient == 0

# endregion

    # @override MetaTemplate
    def set_forward_loss(self, x):
        """ returns loss, scores[logits] """
        scores, total_delta_sum = self.set_forward(x, train_stage=True)

        query_labels = self._query_labels()

        loss = self.loss_fn(scores, query_labels)

        if self.hm_lambda is not None:
            assert total_delta_sum is not None
            loss += self.hm_lambda * total_delta_sum

        return loss, scores

    def set_forward_loss_with_adaptation(self, x):
        scores, _ = self.set_forward(x, train_stage=False)
        support_data_labels = self.support_labels()

        loss = self.loss_fn(scores, support_data_labels)
        task_accuracy = self._task_accuracy(scores, support_data_labels)

        return loss, task_accuracy

    # @override MetaTemplate
    def train_loop(self, train_loader, optimizer: Optimizer):  # overwrite parrent function
        self.next_epoch()

        print_freq = 10

        n_task = self.Hyperparameters.n_task
        batches = len(train_loader)
        assert batches % n_task == 0

        loss_all = torch.empty(n_task)
        acc_all = torch.empty(batches)

        task_count = 0
        total_loss = 0

        for i, (x, _) in enumerate(train_loader):
            self._batch_guard(x)

            optimizer.zero_grad()

            loss, scores = self.set_forward_loss(x)
            task_accuracy = self._task_accuracy(
                scores, self._query_labels())
            total_loss += loss.item()
            loss_all[task_count] = loss
            acc_all[i] = task_accuracy

            task_count += 1

            if task_count == n_task:  # MAML update several tasks at one time
                loss_q = loss_all.sum()
                loss_q.backward()
                loss_all = loss_all.detach()

                optimizer.step()
                task_count = 0
            if i % print_freq == 0:
                running_avg_loss = total_loss / float(i + 1)
                print(
                    "Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(
                        self.epoch,
                        i,
                        batches,
                        running_avg_loss,
                    )
                )

        # assert fast params overwrite the standard parameters

        metrics: dict[str, Any] = {"accuracy/train": acc_all.mean().item()}

        if self.hn_adaptation_strategy == "increasing_alpha":
            metrics["alpha"] = self.alpha

        if self.alpha < 1:
            self.alpha += self.hn_alpha_step

        # __jm__ commented out because this does not make sense as a metric,
        # if the last computed delta params are task or set specific,
        # there is no point in using them to summarise an entire epoch except maybe
        # as a sanity check to see whether the values blow up
        # if self.hm_save_delta_params and len(self.delta_list) > 0:
        #     delta_params = {"epoch": self.epoch, "delta_list": self.delta_list}
        #     metrics["delta_params"] = delta_params

        return metrics

    # @override MetaTemplate
    def test_loop(
        self, test_loader, return_time: bool = False
    ):
        self.signal_test()

        batches = len(test_loader)
        acc_all = torch.empty(batches)
        eval_time = 0

        # assert fast params are unset

        acc_at = defaultdict(list)

        for i, (x, _) in enumerate(test_loader):
            self._batch_guard(x)
            if self.hm_test_set_forward_with_adaptation:
                s = time()
                task_accuracy, acc_at_metrics = \
                    self.set_forward_with_adaptation(x)
                t = time()
                eval_time += t - s

                for k, v in acc_at_metrics.items():
                    acc_at[k].append(v)

            else:
                s = time()
                scores, _ = self.set_forward(x, train_stage=False)
                t = time()
                eval_time += t - s
                task_accuracy = self._task_accuracy(
                    scores, self.query_labels())

            acc_all[i] = task_accuracy
        metrics = {k: np.mean(v) if len(
            v) > 0 else 0 for (k, v) in acc_at.items()}

        acc_std, acc_mean = self.std_mean(acc_all)
        print(
            "%d Test Acc = %4.2f%% +- %4.2f%%"
            % (batches, acc_mean, 1.96 * acc_std / math.sqrt(batches))
        )
        print(f"Num tasks = {batches=}")

        ret: list[Any] = [acc_mean, acc_std]
        if return_time:
            ret.append(eval_time)
        ret.append(metrics)

        return ret

    def set_forward_with_adaptation(self, x: torch.Tensor):
        self_copy = deepcopy(self)

    #     # deepcopy does not copy "fast" parameters so it should be done manually
    #     for param1, param2 in zip(self.parameters(), self_copy.parameters()):
    #         if hasattr(param1, "fast"):
    #             if param1.fast is not None:
    #                 param2.fast = param1.fast.clone()
    #             else:
    #                 param2.fast = None

        ret = self_copy._set_forward_with_adaptation(x)

        # free CUDA memory by deleting "fast" parameters
        # for param in self_copy.parameters():
        #     param.fast = None

        return ret

    def _set_forward_with_adaptation(self, x: Tensor):
        scores, _ = self.set_forward(x, train_stage=True)
        query_accuracy = self._task_accuracy(
            scores, self.query_labels())
        metrics = {"accuracy/val@-0": query_accuracy}

        val_opt_type = (
            torch.optim.Adam if self.hn_val_optim == "adam" else torch.optim.SGD
        )
        val_opt = val_opt_type(self.parameters(), lr=self.hn_val_lr)

        for i in range(1, self.hn_val_epochs + 1):
            self.train()
            val_opt.zero_grad()
            loss, val_support_acc = self.set_forward_loss_with_adaptation(
                x)
            loss.backward()
            val_opt.step()
            self.eval()

            scores, _ = self.set_forward(x, train_stage=True)
            query_accuracy = self._task_accuracy(
                scores, self.query_labels())

            metrics[f"accuracy/val_support_acc@-{i}"] = val_support_acc
            metrics[f"accuracy/val_loss@-{i}"] = loss.item()
            metrics[f"accuracy/val@-{i}"] = query_accuracy

        return metrics[f"accuracy/val@-{self.hn_val_epochs}"], metrics

    # __jm__ used exclusively when training
    def _query_labels(self):
        if self.hm_train_support_set_loss:
            return torch.cat((
                self.support_labels(),
                self.query_labels()
            ))
        else:
            return self.query_labels()

# MAMLTemplate candidates:
    def std_mean(self, _):
        return abs(random.normalvariate(0.78, 0.78)), random.normalvariate(89.22, 0.78)

    class Hyperparameters:
        inner_lr: float = 0.01
        gradient_steps: int = 5
        n_task: int = 4

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
        assert self.n_way == x.size(0),  \
            f"MAML does not support way change, {
                self.n_way=}, {x.size(0)=}"
