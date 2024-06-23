from collections import defaultdict
from copy import deepcopy
import math
from time import time
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F

from methods.meta_template import MetaTemplate, TestResults
import backbone
from methods.hypernets.utils import (
    get_param_dict,
    kl_diag_gauss_with_standard_gauss,
    reparameterize,
)
from modules.container import MetaSequential
from modules.linear import MetaLinear
from modules.module import MetaModule
from parsers.hypermaml import HyperMAMLParams
from parsers.hypernet import Adaptation, HypernetParams
from parsers.parsers import ParamHolder
from torch.nn.parameter import Parameter


class BHyperNet(MetaModule):
    """bayesian hypernetwork for target network params"""

    def __init__(self, hn_head_len: int, hn_hidden_size: int, head_in: int, head_out: int):
        super().__init__()

        head = [MetaLinear(head_in, hn_hidden_size), nn.ReLU()]

        assert self.hn_head_len >= 2
        middle = [
            MetaLinear(hn_hidden_size, hn_hidden_size),
            nn.ReLU()
        ] * (hn_head_len - 2)

        self.head = MetaSequential(*head, *middle)

        # tails to equate weights with distributions
        self.tail_mean = MetaLinear(hn_hidden_size, head_out)
        self.tail_logvar = MetaLinear(hn_hidden_size, head_out)

    def forward(self, x, params=None) -> tuple[Tensor, Tensor]:
        out = self.head(x, params=self.get_subdict(params, 'head'))
        out_mean = self.tail_mean(
            out, params=self.get_subdict(params, 'tail_mean'))
        out_logvar = self.tail_logvar(
            out, params=self.get_subdict(params, 'tail_logvar'))
        return out_mean, out_logvar


class BayesHMAML(MetaTemplate):
    def __init__(self, model_func, n_way: int, n_support: int, n_query: int, params: ParamHolder, approx=False):
        super().__init__(
            model_func, n_way, n_support, n_query
        )
        self.approx = approx

        # loss function component
        # self.loss_fn is used within class with the intent of being CE, and is not the model's representative loss fn.
        self.loss_fn: Callable[[Tensor, Tensor],
                               Tensor] = nn.CrossEntropyLoss()
        self.loss_kld = kl_diag_gauss_with_standard_gauss  # Kullback–Leibler divergence
        self.kl_scale = params.kl_scale
        self.kl_step = None  # increase step for share of kld in loss
        self.kl_stop_val = params.kl_stop_val

        self.alpha = 0

        hn_params: HypernetParams = params
        self.hn_adaptation_strategy: Adaptation = hn_params.adaptation_strategy
        self.hn_val_epochs = hn_params.val_epochs
        self.hn_alpha_step = hn_params.alpha_step

        hm_params: HyperMAMLParams = params
        self.hm_use_class_batch_input = hm_params.use_class_batch_input
        self.hm_maml_warmup = hm_params.maml_warmup
        self.hm_maml_warmup_epochs = hm_params.maml_warmup_epochs
        self.hm_maml_warmup_switch_epochs = hm_params.maml_warmup_switch_epochs
        self.hm_maml_update_feature_net = hm_params.maml_update_feature_net
        self.hm_support_set_loss = hm_params.support_set_loss
        self.hm_detach_after_feature_net = hm_params.detach_feature_net
        self.hm_lambda = hm_params.hm_lambda
        self.hm_set_forward_with_adaptation = hm_params.set_forward_with_adaptation
        self.enhance_embeddings = hm_params.enhance_embeddings
        # num of weight set draws for softvoting
        self.weight_set_num_train = hm_params.weight_set_num_train  # train phase
        self.weight_set_num_test = hm_params.weight_set_num_test  # test phase

        assert self.weight_set_num_test is None \
            or self.weight_set_num_test > 0

        self._init_classifier(
            depth=hn_params.tn_depth,
            hidden_size=hn_params.tn_hidden_size
        )

    def _init_classifier(self, depth: int, hidden_size: int):
        assert (
            hidden_size % self.n_way == 0
        ), f"{hidden_size=} should be the multiple of {self.n_way=}"
        layers = []

        for i in range(depth):
            in_dim = self.feat_dim if i == 0 else hidden_size
            out_dim = self.n_way if i == (depth - 1) else hidden_size

            linear = backbone.BLinear_fw(in_dim, out_dim)
            with torch.no_grad():
                linear.bias.fill_(0)

            layers.append(linear)

        self.classifier = nn.Sequential(*layers)

    def _init_hypernet_modules(self, embedding_size: int, hn_head_len: int, hn_hidden_size: int):

        target_net_param_dict = get_param_dict(self.classifier)

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
            if self.hm_use_class_batch_input and name[-4:] == "bias":
                # notice head_out val when using this strategy
                continue

            # assert param.shape[0] % self.n_way == 0 ?
            bias_size = param.shape[0] // self.n_way

            head_in = embedding_size

            # assert param.numel() % self.n_way == 0 ?
            head_out = (
                (param.numel() // self.n_way) + bias_size
                if self.hm_use_class_batch_input
                else param.numel()
            )

            # make hypernetwork for target network param
            self.hypernet_heads[name] = BHyperNet(
                hn_head_len,
                hn_hidden_size,
                head_in,
                head_out,
            )

    def forward_hn(self, support_embeddings):
        def reshape(support_embeddings: Tensor, use_class_batch_input: bool):
            if use_class_batch_input:
                support_embeddings = support_embeddings.reshape(self.n_way, -1)
            else:
                support_embeddings = support_embeddings.flatten()
            return support_embeddings

        def adapt(delta_mean: Tensor, logvar: Tensor, adaptation_strategy: Adaptation, alpha: float):
            match adaptation_strategy:
                case 'increasing_alpha':
                    if alpha < 1:
                        delta_mean = alpha * delta_mean
                        logvar = alpha * logvar
                case None:
                    pass

            return delta_mean, logvar

        if self.hm_detach_before_hyper_net:
            support_embeddings = support_embeddings.detach()

        delta_params_list = []
        for name, param_net in self.hypernet_heads.items():
            support_embeddings = reshape(
                support_embeddings, self.hm_use_class_batch_input)
            delta_mean, logvar = param_net(
                support_embeddings
            )
            delta_mean, logvar = adapt(
                delta_mean,
                logvar,
                self.hn_adaptation_strategy,
                self.alpha
            )

            if self.hm_use_class_batch_input:

                bias_neurons_num = self.target_net_param_shapes[name][0] // self.n_way

                weights_delta_mean = (
                    delta_mean[:, :-bias_neurons_num]
                    .contiguous()
                    .view(*self.target_net_param_shapes[name])
                )
                bias_delta_mean = delta_mean[:, -bias_neurons_num:].flatten()

                weights_logvar = (
                    logvar[:, :-bias_neurons_num]
                    .contiguous()
                    .view(*self.target_net_param_shapes[name])
                )
                bias_logvar = logvar[:, -bias_neurons_num:].flatten()

                delta_params_list.extend((
                    [weights_delta_mean, weights_logvar],
                    [bias_delta_mean, bias_logvar],
                ))
            else:
                if name in self.target_net_param_shapes.keys():
                    delta_mean = delta_mean.reshape(
                        self.target_net_param_shapes[name])
                    logvar = logvar.reshape(self.target_net_param_shapes[name])
                delta_params_list.extend((
                    [delta_mean, logvar],
                ))

        return delta_params_list

    def _update_weight(self, weight, update_mean, logvar, train_stage=False):
        """get distribution associated with weight. Sample weights for target network."""
        if update_mean is None and logvar is None:
            return
        # if weight.mu is None:
        if not hasattr(weight, "mu") or weight.mu is None:
            weight.mu = None  # __jm__ what the fuck
            weight.mu = weight - update_mean
        else:
            weight.mu = weight.mu - update_mean

        if logvar is None:  # used in maml warmup
            weight.fast = []
            weight.fast.append(weight.mu)
        else:
            weight.logvar = logvar

            weight.fast = []
            if train_stage:
                for _ in range(
                    self.weight_set_num_train
                ):  # sample fast parameters for training
                    weight.fast.append(reparameterize(
                        weight.mu, weight.logvar))
            else:
                if self.weight_set_num_test is not None:
                    for _ in range(
                        self.weight_set_num_test
                    ):  # sample fast parameters for testing
                        weight.fast.append(reparameterize(
                            weight.mu, weight.logvar))
                else:
                    weight.fast.append(weight.mu)  # return expected value

    def _scale_step(self):
        """calculate regularization step for kld"""
        if self.kl_step is None:
            # scale step is calculated so that share of kld in loss increases kl_scale -> kl_stop_val
            self.kl_step = np.power(
                1 / self.kl_scale * self.kl_stop_val, 1 / self.stop_epoch
            )

        self.kl_scale = self.kl_scale * self.kl_step

    def _get_p_value(self, epoch):
        l = self.hm_maml_warmup_epochs
        r = l + self.hm_maml_warmup_switch_epochs
        e = epoch

        linear = (r - e) / (r - l + 1)
        if e < l:
            return 1.0
        if e > r:
            return 0.0
        return linear

    def _update_network_weights(
        self,
        delta_params_list,
        support_embeddings,
        support_data_labels,
        train_stage=False,
    ):
        p = self._get_p_value() if self.hm_maml_warmup else 0
        warmup_ended = (p == 0)
        if warmup_ended:
            for k, weight in enumerate(self.classifier.parameters()):
                update_mean, logvar = delta_params_list[k]
                self._update_weight(weight, update_mean, logvar, train_stage)
            return

        # ===================================
        fast_parameters: list[Parameter] = []

        # see gradient update of feature to see why this is commented out
        # if self.hm_maml_update_feature_net:
        #     fet_fast_parameters = list(self.feature.parameters())
        #     # for weight in self.feature.parameters():
        #     #     weight.fast = None
        #     self.feature.zero_grad()
        #     fast_parameters.extend(fet_fast_parameters)

        # if self.maml_only_adapt_classifier?: could be used here?
        clf_fast_parameters = list(self.classifier.parameters())
        # for weight in self.classifier.parameters():
        #     weight.fast = None
        self.classifier.zero_grad()
        fast_parameters.extend(clf_fast_parameters)

        lr = self._hyperparameters()['inner_lr']
        gradient_steps = self._hyperparameters()['gradient_steps']

        for _task_step in range(gradient_steps):
            scores = self.classifier(support_embeddings)
            set_loss = self.loss_fn(scores, support_data_labels)

            reduction = self.kl_scale
            for weight in self.classifier.parameters():
                if weight.logvar is not None:
                    if weight.mu is not None:
                        # set_loss = set_loss + self.kl_w * reduction * self.loss_kld(weight.mu, weight.logvar)
                        set_loss = set_loss + reduction * self.loss_kld(
                            weight.mu, weight.logvar
                        )
                    else:
                        # set_loss = set_loss + self.kl_w * reduction * self.loss_kld(weight, weight.logvar)
                        set_loss = set_loss + reduction * self.loss_kld(
                            weight, weight.logvar
                        )

            grad = torch.autograd.grad(
                set_loss, fast_parameters,
                create_graph=(not self.approx), allow_unused=True
            )  # build full graph support gradient of gradient

            # classifier_offset = (
            #     len(fet_fast_parameters)
            #     if self.hm_maml_update_feature_net
            #     else 0
            # )
            classifier_offset = 0

            fn_grad, tn_grad = grad[:classifier_offset], grad[classifier_offset:]

            # for weight, g in zip(self.feature.parameters(), fn_grad):
            #     update_value = p * lr * g
            #     # __jm__ TODO: how to _update_weight in contrast to hmaml?
            #     self._update_weight(weight, update_value)

            # update weights of classifier network by adding gradient and output of hypernetwork
            # only uses gradient when p == 1
            for weight, g, (update_mean, logvar) in zip(self.classifier.parameters(), tn_grad, delta_params_list):
                update_grad = p * lr * g
                update_hn = (1 - p) * update_mean
                update_value = update_grad + update_hn
                self._update_weight(
                    weight,
                    update_value,
                    logvar,
                    train_stage,
                )

    def _get_list_of_delta_params(
        self, support_embeddings, support_labels
    ):

        if self.enhance_embeddings:
            with torch.no_grad():
                logits = self.classifier.forward(support_embeddings).detach()
                logits = F.softmax(logits, dim=1)

            labels = support_labels.view(support_embeddings.shape[0], -1)
            support_embeddings = torch.cat(
                (support_embeddings, logits, labels), dim=1)

        # for weight in self.parameters():
        #     weight.fast = None
        # for weight in self.classifier.parameters():
        #     weight.mu = None
        #     # weight.logvar = None
        self.zero_grad()

        support_embeddings = self.apply_embeddings_strategy(support_embeddings)

        delta_params = self.forward_hn(support_embeddings)

        return delta_params

    def apply_embeddings_strategy(self, embeddings: Tensor) -> Tensor:
        match self.hn_sup_aggregation:
            case 'mean':
                new_embeddings = torch.zeros(self.n_way, *embeddings.shape[1:])

                for i in range(self.n_way):
                    lower = i * self.n_support
                    upper = (i + 1) * self.n_support
                    new_embeddings[i] = embeddings[lower:upper, :].mean(dim=0)

                return new_embeddings
            case _:
                return embeddings

    # __jm__ copied over from HyperMAML
    # @override MetaTemplate
    def forward(self, x):
        out = self.feature.forward(x)

        if self.hm_detach_after_feature_net:
            out = out.detach()

        scores = self.classifier.forward(out)
        return scores

    # __jm__ copied over from HyperMAML
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
        support_data_labels = self.support_labels()

        # self.zero_grad() ? is called in _get_list_of_delta_params

        support_embeddings = self.feature(x_support)

        if self.hm_detach_after_feature_net:
            support_embeddings = support_embeddings.detach()

        is_hypernet_ready = (
            not self.hm_maml_warmup) or self._get_p_value() < 1

        delta_params_list: list[Tensor] = self._get_list_of_delta_params(
            support_embeddings,
            support_data_labels
        )

        self._update_network_weights(
            delta_params_list,
            support_embeddings,
            support_data_labels,
        )

        # sum of delta params for regularization
        total_delta_sum: Optional[float] = None
        if self.hm_lambda is not None:
            total_delta_sum = sum(delta_params.pow(2.0).sum().item()
                                  for delta_params in delta_params_list)

        def x_case() -> Tensor:
            if self.hm_set_forward_with_adaptation and not train_stage:
                return x_support
            if self.hm_support_set_loss and train_stage and is_hypernet_ready:
                return torch.cat((x_support, x_query))
            return x_query

        x = x_case()
        scores = self.forward(x)
        return scores, total_delta_sum

    def set_forward_loss(self, x):
        """Adapt and forward using x. Return scores and total losses"""
        scores, total_delta_sum = self.set_forward(
            x  # , train_stage=True
        )

        # calc_sigma = calc_sigma and (self.epoch == self.stop_epoch - 1 or self.epoch % 100 == 0)
        # sigma, mu = self._mu_sigma(calc_sigma)

        query_labels = self._query_labels()

        reduction = self.kl_scale

        loss_ce = self.loss_fn(scores, query_labels)

        loss_kld = torch.zeros_like(loss_ce)

        for _name, weight in self.classifier.named_parameters():
            if weight.mu is not None and weight.logvar is not None:
                val = self.loss_kld(weight.mu, weight.logvar)
                # loss_kld = loss_kld + self.kl_w * reduction * val
                loss_kld = loss_kld + reduction * val

        loss = loss_ce + loss_kld

        if self.hm_lambda is not None:
            assert total_delta_sum is not None
            loss += self.hm_lambda * total_delta_sum

        return loss, loss_ce, loss_kld, scores

    def set_forward_loss_with_adaptation(self, x):
        """returns loss and accuracy from adapted model (copy)"""
        scores, _ = self.set_forward(
            x  # , train_stage=False
        )  # scores from adapted copy
        support_data_labels = self.support_labels()

        reduction = self.kl_scale
        loss_ce = self.loss_fn(scores, support_data_labels)
        loss_kld = torch.zeros_like(loss_ce)

        for _name, weight in self.classifier.named_parameters():
            if weight.mu is not None and weight.logvar is not None:
                # loss_kld = loss_kld + self.kl_w * reduction * self.loss_kld(weight.mu, weight.logvar)
                loss_kld = loss_kld + reduction * self.loss_kld(
                    weight.mu, weight.logvar
                )

        loss = loss_ce + loss_kld

        task_accuracy = self._task_accuracy(scores, support_data_labels)

        return loss, task_accuracy

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10

        n_task = self._hyperparameters()['n_task']
        batches = len(train_loader)
        assert batches % n_task == 0

        loss_all = torch.empty(n_task)
        loss_ce_all = torch.empty(n_task)
        loss_kld_all = torch.empty(n_task)
        acc_all = torch.empty(batches)

        task_count = 0
        total_loss = 0

        # train
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML does not support way change"

            optimizer.zero_grad()

            loss, loss_ce, loss_kld, scores = self.set_forward_loss(x)
            task_accuracy = self._task_accuracy(scores, self._query_labels())
            total_loss += loss.item()
            loss_all[task_count] = loss
            loss_ce_all[i] = loss_ce.item()
            loss_kld_all[i] = loss_kld.item()
            # loss_kld_no_scale_all.append(loss_kld_no_scale.item())
            acc_all[i] = task_accuracy

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = loss_all.sum()
                loss_q.backward()

                optimizer.step()
                task_count = 0
            if i % print_freq == 0:
                running_avg_loss = total_loss / float(i + 1)
                print(
                    "Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}".format(
                        epoch,
                        i,
                        batches,
                        running_avg_loss,
                    )
                )

        self._scale_step()

        metrics: dict[str, Any] = {"accuracy/train": acc_all.mean().item()}
        metrics["loss_ce"] = loss_ce_all.mean().item()
        metrics["loss_kld"] = loss_kld_all.mean().item()

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

    def set_forward_with_adaptation(self, x: torch.Tensor):
        self_copy = deepcopy(self)

        # deepcopy does not copy "fast" parameters so it should be done manually
        for param1, param2 in zip(
            self.feature.parameters(), self_copy.feature.parameters()
        ):
            if hasattr(param1, "fast"):
                if param1.fast is not None:
                    param2.fast = param1.fast.clone()
                else:
                    param2.fast = None

        for param1, param2 in zip(
            self.classifier.parameters(), self_copy.classifier.parameters()
        ):
            if hasattr(param1, "fast"):
                if param1.fast is not None:
                    param2.fast = list(param1.fast)
                else:
                    param2.fast = None
            if hasattr(param1, "mu"):
                if param1.mu is not None:
                    param2.mu = param1.mu.clone()
                else:
                    param2.mu = None
            if hasattr(param1, "logvar"):
                if param1.logvar is not None:
                    param2.logvar = param1.logvar.clone()
                else:
                    param2.logvar = None

        ret = self_copy._set_forward_with_adaptation(x)

        # free CUDA memory by deleting "fast" parameters
        for param in self_copy.parameters():
            param.fast = None
            param.mu = None
            param.logvar = None

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

    def _query_labels(self):
        if self.hm_support_set_loss:
            return torch.cat((
                self.support_labels(),
                self.query_labels()
            ))
        else:
            return self.query_labels()

    def _task_accuracy(self, out: Tensor, y_true: Tensor) -> float:
        _max_scores, max_labels = torch.max(out, dim=1)
        max_labels = max_labels.flatten()
        correct_preds_count = torch.sum(max_labels == y_true)
        task_accuracy = (correct_preds_count / len(y_true)) * 100
        return task_accuracy.item()

    # @override MetaTemplate
    def test_loop(
        self, test_loader, return_time: bool = False
    ) -> TestResults:
        self.signal_test()

        batches = len(test_loader)
        acc_all = torch.empty(batches)
        eval_time = 0

        # assert fast params are unset

        acc_at = defaultdict(list)

        for i, (x, _) in enumerate(test_loader):
            self._batch_guard(x)
    # __jm__ TOMORROW: REFACTOR
            if self.hm_set_forward_with_adaptation:
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
    # __jm__ TOMORROW: REFACTOR
        metrics = {k: np.mean(v) if len(
            v) > 0 else 0 for (k, v) in acc_at.items()}

        acc_std, acc_mean = torch.std_mean(acc_all)
        print(
            "%d Test Acc = %4.2f%% +- %4.2f%%"
            % (batches, acc_mean, 1.96 * acc_std / math.sqrt(batches))
        )
        print(f"Num tasks = {batches=}")

        return TestResults(acc_mean.item(), acc_std.item(), eval_time, metrics)
