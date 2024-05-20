from abc import abstractmethod
from collections import defaultdict
from typing import Tuple
from torch import Tensor

import numpy as np
import torch
import torch.nn as nn

# policy should be no default implementations in template, lest they be actually obvious


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, n_query, change_way=True):
        super().__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query  # (change depends on input)
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim
        # some methods allow different way classification during training and test
        self.change_way = change_way

    @abstractmethod
    def set_forward(self, x: Tensor) -> Tensor:
        _loss, scores = self.set_forward_loss(x)
        return scores

    @abstractmethod
    def set_forward_loss(self, x: Tensor) -> Tensor:
        raise ValueError(
            "MAML performs further adapation simply by increasing task_upate_num"
        )

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise ValueError(
            "MAML performs further adapation simply by increasing task_upate_num"
        )

    def parse_feature(self, x: Tensor, is_feature: bool) -> Tuple[Tensor, Tensor]:
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(
                self.n_way * (self.n_support + self.n_query), *x.size()[2:]
            )
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, : self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x: Tensor):
        raise ValueError(
            "MAML performs further adapation simply by increasing task_upate_num"
        )  # scores = self.set_forward(x)
    #     y_query = np.repeat(range(self.n_way), self.n_query)

    #     topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    #     topk_ind = topk_labels.cpu().numpy()
    #     top1_correct = np.sum(topk_ind[:, 0] == y_query)
    #     return float(top1_correct), len(y_query)

    def train_loop(self, epoch: int, train_loader, optimizer):
        raise ValueError(
            "MAML performs further adapation simply by increasing task_upate_num"
        )  # print_freq = 10

    #     avg_loss = 0
    #     for i, (x, _) in enumerate(train_loader):
    #         self.n_query = x.size(1) - self.n_support
    #         if self.change_way:
    #             self.n_way = x.size(0)
    #         optimizer.zero_grad()
    #         loss = self.set_forward_loss(x)
    #         loss.backward()
    #         optimizer.step()
    #         avg_loss = avg_loss + loss.item()

    #         if i % print_freq == 0:
    #             # print(optimizer.state_dict()['param_groups'][0]['lr'])
    #             print(
    #                 "Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(
    #                     epoch, i, len(train_loader), avg_loss / float(i + 1)
    #                 )
    #             )

    def test_loop(self, test_loader, record=None, return_std: bool = False):
        raise ValueError(
            "MAML performs further adapation simply by increasing task_upate_num"
        )  # correct = 0
    #     count = 0
    #     acc_all = []
    #     acc_at = defaultdict(list)

    #     iter_num = len(test_loader)
    #     for i, (x, _) in enumerate(test_loader):
    #         self.n_query = x.size(1) - self.n_support
    #         if self.change_way:
    #             self.n_way = x.size(0)
    #         y_query = np.repeat(range(self.n_way), self.n_query)

    #         try:
    #             scores, acc_at_metrics = self.set_forward_with_adaptation(x)
    #             for k, v in acc_at_metrics.items():
    #                 acc_at[k].append(v)
    #         except Exception as e:
    #             scores = self.set_forward(x)

    #         scores = scores.reshape((self.n_way * self.n_query, self.n_way))

    #         topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    #         topk_ind = topk_labels.cpu().numpy()
    #         top1_correct = np.sum(topk_ind[:, 0] == y_query)
    #         correct_this = float(top1_correct)
    #         count_this = len(y_query)
    #         acc_all.append(correct_this / count_this * 100)

    #     metrics = {k: np.mean(v) if len(
    #         v) > 0 else 0 for (k, v) in acc_at.items()}

    #     acc_all = np.asarray(acc_all)
    #     acc_mean = np.mean(acc_all)
    #     acc_std = np.std(acc_all)
    #     print(metrics)
    #     print(
    #         "%d Test Acc = %4.2f%% +- %4.2f%%"
    #         % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
    #     )
    #     if return_std:
    #         return acc_mean, acc_std, metrics
    #     else:
    #         return acc_mean, metrics

    def set_forward_adaptation(
        self, x: Tensor, is_feature=True
    ):  # further adaptation, default is fixing feature and train a new softmax clasifier
        raise ValueError(
            "MAML performs further adapation simply by increasing task_upate_num"
        )  # assert is_feature == True, "Feature is fixed in further adaptation"
    #     z_support, z_query = self.parse_feature(x, is_feature)

    #     z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
    #     z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

    #     y_support = torch.from_numpy(
    #         np.repeat(range(self.n_way), self.n_support))

    #     linear_clf = nn.Linear(self.feat_dim, self.n_way)

    #     set_optimizer = torch.optim.SGD(
    #         linear_clf.parameters(),
    #         lr=0.01,
    #         momentum=0.9,
    #         dampening=0.9,
    #         weight_decay=0.001,
    #     )

    #     loss_function = nn.CrossEntropyLoss()

    #     batch_size = 4
    #     support_size = self.n_way * self.n_support
    #     for epoch in range(100):
    #         rand_id = np.random.permutation(support_size)
    #         for i in range(0, support_size, batch_size):
    #             set_optimizer.zero_grad()
    #             selected_id = torch.from_numpy(
    #                 rand_id[i: min(i + batch_size, support_size)]
    #             )
    #             z_batch = z_support[selected_id]
    #             y_batch = y_support[selected_id]
    #             scores = linear_clf(z_batch)
    #             loss = loss_function(scores, y_batch)
    #             loss.backward()
    #             set_optimizer.step()

    #     scores = linear_clf(z_query)
    #     return scores

    def _make_labels(self, n_set_length: int) -> Tensor:
        return torch.repeat_interleave(torch.arange(self.n_way), n_set_length)

    def query_labels(self) -> Tensor:
        return self._make_labels(self.n_query)

    def support_labels(self) -> Tensor:
        return self._make_labels(self.n_support)
