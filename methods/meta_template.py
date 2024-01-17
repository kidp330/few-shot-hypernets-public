from abc import abstractmethod
from collections import namedtuple
from math import prod
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class MetaTemplate(pl.LightningModule):
    def __init__(
        self,
        model_func: pl.LightningModule,
        n_way: int | None = None,
        n_support: int | None = None,
        n_query: int | None = None,
        change_way=True,
    ):
        super().__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim

        self.change_way = change_way  # some methods allow different_way classification during training and test

    # TODO: __jm__ describe these abmethods as best as possible
    @abstractmethod
    def set_forward(self, x, is_feature=False):
        pass

    @abstractmethod
    def set_forward_adaptation(self, x, is_feature=True):
        pass

    @abstractmethod
    def set_forward_loss(self, x) -> dict[str, any]:
        # return {"loss": ..., ""}
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def get_few_shot_dimensions(self, dl: DataLoader):
        X: torch.Tensor
        X, _y = next(iter(dl))
        n_classes, x_dims, *_ = X.shape
        print(n_classes, x_dims)
        return n_classes, x_dims

    def test_change_way_violations(self, n_classes, x_dims):
        if self.n_way is None or self.change_way is True:
            self.n_way = n_classes
        assert self.n_way == n_classes

        if self.n_support is None:
            assert self.n_query is not None and x_dims >= self.n_query
            self.n_support = x_dims - self.n_query
        # elif self.n_query is None:
        assert x_dims >= self.n_support
        self.n_query = x_dims - self.n_support
        # else:
        #     assert self.n_support + self.n_query == x_dims

    # @override pl.LightningModule
    def configure_optimizers(self):
        pass

    # @override pl.LightningModule
    def on_train_start(self):
        super().on_train_start()
        self.test_change_way_violations(
            *self.get_few_shot_dimensions(self.trainer.train_dataloader)
        )

    # these are important - lightning turns off gradient in validation and test by default
    # @override pl.LightningModule
    def on_validation_start(self) -> None:
        super().on_validation_start()
        torch.set_grad_enabled(True)
        self.test_change_way_violations(
            *self.get_few_shot_dimensions(self.val_dataloader())
        )

    # @override pl.LightningModule
    def on_test_start(self) -> None:
        super().on_test_start()
        torch.set_grad_enabled(True)
        self.test_change_way_violations(
            *self.get_few_shot_dimensions(self.trainer.test_dataloaders)
        )

    def parse_feature(self, x, is_feature) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(
                self.n_way * (self.n_support + self.n_query), *x.size()[2:]
            )
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, : self.n_support]
        z_query = z_all[:, self.n_support :]

        return z_support, z_query

    # __jm__ where's this being used
    def correct(self, x):
        scores = self.set_forward(x)
        y_query = torch.repeat_interleave(torch.range(self.n_way), self.n_query)

        _topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        # topk_ind = topk_labels.cpu().numpy()
        # top1_correct = np.sum(topk_ind[:, 0] == y_query)
        top1_correct = torch.sum(topk_labels[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def training_step(self, batch, _batch_idx):
        x, _ = batch
        loss = self.set_forward_loss(x)
        # self.log("train_loss", loss) __jm__ is this necessary?
        return loss

    def test_step(self, batch, _batch_idx):
        x, _ = batch
        y_query = torch.repeat_interleave(torch.range(self.n_way), self.n_query)

        metrics = {}

        set_forward_opt = getattr(self, "set_forward_with_adaptation", default=None)
        if set_forward_opt is not None and callable(set_forward_opt):
            scores, metrics = set_forward_opt(x)
        else:
            scores = self.set_forward(x)

        scores = scores.reshape((self.n_way * self.n_query, self.n_way))

        # TODO: wtf does this mean...
        _topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        top1_correct = torch.sum(topk_labels[:, 0] == y_query)
        correct_this = float(top1_correct)
        count_this = len(y_query)
        metrics["test_acc"] = correct_this / count_this * 100

        # __jm__ huh
        # print(
        #     "%d Test Acc = %4.2f%% +- %4.2f%%"
        #     % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        # )

        self.log_dict(metrics)
        return scores
