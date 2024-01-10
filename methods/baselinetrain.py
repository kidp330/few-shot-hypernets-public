import torch
import torch.nn as nn
import pytorch_lightning as pl

import backbone
from typing import Literal

from methods.meta_template import MetaTemplate


class BaselineTrain(pl.LightningModule):
    def __init__(
        self,
        model_func: pl.LightningModule,
        n_classes: int,
        scale_factor: int,
        loss_type: Literal["softmax", "dist"] = "softmax",
    ):
        super().__init__()
        self.feature = model_func()

        if loss_type == "softmax":
            self.linear_clf = nn.Linear(self.feature.final_feat_dim, n_classes)
            self.linear_clf.bias.data.fill_(0)
        elif loss_type == "dist":  # Baseline ++
            self.linear_clf = backbone.distLinear(
                indim=self.feature.final_feat_dim,
                outdim=n_classes,
                scale_factor=scale_factor,
            )

        self.loss_type = loss_type
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.linear_clf.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        return self.loss_fn(scores, y)

    def training_step(self, train_batch, _batch_idx):
        x, y = train_batch
        loss = self.forward_loss(x, y)
        self.log("train_loss", loss)
        return loss


class BaselineFinetune(MetaTemplate):
    def __init__(self, baseline: BaselineTrain):
        super().__init__(model_func=baseline.feature, change_way=True)
        self.baseline = baseline

    # @override MetaTemplate
    def set_forward(self, x, is_feature=True):
        # Baseline always does adaptation
        return self.set_forward_adaptation(x, is_feature)

    # TODO: use lightning trainer here
    # @override MetaTemplate
    def set_forward_adaptation(self, x, is_feature=True):
        assert is_feature == True, "Baseline only supports testing with feature"
        z_support, z_query = self.parse_feature(
            x, is_feature
        )  # __jm__ from MetaTemplate

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.repeat_interleave(range(self.n_way), self.n_support)

        set_optimizer = torch.optim.SGD(
            self.linear_clf.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.001,
        )

        loss_function = nn.CrossEntropyLoss()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for _epoch in range(100):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = rand_id[i : min(i + batch_size, support_size)]
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = self.linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()
        scores = self.linear_clf(z_query)
        return scores

    # @override MetaTemplate
    def set_forward_loss(self, x):
        raise NotImplementedError(
            "Baseline predicts on pretrained feature and does not support finetune backbone"
        )

    # @override MetaTemplate
    def forward(self, x):
        return self.baseline.forward(x)
