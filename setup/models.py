# region imports
# NOTE: maybe even move dict definitions here?
import backbone
from io_utils import (
    model_dict,
    method_dict
)
from methods.hypernets import hypernet_types
# endregion

# region type imports
from methods.baselinetrain import BaselineTrain

from io_params import ParamHolder, Arg
from methods.meta_template import MetaTemplate
from methods.relationnet import RelationNet
# endregion


# __jm__ TODO: deprecate, defaults should be defaults, a 'model-optimal' value is sth else
def _set_default_stop_epoch(params: ParamHolder):
    if params.method in ["baseline", "baseline++"]:
        if params.dataset in ["omniglot", "cross_char"]:
            params.stop_epoch = 5
        elif params.dataset in ["CUB"]:
            # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            params.stop_epoch = 200
        else:
            params.stop_epoch = 400  # default
    else:  # meta-learning methods
        if params.n_shot == 5:
            params.stop_epoch = 400
        else:
            params.stop_epoch = 600  # default


def _setup_baseline_model(params: ParamHolder) -> BaselineTrain:
    if params.dataset == "omniglot":
        assert (
            params.num_classes >= 4112
        ), "class number need to be larger than max label id in base class"
    elif params.dataset == "cross_char":
        assert (
            params.num_classes >= 1597
        ), "class number need to be larger than max label id in base class"

    if params.dataset == "omniglot":
        scale_factor = 10
    else:
        scale_factor = 2

    model = BaselineTrain(
        model_dict[params.model],
        n_classes=params.num_classes,
        scale_factor=scale_factor,
        loss_type="softmax" if params.method == "baseline" else "dist",
    )

    # __jm__ TODO: remove
    if params.stop_epoch is None:
        _set_default_stop_epoch(params)

    return model


def _setup_adaptive_model(
    params: ParamHolder,
    train_few_shot_params: dict[str, int],
) -> MetaTemplate:
    if params.method in ["DKT", "protonet", "matchingnet"]:
        model = method_dict[params.method](
            model_dict[params.model], **train_few_shot_params
        )
        if params.method == "DKT":
            model.init_summary()
    elif params.method in ["relationnet", "relationnet_softmax"]:
        def feature_model():
            match params.model:
                case Arg.Model.Conv4:
                    return backbone.Conv4NP
                case Arg.Model.Conv6:
                    return backbone.Conv6NP
                case Arg.Model.Conv4S:
                    return backbone.Conv4SNP
                case _:
                    return lambda: model_dict[params.model](flatten=False)

        model = RelationNet(
            feature_model=feature_model,
            loss_type="mse" if params.method == "relationnet" else "softmax",
            **train_few_shot_params,
        )
    elif params.method in ["maml", "maml_approx", "hyper_maml", "bayes_hmaml"]:
        # TODO: there must be a better way to do this
        # NOTE: check if this should be done for hyper/bayes maml
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = method_dict[params.method](
            model_dict[params.model],
            params=params,
            approx=(params.method == "maml_approx"),
            **train_few_shot_params,
        )
        if params.dataset in [
            "omniglot",
            "cross_char",
        ]:  # maml use different parameter in omniglot
            model.n_task = 32
            # NOTE: check if this should be done for hyper/bayes maml
            model.task_update_num = 1
            model.train_lr = 0.1

            # __jm__ uhh I don't like this
            params.stop_epoch *= model.n_task  # MAML runs a few tasks per epoch
    elif params.method in hypernet_types.keys():
        hn_type = hypernet_types[params.method]
        model = hn_type(
            model_dict[params.model], params=params, **train_few_shot_params
        )

    return model


def initialize_model(params: ParamHolder) -> MetaTemplate:
    if params.method in ["baseline", "baseline++"]:
        return _setup_baseline_model(params)
    else:
        return _setup_adaptive_model(params)
