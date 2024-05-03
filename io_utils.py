from typing import Literal

import pytorch_lightning as pl

from io_params import Arg
import backbone

from methods.DKT import DKT
from methods.baselinetrain import BaselineTrain
from methods.hypernets import hypernet_types
from methods.hypernets.bayeshmaml import BayesHMAML
from methods.hypernets.hypermaml import HyperMAML
from methods.hypernets.hypernet_kernel import HyperShot
from methods.hypernets.hypernet_poc import HyperNetPOC, HypernetPPA
from methods.maml import MAML
from methods.matchingnet import MatchingNet
from methods.protonet import ProtoNet

model_dict: dict[str, pl.LightningModule] = {
    Arg.Model.Conv4: backbone.Conv4,
    Arg.Model.Conv4Pool: backbone.Conv4Pool,
    Arg.Model.Conv4S: backbone.Conv4S,
    Arg.Model.Conv6: backbone.Conv6,
    Arg.Model.ResNet10: backbone.ResNet10,
    Arg.Model.ResNet18: backbone.ResNet18,
    Arg.Model.ResNet34: backbone.ResNet34,
    Arg.Model.ResNet50: backbone.ResNet50,
    Arg.Model.ResNet101: backbone.ResNet101,
    Arg.Model.Conv4WithKernel: backbone.Conv4WithKernel,
    Arg.Model.ResNetWithKernel: backbone.ResNetWithKernel,
}

method_dict: dict[str, pl.LightningModule] = {
    Arg.Method.baseline: BaselineTrain,
    Arg.Method.baselinepp: BaselineTrain,
    Arg.Method.DKT: DKT,
    Arg.Method.matchingnet: MatchingNet,
    Arg.Method.protonet: ProtoNet,
    Arg.Method.maml: MAML,
    Arg.Method.maml_approx: MAML,
    Arg.Method.hyper_maml: HyperMAML,
    Arg.Method.bayes_hmaml: BayesHMAML,
    Arg.Method.hyper_shot: HyperShot,
    Arg.Method.hn_ppa: HypernetPPA,
    Arg.Method.hn_poc: HyperNetPOC,
}

illegal_models = [
    "maml",
    "maml_approx",
    "hyper_maml",
    "bayes_hmaml",
    "DKT",
] + list(hypernet_types.keys())


def parse_args(script: Literal["train", "test", "save_features"]):
    pass  # TAP doesn't validate arg to script mapping, TODO __jm__
    # parser = argparse.ArgumentParser(
    #     description=f"few-shot script {script}",
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    # )
    # def add_argument(attr: str):
    #     attr_str = attr
    #     attr = getattr(ParamStruct, attr)
    #     if attr_str == "stop_epoch":
    #         print(attr)
    #     kwargs = {
    #         "default": attr,
    #         "help": attr.__doc__,
    #     }
    #     attr_type = type(attr)
    #     if attr is not None and (attr_type in [int, float] or callable(attr_type)):
    #         kwargs["type"] = attr_type
    #     else:
    #         kwargs["type"] = str
    #     if attr_type is bool:
    #         kwargs["action"] = "store_true"
    #         del kwargs["type"]
    #     elif attr_str in ["milestones"]:
    #         kwargs["nargs"] = "+"
    #         kwargs["type"] = int
    #     elif isinstance(attr, StrEnum):
    #         kwargs["choices"] = Arg.list(attr_type)  # __jm__ probably not gonna work

    #     if attr_str == "stop_epoch":
    #         print(attr_str, kwargs)

    #     parser.add_argument(f"--{attr_str}", **kwargs)

    # for arg in [
    #     "seed",
    #     "dataset",
    #     "model",
    #     "method",
    #     "train_n_way",
    #     "test_n_way",
    #     "n_shot",
    #     "train_aug",
    #     "checkpoint_suffix",
    #     "lr",
    #     "optim",
    #     "n_val_perms",
    #     "lr_scheduler",
    #     "milestones",
    #     "maml_save_feature_network",
    #     "maml_adapt_classifier",
    #     "evaluate_model",
    # ]:
    #     add_argument(arg)

    # if script == "train":
    #     for arg in [
    #         "num_classes",
    #         "save_freq",
    #         "start_epoch",
    #         "stop_epoch",
    #         "resume",
    #         "warmup",
    #         "es_epoch",
    #         "es_threshold",
    #         "eval_freq",
    #     ]:
    #         add_argument(arg)

    # elif script == "save_features":
    #     for arg in [
    #         "split",
    #         "save_iter",
    #     ]:
    #         add_argument(arg)

    # elif script == "test":
    #     for arg in [
    #         "split",
    #         "save_iter",
    #         "adaptation",
    #         "repeat",
    #     ]:
    #         add_argument(arg)
    # else:
    #     raise ValueError("Unknown script")

    # parser = hn_args.add_hn_args_to_parser(parser)


# __jm__ leave this for later
# def parse_args_regression(script):
#     parser = argparse.ArgumentParser(description="few-shot script %s" % (script))
#     parser.add_argument(
#         "--seed",
#         default=0,
#         type=int,
#         help="Seed for Numpy and pyTorch. Default: 0 (None)",
#     )
#     parser.add_argument("--model", default="Conv3", help="model: Conv{3} / MLP{2}")
#     parser.add_argument("--method", default="DKT", help="DKT / transfer")
#     parser.add_argument("--dataset", default="QMUL", help="QMUL / sines")
#     parser.add_argument(
#         "--spectral",
#         action="store_true",
#         help="Use a spectral covariance kernel function",
#     )

#     if script == "train_regression":
#         parser.add_argument("--start_epoch", default=0, type=int, help="Starting epoch")
#         parser.add_argument(
#             "--stop_epoch", default=100, type=int, help="Stopping epoch"
#         )  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
#         parser.add_argument(
#             "--resume",
#             action="store_true",
#             help="continue from previous trained model with largest epoch",
#         )
#     elif script == "test_regression":
#         parser.add_argument(
#             "--n_support",
#             default=5,
#             type=int,
#             help="Number of points on trajectory to be given as support points",
#         )
#         parser.add_argument(
#             "--n_test_epochs", default=10, type=int, help="How many test people?"
#         )
#     return parser.parse_args()
