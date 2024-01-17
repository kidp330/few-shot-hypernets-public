import argparse
import glob
import os
import sys
from pathlib import Path
from enum import StrEnum
from typing import Literal

import numpy as np
import pytorch_lightning as pl

import neptune
from neptune.exceptions import NeptuneException
from neptune import Run

from io_params import Arg, ParamStruct, ParamHolder
import backbone
import configs
import hn_args

from methods.DKT import DKT
from methods.baselinetrain import BaselineTrain
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


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, "{:d}.tar".format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, "*.tar"))
    if len(filelist) == 0:
        return None
    last_model_files = [x for x in filelist if os.path.basename(x) == "last_model.tar"]
    if len(last_model_files) == 1:
        return last_model_files[0]

    filelist = [x for x in filelist if os.path.basename(x) != "best_model.tar"]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, "{:d}.tar".format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, "best_model.tar")
    if os.path.isfile(best_file):
        return best_file
    return get_resume_file(checkpoint_dir)


def setup_neptune(params) -> Run | None:
    run_name = (
        Path(params.checkpoint_dir)
        .relative_to(Path(configs.save_dir) / "checkpoints")
        .name
    )
    run_file = Path(params.checkpoint_dir) / "NEPTUNE_RUN.txt"

    run_id = None
    if params.resume and run_file.exists():
        with run_file.open("r") as f:
            run_id = f.read()
            print("Resuming neptune run", run_id)

    try:
        run = neptune.init_run(
            name=run_name,
            source_files="**/*.py",
            tags=[params.checkpoint_suffix] if params.checkpoint_suffix != "" else [],
            with_id=run_id,
        )
        with run_file.open("w") as f:
            f.write(run["sys/id"].fetch())
            print("Starting neptune run", run["sys/id"].fetch())
        run["params"] = params.as_dict()
        run["cmd"] = f"python {' '.join(sys.argv)}"
        return run

    except NeptuneException as e:
        print("Cannot initialize neptune because of", e)
        return None
