import argparse
from enum import StrEnum, auto
import glob
import os
import sys
from pathlib import Path
from typing import Literal, Optional
import pytorch_lightning as pl
from methods.DKT import DKT
from methods.baselinetrain import BaselineTrain
from methods.hypernets.bayeshmaml import BayesHMAML
from methods.hypernets.hypermaml import HyperMAML
from methods.hypernets.hypernet_kernel import HyperShot
from methods.hypernets.hypernet_poc import HyperNetPOC, HypernetPPA
from methods.maml import MAML
from methods.matchingnet import MatchingNet
from methods.protonet import ProtoNet
import neptune
from neptune.exceptions import NeptuneException

import numpy as np
from neptune import Run

import backbone
import configs
import hn_args


class Arg:
    class Method(StrEnum):
        baseline = auto()
        baselinepp = "baseline++"
        DKT = "DKT"
        protonet = auto()
        matchingnet = auto()
        relationnet = auto()
        relationnet_softmax = auto()
        maml = auto()
        maml_approx = auto()
        hyper_maml = auto()
        bayes_hmaml = auto()
        hyper_shot = auto()
        hn_ppa = auto()
        hn_poc = auto()

    class Model(StrEnum):
        Conv4 = "Conv4"
        Conv4Pool = "Conv4Pool"
        Conv4S = "Conv4S"
        Conv6 = "Conv6"
        ResNet10 = "ResNet10"
        ResNet18 = "ResNet18"
        ResNet34 = "ResNet34"
        ResNet50 = "ResNet50"
        ResNet101 = "ResNet101"
        Conv4WithKernel = "Conv4WithKernel"
        ResNetWithKernel = "ResNetWithKernel"

    class Dataset(StrEnum):
        CUB = "CUB"
        miniImagenet = auto()
        cross = auto()
        omniglot = auto()
        # emnist = auto()
        cross_char = auto()

    class Optim(StrEnum):
        adam = auto()
        sgd = auto()

    class Scheduler(StrEnum):
        none = auto()
        multisteplr = auto()
        cosine = auto()
        reducelronplateau = auto()

    class Split(StrEnum):
        novel = auto()
        base = auto()
        val = auto()

    @classmethod
    def list(_self, cls) -> list[str]:
        return list(map(lambda c: c.value, cls))


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


class ParamStruct:
    seed: int = 0
    "Seed for Numpy and pyTorch."

    dataset = Arg.Dataset.CUB
    "The dataset used for training the model. Refer to Arg.Dataset for allowed values"

    model = Arg.Model.Conv4
    "The model used for prediction. Refer to Arg.Model for allowed values"
    # 50 and 101 are not used in the paper

    method = Arg.Method.baseline
    "The method utilized in conjunction with the model. Refer to Arg.Method for allowed values"
    # relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency

    train_n_way = 5
    "Class num to classify for training"
    # baseline and baseline++ ignore this parameter

    test_n_way = 5
    "Class num to classify for testing (validation)"
    # baseline and baseline++ ignore this parameter

    n_shot = 5
    "Number of labeled data in each class, same as n_support"
    # baseline and baseline++ only use this parameter in finetuning

    train_aug = False
    "Whether to perform data augmentation during training"

    checkpoint_suffix = ""
    "Suffix for custom experiment differentiation"
    # saved in save/checkpoints/[dataset]

    lr = 1e-3
    "Learning rate"

    optim = Arg.Optim.adam
    "Optimizer"

    n_val_perms = 1
    "Number of task permutations in evaluation."

    lr_scheduler = Arg.Scheduler.none
    "LR scheduler"

    milestones: Optional[list[int]] = None
    "Milestones for multisteplr"

    maml_save_feature_network = False
    "Whether to save feature net used in MAML"

    maml_adapt_classifier = False
    "Adapt only the classifier during second gradient calculation"

    evaluate_model = False
    "Skip train phase and perform final test"

    # region train

    num_classes = 200
    "Total number of classes in softmax, only used in baseline"
    # make it larger than the maximum label value in base class

    save_freq = 500
    "Save frequency"

    start_epoch = 0
    "Starting epoch"

    stop_epoch: Optional[int] = None
    "Stopping epoch"
    # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py

    resume = False
    "Continue from previous trained model with largest epoch"

    warmup = False
    "Continue from baseline, neglected if resume is true"
    # never used in the paper

    es_epoch = 250
    "Check if val accuracy threshold achieved at this epoch, stop if not."

    es_threshold: float = 50.0
    "Validation accuracy threshold for early stopping."

    eval_freq = 1
    "Evaluation frequency"

    # endregion

    split = Arg.Split.novel
    "Split dataset into /base/val/novel/"
    # default novel, but you can also test base/val class accuracy if you want

    save_iter: Optional[int] = None
    "save feature from the model trained in x epoch, use the best model if x is None"

    adaptation = False
    "Further adaptation in test time or not"

    repeat = 5
    "Repeat the test N times with different seeds and take the mean. The seeds range is [seed, seed+repeat]"

    n_query: int | None = None
    "This parameter is computed at runtime based on n_way"


class ParamHolder:
    def __init__(self, params):
        super().__init__()
        self.params: ParamStruct = params
        self.history = set()

    def __getattr__(self, item):
        it = getattr(self.params, item)
        if item not in self.history:
            print("Getting", item, "=", it)
            self.history.add(item)
        return it

    def get_ignored_args(self):
        return sorted([k for k in vars(self.params).keys() if k not in self.history])


def parse_args(script: Literal["train", "test", "save_features"]):
    parser = argparse.ArgumentParser(
        description=f"few-shot script {script}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    def add_argument(attr: str):
        attr_str = attr
        attr = getattr(ParamStruct, attr)
        kwargs = {
            "default": attr,
            "type": type(attr),
            "help": attr.__doc__,
        }
        if type(attr) is bool:
            kwargs["action"] = "store_true"
        elif attr_str in ["milestones"]:
            kwargs["nargs"] = "+"
            kwargs["type"] = int
        elif isinstance(StrEnum, attr):
            kwargs["choices"] = Arg.list(type(attr))  # __jm__ probably not gonna work

        print(attr_str, kwargs)

        parser.add_argument(f"--{attr_str}", kwargs=kwargs)

    map(
        add_argument,
        [
            "seed",
            "dataset",
            "model",
            "method",
            "train_n_way",
            "test_n_way",
            "n_shot",
            "train_aug",
            "checkpoint_suffix",
            "lr",
            "optim",
            "n_val_perms",
            "lr_scheduler",
            "milestones",
            "maml_save_feature_network",
            "maml_adapt_classifier",
            "evaluate_model",
        ],
    )

    if script == "train":
        map(
            add_argument,
            [
                "num_classes",
                "save_freq",
                "start_epoch",
                "stop_epoch",
                "resume",
                "warmup",
                "es_epoch",
                "es_threshold",
                "eval_freq",
            ],
        )

    elif script == "save_features":
        map(
            add_argument,
            [
                "split",
                "save_iter",
            ],
        )
    elif script == "test":
        map(
            add_argument,
            [
                "split",
                "save_iter",
                "adaptation",
                "repeat",
            ],
        )
    else:
        raise ValueError("Unknown script")

    parser = hn_args.add_hn_args_to_parser(parser)
    return ParamHolder(parser.parse_args())


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
    else:
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
        run["params"] = vars(params.params)
        run["cmd"] = f"python {' '.join(sys.argv)}"
        return run

    except NeptuneException as e:
        print("Cannot initialize neptune because of", e)
        return None
