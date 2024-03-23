from enum import StrEnum, auto
from typing import Optional
from tap import Tap
from pathlib import Path


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


class ParamStruct(Tap):
    seed: int = 0
    "Seed for Numpy and pyTorch."

    dataset: Optional[Arg.Dataset] = None
    "The dataset used for training the model. Refer to Arg.Dataset for allowed values"

    model = Arg.Model.Conv4
    "The model used for prediction. Refer to Arg.Model for allowed values"
    # 50 and 101 are not used in the paper

    method: Optional[Arg.Method] = None
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

    checkpoint_suffix: str = ""
    "Suffix for custom experiment differentiation"
    # saved in save/checkpoints/[dataset]

    lr: float = 1e-3
    "Learning rate"

    optim = Arg.Optim.adam
    "Optimizer"

    n_val_perms: int = 1
    "Number of task permutations in evaluation."

    lr_scheduler = Arg.Scheduler.none
    "LR scheduler"

    milestones: list[int] | None = None
    "Milestones for multisteplr"

    maml_save_feature_network = False
    "Whether to save feature net used in MAML"

    maml_adapt_classifier = False
    "Adapt only the classifier during second gradient calculation"

    evaluate_model = False
    "Skip train phase and perform final test"

    # region train

    num_classes: int = 200
    "Total number of classes in softmax, only used in baseline"
    # make it larger than the maximum label value in base class

    save_freq: int = 500
    "Save frequency"
    # TODO: pass to pl.Trainer without perhaps defining a custom callback

    stop_epoch: Optional[int] = None
    "Stopping epoch"
    # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py

    resume: bool = False
    "Continue from previous trained model with largest epoch"

    warmup: bool = False
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

    n_query: Optional[int] = None
    "By default, this parameter is computed at runtime based on n_way"

    args_file: Optional[Path] = None
    "Path to a .json file specifying arguments of a previous run may be provided"


class ParamHolder(ParamStruct):
    def __init__(self):
        super().__init__()
        self.history = set()

    def __getattr__(self, item):
        it = super().__getattribute__(item)
        if item not in self.history:
            print("Getting", item, "=", it)
            self.history.add(item)
        return it

    # TODO: seems to be not working correctly after refactor
    def get_ignored_args(self) -> list[str]:
        return sorted([k for k in vars(self).keys() if k not in self.history])
