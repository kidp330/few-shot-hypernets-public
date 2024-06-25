from typing import Optional
from tap import Tap
from pathlib import Path
from parsers.types.general import *


class GeneralParams(Tap):
    seed: Optional[int]
    "Seed for Numpy and pyTorch. Run is non-deterministic when not set"

    dataset: Dataset
    "The dataset used for training the model. Refer to Dataset for allowed values"

    model: Model = 'Conv4'
    "The model used for prediction. Refer to Model for allowed values"
    # 50 and 101 are not used in the paper

    method: Method
    "The method utilized in conjunction with the model. Refer to Method for allowed values"
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

    optim: Optim = 'adam'
    "Optimizer"

    n_val_perms: int = 1
    "Number of task permutations in evaluation."

    lr_scheduler: Scheduler = 'none'
    "LR scheduler"

    milestones: list[int] | None = None
    "Milestones for multisteplr"

    maml_save_feature_network = False
    "Whether to save feature net used in MAML"

    maml_only_adapt_classifier = False
    "Adapt only the classifier during second gradient calculation"

    # region test

    split: Split = 'novel'
    "Split dataset into /base/val/novel/"
    # default novel, but you can also test base/val class accuracy if you want

    save_iter: Optional[int] = None
    "save feature from the model trained in x epoch, use the best model if x is None"

    adaptation = False
    "Further adaptation in test time or not"

    # endregion

    repeat = 5
    "Repeat the test N times with different seeds and take the mean. The seeds range is [seed, seed+repeat]"

    checkpoint_dir: Optional[Path] = None

    # __jm__ TODO:
    args_file: Optional[Path] = None
    "[NOT IMPLEMENTED] Path to a .json file specifying arguments of a previous run may be provided"
