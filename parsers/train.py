from typing import Optional

from parsers.parsers import ParamHolder


class TrainParams(ParamHolder):
    num_classes: int = 200
    "total number of classes in softmax, only used in baseline"
    # make it larger than the maximum label value in base class

    save_freq: int = 500
    "Save frequency"

    start_epoch: int = 0
    "Starting epoch"

    stop_epoch: Optional[int] = None
    "Stopping epoch"
    # for meta-learning methods each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py

    resume: bool = False
    "continue from previous trained model with largest epoch"

    warmup: bool = False
    "continue from baseline, neglected if resume is true"
    # never used in the paper

    es_epoch: int = 250
    "Check if val accuracy threshold achieved at this epoch stop if not."

    es_threshold: float = 50.0
    "Val accuracy threshold for early stopping"

    eval_freq: int = 1
    "Evaluation frequency"
