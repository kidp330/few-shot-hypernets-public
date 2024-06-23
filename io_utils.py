import argparse
import glob
import os
from pathlib import Path
from typing import Callable


import backbone

model_dict: dict[str, Callable] = dict(
    Conv4=backbone.Conv4,
    Conv4Pool=backbone.Conv4Pool,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101,
    Conv4WithKernel=backbone.Conv4WithKernel,
    ResNetWithKernel=backbone.ResNetWithKernel,
)


def parse_args_regression(script):
    parser = argparse.ArgumentParser(
        description="few-shot script %s" % (script))
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for Numpy and pyTorch. Default: 0 (None)",
    )
    parser.add_argument("--model", default="Conv3",
                        help="model: Conv{3} / MLP{2}")
    parser.add_argument("--method", default="DKT", help="DKT / transfer")
    parser.add_argument("--dataset", default="QMUL", help="QMUL / sines")
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="Use a spectral covariance kernel function",
    )

    if script == "train_regression":
        parser.add_argument("--start_epoch", default=0,
                            type=int, help="Starting epoch")
        parser.add_argument(
            "--stop_epoch", default=100, type=int, help="Stopping epoch"
        )  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument(
            "--resume",
            action="store_true",
            help="continue from previous trained model with largest epoch",
        )
    elif script == "test_regression":
        parser.add_argument(
            "--n_support",
            default=5,
            type=int,
            help="Number of points on trajectory to be given as support points",
        )
        parser.add_argument(
            "--n_test_epochs", default=10, type=int, help="How many test people?"
        )
    return parser.parse_args()


def get_assigned_file(checkpoint_dir: Path, num: int):
    assign_file = checkpoint_dir / f"{num}.tar"
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, "*.tar"))
    if len(filelist) == 0:
        return None
    last_model_files = [
        x for x in filelist if os.path.basename(x) == "last_model.tar"]
    if len(last_model_files) == 1:
        return last_model_files[0]

    filelist = [x for x in filelist if os.path.basename(x) != "best_model.tar"]
    max_epoch = max(int(os.path.splitext(os.path.basename(x))[0])
                    for x in filelist)
    resume_file = os.path.join(checkpoint_dir, "{:d}.tar".format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, "best_model.tar")
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
