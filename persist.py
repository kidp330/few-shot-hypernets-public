import os
import sys

import configs

from pathlib import Path

from io_params import ParamHolder


def save_run_params(checkpoint_dir: Path, params: ParamHolder):
    params_save_dir = checkpoint_dir / "args.json"
    if params.args_file == params_save_dir:
        return

    params.save(params_save_dir)
    with (checkpoint_dir / "rerun.sh").open("w") as f:
        print("python", " ".join(sys.argv), file=f)

# def get_assigned_file(checkpoint_dir, num):
#     assign_file = os.path.join(checkpoint_dir, "{:d}.tar".format(num))
#     return assign_file


def get_checkpoint_file(checkpoint_dir: Path) -> Path:
    pl_dir = checkpoint_dir / "lightning_logs"
    checkpoints_glob = pl_dir.glob("version_*/checkpoints/*")
    latest_checkpoint = max(checkpoints_glob, key=os.path.getctime)
    return latest_checkpoint

# def get_best_file(checkpoint_dir):
#     best_file = os.path.join(checkpoint_dir, "best_model.tar")
#     if os.path.isfile(best_file):
#         return best_file
#     return get_resume_file(checkpoint_dir)

# TODO: hash params instead? like nix


def get_checkpoint_dir(params: ParamHolder) -> Path:
    params.checkpoint_dir = (
        configs.save_dir / "checkpoints" / params.dataset / params.model / params.method
    )
    # if params.train_aug:
    #     params.checkpoint_dir += "_aug"
    # if not params.method in ["baseline", "baseline++"]:
    #     params.checkpoint_dir += "_%dway_%dshot" % (
    #         params.train_n_way, params.n_shot)
    # if params.checkpoint_suffix != "":
    #     params.checkpoint_dir = params.checkpoint_dir + "_" + params.checkpoint_suffix
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(f"{params.checkpoint_dir=}")

    return params.checkpoint_dir
