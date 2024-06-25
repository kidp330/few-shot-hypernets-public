import json
import sys
from collections import defaultdict
from typing import Any, Literal, Sized,  Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from methods.meta_template import MetaTemplate
from neptune_utils import Run
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import os

import configs
from torch.optim.lr_scheduler import LRScheduler
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
# from methods.hypernets.hypernet_poc import HyperNetPOC
# from methods.hypernets import hypernet_types
from methods.maml import MAML
from methods.hypernets.bayeshmaml import BayesHMAML
from methods.hypernets.hypermaml import HyperMAML
from io_utils import get_resume_file
import neptune_utils as neptune
import shutil

import matplotlib.pyplot as plt
from pathlib import Path

from parsers.train import TrainParams
from parsers.types.general import Dataset, Optim, Scheduler
from save_features import do_save_fts

from common import set_seed, setup_model, _image_size, params_guards


def set_optimizer(optim: Optim, parameters, lr: float):
    return {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }[optim](parameters, lr=lr)


def load_metrics(checkpoint_dir: Path):
    with (checkpoint_dir / "metrics.json").open("r") as f:
        metrics_per_epoch = defaultdict(list, json.load(f))
        # assert all(len(metric) == start_epoch for metric in metrics_per_epoch.values())
        return metrics_per_epoch


def save_epoch_state(epoch: int, model: MetaTemplate, checkpoint_dir: Path, filename: str):
    torch.save({'epoch': epoch, 'state': model.state_dict()},
               checkpoint_dir / f'{filename}.tar')


def train(base_loader: DataLoader, val_loader: DataLoader,
          model: MetaTemplate,
          checkpoint_dir: Path,
          start_epoch: int, stop_epoch: int,
          params: TrainParams,
          neptune_run: Optional[Run] = None):
    print(f"Total epochs: {stop_epoch}")

    optimizer = set_optimizer(params.optim, model.parameters(), params.lr)

    max_acc = 0
    max_train_acc = 0

    max_acc_adaptation_dict = defaultdict(list)
    if params.test_set_forward_with_adaptation:
        max_acc_adaptation_dict["adaptation/accuracy/val_max"] = [0] * \
            (params.hn_val_epochs + 1)
        max_acc_adaptation_dict["adaptation/accuracy/val_support_max"] = [None] + \
            [0] * params.hn_val_epochs

    metrics_per_epoch = load_metrics(checkpoint_dir) \
        if params.resume \
        else defaultdict(list)

    if metrics_per_epoch["accuracy/val_max"] != []:
        max_acc = metrics_per_epoch["accuracy/val_max"][-1]
    if metrics_per_epoch["accuracy/train_max"] != []:
        max_train_acc = metrics_per_epoch["accuracy/train_max"][-1]

    scheduler = get_scheduler(
        params.lr_scheduler, params.milestones, optimizer, stop_epoch)

    print("Starting training")
    # print("Params accessed until this point:")
    # print("\n\t".join(sorted(params.history)))
    # print("Params ignored until this point:")
    # print("\n\t".join(params.get_ignored_args()))
    # __jm__ need issue for this

    for epoch in range(start_epoch, stop_epoch):
        if epoch >= params.es_epoch and max_acc < params.es_threshold:
            print(f"Breaking training at {epoch=} because {max_acc=} is lower than {params.es_threshold=}")  # nopep8
            break
        # __jm__ commenting out for hyperMAML, might break other models
        # model.epoch = epoch
        # model.start_epoch = start_epoch
        # model.stop_epoch = stop_epoch

        model.train()
        metrics = model.train_loop(epoch, base_loader, optimizer)
        scheduler.step()

        if epoch % params.eval_freq != 0 and epoch not in [params.es_epoch - 1, stop_epoch - 1]:
            continue

        # model.eval()
        test_results = model.test_loop(val_loader)
        acc = test_results.accuracy_mean
        test_loop_metrics = test_results.metrics

        print(f"Epoch {epoch}/{stop_epoch}  | Max test acc {max_acc:.2f} | Test acc {acc:.2f} | Metrics: {test_loop_metrics}")  # nopep8

        metrics: dict[str, Any] = metrics or dict()
        metrics["lr"] = scheduler.get_last_lr()
        metrics["accuracy/val"] = acc
        metrics["accuracy/val_max"] = max_acc
        metrics["accuracy/train_max"] = max_train_acc
        metrics = {
            **metrics,
            **test_loop_metrics,
            **max_acc_adaptation_dict
        }

        if metrics["accuracy/train"] > max_train_acc:
            max_train_acc = metrics["accuracy/train"]

        if params.test_set_forward_with_adaptation:

            def elementwise_max(xs, ys): return [
                max(*pair) for pair in zip(xs, ys)]

            max_acc_adaptation_dict["adaptation/accuracy/val_max"] = elementwise_max(
                max_acc_adaptation_dict["adaptation/accuracy/val_max"], metrics["adaptation/accuracy/val"])

            max_acc_adaptation_dict["adaptation/accuracy/val_support_max"] = elementwise_max(
                max_acc_adaptation_dict["adaptation/accuracy/val_support_max"], metrics["adaptation/accuracy/val_support_acc"])

        path_model = checkpoint_dir / 'last_model.tar'
        path_feature = checkpoint_dir / 'last_feature_net.tar'

        torch.save({'epoch': epoch, 'state': model.state_dict()}, path_model)
        if params.maml_save_feature_network and isinstance(model, (MAML, HyperMAML, BayesHMAML)):
            torch.save(
                {'epoch': epoch, 'state': model.feature.state_dict()}, path_feature)

        if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
            print("--> Best model! save...")
            max_acc = acc
            shutil.copyfile(path_model, checkpoint_dir / 'best_model.tar')
            if path_feature.is_file():
                shutil.copyfile(path_feature, checkpoint_dir /
                                'best_feature_net.tar')

        if epoch % params.save_freq == 0 or epoch == stop_epoch - 1:
            shutil.copyfile(path_model, checkpoint_dir / f'{epoch}.tar')

        with (checkpoint_dir / "metrics.json").open("a") as f:
            json.dump(metrics, f, indent=2)

        for m, v in metrics.items():

            metrics_per_epoch[m].append(v)

            if neptune_run is not None:
                neptune_run[m].log(v, step=epoch)

    if neptune_run is None:
        return
    neptune.track(neptune_run, checkpoint_dir, "best_model")
    neptune.track(neptune_run, checkpoint_dir, "last_model")
    if params.maml_save_feature_network:
        neptune.track(neptune_run, checkpoint_dir, "best_feature_net")
        neptune.track(neptune_run, checkpoint_dir, "last_feature_net")


def plot_metrics(metrics_per_epoch: dict[str, list[float] | float], epoch: int, fig_dir: Path):
    for m, values in metrics_per_epoch.items():
        plt.figure()
        if "accuracy" in m:
            plt.ylim((0, 100))
        if isinstance(values, Sized):
            means: list[float] = [float(np.mean(v)) if isinstance(
                v, list) else v for v in values]
            stds: list[float] = [float(np.std(v)) if isinstance(
                v, list) else v for v in values]

            plt.errorbar(
                list(range(len(values))),
                means,
                stds,
                ecolor="black",
                fmt="o",
            )
        plt.grid()
        plt.title(f"{epoch}- {m}")
        plt.savefig(fig_dir / f"{m}.png")
        plt.close()


def get_scheduler(scheduler: Scheduler, milestones: Optional[list[int]], optimizer: torch.optim.Optimizer, stop_epoch: int) -> LRScheduler:
    match scheduler:
        case "multisteplr":
            milestones = list(range(0, stop_epoch,
                                    stop_epoch // 4))[1:]
            if params.milestones is not None:
                milestones = params.milestones

            return lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                            gamma=0.3)
        case "none":
            return lr_scheduler.MultiStepLR(optimizer,
                                            milestones=list(
                                                range(0, stop_epoch, stop_epoch // 4))[1:],
                                            gamma=1)
        case "cosine":
            return lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=stop_epoch
            )
        case "reducelronplateau":
            raise NotImplemented()


def get_dataloaders(params: TrainParams) -> tuple[DataLoader, DataLoader]:
    match params.method:
        case 'baseline' | 'baseline++':
            base_datamgr = SimpleDataManager(image_size, batch_size=16)
            base_loader = base_datamgr.get_data_loader(
                base_file, aug=params.train_aug)
            val_datamgr = SimpleDataManager(image_size, batch_size=64)
            val_loader = val_datamgr.get_data_loader(val_file, aug=False)
            return base_loader, val_loader
        case _:
            n_query = max(1, int(
                # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
                16 * params.test_n_way / params.train_n_way))

            # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor
            base_datamgr = SetDataManager(
                image_size,
                n_way=params.test_n_way,
                n_support=params.n_shot,
                n_query=n_query)  # n_eposide=100
            base_loader = base_datamgr.get_data_loader(
                base_file, aug=params.train_aug)

            val_datamgr = SetDataManager(
                image_size, n_way=params.test_n_way, n_support=params.n_shot, n_query=n_query)
            val_loader = val_datamgr.get_data_loader(val_file, aug=False)

            return base_loader, val_loader


if __name__ == '__main__':
    backbone.set_default_device()
    params = TrainParams().parse_args()
    set_seed(params.seed)

    def _files(dataset: Dataset) -> tuple[Path, Path]:
        match dataset:
            case 'cross':
                return configs.data_dir['miniImagenet'] / 'all.json', configs.data_dir['CUB'] / 'val.json'
            case 'cross_char':
                return configs.data_dir['omniglot'] / 'noLatin.json', configs.data_dir['emnist'] / 'val.json'
            case _:
                return configs.data_dir[params.dataset] / 'base.json', configs.data_dir[params.dataset] / 'val.json'
    base_file, val_file = _files(params.dataset)

    image_size = _image_size(params)

    def stop_epoch_default():
        if params.method in ['baseline', 'baseline++']:
            match params.dataset:
                case 'omniglot' | 'cross_char':
                    return 5
                case 'CUB':
                    # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
                    return 200
                case 'miniImagenet', 'cross':
                    return 400
                case _:
                    return 400  # default
        if params.n_shot == 1:
            return 600
        elif params.n_shot == 5:
            return 400
        return 600  # default
    if params.stop_epoch is None:
        params.stop_epoch = stop_epoch_default()

    model = setup_model(params)

    params.checkpoint_dir = configs.save_dir / 'checkpoints' / \
        params.dataset / params.model / params.method

    if params.train_aug:
        params.checkpoint_dir /= '_aug'
    if params.method not in ['baseline', 'baseline++']:
        params.checkpoint_dir /= f'_{params.train_n_way}way_{params.n_shot}shot'  # nopep8
    params.checkpoint_dir /= params.checkpoint_suffix

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(f"{params.checkpoint_dir=}")

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method in ['maml', 'maml_approx', 'hyper_maml', 'bayes_hmaml']:
        # maml use multiple tasks in one update
        # __jm__ HACK
        n_task = 4
        stop_epoch = params.stop_epoch * n_task

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        print(resume_file)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])
            print("Resuming training from", resume_file, "epoch", start_epoch)

    elif params.warmup:  # We also support warmup from pretrained baseline feature, but we never used it in our paper
        baseline_checkpoint_dir = configs.save_dir / 'checkpoints' / \
            params.dataset / params.model / 'baseline'
        if params.train_aug:
            baseline_checkpoint_dir /= '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        assert warmup_resume_file is not None
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.",
                                         "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    with (params.checkpoint_dir / "args.json").open("w") as f:
        # default is needed for Path-type parameters
        json.dump(params.as_dict(), f, indent=2, default=lambda obj: str(obj))

    with (params.checkpoint_dir / "rerun.sh").open("w") as f:
        print("python", " ".join(sys.argv), file=f)

    neptune_run = neptune.setup(params, model)
    train(*get_dataloaders(params), model, params.checkpoint_dir, start_epoch, stop_epoch, params,
          neptune_run=neptune_run)

    params.split = "novel"
    params.save_iter = -1

    try:
        do_save_fts(params)
    except Exception as e:
        print("Cannot save features bc of", e)

    # val_datasets = [params.dataset]
    # if params.dataset in ["cross", "miniImagenet"]:
    #     val_datasets = ["cross", "miniImagenet"]
    # for d in val_datasets:
    #     print("Evaluating on", d)
    #     params.dataset = d
    #     # num of epochs for finetuning on testing.
    #     for hn_val_epochs in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 200]:
    #         params.hn_val_epochs = hn_val_epochs
    #         params.hm_set_forward_with_adaptation = True
    #         # add default test params
    #         params.adaptation = True
    #         params.repeat = 5
    #         print(f"Testing with {hn_val_epochs=}")
    #         test_results = perform_test(params)
    #         if neptune_run is not None:
    #             neptune_run[f"full_test/{d}/metrics @ {
    #                 hn_val_epochs}"] = test_results
    #         neptune_run.stop()
