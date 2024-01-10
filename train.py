import json
import sys

# from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import random
from methods.meta_template import MetaTemplate
from neptune import Run
import torch.optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os

import matplotlib.pyplot as plt


import configs
import backbone
from data.datamgr import DataManager, SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.hypernets import hypernet_types
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import (
    ParamStruct,
    model_dict,
    method_dict,
    parse_args,
    get_resume_file,
    setup_neptune,
    ParamHolder,
    Arg,
)

import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


# from neptune.types import File

# import matplotlib.pyplot as plt
from pathlib import Path

from save_features import do_save_fts
from test import perform_test


def _set_seed(seed, verbose=True):
    if seed != 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if verbose:
            print("[INFO] Setting SEED: " + str(seed))
    else:
        if verbose:
            print("[INFO] Setting SEED: None")


def train(
    base_loader: DataLoader,
    val_loader: DataLoader,
    model: pl.LightningModule,
    optimization: str,
    _start_epoch: int,
    stop_epoch: int,
    params: ParamHolder,
    *,
    neptune_run: Run | None = None,
):
    print("Tot epochs: " + str(stop_epoch))
    if optimization == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif optimization == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    else:
        raise ValueError(
            f"Unknown optimization {optimization}, please define by yourself"
        )
    scheduler = get_scheduler(params, optimizer, stop_epoch)

    class ConfigureOptimizers(pl.Callback):
        def configure_optimizers(self):
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(
            params.checkpoint_dir
        )  # __jm__ lightning should create a checkpoint itself

    loggers = [TensorBoardLogger(params.checkpoint_dir)]
    if neptune_run is not None:
        loggers.append(NeptuneLogger(run=neptune_run))

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=ConfigureOptimizers(),
        max_epochs=stop_epoch,
        # deterministc=,
        # benchmark=,
        # profiler=,
        # detect_anomaly=,
    )

    print("Starting training")
    print("Params accessed until this point:")
    print("\n\t".join(sorted(params.history)))
    print("Params ignored until this point:")
    print("\n\t".join(params.get_ignored_args()))

    trainer.fit(model, train_dataloaders=base_loader, val_dataloaders=val_loader)

    # deal with
    # es_epoch, es_threshold
    # eval_freq

    # delta_params_list = []
    # log lr if not logged already?
    # for epoch in range(start_epoch, stop_epoch):
    #     if epoch >= params.es_epoch:
    #         if max_acc < params.es_threshold:
    #             print(
    #                 "Breaking training at epoch",
    #                 epoch,
    #                 "because max accuracy",
    #                 max_acc,
    #                 "is lower than threshold",
    #                 params.es_threshold,
    #             )
    #             break
    #     model.epoch = epoch
    #     model.start_epoch = start_epoch
    #     model.stop_epoch = stop_epoch

    #     model.train()
    #     if params.method in ["hyper_maml", "bayes_hmaml"]:
    #         metrics = model.train_loop(epoch, base_loader, optimizer)
    #     else:
    #         metrics = model.train_loop(
    #             epoch, base_loader, optimizer
    #         )  # model are called by reference, no need to return

    #     scheduler.step()
    #     model.eval()

    #     delta_params = metrics.pop("delta_params", None)
    #     if delta_params is not None:
    #         delta_params_list.append(delta_params)

    #     if (epoch % params.eval_freq == 0) or epoch in [
    #         params.es_epoch - 1,
    #         stop_epoch - 1,
    #     ]:
    #         try:
    #             acc, test_loop_metrics = model.test_loop(val_loader)
    #         except:
    #             acc = model.test_loop(val_loader)
    #             test_loop_metrics = dict()
    #         print(
    #             f"Epoch {epoch}/{stop_epoch}  | Max test acc {max_acc:.2f} | Test acc {acc:.2f} | Metrics: {test_loop_metrics}"
    #         )

    #         metrics = metrics or dict()
    #         metrics["lr"] = scheduler.get_lr()
    #         metrics["accuracy/val"] = acc
    #         metrics["accuracy/val_max"] = max_acc
    #         metrics["accuracy/train_max"] = max_train_acc
    #         # metrics = {**metrics, **test_loop_metrics, **max_acc_adaptation_dict}

    #         # __jm__ huh
    #         if params.hm_set_forward_with_adaptation:
    #             for i in range(params.hn_val_epochs + 1):
    #                 if i != 0:
    #                     metrics[
    #                         f"accuracy/val_support_max@-{i}"
    #                     ] = max_acc_adaptation_dict[f"accuracy/val_support_max@-{i}"]
    #                 metrics[f"accuracy/val_max@-{i}"] = max_acc_adaptation_dict[
    #                     f"accuracy/val_max@-{i}"
    #                 ]

    #         if metrics["accuracy/train"] > max_train_acc:
    #             max_train_acc = metrics["accuracy/train"]

    #         if params.hm_set_forward_with_adaptation:
    #             for i in range(params.hn_val_epochs + 1):
    #                 if (
    #                     i != 0
    #                     and metrics[f"accuracy/val_support_acc@-{i}"]
    #                     > max_acc_adaptation_dict[f"accuracy/val_support_max@-{i}"]
    #                 ):
    #                     max_acc_adaptation_dict[
    #                         f"accuracy/val_support_max@-{i}"
    #                     ] = metrics[f"accuracy/val_support_acc@-{i}"]

    #                 if (
    #                     metrics[f"accuracy/val@-{i}"]
    #                     > max_acc_adaptation_dict[f"accuracy/val_max@-{i}"]
    #                 ):
    #                     max_acc_adaptation_dict[f"accuracy/val_max@-{i}"] = metrics[
    #                         f"accuracy/val@-{i}"
    #                     ]

    #         if (
    #             acc > max_acc
    #         ):  # for baseline and baseline++, we don't use validation here so we let acc = -1
    #             print("--> Best model! save...")
    #             max_acc = acc
    #             outfile = os.path.join(params.checkpoint_dir, "best_model.tar")
    #             torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

    #             if params.maml_save_feature_network and params.method in [
    #                 "maml",
    #                 "hyper_maml",
    #                 "bayes_hmaml",
    #             ]:
    #                 outfile = os.path.join(
    #                     params.checkpoint_dir, "best_feature_net.tar"
    #                 )
    #                 torch.save(
    #                     {"epoch": epoch, "state": model.feature.state_dict()}, outfile
    #                 )

    #         outfile = os.path.join(params.checkpoint_dir, "last_model.tar")
    #         torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

    #         if params.maml_save_feature_network and params.method in [
    #             "maml",
    #             "hyper_maml",
    #             "bayes_hmaml",
    #         ]:
    #             outfile = os.path.join(params.checkpoint_dir, "last_feature_net.tar")
    #             torch.save(
    #                 {"epoch": epoch, "state": model.feature.state_dict()}, outfile
    #             )

    #         if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
    #             outfile = os.path.join(params.checkpoint_dir, "{:d}.tar".format(epoch))
    #             torch.save({"epoch": epoch, "state": model.state_dict()}, outfile)

    #         if metrics is not None:
    #             for k, v in metrics.items():
    #                 metrics_per_epoch[k].append(v)

    #         with (Path(params.checkpoint_dir) / "metrics.json").open("w") as f:
    #             json.dump(metrics_per_epoch, f, indent=2)

    #         if neptune_run is not None:
    #             for m, v in metrics.items():
    #                 neptune_run[m].log(v, step=epoch)

    # if neptune_run is not None:
    #     neptune_run["best_model"].track_files(
    #         os.path.join(params.checkpoint_dir, "best_model.tar")
    #     )
    #     neptune_run["last_model"].track_files(
    #         os.path.join(params.checkpoint_dir, "last_model.tar")
    #     )

    #     if params.maml_save_feature_network:
    #         neptune_run["best_feature_net"].track_files(
    #             os.path.join(params.checkpoint_dir, "best_feature_net.tar")
    #         )
    #         neptune_run["last_feature_net"].track_files(
    #             os.path.join(params.checkpoint_dir, "last_feature_net.tar")
    #         )

    # if len(delta_params_list) > 0 and params.hm_save_delta_params:
    #     with (
    #         Path(params.checkpoint_dir)
    #         / f"delta_params_list_{len(delta_params_list)}.json"
    #     ).open("w") as f:
    #         json.dump(delta_params_list, f, indent=2)

    return model


def plot_metrics(
    metrics_per_epoch: dict[str, float | list[float]], epoch: int, fig_dir: Path
):
    for m, values in metrics_per_epoch.items():
        plt.figure()
        if "accuracy" in m:
            plt.ylim((0, 100))
        plt.errorbar(
            list(range(len(values))),
            [np.mean(v) if isinstance(v, list) else v for v in values],
            [np.std(v) if isinstance(v, list) else 0 for v in values],
            ecolor="black",
            fmt="o",
        )
        plt.grid()
        plt.title(f"{epoch}- {m}")
        plt.savefig(fig_dir / f"{m}.png")
        plt.close()


def get_scheduler(params, optimizer, stop_epoch=None) -> lr_scheduler._LRScheduler:
    if params.lr_scheduler == "cosine":
        T_0 = stop_epoch if stop_epoch is not None else params.stop_epoch // 4
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0)
    elif params.lr_scheduler in ["none", "multisteplr"]:
        if params.milestones is not None:
            milestones = params.milestones
        else:
            milestones = list(range(0, params.stop_epoch, params.stop_epoch // 4))[1:]
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=1,
        )
    raise TypeError(params.lr_scheduler)


# region dataloaders
def get_image_size(model: Arg.Model, dataset: Arg.Dataset):
    if "Conv" in model:
        if dataset in ["omniglot", "cross_char"]:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224
    return image_size


def get_train_val_files(dataset: Arg.Dataset) -> tuple[os.PathLike, os.PathLike]:
    if dataset == "cross":
        base_file = os.path.join(configs.data_dir["miniImagenet"], "all.json")
        val_file = os.path.join(configs.data_dir["CUB"], "val.json")
    elif dataset == "cross_char":
        base_file = os.path.join(configs.data_dir["omniglot"], "noLatin.json")
        val_file = os.path.join(configs.data_dir["emnist"], "val.json")
    else:
        base_file = os.path.join(configs.data_dir[dataset], "base.json")
        val_file = os.path.join(configs.data_dir[dataset], "val.json")
    return base_file, val_file


def get_train_val_dataloaders(
    params: ParamHolder,
    get_datamgrs_callback: Callable[[int], tuple[DataManager, DataManager]],
):
    image_size = get_image_size(params.model, params.dataset)
    train_val_files = get_train_val_files(params.dataset)

    base_file, val_file = train_val_files

    base_datamgr, val_datamgr = get_datamgrs_callback(image_size, train_val_files)

    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    return base_loader, val_loader


# endregion


# region baseline
def set_default_stop_epoch(params: ParamHolder):
    if params.method in ["baseline", "baseline++"]:
        if params.dataset in ["omniglot", "cross_char"]:
            params.stop_epoch = 5
        elif params.dataset in ["CUB"]:
            params.stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
        else:
            params.stop_epoch = 400  # default
    else:  # meta-learning methods
        if params.n_shot == 5:
            params.stop_epoch = 400
        else:
            params.stop_epoch = 600  # default


def setup_baseline_dataloaders(params: ParamHolder):
    return get_train_val_dataloaders(
        params,
        lambda image_size: (
            (base_datamgr := SimpleDataManager(image_size, batch_size=16)),
            (val_datamgr := SimpleDataManager(image_size, batch_size=64)),
            (base_datamgr, val_datamgr),
        )[-1],
    )


def setup_baseline_model(params: ParamHolder) -> BaselineTrain:
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

    return model


def setup_baseline(params: ParamHolder) -> tuple[BaselineTrain, DataLoader, DataLoader]:
    return setup_baseline_model(params), *setup_baseline_dataloaders(params)


# endregion

# region adaptive


def setup_adaptive_dataloaders(
    params: ParamHolder,
    train_few_shot_params: dict[str, int],
) -> tuple[DataLoader, DataLoader]:
    return get_train_val_dataloaders(
        params,
        lambda image_size: (
            (base_mgr := SetDataManager(image_size, **train_few_shot_params)),
            (val_mgr := SetDataManager(image_size, **train_few_shot_params)),
            (base_mgr, val_mgr),
        )[-1],
    )


def setup_adaptive_model(
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
        model_dict_mod = model_dict | {
            "Conv4": backbone.Conv4NP,
            "Conv6": backbone.Conv6NP,
            "Conv4S": backbone.Conv4SNP,
        }

        model = RelationNet(
            feature_model=lambda: model_dict_mod[params.model](flatten=False),
            loss_type="mse" if params.method == "relationnet" else "softmax",
            **train_few_shot_params,
        )
    elif params.method in ["maml", "maml_approx", "hyper_maml", "bayes_hmaml"]:
        # __jm__ huh
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(
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
            model.task_update_num = 1
            model.train_lr = 0.1

            params.stop_epoch *= model.n_task  # MAML runs a few tasks per epoch
    elif params.method in hypernet_types.keys():
        hn_type = hypernet_types[params.method]
        model = hn_type(
            model_dict[params.model], params=params, **train_few_shot_params
        )

    return model


def setup_adaptive(params: ParamHolder) -> tuple[MetaTemplate, DataLoader, DataLoader]:
    # __jm__ n_query is 'hardcoded' here - make it configurable?
    params.n_query: ParamStruct.n_query = max(
        1, int(16 * params.test_n_way / params.train_n_way)
    )  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    print(f"{params.n_query=}")

    train_few_shot_params = dict(
        n_way=params.train_n_way, n_support=params.n_shot, n_query=params.n_query
    )

    model = setup_adaptive_model(params, train_few_shot_params)
    dataloaders = setup_adaptive_dataloaders(params, train_few_shot_params)
    return model, *dataloaders


# endregion


def main():
    params = parse_args("train")
    _set_seed(params.seed)

    if params.dataset in ["omniglot", "cross_char"]:
        assert params.model == "Conv4" and not params.train_aug, (
            f"model = {params.model}, train_aug= {params.train_aug} "
            f"omniglot only support Conv4 without augmentation"
        )
        # params.model = 'Conv4S'
        # no need for this, since omniglot is loaded as RGB
    if params.stop_epoch is None:
        set_default_stop_epoch(params)

    if params.method in ["baseline", "baseline++"]:
        model, base_loader, val_loader = setup_baseline(params)
    else:
        model, base_loader, val_loader = setup_adaptive(params)

    params.checkpoint_dir = f"{configs.save_dir}/checkpoints/{params.dataset}/{params.model}_{params.method}"

    if params.train_aug:
        params.checkpoint_dir += "_aug"
    if not params.method in ["baseline", "baseline++"]:
        params.checkpoint_dir += "_%dway_%dshot" % (params.train_n_way, params.n_shot)
    if params.checkpoint_suffix != "":
        params.checkpoint_dir = params.checkpoint_dir + "_" + params.checkpoint_suffix
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(params.checkpoint_dir)

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        print(resume_file)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            params.start_epoch = tmp["epoch"] + 1
            model.load_state_dict(tmp["state"])
            print("Resuming training from", resume_file, "epoch", params.start_epoch)

    # We also support warmup from pretrained baseline feature, but we never used it in our paper
    elif params.warmup:
        baseline_checkpoint_dir = (
            f"{configs.save_dir}/checkpoints/{params.dataset}/{params.model}_baseline"
        )
        if params.train_aug:
            baseline_checkpoint_dir += "_aug"
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp["state"]
            state_keys = list(state.keys())
            for _i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace(
                        "feature.", ""
                    )  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError("No warm_up file")

    args_dict = vars(params.params)
    with (Path(params.checkpoint_dir) / "args.json").open("w") as f:
        json.dump(
            {
                k: v if isinstance(v, (int, str, bool, float)) else str(v)
                for (k, v) in args_dict.items()
            },
            f,
            indent=2,
        )

    with (Path(params.checkpoint_dir) / "rerun.sh").open("w") as f:
        print("python", " ".join(sys.argv), file=f)

    neptune_run = setup_neptune(params)
    if neptune_run is not None:
        neptune_run["model"] = str(model)

    # train and test are split -- __jm__

    if not params.evaluate_model:
        model = train(
            base_loader,
            val_loader,
            model,
            params.optim,
            params.start_epoch,
            params.stop_epoch,
            params,
            neptune_run=neptune_run,
        )

    params.split = "novel"
    params.save_iter = -1

    try:
        do_save_fts(params)
    except Exception as e:
        print("Cannot save features bc of", e)

    val_datasets = [params.dataset]
    if params.dataset in ["cross", "miniImagenet"]:
        val_datasets = ["cross", "miniImagenet"]

    for d in val_datasets:
        print("Evaluating on", d)
        params.dataset = d
        # num of epochs for finetuning on testing.
        for hn_val_epochs in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 100, 200]:
            params.hn_val_epochs = hn_val_epochs
            params.hm_set_forward_with_adaptation = True
            # add default test params
            params.adaptation = True
            params.repeat = 5
            print(f"Testing with {hn_val_epochs=}")
            test_results = perform_test(params)
            if neptune_run is not None:
                neptune_run[f"full_test/{d}/metrics @ {hn_val_epochs}"] = test_results
            neptune_run.stop()


if __name__ == "__main__":
    main()
