import os
import random
import time
from pathlib import Path
from typing import Type

import numpy as np
import torch
import torch.optim
import torch.utils.data.sampler
import setup
import backbone
import configs
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineFinetune
from methods.DKT import DKT
from methods.hypernets import hypernet_types
from methods.hypernets.bayeshmaml import BayesHMAML
from methods.hypernets.hypermaml import HyperMAML
from methods.hypernets.hypernet_poc import HyperNetPOC
from methods.maml import MAML
from methods.matchingnet import MatchingNet
from methods.protonet import ProtoNet
from methods.relationnet import RelationNet

import setup
import persist
import pytorch_lightning as pl
from io_params import Arg, ParamHolder


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


def feature_evaluation(
    cl_data_file, model, n_way=5, n_support=5, n_query=15, adaptation=False
):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append(
            [np.squeeze(img_feat[perm_ids[i]])
             for i in range(n_support + n_query)]
        )  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    if adaptation:
        scores = model.set_forward_adaptation(z_all, is_feature=True)
    else:
        scores, _ = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc


def single_test(params):
    acc_all = []

    iter_num = 600

    # few_shot_params["n_query"] = 15

    # __jm__ huh
    # if params.dataset == "cross":
    #     if not Path(checkpoint_dir).exists():
    #         checkpoint_dir = checkpoint_dir.replace("cross", "miniImagenet")

    # assert Path(checkpoint_dir).exists(), checkpoint_dir

    # modelfile   = get_resume_file(checkpoint_dir)

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split

    # eval_time = 0
    # testing without features
    if params.method in [
        "maml",
        "maml_approx",
        "hyper_maml",
        "bayes_hmaml",
        "DKT",
    ] + list(
        hypernet_types.keys()
    ):  # maml do not support testing with feature
        if "Conv" in params.model:
            if params.dataset in ["omniglot", "cross_char"]:
                image_size = 28
            else:
                image_size = 84
        else:
            image_size = 224

        datamgr = SetDataManager(
            image_size, n_eposide=iter_num, **few_shot_params)

        if params.dataset == "cross":
            if split == "base":
                loadfile = configs.data_dir["miniImagenet"] + "all.json"
            else:
                loadfile = configs.data_dir["CUB"] + split + ".json"
        elif params.dataset == "cross_char":
            if split == "base":
                loadfile = configs.data_dir["omniglot"] + "noLatin.json"
            else:
                loadfile = configs.data_dir["emnist"] + split + ".json"
        else:
            loadfile = configs.data_dir[params.dataset] + split + ".json"

        novel_loader = datamgr.get_data_loader(loadfile, aug=False)
        if params.adaptation:
            model.task_update_num = (
                100 if params.hn_val_epochs == -1 else params.hn_val_epochs
            )
            # We perform adaptation on MAML simply by updating more times.

        model.eval()
        model.single_test = True

        if isinstance(model, (MAML, BayesHMAML, HyperMAML)):
            acc_mean, acc_std, eval_time, *_ = model.test_loop(
                novel_loader, return_std=True, return_time=True
            )
        else:
            acc_mean, acc_std, * \
                _ = model.test_loop(novel_loader, return_std=True)

    # testing with features
    else:
        novel_file = os.path.join(
            checkpoint_dir.replace(
                "checkpoints", "features"), split_str + ".hdf5"
        )  # defaut split = novel, but you can also test base or val classes
        cl_data_file = feat_loader.init_baloader(novel_file)

        for i in range(iter_num):
            acc = feature_evaluation(
                cl_data_file, model, adaptation=params.adaptation, **few_shot_params
            )
            acc_all.append(acc)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print(
            "%d Test Acc = %4.2f%% +- %4.2f%%"
            % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        )

    # __jm__ TODO: don't do this here, delegate to separate function after figuring out how PL saves test results
    # region save record/results.txt
    # with open("./record/results.txt", "a") as f:
    #     timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    #     aug_str = "-aug" if params.train_aug else ""
    #     aug_str += "-adapted" if params.adaptation else ""
    #     if params.method in ["baseline", "baseline++"]:
    #         exp_setting = "%s-%s-%s-%s%s %sshot %sway_test" % (
    #             params.dataset,
    #             split_str,
    #             params.model,
    #             params.method,
    #             aug_str,
    #             params.n_shot,
    #             params.test_n_way,
    #         )
    #     else:
    #         exp_setting = "%s-%s-%s-%s%s %sshot %sway_train %sway_test" % (
    #             params.dataset,
    #             split_str,
    #             params.model,
    #             params.method,
    #             aug_str,
    #             params.n_shot,
    #             params.train_n_way,
    #             params.test_n_way,
    #         )
    #     acc_str = "%d Test Acc = %4.2f%% +- %4.2f%%" % (
    #         iter_num,
    #         acc_mean,
    #         1.96 * acc_std / np.sqrt(iter_num),
    #     )
    #     f.write("Time: %s, Setting: %s, Acc: %s \n" %
    #             (timestamp, exp_setting, acc_str))
    # endregion

    print("Test loop time:", eval_time)
    return acc_mean, eval_time


def main():
    params = ParamHolder().parse_args()
    if Path(params.args_file).is_file():
        cli_resume = params.resume
        params.load(params.args_file)
        # overwrite
        params.resume = cli_resume

    if params.dataset in ["omniglot", "cross_char"]:
        assert params.model == "Conv4" and not params.train_aug, (
            f"model = {params.model}, train_aug= {params.train_aug} "
            f"omniglot only support Conv4 without augmentation"
        )

    checkpoint_dir = persist.get_checkpoint_dir(params)
    persist.save_run_params(checkpoint_dir, params)
    checkpoint_file = persist.get_checkpoint_file(
        checkpoint_dir) if params.resume else None
    loggers = setup.setup_loggers(checkpoint_dir, params)

    _set_seed(params.seed)

    model = setup.initialize_model(params)
    test_dataloader = setup.initialize_test_dataloader(params)
    # print({b[0].shape for b in test_dataloader})

    optimizer: dict[Arg.Optim, torch.optim.Optimizer] = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD
    }[params.optim](model.parameters(), lr=params.lr)

    scheduler = setup.get_scheduler(params, optimizer)

    # class ConfigureOptimizers(pl.Callback):
    #     def configure_optimizers(self):
    #         return {"optimizer": optimizer, "lr_scheduler": scheduler}
    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
    }
    # HACK, TODO: check if this works

    trainer = pl.Trainer(
        logger=loggers,
        max_epochs=params.stop_epoch,
        # deterministc=,
        # benchmark=,
        # profiler=,
        # detect_anomaly=,
    )

    print("Starting testing")
    print("Params accessed until this point:")
    print("\n\t".join(sorted(params.history)))
    print("Params ignored until this point:")
    print("\n\t".join(params.get_ignored_args()))

    return trainer.test(
        model=model,
        dataloaders=test_dataloader,
        ckpt_path=params.ckpt_path
    )


if __name__ == "__main__":
    main()
