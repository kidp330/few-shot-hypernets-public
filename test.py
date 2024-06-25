from pathlib import Path
import random
import time
from typing import Literal

import numpy as np
import torch
import torch.optim
import torch.utils.data.sampler

import backbone
import configs
from data.datamgr import SetDataManager
from io_utils import get_assigned_file, get_best_file, model_dict
from methods.DKT import DKT
from methods.hypernets.bayeshmaml import BayesHMAML
from methods.hypernets.hypermaml import HyperMAML
from methods.maml import MAML
from methods.matchingnet import MatchingNet
from methods.protonet import ProtoNet
from methods.relationnet import RelationNet
from methods.meta_template import MetaTemplate
from methods.baselinetrain import BaselineTrain
from methods.relationnet import RelationNet
# from methods.hypernets.hypernet_poc import HyperNetPOC
from methods.hypernets import hypernet_types
import data.feature_loader as feat_loader
from common import params_guards, set_seed, setup_model, _image_size

import random

import configs
import backbone

from modules.module import MetaModule
from parsers.train import TrainParams


def feature_evaluation(
    cl_data_file, model: MetaTemplate, n_way=5, n_support=5, n_query=15, adaptation=False
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


def single_test(params: TrainParams):
    iter_num = 600

    n_query = max(
        1, int(16 * params.test_n_way / params.train_n_way)
    )  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    print(f"{n_query=}")

    few_shot_params = dict(
        n_way=params.test_n_way, n_support=params.n_shot, n_query=n_query
    )

    params_guards(params)

    checkpoint_dir = configs.save_dir / 'checkpoints' / \
        params.dataset / params.model / params.method

    if params.train_aug:
        checkpoint_dir /= '_aug'
    if params.method not in ['baseline', 'baseline++']:
        checkpoint_dir /= f'_{params.train_n_way}way_{params.n_shot}shot'
    checkpoint_dir /= params.checkpoint_suffix

    if params.dataset == "cross" and not checkpoint_dir.exists():
        checkpoint_dir = Path(
            str(checkpoint_dir).replace("cross", "miniImagenet"))

    assert checkpoint_dir.exists()

    # modelfile = get_resume_file(checkpoint_dir)

    model = setup_model(params)

    if not params.method in ["baseline", "baseline++"]:
        if params.save_iter is not None:
            modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
        else:
            modelfile = get_best_file(checkpoint_dir)

        if modelfile is not None:
            print(f"Using {modelfile=}")
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp["state"])
        else:
            print(f"[WARNING] Cannot find 'best_file.tar' in {
                  checkpoint_dir=}")

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split

    eval_time = 0
    if params.method in [
        "maml",
        "maml_approx",
        "hyper_maml",
        "bayes_hmaml",
        "DKT",
    ] + list(
        hypernet_types.keys()
    ):  # maml do not support testing with feature
        image_size = _image_size(params)

        datamgr = SetDataManager(
            image_size, n_eposide=iter_num, **few_shot_params)

        def _loadfile():
            match params.dataset:
                case 'cross':
                    match split:
                        case 'base':
                            return configs.data_dir["miniImagenet"] / "all.json"
                        case _:
                            return configs.data_dir["CUB"] / f'{split}.json'
                case 'cross_char':
                    match split:
                        case 'base':
                            return configs.data_dir["omniglot"] / "noLatin.json"
                        case _:
                            return configs.data_dir["emnist"] / f'{split}.json'
                case _:
                    return configs.data_dir[params.dataset] / f'{split}.json'
        loadfile = _loadfile()

        novel_loader = datamgr.get_data_loader(loadfile, aug=False)
        if params.adaptation:
            model.task_update_num = (
                100 if params.hn_val_epochs == -1 else params.hn_val_epochs
            )
            # We perform adaptation on MAML simply by updating more times.

        # model.eval() # __jm__ massively hinders MAML accuracy
        # __jm__ removed from HyperMAML: who thought this would be a good idea?
        # model.single_test = True

        # __jm__ non unified interface
        test_results = model.test_loop(novel_loader)
        acc_mean = test_results.accuracy_mean
        acc_std = test_results.accuracy_std
    else:
        # defaut split = novel, but you can also test base or val classes
        novel_file = Path(str(checkpoint_dir).replace(
            "checkpoints", "features")) / f"{split_str}.hdf5"
        cl_data_file = feat_loader.init_loader(novel_file)

        acc_all = torch.empty(iter_num)
        for i in range(iter_num):
            acc_all[i] = feature_evaluation(
                cl_data_file, model, adaptation=params.adaptation, **few_shot_params)

        acc_std, acc_mean = torch.std_mean(acc_all)
        percentile_97_5 = 1.96 * acc_std / np.sqrt(iter_num)
        print(
            f"{iter_num} Test Acc = {acc_mean:4.2f} +- {percentile_97_5:4.2f}"
            % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num))
        )

    # region format and write results
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    aug_str = "-aug" if params.train_aug else ""
    aug_str += "-adapted" if params.adaptation else ""

    def _exp_setting():
        match params.method:
            case 'baseline' | 'baseline++':
                return f"""{params.dataset}-{split_str}-{params.model}-{params.method}{aug_str}
                            {params.n_shot}shot {params.test_n_way}way_test"""
            case _:
                return f"""{params.dataset}-{split_str}-{params.model}-{params.method}{aug_str}
                            {params.n_shot}shot {params.train_n_way}way_train {params.test_n_way}way_test"""
    exp_setting = _exp_setting()

    percentile_97_5 = 1.96 * acc_std / np.sqrt(iter_num)
    acc_str = f"{iter_num} Test Acc = {acc_mean} +- {percentile_97_5:4.2f}"  # nopep8
    with open("./record/results.txt", "a") as f:
        f.write(f"Time: {timestamp}, Setting: {exp_setting}, Acc: {acc_str}\n")
    # endregion

    print("Test loop time:", eval_time)
    return acc_mean, eval_time


def perform_test(params: TrainParams):
    seed = params.seed
    repeat = params.repeat
    # repeat the test N times changing the seed in range [seed, seed+repeat]
    accuracy_all = torch.empty(repeat)
    time_all = torch.empty(repeat)
    for i in range(repeat):
        if seed is not None:
            set_seed(seed + i)
        else:
            set_seed(None)
        acc, test_time = single_test(params)
        accuracy_all[i] = acc
        time_all[i] = test_time

    std_acc, mean_acc = torch.std_mean(accuracy_all)
    std_time, mean_time = torch.std_mean(time_all)

    print("-----------------------------")
    print(f"Seeds = {repeat} | Overall Test Acc = {mean_acc:.2f} +- {std_acc:.2f}. Eval time: {mean_time:.2f} +- {std_time:.2f}")  # nopep8
    print("-----------------------------")


def main():
    params = TrainParams().parse_args()  # __jm__ TODO: TestParams class
    perform_test(params)


if __name__ == "__main__":
    backbone.set_default_device()
    main()
