import os
import random
import time
from pathlib import Path
from typing import Type

import numpy as np
import torch
import torch.optim
import torch.utils.data.sampler

import backbone
import configs
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from io_utils import get_assigned_file, get_best_file, model_dict
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
from parsers.parsers import ParamHolder


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


def single_test(params: ParamHolder):
    acc_all = []
    iter_num = 600

    n_query = max(
        1, int(16 * params.test_n_way / params.train_n_way)
    )  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    print(f"{n_query=}")

    few_shot_params = dict(
        n_way=params.test_n_way, n_support=params.n_shot, n_query=n_query
    )

    if params.dataset in ["omniglot", "cross_char"]:
        assert params.model == "Conv4" and not params.train_aug, (
            f"model = {params.model}, train_aug= {params.train_aug} "
            f"omniglot only support Conv4 without augmentation"
        )
        # params.model = 'Conv4S'

    if params.method == "baseline":
        pass  # __jm__ stub
    #     model = BaselineFinetune(model_dict[params.model], **few_shot_params)
    # elif params.method == "baseline++":
    #     model = BaselineFinetune(
    #         model_dict[params.model], loss_type="dist", **few_shot_params
    #     )
    elif params.method == "protonet":
        model = ProtoNet(model_dict[params.model], **few_shot_params)
    elif params.method == "DKT":
        model = DKT(model_dict[params.model], **few_shot_params)
    elif params.method == "matchingnet":
        model = MatchingNet(model_dict[params.model], **few_shot_params)
    elif params.method in ["relationnet", "relationnet_softmax"]:
        if params.model == "Conv4":
            feature_model = backbone.Conv4NP
        elif params.model == "Conv6":
            feature_model = backbone.Conv6NP
        elif params.model == "Conv4S":
            feature_model = backbone.Conv4SNP
        else:
            def feature_model(): return model_dict[params.model](flatten=False)
        loss_type = "mse" if params.method == "relationnet" else "softmax"
        model = RelationNet(
            feature_model, loss_type=loss_type, **few_shot_params)
    elif params.method in ["maml", "maml_approx"]:
        model = MAML(
            model_dict[params.model],
            params=params,
            approx=(params.method == "maml_approx"),
            **few_shot_params,
        )
        # __jm__ TODO: uncomment
        # if params.dataset in [
        #     "omniglot",
        #     "cross_char",
        # ]:  # maml use different parameter in omniglot
        #     model.n_task = 32
        #     model.task_update_num = 1
        #     model.train_lr = 0.1
    # elif params.method in list(hypernet_types.keys()):
    #     few_shot_params["n_query"] = 15
    #     hn_type: Type[HyperNetPOC] = hypernet_types[params.method]
    #     model = hn_type(model_dict[params.model],
    #                     params=params, **few_shot_params)
        # model = HyperNetPOC(model_dict[params.model], **few_shot_params)
    elif params.method == "hyper_maml" or params.method == "bayes_hmaml":
        if params.method == "bayes_hmaml":
            model = BayesHMAML(
                model_dict[params.model],
                *few_shot_params.values(),
                params=params,
                approx=(params.method == "maml_approx"),
            )
        else:
            model = HyperMAML(
                model_dict[params.model],
                *few_shot_params.values(),
                params=params,
                approx=(params.method == "maml_approx"),
            )
        # __jm__ TODO: uncomment
        # if params.dataset in [
        #     "omniglot",
        #     "cross_char",
        # ]:  # maml use different parameter in omniglot
        #     model.n_task = 32
        #     model.train_lr = 0.1
    else:
        raise ValueError("Unknown method")

    checkpoint_dir = configs.save_dir / 'checkpoints' / \
        params.dataset / params.model / params.method

    if params.train_aug:
        checkpoint_dir /= "_aug"
    if not params.method in ["baseline", "baseline++"]:
        checkpoint_dir /= f"_{params.train_n_way}way_{
            params.n_shot}shot"
    if params.checkpoint_suffix != "":
        checkpoint_dir /= params.checkpoint_suffix

    if params.dataset == "cross":
        if not checkpoint_dir.exists():
            checkpoint_dir = checkpoint_dir.replace("cross", "miniImagenet")

    assert checkpoint_dir.exists()

    # modelfile = get_resume_file(checkpoint_dir)

    if not params.method in ["baseline", "baseline++"]:
        if params.save_iter is not None:
            modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
        else:
            modelfile = get_best_file(checkpoint_dir)

        print("Using model file", modelfile)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp["state"])
        else:
            print("[WARNING] Cannot find 'best_file.tar' in: " +
                  str(checkpoint_dir))

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
                loadfile = configs.data_dir["miniImagenet"] / "all.json"
            else:
                loadfile = configs.data_dir["CUB"] / f'{split}.json'
        elif params.dataset == "cross_char":
            if split == "base":
                loadfile = configs.data_dir["omniglot"] / "noLatin.json"
            else:
                loadfile = configs.data_dir["emnist"] / f'{split}.json'
        else:
            loadfile = configs.data_dir[params.dataset] / f'{split}.json'

        novel_loader = datamgr.get_data_loader(loadfile, aug=False)
        if params.adaptation:
            model.task_update_num = (
                100 if params.hn_val_epochs == -1 else params.hn_val_epochs
            )
            # We perform adaptation on MAML simply by updating more times.

        model.eval()
        # __jm__ removed from HyperMAML: who thought this would be a good idea?
        # model.single_test = True

        # __jm__ non unified interface
        acc_mean, acc_std, *_ = model.test_loop(novel_loader, return_time=True)

    else:
        # defaut split = novel, but you can also test base or val classes
        novel_file = Path(str(checkpoint_dir).replace(
            "checkpoints", "features")) / f"{split_str}.hdf5"
        cl_data_file = feat_loader.init_loader(novel_file)

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
    with open("./record/results.txt", "a") as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aug_str = "-aug" if params.train_aug else ""
        aug_str += "-adapted" if params.adaptation else ""
        if params.method in ["baseline", "baseline++"]:
            exp_setting = "%s-%s-%s-%s%s %sshot %sway_test" % (
                params.dataset,
                split_str,
                params.model,
                params.method,
                aug_str,
                params.n_shot,
                params.test_n_way,
            )
        else:
            exp_setting = "%s-%s-%s-%s%s %sshot %sway_train %sway_test" % (
                params.dataset,
                split_str,
                params.model,
                params.method,
                aug_str,
                params.n_shot,
                params.train_n_way,
                params.test_n_way,
            )

        percentile975 = 1.96 * acc_std / np.sqrt(iter_num)
        # __jm__ wtf
        print(acc_mean)
        acc_str = f"{iter_num} Test Acc = {acc_mean} +- {percentile975:4.2f}"  # nopep8
        f.write(f"Time: {timestamp}, Setting: {exp_setting}, Acc: {acc_str}\n")

    print("Test loop time:", eval_time)
    return acc_mean, eval_time


def perform_test(params):
    seed = params.seed
    repeat = params.repeat
    # repeat the test N times changing the seed in range [seed, seed+repeat]
    accuracy_list = list()
    time_list = list()
    for i in range(seed, seed + repeat):
        if seed != 0:
            _set_seed(i)
        else:
            _set_seed(0)
        acc, test_time = single_test(params)
        accuracy_list.append(acc)
        time_list.append(test_time)

    mean_acc = np.mean(accuracy_list)
    std_acc = np.std(accuracy_list)
    mean_time = np.mean(time_list)
    std_time = np.std(time_list)
    print("-----------------------------")
    print(
        f"Seeds = {repeat} | Overall Test Acc = {
            mean_acc:.2f} +- {std_acc:.2f}. Eval time: {mean_time:.2f} +- {std_time:.2f}"
    )
    print("-----------------------------")
    return {
        "accuracy_mean": mean_acc,
        "accuracy_std": std_acc,
        "time_mean": mean_time,
        "time_std": std_time,
        "n_seeds": repeat,
    }


def main():
    params = ParamHolder().parse_args()  # __jm__ TODO: TestParams class
    perform_test(params)


if __name__ == "__main__":
    backbone.set_default_device()
    main()
