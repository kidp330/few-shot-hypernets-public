# region imports

import configs
# endregion

# region type imports
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Callable

from io_params import ParamHolder, Arg
from data.datamgr import (
    DataManager,
    SimpleDataManager,
    SetDataManager
)
# endregion


def __get_train_val_files(dataset: Arg.Dataset) -> tuple[Path, Path]:
    if dataset == "cross":
        base_file = configs.data_dir["miniImagenet"] / "all.json"
        val_file = configs.data_dir["CUB"] / "val.json"
    elif dataset == "cross_char":
        base_file = configs.data_dir["omniglot"] / "noLatin.json"
        val_file = configs.data_dir["emnist"] / "val.json"
    else:
        base_file = configs.data_dir[dataset] / "base.json"
        val_file = configs.data_dir[dataset] / "val.json"
    return base_file, val_file


def __get_image_size(model: Arg.Model, dataset: Arg.Dataset):
    if "Conv" in model:
        if dataset in ["omniglot", "cross_char"]:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224
    return image_size


def __get_dataloaders_generic(
    params: ParamHolder,
    get_datamgrs_callback: Callable[[int], tuple[DataManager, DataManager]],
):
    # NOTE: how to handle this for protonet, matchingnet
    image_size = __get_image_size(params.model, params.dataset)
    train_val_files = __get_train_val_files(params.dataset)

    base_file, val_file = train_val_files

    base_datamgr, val_datamgr = get_datamgrs_callback(image_size)

    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    return base_loader, val_loader


def setup_simple_dataloaders(params: ParamHolder) -> tuple[DataLoader, DataLoader]:
    return __get_dataloaders_generic(
        params,
        lambda image_size: (
            (base_datamgr := SimpleDataManager(image_size, batch_size=16)),
            (val_datamgr := SimpleDataManager(image_size, batch_size=64)),
            (base_datamgr, val_datamgr),
        )[-1],
    )


def setup_set_dataloaders(
    params: ParamHolder,
) -> tuple[DataLoader, DataLoader]:
    # __jm__ n_query is 'hardcoded' here - make it configurable?
    params.n_query = max(
        1, int(16 * params.test_n_way / params.train_n_way)
    )  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    print(f"{params.n_query=}")

    return __get_dataloaders_generic(
        params,
        lambda image_size: (
            (base_mgr := SetDataManager(image_size,
                                        n_way=params.train_n_way,
                                        n_support=params.n_shot,
                                        n_query=params.n_query
                                        )),
            (val_mgr := SetDataManager(image_size,
                                       n_way=params.test_n_way,
                                       n_support=params.n_shot,
                                       n_query=params.n_query
                                       )),
            (base_mgr, val_mgr),
        )[-1],
    )


def initialize_dataloaders(params: ParamHolder) -> tuple[DataLoader, DataLoader]:
    if params.method in ["baseline", "baseline++"]:
        return setup_simple_dataloaders(params)
    else:
        return setup_set_dataloaders(params)
