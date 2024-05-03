import torch
import os
import h5py
from pathlib import Path
import configs
import backbone
from data.datamgr import SimpleDataManager
from methods.hypernets import hypernet_types
from io_utils import (
    model_dict,
    illegal_models,
    parse_args,
)
from persist import get_checkpoint_dir
from setup.dataloaders import __get_image_size
import time

def save_features(model, data_loader, outfile):
    with h5py.File(outfile, "w") as f:
        max_count = len(data_loader) * data_loader.batch_size
        all_labels = f.create_dataset("all_labels", (max_count,), dtype="i")
        all_feats = None
        count = 0
        for i, (x, y) in enumerate(data_loader):
            if i % 10 == 0:
                print("{:d}/{:d}".format(i, len(data_loader)))
            feats = model(x)
            if all_feats is None:
                all_feats = f.create_dataset(
                    "all_feats", [max_count] + list(feats.size()[1:]), dtype="f"
                )
            all_feats[count: count + feats.size(0)] = feats.data.cpu().numpy()
            all_labels[count: count + feats.size(0)] = y.cpu().numpy()
            count = count + feats.size(0)

        count_var = f.create_dataset("count", (1,), dtype="i")
        count_var[0] = count


def _get_embeddings_file(checkpoint_file: Path, split: str) -> Path:
    dir = checkpoint_file.parent / "embeddings" / split
    return dir / time.now() + ".hdf5"
    if params.save_iter != -1:
        outfile = os.path.join(
            checkpoint_dir.replace("checkpoints", "features"),
            split + "_" + str(params.save_iter) + ".hdf5",
        )
    else:
        outfile = os.path.join(
            checkpoint_dir.replace("checkpoints", "features"), split + ".hdf5"
        )
    return outfile

def get_features_dir

def do_save_fts(params):
    assert (
        params.method not in illegal_models
    ), "Chosen method does not support saving features"

    image_size = __get_image_size(params.model, params.dataset)

    split = params.split
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

    features_dir = get_features_dir(params)

    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False)

    if params.method in ["relationnet", "relationnet_softmax"]:
        if params.model == "Conv4":
            model = backbone.Conv4NP()
        elif params.model == "Conv6":
            model = backbone.Conv6NP()
        elif params.model == "Conv4S":
            model = backbone.Conv4SNP()
        else:
            model = model_dict[params.model](flatten=False)
    elif params.method in ["maml", "maml_approx"]:
        raise ValueError("MAML do not support save feature")
    else:
        model = model_dict[params.model]()

    tmp = torch.load(modelfile)
    state = tmp["state_dict"]
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace(
                "feature.", ""
            )  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_state_dict(state)
    model.eval()

    dirname = Path(outfile).parent
    if not dirname.is_dir():
        os.makedirs(dirname)
    save_features(model, data_loader, outfile)


if __name__ == "__main__":
    params = parse_args("save_features")
    do_save_fts(params)
