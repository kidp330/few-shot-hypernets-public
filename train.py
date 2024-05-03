
import torch
import random
import torch.optim
import pytorch_lightning as pl
import os

import setup
from persist import (
    get_checkpoint_dir,
    get_checkpoint_file,
    save_run_params,
)

from pathlib import Path
from torch.utils.data import DataLoader

from io_utils import Arg
from io_params import ParamHolder

# TODO: remove these imports
import numpy as np
import matplotlib.pyplot as plt

# from test import perform_test


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

# def train_resume(checkpoint_path: os.PathLike, )


def train(
    model: pl.LightningModule,
    base_loader: DataLoader,
    val_loader: DataLoader,
    loggers: list[pl._logger],
    checkpoint_file: os.PathLike,
    params: ParamHolder,
):
    pass
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


# __jm__ TODO: put this somewhere else
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


def main():
    # params = parse_args("train")
    params = ParamHolder().parse_args()
    if params.args_file is not None and os.path.isfile(params.args_file):
        cli_resume = params.resume
        params.load(params.args_file)
        # overwrite
        params.resume = cli_resume
    else:
        assert \
            params.dataset is not None and \
            params.method is not None

    if params.dataset in ["omniglot", "cross_char"]:
        assert params.model == "Conv4" and not params.train_aug, (
            f"model = {params.model}, train_aug= {params.train_aug} "
            f"omniglot only support Conv4 without augmentation"
        )
        # params.model = 'Conv4S'
        # no need for this, since omniglot is loaded as RGB

    # print(f"{len(base_loader)=}")
    # for b in base_loader:
    #     print(f"{len(b)=}")
    #     print(f"{b[0].shape=}")
    #     print(f"{b[1].shape=}")
    #     return

    # train and test are split -- __jm__

    # things before call to train can be done mostly concurrently
    checkpoint_dir = get_checkpoint_dir(params)
    save_run_params(checkpoint_dir, params)
    checkpoint_file = get_checkpoint_file(
        checkpoint_dir) if params.resume else None
    loggers = setup.setup_loggers(checkpoint_dir, params)

    _set_seed(params.seed)

    model = setup.initialize_model(params)
    base_dataloader, val_dataloader = setup.initialize_dataloaders(params)
    # parse_dataloaders
    print({b[0].shape for b in base_dataloader})

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

    print("Starting training")
    print("Params accessed until this point:")
    print("\n\t".join(sorted(params.history)))
    print("Params ignored until this point:")
    print("\n\t".join(params.get_ignored_args()))

    trainer.fit(
        model,
        train_dataloaders=base_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=checkpoint_file,
    )

    # __jm__ train() should only train - its easy to make pipelines afterwards
    # params.split = "novel"
    # params.save_iter = -1

    # try:
    #     do_save_fts(params)
    # except Exception as e:
    #     print("Cannot save features bc of", e)

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
    #             neptune_run[f"full_test/{d}/metrics @ {hn_val_epochs}"] = test_results
    #         neptune_run.stop()


if __name__ == "__main__":
    main()
