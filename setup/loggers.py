import configs
import neptune
import sys

from pathlib import Path
from neptune import Run
from neptune.exceptions import NeptuneException
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger

from io_params import ParamHolder


def _setup_neptune(params) -> Run | None:
    run_name = (
        Path(params.checkpoint_dir)
        .relative_to(Path(configs.save_dir) / "checkpoints")
        .name
    )
    run_file = Path(params.checkpoint_dir) / "NEPTUNE_RUN.txt"

    run_id = None
    if params.resume and run_file.exists():
        with run_file.open("r") as f:
            run_id = f.read()
            print("Resuming neptune run", run_id)

    try:
        run = neptune.init_run(
            name=run_name,
            source_files="**/*.py",
            tags=[params.checkpoint_suffix] if params.checkpoint_suffix != "" else [],
            with_id=run_id,
        )
        with run_file.open("w") as f:
            f.write(run["sys/id"].fetch())
            print("Starting neptune run", run["sys/id"].fetch())
        run["params"] = params.as_dict()
        run["cmd"] = f"python {' '.join(sys.argv)}"
        return run

    except NeptuneException as e:
        print("Cannot initialize neptune because of", e)
        return None


def setup_loggers(checkpoint_dir: str, params: ParamHolder) -> list[Logger]:
    loggers = [TensorBoardLogger(checkpoint_dir)]
    neptune_run = _setup_neptune(params)
    # if neptune_run is not None:
    #     neptune_run["model"] = str(model)
    if neptune_run is not None:
        loggers.append(NeptuneLogger(run=neptune_run))
    return loggers
