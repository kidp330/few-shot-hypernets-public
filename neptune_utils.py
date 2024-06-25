from pathlib import Path
import sys
from typing import Optional

import neptune
from neptune import Run

import configs
from methods.meta_template import MetaTemplate
from parsers.train import TrainParams


def track(run: Run, checkpoint_dir: Path, file: str):
    run[file].track_files(str(checkpoint_dir / f"{file}.tar"))


def setup(params: TrainParams, model: MetaTemplate) -> Optional[Run]:
    try:
        assert params.checkpoint_dir is not None
        run_name = (
            params.checkpoint_dir
            .relative_to(configs.save_dir / "checkpoints")
            .name
        )
        run_file = params.checkpoint_dir / "NEPTUNE_RUN.txt"

        run_id = None
        if params.resume and run_file.exists():
            with run_file.open("r") as f:
                run_id = f.read()
                print(f"Resuming neptune run {run_id}")

        run = neptune.init_run(
            name=run_name,
            source_files="**/*.py",
            tags=[params.checkpoint_suffix] if params.checkpoint_suffix != "" else [],
            with_id=run_id,
        )
        with run_file.open("w") as f:
            f.write(run["sys/id"].fetch())
            print("Starting neptune run", run["sys/id"].fetch())

        run["params"] = vars(params.params)
        run["cmd"] = f"python {' '.join(sys.argv)}"
        run["model"] = model

        return run

    except Exception as e:
        print("Cannot initialize neptune because of", e)
    return None
