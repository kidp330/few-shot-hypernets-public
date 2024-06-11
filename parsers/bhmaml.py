from typing import Optional
from tap import Tap


class BHyperMAMLParams(Tap):
    weight_set_num_train: int = 1
    'number of randomly generated weights for training (default 1)'

    weight_set_num_test: Optional[int] = 20
    'number of randomly generated weights for test (default 20), if set to None expected value is generated'

    kl_stop_val: float = 1e-3
    'final value of kld_scale (default 1e-3)'

    kl_scale: float = 1e-24
    'initial value of kld_scale (default 1e-24)'
