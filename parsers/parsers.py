from .bhmaml import BHyperMAMLParams
from .general import GeneralParams
from .hypermaml import HyperMAMLParams
from .hypernet import HypernetParams


class ParamHolder(GeneralParams, HypernetParams, HyperMAMLParams, BHyperMAMLParams):
    def __init__(self):
        super().__init__()
        self.history = set()

    def __getattr__(self, item):
        it = super().__getattribute__(item)
        if item not in self.history:
            print("Getting", item, "=", it)
            self.history.add(item)
        return it

    # TODO: seems to be not working correctly after refactor
    def get_ignored_args(self) -> list[str]:
        return sorted([k for k in vars(self).keys() if k not in self.history])
