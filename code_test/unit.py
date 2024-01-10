from pytorch_lightning import Trainer
import methods.DKT
import unittest
from itertools import product
from io_utils import model_dict


class TestModels(unittest.TestCase):
    def test_baseline(self):
        trainer = Trainer(fast_dev_run=True)

    # @parameterized.expand(
    #     product(
    #         model_dict,
    #         ["omniglot", "cross_char"],
    #         ["maml", "maml_approx"],
    #     ),
    #     testcase_func_name=lambda f, n, p: f"{n}_{p[0]}",
    # )
    # def test_model_fit(self, model, dataset, method):
    #     print(model, dataset, method)
