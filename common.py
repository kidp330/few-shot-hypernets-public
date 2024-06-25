import backbone
import random
from typing import Literal, Optional
import numpy as np
import torch

from io_utils import model_dict
from methods.DKT import DKT
from methods.baselinetrain import BaselineTrain
from methods.hypernets.bayeshmaml import BayesHMAML
from methods.hypernets.hypermaml import HyperMAML
from methods.maml import MAML
from methods.matchingnet import MatchingNet
from methods.meta_template import MetaTemplate
from methods.protonet import ProtoNet
from methods.relationnet import RelationNet
from modules.module import MetaModule
from parsers.train import TrainParams


def setup_model(params: TrainParams) -> MetaTemplate:
    match params.method:
        case 'baseline' | 'baseline++':
            return BaselineTrain(model_dict[params.model], params.num_classes, loss_type='softmax' if params.method == 'baseline' else 'dist')

        case 'DKT' | 'protonet' | 'matchingnet' | 'relationnet' | 'relationnet_softmax' | 'maml' | 'maml_approx' | 'hyper_maml' | 'bayes_hmaml':
            n_query = max(1, int(
                # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
                16 * params.test_n_way / params.train_n_way))
            print(f"{n_query=}")

            n_way = params.train_n_way
            n_support = params.n_shot
            train_few_shot_params = dict(
                n_way=n_way,
                n_support=n_support,
                n_query=n_query
            )

            match params.method:
                case 'DKT':
                    model = DKT(model_dict[params.model],
                                n_way=params.train_n_way,
                                n_support=params.n_shot,
                                n_query=n_query)
                    model.init_summary()
                    return model
                case 'protonet':
                    return ProtoNet(model_dict[params.model], **train_few_shot_params)
                case 'matchingnet':
                    return MatchingNet(model_dict[params.model], **train_few_shot_params)
                case 'relationnet' | 'relationnet_softmax':
                    match params.model:
                        case 'Conv4':
                            feature_model = backbone.Conv4NP
                        case 'Conv6':
                            feature_model = backbone.Conv6NP
                        case 'Conv4S':
                            feature_model = backbone.Conv4SNP
                        case _:
                            def feature_model() -> MetaModule:
                                return model_dict[params.model](flatten=False)

                    loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
                    return RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
                case 'maml' | 'maml_approx':
                    # maml uses different parameter in omniglot
                    hypers = MAML.Hyperparameters()
                    if params.dataset in ['omniglot', 'cross_char']:
                        hypers = MAML.Hyperparameters(32//2, 0.1, 1)
                    return MAML(
                        model_dict[params.model],
                        n_way, n_support, n_query,
                        params, hypers,
                        approx=(params.method == 'maml_approx')
                    )
                # __jm__ TODO:
                # elif params.method in hypernet_types.keys():
                #     hn_type: Type[HyperNetPOC] = hypernet_types[params.method]
                #     return hn_type(model_dict[params.model],
                #                     params=params, **train_few_shot_params)
                case 'hyper_maml':
                    return HyperMAML(model_dict[params.model], n_way, n_support, n_query, params, approx=False)
                case 'bayes_hmaml':
                    return BayesHMAML(model_dict[params.model], n_way, n_support, n_query, params, approx=False)
                    # maml use different parameter in omniglot
                    # __jm__ TODO:
                    # if params.dataset in ['omniglot', 'cross_char']:
                    #     model.n_task = 32
                    #     model.task_update_num = 1
                    #     model.train_lr = 0.1
        # case list(hypernet_types.keys()):
        case _:
            raise ValueError('Unknown method')


def params_guards(params: TrainParams):
    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
        # params.model = 'Conv4S'
        # no need for this, since omniglot is loaded as RGB

    if params.method in ['baseline', 'baseline++']:
        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'


def _image_size(params: TrainParams) -> Literal[28] | Literal[84] | Literal[224]:
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            return 28
        return 84
    return 224


def set_seed(seed: Optional[int]):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if __debug__:
        print(f"[INFO] Setting SEED: {seed}")
