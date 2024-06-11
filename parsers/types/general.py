from typing import Literal, Optional

Method = Literal[
    'baseline',
    'baseline++',
    'DKT',
    'protonet',
    'matchingnet',
    'relationnet',
    'relationnet_softmax',
    'maml',
    'maml_approx',
    'hyper_maml',
    'bayes_hmaml',
    'hyper_shot',
    'hn_ppa',
    'hn_poc',
]

Model = Literal[
    "Conv4",
    "Conv4Pool",
    "Conv4S",
    "Conv6",
    "ResNet10",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "Conv4WithKernel",
    "ResNetWithKernel",
]

Dataset = Literal[
    'CUB',  # too large for CPU testing
    # 'miniImagenet', # TODO: unable to download using old script due to copyright issues
    # 'cross',
    'omniglot',
    # 'emnist', # TODO: it seems this dataset was not tested independently
    'cross_char',
]

Optim = Literal[
    'adam',
    'sgd',
]

Scheduler = Literal[
    'none',
    'multisteplr',
    'cosine',
    'reducelronplateau',
]

Split = Literal[
    'novel',
    'base',
    'val'
]
