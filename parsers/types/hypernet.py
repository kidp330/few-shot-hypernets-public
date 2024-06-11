
from typing import Literal, Optional


Adaptation = Optional[Literal[
    'increasing_alpha',
]]

KernelInvariance = Literal[
    'attention',
    'convolution',
]

KernelInvariancePooling = Literal[
    'average',
    'mean',
    'min',
    'max',
]

AggregationType = Literal[
    "concat",
    "mean",
    "max_pooling",
    "min_pooling",
]
