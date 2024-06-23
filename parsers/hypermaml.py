from typing import Literal, Optional
from tap import Tap

UpdateOperator = Literal[
    'minus',
    'plus',
    'multiply',
]


class HyperMAMLParams(Tap):
    use_class_batch_input: bool = False
    'Strategy for handling query set embeddings as an input of hyper network'

    use_enhance_embeddings: bool = False
    "Flag that indicates if embeddings should be concatenated with logits and labels"

    update_operator: UpdateOperator = 'minus'
    "Choice of operator to use with update value for weight update"

    hm_lambda: Optional[float] = None
    'Regularization coefficient for the output of the hypernet'

    maml_warmup: bool = False
    "Train the model using gradient method (MAML) before switching to using the hypernetwork"

    # rolled back for now because I'm not sure whether computing multiple gradient steps over the same embeddings
    # (not recomputing the input by passing it through feature net) is the right implementation
    # self.hm_maml_update_feature_net = hm_params.maml_update_feature_net
    # maml_update_feature_net: bool = False
    # "Train feature net in the inner loop of MAML"

    maml_warmup_epochs: int = 100
    "The first n epochs where model is trained using a standard optimizer, not hypernetwork"

    maml_warmup_switch_epochs: int = 1000
    "The number of epochs for switching from MAML to HyperMAML"

    detach_feature_net: bool = False
    "Freeze feature network"

    detach_before_hyper_net: bool = False
    "Do not calculate gradient which comes from hypernetwork"

    train_support_set_loss: bool = False
    "Use both query and support data when calculating loss"

    test_set_forward_with_adaptation: bool = False
    "Adapt network before test"
