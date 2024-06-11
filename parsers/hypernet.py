from tap import Tap
from parsers.types.general import Optim
from parsers.types.hypernet import *


class HypernetParams(Tap):
    adaptation_strategy: Adaptation = None
    'strategy used for manipulating alpha parameter'

    alpha_step: float = 0
    'step used to increase alpha from 0 to 1 during adaptation to new task'

    hidden_size: int = 256
    "HN hidden size"

    tn_hidden_size: int = 120
    "Target network hidden size"

    taskset_size: int = 1
    "Taskset size"

    neck_len: int = 0
    "Number of layers in the neck of the hypernet"

    head_len: int = 2
    "Number of layers in the heads of the hypernet, must be >= 1"

    taskset_repeats: str = "10:10-20:5-30:2"
    "A sequence of <epoch:taskset_repeats_until_the_epoch>"

    taskset_print_every: int = 20
    "It's a utility"

    detach_ft_in_hn: int = 10000
    "Detach FE output before hypernetwork in training *after* this epoch"

    detach_ft_in_tn: int = 10000
    "Detach FE output before target network in training *after* this epoch"

    tn_depth: int = 1
    "Depth of target network"

    dropout: float = 0
    "Dropout probability in hypernet"

    sup_aggregation: AggregationType = "concat"
    "How to aggregate supports from the same class"

    transformer_layers_no: int = 1
    "Number of layers in transformer"

    transformer_heads_no: int = 1
    "Number of attention heads in transformer"

    transformer_feedforward_dim: int = 512
    "Transformer's feedforward dimensionality"

    attention_embedding: bool = False
    "Utilize attention-based embedding"

    kernel_layers_no: int = 2
    "Depth of a kernel network"

    kernel_hidden_dim: int = 128
    "Hidden dimension of a kernel network"

    kernel_transformer_layers_no: int = 1
    "Number of layers in kernel's transformer"

    kernel_transformer_heads_no: int = 1
    "Number of attention heads in kernel's transformer"

    kernel_transformer_feedforward_dim: int = 512
    "Kernel transformer's feedforward dimensionality"

    kernel_out_size: int = 1600
    "Kernel output dim"

    kernel_invariance: bool = False
    "Should the HyperNet's kernel be sequence invariant"

    kernel_invariance_type: KernelInvariance = 'attention'
    "The type of invariance operation for the kernel's output"

    kernel_convolution_output_dim: int = 256
    "Kernel convolution's output dim"

    kernel_invariance_pooling: KernelInvariancePooling = 'mean'
    "The type of invariance operation for the kernel's output"

    use_support_embeddings: bool = False
    "Concatenate support embeddings with kernel features"

    no_self_relations: bool = False
    "Multiply matrix K to remove self relations (i.e., kernel(x_i, x_i))"

    use_cosine_distance: bool = False
    "Use cosine distance instead of a more specific kernel"

    use_scalar_product: bool = False
    "Use scalar product instead of a more specific kernel"

    use_cosine_nn_kernel: bool = False
    "Use cosine distance in NNKernel"

    val_epochs: int = 0
    "Epochs for finetuning on support set during validation. We recommend to set this to >0 only during testing."

    val_lr: float = 1e-4
    "LR for finetuning on support set during validation"

    val_optim: Optim = "adam"
    "Optimizer for finetuning on support set during validation"
