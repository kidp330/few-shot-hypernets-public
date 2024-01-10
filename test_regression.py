import numpy as np
import torch
import torch.optim as optim

import backbone
import configs
from io_utils import parse_args_regression
from methods.DKT_regression import DKT
from methods.feature_transfer_regression import FeatureTransfer

params = parse_args_regression("test_regression")
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params.checkpoint_dir = "%scheckpoints/%s/%s_%s" % (
    configs.save_dir,
    params.dataset,
    params.model,
    params.method,
)
bb = backbone.Conv3()

if params.method == "DKT":
    model = DKT(bb)
    optimizer = None
elif params.method == "transfer":
    model = FeatureTransfer(bb)
    optimizer = optim.Adam([{"params": model.parameters(), "lr": 0.001}])
else:
    raise ValueError("Unrecognised method")

model.load_checkpoint(params.checkpoint_dir)

mse_list = []
for epoch in range(params.n_test_epochs):
    mse = float(model.test_loop(params.n_support, optimizer).cpu().detach().numpy())
    mse_list.append(mse)

print("-------------------")
print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
print("-------------------")
