# We also support warmup from pretrained baseline feature, but we never used it in our paper
# elif params.warmup:
#     baseline_checkpoint_dir = (
#         f"{configs.save_dir}/checkpoints/{params.dataset}/{params.model}_baseline"
#     )
#     if params.train_aug:
#         baseline_checkpoint_dir += "_aug"
#     warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
#     tmp = torch.load(warmup_resume_file)
#     if tmp is not None:
#         state = tmp["state"]
#         state_keys = list(state.keys())
#         for _i, key in enumerate(state_keys):
#             if "feature." in key:
#                 newkey = key.replace(
#                     "feature.", ""
#                 )  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
#                 state[newkey] = state.pop(key)
#             else:
#                 state.pop(key)
#         model.feature.load_state_dict(state)
#     else:
#         raise ValueError("No warm_up file")
