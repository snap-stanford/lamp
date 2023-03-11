from collections import OrderedDict
from copy import deepcopy
import math
import matplotlib.pyplot as plt
from numbers import Number
import numpy as np
import os
import pdb
import sys
import torch
from torch.fft import fft
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import yaml

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from lamp.pytorch_net.util import lcm, L2Loss, Attr_Dict, Printer, zero_grad_hook_multi, get_triu_block, get_tril_block, My_Tuple

p = Printer(n_digits=6)
INVALID_VALUE = -200

PDE_PATH = "data/"
EXP_PATH = "./results/"
DESIGN_PATH = ".."
MPPDE1D_PATH = "mppde1d_data/"
ARCSIMMESH_PATH = "arcsimmesh_data/"
FNO_PATH = "fno_data/"


try:
    import trimesh
    from trimesh.curvature import vertex_defects
except:
    print("Cannot import trimesh. If do not need trimesh, can ignore this message.")

def add_edge_normal_curvature(data_arcsimmesh):
    first = data_arcsimmesh.x[data_arcsimmesh.xfaces.T[:,1],3:] - data_arcsimmesh.x[data_arcsimmesh.xfaces.T[:,0],3:]
    second = data_arcsimmesh.x[data_arcsimmesh.xfaces.T[:,2],3:] - data_arcsimmesh.x[data_arcsimmesh.xfaces.T[:,1],3:]
    face_normals = torch.cross(first, second, dim=1)
    mean_vnormals = torch.tensor(trimesh.geometry.mean_vertex_normals(data_arcsimmesh.x.shape[0], data_arcsimmesh.xfaces.T.cpu().numpy(), face_normals.cpu().numpy())).cuda()
    x_receiver = torch.gather(mean_vnormals, 0, data_arcsimmesh.edge_index[0,:].unsqueeze(-1).repeat(1, mean_vnormals.shape[1]))
    x_sender = torch.gather(mean_vnormals, 0, data_arcsimmesh.edge_index[1,:].unsqueeze(-1).repeat(1,mean_vnormals.shape[1]))
    normal_angles = torch.acos((x_receiver*x_sender).sum(dim=1))
    data_arcsimmesh.edge_curvature = normal_angles.reshape(normal_angles.shape[0], 1)
    return data_arcsimmesh

def load_data(args, **kwargs):
    """Load data."""

    def get_train_val_idx(num_train_val, chunk_size=200):
        train_idx = (np.arange(0, num_train_val, chunk_size)[:, None] + np.arange(int(chunk_size * 0.8))[None, :]).reshape(-1)
        train_idx = train_idx[train_idx < num_train_val]
        val_idx = np.sort(np.array(list(set(np.arange(num_train_val)) - set(train_idx))))
        return torch.LongTensor(train_idx), torch.LongTensor(val_idx)

    def get_train_val_idx_random(num_train_val, train_fraction):
        train_idx = np.random.choice(num_train_val, size=int(num_train_val * train_fraction), replace=False)
        val_idx = np.sort(np.array(list(set(np.arange(num_train_val)) - set(train_idx))))
        return torch.LongTensor(train_idx), torch.LongTensor(val_idx)

    def to_deepsnap(pyg_dataset, args):
        """Transform the PyG dataset into deepsnap format."""
        graph_list = []
        #pdb.set_trace()
        for i, data in enumerate(pyg_dataset):
            if args.dataset.startswith("PL") or args.dataset.startswith("VL"):
                # Single/multi grid types, single/multi- particle types:
                """
                node_features:
                    '0': field:
                        node_feature: [N0, input_steps, 6];    node_label: [N0, output_steps, 6]
                        node_pos: [N0, input_steps, pos_dims]; node_pos_label: [N0, output_steps, pos_dims]
                    '1': ion, '2': electron:
                        node_feature: [N, input_steps, 5];     node_label: [N, output_steps, 3]
                        node_pos:     [N, input_steps, pos_dims]; node_pos_label: [N, output_steps, pos_dims]
                """
                G = HeteroGraph(
                    edge_index=data["edge_index"][0],
                    node_feature=data["node_feature"],
                    node_label=data["node_label"],
                    node_pos=data["node_pos"],
                    node_pos_label=data["node_pos_label"],
                    mask=data["mask"],
                    original_shape=data["original_shape"],
                    dyn_dims=data["dyn_dims"],
                    compute_func=data["compute_func"],
                    grid_keys=data["grid_keys"],
                    part_keys=data["part_keys"],
                    params=to_tuple_shape(data.params) if hasattr(data, "params") else (),
                )
            elif args.dataset.startswith("arcsimmesh"):
                G = HeteroGraph(
                    edge_index={("n0", "0", "n0"): data.edge_index},
                    node_feature={"n0": add_data_noise(data.x, args.data_noise_amp)},
                    node_label={"n0": add_data_noise(data.y, args.data_noise_amp)},
                    # mask={"n0": data.mask},
                    directed=True,
                    original_shape=(("n0", data.original_shape),),
                    dyn_dims=(("n0", to_tuple_shape(data.dyn_dims)),),
                    compute_func=(("n0", to_tuple_shape(data.compute_func)),),
                    #param={"n0": data.param[None]} if hasattr(data, "param") else {"n0": torch.ones(1,1)},
                    grid_keys=("n0",),
                    part_keys=(),
                    #params=to_tuple_shape(data.params) if hasattr(data, "params") else (),
                )
            else:
                # Single grid type, no particle:
                if i == 0:
                    original_shape = to_tuple_shape(data.original_shape)
                #pdb.set_trace()
                G = HeteroGraph(
                    edge_index={("n0", "0", "n0"): data.edge_index},
                    node_feature={"n0": add_data_noise(data.x, args.data_noise_amp)},
                    node_label={"n0": add_data_noise(data.y, args.data_noise_amp)},
                    mask={"n0": data.mask},
                    directed=True,
                    original_shape=(("n0", original_shape),),
                    dyn_dims=(("n0", to_tuple_shape(data.dyn_dims)),),
                    compute_func=(("n0", to_tuple_shape(data.compute_func)),),
                    param={"n0": data.param[None]} if hasattr(data, "param") else {"n0": torch.ones(1,1)},
                    grid_keys=("n0",),
                    part_keys=(),
                    params=to_tuple_shape(data.params) if hasattr(data, "params") else (),
                )
            if hasattr(data, "x_pos"):
                x_pos = deepcopy(data.x_pos)
                G["node_pos"] = {"n0": x_pos}
            if hasattr(data, "is_1d_periodic"):
                G["is_1d_periodic"] = data.is_1d_periodic
            if hasattr(data, "is_normalize_pos"):
                G["is_normalize_pos"] = data.is_normalize_pos
            if hasattr(data, "dataset"):
                G["dataset"] = data.dataset
            if hasattr(data, "xfaces"):
                G["xfaces"] = {"n0": data.xfaces}
            if hasattr(data, "y_tar"):
                G["y_tar"] = {"n0": data.y_tar}
            if hasattr(data, "y_back"):
                G["y_back"] = {"n0": data.y_back}
            if hasattr(data, "yface_list"):
                G["yface_list"] = {"n0": data.yface_list}
            if hasattr(data, "yedge_index"):
                G["yedge_index"] = {"n0": data.yedge_index}
            if hasattr(data, "x_bdd"):
                G["x_bdd"] = {"n0": data.x_bdd}
            if hasattr(data, "mask_non_badpoints"):
                G["mask_non_badpoints"] = data.mask_non_badpoints
            if hasattr(data, "edge_attr"):
                G["edge_attr"] = {("n0", "0", "n0"): data.edge_attr}
            graph_list.append(G)
            if i % 500 == 0 and i > 0:
                p.print(i)
        p.print(i + 1)
        dataset = GraphDataset(graph_list, task='node', minimum_node_per_graph=1)
        dataset.n_simu = pyg_dataset.n_simu
        dataset.time_stamps = pyg_dataset.time_stamps
        return dataset

    p = Printer()
    train_val_fraction = 0.9
    train_fraction = args.train_fraction
    multi_step_dict = parse_multi_step(args.multi_step)
    latent_multi_step_dict = parse_multi_step(args.latent_multi_step if args.latent_multi_step is not None else args.multi_step)
    if hasattr(args, "disc_coef") and (args.disc_coef > 0 or args.disc_coef == -1) and args.disc_t != "None":
        disc_t_list = parse_string_idx_to_list(args.disc_t, max_t=max(latent_multi_step_dict.keys()), is_inclusive=True)
    else:
        disc_t_list = list(latent_multi_step_dict.keys())
    if args.is_ebm:
        ebm_t_list = parse_string_idx_to_list(args.ebm_t, max_t=max(latent_multi_step_dict.keys()), is_inclusive=True)
    else:
        ebm_t_list = list(latent_multi_step_dict.keys())
    max_pred_steps = max(list(multi_step_dict.keys()) + list(latent_multi_step_dict.keys()) + [1] + disc_t_list + ebm_t_list) * args.temporal_bundle_steps
    filename_train_val = os.path.join(PDE_PATH_LOCAL, "deepsnap", "{}_train_val_in_{}_out_{}{}{}{}{}{}.p".format(
        args.dataset, args.input_steps * args.temporal_bundle_steps, max_pred_steps, 
        "_itv_{}".format(args.time_interval) if args.time_interval > 1 else "",
        "_yvar_{}".format(args.is_y_variable_length) if args.is_y_variable_length is True else "",
        "_noise_{}".format(args.data_noise_amp) if args.data_noise_amp > 0 else "",
        "_periodic_{}".format(args.is_1d_periodic) if args.is_1d_periodic else "",
        "_normpos_{}".format(args.is_normalize_pos) if args.is_normalize_pos is False else "",
    ))
    filename_test = os.path.join(PDE_PATH_LOCAL, "deepsnap", "{}_test_in_{}_out_{}{}{}{}{}{}.p".format(
        args.dataset, args.input_steps * args.temporal_bundle_steps, max_pred_steps, 
        "_itv_{}".format(args.time_interval) if args.time_interval > 1 else "",
        "_yvar_{}".format(args.is_y_variable_length) if args.is_y_variable_length is True else "",
        "_noise_{}".format(args.data_noise_amp) if args.data_noise_amp > 0 else "",
        "_periodic_{}".format(args.is_1d_periodic) if args.is_1d_periodic else "",
        "_normpos_{}".format(args.is_normalize_pos) if args.is_normalize_pos is False else "",
    ))
    is_to_deepsnap = True
    if (os.path.isfile(filename_train_val) or args.is_test_only) and os.path.isfile(filename_test) and args.n_train == "-1" and not args.dataset.startswith("arcsimmesh"):
        if not args.is_test_only:
            p.print(f"Loading {filename_train_val}")
            loaded = pickle.load(open(filename_train_val, "rb"))
            if isinstance(loaded, tuple):
                dataset_train, dataset_val = loaded
            else:
                dataset_train_val = loaded
        p.print(f"Loading {filename_test}")
        dataset_test = pickle.load(open(filename_test, "rb"))
        p.print("Loaded pre-saved deepsnap file at {}.".format(filename_test))

    else:
        p.print("{} does not exist. Generating...".format(filename_test))
        is_save = True  # If True, will save generated deepsnap dataset.

        if "fno" in args.dataset:
            if not args.is_test_only:
                pyg_dataset_train_val = FNOData(
                    dataset=args.dataset,
                    input_steps=args.input_steps * args.temporal_bundle_steps,
                    output_steps=max_pred_steps,
                    time_interval=args.time_interval,
                    is_y_diff=args.is_y_diff,
                    is_train=True,
                    is_y_variable_length=args.is_y_variable_length,
                )
            pyg_dataset_test = FNOData(
                dataset=args.dataset,
                input_steps=args.input_steps * args.temporal_bundle_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                is_y_diff=args.is_y_diff,
                is_train=False,
                is_y_variable_length=args.is_y_variable_length,
            )
        elif args.dataset.startswith("movinggas"):
            if not args.is_test_only:
                pyg_dataset_train_val = MovingGas(
                    dataset=args.dataset,
                    input_steps=args.input_steps * args.temporal_bundle_steps,
                    output_steps=max_pred_steps,
                    time_interval=args.time_interval,
                    is_y_diff=args.is_y_diff,
                    is_train=True,
                )
            pyg_dataset_test = MovingGas(
                dataset=args.dataset,
                input_steps=args.input_steps * args.temporal_bundle_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                is_y_diff=args.is_y_diff,
                is_train=False,
            )
        elif args.dataset in ["karman3d", "karman3d-small"]:
            if not args.is_test_only:
                pyg_dataset_train_val = Karman3D(
                    dataset=args.dataset,
                    input_steps=args.input_steps,
                    output_steps=max_pred_steps,
                    time_interval=args.time_interval,
                    is_y_diff=args.is_y_diff,
                    is_train=True,
                )
            pyg_dataset_test = Karman3D(
                dataset=args.dataset,
                input_steps=args.input_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                is_y_diff=args.is_y_diff,
                is_train=False,
            )
        elif args.dataset in ["karman3d-large", "karman3d-large-s", "karman3d-large-d", "karman3d-large-s-d"]:
            pyg_dataset_train_val = Karman3D(
                    dataset=args.dataset,
                    input_steps=args.input_steps,
                    output_steps=max_pred_steps,
                    time_interval=args.time_interval,
                    is_y_diff=args.is_y_diff,
                    is_train=True,
                    data_format="deepsnap",
                )
            pyg_dataset_test = Karman3D(
                dataset=args.dataset,
                input_steps=args.input_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                is_y_diff=args.is_y_diff,
                is_train=False,
                data_format="deepsnap",
            )
            is_to_deepsnap = False
        elif args.dataset.startswith("mppde1d"):
            if not args.is_test_only:
                pyg_dataset_train = MPPDE1D(
                    dataset=args.dataset,
                    input_steps=args.input_steps * args.temporal_bundle_steps,
                    output_steps=max_pred_steps,
                    time_interval=args.time_interval,
                    is_y_diff=args.is_y_diff,
                    is_1d_periodic=args.is_1d_periodic,
                    is_normalize_pos=args.is_normalize_pos,
                    split="train",
                )
                pyg_dataset_val = MPPDE1D(
                    dataset=args.dataset,
                    input_steps=args.input_steps * args.temporal_bundle_steps,
                    output_steps=max_pred_steps,
                    time_interval=args.time_interval,
                    is_y_diff=args.is_y_diff,
                    is_1d_periodic=args.is_1d_periodic,
                    is_normalize_pos=args.is_normalize_pos,
                    split="valid",
                )
            pyg_dataset_test = MPPDE1D(
                dataset=args.dataset,
                input_steps=args.input_steps * args.temporal_bundle_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                is_y_diff=args.is_y_diff,
                is_1d_periodic=args.is_1d_periodic,
                is_normalize_pos=args.is_normalize_pos,
                split="test",
            )
        elif args.dataset.startswith("arcsimmesh"):
            if args.algo.startswith("rlgnnremesher"):
                max_pred_steps = args.rl_horizon
            if not args.is_test_only:
                pyg_dataset_train_val = ArcsimMesh(
                    dataset=args.dataset,
                    input_steps=args.input_steps * args.temporal_bundle_steps,
                    output_steps=max_pred_steps,
                    time_interval=args.time_interval,
                    use_fineres_data=args.use_fineres_data,
                    is_train=True
                    #is_y_diff=args.is_y_diff,
                    #split="train",
                )
            pyg_dataset_test = ArcsimMesh(
                dataset=args.dataset,
                input_steps=args.input_steps * args.temporal_bundle_steps,
                output_steps=max_pred_steps,
                time_interval=args.time_interval,
                is_shifted_data=args.is_shifted_data,
                use_fineres_data=args.use_fineres_data,
                is_train=False,
                #is_y_diff=args.is_y_diff,
                #split="test",
            )
            is_to_deepsnap = False
        else:
            raise
   
        if args.n_train != "-1":
            # Test overfitting:
            is_save = False
            if not args.is_test_only:
                if "pyg_dataset_train_val" in locals():
                    pyg_dataset_train_val = get_elements(pyg_dataset_train_val, args.n_train)
                else:
                    pyg_dataset_train_val = get_elements(pyg_dataset_train, args.n_train)
            pyg_dataset_test = pyg_dataset_train_val
            p.print(", using the following elements {}.".format(args.n_train))
        else:
            p.print(":")

        # Transform to deepsnap format:
        if is_to_deepsnap:
            if not args.is_test_only:
                if "pyg_dataset_train_val" in locals():
                    dataset_train_val = to_deepsnap(pyg_dataset_train_val, args)
                else:
                    dataset_train = to_deepsnap(pyg_dataset_train, args)
                    dataset_val = to_deepsnap(pyg_dataset_val, args)
            dataset_test = to_deepsnap(pyg_dataset_test, args)
        else:
            if not args.is_test_only:
                dataset_train_val = pyg_dataset_train_val
            dataset_test = pyg_dataset_test

        # Save pre-processed dataset into file:
        if is_save:
            if not args.is_test_only:
                if "pyg_dataset_train_val" in locals():
                    if not os.path.isfile(filename_train_val):
                        pickle.dump(dataset_train_val, open(filename_train_val, "wb"))
                else:
                    pickle.dump((dataset_train, dataset_val), open(filename_train_val, "wb"))
            try:
                pickle.dump(dataset_test, open(filename_test, "wb"))
                p.print("saved generated deepsnap dataset to {}".format(filename_test))
            except Exception as e:
                p.print(f"Cannot save dataset object. Reason: {e}")


    # Split into train, val and test:
    collate_fn = deepsnap_Batch.collate() if is_to_deepsnap else MeshBatch(is_absorb_batch=True, is_collate_tuple=True).collate()
    if not args.is_test_only:
        if args.n_train == "-1":
            if "dataset_train" in locals() and "dataset_val" in locals():
                dataset_train_val = (dataset_train, dataset_val)
            elif args.dataset_split_type == "standard":
                if args.dataset.startswith("VL") or args.dataset.startswith("PL") or args.dataset.startswith("PIL"):
                    train_idx, val_idx = get_train_val_idx(len(dataset_train_val), chunk_size=200)
                    dataset_train = dataset_train_val[train_idx]
                    dataset_val = dataset_train_val[val_idx]
                else:
                    num_train = int(len(dataset_train_val) * train_fraction)
                    dataset_train, dataset_val = dataset_train_val[:num_train], dataset_train_val[num_train:]
            elif args.dataset_split_type == "random":
                train_idx, val_idx = get_train_val_idx_random(len(dataset_train_val), train_fraction=train_fraction)
                dataset_train, dataset_val = dataset_train_val[train_idx], dataset_train_val[val_idx]
            elif args.dataset_split_type == "order":
                n_train = int(len(dataset_train_val) * train_fraction)
                dataset_train, dataset_val = dataset_train_val[:n_train], dataset_train_val[n_train:]
            else:
                raise Exception("dataset_split_type '{}' is not valid!".format(args.dataset_split_type))
        else:
            # train, val, test are all the same as designated by args.n_train:
            dataset_train = deepcopy(dataset_train_val) if is_to_deepsnap else dataset_train_val
            dataset_val = deepcopy(dataset_train_val) if is_to_deepsnap else dataset_train_val
        train_loader = DataLoader(dataset_train, num_workers=args.n_workers, collate_fn=collate_fn,
                                  batch_size=args.batch_size, shuffle=True if args.dataset_split_type!="order" else False, drop_last=True)
        val_loader = DataLoader(dataset_val, num_workers=args.n_workers, collate_fn=collate_fn,
                                batch_size=args.val_batch_size if not args.algo.startswith("supn") else 1, shuffle=False, drop_last=False)
    else:
        dataset_train_val = None
        train_loader, val_loader = None, None
    test_loader = DataLoader(dataset_test, num_workers=args.n_workers, collate_fn=collate_fn,
                             batch_size=args.val_batch_size if not args.algo.startswith("supn") else 1, shuffle=False, drop_last=False)
    return (dataset_train_val, dataset_test), (train_loader, val_loader, test_loader)

    


def update_legacy_default_hyperparam(Dict):
    """Default hyperparameters for legacy settings."""
    default_param = {
        # Dataset:
        "time_interval": 1,
        "sector_size": "-1",
        "sector_stride": "-1",
        "seed": -1,
        "dataset_split_type": "standard",
        "train_fraction": float(8/9),
        "temporal_bundle_steps": 1,
        "is_y_variable_length": False,
        "data_noise_amp": 0,
        "data_dropout": "None",

        # Model:
        "latent_multi_step": None,
        "padding_mode": "zeros",
        "latent_noise_amp": 0,
        "decoder_last_act_name": "linear",
        "hinge": 1,
        "contrastive_rel_coef": 0,
        "n_conv_layers_latent": 1,
        "is_latent_flatten": True,
        "channel_mode": "exp-16",
        "no_latent_evo": False,
        "reg_type": "None",
        "reg_coef": 0,
        "is_reg_anneal": True,
        "forward_type": "Euler",
        "evo_groups": 1,
        "evo_conv_type": "cnn",
        "evo_pos_dims": -1,
        "evo_inte_dims": -1,
        "decoder_act_name": "None",
        "vae_mode": "None",
        "uncertainty_mode": "None",

        # Training:
        "is_pretrain_autoencode": False,
        "is_vae": False,
        "epochs_pretrain": 0,
        "dp_mode": "None",
        "latent_loss_normalize_mode": "None",
        "reinit_mode": "None",
        "is_clip_grad": False,
        "multi_step_start_epoch": 0,
        "epsilon_latent_loss": 0,
        "test_interval": 1,
        "lr_min_cos": 0,
        "is_prioritized_dropout": False,

        "noise_amp": 0,

        #RL:
        "processor_aggr":"max",
        "fix_alt_evolution_model":False,
        "test_reward_random_sample":False,
        
    }
    for key, item in default_param.items():
        if key not in Dict:
            Dict[key] = item
    if "seed" in Dict and Dict["seed"] is None:
        Dict["seed"] = -1
    return Dict

    
class MeshBatch(object):
    def __init__(self, is_absorb_batch=False, is_collate_tuple=False):
        """
        
        Args:
            is_collate_tuple: if True, will collate inside the tuple.
        """
        self.is_absorb_batch = is_absorb_batch
        self.is_collate_tuple = is_collate_tuple

    def collate(self):
        import re
        if torch.__version__.startswith("1.9") or torch.__version__.startswith("1.10") or torch.__version__.startswith("1.11"):
            from torch._six import string_classes
            from collections import abc as container_abcs
        else:
            from torch._six import container_abcs, string_classes, int_classes
        from pstar import pdict, plist
        default_collate_err_msg_format = (
            "collate_fn: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")
        np_str_obj_array_pattern = re.compile(r'[SaUO]')
        def default_convert(data):
            r"""Converts each NumPy array data field into a tensor"""
            elem_type = type(data)
            if isinstance(data, torch.Tensor):
                return data
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                    and elem_type.__name__ != 'string_':
                # array of string classes and object
                if elem_type.__name__ == 'ndarray' \
                        and np_str_obj_array_pattern.search(data.dtype.str) is not None:
                    return data
                return torch.as_tensor(data)
            elif isinstance(data, container_abcs.Mapping):
                return {key: default_convert(data[key]) for key in data}
            elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
                return elem_type(*(default_convert(d) for d in data))
            elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
                return [default_convert(d) for d in data]
            else:
                return data

        def collate_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                tensor = torch.cat(batch, 0, out=out)
                if self.is_absorb_batch:
                    # pdb.set_trace()
                    if tensor.shape[1] == 0:
                        tensor = tensor.view(tensor.shape[0], 0)
                    else:
                        tensor = tensor.view(-1, *tensor.shape[2:])
                return tensor
            elif elem is None:
                return None
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
                if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                    # array of string classes and object
                    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                        raise TypeError(default_collate_err_msg_format.format(elem.dtype))
                    return collate_fn([torch.as_tensor(b) for b in batch])
                elif elem.shape == ():  # scalars
                    return torch.as_tensor(batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float64)
            elif isinstance(elem, int):
                return torch.tensor(batch)
            elif isinstance(elem, string_classes):
                return batch
            elif isinstance(elem, container_abcs.Mapping):
                Dict = {}
                for key in elem:
                    if key == "node_feature":
                        Dict["vers"] = collate_trans_fn([d[key] for d in batch])
                        Dict[key] = collate_fn([d[key] for d in batch])
                        Dict["batch"] = {"n0": []}
                        batch_nodes = [d[key]["n0"] for d in batch]
                        for i in range(len(batch_nodes)):
                            item = torch.full((batch_nodes[i].shape[0],), i, dtype=torch.long)
                            Dict["batch"]["n0"].append(item)
                        Dict["batch"]["n0"] = torch.cat(Dict["batch"]["n0"])
                    elif key in ["y_tar", "reind_yfeatures"]:
                        # pdb.set_trace()
                        Dict[key] = collate_ytar_trans_fn([d[key] for d in batch])
                    elif key in ["history"]:
                        Dict[key] = collate_fn([d[key] for d in batch])
                        Dict["batch_history"] = {"n0": []}
                        batch_nodes = [d[key]["n0"] for d in batch]
                        for i in range(len(batch_nodes)):
                            item = torch.full((batch_nodes[i][0].shape[0],), i, dtype=torch.long)
                            Dict["batch_history"]["n0"].append(item)
                        Dict["batch_history"]["n0"] = torch.cat(Dict["batch_history"]["n0"])
                    elif key == "edge_index":
                        Dict[key] = collate_edgeidshift_fn([d[key] for d in batch])
                    elif key == "yedge_index":
                        # pdb.set_trace()
                        Dict[key] = collate_y_edgeidshift_fn([d[key] for d in batch])
                    elif key == "xfaces":
                        Dict[key] = collate_xfaceshift_fn([d[key] for d in batch])
                    elif key in ["bary_indices", "hist_indices"]:
                        #pdb.set_trace()
                        Dict[key] = collate_bary_indices_fn([d[key] for d in batch])
                    elif key in ["yface_list", "xface_list"]:
                         Dict[key] = collate_fn([d[key] for d in batch])
                    else:
                        Dict[key] = collate_fn([d[key] for d in batch])
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple:
                return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
            elif isinstance(elem, My_Tuple):
                it = iter(batch)
                elem_size = len(next(it))
                if not all(len(elem) == elem_size for elem in it):
                    raise RuntimeError('each element in list of batch should be of equal size')
                transposed = zip(*batch)
                return elem.__class__([collate_fn(samples) for samples in transposed])
            elif isinstance(elem, tuple):
                # pdb.set_trace()
                if self.is_collate_tuple:
                    #pdb.set_trace()
                    if len(elem) == 0:
                        return batch[0]
                    elif isinstance(elem[0], torch.Tensor):
                        newbatch = ()
                        for i in range(len(elem)):
                            newbatch = newbatch + tuple([torch.cat([tup[i] for tup in batch], dim=0)])
                        return newbatch
                    elif type(elem[0]) == list:
                        newbatch = ()
                        for i in range(len(elem)):
                            cumsum = 0
                            templist = []
                            for k in range(len(batch)):
                                shiftbatch = np.array(batch[k][i]) + cumsum
                                cumsum = shiftbatch.max() + 1
                                templist.extend(shiftbatch.tolist())
                            newbatch = newbatch + (templist,)
                    elif type(elem[0]).__module__ == np.__name__:
                        newbatch = ()
                        for i in range(len(elem)):
                            cumsum = 0
                            templist = []
                            for k in range(len(batch)):
                                shiftbatch = batch[k][i] + cumsum
                                cumsum = shiftbatch.max() + 1
                                templist.extend(shiftbatch.tolist())
                            newbatch = newbatch + (templist,)
                    else:
                        newbatch = batch[0]
                    return newbatch
                else:
                    return batch
            elif isinstance(elem, container_abcs.Sequence):
                # check to make sure that the elements in batch have consistent size
                it = iter(batch)
                elem_size = len(next(it))
                if not all(len(elem) == elem_size for elem in it):
                    raise RuntimeError('each element in list of batch should be of equal size')
                transposed = zip(*batch)
                return  [collate_fn(samples) for samples in transposed]
            elif elem.__class__.__name__ == 'Dictionary':
                return batch
            elif elem.__class__.__name__ == 'DGLHeteroGraph':
                import dgl
                return dgl.batch(batch)
            raise TypeError(default_collate_err_msg_format.format(elem_type))

        def collate_bary_indices_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, container_abcs.Mapping):
                Dict = {key: collate_bary_indices_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            elif isinstance(elem, tuple):
                # pdb.set_trace()
                if self.is_collate_tuple:
                    #pdb.set_trace()
                    if type(elem[0]).__module__ == np.__name__:
                        newbatch = ()
                        for i in range(len(elem)):
                            cumsum = 0
                            templist = []
                            for k in range(len(batch)):
                                shiftbatch = batch[k][i] + cumsum
                                cumsum = shiftbatch.max() + 1
                                templist.extend(shiftbatch)
                            newbatch = newbatch + (templist,)
                        return newbatch
                else:
                    return batch
            raise TypeError(default_collate_err_msg_format.format(elem_type))
            
        def collate_y_edgeidshift_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                cumsum = 0
                newbatch = []
                for i in range(len(batch)):
                    shiftbatch = batch[i] + cumsum
                    newbatch.append(shiftbatch)
                    cumsum = shiftbatch.max().item() + 1
                tensor = torch.cat(newbatch, dim=1)
                return tensor
            elif isinstance(elem, container_abcs.Mapping):
                Dict = {key: collate_y_edgeidshift_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            elif isinstance(elem, tuple):
                if self.is_collate_tuple:
                    if isinstance(elem[0], torch.Tensor):
                        newbatch = ()
                        for i in range(len(elem)):
                            cumsum = 0
                            tempbatch = []
                            for tup in batch:                    
                                shiftbatch = tup[i] + cumsum
                                tempbatch.append(shiftbatch)
                                cumsum = shiftbatch.max().item() + 1
                            newbatch = newbatch + tuple([torch.cat(tempbatch, dim=-1)])
                        return newbatch
                else:
                    return batch            
            raise TypeError(default_collate_err_msg_format.format(elem_type))  
            
        def collate_edgeidshift_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                cumsum = 0
                newbatch = []
                for i in range(len(batch)):                    
                    shiftbatch = batch[i] + cumsum
                    newbatch.append(shiftbatch)
                    cumsum = shiftbatch.max().item() + 1
                tensor = torch.cat(newbatch, dim=1)
                return tensor
            elif isinstance(elem, container_abcs.Mapping):
                Dict = {key: collate_edgeidshift_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            raise TypeError(default_collate_err_msg_format.format(elem_type))  
            
        def collate_xfaceshift_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                cumsum = 0
                newbatch = []
                for i in range(len(batch)):                    
                    shiftbatch = batch[i] + cumsum
                    newbatch.append(shiftbatch)
                    cumsum = shiftbatch.max().item() + 1
                tensor = torch.cat(newbatch, dim=-1)
                return tensor
            elif isinstance(elem, container_abcs.Mapping):
                Dict = {key: collate_xfaceshift_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            raise TypeError(default_collate_err_msg_format.format(elem_type))  
            
        def collate_trans_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                batch = [batch[i] + 2*i for i in range(len(batch))]
                try:
                    tensor = torch.cat(batch, 0, out=out)
                except:
                    pdb.set_trace()
                if self.is_absorb_batch:
                    # pdb.set_trace()
                    # if tensor.shape[1] == 0:
                    #     tensor = tensor.view(tensor.shape[0]*tensor.shape[1], 0)
                    # else:
                    tensor = tensor.view(-1, *tensor.shape[2:])
                return tensor
            elif isinstance(elem, container_abcs.Mapping):
                Dict = {key: collate_trans_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            raise TypeError(default_collate_err_msg_format.format(elem_type))   
            
        def collate_ytar_trans_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            # pdb.set_trace()
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, tuple):
                # pdb.set_trace()
                if self.is_collate_tuple:
                    #pdb.set_trace()
                    if len(elem) == 0:
                        return batch[0]
                    elif isinstance(elem[0], torch.Tensor):
                        newbatch = ()
                        for i in range(len(elem)):
                            templist = [batch[j][i] + 2*j for j in range(len(batch))]
                            newbatch = newbatch + (torch.cat(templist, dim=0),)
                        return newbatch
            elif isinstance(elem, container_abcs.Mapping):
                Dict = {key: collate_ytar_trans_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            raise TypeError(default_collate_err_msg_format.format(elem_type))   
        return collate_fn


def add_edge_angle_curvature(data_arcsimmesh):
    mesh = trimesh.Trimesh(vertices=data_arcsimmesh.x.numpy(), faces=data_arcsimmesh.xfaces.T.numpy(), process=False)
    vdefects = vertex_defects(mesh)
    tensor_defects = torch.tensor(vdefects).reshape(vdefects.shape[0], 1)
    x_receiver = torch.gather(tensor_defects, 0, data_arcsimmesh.edge_index[0,:].unsqueeze(-1).repeat(1, tensor_defects.shape[1]))
    x_sender = torch.gather(tensor_defects, 0, data_arcsimmesh.edge_index[1,:].unsqueeze(-1).repeat(1,tensor_defects.shape[1]))
    edge_curvature = (x_receiver + x_sender)/2
    data_arcsimmesh.edge_curvature = edge_curvature.abs()
    return data_arcsimmesh

def add_edge_normal_curvature(data_arcsimmesh):
    first = data_arcsimmesh.x[data_arcsimmesh.xfaces.T[:,1],3:] - data_arcsimmesh.x[data_arcsimmesh.xfaces.T[:,0],3:]
    second = data_arcsimmesh.x[data_arcsimmesh.xfaces.T[:,2],3:] - data_arcsimmesh.x[data_arcsimmesh.xfaces.T[:,1],3:]
    face_normals = torch.cross(first, second, dim=1)
    mean_vnormals = torch.tensor(trimesh.geometry.mean_vertex_normals(data_arcsimmesh.x.shape[0], data_arcsimmesh.xfaces.T.cpu().numpy(), face_normals.cpu().numpy())).cuda()
    x_receiver = torch.gather(mean_vnormals, 0, data_arcsimmesh.edge_index[0,:].unsqueeze(-1).repeat(1, mean_vnormals.shape[1]))
    x_sender = torch.gather(mean_vnormals, 0, data_arcsimmesh.edge_index[1,:].unsqueeze(-1).repeat(1,mean_vnormals.shape[1]))
    normal_angles = torch.acos((x_receiver*x_sender).sum(dim=1))
    data_arcsimmesh.edge_curvature = normal_angles.reshape(normal_angles.shape[0], 1)
    return data_arcsimmesh


##>>> Original util.py:

def test_graph_equivalence(graph1, graph2):
    
    equiv = True
    for mt, edges in graph1.edge_index.items():
        e1 = np.transpose(edges.cpu().numpy())
        e2 = np.transpose(graph2.edge_index[mt].cpu().numpy())
        #print(mt)
        #print(f'shape 1: {e1.shape}, shape 2: {e2.shape}')
        for e in e1:
            if e not in e2:
                equiv = False 
                print(f'{mt}: edge {e} in 1 but not 2')
        for e in e2:
            if e not in e1:
                equiv = False
                print(f'{mt}: edge {e} in 1 but not 1')
    if equiv:
        print('graphs have equivalent edges')
        # edges1 = {edge[1]: [] for edge in np.transpose(edges.cpu().numpy())}
        # edges2 = {edge[1]: [] for edge in np.transpose(graph2.edge_index[mt].cpu().numpy())}
        # for edge in np.transpose(edges.cpu().numpy()):
        #     edges1[edge[1]].append(edge[0])
        # for edge in np.transpose(graph2.edge_index[mt].cpu().numpy()):
        #     edges2[edge[1]].append(edge[0])
        
        # for target, source in edges2.items():
        #     try:
        #         if np.sort(edges1[target]) != np.sort(source):
        #             print(f'edge missmatch message type {mt}: {(target, edges1[target])} != {(target, source)}')
        #     except:
        #         print(f'edge missmatch message type {mt}: {target} is not a target node in graph1')
        

    for nt, features in graph1.node_feature.items():
            if features.numpy().all() != graph2.node_feature[nt].numpy().all():
                print(f'node feature missmatch on node type {nt}')
            if graph1.node_label[nt].numpy().all() != graph2.node_label[nt].numpy().all():
                print(f'node label missmatch on node type {nt}')
    
    print('graph1 and graph2 are equivalent')



def flatten(tensor):
    """Flatten the tensor except the first dimension."""
    return tensor.reshape(tensor.shape[0], -1)


def get_activation(act_name, inplace=False):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "linear":
        return nn.Identity()
    elif act_name == "leakyrelu":
        return nn.LeakyReLU(inplace=inplace)
    elif act_name == "leakyrelu0.2":
        return nn.LeakyReLU(inplace=inplace, negative_slope=0.2)
    elif act_name == "elu":
        return nn.ELU(inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "softplus":
        return nn.Softplus()
    elif act_name == "exp":
        return Exp()
    elif act_name == "sine":
        from siren_pytorch import Sine
        return Sine()
    elif act_name == "rational":
        return Rational()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "celu":
        return nn.CELU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "prelu":
        return nn.PReLU()
    elif act_name == "rrelu":
        return nn.RReLU()
    elif act_name == "mish":
        return nn.Mish()
    else:
        raise Exception("act_name '{}' is not valid!".format(act_name))


class Apply_Activation(nn.Module):
    def __init__(self, apply_act_idx, act_name="relu", dim=1):
        super().__init__()
        if isinstance(apply_act_idx, str):
            apply_act_idx = [int(ele) for ele in apply_act_idx.split(",")]
        self.apply_act_idx = apply_act_idx
        self.act = get_activation(act_name)
        self.dim = dim

    def forward(self, input):
        assert len(input.shape) >= 2  # []
        out = []
        for i in range(input.shape[self.dim]):
            if i in self.apply_act_idx:
                if self.dim == 1:
                    out.append(self.act(input[:,i]))
                elif self.dim == 2:
                    out.append(self.act(input[:,:,i]))
                else:
                    raise
            else:
                if self.dim == 1:
                    out.append(input[:,i])
                elif self.dim == 2:
                    out.append(input[:,:,i])
                else:
                    raise
        return torch.stack(out, self.dim)


# def get_LCM_input_shape(input_shape):
#     """Get the LCM input shape among many shapes."""
#     input_shape_dict = dict(input_shape)
#     dim_dict = {i: [] for i in range(3)}
#     for i in range(3):
#         for key, shape in input_shape_dict.items():
#             if len(shape) > i:
#                 dim_dict[i].append(shape[i])
#             else:
#                 dim_dict[i].append(1)

#     dim_dict_final = {}
#     for i in range(3):
#         dim_lcm = lcm(dim_dict[i])
#         for value in dim_dict[i]:
#             assert dim_lcm == value or value == 1
#         if dim_lcm > 1:
#             dim_dict_final[i] = dim_lcm
#     input_shape_final = tuple(list(dim_dict_final.values()))
#     return input_shape_final


def expand_same_shape(x_list, input_shape_LCM):
    pos_dim_max = len(input_shape_LCM)
    x_expand_list = []
    for x in x_list:
        shape = x.shape
        if len(x.shape) < pos_dim_max + 2:
            for i in range(pos_dim_max + 2 - len(x.shape)):
                x = x.unsqueeze(2)  # Here the field [B, C, X] is expanded to the full distribution [B, C, U, X]
            x = x.expand(x.shape[:2] + input_shape_LCM)
            x_expand_list.append(x)
        else:
            x_expand_list.append(x)
    return x_expand_list


def get_data_next_step(
    model,
    data,
    use_grads=True,
    is_y_diff=False,
    return_data=True,
    forward_func_name=None,
    is_rollout=False,
    uncertainty_mode="None",
):
    """Apply the model to data and obtain the data at the next time step without grads.

    Args:
        data:
            The returned data has features of
                [computed_features [not containing grad], static_features, dyn_features]

            The input data.node_feature does not contain the grads information.
            if return_data is False, will only return pred.
            if return_data is True, will also return the full data incorporating the prediction.

        forward_func_name: if None, will use the model's own forward function. If a string, will use model.forward_func_name as the forward function.
        is_rollout: if is_rollout=True, will stop gradient.

    Returns:
        pred: {key: [n_nodes, pred_steps, dyn_dims]}
    """
    dyn_dims = dict(to_tuple_shape(data.dyn_dims))  # data.node_feature: [n_nodes, input_steps, static_dims + dyn_dims]
    compute_func_dict = dict(to_tuple_shape(data.compute_func))
    static_dims = {key: data.node_feature[key].shape[-1] - dyn_dims[key] - compute_func_dict[key][0] for key in data.node_feature}

    # Compute pred:
    # After this application, the data.node_feature may append the grads information at the left:
    if is_rollout:
        with torch.no_grad():
            if forward_func_name is None:
                pred, info = model(data, use_grads=use_grads, uncertainty_mode=uncertainty_mode)  # pred: [n_nodes, pred_steps, dyn_dims]
            else:
                pred, info = getattr(model, forward_func_name)(data, use_grads=use_grads, uncertainty_mode=uncertainty_mode)  # pred: [n_nodes, pred_steps, dyn_dims]
    else:
        if forward_func_name is None:
            pred, info = model(data, use_grads=use_grads, uncertainty_mode=uncertainty_mode)  # pred: [n_nodes, pred_steps, dyn_dims]
        else:
            pred, info = getattr(model, forward_func_name)(data, use_grads=use_grads, uncertainty_mode=uncertainty_mode)  # pred: [n_nodes, pred_steps, dyn_dims]

    if not return_data:
        return None, pred, info
    # Update data:
    for key in pred:
        compute_dims = compute_func_dict[key][0]
        dynamic_features = pred[key]
        if uncertainty_mode != "None" and len(uncertainty_mode.split("^")) > 1 and uncertainty_mode.split("^")[1] == "samplefull":
            dynamic_features = dynamic_features + info["preds_ls"][key].exp() * torch.randn_like(info["preds_ls"][key])
        if is_y_diff:
            dynamic_features = dynamic_features + data.node_feature[key][..., -dyn_dims[key]:]
        # Append the computed node features:
        # [computed + static + dynamic]
        input_steps = data.node_feature[key].shape[-2]
        if input_steps > 1:
            dynamic_features = torch.cat([data.node_feature[key][...,-dyn_dims[key]:], dynamic_features], -2)[...,-input_steps:,:]
        static_features = data.node_feature[key][..., -static_dims[key]-dyn_dims[key]:-dyn_dims[key]]
        # The returned data will not contain grad information:
        if compute_dims > 0:
            compute_features = compute_func_dict[key][1](dynamic_features)
            node_features = torch.cat([compute_features, static_features, dynamic_features], -1)
        else:
            node_features = torch.cat([static_features, dynamic_features], -1)
        data.node_feature[key] = node_features
    return data, pred, info


def get_loss_ar(
    model,
    data,
    multi_step,
    use_grads=True,
    is_y_diff=False,
    loss_type="mse",
    **kwargs
):
    """Get auto-regressive loss for multiple steps."""
    multi_step_dict = parse_multi_step(multi_step)
    if len(multi_step_dict) == 1 and next(iter(multi_step_dict)) == 1:
        # Single-step prediction:
        pred, _ = model(data, use_grads=use_grads)
        loss = loss_op(pred, data.node_label, mask=data.mask, y_idx=0, loss_type=loss_type, **kwargs)
    else:
        # Multi-step prediction:
        max_step = max(list(multi_step_dict.keys()))
        loss = 0
        dyn_dims = dict(to_tuple_shape(data.dyn_dims))
        for i in range(1, max_step + 1):
            if i != max_step:
                data, _ = get_data_next_step(model, data, use_grads=use_grads, is_y_diff=is_y_diff, return_data=True)
                if i in multi_step_dict:
                    pred_new = {key: item[..., -dyn_dims[key]:] for key, item in data.node_feature.items()}
            else:
                _, pred_new = get_data_next_step(model, data, use_grads=use_grads, is_y_diff=is_y_diff, return_data=False)
            if i in multi_step_dict:
                loss_i = loss_op(pred_new, data.node_label, data.mask, y_idx=i-1, loss_type=loss_type, **kwargs)
                loss = loss + loss_i * multi_step_dict[i]
    return loss


def get_precision_floor(loss_type):
    """Get precision_floor from loss_type, if mselog, huberlog or l1 is inside loss_type. Otherwise return None"""
    precision_floor = None
    if loss_type is not None and ("mselog" in loss_type or "huberlog" in loss_type or "l1log" in loss_type):
        string_all = loss_type.split("+")
        for string in string_all:
            if "mselog" in string or "huberlog" in string or "l1log" in string:
                precision_floor = eval(string.split("#")[1])
                break
    return precision_floor


def loss_op(
    pred,
    y,
    mask=None,
    pred_idx=None,
    y_idx=None,
    dyn_dims=None,
    loss_type="mse",
    keys=None,
    reduction="mean",
    time_step_weights=None,
    normalize_mode="None",
    is_y_variable_length=False,
    preds_ls=None,
    **kwargs
):
    """Compute loss.

    Args:
        pred: shape [n_nodes, pred_steps, features]
        y: shape [n_nodes, out_steps, dyn_dims]
        mask: shape [n_nodes]
        pred_idx: range(0, pred_steps)
        y_idx: range(0, out_steps)
        dyn_dims: dictionary of {key: number of dynamic dimensions}. If not None, will get loss from pred[..., -dyn_dims:].
        loss_type: choose from "mse", "huber", "l1" and "dl", or use e.g. "0:huber^1:mse" for {"0": "huber", "1": "mse"}.
                   if "+" in loss_type, the loss will be the sum of multiple loss components added together.
                   E.g., if loss_type == '0:mse^2:mse+l1log#1e-3', then the loss is
                   {"0": mse, "2": mse_loss + l1log loss}
        keys: if not None, will only go through the keys provided. If None, will use the keys in "pred".
        time_step_weights: if not None but an array, will weight each time step by some coefficients.
        reduction: choose from "mean", "sum", "none" and "mean-dyn" (mean on the loss except on the last dyn_dims dimension).
        normalize_mode: choose from "target", "targetindi", "None". If "target", will divide the loss by the global norm of the target. 
                   if "targetindi", will divide the each individual loss by the individual norm of the target example. 
                   Default "None" will not normalize.
        **kwargs: additional kwargs for loss function.

    Returns:
        loss: loss.
    """
    # Make pred and y both dictionary:
    if (not isinstance(pred, dict)) and (not isinstance(y, dict)):
        pred = {"key": pred}
        y = {"key": y}
        if mask is not None:
            mask = {"key": mask}
    if keys is None:
        keys = list(pred.keys())

    # Individual loss components:
    loss = 0
    if time_step_weights is not None:
        assert len(time_step_weights.shape) == 1
        reduction_core = "none"
    else:
        if reduction == "mean-dyn":
            # Will perform mean on the loss except on the last dyn_dims dimension:
            reduction_core = "none"
        else:
            reduction_core = reduction
    if is_y_variable_length and loss_type != "lp":
        reduction_core = "none"
    if "^" in loss_type:
        # Different loss for different keys:
        loss_type_dict = parse_loss_type(loss_type)
    else:
        loss_type_dict = {key: loss_type for key in keys}

    # Compute loss
    for key in keys:
        # Specify which time step do we want to use from y:
        #   y has shape of [n_nodes, output_steps, dyn_dims]
        if pred[key] is None:
            # Due to latent level turning off:
            assert y[key] is None
            continue
        elif isinstance(pred[key], list) and len(pred[key]) == 0:
            # The multi_step="" or latent_multi_step="":
            continue
        if pred_idx is not None:
            if not isinstance(pred_idx, list):
                pred_idx = [pred_idx]
            pred_core = pred[key][..., pred_idx, :]
            if preds_ls is not None:
                preds_ls_core = preds_ls[key][..., pred_idx, :]
            else:
                preds_ls_core = None
        else:
            pred_core = pred[key]
            if preds_ls is not None:
                preds_ls_core = preds_ls[key]
            else:
                preds_ls_core = None
        if y_idx is not None:
            if not isinstance(y_idx, list):
                y_idx = [y_idx]
            y_core = y[key][..., y_idx, :]
        else:
            y_core = y[key]

        # y_core: [n_nodes, output_steps, dyn_dims]
        if is_y_variable_length:
            is_nan_full = (y_core == INVALID_VALUE).view(kwargs["batch_size"], -1, *y_core.shape[-2:])  # [batch_size, n_nodes_per_batch, output_steps, dyn_dims]
            n_nodes_per_batch = is_nan_full.shape[1]
            is_not_nan_batch = ~is_nan_full.any(1).any(-1)  # [batch_size, output_steps]
            if is_not_nan_batch.sum() == 0:
                continue
            is_not_nan_batch = is_not_nan_batch[:,None,:].expand(kwargs["batch_size"], n_nodes_per_batch, is_not_nan_batch.shape[-1])  # [batch_size, n_nodes_per_batch, output_steps]
            is_not_nan_batch = is_not_nan_batch.reshape(-1, is_not_nan_batch.shape[-1])  # [n_nodes, output_steps]
            if loss_type == "lp":
                kwargs["is_not_nan_batch"] = is_not_nan_batch[..., None]  # [n_nodes, output_steps, 1]
        else:
            is_not_nan_batch = None

        if dyn_dims is not None:
            y_core = y_core[..., -dyn_dims[key]:]
        if mask is not None:
            pred_core = pred_core[mask[key]]  # [n_nodes, pred_steps, dyn_dims]
            y_core = y_core[mask[key]]      # [n_nodes, output_steps, dyn_dims]
            if preds_ls is not None:
                preds_ls_core = preds_ls_core[mask[key]]
        # Compute loss:
        loss_i = loss_op_core(pred_core, y_core, reduction=reduction_core, loss_type=loss_type_dict[key], normalize_mode=normalize_mode, preds_ls_core=preds_ls_core, **kwargs)

        if time_step_weights is not None:
            shape = loss_i.shape
            assert len(shape) >= 3  # [:, time_steps, ...]
            time_step_weights = time_step_weights[None, :]
            for i in range(len(shape) - 2):
                time_step_weights = time_step_weights.unsqueeze(-1)  # [:, time_steps, [1, ...]]
            loss_i = loss_i * time_step_weights
            if is_y_variable_length and is_not_nan_batch is not None:
                loss_i = loss_i[is_not_nan_batch]
            if reduction == "mean-dyn":
                assert len(shape) == 3
                loss_i = loss_i.mean((0,1))
            else:
                loss_i = reduce_tensor(loss_i, reduction)
        elif is_y_variable_length and is_not_nan_batch is not None and loss_type != "lp":
            loss_i = loss_i[is_not_nan_batch]
            loss_i = reduce_tensor(loss_i, reduction)
        else:
            if reduction == "mean-dyn":
                assert len(loss_i.shape) == 3
                loss_i = loss_i.mean((0,1))  # [dyn_dims,]

        if loss_type == "rmse":
            loss_i = loss_i.sqrt()
        loss = loss + loss_i
    return loss


def reduce_tensor(tensor, reduction, dims_to_reduce=None, keepdims=False):
    """Reduce tensor using 'mean' or 'sum'."""
    if reduction == "mean":
        if dims_to_reduce is None:
            tensor = tensor.mean()
        else:
            tensor = tensor.mean(dims_to_reduce, keepdims=keepdims)
    elif reduction == "sum":
        if dims_to_reduce is None:
            tensor = tensor.sum()
        else:
            tensor = tensor.sum(dims_to_reduce, keepdims=keepdims)
    elif reduction == "none":
        pass
    else:
        raise
    return tensor


def loss_op_core(pred_core, y_core, reduction="mean", loss_type="mse", normalize_mode="None", zero_weight=1, preds_ls_core=None, **kwargs):
    """Compute the loss. Here pred_core and y_core must both be tensors and have the same shape. 
    Generically they have the shape of [n_nodes, pred_steps, dyn_dims].
    For hybrid loss_type, e.g. "mse+huberlog#1e-3", will recursively call itself.
    """
    if "+" in loss_type:
        loss_list = []
        precision_floor = get_precision_floor(loss_type)
        for loss_component in loss_type.split("+"):
            loss_component_coef = eval(loss_component.split(":")[1]) if len(loss_component.split(":")) > 1 else 1
            loss_component = loss_component.split(":")[0] if len(loss_component.split(":")) > 1 else loss_component
            if precision_floor is not None and not ("mselog" in loss_component or "huberlog" in loss_component or "l1log" in loss_component):
                pred_core_new = torch.exp(pred_core) - precision_floor
            else:
                pred_core_new = pred_core
            loss_ele = loss_op_core(
                pred_core=pred_core_new,
                y_core=y_core,
                reduction=reduction,
                loss_type=loss_component,
                normalize_mode=normalize_mode,
                zero_weight=zero_weight,
                preds_ls_core=preds_ls_core,
                **kwargs
            ) * loss_component_coef
            loss_list.append(loss_ele)
        loss = torch.stack(loss_list).sum(dim=0)
        return loss

    if normalize_mode != "None":
        assert normalize_mode in ["targetindi", "target"]
        dims_to_reduce = list(np.arange(2, len(y_core.shape)))  # [2, ...]
        epsilon_latent_loss = kwargs["epsilon_latent_loss"] if "epsilon_latent_loss" in kwargs else 0
        if normalize_mode == "target":
            dims_to_reduce.insert(0, 0)  # [0, 2, ...]

    if loss_type.lower() in ["mse", "rmse"]:
        if normalize_mode in ["target", "targetindi"]:
            loss = F.mse_loss(pred_core, y_core, reduction='none')
            loss = (loss + epsilon_latent_loss) / (reduce_tensor(y_core.square(), "mean", dims_to_reduce, keepdims=True) + epsilon_latent_loss)
            loss = reduce_tensor(loss, reduction=reduction)
        else:
            if zero_weight == 1:
                if preds_ls_core is None:
                    loss = F.mse_loss(pred_core, y_core, reduction=reduction)
                else:
                    loss_tensor = ((pred_core - y_core) / preds_ls_core.exp().clip(min=1e-5)).square() + preds_ls_core * 2
                    loss = reduce_tensor(loss_tensor, reduction=reduction)
            else:
                loss_inter = F.mse_loss(pred_core, y_core, reduction="none")
                zero_mask = y_core.abs() < 1e-8
                nonzero_mask = ~zero_mask
                loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
                loss = reduce_tensor(loss, reduction=reduction)
    elif loss_type.lower() == "huber":
        if normalize_mode in ["target", "targetindi"]:
            loss = F.smooth_l1_loss(pred_core, y_core, reduction='none')
            loss = (loss + epsilon_latent_loss) / (reduce_tensor(y_core.abs(), "mean", dims_to_reduce, keepdims=True) + epsilon_latent_loss)
            loss = reduce_tensor(loss, reduction=reduction)
        else:
            if zero_weight == 1:
                if preds_ls_core is None:
                    loss = F.smooth_l1_loss(pred_core, y_core, reduction=reduction)
                else:
                    loss = F.smooth_l1_loss((pred_core - y_core) / preds_ls_core.exp().clip(min=1e-5), torch.zeros_like(y_core), reduction=reduction) + reduce_tensor(preds_ls_core, reduction=reduction)
            else:
                loss_inter = F.smooth_l1_loss(pred_core, y_core, reduction="none")
                zero_mask = y_core.abs() < 1e-6
                nonzero_mask = ~zero_mask
                loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
                loss = reduce_tensor(loss, reduction)
    elif loss_type.lower() == "l1":
        if normalize_mode in ["target", "targetindi"]:
            loss = F.l1_loss(pred_core, y_core, reduction='none')
            loss = (loss + epsilon_latent_loss) / (reduce_tensor(y_core.abs(), "mean", dims_to_reduce, keepdims=True) + epsilon_latent_loss)
            loss = reduce_tensor(loss, reduction)
        else:
            if zero_weight == 1:
                if preds_ls_core is None:
                    loss = F.l1_loss(pred_core, y_core, reduction=reduction)
                else:
                    loss_tensor = (pred_core - y_core).abs() / preds_ls_core.exp().clip(min=1e-5) + preds_ls_core
                    loss = reduce_tensor(loss_tensor, reduction=reduction)
            else:
                loss_inter = F.l1_loss(pred_core, y_core, reduction="none")
                zero_mask = y_core.abs() < 1e-6
                nonzero_mask = ~zero_mask
                loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
                loss = reduce_tensor(loss, reduction)
    elif loss_type.lower() == "l2":
        first_dim = kwargs["first_dim"] if "first_dim" in kwargs else 2
        if normalize_mode in ["target", "targetindi"]:
            loss = L2Loss(reduction='none', first_dim=first_dim)(pred_core, y_core)
            y_L2 = L2Loss(reduction='none', first_dim=first_dim)(torch.zeros_like(y_core), y_core)
            if normalize_mode == "target":
                y_L2 = y_L2.mean(0, keepdims=True)
            loss = loss / y_L2
            loss = reduce_tensor(loss, reduction)
        else:
            if zero_weight == 1:
                loss = L2Loss(reduction=reduction, first_dim=first_dim)(pred_core, y_core)
            else:
                loss_inter = L2Loss(reduction="none", first_dim=first_dim)(pred_core, y_core)
                zero_mask = y_core.abs() < 1e-6
                nonzero_mask = ~zero_mask
                loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
                loss = reduce_tensor(loss, reduction)
    elif loss_type.lower() == "lp":
        assert normalize_mode not in ["target", "targetindi"]
        assert zero_weight == 1
        batch_size = kwargs["batch_size"]
        pred_core_reshape = pred_core.reshape(batch_size, -1)  # [B, -1]
        y_core_reshape = y_core.reshape(batch_size, -1)  # [B, -1]
        loss = LpLoss(reduction=True, size_average=True if reduction=="mean" else False)(pred_core_reshape, y_core_reshape, mask=kwargs["is_not_nan_batch"] if "is_not_nan_batch" in kwargs else None)
    elif loss_type.lower().startswith("mpe"):
        exponent = eval(loss_type.split("-")[1])
        if normalize_mode in ["target", "targetindi"]:
            loss = (pred_core - y_core).abs() ** exponent
            loss = (loss + epsilon_latent_loss) / (reduce_tensor(y_core.abs() ** exponent, "mean", dims_to_reduce, keepdims=True) + epsilon_latent_loss)
            loss = reduce_tensor(loss, reduction)
        else:
            if zero_weight == 1:
                if preds_ls_core is None:
                    loss = reduce_tensor((pred_core - y_core).abs() ** exponent, reduction=reduction)
                else:
                    loss_tensor = ((pred_core - y_core)/preds_ls_core.exp()).abs() ** exponent + preds_ls_core
                    loss = reduce_tensor(loss_tensor, reduction=reduction)
            else:
                loss_inter = (pred_core - y_core).abs() ** exponent
                zero_mask = y_core.abs() < 1e-8
                nonzero_mask = ~zero_mask
                loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
                loss = reduce_tensor(loss, reduction)
    elif loss_type.lower() == "dl":
        if zero_weight == 1:
            loss = DLLoss(pred_core, y_core, reduction=reduction, **kwargs)
        else:
            loss_inter = DLLoss(pred_core, y_core, reduction="none", **kwargs)
            zero_mask = y_core.abs() < 1e-6
            nonzero_mask = ~zero_mask
            loss = loss_inter * nonzero_mask + loss_inter * zero_mask * zero_weight
            loss = reduce_tensor(loss, reduction)
    # loss where the target is taking the log scale:
    elif loss_type.lower().startswith("mselog"):
        precision_floor = eval(loss_type.split("#")[1])
        loss = F.mse_loss(pred_core, torch.log(y_core + precision_floor), reduction=reduction)
    elif loss_type.lower().startswith("huberlog"):
        precision_floor = eval(loss_type.split("#")[1])
        loss = F.smooth_l1_loss(pred_core, torch.log(y_core + precision_floor), reduction=reduction)
    elif loss_type.lower().startswith("l1log"):
        precision_floor = eval(loss_type.split("#")[1])
        loss = F.l1_loss(pred_core, torch.log(y_core + precision_floor), reduction=reduction)
    else:
        raise Exception("loss_type {} is not valid!".format(loss_type))
    return loss


def loss_hybrid(
    preds,
    node_label,
    mask,
    node_pos_label,
    input_shape,
    pred_idx=None,
    y_idx=None,
    dyn_dims=None,
    loss_type="mse",
    part_keys=None,
    reduction="mean",
    **kwargs
):
    """Compute the loss at particle locations using by interpolating the values at the field grid.

    Args:
        preds:      {key: [n_nodes, pred_steps, dyn_dims]}
        node_label: {key: [n_nodes, output_steps, [static_dims + compute_dims] + dyn_dims]}
        mask:       {key: [n_nodes]}
        node_pos_label: {key: [n_nodes, output_steps, pos_dims]}. Both used to obtain the pos_grid, and also compute the loss for the density.
        input_shape: a tuple of the actual grid shape.
        pred_idx: index for the pred_steps in preds
        y_idx:    index for the output_steps in node_label
        dyn_dims: the last dyn_dims to obtain from node_label. If None, use full node_label.
        loss_type: loss_type.
        part_keys: keys for particle node types.
        reduction: choose from "mean", "sum" and "none".
        **kwargs: additional kwargs for loss_core.

    Returns:
        if reduction is 'none':
            loss_dict = {"density": {key: loss_matrix with shape [B, n_grid, dyn_dims]}, "feature": {key: loss_matrix}}
        else:
            loss_dict = {"density": loss_density, "feature": loss_feature}
    """
    grid_key = None
    for key in node_pos_label:
        if key not in part_keys:
            grid_key = key
    assert grid_key is not None
    pos_dims = len(input_shape)
    pos_grid = node_pos_label[grid_key][:, y_idx].reshape(-1, *input_shape, pos_dims)  # [B, n_grid, pos_dims]
    batch_size = pos_grid.shape[0]
    n_grid = pos_grid.shape[1]
    pos_dict = {}
    for dim in range(pos_grid.shape[-1]):
        pos_dict[dim] = {"pos_min": pos_grid[..., dim].min(),
                         "pos_max": pos_grid[..., dim].max()}

    loss_dict = {"density": {}, "feature": {}}
    for key in part_keys:
        # Obtain the index information for each position dimension:
        pos_part = node_pos_label[key][:, pred_idx].reshape(batch_size, -1, pos_dims) # [B, n_part, pos_dims]
        n_part = pos_part.shape[1]
        idx_dict = {}
        for dim in range(pos_dims):
            # idx_dict records the left index and remainder for each pos_part[..., dim] (with shape of [B, n_part]):
            idx_left, idx_remainder = get_idx_rel(pos_part[..., dim], pos_dict[dim]["pos_min"], pos_dict[dim]["pos_max"], n_grid)
            idx_dict[dim] = {}
            idx_dict[dim]["idx_left"] = idx_left
            idx_dict[dim]["idx_remainder"] = idx_remainder

        # density_grid_logit, prection of the density logit at each location, shape [B, prod([pos_dims])]
        density_grid_logit = preds[key][:, pred_idx, 0].reshape(batch_size, -1)
        density_grid_logprob = F.log_softmax(density_grid_logit, dim=-1)  # [B, prod([pos_dims])]
        if pos_dims == 1:
            dim = 0
            # Density loss. density_part_logprob: [B, n_part]:
            density_part_logprob = torch.gather(density_grid_logprob, dim=1, index=idx_dict[dim]["idx_left"]) * (1 - idx_dict[dim]["idx_remainder"]) + \
                                   torch.gather(density_grid_logprob, dim=1, index=idx_dict[dim]["idx_left"]+1) * idx_dict[dim]["idx_remainder"]
            loss_dict["density"][key] = -density_part_logprob.mean()

            # Field value loss:
            if dyn_dims is not None:
                node_label_core = node_label[key][:, y_idx, -dyn_dims[key]:].reshape(batch_size, n_part, dyn_dims[key])  # [B, n_part, dyn_dims]
            else:
                node_label_core = node_label[key][:, y_idx].reshape(batch_size, n_part, -1)  # [B, n_part, dyn_dims]
            feature_size = node_label_core.shape[-1]
            node_feature_pred_grid = preds[key][:, pred_idx, 1:].reshape(batch_size, n_grid, feature_size)  # [B, n_grid, feature_size]
            idx_left = idx_dict[dim]["idx_left"][...,None].expand(batch_size, n_part, feature_size)
            # node_feature_pred_part: [B, n_part, feature_size]:
            node_feature_pred_part = torch.gather(node_feature_pred_grid, dim=1, index=idx_left) * (1 - idx_dict[dim]["idx_remainder"])[..., None] + \
                                     torch.gather(node_feature_pred_grid, dim=1, index=idx_left+1) * idx_dict[dim]["idx_remainder"][..., None]
            loss_dict["feature"][key] = loss_op_core(node_feature_pred_part, node_label_core, reduction=reduction, loss_type=loss_type, **kwargs)
        else:
            raise Exception("Currently only supports pos_dims=1!")

    if reduction != "none":
        for mode in ["density", "feature"]:
            if reduction == "mean":
                loss_dict[mode] = torch.stack(list(loss_dict[mode].values())).mean()
            elif reduction == "sum":
                loss_dict[mode] = torch.stack(list(loss_dict[mode].values())).sum()
            else:
                raise
    return loss_dict


def get_idx_rel(pos, pos_min, pos_max, n_grid):
    """
    Obtain the left index on the grid as well as the relative distance to the left index.

    Args:
        pos: any tensor
        pos_min, pos_max: position of the left and right end of the grid
        n_grid: number of grid vertices

    Returns:
        idx_left: the left index. The pos is within [left_index, left_index + 1). Same shape as pos.
        idx_remainder: distance to the left index (in index space). Same shape as pos.
    """
    idx_real = (pos - pos_min) / (pos_max - pos_min) * (n_grid - 1)
    idx_left = idx_real.long()
    idx_remainder = idx_real - idx_left
    return idx_left, idx_remainder


def DLLoss(pred, y, reduction="mean", quantile=0.5):
    """Compute the Description Length (DL) loss, according to AI Physicist (Wu and Tegmark, 2019)."""
    diff = pred - y
    if quantile == 0.5:
        precision_floor = diff.abs().median().item() + 1e-10
    else:
        precision_floor = diff.abs().quantile(quantile).item() + 1e-10
    loss = torch.log(1 + (diff / precision_floor) ** 2)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    else:
        raise Exception("Reduction can only choose from 'mean', 'sum' and 'none'.")
    return loss


def to_cpu(state_dict):
    state_dict_cpu = {}
    for k, v in state_dict.items():
        state_dict_cpu[k] = v.cpu()
    return state_dict_cpu


def get_root_dir():
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("plasma")
    dirname = "/".join(dirname_split[:index + 1])
    return dirname


def to_tuple_shape(item):
    """Transform [tuple] or tuple into tuple."""
    if isinstance(item, list) or isinstance(item, torch.Tensor):
        item = item[0]
    assert isinstance(item, tuple) or isinstance(item, Number) or isinstance(item, torch.Tensor) or isinstance(item, str) or isinstance(item, bool)
    return item


def parse_multi_step(string):
    """
    Parse multi-step prediction setting from string to multi_step_dict.

    Args:
        string: default "1", meaning only 1 step MSE. "1^2:1e-2^4:1e-3" means loss has 1, 2, 4 steps, with the number after ":" being the scale.'

    Returns:
        multi_step_dict: E.g. {1: 1, 2: 1e-2, 4: 1e-3} for string="1^2:1e-2^4:1e-3".
    """
    if string == "":
        return {}
    multi_step_dict = {}
    if "^" in string:
        string_split = string.split("^")
    else:
        string_split = string.split("$")
    for item in string_split:
        item_split = item.split(":")
        time_step = eval(item_split[0])
        multi_step_dict[time_step] = eval(item_split[1]) if len(item_split) == 2 else 1
    multi_step_dict = OrderedDict(sorted(multi_step_dict.items()))
    return multi_step_dict


def parse_lr_special(string):
    """
    Parse lr_special

    Args:
        string: default "None". Example: "decoder+evolution_op:1e-3^encoder:1e-2"

    Returns:
        lr_special_dict: E.g. {decoder: 1e-3, decoder: 1e-3, encoder: 1e-2} for string="decoder+evolution_op:1e-3^encoder:1e-2".
    """
    if string == "None":
        return {}
    lr_special_dict = {}
    for item in string.split("^"):
        # item: decoder+evolution_op:1e-3
        keys, lr = item.split(":")
        for key in keys.split("+"):
            lr_special_dict[key] = eval(lr)
    return lr_special_dict


def parse_act_name(string):
    """
    Parse act_name_dict from string.

    Returns:
        act_name_dict: E.g. {"1": "softplus", "2": "elu"} for string="1:softplus^2:elu".
    """
    act_name_dict = {}
    string_split = string.split("^")
    for item in string_split:
        item_split = item.split(":")
        assert len(item_split) == 2
        time_step = item_split[0]
        act_name_dict[time_step] = item_split[1]
    return act_name_dict


def parse_loss_type(string):
    """
    Parse loss_type_dict from string.

    Returns:
        loss_type_dict: E.g.
            string == "1:mse^2:huber" => loss_type_dict = {"1": "mse", "2": "huber"} for .
            string == '0:mse^2:mse+l1log#1e-3' => loss_type_dict = {"0": mse, "2": "mse+l1log#1e-3"}
    """
    loss_type_dict = {}
    string_split = string.split("^")
    for item in string_split:
        item_split = item.split(":")
        assert len(item_split) == 2
        key = item_split[0]
        loss_type_dict[key] = item_split[1]
    return loss_type_dict


def parse_hybrid_targets(hybrid_targets, default_value=1.):
    """
    Example: M:0.1^xu
    """
    if hybrid_targets == "all":
        hybrid_target_dict = {key: 1 for key in ["M", "MNT", "xu", "J", "field", "full"]}
    elif isinstance(hybrid_targets, str):
        hybrid_target_dict = {}
        for item in hybrid_targets.split("^"):
            hybrid_target_dict[item.split(":")[0]] = eval(item.split(":")[1]) if len(item.split(":")) > 1 else default_value
    else:
        raise
    return hybrid_target_dict


def parse_reg_type(reg_type):
    """Parse reg_type and returns reg_type_core and reg_target.

    reg_type has the format of f"{reg-type}[-{model-target}]^..." as splited by "^"
        where {reg-type} chooses from "srank", "spectral", "Jsim" (Jacobian simplicity), "l2", "l1".
        The optional {model-target} chooses from "all" or "evo" (only effective for Contrastive).
        If not appearing, default "all". The "Jsim" only targets "evo".
    """
    reg_type_list = []
    for reg_type_ele in reg_type.split("^"):
        reg_type_split = reg_type_ele.split("-")
        reg_type_core = reg_type_split[0]
        if len(reg_type_split) == 1:
            reg_target = "all"
        else:
            assert len(reg_type_split) == 2
            reg_target = reg_type_split[1]
        if reg_type_core == "Jsim":
            assert len(reg_type_split) == 1 or reg_target == "evo"
            reg_target = "evo"
        elif reg_type_core == "None":
            reg_target = "None"
        reg_type_list.append((reg_type_core, reg_target))
    return reg_type_list


def get_cholesky_inverse(scale_tril_logit, size):
    """Get the cholesky-inverse from the lower triangular matrix.

    Args:
        scale_tril_logit: has shape of [B, n_components, size*(size+1)/2]. It should be a logit
            where the diagonal element will be passed into softplus.
        size: dimension of the matrix.

    Returns:
        cholesky_inverse: has shape of [B, n_components, size, size]
    """
    n_components = scale_tril_logit.shape[-2]
    scale_tril = fill_triangular(scale_tril_logit.view(-1, scale_tril_logit.shape[-1]), dim=size)
    scale_tril = matrix_diag_transform(scale_tril, F.softplus)
    cholesky_inverse = torch.stack([torch.cholesky_inverse(matrix) for matrix in scale_tril]).reshape(-1, n_components, size, size)
    return cholesky_inverse, scale_tril.reshape(-1, n_components, size, size)


class Rational(torch.nn.Module):
    """Rational Activation function.
    Implementation provided by Mario Casado (https://github.com/Lezcano)
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                         [1.5957, 2.383],
                                         [0.5, 0.0],
                                         [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output


def get_device(args):
    """Initialize device."""
    if len(args.gpuid.split(",")) > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid # later retrieved to set gpuids
        # https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018/9
        cuda_str = args.gpuid.split(",")[0] # first device
    else:
        cuda_str = args.gpuid
    is_cuda = eval(cuda_str)
    if not isinstance(is_cuda, bool):
        is_cuda = "cuda:{}".format(is_cuda)
    device = torch.device(is_cuda if isinstance(is_cuda, str) else "cuda" if is_cuda else "cpu")
    return device


def get_normalization(normalization_type, n_channels, n_groups=2):
    """Get normalization layer."""
    if normalization_type.lower() == "bn1d":
        layer = nn.BatchNorm1d(n_channels)
    elif normalization_type.lower() == "bn2d":
        layer = nn.BatchNorm2d(n_channels)
    elif normalization_type.lower() == "gn":
        layer = nn.GroupNorm(num_groups=n_groups, num_channels=n_channels)
    elif normalization_type.lower() == "ln":
        layer = nn.LayerNorm(n_channels)
    elif normalization_type.lower() == "none":
        layer = nn.Identity()
    else:
        raise Exception("normalization_type '{}' is not valid!".format(normalization_type))
    return layer


def get_max_pool(pos_dims, kernel_size):
    if pos_dims == 1:
        return nn.MaxPool1d(kernel_size=kernel_size)
    elif pos_dims == 2:
        return nn.MaxPool2d(kernel_size=kernel_size)
    elif pos_dims == 3:
        return nn.MaxPool3d(kernel_size=kernel_size)
    else:
        raise


def get_regularization(model_list, reg_type_core):
    """Get regularization.

    Args:
        reg_type_core, Choose from:
            "None": no regularization
            "l1": L1 regularization
            "l2": L2 regularization
            "nuc": nuclear regularization
            "fro": Frobenius norm
            "snr": spectral regularization
            "snn": spectral normalization
            "SINDy": L1 regularization on the coefficient of SINDy
            "Hall": regularize all the elements of Hessian
            "Hoff": regularize the off-diagonal elements of Hessian
            "Hdiag": regularize the diagonal elements of Hessian

    Returns:
        reg: computed regularization.
    """
    if reg_type_core in ["None", "snn"]:
        return 0
    else:
        List = []
        if reg_type_core in ["l1", "l2", "nuc", "fro"]:
            for model in model_list:
                for param_key, param in model.named_parameters():
                    if "weight" in param_key and param.requires_grad:
                        if reg_type_core in ["nuc", "fro"]:
                            norm = torch.norm(param, reg_type_core)
                        elif reg_type_core == "l1":
                            norm = param.abs().sum()
                        elif reg_type_core == "l2":
                            norm = param.square().sum()
                        else:
                            raise
                        List.append(norm)
        elif reg_type_core == "snr":
            for model in model_list:
                for module in model.modules():
                    if module.__class__.__name__ == "SpectralNormReg" and hasattr(module, "snreg"):
                        List.append(module.snreg)
        elif reg_type_core == "sindy":
            for model in model_list:
                for module in model.modules():
                    if module.__class__.__name__ == "SINDy":
                        List.append(module.weight.abs().sum())
        elif reg_type_core in ["Hall", "Hoff", "Hdiag"]:
            for model in model_list:
                if hasattr(model, "Hreg"):
                    List.append(model.Hreg)
        else:
            raise Exception("reg_type_core {} is not valid!".format(reg_type_core))
        if len(List) > 0:
            reg = torch.stack(List).sum()
        else:
            reg = 0
        return reg


def get_edge_index_kernel(pos_part, grid_pos, kernel_size, stride, padding, batch_size):
    """Get the edge index from particle to kernel indices.

    Args:
        pos_part: particle position [B, n_part]
        grid_pos: [B, n_grid, steps, pos_dims]
    """
    def get_index_pos(pos_part, pos_min, pos_max, n_grid):
        """Get the index position (index_pos) for each real position."""
        index_pos = (pos_part - pos_min) / (pos_max - pos_min) * (n_grid - 1)
        return index_pos

    def get_kernel_index_range(index_pos, kernel_size, stride, padding):
        """Get the kernel index range(index_left, index_right) for each index_pos."""
        index_left = torch.ceil((index_pos + padding - (kernel_size - 1)) / stride).long()
        index_right = (torch.floor((index_pos + padding) / stride) + 1).long()
        return index_left, index_right

    assert len(pos_part.shape) == 2
    device = pos_part.device
    pos_min = grid_pos.min()
    pos_max = grid_pos.max()
    n_grid = grid_pos.shape[1]
    n_part = pos_part.shape[1]
    n_kern = int((n_grid + 2 * padding - kernel_size) / stride + 1)
    index_pos = get_index_pos(pos_part, pos_min, pos_max, n_grid)  # [B, n_part]

    index_left, index_right = get_kernel_index_range(index_pos, kernel_size, stride, padding)  # [B, n_part]
    assert stride == 1, "Currently only supports stride=1."
    idx_kern_ori = torch.arange(kernel_size - 1).unsqueeze(0).unsqueeze(0).to(device) + index_left.unsqueeze(-1)  # [B, n_part, kernel_size-1], each denotes the index of the kernel that the particle will be mapped to.
    mask_valid = ((0 <= idx_kern_ori) & (idx_kern_ori < n_kern)).view(-1)
    # Compute edge_attr:
    idx_index = idx_kern_ori * stride - padding + (kernel_size - 1) / 2
    pos_kern = idx_index / (n_grid - 1) * (pos_max - pos_min) + pos_min
    pos_diff = pos_part.unsqueeze(-1) - pos_kern  # [B, n_part, kernel_size-1]
    pos_diff = pos_diff.view(-1)[:, None]
    edge_attr = torch.cat([pos_diff.abs(), pos_diff], -1)
    edge_attr = edge_attr[mask_valid]

    # Compute edge_index:
    idx_kern = idx_kern_ori + (torch.arange(batch_size) * n_kern).unsqueeze(-1).unsqueeze(-1).to(device)
    idx_part = (torch.ones(kernel_size - 1).unsqueeze(0).unsqueeze(0).long() * torch.arange(n_part).unsqueeze(0).unsqueeze(-1) + (torch.arange(batch_size) * n_part).unsqueeze(-1).unsqueeze(-1)).to(device)  # [B, n_part, kernel_size-1]
    edge_index = torch.stack([idx_part.view(-1), idx_kern.view(-1)]).to(device)
    edge_index = edge_index[:, mask_valid]
    return edge_index, edge_attr, n_kern


def add_noise(input, noise_amp):
    """Add independent Gaussian noise to each element of the tensor."""
    if not isinstance(input, tuple):
        input = input + torch.randn(input.shape).to(input.device) * noise_amp
        return input
    else:
        List = []
        for element in input:
            if element is not None:
                List.append(add_noise(element, noise_amp))
            else:
                List.append(None)
        return tuple(List)


def get_neg_loss(pred, target, loss_type="mse", time_step_weights=None):
    """Get the negative loss by permuting the target along batch dimension."""
    if not isinstance(pred, tuple):
        assert not isinstance(target, tuple)
        batch_size = pred.shape[0]
        perm_idx = np.random.permutation(batch_size)
        target_neg = target[perm_idx]
        loss_neg = loss_op(pred, target_neg, loss_type=loss_type, time_step_weights=time_step_weights)
    else:
        loss_neg = torch.stack([get_neg_loss(pred_ele, target_ele, loss_type=loss_type, time_step_weights=time_step_weights,
                                            ) for pred_ele, target_ele in zip(pred, target) if pred_ele is not None]).mean()
    return loss_neg


def get_pos_dims_dict(original_shape):
    """Obtain the position dimension based on original_shape.

    Args:
        original_shape: ((key1, shape_tuple1), (key2, shape_tuple2), ...) or the corresponding dict format dict(original_shape).

    Returns:
        pos_dims_dict: {key1: pos_dims1, key2: pos_dims2}
    """
    original_shape_dict = dict(original_shape)
    pos_dims_dict = {key: len(original_shape_dict[key]) for key in original_shape_dict}
    return pos_dims_dict


def process_data_for_CNN(data, use_grads=True, use_pos=False):
    """Process data for CNN, optionally adding gradient and normalized position."""
    data = endow_grads(data) if use_grads else data
    # if use_pos:
    #     for key in data.node_feature:
    #         data.node_feature[key] = torch.cat([data.node_pos[0][0][key].to(data.node_feature[key].device), data.node_feature[key]], -1)
    original_shape = dict(to_tuple_shape(data.original_shape))
    pos_dims = get_pos_dims_dict(original_shape)
    data_node_feature = {}
    for key in data.node_feature:
        if key in to_tuple_shape(data.grid_keys):  # Only for grid nodes
            x = data.node_feature[key]  # x: [n_nodes, input_steps, C]
            x = x.reshape(-1, *original_shape[key], *x.shape[-2:])  # [B, [pos_dims], T, C]
            # assert x.shape[-2] == 1
            """
            if mask_non_badpoints is present, then mask_non_badpoints denotes the points in the input that need to be set to zero.
                and mask denotes the nodes to compute the loss.
            if mask_non_badpoints is not present, then mask acts as both mask_non_badpoints and mask. If it is None, then all nodes are valid.
            """
            if hasattr(data, "mask_non_badpoints"):
                mask_non_badpoints = data.mask_non_badpoints
            else:
                mask_non_badpoints = data.mask
            if mask_non_badpoints is not None:
                mask_non_badpoints = mask_non_badpoints[key].reshape(-1, *original_shape[key])  # [B, [pos_dims]]
                mask_non_badpoints = mask_non_badpoints.unsqueeze(-1).unsqueeze(-1).to(x.device)  # [B, [pos_dims], T:1, C:1]
                x = x * mask_non_badpoints  # [B, [pos_dims], T, C]
            permute_order = (0,) + (1 + pos_dims[key], 2 + pos_dims[key]) + tuple(range(1, 1 + pos_dims[key]))
            x = x.permute(*permute_order)  # [B, T, C, [pos_dims]]
            data_node_feature[key] = x.reshape(x.shape[0], -1, *x.shape[3:])  # [B, T * C, [pos_dims]]
    return data_node_feature


def endow_grads_x(x, original_shape, dyn_dims):
    """
    Append data grad to the left of the x.

    The full x has feature semantics of [compute_dims, static_dims, dyn_dims]
    Let the original x has shape of [nodes, (input_steps), compute_dims + static_dims + dyn_dims]  (in order),
        then the x_new will have shape of [nodes, (input_steps), compute_dims ([dyn_dims * 2 + other_compute_dims]) + static_dims + dyn_dims]
    """
    dyn_dims = to_tuple_shape(dyn_dims)
    pos_dims = len(original_shape)
    x_reshape = x.reshape(-1, *original_shape, *x.shape[1:])  # [batch, [pos_dims], (input_steps), feature_size]
    x_core = x_reshape[..., -dyn_dims:]
    x_diff_list = []
    x_diff_list.append(torch.cat([x_core[:,1:2] - x_core[:,0:1], (x_core[:,2:] - x_core[:,:-2]) / 2, x_core[:,-1:] - x_core[:,-2:-1]], 1))
    if pos_dims >= 2:
        x_diff_list.append(torch.cat([x_core[:,:,1:2] - x_core[:,:,0:1], (x_core[:,:,2:] - x_core[:,:,:-2]) / 2, x_core[:,:,-1:] - x_core[:,:,-2:-1]], 2))
    if pos_dims >= 3:
        x_diff_list.append(torch.cat([x_core[:,:,:,1:2] - x_core[:,:,:,0:1], (x_core[:,:,:,2:] - x_core[:,:,:,:-2]) / 2, x_core[:,:,:,-1:] - x_core[:,:,:,-2:-1]], 3))
    x_new = torch.cat(x_diff_list + [x_reshape], -1)   # [batch, [pos_dims], (input_steps), feature_size]
    x_new = x_new.reshape(-1, *x_new.shape[1+pos_dims:])  # [-1, (input_steps), feature_size]
    return x_new


def endow_grads(data):
    grid_keys = to_tuple_shape(data.grid_keys)
    dyn_dims_dict = dict(to_tuple_shape(data.dyn_dims))
    original_shape = dict(to_tuple_shape(data.original_shape))
    for key in data.node_feature:
        if key in grid_keys:
            data.node_feature[key] = endow_grads_x(data.node_feature[key], original_shape[key], dyn_dims_dict[key])
    return data


def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)
        

class Linear_Ordered(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ordered_neuron_size: int,
        is_reverse: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        static_size: int = 0,
    ) -> None:
        assert in_features == out_features + static_size and ordered_neuron_size > 0
        self.n = int(np.ceil(in_features / ordered_neuron_size))
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear_Ordered, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.static_size = static_size
        self.ordered_neuron_size = ordered_neuron_size
        self.is_reverse = is_reverse

        self.weight = nn.Parameter(torch.zeros((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        """
        The static parameters are by default on the right most columns (represented by B)
        If is_reverse is True: will freeze lower trianguler matrix, where the k th block will only depend on up to k th block as input.
        [[A A A B],
         [0 A A B],
         [0 0 A B]]
        If is_reverse is False: will freeze upper trianguler matrix, where the k th block will depend on k th block and all blocks forward.
        [[A 0 0 0],
         [A A 0 0],
         [A A A 0]]
        """
        if is_reverse:
            freezed_rows, freezed_cols = get_tril_block(out_features, block_size=ordered_neuron_size, diagonal=-1)
        else:
            freezed_rows, freezed_cols = get_triu_block(out_features, block_size=ordered_neuron_size, diagonal=1)
            freezed_rows_static, freezed_cols_static = [], []
            for ii in range(out_features):
                for jj in range(out_features, in_features):
                    freezed_rows_static.append(ii)
                    freezed_cols_static.append(jj)
            freezed_rows_static, freezed_cols_static = np.array(freezed_rows_static), np.array(freezed_cols_static)
            freezed_rows = np.concatenate([freezed_rows, freezed_rows_static])
            freezed_cols = np.concatenate([freezed_cols, freezed_cols_static])
        hook_function = zero_grad_hook_multi(freezed_rows, freezed_cols)
        self.weight.register_hook(hook_function)

    def reset_parameters(self):
        for k in range(self.n):
            if self.is_reverse:
                fan_in = (self.n-k) * self.ordered_neuron_size
                fan_out = self.ordered_neuron_size
            else:
                fan_in = (k+1) * self.ordered_neuron_size
                fan_out = self.ordered_neuron_size
            gain = 1.
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            if self.is_reverse:
                _no_grad_normal_(self.weight[k*self.ordered_neuron_size:(k+1)*self.ordered_neuron_size, k*self.ordered_neuron_size:], 0., std)
            else:
                _no_grad_normal_(self.weight[k*self.ordered_neuron_size:(k+1)*self.ordered_neuron_size, :min((k+1)*self.ordered_neuron_size, self.out_features)], 0., std)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, chunk_size={}, reverse={}, bias={}'.format(
            self.in_features, self.out_features, self.ordered_neuron_size, self.is_reverse, self.bias is not None
        )


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        n_neurons,
        n_layers,
        act_name="rational",
        output_size=None,
        last_layer_linear=True,
        is_res=False,
        normalization_type="None",
        ordered_neuron="None",
    ):
        """
        Args:
            ordered_neuron:
                "None": normal MLP
                "split(r):{chunk_size}": simply ordered into chunk size of {chunk_size}. If there is "r", it means that it is reversed.
                "dropout(r)":{chunk_size}: not only ordered into chunk size of {chunk_size}, but in training time, also performs input dropout. If there is "r", it means that it is reversed.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.n_neurons = n_neurons
        if not isinstance(n_neurons, Number):
            assert n_layers == len(n_neurons), "If n_neurons is not a number, then its length must be equal to n_layers={}".format(n_layers)
        self.n_layers = n_layers
        self.act_name = act_name
        self.output_size = output_size if output_size is not None else n_neurons
        self.last_layer_linear = last_layer_linear
        self.is_res = is_res
        self.normalization_type = normalization_type
        self.normalization_n_groups = 2
        self.ordered_neuron = ordered_neuron
        if self.ordered_neuron == "None":
            self.ordered_neuron_chunk_size = -1
        else:
            self.ordered_neuron_chunk_size = int(self.ordered_neuron.split(":")[1])
        
        if act_name != "siren":
            last_out_neurons = self.input_size
            for i in range(1, self.n_layers + 1):
                out_neurons = self.n_neurons if isinstance(self.n_neurons, Number) else self.n_neurons[i-1]
                if i == self.n_layers and self.output_size is not None:
                    out_neurons = self.output_size

                if i == self.n_layers and self.last_layer_linear == "siren":
                    # Last layer is Siren:
                    from siren_pytorch import Siren
                    setattr(self, "layer_{}".format(i), Siren(
                        last_out_neurons,
                        out_neurons,
                    ))
                else:
                    if self.ordered_neuron == "None":
                        setattr(self, "layer_{}".format(i), nn.Linear(
                            last_out_neurons,
                            out_neurons,
                        ))
                        torch.nn.init.xavier_normal_(getattr(self, "layer_{}".format(i)).weight)
                        torch.nn.init.constant_(getattr(self, "layer_{}".format(i)).bias, 0)
                    else:
                        is_reverse = self.ordered_neuron.startswith("splitr") or self.ordered_neuron.startswith("dropoutr")
                        setattr(self, "layer_{}".format(i), Linear_Ordered(
                            in_features=last_out_neurons,
                            out_features=out_neurons,
                            ordered_neuron_size=self.ordered_neuron_chunk_size,
                            is_reverse=is_reverse,
                            static_size=last_out_neurons-out_neurons,
                        ))
                    last_out_neurons = out_neurons

                # Normalization and activation:
                if i != self.n_layers:
                    # Intermediate layers:
                    if self.act_name != "linear":
                        if self.normalization_type != "None":
                            setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                        setattr(self, "activation_{}".format(i), get_activation(act_name))
                else:
                    # Last layer:
                    if self.last_layer_linear in [False, "False"]:
                        if self.act_name != "linear":
                            if self.normalization_type != "None":
                                setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                            setattr(self, "activation_{}".format(i), get_activation(act_name))
                    elif self.last_layer_linear in [True, "True", "siren"]:
                        pass
                    else:
                        if self.normalization_type != "None":
                            setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                        setattr(self, "activation_{}".format(i), get_activation(self.last_layer_linear))

        else:
            from siren_pytorch import SirenNet, Sine
            if self.last_layer_linear in [False, "False"]:
                if self.act_name == "siren":
                    last_layer = Sine()
                else:
                    last_layer = get_activation(act_name)
            elif self.last_layer_linear in [True, "True"]:
                last_layer = nn.Identity()
            elif self.last_layer_linear == "siren":
                last_layer = Sine()
            else:
                last_layer = get_activation(self.last_layer_linear)
            self.model = SirenNet(
                dim_in=input_size,               # input dimension, ex. 2d coor
                dim_hidden=n_neurons,            # hidden dimension
                dim_out=output_size,             # output dimension, ex. rgb value
                num_layers=n_layers,             # number of layers
                final_activation=last_layer,     # activation of final layer (nn.Identity() for direct output). If last_layer_linear is False, then last activation is Siren
                w0_initial=30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
            )

    def forward(self, x):
        if self.act_name != "siren":
            u = x

            for i in range(1, self.n_layers + 1):
                u = getattr(self, "layer_{}".format(i))(u)

                # Normalization and activation:
                if i != self.n_layers:
                    # Intermediate layers:
                    if self.act_name != "linear":
                        if self.normalization_type != "None":
                            u = getattr(self, "normalization_{}".format(i))(u)
                        u = getattr(self, "activation_{}".format(i))(u)
                else:
                    # Last layer:
                    if self.last_layer_linear in [True, "True", "siren"]:
                        pass
                    else:
                        if self.last_layer_linear in [False, "False"] and self.act_name == "linear":
                            pass
                        else:
                            if self.normalization_type != "None":
                                u = getattr(self, "normalization_{}".format(i))(u)
                            u = getattr(self, "activation_{}".format(i))(u)
            if self.is_res:
                x = x + u
            else:
                x = u
            return x
        else:
            return self.model(x)


class MLP_Coupling(nn.Module):
    def __init__(
        self,
        input_size,
        z_size,
        n_neurons,
        n_layers,
        act_name="rational",
        output_size=None,
        last_layer_linear=True,
        is_res=False,
        normalization_type="None",
        is_prioritized_dropout=False,
    ):
        super(MLP_Coupling, self).__init__()
        self.input_size = input_size
        self.n_neurons = n_neurons
        if not isinstance(n_neurons, Number):
            assert n_layers == len(n_neurons), "If n_neurons is not a number, then its length must be equal to n_layers={}".format(n_layers)
        self.n_layers = n_layers
        self.act_name = act_name
        self.output_size = output_size if output_size is not None else n_neurons
        self.last_layer_linear = last_layer_linear
        self.is_res = is_res
        self.normalization_type = normalization_type
        self.normalization_n_groups = 2
        self.is_prioritized_dropout = is_prioritized_dropout
        assert act_name != "siren"
        last_out_neurons = self.input_size
        for i in range(1, self.n_layers + 1):
            out_neurons = self.n_neurons if isinstance(self.n_neurons, Number) else self.n_neurons[i-1]
            if i == self.n_layers and self.output_size is not None:
                out_neurons = self.output_size

            setattr(self, "layer_{}".format(i), nn.Linear(
                last_out_neurons,
                out_neurons,
            ))
            setattr(self, "z_layer_{}".format(i), nn.Linear(
                z_size,
                out_neurons*2,
            ))
            last_out_neurons = out_neurons
            torch.nn.init.xavier_normal_(getattr(self, "layer_{}".format(i)).weight)
            torch.nn.init.constant_(getattr(self, "layer_{}".format(i)).bias, 0)
            torch.nn.init.xavier_normal_(getattr(self, "z_layer_{}".format(i)).weight)
            torch.nn.init.constant_(getattr(self, "z_layer_{}".format(i)).bias, 0)

            # Normalization and activation:
            if i != self.n_layers:
                # Intermediate layers:
                if self.act_name != "linear":
                    if self.normalization_type != "None":
                        setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                    setattr(self, "activation_{}".format(i), get_activation(act_name))
            else:
                # Last layer:
                if self.last_layer_linear in [False, "False"]:
                    if self.act_name != "linear":
                        if self.normalization_type != "None":
                            setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                        setattr(self, "activation_{}".format(i), get_activation(act_name))
                elif self.last_layer_linear in [True, "True"]:
                    pass
                else:
                    if self.normalization_type != "None":
                        setattr(self, "normalization_{}".format(i), get_normalization(self.normalization_type, last_out_neurons, n_groups=self.normalization_n_groups))
                    setattr(self, "activation_{}".format(i), get_activation(self.last_layer_linear))


    def forward(self, x, z, n_dropout=None):
        u = x

        for i in range(1, self.n_layers + 1):
            u = getattr(self, "layer_{}".format(i))(u)
            z_chunks = getattr(self, "z_layer_{}".format(i))(z)
            z_weight, z_bias = torch.chunk(z_chunks, 2, dim=-1)
            u = u * z_weight + z_bias

            # Normalization and activation:
            if i != self.n_layers:
                # Intermediate layers:
                if self.act_name != "linear":
                    if self.normalization_type != "None":
                        u = getattr(self, "normalization_{}".format(i))(u)
                    u = getattr(self, "activation_{}".format(i))(u)
            else:
                # Last layer:
                if self.last_layer_linear in [True, "True"]:
                    pass
                else:
                    if self.last_layer_linear in [False, "False"] and self.act_name == "linear":
                        pass
                    else:
                        if self.normalization_type != "None":
                            u = getattr(self, "normalization_{}".format(i))(u)
                        u = getattr(self, "activation_{}".format(i))(u)
        if self.is_res:
            x = x + u
        else:
            x = u
        return x


class Channel_Gen(object):
    """Generator to generate number of channels depending on the block id (n)."""
    def __init__(self, channel_mode):
        self.channel_mode = channel_mode

    def __call__(self, n):
        if self.channel_mode.startswith("exp"):
            # Exponential growth:
            channel_mul = int(self.channel_mode.split("-")[1])
            n_channels = channel_mul * 2 ** n
        elif self.channel_mode.startswith("c"):
            # Constant:
            n_channels = int(self.channel_mode.split("-")[1])
        else:
            channels = [int(ele) for ele in self.channel_mode.split("-")]
            n_channels = channels[n]
        return n_channels


def get_batch_size(data):
    """Get batch_size"""
    if hasattr(data, "node_feature"):
        first_key = next(iter(data.node_feature))
        original_shape = dict(to_tuple_shape(data.original_shape))
        batch_size = data.node_feature[first_key].shape[0] // np.prod(original_shape[first_key])
    else:
        batch_size = len(data.t)
    return batch_size


def get_elements(src, string_idx):
    if ":" not in string_idx:
        idx = eval(string_idx)
        return src[idx: idx+1]
    else:
        if string_idx.startswith(":"):
            return src[:eval(string_idx[1:])]
        elif string_idx.endswith(":"):
            return src[eval(string_idx[:-1]):]
        else:
            string_idx_split = string_idx.split(":")
            assert len(string_idx_split) == 2
            return src[eval(string_idx_split[0]): eval(string_idx_split[1])]


def parse_string_idx_to_list(string_idx, max_t=None, is_inclusive=True):
    """Parse a index into actual list. E.g.
    E.g.:
        '4'  -> [4]
        ':3' -> [1, 2, 3]
        '2:' -> [2, 3, ... max_t]
        '2:4' -> [2, 3, 4]
    """
    if isinstance(string_idx, int):
        return [string_idx]
    elif isinstance(string_idx, str):
        if ":" not in string_idx:
            idx = eval(string_idx)
            return [idx]
        else:
            if string_idx.startswith(":"):
                return (np.arange(eval(string_idx[1:])) + (1 if is_inclusive else 0)).tolist()
            elif string_idx.endswith(":"):
                return (np.arange(eval(string_idx[:1]), max_t+1)).tolist()
            else:
                string_idx_split = string_idx.split(":")
                assert len(string_idx_split) == 2
                return (np.arange(eval(string_idx_split[0]), eval(string_idx_split[1])+(1 if is_inclusive else 0))).tolist()


def get_data_comb(dataset):
    """Get collated data for full dataset, collated along the batch dimension."""
    from deepsnap.batch import Batch as deepsnap_Batch
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, collate_fn=deepsnap_Batch.collate(), batch_size=len(dataset))
    for data in data_loader:
        break
    return data


def combine_node_label_time(dataset):
    """Combine the node_label across the time dimension, starting with the first data."""
    data = deepcopy(dataset[0])
    node_label_dict = {key: [] for key in data.node_label}
    for data in dataset:
        for key in data.node_label:
            node_label_dict[key].append(data.node_label[key])
    node_label_dict[key] = torch.cat(node_label_dict[key], 1)
    data.node_label = node_label_dict
    return data



def get_root_dir(level=0):
    """Obtain the root directory of the repo.
    Args:
        level: the relative level w.r.t. the repo.
    """
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("plasma")
    dirname = "/".join(dirname_split[:index + 1 + level])
    return dirname


def is_diagnose(loc, filename):
    """If the given loc and filename matches that of the diagose.yml, will return True and (later) call an pde.set_trace()."""
    try:
        with open(get_root_dir() + "/design/multiscale/diagnose.yml", "r") as f:
            Dict = yaml.load(f, Loader=yaml.FullLoader)
    except:
        return False
    Dict.pop(None, None)
    if not ("loc" in Dict and "dirname" in Dict and "filename" in Dict):
        return False
    if loc == Dict["loc"] and filename == op.path.join(Dict["dirname"], Dict["filename"]):
        return True
    else:
        return False


def interpolate_2d(node_feature_2, length=4):
    from scipy.interpolate import griddata
    node_feature_2 = node_feature_2[:, 1:-1]
    x_mesh, y_mesh = np.meshgrid(np.arange(2*length+2), np.arange(256))
    mesh = np.stack([x_mesh, y_mesh], -1)
    points = np.concatenate([mesh[:, :length], mesh[:, -length:]], 1).reshape(-1, 2)  # (256*length*2, 2)
    values = torch.cat([node_feature_2[:, -length:], node_feature_2[:, :length]], 1).reshape(-1)  # (256*length*2)
    xi = mesh[:, length:-length].reshape(-1, 2)

    interpolation = griddata(points, values, xi, method='cubic').reshape(-1, 2)
    node_feature_2 = torch.cat([torch.FloatTensor(interpolation[...,1:]), node_feature_2, torch.FloatTensor(interpolation[:,:1])], 1)
    return node_feature_2


def get_keys_values(Dict, exclude=None):
    """Obtain the list of keys and values of the Dict, excluding certain keys."""
    if exclude is None:
        exclude = []
    if not isinstance(exclude, list):
        exclude = [exclude]
    keys = []
    values = []
    for key, value in Dict.items():
        if key not in exclude:
            keys.append(key)
            values.append(value)
    return keys, values


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x


def get_prioritized_dropout(x, n_dropout):
    if isinstance(n_dropout, Number):
        if n_dropout > 0:
            x = torch.cat([x[..., :-n_dropout], torch.zeros(*x.shape[:-1], n_dropout, device=x.device)], -1)
    else:
        if n_dropout.sum() == 0:
            return x
        assert x.shape[0] == len(n_dropout)
        x_list = []
        device = x.device
        for i in range(len(x)):
            if n_dropout[i] > 0:
                x_ele_dropout = torch.cat([x[i, ..., :-n_dropout[i]], torch.zeros(*x.shape[1:-1],n_dropout[i], device=device)], -1)
            else:
                x_ele_dropout = x[i]
            x_list.append(x_ele_dropout)
        x = torch.stack(x_list)
    return x


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y, mask=None):
        """
        Args:
            x, y: both have shape of [B, -1]
        """
        num_examples = x.size()[0]

        if mask is None:
            diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
            y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        else:
            mask = mask.view(num_examples, -1).float()
            diff_norms = torch.norm((x.reshape(num_examples,-1) - y.reshape(num_examples,-1)) * mask, self.p, 1)
            y_norms = torch.norm(y.reshape(num_examples,-1) * mask + (1 - mask) * 1e-6, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y, mask=None):
        return self.rel(x, y, mask=mask)


def get_model_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.model_dict
    else:
        return model.model_dict


def add_data_noise(tensor, data_noise_amp):
    if data_noise_amp == 0:
        return tensor
    else:
        return tensor + torch.randn_like(tensor) * data_noise_amp


def deepsnap_to_pyg(data, is_flatten=False, use_pos=False, args_dataset=None):
    assert hasattr(data, "node_feature")
    from torch_geometric.data import Data
    data_pyg = Data(
        x=data.node_feature["n0"],
        y=data.node_label["n0"],
        edge_index=data.edge_index[("n0", "0", "n0")],
        x_pos=data.node_pos["n0"],
        xfaces=data.xfaces["n0"],
        x_bdd=data.x_bdd["n0"],
        original_shape=data.original_shape,
        dyn_dims=data.dyn_dims,
        compute_func=data.compute_func,
    )
    if hasattr(data, "edge_attr"):
        data_pyg.edge_attr = data.edge_attr[("n0", "0", "n0")]
    if hasattr(data, "is_1d_periodic"):
        data_pyg.is_1d_periodic = data.is_1d_periodic
    if hasattr(data, "is_normalize_pos"):
        data_pyg.is_normalize_pos = data.is_normalize_pos
    if hasattr(data, "dataset"):
        data_pyg.dataset = data.dataset
    if hasattr(data, "param"):
        data_pyg.param = data.param["n0"]
    if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
        data_pyg.yedge_index = data.yedge_index["n0"]
        data_pyg.y_tar = data.y_tar["n0"]
        data_pyg.y_back = data.y_back["n0"]
        data_pyg.yface_list = data.yface_list["n0"]
    if hasattr(data, "mask"):
        data_pyg.mask = data.mask["n0"]
    if is_flatten:
        is_1d_periodic = to_tuple_shape(data.is_1d_periodic)
        if is_1d_periodic:
            lst = [data_pyg.x.flatten(start_dim=1)]
        else:
            lst = [data_pyg.x.flatten(start_dim=1), data_pyg.x_bdd]
        if use_pos:
            lst.insert(1, data_pyg.x_pos)
        if hasattr(data_pyg, "dataset") and to_tuple_shape(data_pyg.dataset).startswith("mppde1dh"):
            lst.insert(1, data_pyg.param[:1].expand(data_pyg.x.shape[0], data_pyg.param.shape[-1]))
        data_pyg.x = torch.cat(lst, -1)
        if data_pyg.y is not None:
            data_pyg.y = data_pyg.y.flatten(start_dim=1)
    return data_pyg


def attrdict_to_pygdict(attrdict, is_flatten=False, use_pos=False):    
    pygdict = {
        "x": attrdict["node_feature"]["n0"],
        "y": attrdict["node_label"]["n0"],
        #"y_tar": attrdict["y_tar"]["n0"],
        #"y_back": attrdict["y_back"]["n0"],
        "x_pos": attrdict["x_pos"]["n0"],
        "edge_index": attrdict["edge_index"][("n0","0","n0")],
        "xfaces": attrdict["xfaces"]["n0"],
        #"yface_list": attrdict["yface_list"]["n0"], 
        #"yedge_index": attrdict["yedge_index"]["n0"],
        "x_bdd": attrdict["x_bdd"]["n0"], 
        "original_shape": attrdict["original_shape"],
        "dyn_dims": attrdict["dyn_dims"],
        "compute_func": attrdict["compute_func"],
        "grid_keys": attrdict["grid_keys"],
        "part_keys": attrdict["part_keys"], 
        "time_step": attrdict["time_step"], 
        "sim_id": attrdict["sim_id"],
        "time_interval": attrdict["time_interval"], 
        "cushion_input": attrdict["cushion_input"],
    }
    if "edge_attr" in attrdict:
        pygdict["edge_attr"] = attrdict["edge_attr"]["n0"][0]
    if len(dict(to_tuple_shape(attrdict["original_shape"]))["n0"]) == 0:
        pygdict["bary_weights"] = attrdict["bary_weights"]["n0"]
        pygdict["bary_indices"] = attrdict["bary_indices"]["n0"]
        pygdict["hist_weights"] = attrdict["hist_weights"]["n0"]
        pygdict["hist_indices"] = attrdict["hist_indices"]["n0"]
        pygdict["yedge_index"] = attrdict["yedge_index"]["n0"]
        pygdict["y_back"] = attrdict["y_back"]["n0"]
        pygdict["y_tar"] = attrdict["y_tar"]["n0"]
        pygdict["yface_list"] = attrdict["yface_list"]["n0"]
        pygdict["history"] = attrdict["history"]["n0"]
        pygdict["yfeatures"] = attrdict["yfeatures"]["n0"]
        pygdict["node_dim"] = attrdict["node_dim"]["n0"]
        pygdict["xface_list"] = attrdict["xface_list"]["n0"]
        pygdict["reind_yfeatures"] = attrdict["reind_yfeatures"]["n0"]
        pygdict["batch_history"] = attrdict["batch_history"]["n0"]
    if "onehot_list" in attrdict:
        pygdict["onehot_list"] = attrdict["onehot_list"]["n0"]
        pygdict["kinematics_list"] = attrdict["kinematics_list"]["n0"]   
    if is_flatten:
        if use_pos:
            pygdict["x"] = torch.cat([pygdict["x"].flatten(start_dim=1), pygdict["x_pos"], pygdict["x_bdd"]], -1)
        else:
            pygdict["x"] = torch.cat([pygdict["x"].flatten(start_dim=1), pygdict["x_bdd"]], -1)
    return Attr_Dict(pygdict)


def sample_reward_beta(reward_beta_str,batch_size=1):
    """
    Args:
        reward_beta_str: "0.5-2:linear" (sample from 0.5 to 2, with uniform sampling), "0.0001-1:log" (sample from 0.0001 to 1, with uniform sampling in log scale). Default "1".
    """
    if len(reward_beta_str.split(":")) == 1:
        value_str, mode = reward_beta_str, "linear"
    else:
        value_str, mode = reward_beta_str.split(":")
    if len(value_str.split("-")) == 1:
        min_value, max_value = eval(value_str), eval(value_str)
    else:
        min_value, max_value = value_str.split("-")
        min_value, max_value = eval(min_value), eval(max_value)
    assert min_value <= max_value
    if min_value == max_value:
        return np.ones(batch_size)*min_value
    if mode == "linear":
        reward_beta = np.random.rand(batch_size) * (max_value - min_value) + min_value
    elif mode == "log":
        reward_beta = np.exp(np.random.rand(batch_size) * (np.log(max_value) - np.log(min_value)) + np.log(min_value))
    else:
        raise
    return reward_beta


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_grad_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0).item()
    return total_norm


def copy_data(data, detach=True):
    """Copy Data instance, and detach from source Data."""
    if isinstance(data, dict):
        dct = {key: copy_data(value) for key, value in data.items()}
        if data.__class__.__name__ == "Attr_Dict":
            dct = Attr_Dict(dct)
        return dct
    elif isinstance(data, list):
        return [copy_data(ele) for ele in data]
    elif isinstance(data, torch.Tensor):
        if detach:
            return data.detach().clone()
        else:
            return data.clone()
    elif isinstance(data, tuple):
        return tuple(copy_data(ele) for ele in data)
    elif data.__class__.__name__ in ['HeteroGraph', 'Data']:
        dct = Attr_Dict({key: copy_data(value) for key, value in vars(data).items()})
        assert len(dct) > 0, "Did not clone anything. Check that your PyG version is below 1.8, preferablly 1.7.1. Follow the the ./design/multiscale/README.md to install the correct version of PyG."
        return dct
    elif data is None:
        return data
    else:
        return deepcopy(data)


def detach_data(data):
    if hasattr(data, "detach"):
        return data.detach()
    elif data is None:
        return data
    else:
        for key, item in vars(data).items():
            if hasattr(item, "detach"):
                setattr(data, key, item.detach())
        return data


def edge_index_to_num(edge_index, N):
    #print(edge_index.max())
    assert edge_index.max() < N
    assert N.dtype == torch.int64
    assert edge_index.dtype == torch.int64
    
    return edge_index[0] * N + edge_index[1]


def pyg2obj(pygmesh, path, filename=None):
    if not filename:
        objfile = open(os.path.join(path, "pyg2meshobj.obj"), 'w')
    else:
        objfile = open(os.path.join(path, filename), 'w')
    # vt 
    for v in pygmesh.x[:,:2].cpu().detach().numpy():
        objfile.write("vt ")
        # print(v[1], v[1].item())
        if v[0].item().is_integer():
            objfile.write(str(int(v[0].item())))
        else:
            objfile.write(str(v[0]))
        objfile.write(" ")
        if v[1].item().is_integer():
            objfile.write(str(int(v[1].item())))
        else:
            objfile.write(str(v[1]))
        objfile.write("\n")

    # v, yn
    for v in pygmesh.x.cpu().detach().numpy():
        objfile.write("v ")
        if v[3].item().is_integer():
            objfile.write(str(int(v[3].item())))
        else:
            objfile.write(str(v[3]))
        objfile.write(" ")
        if v[4].item().is_integer():
            objfile.write(str(int(v[4].item())))
        else:
            objfile.write(str(v[4]))
        objfile.write(" ")
        if v[5].item().is_integer():
            objfile.write(str(int(v[5].item())))
        else:
            objfile.write(str(v[5]))
        objfile.write("\n")
        objfile.write("ny ")
        if v[0].item().is_integer():
            objfile.write(str(int(v[0].item())))
        else:
            objfile.write(str(v[0]))
        objfile.write(" ")
        if v[1].item().is_integer():
            objfile.write(str(int(v[1].item())))
        else:
            objfile.write(str(v[1]))
        objfile.write(" ")
        if v[2].item().is_integer():
            objfile.write(str(int(v[2].item())))
        else:
            objfile.write(str(v[2]))
        objfile.write("\n")
        objfile.write("nv ")
        if v[6].item().is_integer():
            objfile.write(str(int(v[6].item())))
        else:
            objfile.write(str(v[6]))
        objfile.write(" ")
        if v[7].item().is_integer():
            objfile.write(str(int(v[7].item())))
        else:
            objfile.write(str(v[7]))
        objfile.write(" ")
        if v[8].item().is_integer():
            objfile.write(str(int(v[8].item())))
        else:
            objfile.write(str(v[8]))
        objfile.write("\n")

    # face
    for face in pygmesh.xfaces.T.cpu().detach().numpy():
        objfile.write("f ")
        objfile.write(str(int(face[0]+1))+"/"+str(int(face[0]+1)))
        objfile.write(" ")
        objfile.write(str(int(face[1]+1))+"/"+str(int(face[1]+1)))
        objfile.write(" ")
        objfile.write(str(int(face[2]+1))+"/"+str(int(face[2]+1)))
        objfile.write("\n")

    objfile.close()     
    
def verface2obj(vertices, faces, path, filename=None):
    if not filename:
        objfile = open(os.path.join(path, "pyg2meshobj.obj"), 'w')
    else:
        objfile = open(os.path.join(path, filename), 'w')
    # vt 
    for v in vertices[:,:2].cpu().detach().numpy():
        objfile.write("vt ")
        # print(v[1], v[1].item())
        if v[0].item().is_integer():
            objfile.write(str(int(v[0].item())))
        else:
            objfile.write(str(v[0]))
        objfile.write(" ")
        if v[1].item().is_integer():
            objfile.write(str(int(v[1].item())))
        else:
            objfile.write(str(v[1]))
        objfile.write("\n")

    # v, yn
    for v in vertices.cpu().detach().numpy():
        objfile.write("v ")
        if v[3].item().is_integer():
            objfile.write(str(int(v[3].item())))
        else:
            objfile.write(str(v[3]))
        objfile.write(" ")
        if v[4].item().is_integer():
            objfile.write(str(int(v[4].item())))
        else:
            objfile.write(str(v[4]))
        objfile.write(" ")
        if v[5].item().is_integer():
            objfile.write(str(int(v[5].item())))
        else:
            objfile.write(str(v[5]))
        objfile.write("\n")
        objfile.write("ny ")
        if v[0].item().is_integer():
            objfile.write(str(int(v[0].item())))
        else:
            objfile.write(str(v[0]))
        objfile.write(" ")
        if v[1].item().is_integer():
            objfile.write(str(int(v[1].item())))
        else:
            objfile.write(str(v[1]))
        objfile.write(" ")
        if v[2].item().is_integer():
            objfile.write(str(int(v[2].item())))
        else:
            objfile.write(str(v[2]))
        objfile.write("\n")
        # objfile.write("nv ")
        # if v[6].item().is_integer():
        #     objfile.write(str(int(v[6].item())))
        # else:
        #     objfile.write(str(v[6]))
        # objfile.write(" ")
        # if v[7].item().is_integer():
        #     objfile.write(str(int(v[7].item())))
        # else:
        #     objfile.write(str(v[7]))
        # objfile.write(" ")
        # if v[8].item().is_integer():
        #     objfile.write(str(int(v[8].item())))
        # else:
        #     objfile.write(str(v[8]))
        objfile.write("\n")

    # face
    for face in faces.T.cpu().detach().numpy():
        objfile.write("f ")
        objfile.write(str(int(face[0]+1))+"/"+str(int(face[0]+1)))
        objfile.write(" ")
        objfile.write(str(int(face[1]+1))+"/"+str(int(face[1]+1)))
        objfile.write(" ")
        objfile.write(str(int(face[2]+1))+"/"+str(int(face[2]+1)))
        objfile.write("\n")

    objfile.close()     

