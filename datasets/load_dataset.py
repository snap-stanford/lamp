import argparse
import datetime
from deepsnap.batch import Batch as deepsnap_Batch
from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph
from deepsnap.hetero_graph import HeteroGraph
import numpy as np
import pdb
import pickle
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time


import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from lamp.datasets.mppde1d_dataset import MPPDE1D
from lamp.datasets.arcsimmesh_dataset import ArcsimMesh
from lamp.pytorch_net.util import ddeepcopy as deepcopy, Batch, make_dir
from lamp.utils import p, PDE_PATH, get_elements, is_diagnose, get_keys_values, loss_op, to_tuple_shape, parse_string_idx_to_list, parse_multi_step, get_device, get_activation, get_normalization, to_cpu, add_data_noise


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

    train_val_fraction = 0.9
    train_fraction = args.train_fraction
    multi_step_dict = parse_multi_step(args.multi_step)
    max_pred_steps = max(list(multi_step_dict.keys()) + [1]) * args.temporal_bundle_steps
    filename_train_val = os.path.join(PDE_PATH, "deepsnap", "{}_train_val_in_{}_out_{}{}{}{}{}{}.p".format(
        args.dataset, args.input_steps * args.temporal_bundle_steps, max_pred_steps, 
        "_itv_{}".format(args.time_interval) if args.time_interval > 1 else "",
        "_yvar_{}".format(args.is_y_variable_length) if args.is_y_variable_length is True else "",
        "_noise_{}".format(args.data_noise_amp) if args.data_noise_amp > 0 else "",
        "_periodic_{}".format(args.is_1d_periodic) if args.is_1d_periodic else "",
        "_normpos_{}".format(args.is_normalize_pos) if args.is_normalize_pos is False else "",
    ))
    filename_test = os.path.join(PDE_PATH, "deepsnap", "{}_test_in_{}_out_{}{}{}{}{}{}.p".format(
        args.dataset, args.input_steps * args.temporal_bundle_steps, max_pred_steps, 
        "_itv_{}".format(args.time_interval) if args.time_interval > 1 else "",
        "_yvar_{}".format(args.is_y_variable_length) if args.is_y_variable_length is True else "",
        "_noise_{}".format(args.data_noise_amp) if args.data_noise_amp > 0 else "",
        "_periodic_{}".format(args.is_1d_periodic) if args.is_1d_periodic else "",
        "_normpos_{}".format(args.is_normalize_pos) if args.is_normalize_pos is False else "",
    ))
    make_dir(filename_train_val)
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