#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from copy import deepcopy
from pickle import FALSE
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_scatter import scatter
import torch.nn.functional as F
import numpy as np
import torch_geometric
import torch_scatter

from random import choice
import pdb
import time
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from lamp.datasets.arcsimmesh_dataset import ArcsimMesh
from lamp.datasets.mppde1d_dataset import get_data_pred, update_edge_attr_1d
from lamp.pytorch_net.util import to_np_array, init_args, to_cpu, Attr_Dict
from lamp.pytorch_net.net import fill_triangular, matrix_diag_transform
from lamp.utils_model import FCBlock
from lamp.utils import p, copy_data, sample_reward_beta, deepsnap_to_pyg, attrdict_to_pygdict, loss_op_core, parse_multi_step, to_tuple_shape, get_activation, seed_everything, edge_index_to_num, add_edge_normal_curvature, load_data

try:
    import dolfin as dolfin
except:
    pass

# ## Helper classes:

# In[ ]:


class processor_mean(MessagePassing):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        layer_norm=False,
        act_name='relu',
        edge_attr=False
    ):
        super(processor_mean, self).__init__(aggr="mean")  # "Add" aggregation.
        self.edge_attr = edge_attr
        if self.edge_attr:
            in_features = in_channels * 3
        else:
            in_features = in_channels * 2
            
        self.edge_encoder = FCBlock(in_features=in_features, 
                                    out_features=out_channels,
                                    num_hidden_layers=2,
                                    hidden_features=in_channels,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=layer_norm,
                                   )
        self.node_encoder = FCBlock(in_features=in_channels*2,
                                    out_features=out_channels,
                                    num_hidden_layers=2,
                                    hidden_features=in_channels,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=layer_norm,
                                   )
        self.latent_dim = out_channels
    def forward(self, graph):
        # pdb.set_trace()
        edge_index = graph.edge_index
        # cat features together (eij,vi,ei)
        x_receiver = torch.gather(graph.x, 0, edge_index[0,:].unsqueeze(-1).repeat(1, graph.x.shape[1]))
        x_sender = torch.gather(graph.x, 0, edge_index[1,:].unsqueeze(-1).repeat(1,graph.x.shape[1]))
        # pdb.set_trace()
        if not self.edge_attr:
            edge_features = torch.cat([x_receiver, x_sender], dim=-1)
        else:
            edge_features = torch.cat([x_receiver, x_sender, graph.edge_attr], dim=-1)
        # edge processor
        edge_features = self.edge_encoder(edge_features)
        
        # aggregate edge_features
        #try:
        save_edges = edge_index.clone().detach().cpu()
        save_x = graph.x.clone().detach().cpu()
        save_edgefeat = edge_features.clone().detach().cpu()
        #pdb.set_trace()
        node_features = self.propagate(edge_index, x=graph.x, edge_attr=edge_features)
        #except:
        #    pdb.set_trace()
        # cat features for node processor (vi,\sum_eij)
        features = torch.cat([graph.x, node_features[:,self.latent_dim:]],dim=-1)
        # node processor and update graph
        graph.x = self.node_encoder(features) + graph.x
        if self.edge_attr:
            graph.edge_attr = edge_features
        return graph

    def message(self, x_i, edge_attr):
        z = torch.cat([x_i, edge_attr], dim=-1)
        return z


class processor(MessagePassing):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        layer_norm=False,
        act_name='relu',
        edge_attr=False
    ):
        super(processor, self).__init__(aggr='add')  # "Add" aggregation.
        self.edge_attr = edge_attr
        if self.edge_attr:
            in_features = in_channels * 3
        else:
            in_features = in_channels * 2
            
        self.edge_encoder = FCBlock(in_features=in_features, 
                                    out_features=out_channels,
                                    num_hidden_layers=2,
                                    hidden_features=in_channels,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=layer_norm,
                                   )
        self.node_encoder = FCBlock(in_features=in_channels*2,
                                    out_features=out_channels,
                                    num_hidden_layers=2,
                                    hidden_features=in_channels,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=layer_norm,
                                   )
        self.latent_dim = out_channels
    def forward(self, graph):
        # pdb.set_trace()
        edge_index = graph.edge_index
        # cat features together (eij,vi,ei)
        x_receiver = torch.gather(graph.x, 0, edge_index[0,:].unsqueeze(-1).repeat(1, graph.x.shape[1]))
        x_sender = torch.gather(graph.x, 0, edge_index[1,:].unsqueeze(-1).repeat(1,graph.x.shape[1]))
        # pdb.set_trace()
        if not self.edge_attr:
            edge_features = torch.cat([x_receiver, x_sender], dim=-1)
        else:
            edge_features = torch.cat([x_receiver, x_sender, graph.edge_attr], dim=-1)
        # edge processor
        edge_features = self.edge_encoder(edge_features)
        
        # aggregate edge_features
        #try:
        save_edges = edge_index.clone().detach().cpu()
        save_x = graph.x.clone().detach().cpu()
        save_edgefeat = edge_features.clone().detach().cpu()
        #pdb.set_trace()
        node_features = self.propagate(edge_index, x=graph.x, edge_attr=edge_features)
        #except:
        #    pdb.set_trace()
        # cat features for node processor (vi,\sum_eij)
        features = torch.cat([graph.x, node_features[:,self.latent_dim:]],dim=-1)
        # node processor and update graph
        graph.x = self.node_encoder(features) + graph.x
        if self.edge_attr:
            graph.edge_attr = edge_features
        return graph

    def message(self, x_i, edge_attr):
        z = torch.cat([x_i, edge_attr], dim=-1)
        return z


class normalizer(nn.Module):
    def __init__(self, dim, mean=0, std=1e-8, max_acc = 60*600):
        super().__init__()
        self.acc_sum = nn.Parameter(torch.zeros(dim, device=self.device), requires_grad=False)
        self.acc_sum_squared = nn.Parameter(torch.zeros(dim, device=self.device), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(dim, device=self.device), requires_grad=False)
        self.std = nn.Parameter(torch.ones(dim, device=self.device), requires_grad=False)
        
        self.total_acc = 0
        self.max_acc = max_acc

    def update(self, value, train):
        if self.total_acc<self.max_acc*value.shape[0] and train:
            self.total_acc += value.shape[0]
            self.acc_sum += torch.sum(value,0).data
            self.acc_sum_squared += torch.sum(value**2,0).data
            safe_count = max(1,self.total_acc)
            self.mean = nn.Parameter(self.acc_sum/safe_count)
            self.std = nn.Parameter(torch.maximum(torch.sqrt(self.acc_sum_squared / safe_count - self.mean**2),torch.tensor(1e-5, dtype=float, device=self.device)))
        return (value-self.mean.data)/self.std.data
    
    def reverse(self,value):
        return value*self.std.data+self.mean.data


def get_data_dropout(data, dropout_mode, exclude_idx=None, sample_idx=None):
    """
    Randomly dropout nodes of a path graph. Only work for 1d case.

    Args:
        dropout_mode: "None" or f"node:{dropout_prob}" or f"node:{dropout_prob}:0.3" or "uniform:{interval}".
            if startswith "uniform", the exclude_idx will be used to do uniform subsampling.
    """
    if dropout_mode == "None":
        return data
    if len(dropout_mode.split(":")) == 2:
        dropout_target, dropout_prob = dropout_mode.split(":")
        is_dropout_prob = 1
    elif len(dropout_mode.split(":")) == 3:
        dropout_target, dropout_prob, is_dropout_prob = dropout_mode.split(":")
        is_dropout_prob = float(is_dropout_prob)
    if "-" in dropout_prob:
        dropout_prob_min, dropout_prob_max = dropout_prob.split("-")
        dropout_prob_min, dropout_prob_max = float(dropout_prob_min), float(dropout_prob_max)
    else:
        dropout_prob_min, dropout_prob_max = float(dropout_prob), float(dropout_prob)

    if dropout_target in ["node", "uniform"]:
        if np.random.rand() > is_dropout_prob:
            return data
        if hasattr(data, "node_feature"):
            length = data.node_feature["n0"].shape[0]
            device = data.node_feature["n0"].device
        else:
            length = data.x.shape[0]
            device = data.x.device
        nx = dict(to_tuple_shape(data.original_shape))["n0"][0]
        batch_size = length // nx
        edge_index_new_list = []
        nx_new_sum = 0
        idx_list = []
        batch_list = []
        if dropout_target == "uniform":
            interval = int(dropout_mode.split(":")[1])
            include_idx = np.concatenate([np.arange(0, nx, interval), np.array([nx-1])])
            exclude_idx = [idx for idx in range(nx) if idx not in include_idx]
        for ii in range(batch_size):
            dropout_prob_chosen = dropout_prob_min + np.random.rand() * (dropout_prob_max - dropout_prob_min)
            if exclude_idx is None:
                nx_new = int(nx * (1 - dropout_prob_chosen))
            else:
                if not isinstance(exclude_idx, list):
                    exclude_idx = [exclude_idx]
                nx_new = nx - len(exclude_idx)
            if exclude_idx is None and (sample_idx is not None):
                nx_new = min(nx_new, len(sample_idx))
            if hasattr(data, "node_feature"):
                edge_index_new = data.edge_index[("n0", "0", "n0")][:,:(nx_new-1)*2]
            else:
                edge_index_new = data.edge_index[:,:(nx_new-1)*2]

            if exclude_idx is None:
                idx = np.sort(np.random.choice(np.arange(1, nx-1), size=nx_new-2, replace=False))
                idx = np.concatenate([np.array([0]), idx, np.array([nx-1])])
                if sample_idx is not None:
                    idx = np.sort(np.random.choice(sample_idx, size=nx_new, replace=False))
            else:
                idx = np.array([i for i in range(nx) if i not in exclude_idx])
            idx_list.append(idx + nx * ii)
            edge_index_new_list.append(edge_index_new + nx_new_sum)
            batch_list.append(torch.ones(nx_new, device=device)*ii)
            nx_new_sum += nx_new
        idx_core = np.concatenate(idx_list)
        batch = torch.cat(batch_list)
        edge_index_new_list = torch.cat(edge_index_new_list, -1)
        if hasattr(data, "node_feature"):
            data_new = Attr_Dict({
                "node_feature": {"n0": data.node_feature["n0"][idx_core]},
                "node_label": {"n0": data.node_label["n0"][idx_core]},
                "node_pos": {"n0": data.node_pos["n0"][idx_core]},
                "x_bdd": {"n0": data.x_bdd["n0"][idx_core]},
                "xfaces": data.xfaces,
                "edge_index": {("n0", "0", "n0"): edge_index_new_list},
                "original_shape": (('n0', (nx_new,)),),
                "dyn_dims": data.dyn_dims,
                "compute_func": data.compute_func,
                "dataset": to_tuple_shape(data.dataset),
                "mask": {"n0": data.mask["n0"]},
                "batch": batch,
            })
            if hasattr(data, "edge_attr"):
                rel_pos = data_new.node_pos["n0"][data_new.edge_index[("n0", "0", "n0")][0]] - data_new.node_pos["n0"][data_new.edge_index[("n0", "0", "n0")][1]]
                if data.edge_attr[("n0", "0", "n0")].shape[-1] == 1:
                    data_new["edge_attr"] = {("n0", "0", "n0"): rel_pos}
                elif data.edge_attr[("n0", "0", "n0")].shape[-1] == 2:
                    data_new["edge_attr"] = {("n0", "0", "n0"): torch.cat([rel_pos, rel_pos.abs()], -1)}
                else:
                    raise
        else:
            data_new = Data(
                x=data.x[idx_core],
                y=data.y[idx_core],
                x_pos=data.x_pos[idx_core],
                x_bdd=data.x_bdd[idx_core],
                xfaces=data.xfaces,
                edge_index=edge_index_new_list, # Assuming 1D path graph, and the specific way of edge_index.
                original_shape=(('n0', (nx_new,)),),
                dyn_dims=data.dyn_dims,
                compute_func=data.compute_func,
                dataset=to_tuple_shape(data.dataset),
                mask=data.mask,
                batch=batch,
            )
            if hasattr(data, "edge_attr"):
                rel_pos = data_new.x_pos[data_new.edge_index[0]] - data_new.x_pos[data_new.edge_index[1]]
                if data.edge_attr.shape[-1] == 1:
                    data_new.edge_attr = rel_pos
                elif data.edge_attr.shape[-1] == 2:
                    data_new.edge_attr = torch.cat([rel_pos, rel_pos.abs()], -1)
                else:
                    raise
        if hasattr(data, "param"):
            data_new.param = data.param
        if hasattr(data, "dataset"):
            data_new.dataset = data.dataset
    else:
        raise
    return data_new


def get_minus_reward(loss, state, time, reward_mode="None", reward_beta=1.):
    """
    Args:
        reward_mode:
            "None": r = loss + time
            ""
    """
    if reward_mode in ["None", "loss+time"]:
        return loss.item() + time * reward_beta
    elif reward_mode == "loss+state":
        return loss.item() + len(state)/1e5 * reward_beta
    else:
        raise


def get_reward_batch(
    loss,
    state,
    time,
    loss_alt=None,
    state_alt=None,
    time_alt=None,
    reward_mode="None",
    reward_beta=1.,
    reward_loss_coef=100,
    prefix="",
):
    """
    Minus reward in batch.

    Args:
        loss: shape of [B, 1]
        reward_mode:
            "None": -r = loss + time * reward_beta
            "loss+state": -r = loss + state_size * reward_beta

    Returns:
        reward: shape [B, 1]
    """
    if len(reward_beta.shape)==1: reward_beta=reward_beta[:,None]

    # if isinstance(reward_beta, str):
    #     reward_beta = sample_reward_beta(reward_beta)

    if reward_mode in ["None", "loss+time"]:
        loss = loss.detach()
        reward = -(loss * reward_loss_coef * (1-reward_beta) + time * 10 * reward_beta)
        info = {prefix+"r_loss": -loss * reward_loss_coef * (1-reward_beta),
                prefix+"r_time": -time * reward_beta,
                prefix+"r_beta": torch.tensor(reward_beta, dtype=torch.float32),
               }

    elif reward_mode == "loss+state":
        batch = state.batch
        state_size = torch.unique(batch, return_counts=True)[1].unsqueeze(-1).type(torch.float32)
        assert loss.shape == state_size.shape
        loss = loss.detach()
        reward = -(loss * reward_loss_coef * (1-reward_beta) + state_size/400 * reward_beta)
        info = {prefix+"r_loss": -loss * reward_loss_coef * (1-reward_beta),
                prefix+"r_state": -state_size/400 * reward_beta,
                prefix+"r_beta": torch.tensor(reward_beta, dtype=torch.float32),
               }
    
    elif reward_mode == "loss":
        batch = state.batch
        loss = loss.detach()
        reward = -loss * reward_loss_coef * (1-reward_beta)
        info = {prefix+"r_loss": -loss * reward_loss_coef * (1-reward_beta),
                prefix+"r_beta": torch.tensor(reward_beta, dtype=torch.float32),
               }

    elif reward_mode == "state":
        state_size = torch.unique(state.batch, return_counts=True)[1].unsqueeze(-1).type(torch.float32)
        assert loss.shape == state_size.shape
        reward = -state_size/400 * reward_beta
        info = {prefix+"r_state": -state_size/400 * reward_beta,
                prefix+"r_beta": torch.tensor(reward_beta, dtype=torch.float32),
               }

    elif reward_mode in ["lossdiff+statediff", "statediff", "lossdiff", "lossdiff+timediff", "timediff"]:
        # assert (0 <= reward_beta).any() and (reward_beta <= 1).any()
        state_size = torch.unique(state.batch, return_counts=True)[1].unsqueeze(-1).type(torch.float32)
        state_size_alt = torch.unique(state_alt.batch, return_counts=True)[1].unsqueeze(-1).type(torch.float32)
        loss_diff = (loss_alt - loss).detach()
        state_diff = state_size_alt - state_size
        time_diff = (time_alt - time)
        info = {prefix+"v/lossdiff": loss_diff * reward_loss_coef,
                prefix+"v/statediff": state_diff / 100,
                prefix+"v/timediff": torch.tensor(time_diff) * 100,
                prefix+"v/beta": reward_beta.float(),
                prefix+"v/state_size": state_size,

                prefix+"r/lossdiff": loss_diff * reward_loss_coef * (1-reward_beta),
                prefix+"r/statediff": state_diff / 100 * reward_beta,
                prefix+"r/timediff": torch.tensor(time_diff) * 100 * reward_beta,
               }
        
        if reward_mode == "statediff":
            reward = state_diff / 100 * reward_beta
        elif reward_mode == "lossdiff":
            reward = loss_diff * reward_loss_coef * (1-reward_beta)
        elif reward_mode == "lossdiff+statediff":
            # print("loss_diff",loss_diff.mean())
            reward = loss_diff * reward_loss_coef * (1-reward_beta) + state_diff / 100 * reward_beta
        elif reward_mode == "lossdiff+timediff":
            reward = loss_diff * reward_loss_coef * (1-reward_beta) + torch.tensor(time_diff)[None] * 100 * reward_beta
        elif reward_mode == "timediff":
            reward = torch.tensor(time_diff)[None] * 100 * reward_beta
        else:
            raise

    else:
        raise
    return reward, info


def parse_rl_coefs(rl_coefs):
    if rl_coefs == "None":
        return {}
    rl_coefs_dict = {}
    for item in rl_coefs.split("+"):
        key, value = item.split(":")
        rl_coefs_dict[key] = float(value)
    return rl_coefs_dict



class Value_Model(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        edge_dim,
        latent_dim=32,
        num_pool=1,
        act_name="relu",
        act_name_final="relu",
        layer_norm=False,
        batch_norm=False,
        num_steps=3,
        pooling_type="global_mean_pool",
        edge_attr=False,
        use_pos=False,
        final_ratio=0.1,
        final_pool="global_mean_pool",
        reward_condition=False,
        processors=[None],
        encoder_edge=None,
        encoder_nodes=None,
    ):
        super().__init__()
        """
        input_size - input feature size
        edge_dim - edge dimension size
        latent_dim - latent dimension size of node features
        num_pool - number of pooling steps
        act_name - activation function
        """
        self.input_size = input_size
        self.output_size = output_size
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim
        self.num_pool = num_pool
        self.act_name = act_name
        self.act_name_final = act_name_final
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.num_steps = num_steps
        self.pooling_type = pooling_type
        self.edge_attr = edge_attr
        self.use_pos = use_pos
        self.final_ratio = final_ratio
        self.final_pool = final_pool
        self.reward_condition = reward_condition
        if self.edge_dim>6:
            import dolfin as dolfin
            self.dolfin = dolfin
            # self.input_size = 8
            
        if (processors[0]==None):
            self.encoder_edge = FCBlock(in_features=edge_dim,
                                        out_features=latent_dim,
                                        num_hidden_layers=2,
                                        hidden_features=latent_dim,
                                        outermost_linear=True,
                                        nonlinearity=act_name,
                                        layer_norm=layer_norm,
                                       )
            self.encoder_nodes = FCBlock(in_features=self.input_size + (5 if self.reward_condition else 0),
                                         out_features=latent_dim,
                                         num_hidden_layers=2,
                                         hidden_features=latent_dim,
                                         outermost_linear=True,
                                         nonlinearity=act_name,
                                         layer_norm=layer_norm,
                                        )
        else:
            self.encoder_edge = encoder_edge
            self.encoder_nodes = encoder_nodes
            
        if pooling_type in ["global_max_pool","global_mean_pool","global_mean_pool"]:
            if (processors[0]==None):
                self.processors = []
                for _ in range(num_steps):
                    self.processors.append((processor(latent_dim, 
                                                      latent_dim, 
                                                      layer_norm=layer_norm, 
                                                      act_name=act_name, 
                                                      edge_attr=edge_attr)))
                    if self.batch_norm:
                        self.processors.append(torch.nn.BatchNorm1d(latent_dim))
                        self.processors.append(torch.nn.BatchNorm1d(latent_dim))
                self.processors = torch.nn.Sequential(*self.processors)
            else:
                self.processors = processors
                
        elif pooling_type in ["TopKPooling"]:
            ratio = float(np.exp(np.log(self.final_ratio)/self.num_pool))
            self.topKpoolings = []
            self.processors = []
            for _ in range(num_pool):
                self.topKpoolings.append(torch_geometric.nn.TopKPooling(latent_dim ,ratio))
                self.processors.append((processor(latent_dim, 
                                                  latent_dim, 
                                                  layer_norm=layer_norm, 
                                                  act_name=act_name, 
                                                  edge_attr=edge_attr)))
                if self.batch_norm:
                    self.processors.append(torch.nn.BatchNorm1d(latent_dim))
                    self.processors.append(torch.nn.BatchNorm1d(latent_dim))
            self.processors = torch.nn.Sequential(*self.processors)
            self.topKpoolings = torch.nn.Sequential(*self.topKpoolings)
                
        self.fc1 = nn.Linear(latent_dim + (5 if self.reward_condition else 0), 1)
        self.fc2 = nn.Linear(latent_dim + (5 if self.reward_condition else 0), 1)
        self.activation1 = get_activation(self.act_name_final)
        self.activation2 = get_activation(self.act_name_final)

    def encoder(self, graph):
        """ Encode node and edge features.

        Args:
            graph: pyg data instance for graph.
        """
        graph.x = self.encoder_nodes(graph.x)
        # pdb.set_trace()
        graph.edge_attr = self.encoder_edge(graph.edge_attr)
        return graph

    def get_reward(self, graph_latent,batch,reward_beta=None):
        """ perform pooling and output the graph-level latent representation      
        """
        if self.reward_condition:
            graph_latent = torch.cat([graph_latent,reward_beta[:,None].repeat([1,5])],dim=-1).float()
            
        reward_1 = self.activation1((self.fc1(graph_latent)))
        reward_2 = self.activation2((self.fc2(graph_latent)))
        return reward_1, reward_2

    def global_pooling(self, graph_evolved,reward_beta=None):
        # message passing steps with different MLP each time
        for i in range(self.num_steps):
            # if self.reward_condition:
            #     beta_batch = torch.zeros(graph_evolved.x.shape[0]).to(graph_evolved.x.device)
            #     beta_batch = reward_beta[graph_evolved.batch.to(torch.int64)][:,None]
            #     graph_evolved.x = torch.cat([graph_evolved.x,beta_batch],dim=-1).float()
            #     pdb.set_trace()
            
            if self.batch_norm:
                graph_evolved = self.processors[i*3](graph_evolved)
                graph_evolved.x = self.processors[i*3+1](graph_evolved.x)
                graph_evolved.edge_attr = self.processors[i*3+2](graph_evolved.edge_attr)
            else:
                graph_evolved = self.processors[i](graph_evolved)
        batch = graph_evolved.batch.long()
        if self.pooling_type=="global_max_pool":
            graph_latent = torch_geometric.nn.global_max_pool(graph_evolved.x, batch)
        elif self.pooling_type=="global_add_pool":
            graph_latent = torch_geometric.nn.global_add_pool(graph_evolved.x, batch)
        elif self.pooling_type=="global_mean_pool":
            graph_latent = torch_geometric.nn.global_mean_pool(graph_evolved.x, batch)
        return graph_latent

    def topk_pooling(self,graph_evolved,reward_beta=None):

        for i in range(self.num_pool):
                        
            node_feature,edge_index,edge_attr,batch,perm,score = self.topKpoolings[i](graph_evolved.x,graph_evolved.edge_index,graph_evolved.edge_attr,graph_evolved.batch,)
            graph_evolved.x = node_feature
            # print("pooling layer:{},node feature shape:{}".format(i,graph_evolved.x.shape))

            graph_evolved.edge_index = edge_index
            graph_evolved.edge_attr = edge_attr
            graph_evolved.batch = batch
            graph_evolved.perm = perm 
            graph_evolved.score = score

            if self.batch_norm:
                graph_evolved = self.processors[i*3](graph_evolved)
                graph_evolved.x = self.processors[i*3+1](graph_evolved.x)
                graph_evolved.edge_attr = self.processors[i*3+2](graph_evolved.edge_attr)
            else:
                graph_evolved = self.processors[i](graph_evolved)

        if self.final_pool=="global_max_pool":
            graph_latent = torch_geometric.nn.global_max_pool(graph_evolved.x, graph_evolved.batch)
        elif self.final_pool=="global_add_pool":
            graph_latent = torch_geometric.nn.global_add_pool(graph_evolved.x, graph_evolved.batch)
        elif self.final_pool=="global_mean_pool":
            graph_latent = torch_geometric.nn.global_mean_pool(graph_evolved.x, graph_evolved.batch.long())

        return graph_latent

    def forward(self, graph, reward_beta=None,**kwargs):
        """Given the data (graph) and a reward_beta, return a value estimating the cumulative expected reward.

        Args:
            data: graph
            reward_beta: float, nonnegative
        """
        if hasattr(graph, "node_feature"):
            if len(dict(to_tuple_shape(graph.original_shape))["n0"]) == 0:
                batch = graph.batch
                interp_index = graph.interp_index
                graph = attrdict_to_pygdict(graph, is_flatten=True, use_pos=self.use_pos)  
                graph.batch = batch
                if graph.x.shape[1]!=6:
                    if hasattr(graph, "onehot_list"):
                        onehot = graph.onehot_list[0]
                        assert onehot.shape[0]==graph.x.shape[0]
                        raw_kinems = self.compute_kinematics(graph, interp_index)
                        kinematics = graph.onehot_list[0][:,:1] * raw_kinems
                        try:
                            graph.x = torch.cat([graph.x, onehot, kinematics], dim=-1)
                        except:
                            pdb.set_trace()
            else:
                batch = graph.batch.clone()
                graph = deepsnap_to_pyg(graph, is_flatten=True, use_pos=self.use_pos)
        elif hasattr(graph, "x"):
            batch = graph.batch.clone()
            x_coords = graph.history[-1]
            interp_index = graph.interp_index
            if graph.x.shape[1]!=6:
                if hasattr(graph, "onehot_list"):
                    onehot = graph.onehot_list[0]
                    assert onehot.shape[0]==graph.x.shape[0]
                    raw_kinems = self.compute_kinematics(graph, interp_index)
                    kinematics = graph.onehot_list[0][:,:1] * raw_kinems
                    graph.x = torch.cat([graph.x, onehot, kinematics], dim=-1)
        
        if self.reward_condition:
            beta_batch = torch.zeros(graph.x.shape[0]).to(graph.x.device)
            beta_batch = reward_beta[batch.to(torch.int64)][:,None]
            graph.x = torch.cat([graph.x,beta_batch.repeat([1,5])],dim=-1).float()
        graph_evolved = self.encoder(graph)
        graph_evolved.batch = batch

        if self.pooling_type in ["global_max_pool","global_mean_pool","global_min_pool"]:
            graph_latent = self.global_pooling(graph_evolved,reward_beta=reward_beta)
        elif self.pooling_type in ["TopKPooling"]:
            graph_latent = self.topk_pooling(graph_evolved,reward_beta=reward_beta)

        reward_1, reward_2 = self.get_reward(graph_latent,reward_beta=reward_beta,batch=batch)
        value = reward_1 + reward_beta[:,None] * reward_2
        return value/10
    
    def compute_kinematics(self, data, index):
        # return 
        ybatch = ((data.reind_yfeatures[index][:,0] - data.yfeatures[index][:,0])/2).T
        # onehot_list = data.onehot_list[index+1][:,0] #for handles
        handles_index = torch.where(data.onehot_list[index+1][:,0]==1)[0]
        
        handles_batch_info = ybatch[handles_index]
        vers = data.yfeatures[index][handles_index][:,3:] #handle value at t+1 
        kinematic = torch.zeros_like(data.history[-1][:,3:]) 
        
        handles_index_tar = torch.where(data.onehot_list[0][:,0]==1)[0]
        kinematic[handles_index_tar] = (vers - data.history[-1][handles_index_tar][:,3:].clone().detach())
        # assert ((data.history[-1][:,:2][handles_index_tar]-data.yfeatures[index][handles_index][:,:2]).abs()).mean()<1e-12
        # try:
        #     assert ((ybatch[handles_index].to(torch.int64) == data.batch[handles_index_tar])*1-1).sum()==0
        # except:
        #     pdb.set_trace()
        return kinematic
    
    def pyg_to_dolfin_mesh(self, vers, faces):
        tmesh = self.dolfin.Mesh()
        editor = self.dolfin.MeshEditor()
        editor.open(tmesh, 'triangle', 2, 2)

        editor.init_vertices(vers.shape[0])
        for i in range(vers.shape[0]):
            editor.add_vertex(i, vers[i,:2].cpu().numpy())
        editor.init_cells(faces.shape[0])
        for f in range(faces.shape[0]):
            editor.add_cell(f, faces[f].cpu().numpy())

        editor.close()
        return tmesh
    
    def generate_barycentric_interpolated_data_forward(self, mesh, bvhtree, outvec, tarvers):
        faces, weights = self.generate_baryweight(tarvers, mesh, bvhtree)
        indices = mesh.cells()[faces].astype('int64')
        fweights = torch.tensor(np.array(weights), device=outvec.device, dtype=torch.float32)
        return torch.matmul(fweights, outvec[indices,:]).diagonal().T
        
    def generate_baryweight(self, tarvers, mesh, bvh_tree):
        faces = []
        weights = []
        for query in tarvers:
            face = bvh_tree.compute_first_entity_collision(self.dolfin.Point(query))
            while (mesh.num_cells() <= face):
                #print("query: ", query)
                if query[0] < 0.5:
                    query[0] += 1e-15
                elif query[0] >= 0.5:
                    query[0] -= 1e-15
                if query[1] < 0.5:
                    query[1] += 1e-15
                elif query[1] >= 0.5:
                    query[1] -= 1e-15            
                face = bvh_tree.compute_first_entity_collision(self.dolfin.Point(query))
            faces.append(face)
            face_coords = mesh.coordinates()[mesh.cells()[face]]
            mat = face_coords.T[:,[0,1]] - face_coords.T[:,[2,2]]
            const = query - face_coords[2,:]
            weight = np.linalg.solve(mat, const)
            final_weights = np.concatenate([weight, np.ones(1) - weight.sum()], axis=-1)
            weights.append(final_weights)
        return faces, weights

    @property
    def model_dict(self):
        model_dict = {"type": "Value_Model"}
        model_dict["input_size"] = self.input_size
        model_dict["output_size"] = self.output_size
        model_dict["edge_dim"] = self.edge_dim
        model_dict["latent_dim"] = self.latent_dim
        model_dict["num_pool"] = self.num_pool
        model_dict["act_name_final"] = self.act_name_final
        model_dict["pooling_type"] = self.pooling_type
        model_dict["batch_norm"] = self.batch_norm
        model_dict["layer_norm"] = self.layer_norm
        model_dict["num_steps"] = self.num_steps
        model_dict["act_name"] = self.act_name
        model_dict["edge_attr"] = self.edge_attr
        model_dict["use_pos"] = self.use_pos
        model_dict["final_ratio"] = self.final_ratio
        model_dict["final_pool"] = self.final_pool
        model_dict["reward_condition"] = self.reward_condition
        model_dict["state_dict"] = to_cpu(self.state_dict())
        return model_dict



# ## GNNRemesher:

# In[ ]:


class GNNRemesher(nn.Module):
    """ Perform remeshing for given mesh data.

    Attributes::
        nmax: maximun number to give hash for edge indices.
        device: gpu device.
        encoder_edge: encoder for edge features.
        encoder_nodes: encoder for node features.
        pdist: distance function used to detect obtuse triagnles.
        num_steps: message passing step.
        processors: message passing networks.
        diffMLP: use different message passing networks if True.
        decoder_node: decoder for node features.
        samplemode: strategy to sample independent edges.
        use_encoder: use encoder if True.
        is_split_test: perform split function for original mesh data.
        is_flip_test: perform flip function for original mesh data.
        is_coarsen_test: perform coarsen function for original mesh data.
        skip_split: skip split action if True.
        skip_flip: skip flip action if True.
        is_forward: perform forwarding by gnn before remeshing actions.
    """
    def __init__(
        self,
        input_size,
        output_size,
        edge_dim,
        sizing_field_dim,
        nmax,
        reward_dim=16,
        latent_dim=32,
        num_steps=3,
        layer_norm=False,
        act_name="relu",
        var=0,
        batch_norm=False,
        normalize = False,
        diffMLP=False,
        checkpoints=None,
        samplemode="random",
        use_encoder=False,
        edge_attr=False,
        is_split_test=False,
        is_flip_test=False,
        is_coarsen_test=False,
        skip_split=False,
        skip_flip=False,
        is_y_diff=False,
        edge_threshold=0.,
        noise_amp=0.,
        correction_rate=0.,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.edge_dim = edge_dim
        self.sizing_field_dim = sizing_field_dim
        if self.sizing_field_dim == 4:
            import dolfin as dolfin
            self.dolfin = dolfin
        self.reward_dim = reward_dim
        self.nmax = nmax.type(torch.int64) if isinstance(nmax, torch.Tensor) else torch.tensor(nmax, dtype=torch.int64)
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.layer_norm = layer_norm
        self.act_name = act_name
        self.var = var
        self.batch_norm = batch_norm
        self.normalize = normalize
        self.diffMLP = diffMLP
        self.checkpoints = checkpoints
        self.is_y_diff = is_y_diff
        self.edge_threshold = edge_threshold
        self.noise_amp = noise_amp
        self.correction_rate = correction_rate
        
        self.encoder_edge = FCBlock(in_features=edge_dim,
                                    out_features=latent_dim,
                                    num_hidden_layers=2,
                                    hidden_features=latent_dim,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=layer_norm,
                                   )
        
        if self.edge_threshold > 0.:
            self.world_encoder_edge = FCBlock(in_features=int(edge_dim/2),
                            out_features=latent_dim,
                            num_hidden_layers=2,
                            hidden_features=latent_dim,
                            outermost_linear=True,
                            nonlinearity=act_name,
                            layer_norm=layer_norm,
                           )
        
        self.encoder_nodes = FCBlock(in_features=input_size,
                                     out_features=latent_dim,
                                     num_hidden_layers=2,
                                     hidden_features=latent_dim,
                                     outermost_linear=True,
                                     nonlinearity=act_name,
                                     layer_norm=layer_norm,
                                    )

        self.pdist = torch.nn.PairwiseDistance(p=2)

        if diffMLP:
            # message passing with different MLP for each steps
            self.processors = []
            for _ in range(num_steps):
                self.processors.append((processor(latent_dim, latent_dim, layer_norm=layer_norm, act_name=act_name, edge_attr=edge_attr)))
                if batch_norm:
                    self.processors.append(torch.nn.BatchNorm1d(latent_dim))
                    self.processors.append(torch.nn.BatchNorm1d(latent_dim))
            self.processors = torch.nn.Sequential(*self.processors)
        else:
            self.processors = ((processor(latent_dim, latent_dim, layer_norm=layer_norm, act_name=act_name, edge_attr=edge_attr)))

        self.decoder_node = FCBlock(in_features=latent_dim,
                                    out_features=output_size+sizing_field_dim,
                                    num_hidden_layers=3,
                                    hidden_features=latent_dim,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=False,
                                   )
        if self.reward_dim > 0:
            self.reward_model = FCBlock(in_features=latent_dim+1,
                                        out_features=2,
                                        num_hidden_layers=3,
                                        hidden_features=reward_dim,
                                        outermost_linear=True,
                                        nonlinearity=act_name,
                                        layer_norm=False,
                                       )

        if self.normalize:
            if checkpoints==None:
                self.normalizer_node_feature = normalizer(input_size)
                self.normalizer_edge_feature = normalizer(edge_dim)
                self.normalizer_v_gt = normalizer(1)
            else:
                self.normalizer_node_feature = normalizer(input_size,max_acc=0)
                self.normalizer_edge_feature = normalizer(edge_dim,max_acc=0)
                self.normalizer_v_gt = normalizer(1,max_acc=0)             

        self.samplemode = samplemode
        self.use_encoder = use_encoder
        self.edge_attr = edge_attr
        self.is_split_test = is_split_test
        self.is_flip_test = is_flip_test
        self.is_coarsen_test = is_coarsen_test
        self.skip_split = skip_split
        self.skip_flip = skip_flip

    def generate_world_edge(self, graph):
        total_dim = graph.node_dim[0].sum().item()
        diagonal_mask = torch.zeros(total_dim, total_dim, device=graph.x.device)
        sum = 0
        for dim in graph.node_dim[0].reshape(graph.node_dim[0].shape[0]).tolist():
            diagonal_mask[sum:sum+dim, sum:sum+dim] = 1
            sum += dim
        
        distance = torch.sqrt(torch.sum((graph.x[None] - graph.x[:,None])**2, dim=-1))
        distance = distance * diagonal_mask
        mask = (0 < distance) * (distance < self.edge_threshold) 
        x = torch.arange(graph.x.shape[0])
        y = torch.arange(graph.x.shape[0])
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        candidate_edges = torch.stack([grid_x[mask], grid_y[mask]]).to(graph.x.device)
        if candidate_edges.shape[1] == 0:
            return candidate_edges, torch.empty([0, 2*graph.x.shape[1]], device=graph.x.device)
        can_hashes = edge_index_to_num(candidate_edges, self.nmax)
        edge_hashes = edge_index_to_num(graph.edge_index, self.nmax)
        world_mask = torch.logical_not(torch.isin(can_hashes, edge_hashes))
        world_edge_index = candidate_edges[:, world_mask]
        
        x_receiver = torch.gather(graph.x, 0, world_edge_index[0,:].unsqueeze(-1).repeat(1,graph.x.shape[1]))
        x_sender = torch.gather(graph.x, 0, world_edge_index[1,:].unsqueeze(-1).repeat(1,graph.x.shape[1]))
        rel_vec = x_receiver - x_sender
        new_edge_attr = torch.cat([rel_vec, rel_vec.abs()], dim=-1)        
        return world_edge_index, new_edge_attr

    def encoder(self, graph):
        """ Encode node and edge features.

        Args:
            graph: pyg data instance for graph.
        """
        if self.var > 0:
            noise = (torch.normal(0,1,size=(graph.x.shape[0],graph.x.shape[1]-1))*self.var).to(graph.x.device)
            graph.x[:,:-1] = graph.x[:,:-1] + graph.x[:,[-1]]*noise
            graph.noise = graph.x[:,[-1]]*noise
        else:
            graph.noise = torch.zeros_like(graph.x)

        if self.edge_attr:
            graph.edge_attr = self.encoder_edge(graph.edge_attr)
            if self.edge_threshold > 0:
                graph.orig_edge_shape = graph.edge_index.shape
                world_edge_index, new_edge_attr = self.generate_world_edge(graph)
                graph.edge_index = torch.cat([graph.edge_index, world_edge_index], dim=-1)
                graph.edge_attr = torch.cat([graph.edge_attr, self.world_encoder_edge(new_edge_attr)], dim=0)         
        graph.x = self.encoder_nodes(graph.x)
        return graph

    def decoder(self, graph):
        """ Decode node features.

        Args:
            graph: pyg data instance for graph.       
        """
        graph.x = self.decoder_node(graph.x)
        if self.edge_threshold > 0:
            graph.edge_index = graph.edge_index[:, :graph.orig_edge_shape[1]]
        return graph

    def forward_one_step(
        self,
        graph,
        train=False,
        use_pos=False,
        **kwargs
    ):
        """ Forward graph for 1 step into the future.

        Forward graph using encoder, processor, and decoder.

        Args:
            graph: pyg data instance for graph.
            train: if it is training mode

        Returns:
            graph_evolved: graph evolved.
        """
        # Reformat the data:
        if hasattr(graph, "node_feature"):
            if len(dict(graph.original_shape)["n0"]) == 0:
                graph = attrdict_to_pygdict(graph, is_flatten=True, use_pos=use_pos)
            else:
                graph = deepsnap_to_pyg(graph, is_flatten=True, use_pos=use_pos)

        # normalize the dataset
        if self.normalize:
            graph.x = self.normalizer_node_feature.update(graph.x,train)
            graph.edge_attr = self.normalizer_edge_feature.update(graph.edge_attr,train)
            graph.v_gt = self.normalizer_v_gt.update(graph.v_gt,train)
        
        # if is_y_diff, then store the initial u and add residual connect to the end
        if self.is_y_diff:
            feature_init = graph.x.clone()

        #encode edges and nodes to latent dim
        if self.use_encoder:
            graph_evolved = self.encoder(graph)
        else:
            # graph_evolved = graph.clone()
            graph_evolved = graph
        if self.diffMLP:
            # message passing steps with different MLP each time
            for i in range(self.num_steps):
                if self.batch_norm:
                    graph_evolved = self.processors[i*3](graph_evolved)
                    graph_evolved.x = self.processors[i*3+1](graph_evolved.x)
                    graph_evolved.edge_attr = self.processors[i*3+2](graph_evolved.edge_attr)
                else:
                    graph_evolved = self.processors[i](graph_evolved)
        else:
            # message passing steps with same MLP each time
            for _ in range(self.num_steps):
                graph_evolved = self.processors(graph_evolved)

        pred = {}
        # Reward:
        if self.reward_dim > 0:
            reward_beta = torch.tensor(kwargs["reward_beta"] if "reward_beta" in kwargs else 1, device=graph.x.device, dtype=graph.x.dtype)[None,None].expand(graph.x.shape[0],1)
            reward_raw = self.reward_model(torch.cat([graph_evolved.x, reward_beta], -1))  # [n_nodes, 2]
            pred["minus_reward"] = reward_raw[:,0].mean() + reward_raw[:,1].sum() / 10000
        # Decoding:
        if self.use_encoder:
            graph_evolved = self.decoder(graph_evolved)
        graph_evolved.x = graph_evolved.x/10 #div 10 for 46, div 100 for 31
        # if is_y_diff then add residual connection here as euler integration
        if self.is_y_diff:
            graph_evolved.x = graph_evolved.x+torch.cat([feature_init[:,:graph_evolved.x.shape[-1]], torch.zeros(feature_init.shape[0], self.sizing_field_dim, device=feature_init.device)], dim=-1)

        if self.normalize and not train:
            graph_evolved.x = self.normalizer_v_gt.reverse(graph_evolved.x)

        pred["n0"] = graph_evolved.x
        info = {}
        return pred, info

    def forward(
        self,
        data,
        pred_steps=1,
        train=False,
        use_pos=False,
        is_deepsnap=None,
        returns_data=False,
        **kwargs
    ):
        """
        Evolve the system a few time steps into the future.

        Args:
            data: pyg graph object
            pred_steps: an integer, or a list of integers indicating which steps we want the prediction.

        Returns:
            preds: a dict {"n0": tensor} where tensor has shape [n_nodes, n_steps, feature_size]
            info: additional information
        """
        # Reformat the data:
        # pdb.set_trace()
        if returns_data:
            with torch.no_grad():
                data_clone = copy_data(data)

        if hasattr(data, "node_feature"):
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0 and is_deepsnap in [False, None]:         
                data = attrdict_to_pygdict(data, is_flatten=True, use_pos=use_pos)
            else:
                data = deepsnap_to_pyg(data, is_flatten=True, use_pos=use_pos)

        if not isinstance(pred_steps, list) and not isinstance(pred_steps, np.ndarray):
            pred_steps = [pred_steps]
        max_pred_step = max(pred_steps + [0])
        pred = {"n0": []}
        if self.reward_dim > 0:
            pred["minus_reward"] = []
        if hasattr(data, "edge_attr"):
            edge_attr_clone = deepcopy(data.edge_attr)
        for k in range(1, max_pred_step+1):
            pred_k, _ = self.forward_one_step(
                data,
                train=train,
                is_last_layer=(k==max_pred_step),
                **kwargs
            )
            x_core = data.x[..., :self.output_size]
            lst = [x_core, data.x_bdd]
            if use_pos:
                lst.insert(1, data.x_pos)
            if hasattr(data, "dataset") and to_tuple_shape(data.dataset).startswith("mppde1dh"):
                lst.insert(1, data.param[:1].expand(data.x.shape[0], data.param.shape[-1]))
            data.x = torch.cat(lst, -1)
            if hasattr(data, "edge_attr"):
                data.edge_attr = edge_attr_clone
            pred["n0"].append(pred_k["n0"])
            if self.reward_dim > 0:
                pred["minus_reward"].append(pred_k["minus_reward"])
        pred["n0"] = torch.stack(pred["n0"], -2)
        if self.reward_dim > 0:
            pred["minus_reward"] = torch.stack(pred["minus_reward"])
        info = {}
        if returns_data:
            # Get the data for the next time step
            data_new = get_data_pred(pred["n0"], step=0, data=data_clone)
            if "minus_reward" in pred:
                info["minus_reward"] = pred["minus_reward"]
            return data_new, info
        else:
            return pred, info


    def get_loss(self, data, args, **kwargs):
        multi_step_dict = parse_multi_step(args.multi_step)
        # Reformat the data:
        if hasattr(data, "node_feature"):
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                data = attrdict_to_pygdict(data, is_flatten=True, use_pos=args.use_pos)
            else:
                data = deepsnap_to_pyg(data, is_flatten=True, use_pos=args.use_pos)
        if args.data_dropout != "None" and self.training:
            if args.dataset.startswith("mppde1d"):
                if args.exclude_bdd:
                    nx = dict(to_tuple_shape(data.original_shape))["n0"][0]
                    sample_idx = np.arange(1, nx-1)
                else:
                    sample_idx = None
                data = get_data_dropout(data, dropout_mode=args.data_dropout, sample_idx=sample_idx)

        self.info = {}
        if self.reward_dim > 0:
            reward_beta = sample_reward_beta(args.reward_beta)
            kwargs["reward_beta"] = reward_beta
        if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
            t_start = time.time()
            pred, _ = self.interpolation_hist_forward(data, pred_steps=list(multi_step_dict.keys()), use_pos=args.use_pos, **kwargs)
            t_end = time.time()
            ylist = data.y_back
        else:  
            t_start = time.time()
            pred, _ = self(data, pred_steps=list(multi_step_dict.keys()), use_pos=args.use_pos, **kwargs)
            t_end = time.time()
            y = data.y.reshape(data.y.shape[0], -1, self.output_size)

        loss = 0
        # pdb.set_trace()
        if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
            kwargs = {"first_dim": 1}
            for pred_idx, k in enumerate(multi_step_dict):
                loss_k = loss_op_core(
                    pred["n0"][pred_idx][:,:self.output_size].flatten(start_dim=1),
                    ylist[k-1].to(data.x.device),
                    loss_type=args.loss_type,
                    normalize_mode=args.latent_loss_normalize_mode,
                    **kwargs,
                ) * multi_step_dict[k]
                loss = loss + loss_k
                self.info[f"loss_{k}"] = loss_k.item()
        else:
            for pred_idx, k in enumerate(multi_step_dict):
                loss_k = loss_op_core(
                    pred["n0"][...,pred_idx,:self.output_size].flatten(start_dim=1),
                    y[...,k-1,:],
                    loss_type=args.loss_type,
                    first_dim=1,
                    normalize_mode=args.latent_loss_normalize_mode,
                ) * multi_step_dict[k]
                loss = loss + loss_k
                self.info[f"loss_{k}"] = loss_k.item()
        if self.reward_dim > 0:
            rl_coefs_dict = parse_rl_coefs(args.rl_coefs)
            minus_reward = get_minus_reward(
                loss=loss,
                state=pred["n0"][0] if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0 else pred["n0"][...,0,:self.output_size],
                time=t_end - t_start,
                reward_mode=args.reward_mode,
                reward_beta=reward_beta,
            )
            kwargs = {"first_dim": 1}
            loss_reward = loss_op_core(
                pred["minus_reward"].sum(),
                torch.tensor(minus_reward, device=loss.device, dtype=torch.float32),
                loss_type=args.loss_type,
                **kwargs,
            ) * rl_coefs_dict["reward"]
            loss = loss + loss_reward
            self.info["loss_reward"] = loss_reward.item()
        return loss


    @property
    def model_dict(self):
        model_dict = {"type": "GNNRemesher"}
        model_dict["input_size"] = self.input_size
        model_dict["output_size"] = self.output_size
        model_dict["edge_dim"] = self.edge_dim
        model_dict["sizing_field_dim"] = self.sizing_field_dim
        model_dict["reward_dim"] = self.reward_dim
        model_dict["nmax"] = self.nmax
        model_dict["latent_dim"] = self.latent_dim
        model_dict["num_steps"] = self.num_steps
        model_dict["layer_norm"] = self.layer_norm
        model_dict["act_name"] = self.act_name
        model_dict["var"] = self.var
        model_dict["batch_norm"] = self.batch_norm
        model_dict["normalize"] = self.normalize
        model_dict["diffMLP"] = self.diffMLP
        model_dict["checkpoints"] = self.checkpoints
        model_dict["samplemode"] = self.samplemode
        model_dict["use_encoder"] = self.use_encoder
        model_dict["edge_attr"] = self.edge_attr
        model_dict["is_split_test"] = self.is_split_test
        model_dict["is_flip_test"] = self.is_flip_test
        model_dict["is_coarsen_test"] = self.is_coarsen_test
        model_dict["skip_split"] = self.skip_split
        model_dict["skip_flip"] = self.skip_flip
        model_dict["is_y_diff"] = self.is_y_diff
        model_dict["state_dict"] = to_cpu(self.state_dict())
        model_dict["edge_threshold"] = self.edge_threshold
        model_dict["noise_amp"] = self.noise_amp
        model_dict["correction_rate"] = self.correction_rate
        return model_dict


class GNNRemesherPolicy(GNNRemesher):
    """ Perform remeshing for given mesh data.

    Attributes::
        See the attributes in the GNNRemesher.
    """
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    def one_step_interpolation_forward(
        self,
        data,
        interp_index,
        pred_steps=1,
        train=False,
        use_pos=False,
        is_deepsnap=None,
        returns_data=False,
        is_timing=0,
        use_remeshing=False,
        changing_mesh=False,
        opt_evl=False,
        mode=None,
        debug=False,
        data_gt=None,
        noise_amp_val=0,
        **kwargs
    ):
        """
        Evolve the system one time step into the future.

        Args:
            data: pyg graph object
            pred_steps: an integer, or a list of integers indicating which steps we want the prediction.

        Returns:
            preds: a dict {"n0": tensor} where tensor has shape [n_nodes, n_steps, feature_size]
            info: additional information
        """
        # Reformat the data:
        p.print("0.0000", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        if hasattr(data, "node_feature"):
            batch = data.batch["n0"]
            data = attrdict_to_pygdict(data, is_flatten=True, use_pos=use_pos)
            data.batch = batch
        if not isinstance(pred_steps, list) and not isinstance(pred_steps, np.ndarray):
            pred_steps = [pred_steps]
        max_pred_step = max(pred_steps + [0])
        # node_dims = tuple([nd.clone() for nd in list(data.node_dim)])
        pred = {"n0": []}
        if self.reward_dim > 0:
            pred["minus_reward"] = []
        # pdb.set_trace()
        info = {}
        
        k=1
        if debug:
            # load the ground truth fine mesh 
            ybatch = ((data_gt.reind_yfeatures["n0"][interp_index][:,0] - data_gt.yfeatures["n0"][interp_index][:,0])/2).T
            vers = (data.history[-1] + 2*data.batch.repeat((data.history[-1].shape[1], 1)).T)
            faces = data.xfaces.T
            tarvers = data_gt.reind_yfeatures["n0"][interp_index][:,:2].clone().detach().cpu().numpy()
            interpolate_out = generate_barycentric_interpolated_data(vers.clone().detach(), faces, vers, tarvers)
            interpolate_out[:,:2] =  torch.tensor(tarvers,device=interpolate_out.device)  
            gt = (interpolate_out - 2*ybatch.repeat((interpolate_out.shape[-1], 1)).T).to(data.x.device)
            data.history = (data.history[0],gt)
            data.batch = None
            data.batch = torch.tensor(ybatch,device=ybatch.device,dtype=data.batch_history.dtype)
            data.xfaces = (torch.tensor(data_gt.yface_list["n0"][interp_index]).T).to(data.x.device)
            data.xface_list = (data.xface_list[0],data.xfaces)   
            data.x_coords = gt.cuda()
            data.x_pos = data.x_coords[:,:2]
            temp_onehots = list(data.onehot_list)
            temp_onehots[0] = temp_onehots[interp_index+1]
            data.onehot_list = tuple(temp_onehots)
            data.edge_index = data_gt.yedge_index["n0"][interp_index].to(data.edge_index.device)

        if True:
        # if use_remeshing or changing_mesh or mode=="test_remeshing_2":
            p.print("0.0001", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            shift_history = [data.history[0]+2*data.batch_history.repeat((data.history[0].shape[1], 1)).T, data.history[1]+2*data.batch.repeat((data.history[1].shape[1], 1)).T]
            tarvers = shift_history[-1][:,:2].clone().detach()
            vers = shift_history[0][:,:2].clone().detach()
            
            if not isinstance(data.xface_list[0], torch.Tensor):
                faces = torch.tensor(data.xface_list[0], device=vers.device)
            else:
                faces = data.xface_list[0]
            p.print("0.0002", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            mesh = self.pyg_to_dolfin_mesh(vers, faces)
            p.print("0.0003", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            bvhtree = mesh.bounding_box_tree()  
            p.print("0.0004", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            faces, weights = self.generate_baryweight(tarvers.cpu().numpy(), mesh, bvhtree)
            p.print("0.0005", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            tindices = mesh.cells()[faces].astype('int64')
            tfweights = torch.tensor(np.array(weights), device=tarvers.device, dtype=torch.float32)
            hist_wfeature = torch.matmul(tfweights, data.history[0][tindices,:]).diagonal().T[:,self.output_size:]
            # ((vers-tarvers).abs()>1e-7).sum()
            # ((data.history[0][:,3:]-hist_wfeature).abs()>1e-7).sum()
            # ((data.history[0]-data.history[1]).abs()>1e-8).sum()
            past_hist_wfeature = hist_wfeature.clone()
            shift_history = [shift_history[0]+2*data.batch_history.repeat((data.history[0].shape[1], 1)).T, shift_history[1]-2*data.batch.repeat((data.history[1].shape[1], 1)).T]
        # else:
        #     hist_wfeature = data.history[0][:,3:]
        #     past_hist_wfeature = hist_wfeature.clone()
        p.print("0.0006", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        # Update edge_attributes
        edge_index = data.edge_index
        x_receiver = torch.gather(data.history[-1], 0, edge_index[0,:].unsqueeze(-1).repeat(1,data.history[-1].shape[1]))
        x_sender = torch.gather(data.history[-1], 0, edge_index[1,:].unsqueeze(-1).repeat(1,data.history[-1].shape[1]))
        p.print("0.00061", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        relative_pos = x_receiver - x_sender
        data.edge_attr = torch.cat([relative_pos, relative_pos.abs()], dim=-1)
        # Compute x feature for nodes by taking subtraction
        # pdb.set_trace()
        if noise_amp_val > 0 and opt_evl: # and interp_index==0:
            # pdb.set_trace()
            noise = torch.normal(0, noise_amp_val, size=data.history[-1][:,3:].shape, device=data.history[-1][:,3:].device)
            velocity = data.history[-1][:,3:] + noise - hist_wfeature
        else:
            velocity = data.history[-1][:,3:] - hist_wfeature
            
        data.x = velocity
        p.print("0.00062", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        # pdb.set_trace()
        if hasattr(data, "onehot_list"):
            onehot = data.onehot_list[0]
            raw_kinems = self.compute_kinematics(data, interp_index)
            p.print("0.00063", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            kinematics = data.onehot_list[0][:,:1] * raw_kinems
            data.x = torch.cat([data.x, onehot, kinematics], dim=-1)

        # data.node_dim = (node_dims[0],)
        p.print("0.0007", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        pred_k, _ = self.forward_one_step(
            data,
            train=train,
            is_last_layer=True,
            **kwargs
        )
        p.print("0.0008", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        if noise_amp_val > 0 and opt_evl: # and interp_index==0:
            data.x[:,:self.output_size] = data.x[:,:self.output_size] + 0.9 * noise
        data.x = data.x[:, :self.output_size]

        outvec = data.x + 2*data.history[-1][:,3:] - past_hist_wfeature
        next_feature = torch.cat([data.history[-1][:,:3].clone(), outvec], dim=-1)
        # if noise_amp_val > 0 and opt_evl:
        # if True:
        data.history = (data.history[-1], next_feature)
        # if use_remeshing:
        #     data.history = (data.history[0], next_feature)
        # else:
        #     data.history = (data.history[-1], next_feature)
        
        # if changing_mesh or mode=="test_remeshing_2":
        data.batch_history = data.batch
        data.xface_list = (data.xfaces.T, data.xfaces.T)
            
        edge_index = data.edge_index
        x_receiver = torch.gather(data.history[-1], 0, edge_index[0,:].unsqueeze(-1).repeat(1,data.history[-1].shape[1]))
        x_sender = torch.gather(data.history[-1], 0, edge_index[1,:].unsqueeze(-1).repeat(1,data.history[-1].shape[1]))
        relative_pos = x_receiver - x_sender
        data.edge_attr = torch.cat([relative_pos, relative_pos.abs()], dim=-1)
        
        p.print("0.0009", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        info["final_gnnout"] = outvec.clone()
        if self.reward_dim > 0:
            pred["minus_reward"].append(pred_k["minus_reward"])
        pred["n0"].append(data.x)
        if self.reward_dim > 0:
            pred["minus_reward"] = torch.stack(pred["minus_reward"])
        velocity = data.history[-1][:,3:] - data.history[0][:,3:]
        data.x = velocity
        return pred, info, data
    
    def generate_barycentric_interpolated_data_forward(self, mesh, bvhtree, outvec, tarvers):
        faces, weights = self.generate_baryweight(tarvers, mesh, bvhtree)
        indices = mesh.cells()[faces].astype('int64')
        fweights = torch.tensor(np.array(weights), device=outvec.device, dtype=torch.float32)
        return torch.matmul(fweights, outvec[indices,:]).diagonal().T
        
    def compute_kinematics(self, data, index):
        # return 
        ybatch = ((data.reind_yfeatures[index][:,0] - data.yfeatures[index][:,0])/2).T
        # onehot_list = data.onehot_list[index+1][:,0] #for handles
        handles_index = torch.where(data.onehot_list[index+1][:,0]==1)[0]

        handles_batch_info = ybatch[handles_index]
        vers = data.yfeatures[index][handles_index][:,3:] #handle value at t+1 
        kinematic = torch.zeros_like(data.history[-1][:,3:]) 
        handles_index_tar = torch.where(data.onehot_list[0][:,0]==1)[0]
        kinematic[handles_index_tar] = (vers - data.history[-1][handles_index_tar][:,3:].clone().detach())
    
        # assert ((data.history[-1][:,:2][handles_index_tar]-data.yfeatures[index][handles_index][:,:2]).abs()).mean()<1e-12
        # assert ((ybatch[handles_index].to(torch.int64) == data.batch[handles_index_tar])*1-1).sum()==0
        return kinematic

    def interpolation_hist_forward(
        self,
        data,
        pred_steps=1,
        train=False,
        use_pos=False,
        is_non_remeshing=False,
        **kwargs
    ):
        """
        Evolve the system a few time steps into the future.

        Args:
            data: pyg graph object
            pred_steps: an integer, or a list of integers indicating which steps we want the prediction.

        Returns:
            preds: a dict {"n0": tensor} where tensor has shape [n_nodes, n_steps, feature_size]
            info: additional information
        """
        # Reformat the data:
        if hasattr(data, "node_feature"):
            data = attrdict_to_pygdict(data, is_flatten=True, use_pos=use_pos)        
        if not isinstance(pred_steps, list) and not isinstance(pred_steps, np.ndarray):
            pred_steps = [pred_steps]
        max_pred_step = max(pred_steps + [0])
        node_dims = tuple([nd.clone() for nd in list(data.node_dim)])
        pred = {"n0": []}
        if self.reward_dim > 0:
            pred["minus_reward"] = []
        # pdb.set_trace()
        info = {"data": [], "data_x": [], "data_face": [], "y_tar": []}
        for k in range(1, max_pred_step+1):
            if is_non_remeshing:
                if k == 1:
                    tindices = np.array(data.hist_indices[k-1]).astype('int64')
                    tfweights = data.hist_weights[k-1].to(dtype=torch.float32)
                    hist_wfeature = torch.matmul(tfweights, data.history[0][tindices,:]).diagonal().T[:,self.output_size:]
                else:
                    hist_wfeature = data.history[0][:,self.output_size:]
            else:
                tindices = np.array(data.hist_indices[k-1]).astype('int64')
                tfweights = data.hist_weights[k-1].to(dtype=torch.float32)
                hist_wfeature = torch.matmul(tfweights, data.history[0][tindices,:]).diagonal().T[:,self.output_size:]
            # add noise
            # pdb.set_trace()
            if self.noise_amp > 0:
                noise = torch.normal(0, self.noise_amp, size=data.history[-1][:,3:].shape, device=data.history[-1][:,3:].device)
                data.history[-1][:,3:] = data.history[-1][:,3:] + noise
            past_hist_wfeature = hist_wfeature.clone()
            # Compute x feature for nodes by taking subtraction
            velocity = data.history[-1][:,3:] - hist_wfeature
            data = data.copy()
            data.x = velocity
            if hasattr(data, "onehot_list"):
                if is_non_remeshing:
                    onehot = data.onehot_list[0]
                    raw_kinems = self.compute_kinematics(data, k-1)
                    kinematics = data.onehot_list[0][:,:1] * raw_kinems
                    data.x = torch.cat([data.x, onehot, kinematics], dim=-1)
                else:
                    onehot = data.onehot_list[k-1]
                    hand_vel = data.kinematics_list[k-1]
                    data.x = torch.cat([data.x, onehot, hand_vel], dim=-1)
            data.node_dim = (node_dims[k-1],)
            pred_k, _ = self.forward_one_step(
                data,
                train=train,
                is_last_layer=(k==max_pred_step),
                **kwargs
            )
            # Correct acceleration, that is pred_k
            # pdb.set_trace()
            if not self.correction_rate==0.:
                data.x[:,:self.output_size] = data.x[:,:self.output_size] + self.correction_rate * noise
            # check if change in data.x makes change in pred_k["n0"] 
            ## Take subtraction after the change in data.x
            
            # Update history vertices (2d coords and 3d coords) and faces 
            outvec = data.x[:,:3] + 2*data.history[-1][:,3:] - past_hist_wfeature
            info["data_x"].append(torch.cat([data.history[-1][:,:3], outvec.clone()], dim=-1))
            info["data_face"].append(data.xfaces.clone())

            if is_non_remeshing:
                next_feature = torch.cat([data.history[-1][:, :3], outvec], dim=-1)
            else:
                # outvec = data.x
                fweights = data.bary_weights[k-1]
                indices = data.bary_indices[k-1]                        

                data.xfaces = torch.tensor(data.yface_list[k-1], device=data.xfaces.device).T
                data.edge_index = data.yedge_index[k-1].to(data.edge_index.device)
                data.x_bdd = torch.empty(data.x.shape[0],0, device=data.edge_index.device)

                world_coords = torch.matmul(fweights, outvec[np.array(indices),:]).diagonal().T 
                next_feature = torch.cat([data.yfeatures[k-1][:,:3], world_coords], dim=-1)
            
            if not self.correction_rate==0.:
                data.history
            data.history = (data.history[-1], next_feature)
            # data.xface_list = (data.xface_list[-1], data.yface_list[k-1])            
            if k == max_pred_step:
                info["final_gnnout"] = outvec.clone()
            if self.reward_dim > 0:
                pred["minus_reward"].append(pred_k["minus_reward"])
            pred["n0"].append(data.x[:,:self.output_size])
            
            # Renew edge_attributes using M_{t}
            edge_index = data.edge_index
            x_receiver = torch.gather(next_feature, 0, edge_index[0,:].unsqueeze(-1).repeat(1,next_feature.shape[1]))
            x_sender = torch.gather(next_feature, 0, edge_index[1,:].unsqueeze(-1).repeat(1,next_feature.shape[1]))
            relative_pos = x_receiver - x_sender
            data.edge_attr = torch.cat([relative_pos, relative_pos.abs()], dim=-1)
        if self.reward_dim > 0:
            pred["minus_reward"] = torch.stack(pred["minus_reward"])
        info["data"].append(data)
        return pred, info

    
    def interpolation_forward(
        self,
        data,
        pred_steps=1,
        train=False,
        use_pos=False,
        **kwargs
    ):
        """
        Evolve the system a few time steps into the future.

        Args:
            data: pyg graph object
            pred_steps: an integer, or a list of integers indicating which steps we want the prediction.

        Returns:
            preds: a dict {"n0": tensor} where tensor has shape [n_nodes, n_steps, feature_size]
            info: additional information
        """
        # Reformat the data:
        if hasattr(data, "node_feature"):
            data = attrdict_to_pygdict(data, is_flatten=True, use_pos=use_pos)
        
        if not isinstance(pred_steps, list) and not isinstance(pred_steps, np.ndarray):
            pred_steps = [pred_steps]
        max_pred_step = max(pred_steps + [0])
        pred = {"n0": []}
        if self.reward_dim > 0:
            pred["minus_reward"] = []
        if hasattr(data, "edge_attr"):
            edge_attr_clone = data.edge_attr
        #pdb.set_trace()
        info = {"data": [], "data_x": [], "data_face": []}
        for k in range(1, max_pred_step+1):
            pred_k, _ = self.forward_one_step(
                data,
                train=train,
                is_last_layer=(k==max_pred_step),
                **kwargs
            )
            # pdb.set_trace()
            x_core = data.x[..., :self.output_size]
            lst = [x_core, data.x_bdd]
            if use_pos:
                lst.insert(1, data.x_pos)
            data.x = torch.cat(lst, -1)
            if hasattr(data, "edge_attr"):
                data.edge_attr = edge_attr_clone
            if self.reward_dim > 0:
                pred["minus_reward"].append(pred_k["minus_reward"])
            pred["n0"].append(pred_k["n0"])
            
            pred_list = []
            outvec = data.x
            fweights = data.bary_weights[k-1]
            indices = data.bary_indices[k-1]
            # pdb.set_trace()
            if k == max_pred_step:
                info["final_gnnout"] = outvec.clone()
            data.x = torch.matmul(fweights, outvec[np.array(indices),:]).diagonal().T[:,:self.output_size].reshape(-1,1,self.output_size)[:,0,:]
            # data.x = self.generate_barycentric_interpolated_data(mesh, bvhtree, outvec, tarvers.cpu().numpy())[:,0,:]
            data.xfaces = torch.tensor(data.yface_list[k-1], device=data.xfaces.device).T
            data.edge_index = data.yedge_index[k-1].to(data.edge_index.device)
            data.x_bdd = torch.empty(data.x.shape[0],0, device=data.edge_index.device)
            if k%2 == 0:
                info["data"].append(data)
                info["data_x"].append(data.x.clone())
                info["data_face"].append(data.xfaces.clone())
                

        if self.reward_dim > 0:
            pred["minus_reward"] = torch.stack(pred["minus_reward"])
            
        return pred, info

    def pyg_to_dolfin_mesh(self, vers, faces):
        tmesh = self.dolfin.Mesh()
        editor = self.dolfin.MeshEditor()
        editor.open(tmesh, 'triangle', 2, 2)

        editor.init_vertices(vers.shape[0])
        for i in range(vers.shape[0]):
            editor.add_vertex(i, vers[i,:2].cpu().numpy())
        editor.init_cells(faces.shape[0])
        for f in range(faces.shape[0]):
            editor.add_cell(f, faces[f].cpu().numpy())

        editor.close()
        return tmesh

    def generate_baryweight(self, tarvers, mesh, bvh_tree):
        faces = []
        weights = []
        for query in tarvers:
            face = bvh_tree.compute_first_entity_collision(self.dolfin.Point(query))
            while (mesh.num_cells() <= face):
                #print("query: ", query)
                # if query[0] < 0.5:
                #     query[0] += 1e-15
                # elif query[0] >= 0.5:
                #     query[0] -= 1e-15
                # if query[1] < 0.5:
                #     query[1] += 1e-15
                # elif query[1] >= 0.5:
                #     query[1] -= 1e-15  
                if ((math.trunc(query[0].item())%2)!=0):
                    if (query[0].item() - math.trunc(query[0].item()) < 0.5):
                        query[0] = math.trunc(query[0].item())
                    elif (query[0].item() - math.trunc(query[0].item()) > 0.5):
                        query[0] = math.trunc(query[0].item())+1
                if ((math.trunc(query[1].item())%2)!=0):
                    if (query[1].item() - math.trunc(query[1].item()) < 0.5):
                        query[1] = math.trunc(query[1].item())
                    elif (query[1].item() - math.trunc(query[1].item()) > 0.5):
                        query[1] = math.trunc(query[1].item())+1
                if (query[0].item() - math.trunc(query[0].item())) < 0.5:
                    query[0] += 1e-15
                elif (query[0].item() - math.trunc(query[0].item())) >= 0.5:
                    query[0] -= 1e-15
                if (query[1].item() - math.trunc(query[1].item())) < 0.5:
                    query[1] += 1e-15
                elif (query[1].item() - math.trunc(query[1].item())) >= 0.5:
                    query[1] -= 1e-15           
                face = bvh_tree.compute_first_entity_collision(self.dolfin.Point(query))
            faces.append(face)
            face_coords = mesh.coordinates()[mesh.cells()[face]]
            mat = face_coords.T[:,[0,1]] - face_coords.T[:,[2,2]]
            const = query - face_coords[2,:]
            weight = np.linalg.solve(mat, const)
            final_weights = np.concatenate([weight, np.ones(1) - weight.sum()], axis=-1)
            weights.append(final_weights)
        return faces, weights

    def generate_barycentric_interpolated_data(self, mesh, bvhtree, outvec, tarvers):
        faces, weights = self.generate_baryweight(tarvers, mesh, bvhtree)
        indices = mesh.cells()[faces].astype('int64')
        fweights = torch.tensor(np.array(weights), device=outvec.device, dtype=torch.float32)
        return torch.matmul(fweights, outvec[indices,:]).diagonal().T[:,:self.output_size].reshape(-1,1,self.output_size)


    def remeshing_forward(self, torchmesh, train=False, use_pos=False):
        """ Perform remeshing action on mesh data.
                
        Args:
            torchmesh: pyg data instance for mesh.
            train: 
            
        Returns:
            outmesh: mesh data after gnn forwarding and remeshing.
        """
        # Reformat the data:
        if hasattr(torchmesh, "node_feature"):
            if len(dict(torchmesh.original_shape)["n0"]) == 0:
                torchmesh = attrdict_to_pygdict(torchmesh, is_flatten=True, use_pos=use_pos)
            else:
                torchmesh = deepsnap_to_pyg(torchmesh, is_flatten=True, use_pos=use_pos)

        if torchmesh.xfaces.shape[0] == 0:
            self.skip_flip = True

        # Split edges of mesh
        if self.skip_split:
            split_outmesh = torchmesh
        else:    
            if self.is_split_test:
                split_edges = self.choose_split_edge(torchmesh)
                if split_edges.shape[0] == 0:
                    outmesh = torchmesh
                else:
                    outmesh = self.split(split_edges, torchmesh)
                return outmesh
            else:
                split_edges = self.choose_split_edge(torchmesh)
                split_outmesh = self.split(split_edges, torchmesh)

        # Flip edges of mesh
        if self.skip_flip:
            flipped_outmesh = split_outmesh
        else:
            if self.is_flip_test:
                split_outmesh = torchmesh
                flip_edges = self.choose_flippable_edge(split_outmesh)
                outmesh = self.flip(flip_edges, split_outmesh)
                return outmesh
            else:
                flip_edges = self.choose_flippable_edge(split_outmesh)
                flipped_outmesh = self.flip(flip_edges, split_outmesh)
        
        # Coarsen edges of mesh
        if self.is_coarsen_test:
            flipped_outmesh = torchmesh
            outmesh = self.coarsen(flipped_outmesh)
            if use_pos and (outmesh.xfaces.shape[0] == 0):
                outmesh.x = outmesh.x[:, :self.input_size+1]
            else:
                outmesh.x = outmesh.x[:, :self.input_size]
            return outmesh

        else:
            outmesh = self.coarsen(flipped_outmesh)
            if use_pos and (outmesh.xfaces.shape[0] == 0):
                outmesh.x = outmesh.x[:, :self.input_size+1]
            else:
                outmesh.x = outmesh.x[:, :self.input_size]
            return outmesh


    def _create_degree_tensor(self, torchmesh):
        """ Compute node degree. 
        
        Args:
            torchmesh: pyg data instance for mesh.
            
        Returns:
            deg_tensor: tensors of node degrees of shape [number of nodes].
        """        
        ones = torch.ones(torchmesh.edge_index[0].shape, device=torchmesh.x.device)
        deg_tensor = scatter(ones, torchmesh.edge_index[0, :], reduce="sum")
        return deg_tensor

    def _create_dual_graph(self, torchmesh):
        """ Created dual graph and generate associated information for splitting. 

        Create dual graph whose nodes are edges in original mesh and
        edges are edges in original mesh belonging to a same face.

        Args:
            torchmesh: pyg data instance for mesh.

        Returns:
            dnodes: set of nodes in dual graph. A node is represented 
            by a pair of nodes in original mesh.
            dedge_neighbor_dict: dictionary whose key is nodes in dual
            graph and values are set of nodes in dual graph adjacent to
            the node of the key.
        """        
        dnodes = set()
        dedge_neighbor_dict = dict()
        
        for i, face in enumerate(torchmesh.xfaces.T.cpu().numpy()):
            n0 = tuple(sorted(face[:2]))
            n1 = tuple(sorted(face[1:]))
            n2 = tuple(sorted([face[0], face[2]]))
                            
            try:
                dedge_neighbor_dict[n0].update({n1, n2})
            except KeyError:
                dedge_neighbor_dict[n0] = {n0}.union({n1, n2})
            try:
                dedge_neighbor_dict[n1].update({n0, n2})
            except KeyError:
                dedge_neighbor_dict[n1] = {n1}.union({n0, n2})
            try:
                dedge_neighbor_dict[n2].update({n0, n1})
            except KeyError:
                dedge_neighbor_dict[n2] = {n2}.union({n0, n1})

            dnodes.update({n0, n1, n2})
            
        return dnodes, dedge_neighbor_dict

    def _create_dual_graph_flip(self, torchmesh):
        """ Created dual graph and generate associated information for flipping. 

        Create dual graph whose nodes are edges in original mesh and
        edges are edges in original mesh belonging to a same face.

        Args:
            torchmesh: pyg data instance for mesh.

        Returns:
            dnodes: set of nodes in dual graph. A node is represented 
            by a pair of nodes in the original mesh.
            dedge_neighbor_dict: dictionary each of whose keys is a node
            in dual graph and values is a set of nodes in the dual graph 
            adjacent to the node of the key.
            neighbor_vertices: dictionary each of whose keys is a node 
            in the dual graph and values is a set of vertices adjacent
            to the key vertex.
            new_dnode_face_dict: dictionary each of whose keys is a node 
            in the dual graph and values is a list of nodes in the dual
            graph adjacent to the node of the key vertex.
        """
        dnodes = set()
        dedge_neighbor_dict = dict()
        neighbor_vertices = dict()
        new_dnode_face_dict = dict()

        for i, face in enumerate(torchmesh.xfaces.T.cpu().numpy()):
            n0 = tuple(sorted(face[:2]))
            n1 = tuple(sorted(face[1:]))
            n2 = tuple(sorted([face[0], face[2]]))

            try:
                new_dnode_face_dict[n0].extend([face[[1,2]], face[[0,2]]])
            except:
                new_dnode_face_dict[n0] = [face[[1,2]], face[[0,2]]]
            try:
                new_dnode_face_dict[n1].extend([face[[1,0]], face[[2,0]]])
            except:
                new_dnode_face_dict[n1] = [face[[1,0]], face[[2,0]]]
            try:
                new_dnode_face_dict[n2].extend([face[[0,1]], face[[2,1]]])
            except:
                new_dnode_face_dict[n2] = [face[[0,1]], face[[2,1]]]

            try:
                neighbor_vertices[n0].update(set(face))
                dedge_neighbor_dict[n0].update({n1, n2})
            except KeyError:
                neighbor_vertices[n0] = set(face)
                dedge_neighbor_dict[n0] = {n0}.union({n1, n2})
            try:
                neighbor_vertices[n1].update(set(face))
                dedge_neighbor_dict[n1].update({n0, n2})
            except KeyError:
                neighbor_vertices[n1] = set(face)
                dedge_neighbor_dict[n1] = {n1}.union({n0, n2})
            try:
                neighbor_vertices[n2].update(set(face))
                dedge_neighbor_dict[n2].update({n0, n1})
            except KeyError:
                neighbor_vertices[n2] = set(face)
                dedge_neighbor_dict[n2] = {n2}.union({n0, n1})

            dnodes.update({n0, n1, n2})

        return dnodes, dedge_neighbor_dict, new_dnode_face_dict, neighbor_vertices
    
    def _create_dual_graph_coarse(self, torchmesh):
        """ Created dual graph and generate associated information for coarsening. 
        
        Create dual graph whose nodes are edges in original mesh and
        edges are edges in original mesh belonging to a same face.
        
        Args:
            torchmesh: pyg data instance for mesh.
            
        Returns:
            neighbor_dict: dictionary each of whose keys is a node in the 
            dual graph and values is its node degree in the dual graph.
            node_edge_dict: dictionary each of whose keys is a node in
            the original graph and values is a set of edges connected to 
            the key node.
            node_edgeset_dict: dictionary each of whose keys is a node in
            the original graph and values is a set of vertices connected to
            the key vertex.
            node_face_dict: dictionary each of whose keys is a node in
            the original graph and values is the list of a pair of edges 
            that make angles out of the key node.
        """
        neighbor_dict = dict()
        node_edge_dict = dict()
        node_edgeset_dict = dict()
        node_face_dict = dict()

        for i, face in enumerate(torchmesh.xfaces.cpu().T.numpy()):
            n0 = tuple(sorted(face[:2]))
            n1 = tuple(sorted(face[1:]))
            n2 = tuple(sorted([face[0], face[2]]))

            try:
                node_face_dict[face[0]].extend([face[[1,0]], face[[1,2]], face[[2,0]], face[[2,1]]])
            except:
                node_face_dict[face[0]] = [face[[1,0]], face[[1,2]], face[[2,0]], face[[2,1]]]
            try:
                node_face_dict[face[1]].extend([face[[2,1]], face[[2,0]], face[[0,1]], face[[0,2]]])
            except:
                node_face_dict[face[1]] = [face[[2,1]], face[[2,0]], face[[0,1]], face[[0,2]]]
            try:
                node_face_dict[face[2]].extend([face[[0,2]], face[[0,1]], face[[1,2]], face[[1,0]]])
            except:
                node_face_dict[face[2]] = [face[[0,2]], face[[0,1]], face[[1,2]], face[[1,0]]]

            try:
                neighbor_dict[n0] += 2
            except KeyError:
                neighbor_dict[n0] = 2
            try:
                neighbor_dict[n1] += 2
            except KeyError:
                neighbor_dict[n1] = 2
            try:
                neighbor_dict[n2] += 2
            except KeyError:
                neighbor_dict[n2] = 2

            try:
                node_edge_dict[face[0]].update({n0, n2})
                node_edgeset_dict[face[0]].update(set(n0), set(n2))
            except KeyError:
                node_edge_dict[face[0]] = {n0, n2}
                node_edgeset_dict[face[0]] = {face[0]}.union(set(n0), set(n2))
            try:
                node_edge_dict[face[1]].update({n0, n1})
                node_edgeset_dict[face[1]].update(set(n0), set(n1))
            except KeyError:
                node_edge_dict[face[1]] = {n0, n1}
                node_edgeset_dict[face[1]] = {face[1]}.union(set(n0), set(n1))
            try:
                node_edge_dict[face[2]].update({n1, n2})
                node_edgeset_dict[face[2]].update(set(n1), set(n2))
            except KeyError:
                node_edge_dict[face[2]] = {n1, n2}
                node_edgeset_dict[face[2]] = {face[2]}.union(set(n1), set(n2))

        return neighbor_dict, node_edge_dict, node_edgeset_dict, node_face_dict
    
    def _create_onedim_graph_coarse(self, torchmesh):
        """ Create dictionaries of node and edge corresponence. 

        Args:
            torchmesh: pyg data instance for segment.

        Returns:
            node_edge_dict: dictionary each of whose keys is a node in
            the original graph and values is a set of edges connected to 
            the key node.
            node_edgeset_dict: dictionary each of whose keys is a node in
            the original graph and values is a set of vertices connected to
            the key vertex.
        """
        node_edge_dict = dict()
        node_edgeset_dict = dict()

        for i, edge in enumerate(torchmesh.edge_index.cpu().T.numpy()):
            node0 = edge[0]
            node1 = edge[1]

            try:
                node_edge_dict[node0].update({(node0, node1)})
                node_edgeset_dict[node0].update({node0, node1})
            except KeyError:
                node_edge_dict[node0] = {(node0, node1)}
                node_edgeset_dict[node0] = {node0, node1}
            try:
                node_edge_dict[node1].update({(node1, node0)})
                node_edgeset_dict[node1].update({node0, node1})
            except KeyError:
                node_edge_dict[node1] = {(node1, node0)}
                node_edgeset_dict[node1] = {node0, node1}

        return node_edge_dict, node_edgeset_dict

    def _sample_independent_edges(self, torchmesh):
        """ Find maximal independent splittable edges in mesh. 
        
        Args:
            torchmesh: pyg data instance for mesh.
            
        Returns:
            sampled_nodes: list of maximal independent flippable edges.
        """
        dnodes, dedge_neighbor_dict = self._create_dual_graph(torchmesh)
        sampled_nodes = []
        if self.samplemode.startswith("random"):
            if self.samplemode=="randomseed":
                seed_everything(42)
            while len(dnodes) > 0:
                snode = choice(list(dnodes))
                sampled_nodes.append(snode)
                dnodes = dnodes - dedge_neighbor_dict[snode]
        if self.samplemode.startswith("randomtest"):
            value,index = torch.sort(torch.tensor(sampled_nodes)[:,0])
            first_20_nodes = (torch.tensor(sampled_nodes)[index[:20]])
            total_num_nodes = int(torchmesh.x.shape[0]/2)
            first_20_nodes = (first_20_nodes).tolist()
            sampled_nodes = []
            for edge in first_20_nodes:
                sampled_nodes.append((edge[0],edge[1]))
                sampled_nodes.append((edge[0]+total_num_nodes,edge[1]+total_num_nodes))
        return sampled_nodes
            

    def _sample_flippable_edges(self, torchmesh):
        """ Find maximal independent flippable edges in mesh. 
        
        Args:
            torchmesh: pyg data instance for mesh.
            
        Returns:
            sampled_nodes: list of maximal independent flippable edges.
            new_dnode_face_dict: dictionary each of whose keys is a node 
            in the dual graph and values is a list of nodes in the dual
            graph adjacent to the node of the key vertex.
            neighbor_vertices: dictionary each of whose keys is a node 
            in the dual graph and values is a set of vertices adjacent
            to the key vertex.
        """
        dnodes, dedge_neighbor_dict, new_dnode_face_dict, neighbor_vertices = self._create_dual_graph_flip(torchmesh)
        for item in new_dnode_face_dict.items():
            if len(item[1]) == 2:
                dnodes.remove(item[0])

        sampled_nodes = []
        if self.samplemode.startswith("random"):
            if self.samplemode=="randomseed":
                seed_everything(42)
            while len(dnodes) > 0:
                snode = choice(list(dnodes))
                sampled_nodes.append(snode)
                dnodes = dnodes - dedge_neighbor_dict[snode] 
        if self.samplemode.startswith("randomtest"):
            value,index = torch.sort(torch.tensor(sampled_nodes)[:,0])
            first_20_nodes = (torch.tensor(sampled_nodes)[index[:20]])
            total_num_nodes = int(torchmesh.x.shape[0]/2)
            first_20_nodes = (first_20_nodes).tolist()
            sampled_nodes = []
            for edge in first_20_nodes:
                if edge[0]>=total_num_nodes or edge[1]>=total_num_nodes:continue
                if (edge[0]+total_num_nodes,edge[1]+total_num_nodes) in new_dnode_face_dict.keys():
                    sampled_nodes.append((edge[0],edge[1]))
                    sampled_nodes.append((edge[0]+total_num_nodes,edge[1]+total_num_nodes))
        # print("sampled_nodes",sampled_nodes)
        return sampled_nodes, new_dnode_face_dict, neighbor_vertices

    def _sample_coarsened_nodes(self, torchmesh):
        """ Find maximal independent corsenable edges in mesh. 
        
        Find maximal independent edges in mesh. Remove nodes with degree 2
        and angles more than 90 to provide valid mesh structure.
        
        Args:
            torchmesh: pyg data instance for mesh.
            
        Returns:
            sampled_nodes: list of maximal independent coarsened edges.
            snodes_edges_list: a list of edges connected to the sampled edges.
        """
        # Remove nodes with degree 2
        deg_tensor = self._create_degree_tensor(torchmesh)
        deg4nodes = ((deg_tensor == 4).nonzero(as_tuple=True)[0])
        dnodes = set(deg4nodes.tolist())
        neighbor_dict, node_edge_dict, node_edgeset_dict, node_face_dict = self._create_dual_graph_coarse(torchmesh)
        for item in neighbor_dict.items():
            if item[1] == 2:
                dnodes = dnodes - set(item[0])
        
        # Remove nodes belonging to a face with angles more than 90
        if len(dnodes) > 0:
            dnodes = self.del_node_with_obtuse_triangles(dnodes, node_face_dict, torchmesh)
        # Sample edge list
        snodes_edges_list = []
        sampled_nodes = []
        if self.samplemode.startswith("random"):
            if self.samplemode=="randomseed":
                seed_everything(42)
            while len(dnodes) > 0:
                snode = choice(list(dnodes))
                sampled_nodes.append(snode)
                snodes_edges_list.append(list(node_edge_dict[snode]))
                dnodes = dnodes - node_edgeset_dict[snode]
        if self.samplemode.startswith("randomtest"):
            value,index = torch.sort(torch.tensor(sampled_nodes)[:])
            first_20_nodes = (torch.tensor(sampled_nodes)[index[:]])
            total_num_nodes = int(torchmesh.x.shape[0]/2)
            first_20_nodes = (first_20_nodes).tolist()
            sampled_nodes = []
            snodes_edges_list = []
            for snode in first_20_nodes:
                if snode>total_num_nodes:continue
                snodes_edges_list.append(list(node_edge_dict[snode]))
                snodes_edges_list.append(list(node_edge_dict[snode+total_num_nodes]))
                sampled_nodes.append(snode)
                sampled_nodes.append(snode+total_num_nodes)
        return sampled_nodes, snodes_edges_list
        
    def _sample_onedim_coarsened_nodes_slower(self, torchmesh, is_timing=0):
        """ Find maximal independent corsenable edges in line segment. 

        Find maximal independent edges in segment. Remove nodes with 
        degree 1 (= boundary of the segment).

        Args:
            torchmesh: pyg data instance for line segment.

        Returns:
            sampled_nodes: list of maximal independent coarsened edges.
            snodes_edges_list: a list of edges connected to the sampled edges.
        """
        # Remove nodes with degree 1
        p.print("2.225111", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        deg_tensor = self._create_degree_tensor(torchmesh)
        deg2nodes = ((deg_tensor == 2).nonzero(as_tuple=True)[0])
        dnodes = set(deg2nodes.tolist())

        node_edge_dict, node_edgeset_dict = self._create_onedim_graph_coarse(torchmesh)

        # Sample edge list
        snodes_edges_list = []
        sampled_nodes = []
        p.print("2.225112", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        if self.samplemode.startswith("random"):
            if self.samplemode=="randomseed":
                seed_everything(42)
            while len(dnodes) > 0:
                snode = choice(list(dnodes))
                sampled_nodes.append(snode)
                snodes_edges_list.append(list(node_edge_dict[snode]))
                dnodes = dnodes - node_edgeset_dict[snode]
            p.print("2.22513a", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            return sampled_nodes, snodes_edges_list
        elif self.samplemode.startswith("2"):
            sampled_nodeslist = list(np.arange(1,198,4))
            for snode in sampled_nodeslist:
                sampled_nodes.append(snode)
                snodes_edges_list.append(list(node_edge_dict[snode]))
            p.print("2.225113b", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            return sampled_nodes, snodes_edges_list
        elif self.samplemode.startswith("1"):
            sampled_nodeslist = list(np.arange(1,99,4))
            for snode in sampled_nodeslist:
                sampled_nodes.append(snode)
                snodes_edges_list.append(list(node_edge_dict[snode]))
            p.print("2.225113c", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            return sampled_nodes, snodes_edges_list
    

    def _sample_onedim_coarsened_nodes_gpu(self, torchmesh, is_timing=True):
        """
        Faster version of "_sample_onedim_coarsened_nodes"
        
        Returns:
            coarsened_nodes: [n_chosen_nodes], independent nodes without edges connecting any of them.
            snodes_edges_tensor: [n_edges, 2, 2], the edges starting from the coarsened_nodes as source nodes
        """
        device = torchmesh.x.device
        ones = torch.ones(torchmesh.edge_index[0].shape, device=device)
        deg_tensor = scatter(ones, torchmesh.edge_index[0, :], reduce="sum")
        non_boundary = deg_tensor == 2

        fraction = 0.9
        max_pos = torchmesh.x_pos[torchmesh.batch==0].max().item()
        assert max_pos < 1
        x_pos = (torchmesh.x_pos.detach() + torchmesh.batch[:,None]).squeeze()
        length = len(x_pos)
        argsort = x_pos.argsort()
        non_boundary_mask = non_boundary[argsort]  # w.r.t. index
        chosen_idx = np.sort(np.random.choice(len(argsort), size=int(length*fraction), replace=False))
        chosen_idx_p1 = chosen_idx + 1
        same = np.roll(chosen_idx, shift=-1) == chosen_idx_p1  # If True, this node has connection to the right-ward node that is neighboring
        same = same | np.roll(same, shift=1)
        same[np.random.choice(2)::2] = False  # half of the nodes will be turned to False
        chosen_final = chosen_idx[~same]
        chosen_final_mask = torch.zeros(length, device=device).bool()
        chosen_final_mask[chosen_final] = True
        chosen_final_mask = chosen_final_mask & non_boundary_mask
        chosen_final = torch.where(chosen_final_mask)[0]
        coarsened_nodes = argsort[chosen_final]
        coarsened_nodes_left = argsort[chosen_final-1]
        coarsened_nodes_right = argsort[chosen_final+1]
        snodes_edges_tensor = torch.stack([
            torch.stack([coarsened_nodes,coarsened_nodes_left],-1),
            torch.stack([coarsened_nodes,coarsened_nodes_right],-1)], 1)
        return coarsened_nodes.detach(), snodes_edges_tensor.detach()


    def _sample_onedim_coarsened_nodes(self, torchmesh, is_timing=True):
        """
        Faster version of "_sample_onedim_coarsened_nodes"
        
        Returns:
            coarsened_nodes: [n_chosen_nodes], independent nodes without edges connecting any of them.
            snodes_edges_tensor: [n_edges, 2, 2], the edges starting from the coarsened_nodes as source nodes
        """
        device = torchmesh.x.device
        ones = torch.ones(torchmesh.edge_index[0].shape)
        deg_tensor = scatter(ones, torchmesh.edge_index[0, :].to("cpu"), reduce="sum")
        non_boundary = to_np_array(deg_tensor == 2)

        fraction = 0.9
        x_pos_np = to_np_array(torchmesh.x_pos)
        batch_np = to_np_array(torchmesh.batch)
        max_pos = x_pos_np[batch_np==0].max().item()
        assert max_pos < 1
        x_pos = (x_pos_np + batch_np[:,None]).squeeze()
        length = len(x_pos)
        argsort = x_pos.argsort()
        non_boundary_mask = non_boundary[argsort]  # w.r.t. index
        chosen_idx = np.sort(np.random.choice(len(argsort), size=int(length*fraction), replace=False))
        chosen_idx_p1 = chosen_idx + 1
        same = np.roll(chosen_idx, shift=-1) == chosen_idx_p1  # If True, this node has connection to the right-ward node that is neighboring
        same = same | np.roll(same, shift=1)
        same[np.random.choice(2)::2] = False  # half of the nodes will be turned to False
        chosen_final = chosen_idx[~same]
        chosen_final_mask = np.zeros(length).astype(bool)
        chosen_final_mask[chosen_final] = True
        chosen_final_mask = chosen_final_mask & non_boundary_mask
        chosen_final = np.where(chosen_final_mask)[0]
        coarsened_nodes = argsort[chosen_final]
        coarsened_nodes_left = argsort[chosen_final-1]
        coarsened_nodes_right = argsort[chosen_final+1]
        snodes_edges_tensor = np.stack([
            np.stack([coarsened_nodes,coarsened_nodes_left],-1),
            np.stack([coarsened_nodes,coarsened_nodes_right],-1)], 1)
        return coarsened_nodes, snodes_edges_tensor


    def gen_obtuse_mask(self, vectors):
        """ Compute mask indicating if or not sampled nodes may be coarsened. 
        
        Compute mask indicating if or not sampled nodes may be coarsened.
        If an angle of faces adjacent to a sampled node is more than 90, 
        the sampled node cannot be coarsened.
        
        Args:
            vectors: torch.tensor with of shape (2, 8 * sampled nodes).
            
        Returns:
            min_mask: mask indicating if or not sampled nodes are coarsened.
            The shape is (sampled nodes,)
        """
        distance = self.pdist(vectors[1, ...], vectors[0, ...])
        mult = distance[0::2] * distance[1::2]
        rel_vectors = (vectors[1, ...] - vectors[0, ...]).T
        dot_prod = (rel_vectors[:,0::2] * rel_vectors[:, 1::2]).T
        norm_dot = torch.sum(dot_prod, dim=1)/mult
        resh_ndot = norm_dot.reshape(-1,8)
        mins, _ = torch.min(resh_ndot, 1)
        min_mask = torch.where(mins > 0.1, True, False)
        return min_mask
        
    def del_node_with_obtuse_triangles(self, deg4dnodes, node_face_dict, torchmesh):
        """ Remove nodes with degree 4 such that the nodes are adjacent to obtuse triangles. 
        
        Remove nodes adjacent to obtuse triangles. Nodes adjacent to obtuse faces in
        either of 2d mesh or world mesh will be removed.
        
        Args:
            deg4dnodes: set of nodes whose neighbor degree is 4.
            node_face_dict: dictionary each of whose keys is a node in
            the original graph and values is the list of a pair of edges 
            that make angles out of the key node. 
            torchmesh: original mesh.
            
        Returns:
            dnodes: set of nodes adjacent to acute triangles.
        """
        list_dnodes = list(deg4dnodes)
        if len(list_dnodes) > 1:
            total_edge_pairs = np.concatenate([np.array(node_face_dict[node]).T for node in list_dnodes], axis=-1)
        elif len(list_dnodes) == 1:
            total_edge_pairs = np.array(node_face_dict[list_dnodes[0]]).T
        # Remove nodes based on faces in 2d space
        raw_vectors = torchmesh.x_coords[total_edge_pairs, :][...,:3]
        min_mask = self.gen_obtuse_mask(raw_vectors)
        list_dnodes = torch.tensor(list_dnodes)[min_mask].tolist()
        if len(list_dnodes) == 0:
            return set(list_dnodes)
        
        if len(list_dnodes) > 1:
            total_edge_pairs = np.concatenate([np.array(node_face_dict[node]).T for node in list_dnodes], axis=-1)
        elif len(list_dnodes) == 1:
            total_edge_pairs = np.array(node_face_dict[list_dnodes[0]]).T
        
        # Remove nodes based on faces in 3d space

        world_vectors = torchmesh.x_coords[total_edge_pairs, :][...,3:6]
        min_wmask = self.gen_obtuse_mask(world_vectors)
        dnodes = set(torch.tensor(list_dnodes)[min_wmask].tolist())
        
        return dnodes

    def onedim_edge_split(self, sampnodes, outmesh):
        """ Split edges according to the sampled independent edges. 

        Args:
            sampnodes: sampled indepnedent edges to be split.
            outmesh: mesh data where the sampnodes come from. 

        Returns:
            split_pts: new vertices generated by split action. Shape is [sampnodes, outmesh.x.shape[1]].
            remained_faces: unsplitted faces. Shape is [3, outmesh.x.faces.shape[1] - (faces to be splitted)]. 
            all_refined_faces: generated new faces. Shape is [3, 2*(faces to be splitted)]. 
            remained_edge_index: unsplitted edges. Shape is [2, outmesh.edge_index.shape[1] - 2*sampnodes].
            newly_generated_edges: generated new edges. Shape is [2, 8*sampnodes].
        """
        # Assign hashes to edges in original edge_node_list
        oriented_sampnodes = torch.cat([sampnodes.T, sampnodes.T[[1,0],:]], dim=-1)

        # Precompute edges
        delete_edge_mask = torch.isin(
            edge_index_to_num(outmesh.edge_index, self.nmax), 
            edge_index_to_num(oriented_sampnodes, self.nmax)
        )
        remained_edge_mask = torch.logical_not(delete_edge_mask)
        remained_edge_index = outmesh.edge_index[:, remained_edge_mask]  

        ## Interpolate newly added nodes
        new_vindices = torch.arange(outmesh.x_pos.shape[0], outmesh.x_pos.shape[0]+sampnodes.shape[0], device=outmesh.x.device)
        receivers = outmesh.x[sampnodes[:,0], :]
        senders = outmesh.x[sampnodes[:,1], :]
        split_pts = 1/2 * (receivers + senders)
        receivers_pos = outmesh.x_pos[sampnodes[:,0], :]
        senders_pos = outmesh.x_pos[sampnodes[:,1], :]
        split_pts_pos = 1/2 * (receivers_pos + senders_pos)
        receivers_phy = outmesh.x_phy[sampnodes[:,0], :]
        senders_phy = outmesh.x_phy[sampnodes[:,1], :]
        split_pts_phys = 1/2 * (receivers_phy + senders_phy)
        receivers_batch = outmesh.batch[sampnodes[:,0]]
        senders_batch = outmesh.batch[sampnodes[:,1]]
        split_pts_batch = 1/2 * (receivers_batch + senders_batch)
        ## 
        gen_tensor = torch.cat(
            [sampnodes.T, new_vindices.reshape(1,-1)], dim=0
        )

        new_edge_index = torch.cat(
            [
                gen_tensor[[0,2], :], gen_tensor[[2,0], :],
                gen_tensor[[1,2], :], gen_tensor[[2,1], :]
            ], dim=-1         
        )
        return split_pts, split_pts_pos, split_pts_phys, split_pts_batch, remained_edge_index, new_edge_index    
    
    
    def face_edge_split(self, sampnodes, outmesh):
        """ Split faces and edges according to the sampled independent edges. 
        
        Args:
            sampnodes: sampled indepnedent edges to be split.
            outmesh: mesh data where the sampnodes come from. 
            
        Returns:
            split_pts: new vertices generated by split action. Shape is [sampnodes, outmesh.x.shape[1]].
            remained_faces: unsplitted faces. Shape is [3, outmesh.x.faces.shape[1] - (faces to be splitted)]. 
            all_refined_faces: generated new faces. Shape is [3, 2*(faces to be splitted)]. 
            remained_edge_index: unsplitted edges. Shape is [2, outmesh.edge_index.shape[1] - 2*sampnodes].
            newly_generated_edges: generated new edges. Shape is [2, 8*sampnodes].
        """
        traw_enlist = torch.cat(
            [
                outmesh.xfaces,
                outmesh.xfaces[[2,0,1], :],
                outmesh.xfaces[[1,2,0], :]
            ], -1
        )

        # Assign hashes to edges in original edge_node_list
        raw_hash = edge_index_to_num(traw_enlist[:2, :], self.nmax)
        oriented_sampnodes = torch.cat([sampnodes.T, sampnodes.T[[1,0],:]], dim=-1)
        hash_samnodes = edge_index_to_num(oriented_sampnodes, self.nmax)

        # Precompute edges
        delete_edge_mask = torch.isin(
            edge_index_to_num(outmesh.edge_index, self.nmax), 
            edge_index_to_num(oriented_sampnodes, self.nmax)
        )
        remained_edge_mask = torch.logical_not(delete_edge_mask)
        remained_edge_index = outmesh.edge_index[:, remained_edge_mask]

        # Precompute faces
        ## Splittable edges
        splittable_mask = torch.isin(raw_hash, hash_samnodes)
        oriented_splittable_faces = traw_enlist[:, splittable_mask]

        ## Remaining edges (edges not to be split)
        raw_face_mask = torch.stack(
            [
                torch.isin(edge_index_to_num(outmesh.xfaces[[0,1], :], self.nmax), hash_samnodes),
                torch.isin(edge_index_to_num(outmesh.xfaces[[2,0], :], self.nmax), hash_samnodes),
                torch.isin(edge_index_to_num(outmesh.xfaces[[1,2], :], self.nmax), hash_samnodes)
            ], dim=0
        )
        rem_face_mask = torch.logical_not(torch.sum(raw_face_mask, 0))
        remained_faces = outmesh.xfaces[:, rem_face_mask]
        splittable_sort, _ = torch.sort(oriented_splittable_faces[:2, :], 0)
        splittable_hash = edge_index_to_num(splittable_sort, self.nmax)

        ## sort splittable faces
        _, sort_index = torch.sort(splittable_hash)
        sorted_osplit_faces = oriented_splittable_faces[:, sort_index]    

        ## Interpolate newly added nodes
        inter_sort, _ = torch.sort(sorted_osplit_faces[0:2, :], dim=0)
        split_edges = torch.unique(inter_sort, dim=1).T
        receivers = torch.gather(outmesh.x, 0, split_edges[:, 0].unsqueeze(-1).repeat(1, outmesh.x.shape[1]).to(outmesh.x.device))
        senders = torch.gather(outmesh.x, 0, split_edges[:, 1].unsqueeze(-1).repeat(1, outmesh.x.shape[1]).to(outmesh.x.device))
        split_pts = 1/2 * (receivers + senders)
        
        receivers = torch.gather(outmesh.x_pos, 0, split_edges[:, 0].unsqueeze(-1).repeat(1, outmesh.x_pos.shape[1]).to(outmesh.x.device))
        senders = torch.gather(outmesh.x_pos, 0, split_edges[:, 1].unsqueeze(-1).repeat(1, outmesh.x_pos.shape[1]).to(outmesh.x.device))
        split_pts_pos = 1/2 * (receivers + senders)
        
        receivers = torch.gather(outmesh.batch.unsqueeze(-1), 0, split_edges[:, 0].unsqueeze(-1).repeat(1, outmesh.batch.unsqueeze(-1).shape[1]).to(outmesh.x.device)).flatten()
        senders = torch.gather(outmesh.batch.unsqueeze(-1), 0, split_edges[:, 1].unsqueeze(-1).repeat(1, outmesh.batch.unsqueeze(-1).shape[1]).to(outmesh.x.device)).flatten()
        split_pts_batch = 1/2 * (receivers + senders)
                
        receivers = torch.gather(outmesh.x_phy, 0, split_edges[:, 0].unsqueeze(-1).repeat(1, outmesh.x_phy.shape[1]).to(outmesh.x.device))
        senders = torch.gather(outmesh.x_phy, 0, split_edges[:, 1].unsqueeze(-1).repeat(1, outmesh.x_phy.shape[1]).to(outmesh.x.device))
        split_pts_phys = 1/2 * (receivers + senders)
        
        receivers = torch.gather(outmesh.x_coords, 0, split_edges[:, 0].unsqueeze(-1).repeat(1, outmesh.x_coords.shape[1]).to(outmesh.x.device))
        senders = torch.gather(outmesh.x_coords, 0, split_edges[:, 1].unsqueeze(-1).repeat(1, outmesh.x_coords.shape[1]).to(outmesh.x.device))
        split_pts_x_coords = 1/2 * (receivers + senders)
        
        split_onehot = torch.tensor([0.,1.], dtype=torch.float32, device=outmesh.x.device).repeat(split_edges.shape[0],1)
        # outmesh.kimenatics_onehot = outmesh.onehot_list[-1][:,0]
        # split_kinem = torch.zeros(split_edges.shape[0], 1)

        ## sort newly added vertices
        temp_sort, inverse_indices = torch.unique(
            splittable_hash, sorted=True, return_inverse=True
        )
        new_vertices = torch.arange(outmesh.x.shape[0], outmesh.x.shape[0]+temp_sort.shape[0])    
        scat_new_vertices = new_vertices[inverse_indices][sort_index].to(outmesh.x.device)

        ## Generate universal matrix that can derive edge_index and meshes
        generating_mesh_components = torch.cat(
            [sorted_osplit_faces[0:1, :], scat_new_vertices.reshape(1, -1), sorted_osplit_faces[1:, :]], dim=0
        )

        ## Create newly refined faces
        all_refined_faces = torch.cat(
            [generating_mesh_components[[0,1,3], :], generating_mesh_components[[1,2,3], :]], dim=-1
        )

        ## Create newly refined and added edges
        newly_generated_edges = torch.cat(
            [
                generating_mesh_components[[0,1], :], generating_mesh_components[[1,0], :],
                generating_mesh_components[[1,3], :], generating_mesh_components[[3,1], :],
                generating_mesh_components[[1,2], :], generating_mesh_components[[2,1], :]
            ], dim=-1         
        )
        newly_generated_edges = torch.unique(newly_generated_edges, dim=1)

        return split_pts, split_pts_phys, split_pts_pos,split_pts_batch, split_pts_x_coords, remained_faces, all_refined_faces, remained_edge_index, newly_generated_edges, split_onehot#, split_kinem 
    
    def split(self, split_edges, outmesh):
        """ Perform split action on mesh according to sampled edges.
        
        Args:
            split_edges: sampled edges to be splitten. 
            outmesh: mesh data.
            
        Returns:
            outmesh: mesh data with splitted edges and faces.
        """
        if split_edges.shape[0] == 0:
            return outmesh

        if outmesh.xfaces.shape[0] == 0:
            split_pts,split_pts_pos,split_pts_phys, split_pts_batch, remained_edge_index, new_edge_index = self.onedim_edge_split(split_edges, outmesh)
            outmesh.x_phy = torch.cat([outmesh.x_phy, split_pts_phys], 0)
            outmesh.batch = torch.cat([outmesh.batch, split_pts_batch], 0)
        else:
            # split_pts,split_pts_phys, split_pts_pos, split_pts_batch, split_pts_x_coords, remained_faces, all_refined_faces, remained_edge_index, new_edge_index = self.face_edge_split(split_edges, outmesh)
            split_pts,split_pts_phys, split_pts_pos, split_pts_batch, split_pts_x_coords, remained_faces, all_refined_faces, remained_edge_index, new_edge_index, split_onehot  = self.face_edge_split(split_edges, outmesh)
            outmesh.x_phy = torch.cat([outmesh.x_phy, split_pts_phys], 0)
            # print(outmesh.batch, split_pts_batch)
            outmesh.batch = torch.cat([outmesh.batch, split_pts_batch], 0)
            outmesh.x_coords = torch.cat([outmesh.x_coords, split_pts_x_coords], 0)
            temp_onehots = list(outmesh.onehot_list)
            temp_onehots[0] = torch.cat([outmesh.onehot_list[0], split_onehot], 0)
            outmesh.onehot_list = tuple(temp_onehots)
            # outmesh.kinematics_[-1] = torch.cat([outmesh.kinematics[-1], split_kinem], 0)

        # Update node feature, edge index and faces
        outmesh.x = torch.cat([outmesh.x, split_pts], 0)
        outmesh.edge_index = torch.cat([new_edge_index, remained_edge_index], -1)
        outmesh.x_pos = torch.cat([outmesh.x_pos, split_pts_pos], 0)
        
        if outmesh.xfaces.shape[0] == 0:
            if not (outmesh.x.shape[0] - outmesh.edge_index.shape[1]/2) == outmesh.batch.max()+1:
                print("Split is invalid")
                pdb.set_trace()
                return None
        else:
            outmesh.xfaces = torch.cat([all_refined_faces, remained_faces], dim=-1)
            if not (outmesh.x.shape[0] + outmesh.xfaces.shape[1] - outmesh.edge_index.shape[1]/2) == outmesh.batch.max()+1:
                print("Split is invalid")
                pdb.set_trace()
                return None

        return outmesh
        
    def _compute_sizing_fields(self, outmesh, tor_samnodes):
        """ Reshape sizing fields.
        
        Args:
            outmesh: mesh data.
            tor_samnodes: `torch.tensor` of indices of boundaries (=vertices) of edges 
            to be splitted/flipped. The shape is [(number of edges to be splitted), 2].
            
        Returns:
            sizing_fields: shape [tor_samnodes.shape[-1], 2, 2]if 2d mesh and 
            [tor_samnodes.shape[-1], 1] if 1d segment.
        """        
        # raw_sizing_fields = (outmesh.x[:,self.node_dim:][tor_samnodes[:, 0], :] + outmesh.x[:,self.node_dim:][tor_samnodes[:, 1], :])/2
        if outmesh.xfaces.shape[0] == 0:
            raw_sizing_fields = (outmesh.x[:,-1:][tor_samnodes[:, 0], :] + outmesh.x[:,-1:][tor_samnodes[:, 1], :])/2
            return raw_sizing_fields
        else:
            raw_sizing_fields = (outmesh.x[:,-4:][tor_samnodes[:, 0], :] + outmesh.x[:,-4:][tor_samnodes[:, 1], :])/2
            return raw_sizing_fields.reshape(-1,2,2)
    
    def _compute_coarsening_sizing_fields(self, outmesh, tor_samnodes):
        """ Reshape sizing fields for coarsening.
        
        Args:
            outmesh: mesh data.
            tor_samnodes: `torch.tensor` of indices of boundaries (=vertices) of edges 
            to be splitted/flipped. The shape is [(number of edges to be splitted), 2].
            
        Returns:
            sizing_fields: shape [tor_samnodes.shape[-1], 2, 2] if 2d mesh and 
            [tor_samnodes.shape[-1], 1] if 1d segment.
        """      
        #raw_sizing_fields = (outmesh.x[:,self.node_dim:][tor_samnodes[:,:,0], :] + outmesh.x[:,self.node_dim:][tor_samnodes[:,:,1], :])/2
        if outmesh.xfaces.shape[0] == 0:
            raw_sizing_fields = (outmesh.x[:,-1:][tor_samnodes[:,:,0], :] + outmesh.x[:,-1:][tor_samnodes[:,:,1], :])/2
            return raw_sizing_fields
        else:
            raw_sizing_fields = (outmesh.x[:,-4:][tor_samnodes[:,:,0], :] + outmesh.x[:,-4:][tor_samnodes[:,:,1], :])/2
            sizing_fields = raw_sizing_fields.reshape(-1,2,2)
            return sizing_fields.to(outmesh.x.device)
        
    def gen_delaunay_mask(self, torchmesh, sampled_nodes, new_dnode_face_dict, neighbor_vertices):
        """ Compute delaunay criterion to decide edges to be flipped.
        
        Args:
            torchmesh: mesh data with outmesh.
            sampled_nodes: list of maximal independent flippable edges.
            new_dnode_face_dict: dictionary each of whose keys is a node 
            in the dual graph and values is a list of nodes in the dual
            graph adjacent to the node of the key vertex.
            neighbor_vertices: dictionary each of whose keys is a node 
            in the dual graph and values is a set of vertices adjacent
            to the key vertex.
            
        Returns:
            delaunay_mask: mask indicating whether or not sampled edges 
            can be flipped. The shape is (len(sampled_nodes),).
        """
        indices = torch.tensor(np.array([new_dnode_face_dict[fedge] for fedge in sampled_nodes]))
        abs_vecs = torchmesh.x[indices, :2]
        rel_vecs = abs_vecs[:,:,0,:] - abs_vecs[:,:,1,:]
        trans_rel_vecs = torch.transpose(rel_vecs, 2, 1)

        newindices = torch.tensor([list(neighbor_vertices[fedge]) for fedge in sampled_nodes])
        raw_sizing_tensor = torchmesh.x[newindices, :][..., -1*self.sizing_field_dim:]
        sizing_tensor = raw_sizing_tensor.sum(dim=1).reshape(-1,2,2)/4
        
        right_mul = torch.matmul(sizing_tensor, trans_rel_vecs[...,1::2])
        temp_adj_right_tensor = torch.stack([right_mul.reshape(-1,2).T.flatten()[0::2], right_mul.reshape(-1,2).T.flatten()[1::2]], dim=0)
        adj_right_tensor = torch.stack([right_mul.flatten()[0::2], right_mul.flatten()[1::2]], dim=0)
        quad_tensors = torch.diagonal(torch.matmul(rel_vecs[:,0::2,:].reshape(-1,2), adj_right_tensor))

        ext_rel_right_tensor = torch.cat([rel_vecs[:,0::2,:].reshape(-1,2), torch.zeros(int(rel_vecs.shape[0]*rel_vecs.shape[1]/2),1, device=torchmesh.x.device)], dim=-1)
        ext_rel_left_tensor = torch.cat([rel_vecs[:,1::2,:].reshape(-1,2), torch.zeros(int(rel_vecs.shape[0]*rel_vecs.shape[1]/2),1, device=torchmesh.x.device)], dim=-1)
        ext_rel_right_tensor, ext_rel_left_tensor

        cross_prods = torch.cross(ext_rel_right_tensor, ext_rel_left_tensor, dim=1)[:,2]
        cross_quads = quad_tensors*cross_prods

        delaunay_mask = (cross_quads[0::2] + cross_quads[1::2]) < 0

        return delaunay_mask
    
    def choose_split_edge(self, outmesh):
        """ Choose splittable edges.
        
        Args:
            outmesh: mesh data.
            
        Returns:
            split_edges: edges to be splitten with shape [(number of edges to be splitted), 2].
        """      
        # sample maximal independent splittable edges
        if outmesh.xfaces.shape[0] == 0:
            values, _ = outmesh.edge_index.sort(dim=0)
            tor_samnodes = values.unique(dim=-1).T
        else:
            sampled_nodes = self._sample_independent_edges(outmesh)
            if len(sampled_nodes) == 0:
                return torch.tensor([]).reshape(0,2)
            tor_samnodes =  torch.tensor(sampled_nodes, dtype=torch.int64, device=outmesh.x.device)
            
        # compute edge feature
        if outmesh.xfaces.shape[0] == 0:
            # if is_pos=True
            # Need to fix 
            receivers = outmesh.x[tor_samnodes[:, 0], 1:2]
            senders = outmesh.x[tor_samnodes[:, 1], 1:2]
        else:
            receivers = outmesh.x[tor_samnodes[:, 0], :2]
            senders = outmesh.x[tor_samnodes[:, 1], :2]
        edge_features = receivers - senders

        # compute error estimator
        sizing_fields = self._compute_sizing_fields(outmesh, tor_samnodes) #shape: [sampled_edges, 2, 2] or [sampled_edge, 1]        
        if outmesh.xfaces.shape[0] == 0:
            half = sizing_fields * edge_features #shape: [sampled_edges, 1]
            estimators = edge_features * half
        else:
            half = torch.matmul(sizing_fields, edge_features.reshape(-1,2,1)) #shape: [sampled_edges, 2]
            estimators = torch.matmul(edge_features.reshape(-1,1,2), half).flatten()

        # compute probability on sampled splittable edges
        remesh_prob = torch.sigmoid(estimators - 1.) #.to(outmesh.x.device)
        if self.is_split_test:
            mask = remesh_prob > 0.1
        else:
            mask = remesh_prob > 0.5
        mask = mask.reshape((mask.shape[0], 1))
        split_edges = torch.masked_select(tor_samnodes, mask).reshape(-1,2)
        
        return split_edges
    
    def choose_flippable_edge(self, outmesh):
        """ Choose flippable edges.

        Args:
            outmesh: mesh data.
            
        Returns:
            flip_edges: edges to be splitten with shape [(number of edges to be flipped), 2].
        """      
        sampled_nodes, new_dnode_face_dict, neighbor_vertices = self._sample_flippable_edges(outmesh)
        if len(sampled_nodes) == 0:
            return torch.tensor([]).reshape(0,2)
        delaunay_mask = self.gen_delaunay_mask(outmesh, sampled_nodes, new_dnode_face_dict, neighbor_vertices)
        tor_samnodes = torch.tensor(sampled_nodes, device=outmesh.x.device)[delaunay_mask, :]

        # compute edge feature
        receivers = outmesh.x[tor_samnodes[:, 0], :][:, :2]
        senders = outmesh.x[tor_samnodes[:, 1], :][:, :2]
        edge_features = receivers - senders

        # compute error estimator
        sizing_fields = self._compute_sizing_fields(outmesh, tor_samnodes)
        if outmesh.xfaces.shape[0] == 0:
            half = torch.matmul(sizing_fields, edge_features.reshape(-1,1,1)) #shape: [sampled_edges, 2]
            estimators = torch.matmul(edge_features.reshape(-1,1,1), half).flatten()
        else:
            half = torch.matmul(sizing_fields, edge_features.reshape(-1,2,1)) #shape: [sampled_edges, 2]
            estimators = torch.matmul(edge_features.reshape(-1,1,2), half).flatten()

        # compute probability on sampled splittable edges
        remesh_prob = torch.sigmoid(estimators - 1.) #.to(outmesh.x.device)
        if self.is_flip_test:
            mask = remesh_prob > 0.1
        else:
            mask = remesh_prob > 0.5
        mask = mask.reshape((mask.shape[0], 1))
        flip_edges = torch.masked_select(tor_samnodes, mask).reshape(-1,2)
        
        return flip_edges   

    def choose_onedim_coarsen_edges(self, torchmesh):
        """ Choose edges to be coarsened.

        Args:
            torchmesh: mesh data.

        Returns:
            split_edges: edges to be coarsened with shape [(number of edges to be coarsened), 2].
        """      
        coarsened_nodes, snodes_edges_tensor = self._sample_onedim_coarsened_nodes(torchmesh)
        if len(coarsened_nodes) == 0:
            collapsed_edges = torch.tensor([[],[]])
            merging_nodes = torch.tensor([[],[]])
            return collapsed_edges, merging_nodes

        # snodes_edges_tensor = torch.tensor(snodes_edges_list, device=torchmesh.x.device)

        receivers = torchmesh.x[snodes_edges_tensor[..., 0]]
        senders = torchmesh.x[snodes_edges_tensor[..., 1]]
        edge_features = (receivers - senders)[..., -2:-1]
        sizing_fields = self._compute_coarsening_sizing_fields(torchmesh, snodes_edges_tensor)
        final_mul = edge_features * sizing_fields * edge_features

        # estimators = final_mul.reshape(2, -1)
        estimators = final_mul[...,0].T
        probs = torch.sigmoid(estimators - 1)
        indices = torch.argmax(probs, dim=0)

        max_probs = torch.gather(probs, 0, indices.reshape(1,-1))
        coarsening_mask = max_probs > 0.5
        # torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)

        highest_prob_edges = torch.gather(snodes_edges_tensor, 1, torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)).reshape(indices.shape[0],2).T

        collapsed_edges = highest_prob_edges[:,coarsening_mask.reshape(coarsening_mask.shape[1])]

        merging_nodes = torch.tensor(coarsened_nodes, device=torchmesh.x.device)[coarsening_mask[0,:]]

        return collapsed_edges, merging_nodes  
    
    
    def choose_coarsen_edges(self, torchmesh):
        """ Choose edges to be coarsened.
        
        Args:
            torchmesh: mesh data.
            
        Returns:
            split_edges: edges to be coarsened with shape [(number of edges to be coarsened), 2].
        """      
        coarsened_nodes, snodes_edges_list = self._sample_coarsened_nodes(torchmesh)
        if len(coarsened_nodes) == 0:
            collapsed_edges = torch.tensor([[],[]])
            merging_nodes = torch.tensor([[],[]])
            return collapsed_edges, merging_nodes
        
        snodes_edges_tensor = torch.tensor(snodes_edges_list, device=torchmesh.x.device)

        receivers = torchmesh.x[snodes_edges_tensor[..., 0]]
        senders = torchmesh.x[snodes_edges_tensor[..., 1]]
        edge_features = (receivers - senders)[..., :2]

        sizing_fields = self._compute_coarsening_sizing_fields(torchmesh, snodes_edges_tensor)

        resh_edge_features = edge_features.reshape(-1,1,2)
        batched_sizing_fields = sizing_fields
        
        left_mul = torch.matmul(resh_edge_features, batched_sizing_fields)
        final_mul = torch.matmul(left_mul, resh_edge_features.reshape(-1, 2, 1))

        estimators = final_mul.reshape(4, -1)
        probs = torch.sigmoid(estimators - 1)
        indices = torch.argmax(probs, dim=0)

        max_probs = torch.gather(probs, 0, indices.reshape(1,-1))
        coarsening_mask = max_probs > 0.5
        torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)

        highest_prob_edges = torch.gather(snodes_edges_tensor, 1, torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)).reshape(indices.shape[0],2).T

        collapsed_edges = highest_prob_edges[:,coarsening_mask.reshape(coarsening_mask.shape[1])]

        merging_nodes = torch.tensor(coarsened_nodes, device=torchmesh.x.device)[coarsening_mask[0,:]]

        return collapsed_edges, merging_nodes    


    def compute_new_edge_index(self, torchmesh, collapsed_edges, merging_nodes):
        """ Choose splittable edges.
        
        Args:
            torchmesh: mesh data.
            collapsed_edges: edges to be coarsened. The shape is (number of collapsed edges, 2).
            merging_nodes: nodes to be merged to another nodes along with the collapsed edges.
            The shape is (number of collapsed edges,).
            
        Returns:
            new_edge_index: edge_index with original vertex indices.
            mapping: correspondence between target and source of nodes to be merged.
        """   
        a = collapsed_edges.flatten()
        b = merging_nodes
        raw_index = ((a - b.unsqueeze(0).T) == 0).nonzero()[:,1]

        residue = raw_index.clone()
        row1 = torch.where(residue >= residue.shape[0], 0, residue.shape[0])
        row2 = torch.remainder(residue, residue.shape[0])
        opp_edges = row1 + row2
        merged_nodes = collapsed_edges.flatten()[opp_edges]
        mapping = torch.stack([merging_nodes, merged_nodes])

        edges = torchmesh.edge_index
        hash_aedges = edge_index_to_num(edges, self.nmax)
        hash_acollapses = edge_index_to_num(
            torch.cat(
                [collapsed_edges, 
                 torch.stack([collapsed_edges[1,:], collapsed_edges[0,:]], dim=0)]
                , dim=-1), self.nmax
        )
        colrem_mask = torch.logical_not(torch.isin(hash_aedges, hash_acollapses))
        rem_edges = edges[:,colrem_mask]

        a = rem_edges.flatten()
        b = mapping[0,:]
        raw_index = ((b - a.unsqueeze(0).T) == 0).nonzero().T

        flatten_rems = rem_edges.flatten()
        flatten_rems[raw_index[0,:]] = mapping[1,:][raw_index[1,:]]
        new_edge_index = torch.unique(rem_edges, sorted=False, dim=1).to(torchmesh.x.device)

        return new_edge_index, mapping

    def generate_flipped_edges(self, sampnodes, outmesh):
        """ Flip faces and edges according to the sampled independent edges. 
        
        Args:
            sampnodes: sampled indepnedent edges to be flipped.
            outmesh: mesh data where the sampnodes come from. 
            
        Returns:
            remained_faces: unflipped faces. Shape is [3, outmesh.x.faces.shape[1] - (faces to be flipped)].
            flipped_new_faces: generated new faces. Shape is [3, 2*(faces to be flipped)]. 
            remained_edge_index: unflippd edges. Shape is [2, outmesh.edge_index.shape[1] - 2*sampnodes].
            all_flipped_edge_index: generated new edges. Shape is [2, 2*sampnodes].
        """
        traw_enlist = torch.cat(
            [
                outmesh.xfaces,
                outmesh.xfaces[[2,0,1], :],
                outmesh.xfaces[[1,2,0], :]
            ], -1
        )

        # Assign hashes to edges in original edge_node_list
        raw_hash = edge_index_to_num(traw_enlist[:2, :], self.nmax)
        oriented_sampnodes = torch.cat([sampnodes.T, sampnodes.T[[1,0],:]], dim=-1)
        hash_samnodes = edge_index_to_num(oriented_sampnodes, self.nmax)

        # Precompute edges
        delete_edge_mask = torch.isin(
            edge_index_to_num(outmesh.edge_index, self.nmax), 
            edge_index_to_num(oriented_sampnodes, self.nmax)
        )
        remained_edge_mask = torch.logical_not(delete_edge_mask)
        remained_edge_index = outmesh.edge_index[:, remained_edge_mask]

        # Precompute faces
        ## Flippable edges
        flippable_mask = torch.isin(raw_hash, hash_samnodes)
        oriented_flippable_faces = traw_enlist[:, flippable_mask]

        ## Remaining edges (edges not to be flipped)
        raw_face_mask = torch.stack(
            [
                torch.isin(edge_index_to_num(outmesh.xfaces[[0,1], :], self.nmax), hash_samnodes),
                torch.isin(edge_index_to_num(outmesh.xfaces[[2,0], :], self.nmax), hash_samnodes),
                torch.isin(edge_index_to_num(outmesh.xfaces[[1,2], :], self.nmax), hash_samnodes)
            ], dim=0
        )
        rem_face_mask = torch.logical_not(torch.sum(raw_face_mask, 0))
        remained_faces = outmesh.xfaces[:, rem_face_mask]
        flippable_sort, _ = torch.sort(oriented_flippable_faces[:2, :], 0)
        flippable_hash = edge_index_to_num(flippable_sort, self.nmax)

        ## sort flippable faces
        _, sort_index = torch.sort(flippable_hash)
        sorted_oflip_faces = oriented_flippable_faces[:, sort_index]    

        generating_faces = torch.cat(
            [
                sorted_oflip_faces, 
                 sorted_oflip_faces[-1:, :].reshape(-1,2)[:, [1,0]].flatten().reshape(1,-1)
            ], 
            dim=0
        )
        flipped_new_faces = generating_faces[1:, :]
        all_flipped_edge_index = generating_faces[-2:, :]

        return remained_faces, flipped_new_faces, remained_edge_index, all_flipped_edge_index
    
    def delete_adjacent_faces(self, torchmesh, mapping):
        """Delete faces adjacent to merged edges. 
        
        Args:
            torchmesh: mesh data.
            mapping: correspondence between target and source of nodes to be merged. 
            
        Returns:
            orig_faces[:, remaining_mask]: cleaned faces without collapsed faces,
            but with original indices of vertices.
        """        
        orig_faces = torchmesh.xfaces
        tes22_0 = mapping    
        tes33_0 = orig_faces

        tes33_01 = tes33_0[torch.tensor([0,1], dtype=torch.int64, device=torchmesh.x.device), :]
        tes33_02 = tes33_0[torch.tensor([0,2], dtype=torch.int64, device=torchmesh.x.device), :]
        tes33_12 = tes33_0[torch.tensor([1,2], dtype=torch.int64, device=torchmesh.x.device), :]
        tes33_10 = tes33_0[torch.tensor([1,0], dtype=torch.int64, device=torchmesh.x.device), :]
        tes33_20 = tes33_0[torch.tensor([2,0], dtype=torch.int64, device=torchmesh.x.device), :]
        tes33_21 = tes33_0[torch.tensor([2,1], dtype=torch.int64, device=torchmesh.x.device), :]

        hash22 = edge_index_to_num(tes22_0,  self.nmax)
        hashes33 = torch.stack(
            [
                torch.isin(edge_index_to_num(tes33_01, self.nmax), hash22),
                torch.isin(edge_index_to_num(tes33_02, self.nmax), hash22),
                torch.isin(edge_index_to_num(tes33_12, self.nmax), hash22),
                torch.isin(edge_index_to_num(tes33_10, self.nmax), hash22),
                torch.isin(edge_index_to_num(tes33_20, self.nmax), hash22),
                torch.isin(edge_index_to_num(tes33_21, self.nmax), hash22)
            ], 0
        )

        remaining_mask = torch.logical_not(torch.sum(hashes33, dim=0))

        return orig_faces[:, remaining_mask]

    def clean_isolated_vertices(self, torchmesh, new_edge_index, newly_generated_faces):
        """Clean isolated vertices and reorder the indices. 
        
        Args:
            torchmesh: mesh data.
            new_edge_index: edge_index with original vertex indices.
            newly_generated_faces: indices of new faces.
             
        Returns:
            new_faces: reindexed faces.
            new_edge_index: reindexed edge index.
            new_x: reordered vertices.
        """
        num_nodes = torchmesh.x.shape[0]
        device = torchmesh.x.device
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        edge_index = new_edge_index
        mask[edge_index.flatten()] = 1

        assoc = torch.full((num_nodes, ), -1, dtype=torch.long, device=device)
        assoc[mask] = torch.arange(mask.sum(), device=device)

        new_faces = assoc[newly_generated_faces]
        new_edge_index = assoc[edge_index]
        new_x = torchmesh.x[mask]
        new_x_pos = torchmesh.x_pos[mask]
        new_x_phy = torchmesh.x_phy[mask]
        new_x_batch = torchmesh.batch[mask]
        new_x_coords = torchmesh.x_coords[mask]
        new_onehot = torchmesh.onehot_list[0][mask]
        # new_kinems = torchmesh.kinematics_list[-1][mask]

        return new_faces, new_edge_index, new_x, new_x_pos, new_x_phy, new_x_batch, new_x_coords, new_onehot#, new_kinems
    
    def clean_onedim_isolated_vertices(self, torchmesh, new_edge_index):
        """Clean isolated vertices and reorder the indices. 

        Args:
            torchmesh: mesh data.
            new_edge_index: edge_index with original vertex indices.

        Returns:
            new_edge_index: reindexed edge index.
            new_x: reordered vertices.
        """
        device = torchmesh.x.device
        num_nodes = torchmesh.x.shape[0]
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        edge_index = new_edge_index
        mask[edge_index.flatten()] = 1

        assoc = torch.full((num_nodes, ), -1, dtype=torch.long, device=device)
        assoc[mask] = torch.arange(mask.sum(), device=device)

        new_edge_index = assoc[edge_index]
        new_x = torchmesh.x[mask]
        new_x_phys = torchmesh.x_phy[mask]
        new_x_batch = torchmesh.batch[mask]
        new_x_pos = torchmesh.x_pos[mask]

        return new_edge_index, new_x, new_x_phys,new_x_batch, new_x_pos
    
    def face_adjustment(self, newly_generated_faces, mapping):
        """Clean isolated vertices and reorder the indices. 
        
        Args:
            newly_generated_faces: raw_faces.
            mapping: correspondence between target and source of nodes to be merged. 
             
        Returns:
            newly_generated_faces: indices of new faces.
        """
        a = newly_generated_faces.flatten()
        b = mapping[0, :]
        temp_index = ((b - a.unsqueeze(0).T) == 0).nonzero().T

        newly_generated_faces.flatten()[temp_index[0, :]] = mapping[1, temp_index[1, :]]

        return newly_generated_faces    

    
    def coarsen(self, torchmesh):
        """ Perform coarsen action on mesh according to sampled edges.
        
        Args:
            torchmesh: mesh data.
            
        Returns:
            torchmesh: mesh data with coarsened vertices, edges, and faces.
        """
        if torchmesh.xfaces.shape[0] == 0:
            collapsed_edges, merging_nodes = self.choose_onedim_coarsen_edges(torchmesh)
            if collapsed_edges.shape[1] == 0:
                return torchmesh
            new_edge_index, _ = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
            new_edge_index, new_x = self.clean_onedim_isolated_vertices(torchmesh, new_edge_index)
            if not (new_x.shape[0] - int(new_edge_index.shape[1]/2)) == torchmesh.batch.max()+1:
                print("Coarsen is invalid")
                return torchmesh
            
            torchmesh.edge_index = new_edge_index
            torchmesh.x = new_x
            return torchmesh
        
        else:
            collapsed_edges, merging_nodes = self.choose_coarsen_edges(torchmesh)
            if collapsed_edges.shape[1] == 0:
                return torchmesh
            
            temp_new_edge_index, mapping = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
            temp_newly_generated_faces = self.delete_adjacent_faces(torchmesh, mapping)
            newly_generated_faces = self.face_adjustment(temp_newly_generated_faces, mapping)
            new_faces, new_edge_index, new_x = self.clean_isolated_vertices(torchmesh, temp_new_edge_index, newly_generated_faces)       
            # pdb.set_trace()

            if not (new_x.shape[0] + new_faces.shape[1] - int(new_edge_index.shape[1]/2)) == torchmesh.batch.max()+1:
                print("Coarsen is invalid")
                pdb.set_trace()
                collapsed_edges, merging_nodes = self.choose_coarsen_edges(torchmesh)
                pdb.set_trace()
                new_edge_index, mapping = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
                pdb.set_trace()
                newly_generated_faces = self.delete_adjacent_faces(torchmesh, mapping)
                pdb.set_trace()
                newly_generated_faces = self.face_adjustment(newly_generated_faces, mapping)
                pdb.set_trace()
                new_faces, new_edge_index, new_x = self.clean_isolated_vertices(torchmesh, new_edge_index, newly_generated_faces)       

                return torchmesh

        torchmesh.x = new_x
        torchmesh.edge_index = new_edge_index
        torchmesh.xfaces = new_faces
        
        return torchmesh
    
    def flip(self, flip_edges, outmesh):
        """ Perform flip action on mesh according to sampled edges.
        
        Args:
            torchmesh: mesh data.
            
        Returns:
            torchmesh: mesh data with flipped edges, and faces.
        """
        if flip_edges.shape[0] == 0:
            return outmesh
        remained_faces, flipped_new_faces, remained_edge_index, all_flipped_edge_index = self.generate_flipped_edges(flip_edges, outmesh)
        new_xfaces = torch.cat([remained_faces, flipped_new_faces], dim=-1)
        
        # Sanity check
        temp_edges, _ = torch.cat([new_xfaces[[0,1],:], new_xfaces[[1,2], :], new_xfaces[[2,0], :]], dim=-1).sort(dim=0)
        _, counts = torch.unique(temp_edges, dim=1, return_counts=True)
        if len(set(counts.tolist())) >= 3:
            # print("ignore irregular mesh")
            return outmesh
        
        outmesh.xfaces = new_xfaces    
        outmesh.edge_index = torch.cat([remained_edge_index, all_flipped_edge_index], dim=-1)
        # print("prosessed normally")
        
        if not (outmesh.x.shape[0] + outmesh.xfaces.shape[1] - int(outmesh.edge_index.shape[1]/2)) == outmesh.batch.max()+1:
            print("Flip is invalid")
            return None

        return outmesh


    @property
    def model_dict(self):
        model_dict = super().model_dict
        model_dict["type"] = "GNNRemesherPolicy"
        return model_dict

    
class GNNPolicySizing(GNNRemesherPolicy):
    """
    perform GNN on the current state and generate new mesh
    """
    def __init__(
        self,
        input_size,
        output_size,
        sizing_field_dim,
        nmax,
        edge_dim,
        latent_dim=32,
        num_steps=3,
        layer_norm=False,
        batch_norm=False,
        edge_attr=False,
        edge_threshold=0.,
        skip_split=False,
        skip_flip=False,
        act_name='relu',
        is_split_test=False,
        is_single_action=False,
        dataset=None,
        is_flip_test=False,
        is_coarsen_test=False,
        skip_coarse=False,
        samplemode="random",
        rescale=1000,
        min_edge_size=0.001,
        batch_size=16,
        *args,
        **kwargs
    ):
        nn.Module.__init__(self)
        """
        input_size: input node feature dim
        edge_dim: input edge feature dim
        output_dim: output node feature dim - sizing field dim
        latent_dim: hidden feature size
        num_steps: number of message passing steps
        act_name: activation used

        Note that self.remeshing_forward_GNN:
            input - data: pyg graph
            output - outmesh:
                        outmesh.x = physics parameter interpolated on the new mesh
                        outmesh.index = index of the node_feature when we do the plotting
                                        as newly added node are appended to the node feature list
                                        rather than ordered by their grid location
                     prob = a dict of probably of action
                           e.g. {["flip_edge_remesh_prob",flip_edge_remesh_prob],}
        """
        self.input_size = input_size
        self.output_size = output_size
        self.sizing_field_dim = sizing_field_dim
        self.edge_dim = edge_dim 
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.act_name = act_name
        self.layer_norm = layer_norm
        self.samplemode = samplemode
        self.batch_norm = batch_norm
        self.edge_attr = edge_attr
        self.edge_threshold = edge_threshold
        self.is_flip_test = is_flip_test
        self.dataset = dataset
        self.var = 0
        self.skip_split = skip_split
        self.skip_flip = skip_flip
        self.is_flip_test = is_flip_test
        self.is_split_test = is_split_test
        self.rescale = rescale
        self.skip_coarse = skip_coarse
        self.min_edge_size = min_edge_size
        self.batch_size = batch_size
        self.nmax = nmax.type(torch.int64) if isinstance(nmax, torch.Tensor) else torch.tensor(nmax, dtype=torch.int64)
        self.is_coarsen_test = is_coarsen_test
        self.is_single_action = is_single_action
        
        self.encoder_edge = FCBlock(in_features=edge_dim,
                                    out_features=latent_dim,
                                    num_hidden_layers=2,
                                    hidden_features=latent_dim,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=layer_norm,
                                   )
        self.encoder_nodes = FCBlock(in_features=input_size,
                                     out_features=latent_dim,
                                     num_hidden_layers=2,
                                     hidden_features=latent_dim,
                                     outermost_linear=True,
                                     nonlinearity=act_name,
                                     layer_norm=layer_norm,
                                    )

        self.processors = []
        edge_latent = self.edge_dim if not self.edge_attr else latent_dim 
        for _ in range(num_steps):
            self.processors.append((processor(latent_dim, 
                                              latent_dim, 
                                              layer_norm=layer_norm, 
                                              act_name=act_name, 
                                              edge_attr=edge_attr,
                                             )))
            if self.batch_norm:
                self.processors.append(torch.nn.BatchNorm1d(edge_latent))
                self.processors.append(torch.nn.BatchNorm1d(edge_latent))
        self.processors = torch.nn.Sequential(*self.processors)          
        
        self.decoder_node = FCBlock(in_features=latent_dim,
                                    out_features=output_size,
                                    num_hidden_layers=3,
                                    hidden_features=latent_dim,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=False,
                                   )
        
        self.pdist = torch.nn.PairwiseDistance(p=2)
    
    def get_sizing_field(self, graph, **kwargs):
        if len(dict(to_tuple_shape(graph.original_shape))["n0"]) == 0:
            graph_clone = graph.copy()
        else:
            graph_clone = graph.clone()
        
        for par in (self.encoder_edge.parameters()):
            if torch.isnan(par).any():
                print(par)
                pdb.set_trace()
                
        if torch.isnan(graph.x).any():
                print(graph.x)
                pdb.set_trace()
        graph_evolved = self.encoder(graph)
        if torch.isnan(graph_evolved.x).any():
                print(graph_evolved.x)
                pdb.set_trace()
        for i in range(self.num_steps):
            if self.batch_norm:
                graph_evolved = self.processors[i*3](graph_evolved)
                graph_evolved.x = self.processors[i*3+1](graph_evolved.x)
                graph_evolved.edge_attr = self.processors[i*3+2](graph_evolved.edge_attr)
            else:
                graph_evolved = self.processors[i](graph_evolved)
        
        if torch.isnan(graph_evolved.x).any():
                print(graph_evolved.x)
                pdb.set_trace()
        graph_evolved = self.decoder(graph_evolved)
        if torch.isnan(graph_evolved.x).any():
                print(graph_evolved.x)
                pdb.set_trace()
                
        if self.output_size==3:
            L = matrix_diag_transform(fill_triangular(graph_evolved.x,2),torch.functional.F.softplus)
            graph_evolved.x = (L@(L.transpose(1,2))).reshape([-1,4])
        elif self.output_size==1:
            a = torch.functional.F.softplus(graph_evolved.x)
            if torch.isnan(a).any():
                print(a)
                pdb.set_trace()
            graph_evolved.x = a
            graph_evolved.x = graph_evolved.x/0.0016/0.0016/self.rescale
            
        return graph_evolved
    
    def choose_split_edge_GNN(self, outmesh):
        """ Choose splittable edges.
        
        Args:
            outmesh: mesh data.
            outmesh.x: [position of nodes, sizing field,]
            
        Returns:
            split_edges: edges to be splitten with shape [(number of edges to be splitted), 2].
        """      
        # sample maximal independent splittable edges
        if outmesh.xfaces.shape[0] == 0:
            values, _ = outmesh.edge_index.sort(dim=0)
            tor_samnodes = values.unique(dim=-1).T
        else:
            sampled_nodes = self._sample_independent_edges(outmesh)
            if len(sampled_nodes) == 0:
                return torch.tensor([]).reshape(0,2)
            tor_samnodes =  torch.tensor(sampled_nodes, dtype=torch.int64, device=outmesh.x.device)
        receivers = outmesh.x_pos[tor_samnodes[:, 0], :]
        senders = outmesh.x_pos[tor_samnodes[:, 1], :]
        edge_size = torch.sqrt(((receivers - senders)**2).mean(dim=-1))
    
        mask = edge_size>self.min_edge_size*2
        splitable_edge = torch.where(mask==1)[0]
        tor_samnodes = tor_samnodes[splitable_edge]
        if tor_samnodes.shape[0] == 0:
            return torch.tensor([]).reshape(0,2),[[None]]
        # pdb.set_trace()
        # compute edge feature
        if outmesh.xfaces.shape[0] == 0:
            # if is_pos=True
            # Need to fix 
            receivers = outmesh.x_pos[tor_samnodes[:, 0], :]
            senders = outmesh.x_pos[tor_samnodes[:, 1], :]
        else:
            receivers = outmesh.x[tor_samnodes[:, 0], :2]
            senders = outmesh.x[tor_samnodes[:, 1], :2]
        edge_features = receivers - senders
        # compute error estimator
        sizing_fields = self._compute_sizing_fields(outmesh, tor_samnodes) #shape: [sampled_edges, 2, 2] or [sampled_edge, 1] 
        # compute error estimator
        if outmesh.xfaces.shape[0] == 0:
            half = sizing_fields * edge_features #shape: [sampled_edges, 1]
            estimators = edge_features * half
        else:
            half = torch.matmul(sizing_fields, edge_features.reshape(-1,2,1)) #shape: [sampled_edges, 2]
            estimators = torch.matmul(edge_features.reshape(-1,1,2), half).flatten()
        # compute probability on sampled splittable edges
        # pdb.set_trace()
        
        shape = (estimators.shape)
        # print(estimators)
        if self.is_single_action:
            # remesh_prob - -1 if not selected, 1 if selected
            remesh_prob_softmax, remesh_prob = get_remesh_prob_softmax(estimators, outmesh.batch[tor_samnodes[:,0]],self.current_batch) # shape[#nodes,1]
            remesh_prob = remesh_prob.reshape(shape)
            remesh_prob_softmax = remesh_prob_softmax.reshape(shape)
        else:
            remesh_prob = torch.sigmoid(estimators - 1.) #.to(outmesh.x.device)
            remesh_prob_softmax = remesh_prob
        
        if torch.isnan(remesh_prob).any():
            print(remesh_prob)
            pdb.set_trace()
            
        if self.training:
            mask = torch.rand_like(remesh_prob) < remesh_prob #[num,1]
        else:
            if self.is_split_test:
                mask = remesh_prob > 0.1
            else:
                mask = remesh_prob > 0.5
        remesh_prob = remesh_prob*(mask.float()) + (1-mask.float())*(1-remesh_prob)
        
        mask = mask.reshape((mask.shape[0], 1))
        # pdb.set_trace()
        split_edges = torch.masked_select(tor_samnodes, mask).reshape(-1,2)        
        assert (outmesh.batch[tor_samnodes[:,0]]!=outmesh.batch[tor_samnodes[:,1]]).sum()<1e-4
        
        return split_edges,[remesh_prob.flatten(),tor_samnodes,outmesh.batch[tor_samnodes[:,0]],remesh_prob_softmax.flatten()]
        
    def choose_flippable_edge_GNN(self, outmesh):
        """ Choose flippable edges.

        Args:
            outmesh: mesh data.
            sizing_fields: gnn predicted sizing_fields
            
        Returns:
            flip_edges: edges to be splitten with shape [(number of edges to be flipped), 2].
        """      
        sampled_nodes, new_dnode_face_dict, neighbor_vertices = self._sample_flippable_edges(outmesh)
        if len(sampled_nodes) == 0:
            return torch.tensor([]).reshape(0,2),[[None]]
        delaunay_mask = self.gen_delaunay_mask(outmesh, sampled_nodes, new_dnode_face_dict, neighbor_vertices)
        tor_samnodes = torch.tensor(sampled_nodes, device=outmesh.x.device)[delaunay_mask, :]
        if tor_samnodes.shape[0] == 0:
            return torch.tensor([]).reshape(0,2),[[None]]
          
        # compute edge feature
        if outmesh.xfaces.shape[0] == 0:
            # if is_pos=True
            # Need to fix 
            receivers = outmesh.x_pos[tor_samnodes[:, 0], :]
            senders = outmesh.x_pos[tor_samnodes[:, 1], :]
        else:
            receivers = outmesh.x[tor_samnodes[:, 0], :2]
            senders = outmesh.x[tor_samnodes[:, 1], :2]
        edge_features = receivers - senders
        
        # compute error estimator
        sizing_fields = self._compute_sizing_fields(outmesh, tor_samnodes) #shape: [sampled_edges, 2, 2] or [sampled_edge, 1] 

        # compute error estimator
        if outmesh.xfaces.shape[0] == 0:
            half = torch.matmul(sizing_fields, edge_features.reshape(-1,1,1)) #shape: [sampled_edges, 2]
            estimators = torch.matmul(edge_features.reshape(-1,1,1), half).flatten()
        else:
            half = torch.matmul(sizing_fields, edge_features.reshape(-1,2,1)) #shape: [sampled_edges, 2]
            estimators = torch.matmul(edge_features.reshape(-1,1,2), half).flatten()

        # compute probability on sampled splittable edges
        
        shape = (estimators.shape)
        if self.is_single_action:
            # remesh_prob - -1 if not selected, 1 if selected
            remesh_prob_softmax, remesh_prob = get_remesh_prob_softmax(estimators, outmesh.batch[tor_samnodes[:,0]],self.current_batch) # shape[#nodes,1]
            remesh_prob = remesh_prob.reshape(shape)
            remesh_prob_softmax = remesh_prob_softmax.reshape(shape)
        else:
            remesh_prob = torch.sigmoid(estimators - 1.) #.to(outmesh.x.device)
            remesh_prob_softmax = remesh_prob
            
        remesh_prob = remesh_prob*(mask.float()) + (1-mask.float())*(1-remesh_prob)
       

        if self.training:
            if self.is_split_test:
                mask = torch.rand_like(remesh_prob) < remesh_prob
            else:
                mask = torch.rand_like(remesh_prob) < remesh_prob
        else:
            if self.is_split_test:
                mask = remesh_prob > 0.1
            else:
                mask = remesh_prob > 0.5
        mask = mask.reshape((mask.shape[0], 1))
        flip_edges = torch.masked_select(tor_samnodes, mask).reshape(-1,2)
        
        return flip_edges,[remesh_prob.flatten(),tor_samnodes,outmesh.batch[tor_samnodes[:,0]],remesh_prob_softmax.flatten()]

    def choose_onedim_coarsen_edges_GNN(self, torchmesh):
        """ Choose splittable edges.

        Args:
            torchmesh: mesh data.

        Returns:
            split_edges: edges to be coarsened with shape [(number of edges to be coarsened), 2].
        """      
        
        coarsened_nodes, snodes_edges_tensor = self._sample_onedim_coarsened_nodes(torchmesh)
        if len(coarsened_nodes) == 0:
            collapsed_edges = torch.tensor([[],[]])
            merging_nodes = torch.tensor([[],[]])
            return collapsed_edges, merging_nodes,[[None]]

        # snodes_edges_tensor = torch.tensor(snodes_edges_list, device=torchmesh.x.device)
        receivers = torchmesh.x_pos[snodes_edges_tensor[..., 0]]
        senders = torchmesh.x_pos[snodes_edges_tensor[..., 1]]
        
        edge_features = (receivers - senders)[..., :]

        # compute error estimator
        sizing_fields = self._compute_coarsening_sizing_fields(torchmesh, snodes_edges_tensor)

        
        final_mul = edge_features * sizing_fields * edge_features #[num nodes,2,1]

        # estimators = final_mul.reshape(2, -1)
        estimators = final_mul[...,0].T #[2,num nodes]
        # pdb.set_trace()
        probs = (estimators - 1)
        indices = torch.argmax(probs, dim=0) #[1,num_nodes]
        max_probs = torch.gather(probs, 0, indices.reshape(1,-1))
        shape = (max_probs.shape)
        if self.is_single_action:
            # remesh_prob - -1 if not selected, 1 if selected
            max_probs_softmax, max_probs = get_remesh_prob_softmax(max_probs, torchmesh.batch[coarsened_nodes],self.current_batch) # shape[#nodes,1]
            max_probs = max_probs.reshape(shape)
            max_probs_softmax = max_probs_softmax.reshape(shape)
        else:
            max_probs = torch.sigmoid(max_probs)
            max_probs_softmax = max_probs
            
        if self.training:
            coarsening_mask = torch.rand_like(max_probs)<max_probs
        else:
            coarsening_mask = max_probs > 0.5
        # torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)
        max_probs = max_probs*(coarsening_mask.float()) + (1-max_probs)*(1-coarsening_mask.float())
       
        highest_prob_edges = torch.gather(snodes_edges_tensor, 1, torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)).reshape(indices.shape[0],2).T

        collapsed_edges = highest_prob_edges[:,coarsening_mask.reshape(coarsening_mask.shape[1])]

        merging_nodes = torch.tensor(coarsened_nodes, device=torchmesh.x.device)[coarsening_mask[0,:]]
        assert ((torchmesh.batch[coarsened_nodes]!=torchmesh.batch[torch.tensor(snodes_edges_list)[:,0]][:,0]).sum())<1e-5
        assert ((torchmesh.batch[coarsened_nodes]!=torchmesh.batch[torch.tensor(snodes_edges_list)[:,1]][:,0]).sum())<1e-5
        return collapsed_edges, merging_nodes, [max_probs.flatten(),coarsened_nodes,torchmesh.batch[coarsened_nodes],max_probs_softmax.flatten()]
    
    
    def choose_coarsen_edges_GNN(self, torchmesh):
        """ Choose splittable edges.
        
        Args:
            torchmesh: mesh data.
            
        Returns:
            split_edges: edges to be coarsened with shape [(number of edges to be coarsened), 2].
        """      
        coarsened_nodes, snodes_edges_list = self._sample_coarsened_nodes(torchmesh)
        if len(coarsened_nodes) == 0:
            collapsed_edges = torch.tensor([[],[]])
            merging_nodes = torch.tensor([[],[]])
            return collapsed_edges, merging_nodes,[[None]]
        
        snodes_edges_tensor = torch.tensor(snodes_edges_list, device=torchmesh.x.device)

        receivers = torchmesh.x_pos[snodes_edges_tensor[..., 0]]
        senders = torchmesh.x_pos[snodes_edges_tensor[..., 1]]
        edge_features = (receivers - senders)[..., :2]
        
        sizing_fields = self._compute_coarsening_sizing_fields(torchmesh, snodes_edges_tensor)

        resh_edge_features = edge_features.reshape(-1,1,2)
        batched_sizing_fields = sizing_fields
        
        left_mul = torch.matmul(resh_edge_features, batched_sizing_fields)
        final_mul = torch.matmul(left_mul, resh_edge_features.reshape(-1, 2, 1))
        
        estimators = final_mul.reshape(-1,4).permute(1,0)
        probs = (estimators - 1)
        indices = torch.argmax(probs, dim=0)
    
        max_probs = torch.gather(probs, 0, indices.reshape(1,-1))
        # pdb.set_trace()
        shape = max_probs.shape
        if self.is_single_action:
            # remesh_prob - -1 if not selected, 1 if selected
            max_probs_softmax, max_probs = get_remesh_prob_softmax(max_probs, torchmesh.batch[coarsened_nodes],self.current_batch) # shape[#nodes,1]
            max_probs = max_probs.reshape(shape)
            max_probs_softmax = max_probs_softmax.reshape(shape)
        else:
            max_probs =  torch.sigmoid(max_probs)
            max_probs_softmax = max_probs
            
            
        if self.training:
            coarsening_mask = torch.rand_like(max_probs)<max_probs
        else:
            coarsening_mask = max_probs > 0.5
        max_probs = max_probs*(coarsening_mask.float()) + (1-max_probs)*(1-coarsening_mask.float())
       
    
        torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)

        highest_prob_edges = torch.gather(snodes_edges_tensor, 1, torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)).reshape(indices.shape[0],2).T

        collapsed_edges = highest_prob_edges[:,coarsening_mask.reshape(coarsening_mask.shape[1])]

        merging_nodes = torch.tensor(coarsened_nodes, device=torchmesh.x.device)[coarsening_mask[0,:]]
        return collapsed_edges, merging_nodes, [max_probs.flatten(),coarsened_nodes,torchmesh.batch[coarsened_nodes],max_probs_softmax.flatten()]


    def coarsen_GNN(self, torchmesh):
        """ Perform coarsen action on mesh according to sampled edges.

        Args:
            torchmesh: mesh data.

        Returns:
            torchmesh: mesh data with coarsened vertices, edges, and faces.
        """
        if torchmesh.xfaces.shape[0] == 0:
            collapsed_edges, merging_nodes, max_probs= self.choose_onedim_coarsen_edges_GNN(torchmesh)
            if collapsed_edges.shape[1] == 0:
                return torchmesh, max_probs
            new_edge_index, _ = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
            new_edge_index, new_x, new_x_phys, new_x_batch, new_x_pos = self.clean_onedim_isolated_vertices(torchmesh, new_edge_index)
            if not (new_x.shape[0] - int(new_edge_index.shape[1]/2)) == torchmesh.batch.max()+1:
                print("Coarsen is invalid")
                return torchmesh

            torchmesh.edge_index = new_edge_index
            torchmesh.x = new_x
            torchmesh.x_phy = new_x_phys
            torchmesh.batch = new_x_batch
            torchmesh.x_pos = new_x_pos
            return torchmesh, max_probs
        
        else:
            old_batch = torchmesh.batch.clone()
            collapsed_edges, merging_nodes, max_probs = self.choose_coarsen_edges_GNN(torchmesh)
            if collapsed_edges.shape[1] == 0:
                return torchmesh, max_probs

            temp_new_edge_index, mapping = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
            temp_newly_generated_faces = self.delete_adjacent_faces(torchmesh, mapping)
            newly_generated_faces = self.face_adjustment(temp_newly_generated_faces, mapping)
            # new_faces, new_edge_index, new_x, new_x_pos, new_x_phy, new_x_batch, new_x_coords = self.clean_isolated_vertices(torchmesh, temp_new_edge_index, newly_generated_faces)       
            new_faces, new_edge_index, new_x, new_x_pos, new_x_phy, new_x_batch, new_x_coords, new_onehot = self.clean_isolated_vertices(torchmesh, temp_new_edge_index, newly_generated_faces)       
            # pdb.set_trace()

            if not (new_x.shape[0] + new_faces.shape[1] - int(new_edge_index.shape[1]/2)) == torchmesh.batch.max()+1:
                print("Coarsen is invalid")
                pdb.set_trace()
                collapsed_edges, merging_nodes = self.choose_coarsen_edges_GNN(torchmesh)
                pdb.set_trace()
                new_edge_index, mapping = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
                pdb.set_trace()
                newly_generated_faces = self.delete_adjacent_faces(torchmesh, mapping)
                pdb.set_trace()
                newly_generated_faces = self.face_adjustment(newly_generated_faces, mapping)
                pdb.set_trace()
                new_faces, new_edge_index, new_x = self.clean_isolated_vertices(torchmesh, new_edge_index, newly_generated_faces)       

                return torchmesh

        torchmesh.x = new_x
        torchmesh.x_phy = new_x_phy
        torchmesh.x_coords = new_x_coords
        torchmesh.batch_history = old_batch
        torchmesh.batch = new_x_batch
        torchmesh.x_pos = new_x_pos
        torchmesh.edge_index = new_edge_index
        torchmesh.xfaces = new_faces
        temp_onehots = list(torchmesh.onehot_list)
        temp_onehots[0] = new_onehot
        torchmesh.onehot_list = tuple(temp_onehots)
        # torchmesh.kinematics_list[-1] = new_kinems

        return torchmesh, max_probs


    def act(self, *args, **kwargs):
        """Policy's main function. Given the state (PyG data), returns the action probability and the resulting mesh after the action."""
        return self.remeshing_forward_GNN(*args, **kwargs)


    def remeshing_forward_GNN(self, torchmesh, use_pos=False, **kwarg):
        """ Perform remeshing action on mesh data.

        Args:
            torchmesh: pyg data instance for mesh.

        Returns:
            outmesh: mesh data after gnn forwarding and remeshing.
            logsigmoid_sum: {"action": logprob of shape [batch, 1]}
            entropy_sum: {"action": entropy of shape [batch, 1]}
            remesh_prob: {"action": [prob, node_index , batch_index]}
        """
        # Reformat the data:
        if hasattr(torchmesh, "node_feature"):            
            if len(dict(to_tuple_shape(torchmesh.original_shape))["n0"]) == 0:
                batch = torchmesh.batch['n0']
                x_coords = torchmesh['history']['n0'][1] #
                torchmesh = attrdict_to_pygdict(torchmesh, is_flatten=True, use_pos=use_pos)
            else:
                batch = torchmesh.batch.clone()
                torchmesh = deepsnap_to_pyg(torchmesh, is_flatten=True, use_pos=use_pos)
        elif hasattr(torchmesh, "x"):
            batch = torchmesh.batch.clone()
            x_coords = torchmesh.history[-1]
        self.current_batch = int(batch.max()+1)
        self.device = batch.device
        
        
        if torchmesh.xfaces.shape[0] == 0:
            self.skip_flip = True
        old_xfaces = torchmesh.xfaces.clone()
        old_batch = batch.clone()
        old_x_coords = x_coords.clone()
        torchmesh.batch = batch
        x_phy = torchmesh.x.clone() #torchmesh.x = 6*1 : [x_mesh, y_mesh, z_mesh=0,x_world, y_world, z_world]
        torchmesh.x_phy = x_phy
        
        torchmesh = self.get_sizing_field(torchmesh) #torchmesh.x - predicted sizing field
        if self.output_size==4 or self.output_size==3:
            torchmesh.x_coords = x_coords
            torchmesh.x = torch.cat([x_coords,torchmesh.x],dim=-1)
        
        remesh_prob = {}
        
        # Split edges of mesh
        if self.skip_split:
            split_outmesh = torchmesh
            remesh_prob['split_edge_remesh_prob']=[[None]]
        else:    
            if self.is_split_test:
                split_edges, split_edge_remesh_prob = self.choose_split_edge_GNN(torchmesh)
                # print(split_edges)
                # print(split_edge_remesh_prob[0])
                # if torch.isnan(split_edge_remesh_prob[0]).any():
                #     pdb.set_trace()
                remesh_prob['split_edge_remesh_prob'] = split_edge_remesh_prob
                if split_edges.shape[0] == 0:
                    outmesh = torchmesh
                else:
                    outmesh = self.split(split_edges, torchmesh)
                return outmesh
            else:
                split_edges, split_edge_remesh_prob = self.choose_split_edge_GNN(torchmesh)
                remesh_prob['split_edge_remesh_prob'] = split_edge_remesh_prob
                split_outmesh = self.split(split_edges, torchmesh)

        # Flip edges of mesh
        if self.skip_flip:
            flipped_outmesh = split_outmesh
            remesh_prob['flip_edge_remesh_prob']=[[None]]
        else:
            if self.is_flip_test:
                split_outmesh = torchmesh
                flip_edges, flip_edge_remesh_prob = self.choose_flippable_edge_GNN(split_outmesh)
                remesh_prob['flip_edge_remesh_prob'] = flip_edge_remesh_prob
                outmesh = self.flip(flip_edges, split_outmesh)
                return outmesh
            else:
                flip_edges, flip_edge_remesh_prob = self.choose_flippable_edge_GNN(split_outmesh)
                remesh_prob['flip_edge_remesh_prob'] = flip_edge_remesh_prob
                flipped_outmesh = self.flip(flip_edges, split_outmesh)
                
        # Coarsen edges of mesh
        if self.skip_coarse:
            remesh_prob['coarse_prob']=[[None]]
            outmesh = flipped_outmesh
            index = torch.sort(outmesh.batch)[1]
            a = torch.cat([outmesh.x_pos,outmesh.batch.unsqueeze(-1)],dim=-1)
            outmesh.index = torch_lexsort(a.permute(1,0))
            if self.dataset.startswith("arcsi"):
                outmesh.history = (old_x_coords, outmesh.x_coords)
            outmesh.x = outmesh.x_phy[:,:]
            #reset bdd nodes, all interpolate boundary node are removed

            if self.dataset.startswith("mpp"):
                outmesh.dataset = to_tuple_shape(outmesh.dataset)
                outmesh.x[:,-1] = 1*(outmesh.x[:,-1]==1)
                outmesh.x_bdd = outmesh.x[:,-1:]
                outmesh = update_edge_attr_1d(outmesh)
            logsigmoid_sum, entropy_sum = self.get_batch_logsigmoid_sum(remesh_prob,self.current_batch,self.is_single_action)
            return outmesh, logsigmoid_sum, entropy_sum, remesh_prob
        else:
            if self.is_coarsen_test:
                flipped_outmesh = torchmesh
                outmesh, max_probs = self.coarsen_GNN(flipped_outmesh)
                remesh_prob['coarse_prob'] = max_probs
                if self.dataset.startswith("mpp"):
                    outmesh.dataset = to_tuple_shape(outmesh.dataset)
                    outmesh.x[:,-1] = 1*(outmesh.x[:,-1]==1)
                    outmesh.x_bdd = outmesh.x[:,-1:]
                    outmesh = update_edge_attr_1d(outmesh)
                if self.dataset.startswith("arcsi"):
                    outmesh.history = (old_x_coords,outmesh.x_coords)   
                logsigmoid_sum, entropy_sum = self.get_batch_logsigmoid_sum(remesh_prob,self.current_batch,self.is_single_action)
                return outmesh, logsigmoid_sum, entropy_sum, remesh_prob

            else:
                outmesh, max_probs = self.coarsen_GNN(flipped_outmesh)
                remesh_prob['coarse_prob'] = max_probs
                # index = torch.sort(outmesh.x_pos,0)[1][:,0]
                index = torch.sort(outmesh.batch)[1]
                a = torch.cat([outmesh.x_pos,outmesh.batch.unsqueeze(-1)],dim=-1)
                outmesh.index = torch_lexsort(a.permute(1,0))
                if self.dataset.startswith("arcsi"):
                    outmesh.history = (old_x_coords,outmesh.x_coords)
                    outmesh.xface_list = (old_xfaces.T, outmesh.xfaces.T)
                    outmesh.batch_history = old_batch
                outmesh.x = outmesh.x_phy[:,:]
                #reset bdd nodes, all interpolate boundary node are removed

                if self.dataset.startswith("mpp"):
                    outmesh.dataset = to_tuple_shape(outmesh.dataset)
                    outmesh.x[:,-1] = 1*(outmesh.x[:,-1]==1)
                    outmesh.x_bdd = outmesh.x[:,-1:]
                    outmesh = update_edge_attr_1d(outmesh)
                logsigmoid_sum, entropy_sum = self.get_batch_logsigmoid_sum(remesh_prob,self.current_batch,self.is_single_action)
                return outmesh, logsigmoid_sum, entropy_sum, remesh_prob
    
    def get_batch_logsigmoid_sum(self,remesh_prob,batch_size,single=False):
        logsigmoid_sum = {}
        entropy_sum = {}
        for key in remesh_prob.keys():
            if len(remesh_prob[key][0])==0:
                logsigmoid_sum[key] = torch.zeros([self.current_batch],device=self.device)
                entropy_sum[key] = torch.zeros([self.current_batch],device=self.device)
                continue
            if remesh_prob[key][0][0]!=None:
                if single:
                    prob = remesh_prob[key][3]
                else:
                    prob = remesh_prob[key][0]
                index = remesh_prob[key][2]
                logsigmoid_sum[key] = batch_logsigmoid_sum(prob,index,batch_size)
                entropy_sum[key] = batch_entropy_sum(prob=prob, index=index, batch_size=batch_size, single=single)
            else:
                logsigmoid_sum[key] = torch.zeros([self.current_batch],device=self.device)
                entropy_sum[key] = torch.zeros([self.current_batch],device=self.device)
        return logsigmoid_sum, entropy_sum
    
    
    
    @property
    def model_dict(self):
        model_dict = {"type": "GNNPolicySizing"}
        model_dict["input_size"] = self.input_size
        model_dict["output_size"] = self.output_size
        model_dict["sizing_field_dim"] = self.sizing_field_dim
        model_dict["nmax"] = to_np_array(self.nmax)
        model_dict["edge_dim"] = self.edge_dim
        model_dict["dataset"] = self.dataset
        model_dict["latent_dim"] = self.latent_dim
        model_dict["num_steps"] = self.num_steps
        model_dict["layer_norm"] = self.layer_norm
        model_dict["batch_norm"] = self.batch_norm
        model_dict["edge_attr"] = self.edge_attr
        model_dict["edge_threshold"] = self.edge_threshold
        model_dict["skip_split"] = self.skip_split
        model_dict["skip_flip"] = self.skip_flip
        model_dict["skip_coarse"] = self.skip_coarse
        model_dict["act_name"] = self.act_name
        model_dict["is_split_test"] = self.is_split_test
        model_dict["is_flip_test"] = self.is_flip_test
        model_dict["is_coarsen_test"] = self.is_coarsen_test
        model_dict["samplemode"] = self.samplemode
        model_dict["rescale"] = self.rescale
        model_dict["min_edge_size"] = self.min_edge_size
        model_dict["batch_size"] = self.batch_size
        model_dict["is_single_action"] = self.is_single_action 
        model_dict["state_dict"] = to_cpu(self.state_dict())
        return model_dict


class GNNPolicyAgent(GNNRemesherPolicy):
    """
    perform GNN on the current state and generate new mesh
    """
    def __init__(
        self,
        input_size,
        nmax,
        edge_dim,
        latent_dim=32,
        num_steps=3,
        layer_norm=False,
        batch_norm=False,
        edge_attr=False,
        edge_threshold=0.,
        skip_split=False,
        skip_flip=False,
        act_name='relu',
        dataset=None,
        is_split_test=False,
        is_flip_test=False,
        is_coarsen_test=False,
        samplemode="random",
        skip_coarse=False,
        min_edge_size=0.001,
        rescale=10000,
        batch_size=16,
        is_single_action=False,
        top_k_action=1,
        max_select=1000,
        reward_condition=False,
        offset_split=0.,
        offset_coarse=0.,
        *args,
        **kwargs
    ):
        nn.Module.__init__(self)
        """
        input_size: input node feature dim
        edge_dim: input edge feature dim
        output_dim: output node feature dim - sizing field dim
        latent_dim: hidden feature size
        num_steps: number of message passing steps
        act_name: activation used

        Note that self.remeshing_forward_GNN:
            input - data: pyg graph
            output - outmesh:
                        outmesh.x = physics parameter interpolated on the new mesh
                        outmesh.index = index of the node_feature when we do the plotting
                                        as newly added node are appended to the node feature list
                                        rather than ordered by their grid location
                     prob = a dict of probably of action
                           e.g. {["flip_edge_remesh_prob",flip_edge_remesh_prob],}
        """
        
        self.input_size = input_size
        output_size = 1
        self.output_size = 1
        self.edge_dim = edge_dim 
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        self.act_name = act_name
        self.layer_norm = layer_norm
        self.samplemode = samplemode
        self.batch_norm = batch_norm
        self.edge_attr = edge_attr
        self.dataset = dataset
        self.edge_threshold = edge_threshold
        self.is_flip_test = is_flip_test
        self.skip_coarse = skip_coarse
        self.min_edge_size = min_edge_size
        self.batch_size = batch_size
        self.is_single_action = is_single_action
        self.top_k_action = top_k_action
        self.max_select = max_select
        self.var = 0
        self.skip_split = skip_split
        self.skip_flip = skip_flip
        self.is_flip_test = is_flip_test
        self.is_split_test = is_split_test
        self.reward_condition = reward_condition
        self.rescale = rescale
        self.offset_split = torch.tensor(offset_split)
        self.offset_coarse = torch.tensor(offset_coarse)
        self.nmax = nmax.type(torch.int64) if isinstance(nmax, torch.Tensor) else torch.tensor(nmax, dtype=torch.int64)
        self.is_coarsen_test = is_coarsen_test
        if self.dataset.startswith("a"):
            import dolfin as dolfin
            self.dolfin = dolfin
            self.input_size = 8
            
        self.encoder_edge = FCBlock(in_features=edge_dim,
                                    out_features=latent_dim,
                                    num_hidden_layers=2,
                                    hidden_features=latent_dim,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=layer_norm,
                                   )
        self.encoder_nodes = FCBlock(in_features=self.input_size + (5 if self.reward_condition else 0),
                                     out_features=latent_dim,
                                     num_hidden_layers=2,
                                     hidden_features=latent_dim,
                                     outermost_linear=True,
                                     nonlinearity=act_name,
                                     layer_norm=layer_norm,
                                    )

        self.processors = []
        for _ in range(num_steps):
            self.processors.append((processor(latent_dim,
                                              latent_dim, 
                                              layer_norm=layer_norm, 
                                              act_name=act_name, 
                                              edge_attr=edge_attr,
                                             )))
            if self.batch_norm:
                self.processors.append(torch.nn.BatchNorm1d(latent_dim))
                self.processors.append(torch.nn.BatchNorm1d(latent_dim))
        self.processors = torch.nn.Sequential(*self.processors)          
        
        if not self.skip_split:
            self.decoder_edge_split = FCBlock(in_features=latent_dim + (5 if self.reward_condition else 0),
                                        out_features=output_size,
                                        num_hidden_layers=3,
                                        hidden_features=latent_dim,
                                        outermost_linear=True,
                                        nonlinearity=act_name,
                                        layer_norm=False,
                                       )
        if not self.skip_flip:
            self.decoder_edge_flip = FCBlock(in_features=latent_dim + (5 if self.reward_condition else 0),
                                        out_features=output_size,
                                        num_hidden_layers=3,
                                        hidden_features=latent_dim,
                                        outermost_linear=True,
                                        nonlinearity=act_name,
                                        layer_norm=False,
                                       )
        self.decoder_edge_coarse = FCBlock(in_features=latent_dim + (5 if self.reward_condition else 0),
                                    out_features=output_size,
                                    num_hidden_layers=3,
                                    hidden_features=latent_dim,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=False,
                                   )
        
        self.pdist = torch.nn.PairwiseDistance(p=2)
    
    def get_edge_latents(self, graph, reward_beta=None, **kwargs):
        if self.reward_condition:
            beta_batch = reward_beta[graph.batch.to(torch.int64)][:,None]
            graph.x = torch.cat([graph.x,beta_batch.repeat([1,5])],dim=-1).float()
        # pdb.set_trace()
        graph_evolved = self.encoder(graph)
        for i in range(self.num_steps):
            if self.batch_norm:
                # if self.reward_condition:
                #     pdb.set_trace()
                #     graph_evolved.x = torch.cat([graph_evolved.x,beta_batch],dim=-1).float() 
                # pdb.set_trace()
                graph_evolved = self.processors[i*3](graph_evolved)
                graph_evolved.x = self.processors[i*3+1](graph_evolved.x)
                graph_evolved.edge_attr = self.processors[i*3+2](graph_evolved.edge_attr)
            else:
                # if self.reward_condition:
                #     graph_evolved.x = torch.cat([graph_evolved.x,beta_batch],dim=-1).float() 
                graph_evolved = self.processors[i](graph_evolved)
      
        return graph_evolved
    
    def get_edge_mask(self,sample, full):
        full = full.unsqueeze(-1).repeat([1,1,sample.shape[0]])
        sample = sample.permute(1,0).unsqueeze(-2).repeat([1,full.shape[1],1])
        mask = (((full[0,:]==sample[0,:]) * (full[1,:]==sample[1,:])  ).sum(dim=-1)>0)*1
        return torch.where(mask==1)[0]
    
    def choose_split_edge_GNN(self, outmesh,reward_beta=None):
        """ Choose splittable edges.
        
        Args:
            outmesh: mesh data.
            outmesh.x: [position of nodes, sizing field,]
            reward_beta: [b,1]
            
        Returns:
            split_edges: edges to be splitten with shape [(number of edges to be splitted), 2].
        """      
        # sample maximal independent splittable edges
        if outmesh.xfaces.shape[0] == 0:
            values, _ = outmesh.edge_index.sort(dim=0)
            tor_samnodes = values.unique(dim=-1).T
        else:
            sampled_nodes = self._sample_independent_edges(outmesh)
            if len(sampled_nodes) == 0:
                return torch.tensor([]).reshape(0,2)
            tor_samnodes =  torch.tensor(sampled_nodes, dtype=torch.int64, device=outmesh.x.device)
        
        receivers = outmesh.x_pos[tor_samnodes[:, 0], :]
        senders = outmesh.x_pos[tor_samnodes[:, 1], :]
        edge_size = torch.sqrt(((receivers - senders)**2).mean(dim=-1))
        mask = edge_size>self.min_edge_size*2
        splitable_edge = torch.where(mask==1)[0]
        tor_samnodes = tor_samnodes[splitable_edge]

        
        if outmesh.xfaces.shape[0] == 0:
            # if is_pos=True
            # Need to fix 
            receivers = outmesh.x[tor_samnodes[:, 0], :]
            senders = outmesh.x[tor_samnodes[:, 1], :]
        else:
            receivers = outmesh.x[tor_samnodes[:, 0], :]
            senders = outmesh.x[tor_samnodes[:, 1], :]
        edge_features = (receivers + senders)/2
        if self.reward_condition:
            beta_batch = torch.zeros(edge_features.shape[0]).to(edge_features.device)
            beta_batch = reward_beta[outmesh.batch[tor_samnodes[:,0]].to(torch.int64)][:,None]
            edge_features = torch.cat([edge_features,beta_batch.repeat([1,5])],dim=-1).float()
        remesh_prob = (self.decoder_edge_split(edge_features)/self.rescale)
        
    
        shape = (remesh_prob.shape)
        if self.is_single_action:
            # remesh_prob - -1 if not selected, 1 if selected
            remesh_prob_softmax, remesh_prob = get_remesh_prob_softmax(remesh_prob, outmesh.batch[tor_samnodes[:,0]],self.current_batch,k=self.top_k_action) # shape[#nodes,1]
            remesh_prob = remesh_prob.reshape(shape)
            remesh_prob_softmax = remesh_prob_softmax.reshape(shape)
        else:
            remesh_prob = torch.sigmoid(remesh_prob)
            remesh_prob_softmax = remesh_prob
       
        if self.training:
            mask = torch.rand_like(remesh_prob) < remesh_prob
        else:
            if self.is_split_test:
                mask = remesh_prob > 0.1
            else:
                mask = remesh_prob > 0.5
                
        remesh_prob = remesh_prob*(mask.float()) + (1-(mask.float()))*(1-remesh_prob)
        
        mask = mask.reshape((mask.shape[0], 1))
        split_edges = torch.masked_select(tor_samnodes, mask).reshape(-1,2)        
        assert (outmesh.batch[tor_samnodes[:,0]]!=outmesh.batch[tor_samnodes[:,1]]).sum()<1e-4
        return split_edges,[remesh_prob.flatten(),tor_samnodes,outmesh.batch[tor_samnodes[:,0]],remesh_prob_softmax.flatten()]
    
    def gen_delaunay_mask_agent(self, torchmesh, sampled_nodes, new_dnode_face_dict):
        """ Compute delaunay criterion to decide edges to be flipped.
        
        Args:
            torchmesh: mesh data with outmesh.
            sampled_nodes: list of maximal independent flippable edges.
            new_dnode_face_dict: dictionary each of whose keys is a node 
            in the dual graph and values is a list of nodes in the dual
            graph adjacent to the node of the key vertex.
            
        Returns:
            delaunay_mask: mask indicating whether or not sampled edges 
            can be flipped. The shape is (len(sampled_nodes),).
        """
        pdb.set_trace()
        indices = torch.tensor(np.array([new_dnode_face_dict[fedge] for fedge in sampled_nodes]))
        abs_vecs = torchmesh.x_coords[indices, :2]
        rel_vecs = abs_vecs[:,:,0,:] - abs_vecs[:,:,1,:]
        innerprod = (rel_vecs[:,0::2,:] * rel_vecs[:, 1::2, :]).sum(dim=2, keepdim=True)
        norms = torch.sqrt((rel_vecs * rel_vecs).sum(dim=2, keepdim=True))
        cosines = innerprod/(norms[:,0::2,:]*norms[:,1::2,:])
        delaunay_mask = torch.acos(cosines).sum(dim=1).flatten() > 3.14159265359
        return delaunay_mask
    
    def choose_flippable_edge_GNN(self, outmesh,reward_beta=None):
        """ Choose flippable edges.

        Args:
            outmesh: mesh data.
            sizing_fields: gnn predicted sizing_fields
            
        Returns:
            flip_edges: edges to be splitten with shape [(number of edges to be flipped), 2].
        """      
        sampled_nodes, new_dnode_face_dict, neighbor_vertices = self._sample_flippable_edges(outmesh)
        if len(sampled_nodes) == 0:
            return torch.tensor([]).reshape(0,2)
        # todo: implement delaunay_mask
        delaunay_mask = self.gen_delaunay_mask_agent(outmesh, sampled_nodes, new_dnode_face_dict)
        # print(delaunay_mask)
        tor_samnodes = torch.tensor(sampled_nodes, device=outmesh.x.device)[delaunay_mask, :]
        if outmesh.xfaces.shape[0] == 0:
            # if is_pos=True
            # Need to fix 
            receivers = outmesh.x[tor_samnodes[:, 0], :]
            senders = outmesh.x[tor_samnodes[:, 1], :]
        else:
            receivers = outmesh.x[tor_samnodes[:, 0], :]
            senders = outmesh.x[tor_samnodes[:, 1], :]
        edge_features = (receivers + senders)/2
        if self.reward_condition:
            beta_batch = torch.zeros(edge_features.shape[0]).to(edge_features.device)
            beta_batch = reward_beta[outmesh.batch[tor_samnodes[:,0]].to(torch.int64)][:,None]
            edge_features = torch.cat([edge_features,beta_batch.repeat([1,5])],dim=-1).float()
           
        remesh_prob = (self.decoder_edge_flip(edge_features)/self.rescale)
        
        shape = (remesh_prob.shape)
        if self.is_single_action:
            # remesh_prob - -1 if not selected, 1 if selected
            remesh_prob_softmax, remesh_prob = get_remesh_prob_softmax(remesh_prob, outmesh.batch[tor_samnodes[:,0]],self.current_batch,k=self.top_k_action) # shape[#nodes,1]
            remesh_prob = remesh_prob.reshape(shape)
            remesh_prob_softmax = remesh_prob_softmax.reshape(shape)
        else:
            remesh_prob = torch.sigmoid(remesh_prob)
            remesh_prob_softmax = remesh_prob
            
        if self.training:
            if self.is_split_test:
                mask = torch.rand_like(remesh_prob) < remesh_prob
            else:
                mask = torch.rand_like(remesh_prob) < remesh_prob
        else:
            if self.is_split_test:
                mask = remesh_prob > 0.1
            else:
                mask = remesh_prob > 0.5
        remesh_prob = remesh_prob*(mask.float()) + (1-(mask.float()))*(1-remesh_prob)
        
        mask = mask.reshape((mask.shape[0], 1))
        flip_edges = torch.masked_select(tor_samnodes, mask).reshape(-1,2)
        return flip_edges,[remesh_prob.flatten(),tor_samnodes,outmesh.batch[tor_samnodes[:,0]],remesh_prob_softmax.flatten()]

    def choose_onedim_coarsen_edges_GNN(self, torchmesh,reward_beta=None, is_timing=0):
        """ Choose splittable edges.

        Args:
            torchmesh: mesh data.

        Returns:
            split_edges: edges to be coarsened with shape [(number of edges to be coarsened), 2].
        """      
        p.print("2.22511", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        coarsened_nodes, snodes_edges_list = self._sample_onedim_coarsened_nodes(torchmesh, is_timing=is_timing)  # snodes_edges_tensor: [n_edges, 2, 2]
        if len(coarsened_nodes) == 0:
            collapsed_edges = torch.tensor([[],[]])
            merging_nodes = torch.tensor([[],[]])
            return collapsed_edges, merging_nodes

        p.print("2.22512", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        snodes_edges_tensor = torch.tensor(snodes_edges_list, device=torchmesh.x.device)  # [n_edges, 2, 2]
        receivers = torchmesh.x[snodes_edges_tensor[..., 0]]  # [n_edges, 2, x_feature]
        senders = torchmesh.x[snodes_edges_tensor[..., 1]]    # [n_edges, 2, x_feature]
        edge_features = 0.5*(receivers + senders)[..., :]     # [n_edges, 2, x_feature]
        p.print("2.22513", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        if self.reward_condition:
            beta_batch = torch.zeros(edge_features.shape[0:2]).to(edge_features.device)
            beta_batch = reward_beta[torchmesh.batch[snodes_edges_tensor[...,0]].to(torch.int64)][:,:,None]
            edge_features = torch.cat([edge_features,beta_batch.repeat([1,1,5])],dim=-1).float()        
        p.print("2.22514", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        probs =(self.decoder_edge_coarse(edge_features)/self.rescale).T.reshape(2, -1)
        indices = torch.argmax(probs, dim=0)

        max_probs = torch.gather(probs, 0, indices.reshape(1,-1))
        p.print("2.22515", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        shape = (max_probs.shape)
        if self.is_single_action:
            # remesh_prob - -1 if not selected, 1 if selected
            max_probs_softmax, max_probs = get_remesh_prob_softmax(max_probs, torchmesh.batch[coarsened_nodes],self.current_batch,k=self.top_k_action) # shape[#nodes,1]
            max_probs = max_probs.reshape(shape)
            max_probs_softmax = max_probs_softmax.reshape(shape)
        else:
            max_probs = torch.sigmoid(max_probs)
            max_probs_softmax = max_probs

        p.print("2.22516", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        if self.training:
            coarsening_mask = torch.rand_like(max_probs)<max_probs
        else:
            coarsening_mask = max_probs > 0.5
        max_probs = max_probs*(coarsening_mask.float()) + (1-(coarsening_mask.float()))*(1-max_probs)
        
        highest_prob_edges = torch.gather(snodes_edges_tensor, 1, torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)).reshape(indices.shape[0],2).T

        collapsed_edges = highest_prob_edges[:,coarsening_mask.reshape(coarsening_mask.shape[1])]
        merging_nodes = torch.tensor(coarsened_nodes, device=torchmesh.x.device)[coarsening_mask[0,:]]
        assert ((torchmesh.batch[torch.tensor(coarsened_nodes, device=torchmesh.x.device)]!=torchmesh.batch[snodes_edges_tensor[:,0]][:,0]).sum())<1e-5
        assert ((torchmesh.batch[torch.tensor(coarsened_nodes, device=torchmesh.x.device)]!=torchmesh.batch[snodes_edges_tensor[:,1]][:,0]).sum())<1e-5
        p.print("2.22517", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        return collapsed_edges, merging_nodes, [max_probs.flatten(),coarsened_nodes,torchmesh.batch[coarsened_nodes],max_probs_softmax.flatten()]


    def choose_coarsen_edges_GNN(self, torchmesh,reward_beta=None):
        """ Choose splittable edges.
        
        Args:
            torchmesh: mesh data.
            
        Returns:
            split_edges: edges to be coarsened with shape [(number of edges to be coarsened), 2].
        """      
        coarsened_nodes, snodes_edges_list = self._sample_coarsened_nodes(torchmesh)
        if len(coarsened_nodes) == 0:
            collapsed_edges = torch.tensor([[],[]])
            merging_nodes = torch.tensor([[],[]])
            return collapsed_edges, merging_nodes, [[None]]
        
        snodes_edges_tensor = torch.tensor(snodes_edges_list, device=torchmesh.x.device)
        receivers = torchmesh.x[snodes_edges_tensor[..., 0]]
        senders = torchmesh.x[snodes_edges_tensor[..., 1]]
        edge_features = 0.5*(receivers + senders)[..., :]
        if self.reward_condition:
            beta_batch = torch.zeros(edge_features.shape[0]).to(edge_features.device)
            beta_batch = reward_beta[torchmesh.batch[snodes_edges_tensor[...,0]].to(torch.int64)][:,:,None]
            edge_features = torch.cat([edge_features,beta_batch.repeat([1,1,5])],dim=-1).float()  
            
        probs = (self.decoder_edge_coarse(edge_features)/self.rescale).reshape(-1, 4).permute(1,0)
        indices = torch.argmax(probs, dim=0)
        max_probs = torch.gather(probs, 0, indices.reshape(1,-1))
        
        shape = (max_probs.shape)
        if self.is_single_action:
            # remesh_prob - -1 if not selected, 1 if selected
            max_probs_softmax, max_probs = get_remesh_prob_softmax(max_probs, torchmesh.batch[coarsened_nodes],self.current_batch,k=self.top_k_action) # shape[#nodes,1]
            max_probs = max_probs.reshape(shape)
            max_probs_softmax = max_probs_softmax.reshape(shape)
        else:
            max_probs = torch.sigmoid(max_probs)
            max_probs_softmax = max_probs
            
        if self.training:
            coarsening_mask = torch.rand_like(max_probs) < max_probs
        else:
            coarsening_mask = max_probs > 0.5
        max_probs = max_probs*(coarsening_mask.float()) + (1-(coarsening_mask.float()))*(1-max_probs)
        
        highest_prob_edges = torch.gather(snodes_edges_tensor, 1, torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)).reshape(indices.shape[0],2).T

        collapsed_edges = highest_prob_edges[:, coarsening_mask.reshape(coarsening_mask.shape[1])]

        merging_nodes = torch.tensor(coarsened_nodes, device=torchmesh.x.device)[coarsening_mask[0,:]]
        return collapsed_edges, merging_nodes, [max_probs.flatten(),coarsened_nodes,torchmesh.batch[coarsened_nodes],max_probs_softmax.flatten()]


    def coarsen_GNN(self, torchmesh,reward_beta=None, is_timing=0):
        """ Perform coarsen action on mesh according to sampled edges.

        Args:
            torchmesh: mesh data.

        Returns:
            torchmesh: mesh data with coarsened vertices, edges, and faces.
        """
        if torchmesh.xfaces.shape[0] == 0:
            p.print("2.2251", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            collapsed_edges, merging_nodes, max_probs= self.choose_onedim_coarsen_edges_GNN(torchmesh,reward_beta=reward_beta, is_timing=is_timing)
            p.print("2.2252", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            if collapsed_edges.shape[1] == 0:
                return torchmesh, max_probs
            new_edge_index, _ = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
            p.print("2.2253", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            new_edge_index, new_x, new_x_phys, new_x_batch, new_x_pos = self.clean_onedim_isolated_vertices(torchmesh, new_edge_index)
            p.print("2.2254", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            if not (new_x.shape[0] - int(new_edge_index.shape[1]/2)) == torchmesh.batch.max()+1:
                print("Coarsen is invalid")
                return torchmesh

            torchmesh.edge_index = new_edge_index
            torchmesh.x = new_x
            torchmesh.x_phy = new_x_phys
            torchmesh.batch = new_x_batch
            torchmesh.x_pos = new_x_pos
            p.print("2.2255", precision="millisecond", is_silent=is_timing<3, avg_window=1)
            return torchmesh, max_probs
        
        else:
            old_batch = torchmesh.batch.clone()
            p.print("2.2251", precision="millisecond", is_silent=not is_timing, avg_window=1)
            collapsed_edges, merging_nodes, max_probs = self.choose_coarsen_edges_GNN(torchmesh,reward_beta=reward_beta)
            p.print("2.2252", precision="millisecond", is_silent=not is_timing, avg_window=1)
            if collapsed_edges.shape[1] == 0:
                # print("no collapsed edges")
                return torchmesh, max_probs
            temp_new_edge_index, mapping = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
            p.print("2.2253", precision="millisecond", is_silent=not is_timing, avg_window=1)
            temp_newly_generated_faces = self.delete_adjacent_faces(torchmesh, mapping)
            p.print("2.2254", precision="millisecond", is_silent=not is_timing, avg_window=1)
            newly_generated_faces = self.face_adjustment(temp_newly_generated_faces, mapping)
            # new_faces, new_edge_index, new_x, new_x_pos, new_x_phy, new_x_batch, new_x_coords= self.clean_isolated_vertices(torchmesh, temp_new_edge_index, newly_generated_faces)
            p.print("2.2255", precision="millisecond", is_silent=not is_timing, avg_window=1)
            new_faces, new_edge_index, new_x, new_x_pos, new_x_phy, new_x_batch, new_x_coords, new_onehot = self.clean_isolated_vertices(torchmesh, temp_new_edge_index, newly_generated_faces)  
            p.print("2.2256", precision="millisecond", is_silent=not is_timing, avg_window=1)
            # pdb.set_trace()

            if not (new_x.shape[0] + new_faces.shape[1] - int(new_edge_index.shape[1]/2)) == torchmesh.batch.max()+1:
                print("Coarsen is invalid")
                pdb.set_trace()
                collapsed_edges, merging_nodes = self.choose_coarsen_edges_GNN(torchmesh)
                pdb.set_trace()
                new_edge_index, mapping = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
                pdb.set_trace()
                newly_generated_faces = self.delete_adjacent_faces(torchmesh, mapping)
                pdb.set_trace()
                newly_generated_faces = self.face_adjustment(newly_generated_faces, mapping)
                pdb.set_trace()
                new_faces, new_edge_index, new_x = self.clean_isolated_vertices(torchmesh, new_edge_index, newly_generated_faces)       

                return torchmesh, max_probs

        torchmesh.x = new_x
        torchmesh.x_phy = new_x_phy
        torchmesh.batch_history = old_batch
        torchmesh.batch = new_x_batch
        torchmesh.x_pos = new_x_pos
        torchmesh.edge_index = new_edge_index
        torchmesh.xfaces = new_faces
        torchmesh.x_coords = new_x_coords
        temp_onehots = list(torchmesh.onehot_list)
        temp_onehots[0] = new_onehot
        torchmesh.onehot_list = tuple(temp_onehots)
        # torchmesh.kinematics_list[-1] = new_kinems
        return torchmesh, max_probs


    def act(self, *args, **kwargs):
        """Policy's main function. Given the state (PyG data), returns the action probability and the resulting mesh after the action."""
        return self.remeshing_forward_GNN(*args, **kwargs)


    def remeshing_forward_GNN(self, torchmesh, use_pos=False,interp_index=None, reward_beta=None, **kwarg):
        """ Perform remeshing action on mesh data.

        Args:
            torchmesh: pyg data instance for mesh.
            reward_beta: indicate speed accuracy tradeoff

        Returns:
            outmesh: mesh data after gnn forwarding and remeshing.
            logsigmoid_sum: {"action": logprob of shape [batch, 1]}
            entropy_sum: {"action": entropy of shape [batch, 1]}
            remesh_prob: {"action": [prob, node_index , batch_index]}
        """
        is_timing = kwarg["is_timing"] if "is_timing" in kwarg else 0
        p.print("2.221", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        if hasattr(torchmesh, "node_feature"):
            if len(dict(to_tuple_shape(torchmesh.original_shape))["n0"]) == 0:
                batch = torchmesh.batch['n0']
                x_coords = torchmesh['history']['n0'][1] #
                torchmesh = attrdict_to_pygdict(torchmesh, is_flatten=True, use_pos=use_pos)   
                torchmesh.batch = batch
                if hasattr(torchmesh, "onehot_list"):
                    onehot = torchmesh.onehot_list[0]
                    raw_kinems = self.compute_kinematics(torchmesh, interp_index)
                    kinematics = torchmesh.onehot_list[0][:,:1] * raw_kinems
                    torchmesh.x = torch.cat([torchmesh.x, onehot, kinematics], dim=-1)
            else:
                batch = torchmesh.batch.clone()
                torchmesh = deepsnap_to_pyg(torchmesh, is_flatten=True, use_pos=use_pos)
                x_coords = torchmesh.x_pos.clone()        
        elif hasattr(torchmesh, "x"):
            batch = torchmesh.batch.clone()
            x_coords = torchmesh.history[-1]
            if hasattr(torchmesh, "onehot_list"):
                onehot = torchmesh.onehot_list[0]
                raw_kinems = self.compute_kinematics(torchmesh, interp_index)
                kinematics = torchmesh.onehot_list[0][:,:1] * raw_kinems
                torchmesh.x = torch.cat([torchmesh.x, onehot, kinematics], dim=-1)
                 
                    # if hasattr(torchmesh, "node_feature"):
            #             if len(dict(to_tuple_shape(torchmesh.original_shape))["n0"]) == 0:
            #                 batch = torchmesh.batch['n0']
            #                 x_coords = torchmesh['history']['n0'][1] #
            #                 torchmesh = attrdict_to_pygdict(torchmesh, is_flatten=True, use_pos=use_pos)                
            #             else:
            #                 batch = torchmesh.batch.clone()
            #                 torchmesh = deepsnap_to_pyg(torchmesh, is_flatten=True, use_pos=use_pos)
            #                 x_coords = torchmesh.x_pos.clone()        
            #         elif hasattr(torchmesh, "x"):
            #             batch = torchmesh.batch.clone()
            #             x_coords = torchmesh.history[-1]
        
        self.current_batch = int(batch.max()+1)
        self.device = batch.device
        
        if torchmesh.xfaces.shape[0] == 0:
            self.skip_flip = True
        old_xfaces = torchmesh.xfaces.clone()
        old_batch = batch.clone()
        old_x_coords = x_coords.clone()
        torchmesh.batch = batch
        x_phy = torchmesh.x.clone() #torchmesh.x = 6*1 : [x_mesh, y_mesh, z_mesh=0,x_world, y_world, z_world]
        torchmesh.x_phy = x_phy
        torchmesh.x_coords = x_coords
        torchmesh = self.get_edge_latents(torchmesh,reward_beta=reward_beta) 
        remesh_prob = {}
        
        p.print("2.222", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        # Split edges of mesh
        if self.skip_split:
            split_outmesh = torchmesh
        else:    
            if self.is_split_test:
                p.print("2.2221", precision="millisecond", is_silent=not is_timing, avg_window=1)
                split_edges, split_edge_remesh_prob = self.choose_split_edge_GNN(torchmesh,reward_beta=reward_beta)
                p.print("2.2222", precision="millisecond", is_silent=not is_timing, avg_window=1)
                remesh_prob['split_edge_remesh_prob'] = split_edge_remesh_prob
                if split_edges.shape[0] == 0:
                    outmesh = torchmesh
                else:
                    outmesh = self.split(split_edges, torchmesh)
                return outmesh
            else:
                p.print("2.2221", precision="millisecond", is_silent=not is_timing, avg_window=1)
                split_edges, split_edge_remesh_prob = self.choose_split_edge_GNN(torchmesh,reward_beta=reward_beta)
                p.print("2.2222", precision="millisecond", is_silent=not is_timing, avg_window=1)
                remesh_prob['split_edge_remesh_prob'] = split_edge_remesh_prob
                split_outmesh = self.split(split_edges, torchmesh)
                p.print("2.2223", precision="millisecond", is_silent=not is_timing, avg_window=1)

        p.print("2.223", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        # Flip edges of mesh
        if self.skip_flip:
            flipped_outmesh = split_outmesh
        else:
            if self.is_flip_test:
                split_outmesh = torchmesh
                flip_edges, flip_edge_remesh_prob = self.choose_flippable_edge_GNN(split_outmesh,reward_beta=reward_beta)
                remesh_prob['flip_edge_remesh_prob'] = flip_edge_remesh_prob
                outmesh = self.flip(flip_edges, split_outmesh)
                return outmesh
            else:
                p.print("2.2231", precision="millisecond", is_silent=not is_timing, avg_window=1)
                flip_edges, flip_edge_remesh_prob = self.choose_flippable_edge_GNN(split_outmesh,reward_beta=reward_beta)
                p.print("2.2232", precision="millisecond", is_silent=not is_timing, avg_window=1)
                remesh_prob['flip_edge_remesh_prob'] = flip_edge_remesh_prob
                flipped_outmesh = self.flip(flip_edges, split_outmesh)
                p.print("2.2233", precision="millisecond", is_silent=not is_timing, avg_window=1)
                
        # Coarsen edges of mesh
        p.print("2.224", precision="millisecond", is_silent=is_timing<3, avg_window=1)
        if self.skip_coarse:
            outmesh = flipped_outmesh                
            index = torch.sort(outmesh.batch)[1]
            a = torch.cat([outmesh.x_pos,outmesh.batch.unsqueeze(-1)],dim=-1)
            outmesh.index = torch_lexsort(a.permute(1,0))
            
            outmesh.x = outmesh.x_phy[:,:]
            if self.dataset.startswith("arcsi"):
                outmesh.history = (old_x_coords,outmesh.x_coords)
                outmesh.batch_history = old_batch
            
            outmesh.x_bdd = outmesh.x[:,-1:]
            if self.dataset.startswith("m"):
                outmesh.dataset = to_tuple_shape(outmesh.dataset)
                #reset bdd nodes, all interpolate boundary node are removed
                outmesh.x[:,-1] = 1*(outmesh.x[:,-1]==1)
                outmesh.x_bdd = outmesh.x[:,-1:]
                outmesh = update_edge_attr_1d(outmesh)
            logsigmoid_sum, entropy_sum = self.get_batch_logsigmoid_sum(remesh_prob,self.current_batch,self.is_single_action)
            return outmesh, logsigmoid_sum, entropy_sum, remesh_prob
        else:
            if self.is_coarsen_test:
                flipped_outmesh = torchmesh
                outmesh, max_probs = self.coarsen_GNN(flipped_outmesh,reward_beta=reward_beta)
                remesh_prob['coarse_prob'] = max_probs
                if use_pos and (outmesh.xfaces.shape[0] == 0):
                    outmesh.x = outmesh.x[:, :self.input_size+1]
                else:
                    outmesh.x = outmesh.x[:, :self.input_size]
                logsigmoid_sum, entropy_sum = self.get_batch_logsigmoid_sum(remesh_prob,self.current_batch,self.is_single_action)
                if self.dataset.startswith("arcsi"):
                    outmesh.history = (old_x_coords,outmesh.x_coords)
                    outmesh.batch_history = old_batch
                return outmesh, logsigmoid_sum, entropy_sum, remesh_prob

            else:
                p.print("2.225", precision="millisecond", is_silent=is_timing<3, avg_window=1)
                outmesh, max_probs = self.coarsen_GNN(flipped_outmesh,reward_beta=reward_beta, is_timing=is_timing)
                p.print("2.226", precision="millisecond", is_silent=is_timing<3, avg_window=1)
                remesh_prob['coarse_prob'] = max_probs
                index = torch.sort(outmesh.batch)[1]
                a = torch.cat([outmesh.x_pos,outmesh.batch.unsqueeze(-1)],dim=-1)
                outmesh.index = torch_lexsort(a.permute(1,0))
                if self.dataset.startswith("arcsi"):
                    outmesh.history = (old_x_coords,outmesh.x_coords)
                    outmesh.xface_list = (old_xfaces.T, outmesh.xfaces.T)
                    outmesh.batch_history = old_batch
                outmesh.x = outmesh.x_phy[:,:]
                p.print("2.227", precision="millisecond", is_silent=is_timing<3, avg_window=1)

                if self.dataset.startswith("m"):
                    outmesh.dataset = to_tuple_shape(outmesh.dataset)
                    #reset bdd nodes, all interpolate boundary node are removed
                    outmesh.x[:,-1] = 1*(outmesh.x[:,-1]==1)
                    outmesh.x_bdd = outmesh.x[:,-1:]
                    outmesh = update_edge_attr_1d(outmesh)
                    p.print("2.228", precision="millisecond", is_silent=is_timing<3, avg_window=1)
                logsigmoid_sum, entropy_sum = self.get_batch_logsigmoid_sum(remesh_prob, self.current_batch, self.is_single_action)
                p.print("2.229", precision="millisecond", is_silent=is_timing<3, avg_window=1)
                return outmesh, logsigmoid_sum, entropy_sum, remesh_prob

    def get_batch_logsigmoid_sum(self, remesh_prob, batch_size, single=False):
        logsigmoid_sum = {}
        entropy_sum = {}
        for key in remesh_prob.keys():
            if len(remesh_prob[key][0])==0:
                logsigmoid_sum[key] = torch.zeros([self.current_batch],device=self.device)
                entropy_sum[key] = torch.zeros([self.current_batch],device=self.device)
                continue
            if remesh_prob[key][0][0]!=None:
                if single:
                    prob = remesh_prob[key][3]
                else:
                    prob = remesh_prob[key][0]
                index = remesh_prob[key][2]
                logsigmoid_sum[key] = batch_logsigmoid_sum(prob,index,batch_size)
                entropy_sum[key] = batch_entropy_sum(prob=prob, index=index, batch_size=batch_size, single=single)
            else:
                logsigmoid_sum[key] = torch.zeros([self.current_batch],device=self.device)
                entropy_sum[key] = torch.zeros([self.current_batch],device=self.device)
        return logsigmoid_sum, entropy_sum
    
    @property
    def model_dict(self):
        model_dict = {"type": "GNNPolicyAgent"}
        model_dict["input_size"] = self.input_size
        model_dict["output_size"] = self.output_size
        model_dict["act_name"] = self.act_name
        model_dict["dataset"] = self.dataset
        model_dict["nmax"] = to_np_array(self.nmax)
        model_dict["edge_dim"] = self.edge_dim
        model_dict["latent_dim"] = self.latent_dim
        model_dict["num_steps"] = self.num_steps
        model_dict["layer_norm"] = self.layer_norm
        model_dict["batch_norm"] = self.batch_norm
        model_dict["edge_attr"] = self.edge_attr
        model_dict["offset_split"] = to_np_array(self.offset_split)
        model_dict["offset_coarse"] = to_np_array(self.offset_coarse)
        model_dict["edge_threshold"] = self.edge_threshold
        model_dict["skip_split"] = self.skip_split
        model_dict["skip_flip"] = self.skip_flip
        model_dict["skip_coarse"] = self.skip_coarse
        model_dict["is_split_test"] = self.is_split_test
        model_dict["is_flip_test"] = self.is_flip_test
        model_dict["is_coarsen_test"] = self.is_coarsen_test
        model_dict["samplemode"] = self.samplemode
        model_dict["min_edge_size"] = self.min_edge_size
        model_dict["state_dict"] = to_cpu(self.state_dict())
        model_dict["rescale"] = self.rescale
        model_dict["is_single_action"] = self.is_single_action
        model_dict["top_k_action"] = self.top_k_action
        model_dict["reward_condition"] = self.reward_condition
        return model_dict

class GNNPolicyAgent_Sampling(GNNRemesherPolicy):
    def __init__(
        self,
        input_size,
        nmax,
        edge_dim,
        latent_dim=32,
        edge_attr=False,
        layer_norm = False,
        batch_norm = False,
        num_steps=3,

        val_layer_norm=False,
        val_batch_norm=False,
        val_num_steps=3,
        val_pooling_type="global_mean_pool",
        use_pos=False,
        final_ratio=0.1,
        final_pool="global_mean_pool",

        num_pool=1,
        val_act_name="relu",
        val_act_name_final="relu",
        
        rl_num_steps=3,
        rl_layer_norm=False,
        rl_batch_norm=False,
        edge_threshold=0.,
        skip_split=False,
        skip_flip=False,
        act_name='relu',
        dataset=None,
        is_split_test=False,
        is_flip_test=False,
        is_coarsen_test=False,
        samplemode="random",
        skip_coarse=False,
        min_edge_size=0.000,
        rescale=10000,
        batch_size=16,
        is_single_action=False,
        top_k_action=1,
        reward_condition=False,
        offset_split=0.,
        offset_coarse=0.,
        share_processor=True,

        max_action=100,
        kaction_pooling_type="global_mean_pool",
        processor_aggr="max",
        *args,
        **kwargs
    ):
        nn.Module.__init__(self)
        
        #rl parameters
        self.input_size = input_size
        output_size = 1
        self.edge_attr = edge_attr
        self.output_size = 1
        self.edge_dim = edge_dim 
        self.latent_dim = latent_dim
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.num_steps = num_steps
        
        self.rl_num_steps = rl_num_steps
        self.act_name = act_name
        self.rl_layer_norm = rl_layer_norm
        self.samplemode = samplemode
        self.rl_batch_norm = rl_batch_norm
        self.dataset = dataset
        self.edge_threshold = edge_threshold
        self.is_flip_test = is_flip_test
        self.skip_coarse = skip_coarse
        self.min_edge_size = min_edge_size
        self.batch_size = batch_size
        self.is_single_action = is_single_action
        self.top_k_action = top_k_action
        self.var = 0
        self.skip_split = skip_split
        self.skip_flip = skip_flip
        self.is_flip_test = is_flip_test
        self.is_split_test = is_split_test
        self.reward_condition = reward_condition
        self.rescale = rescale
        self.offset_split = torch.tensor(offset_split)
        self.offset_coarse = torch.tensor(offset_coarse)
        self.nmax = nmax.type(torch.int64) if isinstance(nmax, torch.Tensor) else torch.tensor(nmax, dtype=torch.int64)
        self.is_coarsen_test = is_coarsen_test
        
        
        self.nmax = nmax.type(torch.int64) if isinstance(nmax, torch.Tensor) else torch.tensor(nmax, dtype=torch.int64)
        self.val_layer_norm = val_layer_norm
        self.val_batch_norm = val_batch_norm
        self.val_num_steps = val_num_steps
        self.val_pooling_type = val_pooling_type
        self.use_pos = use_pos
        self.final_ratio = final_ratio
        self.final_pool = final_pool
        self.num_pool = num_pool
        self.val_act_name = val_act_name
        self.val_act_name_final = val_act_name_final
        self.processor_aggr = processor_aggr
        self.max_action = max_action
        self.kaction_pooling_type = kaction_pooling_type
        self.share_processor = share_processor

        if self.dataset.startswith("a"):
            import dolfin as dolfin
            self.dolfin = dolfin
            # self.input_size = 8
        #joint processer
        self.encoder_edge = FCBlock(in_features=edge_dim,
                                    out_features=latent_dim,
                                    num_hidden_layers=2,
                                    hidden_features=latent_dim,
                                    outermost_linear=True,
                                    nonlinearity=act_name,
                                    layer_norm=layer_norm,
                                   )
        self.encoder_nodes = FCBlock(in_features=self.input_size + (5 if self.reward_condition else 0),
                                     out_features=latent_dim,
                                     num_hidden_layers=2,
                                     hidden_features=latent_dim,
                                     outermost_linear=True,
                                     nonlinearity=act_name,
                                     layer_norm=layer_norm,
                                    )

        if self.processor_aggr=="max":
            self.processors = []
            for _ in range(num_steps):
                self.processors.append((processor(latent_dim, 
                                                latent_dim, 
                                                layer_norm=layer_norm, 
                                                act_name=act_name, 
                                                edge_attr=edge_attr,
                                                )))
                if self.batch_norm:
                    self.processors.append(torch.nn.BatchNorm1d(latent_dim))
                    self.processors.append(torch.nn.BatchNorm1d(latent_dim))
            self.processors = torch.nn.Sequential(*self.processors)         
            
            if self.share_processor:
                self.value_processors = self.processors
            else:
                self.value_processors = []
                for _ in range(num_steps):
                    self.value_processors.append((processor(latent_dim, 
                                                    latent_dim, 
                                                    layer_norm=layer_norm, 
                                                    act_name=act_name, 
                                                    edge_attr=edge_attr,
                                                    )))
                    if self.batch_norm:
                        self.value_processors.append(torch.nn.BatchNorm1d(latent_dim))
                        self.value_processors.append(torch.nn.BatchNorm1d(latent_dim))
                self.value_processors = torch.nn.Sequential(*self.value_processors)   
        elif self.processor_aggr=="mean":
            self.processors = []
            for _ in range(num_steps):
                self.processors.append((processor_mean(latent_dim, 
                                                latent_dim, 
                                                layer_norm=layer_norm, 
                                                act_name=act_name, 
                                                edge_attr=edge_attr,
                                                )))
                if self.batch_norm:
                    self.processors.append(torch.nn.BatchNorm1d(latent_dim))
                    self.processors.append(torch.nn.BatchNorm1d(latent_dim))
            self.processors = torch.nn.Sequential(*self.processors)         
            
            if self.share_processor:
                self.value_processors = self.processors
            else:
                self.value_processors = []
                for _ in range(num_steps):
                    self.value_processors.append((processor_mean(latent_dim, 
                                                    latent_dim, 
                                                    layer_norm=layer_norm, 
                                                    act_name=act_name, 
                                                    edge_attr=edge_attr,
                                                    )))
                    if self.batch_norm:
                        self.value_processors.append(torch.nn.BatchNorm1d(latent_dim))
                        self.value_processors.append(torch.nn.BatchNorm1d(latent_dim))
                self.value_processors = torch.nn.Sequential(*self.value_processors)       
        
        #for value model:
        self.value_model = Value_Model(input_size=self.input_size,
                                        output_size=None,
                                        edge_dim=edge_dim,
                                        latent_dim=latent_dim,
                                        num_pool=num_pool,
                                        act_name=val_act_name,
                                        act_name_final=val_act_name_final,
                                        layer_norm=val_layer_norm,
                                        batch_norm=val_batch_norm,
                                        num_steps=val_num_steps,
                                        pooling_type=val_pooling_type,
                                        edge_attr=edge_attr,
                                        use_pos=use_pos,
                                        final_ratio=final_ratio,
                                        final_pool=final_pool,
                                        reward_condition=reward_condition,
                                        processors=self.value_processors,
                                        encoder_nodes=self.encoder_nodes,
                                        encoder_edge=self.encoder_edge
                                      )
        
        #setup prob predictor:
        #given latent graph, do another MLP on each node features, and then do global aggregation
        #then pass to decoder MLP to predict something [B,max_action]
        self.decoder_actionNum_head = FCBlock(in_features=latent_dim + (5 if self.reward_condition else 0),
                                        out_features=latent_dim,
                                        num_hidden_layers=2,
                                        hidden_features=latent_dim,
                                        outermost_linear=True,
                                        nonlinearity=act_name,
                                        layer_norm=False,
                                       )
        

        if not self.skip_split:
            self.decoder_actionNum_MLPsplit = nn.Linear(latent_dim + (5 if self.reward_condition else 0), max_action)
            self.activationsplit = get_activation(self.val_act_name_final)
            self.decoder_edge_split = FCBlock(in_features=latent_dim + (5 if self.reward_condition else 0),
                                        out_features=output_size,
                                        num_hidden_layers=3,
                                        hidden_features=latent_dim,
                                        outermost_linear=True,
                                        nonlinearity=act_name,
                                        layer_norm=False,
                                       )
        # if dataset.startswith("m"):
        #     self.decoder_actionNum_MLPflip = nn.Linear(latent_dim + (5 if self.reward_condition else 0), max_action)
        #     self.decoder_edge_flip = FCBlock(in_features=latent_dim + (5 if self.reward_condition else 0),
        #                                 out_features=output_size,
        #                                 num_hidden_layers=3,
        #                                 hidden_features=latent_dim,
        #                                 outermost_linear=True,
        #                                 nonlinearity=act_name,
        #                                 layer_norm=False,
        #                                )
        # else:
        #     if not self.skip_flip :
        #         self.decoder_actionNum_MLPflip = nn.Linear(latent_dim + (5 if self.reward_condition else 0), max_action)
        #         self.activationflip = get_activation(self.val_act_name_final)
        #         self.decoder_edge_flip = FCBlock(in_features=latent_dim + (5 if self.reward_condition else 0),
        #                                     out_features=output_size,
        #                                     num_hidden_layers=3,
        #                                     hidden_features=latent_dim,
        #                                     outermost_linear=True,
        #                                     nonlinearity=act_name,
        #                                     layer_norm=False,
        #                                    )
        if not self.skip_coarse:    
            self.decoder_actionNum_MLPcoarse = nn.Linear(latent_dim + (5 if self.reward_condition else 0), max_action)
            self.decoder_edge_coarse = FCBlock(in_features=latent_dim + (5 if self.reward_condition else 0),
                                        out_features=output_size,
                                        num_hidden_layers=3,
                                        hidden_features=latent_dim,
                                        outermost_linear=True,
                                        nonlinearity=act_name,
                                        layer_norm=False,
                                    )
            
        self.pdist = torch.nn.PairwiseDistance(p=2)
    
    
    def get_edge_latents(self, graph,reward_beta=None, **kwargs):
        #condition the latents with beta
        if self.reward_condition:
            beta_batch = reward_beta[graph.batch.to(torch.int64)][:,None]
            graph.x = torch.cat([graph.x,beta_batch.repeat([1,5])],dim=-1).float()
        
        #encode the graph
        graph_evolved = self.encoder(graph)
        for i in range(self.num_steps):
            if self.batch_norm:
                graph_evolved = self.processors[i*3](graph_evolved)
                graph_evolved.x = self.processors[i*3+1](graph_evolved.x)
                graph_evolved.edge_attr = self.processors[i*3+2](graph_evolved.edge_attr)
            else:
                graph_evolved = self.processors[i](graph_evolved)
        #return encoded graph
        return graph_evolved
    
    def get_K_latent(self,graph_evolved,reward_beta=None,**kwargs):
        # get node features
        x_features = graph_evolved.x.clone()
        if self.reward_condition:
            beta_batch = reward_beta[graph_evolved.batch.to(torch.int64)][:,None]
            x_features = torch.cat([x_features,beta_batch.repeat([1,5])],dim=-1).float()
        # pass into the head to get latents for predicting action numbers    
        x_features = self.decoder_actionNum_head(x_features)
        # get batch info 
        batch = graph_evolved.batch.long()
        # global pooling to get the global graph latents
        if self.kaction_pooling_type=="global_max_pool":
            graph_latent = torch_geometric.nn.global_max_pool(x_features, batch)
        elif self.kaction_pooling_type=="global_add_pool":
            graph_latent = torch_geometric.nn.global_add_pool(x_features, batch)
        elif self.kaction_pooling_type=="global_mean_pool":
            graph_latent = torch_geometric.nn.global_mean_pool(x_features, batch)
        
        return graph_latent
        
        

    def get_K(self,graph_latent,reward_beta=None,mode=None, eps=1e-7,is_eval_sample=True,**kwargs):
        # condition the global graph latents with beta
        if self.reward_condition:
            graph_latent = torch.cat([graph_latent,reward_beta[:,None].repeat([1,5])],dim=-1).float()
        # a linear layer to predict the logit for different number of actions up to max_action
        if mode=="split":
            logit = ((self.decoder_actionNum_MLPsplit(graph_latent)))/50 #[b,max_action]
            if self.offset_split!=0:
                temp = torch.arange(self.offset_split,0,+2)
                offset = torch.zeros(logit.shape[1],device=logit.device)
                offset[:temp.shape[0]] = temp
                logit+=offset
        elif mode=="flip":
            logit = ((self.decoder_actionNum_MLPflip(graph_latent)))/50 #[b,max_action]
        elif mode=="coarse":
            logit = ((self.decoder_actionNum_MLPcoarse(graph_latent)))/50  #[b,max_action]
            if self.offset_coarse!=0:
                temp = torch.arange(self.offset_coarse,0,-2)
                offset = torch.zeros(logit.shape[1],device=logit.device)
                offset[:temp.shape[0]] = temp
                logit+=offset
        else:
            raise
        # if training - sample from logit 
        if self.training or (is_eval_sample):
        # if self.training:
            sampler = torch.distributions.categorical.Categorical(logits=logit) #B,max_actions
            selected_k = sampler.sample()
            log_p = sampler.log_prob(selected_k)
        # if testing - find argmax
        else:
            sampler = torch.distributions.categorical.Categorical(logits=logit) #B,max_actions
            selected_k = torch.argmax(logit,dim=-1)
            log_p = sampler.log_prob(selected_k)
            
        if torch.isnan(log_p).any():
            print(log_p)
            pdb.set_trace()
        return selected_k[:,None],log_p[:,None],sampler.entropy()[:,None]
    
    def act(self, *args, **kwargs):
        """Policy's main function. Given the state (PyG data), returns the action probability and the resulting mesh after the action."""
        return self.remeshing_forward_GNN(*args, **kwargs)
    
    def sample(self, prob, index=None, K=None, eps=1e-7, is_eval_sample=True):
        """
        prob: n*1
        index: n*1
        K: b*1
        
        mask: n*1
        log prob: b*1 \sum over k_i for each sample i
        """
        all_sampled_idx = []
        # loop over all samples in the batch
        for i in range(self.current_batch):
            if K[i]>0:
                i_index = torch.where(index==i)[0]
                p_i = prob[i_index]
                if p_i.shape[0]>0:
                    assert torch.abs(p_i.sum()-1)<5e-5, "(p_i.sum()-1).abs()={}".format(torch.abs(p_i.sum()-1).item())
                    sample_ki = K[i] if (K[i]<p_i.shape[0]) else p_i.shape[0]
                    if self.training or is_eval_sample:
                        selected_k = torch.multinomial(input=p_i.squeeze(-1),num_samples=int(sample_ki),replacement=True)
                    else:
                        selected_k = torch.sort(p_i.squeeze(-1),descending=True)[1][:int(sample_ki)]
                        
                    sampled_index_i = i_index[selected_k]
                    all_sampled_idx.append(sampled_index_i)
        mask = torch.zeros_like(prob)[:,0]
        if len(all_sampled_idx)>0:
            all_sampled_idx = torch.cat(all_sampled_idx,dim=-1)
            mask[all_sampled_idx]=1
        log_p = torch.log(prob+eps)
        if torch.isnan(log_p).any():
            print(log_p)
            pdb.set_trace()
  
        log_p = torch.zeros(self.current_batch,device=prob.device).scatter_add(0, index[all_sampled_idx], (log_p.reshape(-1))[all_sampled_idx])
        p_entropy = -1*prob*torch.log(prob+eps)
        if torch.isnan(p_entropy).any():
            print(p_entropy)
            pdb.set_trace()
        p_entropy = torch.zeros(self.current_batch,device=prob.device).scatter_add(0, index, p_entropy.reshape(-1))
        return mask[:,None]==1,log_p,p_entropy
    
    def filter_with_dual_graph(self, torchmesh=None, sampled_edges=None):
        """
        torchmesh - mesh info
        sampled_edges - selected edges, shape N,2
        sampled_nodes - sampled nodes for each edge
        """
        _, dedge_neighbor_dict = self._create_dual_graph(torchmesh)
        actual_sampled_edge = []
       
        set_s = set()
        index = []
        # print("sampled_edges",sampled_edges.shape)
        for idx in range(sampled_edges.shape[0]):
            elm = (int(sampled_edges[idx,0]),int(sampled_edges[idx,1]))
            # print(elm,set_s,(elm in set_s))
            if not (elm in set_s):
                actual_sampled_edge.append(elm)
            for elm_i in dedge_neighbor_dict[elm]:
                set_s.add(elm_i)
        # print("actual_sampled_edge",torch.tensor(actual_sampled_edge, dtype=torch.int64, device=sampled_edges.device))
        return torch.tensor(actual_sampled_edge, dtype=torch.int64, device=sampled_edges.device)
     
    def filter_2d_coarse(self,torchmesh,sampled_edges=None):
        # Remove nodes with degree 2
        deg_tensor = self._create_degree_tensor(torchmesh)
        deg4nodes = ((deg_tensor == 4).nonzero(as_tuple=True)[0])
        dnodes = set(deg4nodes.tolist())
        neighbor_dict, node_edge_dict, node_edgeset_dict, node_face_dict = self._create_dual_graph_coarse(torchmesh)
        for item in neighbor_dict.items():
            if item[1] == 2:
                dnodes = dnodes - set(item[0])
        
        # Remove nodes belonging to a face with angles more than 90
        if len(dnodes) > 0:
            dnodes = self.del_node_with_obtuse_triangles(dnodes, node_face_dict, torchmesh)
        snodes_edges_list = []
        actual_sampled_nodes = []
        set_s = set()

        for idx in range(sampled_edges.shape[0]):
            elm1 = int(sampled_edges[idx,0])
            elm2 = int(sampled_edges[idx,1])
            if (elm1 in dnodes):
                if (not (elm1 in set_s)) and (not (elm2 in set_s)):
                    actual_sampled_nodes.append(elm1)
                    snodes_edges_list.append([int(sampled_edges[idx,0]),int(sampled_edges[idx,1])])
                    set_s.add(elm1)
                    for elm_i in node_edgeset_dict[elm1]:
                        set_s.add(elm_i)
            elif (elm2 in dnodes):
                if (not (elm1 in set_s)) and (not (elm2 in set_s)):
                    actual_sampled_nodes.append(elm2)
                    snodes_edges_list.append([int(sampled_edges[idx,0]),int(sampled_edges[idx,1])])
                    set_s.add(elm2)
                    for elm_i in node_edgeset_dict[elm2]:
                        set_s.add(elm_i)
        return actual_sampled_nodes, snodes_edges_list
    
        
    def gen_delaunay_mask_agent(self, torchmesh, sampled_nodes, new_dnode_face_dict):
        """ Compute delaunay criterion to decide edges to be flipped.
        
        Args:
            torchmesh: mesh data with outmesh.
            sampled_nodes: list of maximal independent flippable edges.
            new_dnode_face_dict: dictionary each of whose keys is a node 
            in the dual graph and values is a list of nodes in the dual
            graph adjacent to the node of the key vertex.
            
        Returns:
            delaunay_mask: mask indicating whether or not sampled edges 
            can be flipped. The shape is (len(sampled_nodes),).
        """
        indices = torch.tensor(np.array([new_dnode_face_dict[fedge] for fedge in sampled_nodes]))
        abs_vecs = torchmesh.x_coords[indices, :2]
        rel_vecs = abs_vecs[:,:,0,:] - abs_vecs[:,:,1,:]
        innerprod = (rel_vecs[:,0::2,:] * rel_vecs[:, 1::2, :]).sum(dim=2, keepdim=True)
        norms = torch.sqrt((rel_vecs * rel_vecs).sum(dim=2, keepdim=True))
        cosines = innerprod/(norms[:,0::2,:]*norms[:,1::2,:])
        delaunay_mask = torch.acos(cosines).sum(dim=1).flatten() > 3.146
        return delaunay_mask
            
    def flip_all_possible(self,outmesh):
        sampled_nodes, new_dnode_face_dict, neighbor_vertices = self._sample_flippable_edges(outmesh)
        if len(sampled_nodes) == 0:
            return torch.tensor([]).reshape(0,2)
        delaunay_mask = self.gen_delaunay_mask_agent(outmesh, sampled_nodes, new_dnode_face_dict) 
        tor_samnodes = torch.tensor(sampled_nodes, device=outmesh.x.device)[delaunay_mask, :]
        return tor_samnodes.reshape(-1,2)
    
    def choose_split_edge_GNNtopK_heuristic(self,outmesh):
        outmesh_curvature = add_edge_normal_curvature(outmesh)
        curvatures = outmesh_curvature.edge_curvature.detach()
        curvatures = torch.nan_to_num(curvatures,0)
        split_edges = torch.where(curvatures>0.1)[0]
        sampled_edges = outmesh.edge_index[:,split_edges]
        values, _ = sampled_edges.sort(dim=0)
        sampled_edges = values.unique(dim=-1).T
        if sampled_edges.shape[0]!=0:
            # print(outmesh.x_pos.shape,sampled_edges.shape)
            receivers = outmesh.x_coords[sampled_edges[:, 0], :2]
            senders = outmesh.x_coords[sampled_edges[:, 1], :2]
            edge_size = torch.sqrt(((receivers - senders)**2).mean(dim=-1))
            mask = edge_size>self.min_edge_size*2
            splitable_edge = torch.where(mask==1)[0]
            sampled_edges = sampled_edges[splitable_edge]
        
        split_edges = self.filter_with_dual_graph(outmesh,sampled_edges=sampled_edges)
        return split_edges

    def choose_split_edge_GNNtopK(self, outmesh,K_split=None, reward_beta=None,is_timing=0,is_eval_sample=True):
        """ Choose splittable edges.
        
        Args:
            outmesh: mesh data.
            outmesh.x: [position of nodes, sizing field,]
            reward_beta: [b,1]
            
        Returns:
            split_edges: edges to be splitten with shape [(number of edges to be splitted), 2].
        """      
        # sample maximal independent splittable edges
        values, _ = outmesh.edge_index.sort(dim=0)
        tor_samnodes = values.unique(dim=-1).T
        if self.dataset.startswith("m"):
            boundary_loc = 0.16
            idx = torch.where((torch.abs(outmesh.x_pos[tor_samnodes[:, 0]]-boundary_loc)>1e-6) | (torch.abs(outmesh.x_pos[tor_samnodes[:, 1]])>1e-6))[0]
            tor_samnodes = tor_samnodes[idx]
            idx = torch.where((torch.abs(outmesh.x_pos[tor_samnodes[:, 1]]-boundary_loc)>1e-6) | (torch.abs(outmesh.x_pos[tor_samnodes[:, 0]])>1e-6))[0]
            tor_samnodes = tor_samnodes[idx]
         
        if outmesh.xfaces.shape[0] == 0:
            receivers = outmesh.x_pos[tor_samnodes[:, 0], :]
            senders = outmesh.x_pos[tor_samnodes[:, 1], :]
            edge_size = torch.sqrt(((receivers - senders)**2).mean(dim=-1))
            mask = edge_size>self.min_edge_size*2
            splitable_edge = torch.where(mask==1)[0]
            tor_samnodes = tor_samnodes[splitable_edge]
        else:
            receivers = outmesh.x_coords[tor_samnodes[:, 0], :2]
            senders = outmesh.x_coords[tor_samnodes[:, 1], :2]
            edge_size = torch.sqrt(((receivers - senders)**2).mean(dim=-1))
            mask = edge_size>self.min_edge_size*2
            splitable_edge = torch.where(mask==1)[0]
            tor_samnodes = tor_samnodes[splitable_edge]
            
        receivers = outmesh.x[tor_samnodes[:, 0], :]
        senders = outmesh.x[tor_samnodes[:, 1], :]
        edge_features = (receivers + senders)/2
        if self.reward_condition:
            beta_batch = torch.zeros(edge_features.shape[0]).to(edge_features.device)
            beta_batch = reward_beta[outmesh.batch[tor_samnodes[:,0]].to(torch.int64)][:,None]
            edge_features = torch.cat([edge_features,beta_batch.repeat([1,5])],dim=-1).float()
        remesh_logit = (self.decoder_edge_split(edge_features)/self.rescale) #predicted logit
        remesh_prob_softmax = torch_geometric.utils.softmax(remesh_logit,
                                                            index=outmesh.batch[tor_samnodes[:,0]].to(torch.int64),
                                                            num_nodes=self.current_batch) #num,1 
        p.print("2.22221", precision="millisecond", is_silent=is_timing<4, avg_window=1)
        mask, logp, entropyp = self.sample(remesh_prob_softmax,K=K_split,index=outmesh.batch[tor_samnodes[:,0]].to(torch.int64),is_eval_sample=is_eval_sample) 
        p.print("2.22222", precision="millisecond", is_silent=is_timing<4, avg_window=1)
        if outmesh.xfaces.shape[0] == 0:
            sampled_edges = split_edges = torch.masked_select(tor_samnodes, mask).reshape(-1,2)
        else:
            sampled_edges = torch.masked_select(tor_samnodes, mask).reshape(-1,2)
            split_edges = self.filter_with_dual_graph(outmesh,sampled_edges=sampled_edges)
            p.print("2.22223", precision="millisecond", is_silent=is_timing<4, avg_window=1)
        # pdb.set_trace()
        return split_edges, logp, entropyp, {"split_p":to_np_array(torch.exp(logp)),"split_topK":to_np_array(K_split),"softmax_split":to_np_array(remesh_prob_softmax),'action_num':np.array(sampled_edges.shape[0]).reshape(1,1)}
 
    
    def choose_onedim_coarsen_edges_GNN_topk(self, torchmesh,reward_beta=None, K_coarse=None, is_timing=False,is_eval_sample=True):
        """ Choose splittable edges.

        Args:
            torchmesh: mesh data.

        Returns:
            split_edges: edges to be coarsened with shape [(number of edges to be coarsened), 2].
        """     
        if torchmesh.xfaces.shape[0] == 0:
            boundary_loc = 0.16
            values, _ = torchmesh.edge_index.sort(dim=0)
            tor_samnodes = values.unique(dim=-1).T
            idx = torch.where(torch.abs(torchmesh.x_pos[tor_samnodes[:, 0]]-boundary_loc)>1e-6)[0]
            tor_samnodes = tor_samnodes[idx]
            idx = torch.where(torch.abs(torchmesh.x_pos[tor_samnodes[:, 1]])>1e-6)[0]
            tor_samnodes = tor_samnodes[idx]
            idx = torch.where(torch.abs(torchmesh.x_pos[tor_samnodes[:, 0]])>1e-6)[0]
            tor_samnodes = tor_samnodes[idx]
            idx = torch.where(torch.abs(torchmesh.x_pos[tor_samnodes[:, 1]]-boundary_loc)>1e-6)[0]
            tor_samnodes = tor_samnodes[idx]
            
            mask = torchmesh.x_pos[tor_samnodes[:, 0]] > torchmesh.x_pos[tor_samnodes[:, 1]]
            idx = torch.where(mask)[0]
            tor_samnodes[idx, 0], tor_samnodes[idx, 1] = tor_samnodes[idx, 1], tor_samnodes[idx, 0]
            

        
        receivers = torchmesh.x[tor_samnodes[:, 0]]
        senders = torchmesh.x[tor_samnodes[:, 1]]
        edge_features = 0.5*(receivers + senders)[:, :]
        if self.reward_condition:
            beta_batch = torch.zeros(edge_features.shape[0]).to(edge_features.device)
            beta_batch = reward_beta[torchmesh.batch[tor_samnodes[:,0]].to(torch.int64)][:,None]
            edge_features = torch.cat([edge_features,beta_batch.repeat([1,5])],dim=-1).float()
        remesh_logit = (self.decoder_edge_coarse(edge_features)/self.rescale) #predicted logit
        remesh_prob_softmax = torch_geometric.utils.softmax(remesh_logit,
                                                            index=torchmesh.batch[tor_samnodes[:,0]].to(torch.int64),
                                                            num_nodes=self.current_batch) #num,1 
        mask, logp, entropyp = self.sample(remesh_prob_softmax,K=K_coarse,index=torchmesh.batch[tor_samnodes[:,0]].to(torch.int64),is_eval_sample=is_eval_sample) 
        collapsed_edges = torch.masked_select(tor_samnodes, mask).reshape(-1,2)     
        merging_nodes_src = torchmesh.x_pos[collapsed_edges[:, 0]]
        merging_nodes_tar = torchmesh.x_pos[collapsed_edges[:, 1]]
        merging_nodes = collapsed_edges[:,0]
        mask = torch.isin(collapsed_edges[:,0],collapsed_edges[:,1])
        index_mask = torch.where(mask==0)[0]
        collapsed_edges = collapsed_edges[index_mask]
        merging_nodes = merging_nodes[index_mask]
        
        return collapsed_edges.permute(1,0), merging_nodes, logp, entropyp, {"coarse_p":to_np_array(torch.exp(logp)),"coarse_topK":to_np_array(K_coarse),"softmax_coarse":to_np_array(remesh_prob_softmax)}
    
   
    def choose_coarsen_edges_GNN_topk_filter_after(self, torchmesh,reward_beta=None, K_coarse=None, is_timing=False,is_eval_sample=True):
        values, _ = torchmesh.edge_index.sort(dim=0)
        tor_samnodes = values.unique(dim=-1).T

        receivers = torchmesh.x[tor_samnodes[:, 0], :]
        senders = torchmesh.x[tor_samnodes[:, 1], :]
        edge_features = (receivers + senders)/2
        if self.reward_condition:
            beta_batch = torch.zeros(edge_features.shape[0]).to(edge_features.device)
            beta_batch = reward_beta[torchmesh.batch[tor_samnodes[:,0]].to(torch.int64)][:,None]
            edge_features = torch.cat([edge_features,beta_batch.repeat([1,5])],dim=-1).float()
        remesh_logit = (self.decoder_edge_coarse(edge_features)/self.rescale) #predicted logit
        remesh_prob_softmax = torch_geometric.utils.softmax(remesh_logit,
                                                            index=torchmesh.batch[tor_samnodes[:,0]].to(torch.int64),
                                                            num_nodes=self.current_batch) #num,1 

        mask, logp, entropyp = self.sample(remesh_prob_softmax,K=K_coarse,index=torchmesh.batch[tor_samnodes[:,0]].to(torch.int64),is_eval_sample=is_eval_sample) 
        collapsed_edges = torch.masked_select(tor_samnodes, mask).reshape(-1,2)  
        merging_nodes,collapsed_edges = self.filter_2d_coarse(torchmesh,sampled_edges=collapsed_edges)
        if len(merging_nodes) == 0:
            collapsed_edges = torch.tensor([[],[]])
            merging_nodes = torch.tensor([[],[]])
            return collapsed_edges, merging_nodes, [None],[None],[None]
        collapsed_edges = torch.tensor(collapsed_edges, device=torchmesh.x.device).permute(1,0)
        merging_nodes = torch.tensor(merging_nodes, device=torchmesh.x.device).reshape(-1)
        return collapsed_edges, merging_nodes, logp, entropyp, {"coarse_p":to_np_array(torch.exp(logp)),"coarse_topK":to_np_array(K_coarse),"softmax_coarse":to_np_array(remesh_prob_softmax),'action_num':np.array(merging_nodes.shape[0]).reshape(1,1)}
    
    def choose_coarsen_edges_GNN_topk(self, torchmesh,reward_beta=None, K_coarse=None, is_timing=False,is_eval_sample=True):
        coarsened_nodes, snodes_edges_list = self._sample_coarsened_nodes(torchmesh)
        if len(coarsened_nodes) == 0:
            collapsed_edges = torch.tensor([[],[]])
            merging_nodes = torch.tensor([[],[]])
            return collapsed_edges, merging_nodes, [None],[None],[None]
        snodes_edges_tensor = torch.tensor(snodes_edges_list, device=torchmesh.x.device)
        receivers = torchmesh.x[snodes_edges_tensor[..., 0], :]
        senders = torchmesh.x[snodes_edges_tensor[..., 1], :]
        edge_features = (receivers + senders)/2
        if self.reward_condition:
            beta_batch = torch.zeros(edge_features.shape[0:2]).to(edge_features.device)
            beta_batch = reward_beta[torchmesh.batch[snodes_edges_tensor[...,0]].to(torch.int64)][:,:,None]
            edge_features = torch.cat([edge_features,beta_batch.repeat([1,1,5])],dim=-1).float()  
        #get logit of n,4,1
        remesh_logit = (self.decoder_edge_coarse(edge_features)/self.rescale) #predicted logit
        indices = torch.argmax(remesh_logit.reshape(-1,4).permute(1,0), dim=0)
        
        #max logit n,1
        max_remesh_logit = torch.gather(remesh_logit.reshape(-1,4).permute(1,0), 0, indices.reshape(1,-1)).permute(1,0)
        #max edges to sample n,2
        max_prob_edges = torch.gather(snodes_edges_tensor, 1, torch.stack([indices, indices], dim=0).T.reshape(indices.shape[0],1,2)).reshape(indices.shape[0],2)
        #softmax among selected edges
        remesh_prob_softmax = torch_geometric.utils.softmax(max_remesh_logit,
                                                            index=torchmesh.batch[max_prob_edges[:,0]].to(torch.int64),
                                                            num_nodes=self.current_batch) #num,1 
        #sample selected edges
        mask, logp, entropyp = self.sample(remesh_prob_softmax ,K=K_coarse,index=torchmesh.batch[max_prob_edges[:,0]].to(torch.int64),is_eval_sample=is_eval_sample) 
        #get collapsed edges - mask from max_prob_edges
        collapsed_edges = torch.masked_select(max_prob_edges, mask).reshape(-1,2).permute(1,0)
        #get coarsend_nodes - same mask
        merging_nodes = torch.tensor(coarsened_nodes, device=torchmesh.x.device)[mask[:,0]].reshape(-1)
        return collapsed_edges, merging_nodes, logp, entropyp, {"coarse_p":to_np_array(torch.exp(logp)),"coarse_topK":to_np_array(K_coarse),"softmax_coarse":to_np_array(remesh_prob_softmax),'action_num':np.array(merging_nodes.shape[0]).reshape(1,1)}
    
    def coarsen_GNNtopK(self, torchmesh,reward_beta=None,K_coarse=None, is_timing=False,is_eval_sample=True):
        """ Perform coarsen action on mesh according to sampled edges.

        Args:
            torchmesh: mesh data.

        Returns:
            torchmesh: mesh data with coarsened vertices, edges, and faces.
        """
        if torchmesh.xfaces.shape[0] == 0:
            p.print("2.2251", precision="millisecond", is_silent=not is_timing, avg_window=1)
            collapsed_edges, merging_nodes, logp, entropyp, info = self.choose_onedim_coarsen_edges_GNN_topk(torchmesh,reward_beta=reward_beta,K_coarse=K_coarse,is_timing=is_timing,is_eval_sample=is_eval_sample)
            p.print("2.2252", precision="millisecond", is_silent=not is_timing, avg_window=1)
            
            if collapsed_edges.shape[1] == 0:
                return torchmesh, logp, entropyp, info
            try:
                new_edge_index, _ = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
            except:
                pdb.set_trace()
            p.print("2.2253", precision="millisecond", is_silent=not is_timing, avg_window=1)
            new_edge_index, new_x, new_x_phys, new_x_batch, new_x_pos = self.clean_onedim_isolated_vertices(torchmesh, new_edge_index)
            p.print("2.2254", precision="millisecond", is_silent=not is_timing, avg_window=1)
            if not (new_x.shape[0] - int(new_edge_index.shape[1]/2)) == (torchmesh.batch.max()+1):
                print("Coarsen is invalid")
                return torchmesh

            torchmesh.edge_index = new_edge_index
            torchmesh.x = new_x
            torchmesh.x_phy = new_x_phys
            torchmesh.batch = new_x_batch
            torchmesh.x_pos = new_x_pos
            p.print("2.2255", precision="millisecond", is_silent=not is_timing, avg_window=1)
            return torchmesh, logp, entropyp, info
        else:
            old_batch = torchmesh.batch.clone()
            p.print("2.2251", precision="millisecond", is_silent=not is_timing, avg_window=1)
            collapsed_edges, merging_nodes, logp, entropyp, info = self.choose_coarsen_edges_GNN_topk(torchmesh,reward_beta=reward_beta,K_coarse=K_coarse,is_timing=is_timing,is_eval_sample=is_eval_sample)
            
            p.print("2.2252", precision="millisecond", is_silent=not is_timing, avg_window=1)
            # print(collapsed_edges.shape)
            if collapsed_edges.shape[1] == 0:
                # print("no collapsed edges")
                return torchmesh, logp, entropyp, info
            try:
                temp_new_edge_index, mapping = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
            except:
                pdb.set_trace()
            p.print("2.2253", precision="millisecond", is_silent=not is_timing, avg_window=1)
            temp_newly_generated_faces = self.delete_adjacent_faces(torchmesh, mapping)
            p.print("2.2254", precision="millisecond", is_silent=not is_timing, avg_window=1)
            newly_generated_faces = self.face_adjustment(temp_newly_generated_faces, mapping)
            # new_faces, new_edge_index, new_x, new_x_pos, new_x_phy, new_x_batch, new_x_coords= self.clean_isolated_vertices(torchmesh, temp_new_edge_index, newly_generated_faces)
            p.print("2.2255", precision="millisecond", is_silent=not is_timing, avg_window=1)
            new_faces, new_edge_index, new_x, new_x_pos, new_x_phy, new_x_batch, new_x_coords, new_onehot = self.clean_isolated_vertices(torchmesh, temp_new_edge_index, newly_generated_faces)  
            p.print("2.2256", precision="millisecond", is_silent=not is_timing, avg_window=1)
            # pdb.set_trace()

            if not (new_x.shape[0] + new_faces.shape[1] - int(new_edge_index.shape[1]/2)) == torchmesh.batch.max()+1:
                print("Coarsen is invalid")
                pdb.set_trace()
                collapsed_edges, merging_nodes = self.choose_coarsen_edges_GNN_topk(torchmesh,reward_beta=reward_beta,K_coarse=K_coarse,is_timing=is_timing)
                pdb.set_trace()
                new_edge_index, mapping = self.compute_new_edge_index(torchmesh, collapsed_edges, merging_nodes)
                pdb.set_trace()
                newly_generated_faces = self.delete_adjacent_faces(torchmesh, mapping)
                pdb.set_trace()
                newly_generated_faces = self.face_adjustment(newly_generated_faces, mapping)
                pdb.set_trace()
                new_faces, new_edge_index, new_x = self.clean_isolated_vertices(torchmesh, new_edge_index, newly_generated_faces)       

                return torchmesh, logp, entropyp, info

        torchmesh.x = new_x
        torchmesh.x_phy = new_x_phy
        torchmesh.batch_history = old_batch
        torchmesh.batch = new_x_batch
        torchmesh.x_pos = new_x_pos
        torchmesh.edge_index = new_edge_index
        torchmesh.xfaces = new_faces
        torchmesh.x_coords = new_x_coords
        temp_onehots = list(torchmesh.onehot_list)
        temp_onehots[0] = new_onehot
        torchmesh.onehot_list = tuple(temp_onehots)
        # torchmesh.kinematics_list[-1] = new_kinems
        return torchmesh, logp, entropyp, info


    def remeshing_forward_GNN(self, torchmesh, use_pos=False,reward_beta=None,is_timing=False,interp_index=None,is_eval_sample=True,debug=False,data_gt=None,policy_input_feature="velocity",heuristic=False,**kwarg):
        """ Perform remeshing action on mesh data.

        Args:
            torchmesh: pyg data instance for mesh.
            reward_beta: indicate speed accuracy tradeoff

        Returns:
            outmesh: mesh data after gnn forwarding and remeshing.
            logsigmoid_sum: {"action": logprob of shape [batch, 1]}
            entropy_sum: {"action": entropy of shape [batch, 1]}
            remesh_prob: {"action": [prob, node_index , batch_index]}
        """
        
        # if not debug:
        if debug:
            self.skip_coarse = True
            self.skip_flip = True
            self.skip_split = True
        if hasattr(torchmesh, "node_feature"):
            if len(dict(to_tuple_shape(torchmesh.original_shape))["n0"]) == 0:
                batch = torchmesh.batch['n0']
                x_coords = torchmesh['history']['n0'][1] #
                torchmesh = attrdict_to_pygdict(torchmesh, is_flatten=True, use_pos=use_pos)   
                torchmesh.batch = batch
                x_phy = torchmesh.x.clone()
                if policy_input_feature=="velocity":
                    if hasattr(torchmesh, "onehot_list"):
                        onehot = torchmesh.onehot_list[0]
                        raw_kinems = self.compute_kinematics(torchmesh, interp_index)
                        kinematics = torchmesh.onehot_list[0][:,:1] * raw_kinems
                        if interp_index!=0:
                            velocity = torchmesh.history[-1][:,3:] - torchmesh.history[0][:,3:]
                            torchmesh.x = velocity.clone() 
                        torchmesh.x = torch.cat([torchmesh.x, onehot, kinematics], dim=-1)
                elif policy_input_feature=="coords":
                    torchmesh.x = torchmesh.history[-1].clone() 
            else:
                batch = torchmesh.batch.clone()
                torchmesh = deepsnap_to_pyg(torchmesh, is_flatten=True, use_pos=use_pos)
                x_coords = torchmesh.x_pos.clone()
                x_phy = torchmesh.x.clone()
        elif hasattr(torchmesh, "x"):
            batch = torchmesh.batch.clone()
            x_coords = torchmesh.history[-1]
            x_phy = torchmesh.x.clone() 
            if policy_input_feature=="velocity":
                if hasattr(torchmesh, "onehot_list"):
                    index = 0
                    onehot = torchmesh.onehot_list[index]
                    raw_kinems = self.compute_kinematics(torchmesh, interp_index)
                    kinematics = torchmesh.onehot_list[index][:,:1] * raw_kinems
                    if interp_index!=0:
                        velocity = torchmesh.history[-1][:,3:] - torchmesh.history[0][:,3:]
                        torchmesh.x = velocity.clone() 
                    torchmesh.x = torch.cat([torchmesh.x, onehot, kinematics], dim=-1)
            elif policy_input_feature=="coords":
                torchmesh.x = torchmesh.history[-1].clone()

        self.current_batch = int(batch.max()+1)
        self.device = batch.device
        
        if torchmesh.xfaces.shape[0] == 0:
            self.skip_flip = True
        
        if self.dataset.startswith("a"):
            old_xfaces = torch.tensor(torchmesh.xface_list[0]).T
            old_batch = torchmesh.batch_history.clone()
            old_x_coords = torchmesh.history[0].clone()
            

        torchmesh.batch = batch
        torchmesh.x_phy = x_phy
        torchmesh.x_coords = x_coords #torchmesh.x_coords = 6*1 : [x_mesh, y_mesh, z_mesh=0,x_world, y_world, z_world]
        
        #message passing
        if not heuristic:
            torchmesh = self.get_edge_latents(torchmesh,reward_beta=reward_beta) 
        remesh_logprob = {}
        entropies = {}
        info = {}
        if not heuristic:
            graph_latent = self.get_K_latent(torchmesh,reward_beta=reward_beta)
        p.print("2.222", precision="millisecond", is_silent=not is_timing, avg_window=1)
        ########## splits ##########
        if self.skip_split:
            split_outmesh = torchmesh
        elif heuristic:
            split_edges = self.choose_split_edge_GNNtopK_heuristic(torchmesh)
            split_outmesh = self.split(split_edges, torchmesh)
        else:    
            p.print("2.2221", precision="millisecond", is_silent=not is_timing, avg_window=1)
            K_split,logpK_split, ksplit_entropy = self.get_K(graph_latent, reward_beta=reward_beta, mode="split", is_eval_sample=is_eval_sample) #[B,1],[B,1]
            p.print("2.2222", precision="millisecond", is_silent=not is_timing, avg_window=1)
            entropies["split/k_entropy"] = ksplit_entropy.squeeze()
            remesh_logprob["split/k_logp"] = logpK_split.squeeze()
            split_edges, logp, entropy_p, info_split = self.choose_split_edge_GNNtopK(torchmesh,reward_beta=reward_beta,K_split=K_split,is_timing=is_timing, is_eval_sample=is_eval_sample)
            # print("split_edges",split_edges.shape)
            p.print("2.2223", precision="millisecond", is_silent=not is_timing, avg_window=1)
            # split_edges_filtered = self.(split_edges,torchmesh)
            remesh_logprob['split/logp'] = logp.squeeze()
            entropies["split/entropy"] = entropy_p.squeeze()
            info_split["action_num"] = np.array(split_edges.shape[0]).reshape(1,1) #actual number of splits
            info["split"] = info_split
            split_outmesh = self.split(split_edges, torchmesh)
            p.print("2.2224", precision="millisecond", is_silent=not is_timing, avg_window=1)
        
        p.print("2.223", precision="millisecond", is_silent=not is_timing, avg_window=1)
        ########## flip ##########
        if self.skip_flip or heuristic:
            flipped_outmesh = split_outmesh
        else:
            flip_edges = self.flip_all_possible(split_outmesh)
            flipped_outmesh = self.flip(flip_edges, split_outmesh)
                
        ########## coarse ##########
        p.print("2.224", precision="millisecond", is_silent=not is_timing, avg_window=1)
        if self.skip_coarse or heuristic:
            outmesh = flipped_outmesh                
        else:
            K_coarse,logpK_coarse, kcoarse_entropy= self.get_K(graph_latent, reward_beta=reward_beta, mode="coarse", is_eval_sample=is_eval_sample) #[B,1],[B,1]

            entropies["coarse/k_entropy"] = kcoarse_entropy.squeeze()
            remesh_logprob["coarse/k_logp"] = logpK_coarse.squeeze()
        
            p.print("2.225", precision="millisecond", is_silent=not is_timing, avg_window=1)
            # pdb.set_trace()
            outmesh, logp, entropy_p, info_coarse = self.coarsen_GNNtopK(flipped_outmesh,reward_beta=reward_beta,K_coarse=K_coarse,is_timing=is_timing, is_eval_sample=is_eval_sample)
            # pdb.set_trace()
            p.print("2.226", precision="millisecond", is_silent=not is_timing, avg_window=1)
            if logp[0]!=None:
                remesh_logprob['coarse/logp'] = logp.squeeze()
                entropies["coarse/entropy"] = entropy_p.squeeze()
                info["coarse"] = info_coarse
        
        ########## prepare final output ##########
        index = torch.sort(outmesh.batch)[1]
        a = torch.cat([outmesh.x_pos,outmesh.batch.unsqueeze(-1)],dim=-1)
        outmesh.index = torch_lexsort(a.permute(1,0))
        # pdb.set_trace()
        if self.dataset.startswith("arcsi"):
            outmesh.history = (old_x_coords,outmesh.x_coords)
            outmesh.xface_list = (old_xfaces.T, outmesh.xfaces.T)
            outmesh.batch_history = old_batch
        outmesh.x = outmesh.x_phy[:,:]
        p.print("2.227", precision="millisecond", is_silent=not is_timing, avg_window=1)

        if self.dataset.startswith("m"):
            outmesh.dataset = to_tuple_shape(outmesh.dataset)
            #reset bdd nodes, all interpolate boundary node are removed
            outmesh.x[:,-1] = 1*(outmesh.x[:,-1]==1)
            outmesh.x_bdd = outmesh.x[:,-1:]
            outmesh = update_edge_attr_1d(outmesh)
            p.print("2.228", precision="millisecond", is_silent=not is_timing, avg_window=1)
        p.print("2.229", precision="millisecond", is_silent=not is_timing, avg_window=1)

        if (self.skip_coarse and self.skip_split) or heuristic:
            remesh_logprob = {'coarse/logp':torch.zeros(self.batch_size,device=torchmesh.batch.device)}
            entropies = {'coarse/entropy':torch.zeros(self.batch_size,device=torchmesh.batch.device)}

        return outmesh, remesh_logprob, entropies, info
      
    @property
    def model_dict(self):
        model_dict = {"type": "GNNPolicyAgent_Sampling"}
        model_dict["input_size"] = self.input_size
        model_dict['share_processor'] = self.share_processor
        model_dict["nmax"] = to_np_array(self.nmax)
        model_dict["edge_dim"] = self.edge_dim
        model_dict["latent_dim"] = self.latent_dim
        model_dict["edge_attr"] = self.edge_attr
        model_dict["layer_norm"] = self.layer_norm
        model_dict["batch_norm"] = self.batch_norm
        model_dict["num_steps"] = self.num_steps
        model_dict["val_layer_norm"] = self.val_layer_norm
        model_dict["val_batch_norm"] = self.val_batch_norm
        model_dict["val_num_steps"] = self.val_num_steps
        model_dict["val_pooling_type"] = self.val_pooling_type
        model_dict["use_pos"] = self.use_pos
        model_dict["final_ratio"] = self.final_ratio
        model_dict["edge_threshold"] = self.edge_threshold
        model_dict["final_pool"] = self.final_pool
        model_dict["val_act_name"] = self.val_act_name
        model_dict["val_act_name_final"] = self.val_act_name_final
        
        model_dict["rl_num_steps"] = self.rl_num_steps
        model_dict["rl_layer_norm"] = self.rl_layer_norm
        model_dict["rl_batch_norm"] = self.rl_batch_norm
        model_dict["skip_split"] = self.skip_split
        model_dict["skip_flip"] = self.skip_flip
        model_dict["skip_coarse"] = self.skip_coarse
        model_dict["act_name"] = self.act_name
        model_dict["dataset"] = self.dataset
        model_dict["samplemode"] = self.samplemode

        model_dict["min_edge_size"] = self.min_edge_size
        model_dict["rescale"] = self.rescale
        model_dict["batch_size"] = self.batch_size
        model_dict["is_single_action"] = self.is_single_action
        model_dict["processor_aggr"] = self.processor_aggr

        model_dict["top_k_action"] = self.top_k_action
        model_dict["max_action"] = self.max_action
        model_dict["reward_condition"] = self.reward_condition
        model_dict["offset_split"] = to_np_array(self.offset_split)
        model_dict["offset_coarse"] = to_np_array(self.offset_coarse)
        model_dict["kaction_pooling_type"] = self.kaction_pooling_type

        model_dict["state_dict"] = to_cpu(self.state_dict())
        
        return model_dict

       
    # In[ ]:


def generate_mesh(vers,faces,tarvers):
    mesh = dolfin.Mesh()
    editor = dolfin.MeshEditor()
    editor.open(mesh, 'triangle', 2, 2)
    editor.init_vertices(vers.shape[0])
    for i in range(vers.shape[0]):
        editor.add_vertex(i, vers[i,:2].cpu().numpy())
    editor.init_cells(faces.shape[0])
    for f in range(faces.shape[0]):
        editor.add_cell(f, faces[f].cpu().numpy())
    editor.close()
    bvh_tree = mesh.bounding_box_tree()
    return mesh, bvh_tree

def generate_baryweight_core(vers,faces,tarvers,mesh,bvh_tree):
    faces = []
    weights = []
    # pdb.set_trace()
    for query in tarvers:
        # p.print("99991", precision="millisecond", is_silent=False, avg_window=1)
        face = bvh_tree.compute_first_entity_collision(dolfin.Point(query))
        # p.print("99992", precision="millisecond", is_silent=False, avg_window=1)
        while (mesh.num_cells() <= face):
            #print("query: ", query)
            if query[0] < 0.5:
                query[0] += 1e-15
            elif query[0] >= 0.5:
                query[0] -= 1e-15
            if query[1] < 0.5:
                query[1] += 1e-15
            elif query[1] >= 0.5:
                query[1] -= 1e-15            
            face = bvh_tree.compute_first_entity_collision(dolfin.Point(query))
        # p.print("99993", precision="millisecond", is_silent=False, avg_window=1)
        faces.append(face)
        face_coords = mesh.coordinates()[mesh.cells()[face]]
        mat = face_coords.T[:,[0,1]] - face_coords.T[:,[2,2]]
        const = query - face_coords[2,:]
        # p.print("99994", precision="millisecond", is_silent=False, avg_window=1)
        try:
            weight = np.linalg.solve(mat, const)
        except:
            weight = np.array([1/3,1/3])
        final_weights = np.concatenate([weight, np.ones(1) - weight.sum()], axis=-1)
        weights.append(final_weights)
    return faces,weights, mesh
                       
def generate_baryweight(vers, faces, tarvers):
    mesh, bvh_tree = generate_mesh(vers,faces,tarvers)
    faces, weights, mesh = generate_baryweight_core(vers,faces,tarvers,mesh,bvh_tree)
    # p.print("99995", precision="millisecond", is_silent=False, avg_window=1)
    # pdb.set_trace()
    return faces, weights, mesh

def generate_barycentric_interpolated_data(vers, faces, outvec, tarvers):
    faces, weights, mesh = generate_baryweight(vers, faces, tarvers)
    indices = mesh.cells()[faces].astype('int64')
    fweights = torch.tensor(np.array(weights), device=outvec.device, dtype=torch.float32)
    return torch.matmul(fweights, outvec[indices,:]).diagonal().T# [:,:self.output_size].reshape(-1,1,self.output_size)

class Value_Model_Summation(GNNRemesher):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        kwargs["output_size"] = 2
        super().__init__(*args, **kwargs)

    def forward(self, data, reward_beta):
        """Given the data (graph) and a reward_beta, return a value estimating the cumulative expected reward.

        Args:
            data: graph
            reward_beta: float, nonnegative
        """
        pred, _ = super(Value_Model_Summation, self).forward(data, pred_steps=1)
        value_raw = pred["n0"]
        value = value_raw[:,0].mean() + value_raw[:,1].sum()/10000
        return value

    @property
    def model_dict(self):
        model_dict = super().model_dict
        model_dict["type"] = "Value_Model_Summation"
        return model_dict

def torch_lexsort(a, dim=-1):
    assert dim == -1  # Transpose if you want differently
    assert a.ndim == 2  # Not sure what is numpy behaviour with > 2 dim
    # To be consistent with numpy, we flip the keys (sort by last row first)
    a_unq, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)
    return torch.argsort(inv)

def batch_logsigmoid_sum(prob, index, batch_size, eps=1e-8):
    x = torch.log(prob+eps)
    if torch.isnan(x).any():
        print(x)
        pdb.set_trace()
    idx = index.to(torch.int64)
    idx_unique = idx.unique(sorted=True)
    
    idx_unique_count = torch.zeros(batch_size, device=x.device).to(torch.int64)
    idx_unique_counting = torch.stack([(idx==idx_u).sum() for idx_u in idx_unique])
    idx_unique_count[idx_unique] = idx_unique_counting
    res = torch.zeros(batch_size, device=x.device).scatter_add(0, idx, x)
    res /= torch.maximum(idx_unique_count.float(),torch.tensor(1))
    if torch.isnan(res).any():
        print(res)
        pdb.set_trace()
    return res

def batch_entropy_sum(prob, index, batch_size, single=False, eps=1e-8):
    if single:
        x=-(prob*torch.log(prob+eps))
    else:
        x=-(prob*torch.log(prob+eps) + (1-prob)*torch.log(1-prob+eps))
    if torch.isnan(x).any():
        print(x)
        pdb.set_trace()
    idx = index.to(torch.int64)
    idx_unique = idx.unique(sorted=True)
    idx_unique_count = torch.zeros(batch_size, device=x.device).to(torch.int64)
    idx_unique_counting = torch.stack([(idx==idx_u).sum() for idx_u in idx_unique])
    idx_unique_count[idx_unique] = idx_unique_counting
    res = torch.zeros(batch_size).to(x.device).scatter_add(0, idx, x)
    res /= torch.maximum(idx_unique_count.float(),torch.tensor(1))
    if torch.isnan(res).any():
        print(res)
        pdb.set_trace()
    return res

def get_remesh_prob_softmax(prob,batch,batch_size,k=10):
    """
    prob: N*1
    batch: N
    """
    prob = prob.squeeze()
    prob_exp = torch.exp(prob).to(prob.device)
    idx = batch.to(torch.int64).to(prob.device)
    idx_unique = idx.unique(sorted=True)
    res = torch.zeros(batch_size, device=prob.device).scatter_add(0, idx, prob_exp) #[b,1], sum_i exp(p_i)
    # sigmoid_per_node = torch.zeros(batch.shape, device=prob.device) #[num 1]
    sigmoid_per_node = res[batch.to(torch.int64)] #[num,1]
    sigmoid_per_node = prob_exp/sigmoid_per_node
    sigmoid_per_node_copy = sigmoid_per_node.clone()
    max_prob, max_idx = torch_scatter.scatter_max(sigmoid_per_node_copy, idx)
    max_prob_batch = torch.zeros(batch.shape, device=prob.device)-1 #[num,1]
    max_prob_batch[max_idx] = max_prob
    for i in range(k-1):
        sigmoid_per_node_copy[max_idx] = -100
        #add remornalize 
        max_prob, max_idx = torch_scatter.scatter_max(sigmoid_per_node_copy, idx)
        max_prob_batch[max_idx] = max_prob
    return sigmoid_per_node,max_prob_batch
# ## Test:

# In[ ]:


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..', '..'))

    args = init_args({
        "dataset": "mppde1de-E2-100",
        "n_train": ":300",
        "train_fraction": float(8/9),
        "multi_step": "1",
        "latent_multi_step": "1",
        "is_ebm": False,
        "input_steps": 1,
        "temporal_bundle_steps": 25,
        "time_interval": 1,
        "is_y_variable_length": False,
        "data_noise_amp": 0,
        "is_test_only": False,
        "is_y_diff": False,
        "n_workers": 0,
        "batch_size": 32,
        "val_batch_size": 32,
        "dataset_split_type": "random",
        "algo": "gnnremesher",
    })
    (dataset_train_val, dataset_test), (train_loader, val_loader, test_loader) = load_data(args)
    data = dataset_test[0]
    data1 = deepsnap_to_pyg(data, is_flatten=True, use_pos=False)

    device = "cpu"
    model = GNNRemesher(
        input_size=26,
        output_size=25,
        edge_dim=18,
        sizing_field_dim=0,
        nmax=torch.tensor(10000),
        use_encoder=True,
        is_split_test=False,
        is_flip_test=False,
        is_coarsen_test=False,
        skip_split=False,
        skip_flip=False,
        is_y_diff=False
    ).to(device)
    out = model(data1)



if __name__ == "__main__":
    device = "cuda:0"
    policy_2 = GNNRemesher(
        input_size=9,
        output_size=6,
        edge_dim=18,
        sizing_field_dim=4,
        nmax=torch.tensor(10000),
        use_encoder=True, 
        is_split_test=False, 
        is_flip_test=False, 
        is_coarsen_test=False, 
        skip_split=False, 
        skip_flip=False,
        is_y_diff=False,
    ).to(device)

    # load two-dimensional data:
    dataset_twodim = ArcsimMesh(input_steps=2, traj_len=325)
    idx2 = np.random.randint(0, 300000-1)
    
   # apply gnn_forward:
    # evolved_mesh2 = policy.gnn_forward(data2, train=True)
    # Mock output's mesh 
    data2 = dataset_twodim[idx2].to(device)
    data2.x = torch.cat(
        [data2.x.reshape(-1,9), torch.arange(4*data2.x.shape[0]).reshape(-1,4).to(data2.x.device)], 
        dim=-1)
    evolved_mesh2 = data2
    print(evolved_mesh2)
    
    # apply forward:
    print(policy_2.remeshing_forward(evolved_mesh2, train=False, use_pos=False))

    policy_1 = GNNRemesher(
        1, 
        1,
        edge_dim=2,
        sizing_field_dim=1,
        nmax=torch.tensor(10000), 
        use_encoder=True, 
        is_split_test=False, 
        is_flip_test=False, 
        is_coarsen_test=False, 
        skip_split=False, 
        skip_flip=False
        ).to(device)
    
    # load one-dimensional data:
    from mppde1d_dataset import MPPDE1D
    dataset_onedim = MPPDE1D(
        dataset="mppde1d-E2-100",
        input_steps=1,
        output_steps=1,
        time_interval=1,
        is_y_diff=False,
        split="valid",
        transform=None,
        pre_transform=None,
        verbose=False,
    )
    idx1 = np.random.randint(0, 300-1)
    data1 = dataset_onedim[idx1].to(device)
    
    # apply gnn_forward:
    # evolved_mesh1 = policy.gnn_forward(data1, train=True)
    # Mock output's segment
    data1.x = data1.x.reshape(-1,1)
    data1.x = torch.cat(
        [data1.x, data1.x_pos.reshape(-1, 1), #torch.zeros(torchmesh.x_pos.shape[0], 1), 
         (torch.arange(data1.x.shape[0]).reshape(-1,1)*2).to(data1.x.device)], dim=-1)
    evolved_mesh1 = data1
    print(evolved_mesh1)

    # apply forward:
    print(policy_1.remeshing_forward(evolved_mesh1, train=False, use_pos=True)) 


# In[ ]:




