#!/usr/bin/env python
# coding: utf-8

import datetime
import gc
import matplotlib.pylab as plt
from numbers import Number
import numpy as np
import pdb
import pickle
import random
import scipy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd.functional import jvp
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch_geometric.nn.inits import reset
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import xarray as xr
from tqdm import tqdm
import matplotlib
import math
try:
    import dolfin
except:
    print("Cannot import dolfin. If dolfin is not needed, please ignore this message.")

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from lamp.gnns import GNNRemesher, Value_Model_Summation, Value_Model, GNNRemesherPolicy, GNNPolicySizing, GNNPolicyAgent, get_reward_batch, get_data_dropout, GNNPolicyAgent_Sampling
from lamp.datasets.arcsimmesh_dataset import get_2d_data_pred
from lamp.datasets.mppde1d_dataset import get_data_pred, update_edge_attr_1d
from lamp.utils_model import get_conv_func, get_conv_trans_func, get_Hessian_penalty
from lamp.pytorch_net.util import get_repeat_interleave, Printer, forward_Runge_Kutta, tuple_add, tuple_mul, clip_grad, Batch, make_dir, to_np_array, record_data, make_dir, ddeepcopy as deepcopy, filter_filename, Early_Stopping, str2bool, get_filename_short, print_banner, plot_matrices, to_string, init_args, get_poly_basis_tensor, get_num_params, get_pdict
from lamp.utils import PDE_PATH, EXP_PATH
from lamp.utils import sample_reward_beta, copy_data, requires_grad, endow_grads, process_data_for_CNN, get_regularization, get_batch_size
from lamp.utils import detach_data, get_model_dict, loss_op_core, MLP, MLP_Coupling, get_keys_values, flatten, get_activation, to_cpu, to_tuple_shape, parse_multi_step, parse_act_name, parse_reg_type, loss_op, get_normalization, add_noise, get_neg_loss, get_pos_dims_dict
from lamp.utils import p, seed_everything, is_diagnose, get_precision_floor, parse_string_idx_to_list, parse_loss_type, get_loss_ar, get_max_pool, get_data_next_step, expand_same_shape, add_data_noise
from lamp.pytorch_net.util import Attr_Dict, set_seed, pdump, pload, get_time, check_same_model_dict, Zip, Interp1d_torch
from deepsnap.hetero_graph import HeteroGraph
from deepsnap.hetero_gnn import forward_op
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch as deepsnap_Batch


def get_grid_change(x_pos,x_pos_prev):
    """
    x_pos: n
    x_pos_prev:m 
    """

    x_pos = x_pos[:,0]
    x_pos_prev = x_pos_prev[:,0]
    nm = x_pos[:,None].repeat(x_pos_prev.shape[0],-1) #nm
    mn = x_pos_prev[None,:].repeat(x_pos.shape[0],0) #nm
    a = np.nonzero(((np.abs(nm-mn)<1e-6).sum(axis=0))==0) # m
    b = np.nonzero(((np.abs(nm-mn)<1e-6).sum(axis=1))==0) # n
    return x_pos[b], x_pos_prev[a]
    
def plot_mesh(ax0,pos,face,a=10):
    # ax0.set_axis_off()
    ax0.set_xlim([-0.5, 0.5])
    ax0.set_ylim([-0.5, 0.5])
    ax0.set_zlim([-0.5, 0.1])
    ax0.view_init(a, 40)
    ax0.plot_trisurf(pos[:, 3], pos[:, 4], face, pos[:, 5], shade=True, linewidth = 0.5, edgecolor = 'grey', color="ghostwhite")

        
def plot_2dmesh(ax0,pos,face,a=10):
    ax0.set_xlim([0, 1])
    ax0.set_ylim([0, 1])
    ax0.triplot(pos[:, 0], pos[:, 1], face)

    
def plot_2d_state(wandb,state_info_all,data_clone,sample_idx=0,step_num=0,prefix="train",a=40,traj_index=None):
    total_states = len(state_info_all)
    if (total_states)>50:
        steps = np.arange(0,total_states,10)
    elif (total_states)>10:
        steps = np.arange(0,total_states,2)
    else:
        steps = np.arange(0,total_states)
    total_states = steps.shape[0]
    figure = plt.figure(figsize=(3*total_states,10))
    for i, sample_idx in enumerate(steps):
        state_info_t1 = state_info_all[sample_idx]
        
        ##########check gt#########
        current_time = np.maximum(sample_idx-1,0)
        vers_gt = data_clone.reind_yfeatures["n0"][current_time].detach().cpu().numpy()
        faces_gt = data_clone.yface_list["n0"][current_time]

        faces_gt = np.stack(faces_gt)
        batch_0_idx = np.where(vers_gt[faces_gt[:,0],0]<1.5)
        faces_gt = np.stack(faces_gt)[batch_0_idx]

        batch_0_idx = np.where(vers_gt[:,0]<1.5)
        vers_gt = vers_gt[batch_0_idx][:,:]
        
        ##########check predict#########
        if hasattr(state_info_t1, "node_feature"):
            batch = state_info_t1['batch']['n0'].detach()
            faces = (state_info_t1['xfaces']['n0'].T).detach().cpu().numpy()
            pos = state_info_t1['history']['n0'][-1][:,:]
        elif hasattr(state_info_t1, "x"):
            batch = state_info_t1.batch.detach()
            faces = (state_info_t1.xfaces.T).detach().cpu().numpy()
            pos = state_info_t1.history[-1][:,:].detach()
        
        # pdb.set_trace()
        batch_0_idx = torch.where(batch!=0)[0]
        pos[batch_0_idx] = -10
        pos = pos.detach().cpu().numpy()
        
        ax0= figure.add_subplot(4,total_states,i+1, projection='3d')
        plot_mesh(ax0,pos,faces,a=a)
    
        ax0= figure.add_subplot(4,total_states,i+total_states+1, projection='3d')
        plot_mesh(ax0,vers_gt,faces_gt,a=a)
        
        ax1= figure.add_subplot(4,total_states,i+total_states*2+1)
        plot_2dmesh(ax1,pos,faces,a=a)
    
        ax1= figure.add_subplot(4,total_states,i+total_states*3+1)
        plot_2dmesh(ax1,vers_gt,faces_gt,a=a)

    figure.tight_layout()
    frame1 = figure.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])

    wandb.log({"plots_val/plot_t_view{}_{}_{}".format(str(a),prefix,traj_index): wandb.Image(figure)},step=step_num)
    plt.close('all')
    
def plot_1d_state(wandb,state_info_all,data_clone,sample_idx=0,step_num=0,prefix="train"):
    total_states = len(state_info_all)
    figure = plt.figure(figsize=(4*total_states,8))
    for sample_idx in range(total_states):
        state_info_t0 = state_info_all[sample_idx-1] if sample_idx>0 else None
        state_info_t1 = state_info_all[sample_idx]
                
        batch = state_info_t1['batch']
        x_pos = state_info_t1['x_pos']
        pred = state_info_t1['x']

        x_pos_gt = data_clone.node_pos['n0']
        gt =  data_clone.node_label['n0']
        batch_gt = data_clone.batch
        batch_idx = torch.where(batch==0)[0]    
        batch_idx_gt = torch.where(batch_gt==0)[0]    
        pred = pred[batch_idx].clone().detach().cpu().numpy()
        current_time = pred.shape[1]*np.maximum(sample_idx-1,0)
        gt = gt[batch_idx_gt,current_time:current_time+pred.shape[1]].clone().detach().cpu().numpy()
        x_axis = x_pos[batch_idx].clone().detach().cpu().numpy()
        x_axis_gt = x_pos_gt[batch_idx_gt].clone().detach().cpu().numpy()
        x_axis_idx = np.argsort(x_axis,axis=0)
        x_axis = x_axis[x_axis_idx][:,:,0]

        if state_info_t0==None:
            x_axis_prev = x_axis_gt
        else:
            x_pos_prev = state_info_t0['x_pos']
            batch_idx_prev = torch.where(state_info_t0['batch']==0)[0]    
            x_axis_prev = x_pos_prev[batch_idx_prev].clone().detach().cpu().numpy()
            x_axis_prev_idx = np.argsort(x_axis_prev,axis=0)
            x_axis_prev = x_axis_prev[x_axis_prev_idx][:,0]
        # give a x_pos and x_pos_prev, get set of removed nodes, and added node
        # node removed means nodes in x_pos_prev not in x_pos
        node_removed,node_added = get_grid_change(x_axis,x_axis_prev)


        fontsize = 14
        idx_list = np.arange(0, pred.shape[1], 5)
        color_list = np.linspace(0.01, 0.9, len(idx_list))
        cmap = matplotlib.cm.get_cmap('jet')
        
        plt.subplot(2,total_states,sample_idx+1)
        for i, idx in enumerate(idx_list):
            pred_i = to_np_array(pred[x_axis_idx,idx,:].squeeze())
            rgb = cmap(color_list[i])[:3]
            plt.plot(x_axis,pred_i, color=rgb, label=f"t={np.round(i*0.3, 1)}s",marker="*",linewidth=0.5)
        plt.ylabel("u(t,x)", fontsize=fontsize)
        plt.xlabel("x", fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
        plt.ylim([-2.5,2.5])
        plt.title("Prediction")
        for node in node_removed:
            plt.scatter(x = node,y=2,color="r",marker="v")
        for node in node_added:
            plt.scatter(x = node,y=-2,color="b",marker="s")
        plt.subplot(2,total_states,total_states+sample_idx+1)
        for i, idx in enumerate(idx_list):
            try:
                y_i = to_np_array(gt[...,idx,:])
            except:
                pdb.set_trace()
            rgb = cmap(color_list[i])[:3]
            plt.plot(x_axis_gt, y_i, color=rgb, label=f"t={np.round(i*0.3, 1)}s",marker="*",linewidth=0.5)

        plt.ylabel("u(t,x)", fontsize=fontsize)
        plt.xlabel("x", fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
        plt.ylim([-2.5,2.5])
        plt.title("Ground-truth")

    
    
    wandb.log({"{}_1d_plot_t".format(prefix): wandb.Image(figure)},step=step_num)
    plt.close("all")
    return 
    
    
def log_forward_wandb(wandb,info,data_clone,step_num,plot,prefix,is_timing=0,single=False,args=None,traj_index=0):

    if wandb==None:
        return 
    else:
        state_preds = info['state_preds']
        state_preds_alt = info['state_preds_alt']
        if 'state_preds_fine' in info.keys():
            state_preds_fine = info['state_preds_fine']
        if 'state_preds_interp_alt' in info.keys():
            state_preds_interp_alt = info['state_preds_interp_alt']
        t_evolve = torch.tensor(info['t_evolve'])
        for key in [ "reward_preds", "rewards", "rewards_interp_alt",
                    "evolution/loss", "evolution/loss_alt", "evolution/loss_interp_alt","evolution/loss_fine",
                    "evolution/loss_mse", "evolution/loss_alt_mse", "evolution/loss_interp_alt_mse", "evolution/loss_fine_mse",
                    "r/lossdiff", "r/statediff", "r/timediff",
                    "v/lossdiff", "v/statediff", "v/timediff",
                    "v/beta", "interp_v/beta",
                    "interp_r/lossdiff", "interp_r/statediff", "interp_r/timediff",
                    "interp_v/lossdiff", "interp_v/statediff", "interp_v/timediff", 
                    "v/state_size", "interp_v/state_size",
                    "logpK_split","logpK_coarse","logpK_flip",
                    "ksplit_entropy","kcoarse_entropy","kflip_entropy",
                    "split_edge_remesh_logp","split_edge_remesh_entropy",
                    "coarse_logp","coarse_entropy",
                    "split_edge_remesh_prob", "coarse_prob","flip_edge_remesh_prob",
                    "split/k_entropy","coarse/k_entropy",
                    "split/k_logp","coarse/k_logp",
                    "split/logp","coarse/logp",
                    "split/entropy", "coarse/entropy",]:
            if key in info:
                value = info[key]
                assert isinstance(value, torch.Tensor)
                wandb.log({prefix+f'{key}_mean': value.mean().item()}, step=step_num)
                wandb.log({prefix+f'{key}_std': value.std().item()}, step=step_num)
                # for jj in range(len(value)):
                #     wandb.log({prefix+f'{key}_mean_{jj}': value[jj].mean().item()}, step=step_num)
                #     wandb.log({prefix+f'{key}_std_{jj}': value[jj].std().item()}, step=step_num)
        action_logprobs = info['action_logprobs']
        action_entropies = info['action_entropies']
        action_probs = info['action_probs']
        wandb.log({prefix+'action/logprobs_mean': action_logprobs.mean()}, step=step_num)
        wandb.log({prefix+'action/logprobs_std': action_logprobs.std()}, step=step_num)
        wandb.log({prefix+'action/entropies_mean': action_entropies.mean()}, step=step_num)
        wandb.log({prefix+'action/entropies_std': action_entropies.std()}, step=step_num)

        wandb.log({prefix+'t_evolve': t_evolve.mean()}, step=step_num)
        total_state = len(state_preds)
        p.print("3.2", precision="millisecond", is_silent=is_timing<2, avg_window=1)
        
        # sampled_states = np.linspace(0,total_state-1,max(4,total_state))
        for sample_idx in range(total_state):
            sample_idx = int(np.floor(sample_idx))
            if sample_idx!=len(action_probs) and not (args.algo.startswith("srl")):
                if not single:
                    for key in action_probs[sample_idx].keys():
                        wandb.log({prefix+'action/{}_{}_mean'.format(key, sample_idx): action_probs[sample_idx][key][0].mean().item() if not isinstance(action_probs[sample_idx][key][0], list) else torch.tensor(0.)}, step=step_num)
                        wandb.log({prefix+'action/{}_{}_std'.format(key,sample_idx): action_probs[sample_idx][key][0].std().item() if not isinstance(action_probs[sample_idx][key][0], list) else torch.tensor(0.)}, step=step_num)
                        wandb.log({prefix+'action/{}_{}_num'.format(key, sample_idx): action_probs[sample_idx][key][0].shape[0] if not isinstance(action_probs[sample_idx][key][0], list) else torch.tensor(0.)}, step=step_num)
                        wandb.log({prefix+'action/{}_{}_hist'.format(key, sample_idx): wandb.Histogram(action_probs[sample_idx][key][0].clone().detach().cpu()) if not isinstance(action_probs[sample_idx][key][0], list) else torch.tensor(0.)}, step=step_num)
                else:
                    for key in action_probs[sample_idx].keys():
                        wandb.log({prefix+'action/{}_{}_mean'.format(key, sample_idx): action_probs[sample_idx][key][-1][np.nonzero(action_probs[sample_idx][key][-1]>0)].mean().item() if not isinstance(action_probs[sample_idx][key][0], list) else torch.tensor(0.)}, step=step_num)
                        wandb.log({prefix+'action/{}_{}_std'.format(key,sample_idx): action_probs[sample_idx][key][-1][np.nonzero(action_probs[sample_idx][key][-1]>0)].std().item() if not isinstance(action_probs[sample_idx][key][0], list) else torch.tensor(0.)}, step=step_num)
                        wandb.log({prefix+'action/{}_{}_num'.format(key, sample_idx): action_probs[sample_idx][key][-1].shape[0] if not isinstance(action_probs[sample_idx][key][0], list) else torch.tensor(0.)}, step=step_num)
                        
            elif sample_idx!=len(action_probs) and (args.algo.startswith("srl")) and args.dataset.startswith("a"):
                for key in action_probs[sample_idx].keys():
                    for key_in in action_probs[sample_idx][key].keys():
                        wandb.log({prefix+'action/{}_{}_{}_mean'.format(key, key_in, sample_idx): action_probs[sample_idx][key][key_in].mean().item() if not isinstance(action_probs[sample_idx][key][key_in], list) else torch.tensor(0.)}, step=step_num)
                        wandb.log({prefix+'action/{}_{}_{}_std'.format(key, key_in, sample_idx): action_probs[sample_idx][key][key_in].std().item() if not isinstance(action_probs[sample_idx][key][key_in], list) else torch.tensor(0.)}, step=step_num)
                        # wandb.log({prefix+'prob_{}_{}_{}_num'.format(key, key_in, sample_idx): action_probs[sample_idx][key][key_in].shape[0] if not isinstance(action_probs[sample_idx][key][key_in], list) else torch.tensor(0.)}, step=step_num)
                        wandb.log({prefix+'action/{}_{}_{}_hist'.format(key, key_in, sample_idx): wandb.Histogram(to_np_array(action_probs[sample_idx][key][key_in])) if not isinstance(action_probs[sample_idx][key][key_in], list) else torch.tensor(0.)}, step=step_num)
                        # if key_in.startswith("softmax"):
                        #     wandb.log({prefix+'action/{}_{}_{}_max'.format(key, key_in, sample_idx): action_probs[sample_idx][key][key_in].max().item() if not isinstance(action_probs[sample_idx][key][key_in], list) else torch.tensor(-1)}, step=step_num)
                
                
        if plot and args.dataset.startswith("m"):
            # state_preds_prev = state_preds[sample_idx-1] if sample_idx>0 else None
            # state_preds_alt_prev = state_preds_alt[sample_idx-1] if sample_idx>0 else None
            # if 'state_preds_interp_alt' in info.keys():
            #     state_preds_interp_alt_prev = state_preds_interp_alt[sample_idx-1] if sample_idx>0 else None
            p.print(f"3.3{sample_idx}log", precision="millisecond", is_silent=is_timing<2, avg_window=1)
            plot_1d_state(wandb, state_preds,data_clone, sample_idx, step_num,prefix)
            if 'state_preds_interp_alt' in info.keys():
                plot_1d_state(wandb, state_preds_interp_alt,data_clone, sample_idx, step_num,"alt_inter"+prefix)
            plot_1d_state(wandb, state_preds_alt,data_clone, sample_idx, step_num,"alt"+prefix)
            p.print(f"3.3{sample_idx}plot", precision="millisecond", is_silent=is_timing<2, avg_window=1)
        elif plot and args.dataset.startswith("a"):
            p.print(f"3.3{sample_idx}log", precision="millisecond", is_silent=is_timing<2, avg_window=1)
            plot_2d_state(wandb, state_preds,data_clone, sample_idx, step_num,prefix,a=10,traj_index=traj_index)
            if 'state_preds_interp_alt' in info.keys():
                plot_2d_state(wandb, state_preds_interp_alt,data_clone, sample_idx, step_num,"alt_inter"+prefix,traj_index=traj_index)
            plot_2d_state(wandb, state_preds_alt,data_clone, sample_idx, step_num,"alt"+prefix,a=10,traj_index=traj_index)
            p.print(f"3.3{sample_idx}plot", precision="millisecond", is_silent=is_timing<2, avg_window=1)
              
            p.print(f"3.3{sample_idx}log", precision="millisecond", is_silent=is_timing<2, avg_window=1)
            plot_2d_state(wandb, state_preds,data_clone, sample_idx, step_num,prefix,a=40,traj_index=traj_index)
            if 'state_preds_interp_alt' in info.keys():
                plot_2d_state(wandb, state_preds_interp_alt,data_clone, sample_idx, step_num,"alt_inter"+prefix,traj_index=traj_index)
            # if 'state_preds_fine' in info.keys():
            #     plot_2d_state(wandb, state_preds_fine,data_clone, sample_idx, step_num,"fine"+prefix,traj_index=traj_index)
            plot_2d_state(wandb, state_preds_alt,data_clone, sample_idx, step_num,"alt"+prefix,a=40,traj_index=traj_index)
            p.print(f"3.3{sample_idx}plot", precision="millisecond", is_silent=is_timing<2, avg_window=1)


def log_value_wandb(wandb,info,data_clone,step_num,prefix):
    if wandb==None:
        return 
    else:
        value_loss,value_preds,value_targets = info
        wandb.log({prefix+'value_loss_mean': value_loss.mean()}, step=step_num)
        wandb.log({prefix+'value_loss_std': value_loss.std()}, step=step_num)
        wandb.log({prefix+'value_preds_mean': value_preds.mean()}, step=step_num)
        wandb.log({prefix+'value_preds_std': value_preds.std()}, step=step_num)
        wandb.log({prefix+'value_targets_mean': value_targets.mean()}, step=step_num)
        wandb.log({prefix+'value_targets_std': value_targets.std()}, step=step_num)

def log_actor_wandb(wandb,info,data_clone,step_num,prefix):
    if wandb==None:
        return 
    else:
        actor_loss,action_info = info
        wandb.log({prefix+'actor_loss_mean': actor_loss.mean()}, step=step_num)
        wandb.log({prefix+'actor_loss_std': actor_loss.std()}, step=step_num)
        for key in action_info.keys():
            wandb.log({prefix+'action_info{}'.format(key): action_info[key]}, step=step_num)


class Actor_Critic(nn.Module):
    def __init__(
        self,
        actor,
        critic,
        critic_target=None,
        horizon=4,
    ):
        """
        Inspired from Dreamer v2 (Hafner et al. 2021)
        https://github.com/danijar/dreamerv2/blob/main/dreamerv2/agent.py
        
        """
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.device = list(actor.parameters())[0].device
        self.copy_critic()
        if critic_target is not None:
            self.critic_target = critic_target
        assert horizon >= 1
        self.horizon = horizon
        self.critic_iteration_count = 0
        self.train_value_count = 0
        self.train_actor_count = -1

    def get_tested(self, data_loader, args, wandb=None, step_num=0, **kwargs):
        if wandb!=None:
            self.actor.eval()
            self.critic.eval()

            #sample_diff_beta
            data_record_val = {}
            if len(args.reward_beta.split(":")) == 1:
                value_str, mode = args.reward_beta, "linear"
            else:
                value_str, mode = args.reward_beta.split(":")
            if len(value_str.split("-")) == 1:
                min_value, max_value = eval(value_str), eval(value_str)
            else:
                min_value, max_value = value_str.split("-")
                min_value, max_value = eval(min_value), eval(max_value)
            assert min_value <= max_value
            if max_value-min_value>0:
                sampled_beta = torch.linspace(min_value,max_value,3)
            else:
                sampled_beta = torch.tensor(min_value).reshape([1,1]).to(args.device)
            state_sizes = []
            evolution_losses = []
            for beta in sampled_beta:
                evolution_loss = 0 
                evolution_loss_alt = 0 
                evolution_loss_mse = 0
                evolution_loss_alt_mse = 0
                r_statediff = 0
                v_statediff = 0
                rewards = 0
                count = 0
                state_size = 0
                for j, data in enumerate(data_loader):
                    if args.dataset.startswith("m") and j % 26 == 25:
                        count += 1
                        if data.__class__.__name__ == "Attr_Dict":
                            data = data.to(args.device)
                        else:
                            data.to(args.device)
                        info, data_clone = self.get_loss(
                            data,
                            args,
                            wandb=wandb,
                            step_num=step_num,
                            mode="test",
                            beta=beta,
                            is_gc_collect=False,
                            **kwargs
                        )
                        evolution_loss += info['evolution/loss'].mean().item() #[h,b,1] where [i,:,:] = mse sum over 25 average over all nodes
                        evolution_loss_alt += info['evolution/loss_alt'].mean().item()
                        r_statediff += info['r/statediff'].mean().item()
                        v_statediff += info['v/statediff'].mean().item()
                        rewards += info['rewards'].mean().item()
                        state_size += info['v/state_size'].mean().item()
                        if args.dataset.startswith("m") and j in [25,25+26,25+26*2]:
                            log_forward_wandb(wandb,info,data_clone,step_num,plot=True,prefix="val_{:.3f}:".format(beta.item())+str(j)+"_",is_timing=0,single=False,args=args)
                            del data
                            del data_clone
                    elif args.dataset.startswith("a"):
                        if data.time_step==0:
                            count += 1
                            if count <2:
                                if data.__class__.__name__ == "Attr_Dict":
                                    data = data.to(args.device)
                                else:
                                    data.to(args.device)
                                info, data_clone = self.get_loss(
                                    data,
                                    args,
                                    wandb=wandb,
                                    step_num=step_num,
                                    mode="test",
                                    beta=beta,
                                    is_gc_collect=False,
                                    **kwargs
                                )
                                evolution_loss += info['evolution/loss'].mean().item() #[h,b,1] where [i,:,:] = mse sum over 25 average over all nodes
                                evolution_loss_mse += info['evolution/loss_mse'].mean().item() 
                                evolution_loss_alt += info['evolution/loss_alt'].mean().item()
                                evolution_loss_alt_mse += info['evolution/loss_alt_mse'].mean().item()
                                r_statediff += info['r/statediff'].mean().item()
                                v_statediff += info['v/statediff'].mean().item()
                                rewards += info['rewards'].mean().item()
                                state_size += info['v/state_size'].mean().item()
                                if count in [1]:
                                    print("plot testing")
                                    log_forward_wandb(wandb,info,data_clone,step_num,plot=True,prefix="val/{:.3f}:".format(beta.item())+str(j)+"_",is_timing=0,single=False,args=args,traj_index=count)
                                    del data
                                    del data_clone
                            else:
                                break
                prefix = "val/{:.3f}:".format(beta.item())
                if args.dataset.startswith("a"):
                    if count >0:
                        wandb.log({prefix+'/evolution_loss': evolution_loss/count}, step=step_num)
                        wandb.log({prefix+'/evolution_loss_alt': evolution_loss_alt/count}, step=step_num)
                        wandb.log({prefix+'/evolution_loss_mse': evolution_loss_mse/count}, step=step_num)
                        wandb.log({prefix+'/evolution_loss_alt_mse': evolution_loss_alt_mse/count}, step=step_num)
                        wandb.log({prefix+'r_statediff': r_statediff/count}, step=step_num)
                        wandb.log({prefix+'v_statediff': v_statediff/count}, step=step_num)
                        wandb.log({prefix+'rewards': rewards/count}, step=step_num)
                        wandb.log({prefix+'v_state_size': state_size/count}, step=step_num)
                        evolution_losses.append(evolution_loss_mse/count)
                        state_sizes.append(state_size/count)

                        record_data(data_record_val, [
                                evolution_loss/count,
                                evolution_loss_alt/count,
                                evolution_loss_mse/count,
                                evolution_loss_alt_mse/count,
                                r_statediff/count,
                                v_statediff/count,
                                rewards/count,
                                state_size/count],
                            [
                                prefix+'evolution_loss',
                                prefix+'evolution_loss_alt',
                                prefix+'evolution_loss_mse',
                                prefix+'evolution_loss_alt_mse',
                                prefix+'r_statediff',
                                prefix+'v_statediff',
                                prefix+'rewards',
                                prefix+'v/state_size',
                            ])
                elif args.dataset.startswith("m"):
                    if count >0:
                        wandb.log({prefix+'/evolution_loss': evolution_loss/count*200/25}, step=step_num)
                        wandb.log({prefix+'/evolution_loss_alt': evolution_loss_alt/count*200/25}, step=step_num)
                        wandb.log({prefix+'r_statediff': r_statediff/count}, step=step_num)
                        wandb.log({prefix+'v_statediff': v_statediff/count}, step=step_num)
                        wandb.log({prefix+'rewards': rewards/count}, step=step_num)
                        wandb.log({prefix+'v_state_size': state_size/count}, step=step_num)
                        evolution_losses.append(evolution_loss_mse/count*200/25)
                        state_sizes.append(state_size/count)

                        record_data(data_record_val, [
                                evolution_loss/count*200/25,
                                evolution_loss_alt/count*200/25,
                                r_statediff/count,
                                v_statediff/count,
                                rewards/count,
                                state_size/count],
                            [
                                prefix+'evolution_loss',
                                prefix+'evolution_loss_alt',
                                prefix+'r_statediff',
                                prefix+'v_statediff',
                                prefix+'rewards',
                                prefix+'v/state_size',
                            ])
                    
            if wandb!=None and len(evolution_losses)>0:
                figure = plt.figure()
                plt.scatter(state_sizes, evolution_losses)
                wandb.log({"val_scatter_mse": figure},step=step_num)

            self.actor.train()
            self.critic.train()

            if len(evolution_losses) == 0:
                evolution_losses_mean = None
            else:
                evolution_losses_mean = np.mean(evolution_losses)
            for key, item in data_record_val.items():
                data_record_val[key] = np.mean(item)
            return evolution_losses_mean, data_record_val
        
        return
        
    def get_loss(
        self,
        data,
        args,
        wandb=None,
        step_num=0,
        opt_evl=False,
        opt_actor=True,
        mode="train",
        beta=None,
        data_fine=None,
        data008=None,
        **kwargs
    ):
        """Get total loss from Value_Model and Policy."""
        if not hasattr(data, "dataset"):
            data.dataset = args.dataset
        p.print("1", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        data_clone = deepcopy(data)
        ##########################################
        # Full forward:
        ##########################################
        p.print("2", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        if not len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
            if args.rl_data_dropout.startswith("node"):
                nx = dict(to_tuple_shape(data.original_shape))["n0"][0]
                sample_idx = np.arange(1, nx-1)
                data = get_data_dropout(data, dropout_mode=args.rl_data_dropout, sample_idx=sample_idx)
            elif args.rl_data_dropout.startswith("uniform"):
                data = get_data_dropout(data, dropout_mode=args.rl_data_dropout)
            elif args.rl_data_dropout == "None":
                pass
            else:
                raise
        p.print("2.1", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        # sample batch of reward if exist:
        # pdb.set_trace()
        if args.test_reward_random_sample:
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                sampled_reward = sample_reward_beta(args.reward_beta,int(data.batch["n0"].max().detach().cpu().numpy())+1)
            else:
                sampled_reward = sample_reward_beta(args.reward_beta,int(data.batch.max().detach().cpu().numpy())+1)
            if not torch.is_tensor(sampled_reward):
                if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                    sampled_reward = torch.tensor(sampled_reward, device=data.batch["n0"].device,dtype=data.node_feature["n0"].dtype)
                else:                
                    sampled_reward = torch.tensor(sampled_reward, device=data.batch.device)
            p.print("2.2", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
            # pdb.set_trace()
            evolution_model_alt = kwargs["evolution_model_alt"]
            if evolution_model_alt==None:
                evolution_model_alt = kwargs["evolution_model"]
            _, info = full_forward(
                evolution_model=kwargs["evolution_model"],
                evolution_model_alt = evolution_model_alt,
                policy=self.actor,
                data=data,
                pred_steps=np.arange(1, self.horizon+1),
                args=args,
                data_finegrain=data_clone,
                is_alt_remeshing=args.rl_is_alt_remeshing,
                step_num=step_num,
                sampled_reward=sampled_reward,
                opt_evl=opt_evl,
                data_fine=data_fine,
            )
            # p.print("2.3", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
            log_forward_wandb(wandb, info, data_clone, step_num, True, "train_", is_timing=args.is_timing,single=args.is_single_action,args=args)
            return None
        
        elif mode=="train":     
            # sample batch of reward if exist:
            # pdb.set_trace()
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                sampled_reward = sample_reward_beta(args.reward_beta,int(data.batch["n0"].max().detach().cpu().numpy())+1)
            else:
                sampled_reward = sample_reward_beta(args.reward_beta,int(data.batch.max().detach().cpu().numpy())+1)
            if not torch.is_tensor(sampled_reward):
                if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                    sampled_reward = torch.tensor(sampled_reward, device=data.batch["n0"].device,dtype=data.node_feature["n0"].dtype)
                else:                
                    sampled_reward = torch.tensor(sampled_reward, device=data.batch.device)
            p.print("2.2", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
            # pdb.set_trace()
            evolution_model_alt = kwargs["evolution_model_alt"]
            if evolution_model_alt==None:
                evolution_model_alt = kwargs["evolution_model"]
            _, info = full_forward(
                evolution_model=kwargs["evolution_model"],
                evolution_model_alt = evolution_model_alt,
                policy=self.actor,
                data=data,
                pred_steps=np.arange(1, self.horizon+1),
                args=args,
                data_finegrain=data_clone,
                is_alt_remeshing=args.rl_is_alt_remeshing,
                step_num=step_num,
                sampled_reward=sampled_reward,
                opt_evl=opt_evl,
                data_fine=data_fine,
            )
            p.print("2.3", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)

        elif mode.startswith("test_gt_remesh_heursitc"):
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                sampled_reward = torch.zeros(int(data.batch["n0"].max().detach().cpu().numpy())+1, device=args.device)
            else:
                sampled_reward = torch.zeros(int(data.batch.max().detach().cpu().numpy())+1, device=args.device)
            sampled_reward[:] = beta
            p.print("2.2b", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)

            evolution_model_alt = kwargs["evolution_model_alt"]
            if evolution_model_alt==None:
                evolution_model_alt = kwargs["evolution_model"]
            _, info = full_forward(
                evolution_model=kwargs["evolution_model"],
                evolution_model_alt=evolution_model_alt,
                policy=self.actor,
                data=data,
                pred_steps=np.arange(1, args.pred_steps+1),
                args=args,
                data_finegrain=data_clone,
                is_alt_remeshing=args.rl_is_alt_remeshing,
                step_num=step_num,
                sampled_reward=sampled_reward,
                data_fine=data_fine,
                mode = mode,
                data008 = data008,
                heuristic = True,
                is_gc_collect=kwargs["is_gc_collect"] if "is_gc_collect" in kwargs else False,
            )
            p.print("2.3b", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
            return info, data_clone
        
        elif mode.startswith("test"):
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                sampled_reward = torch.zeros(int(data.batch["n0"].max().detach().cpu().numpy())+1, device=args.device)
            else:
                sampled_reward = torch.zeros(int(data.batch.max().detach().cpu().numpy())+1, device=args.device)
            sampled_reward[:] = beta
            p.print("2.2b", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)

            evolution_model_alt = kwargs["evolution_model_alt"]
            if evolution_model_alt==None:
                evolution_model_alt = kwargs["evolution_model"]
            _, info = full_forward(
                evolution_model=kwargs["evolution_model"],
                evolution_model_alt=evolution_model_alt,
                policy=self.actor,
                data=data,
                pred_steps=np.arange(1, args.pred_steps+1),
                args=args,
                data_finegrain=data_clone,
                is_alt_remeshing=args.rl_is_alt_remeshing,
                step_num=step_num,
                sampled_reward=sampled_reward,
                data_fine=data_fine,
                mode = mode,
                data008 = data008,
                is_gc_collect=kwargs["is_gc_collect"] if "is_gc_collect" in kwargs else False,
            )
            p.print("2.3b", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
            return info, data_clone

        else:
            raise 
        
        if step_num%args.wandb_step==0:
            if step_num%args.wandb_step_plot==0:
                plot=True
            else:
                plot=False
            p.print("3", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
            log_forward_wandb(wandb, info, data_clone, step_num, plot, "train_", is_timing=args.is_timing,single=args.is_single_action,args=args)
            p.print("3.1", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        if opt_evl and (not opt_actor):
            if args.rl_finetune_evalution_mode=="policy:alt":
                loss = info["evolution/loss"].mean() + info["evolution/loss_alt"].mean()
            elif args.rl_finetune_evalution_mode=="policy:fine":
                loss = info["evolution/loss"].mean() + info["evolution/loss_fine"].mean()
            elif args.rl_finetune_evalution_mode=="policy:alt:fine":
                loss = info["evolution/loss"].mean() + info["evolution/loss_fine"].mean() + info["evolution/loss_alt"].mean()
            elif args.rl_finetune_evalution_mode=="policy":
                loss = info["evolution/loss"].mean()
            else:
                raise
            p.print("3.2", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
            return loss
    
        ##########################################
        # Value loss:
        ##########################################
        # pdb.set_trace()
        value_loss, value_preds, value_targets = self.get_value_loss(
            data=data_clone,
            args=args,
            state_preds=info["state_preds"],   # x_k, k=0,1,...,H-1
            rewards=info["reward_preds"] if args.reward_src == "pred" else info["rewards"] if args.reward_src == "env" else None,
            value_target_mode=args.value_target_mode,
            sampled_reward=sampled_reward
        )
        p.print("4", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        if step_num%args.wandb_step==0:
            log_value_wandb(wandb, [value_loss, value_preds, value_targets], data_clone, step_num, "train_")
            p.print("4.1", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
    
        ##########################################
        # Actor loss:
        ##########################################
        actor_loss, action_info = self.get_actor_loss(
            data=data_clone,
            args=args,
            action_logprobs=info["action_logprobs"],   # logprob_k, k=0,1,...,H-1
            action_entropies=info["action_entropies"],  # entropy_k, k=0,1,...,H-1
            value_preds=value_preds,
            value_targets=value_targets,
        )
        p.print("5", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        if step_num%args.wandb_step==0:
            log_actor_wandb(wandb, [actor_loss, action_info], data_clone, step_num, "train_")
            p.print("5.1", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)

        ##########################################
        # Total loss and record:
        ##########################################
        if args.test_value_model:
            loss = (value_loss * args.value_loss_coef).mean()
        else:
            if args.is_alternating_train:
                if self.train_value_count>=0 and (self.train_value_count < args.value_steps):
                    loss = (value_loss * args.value_loss_coef).mean()
                    self.train_value_count += 1
                else:
                    self.train_value_count = -1
                    self.train_actor_count = 0
                if self.train_actor_count>=0 and (self.train_actor_count < args.actor_steps):
                    loss = (actor_loss).mean()
                    self.train_actor_count += 1
                else:
                    self.train_actor_count = -1
                    self.train_value_count = 0
            else:
                loss = (actor_loss + value_loss * args.value_loss_coef).mean()
        p.print("6", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        if args.rl_is_finetune_evolution and opt_evl:
            loss = loss + info["evolution/loss"].mean()

        self.info = {
            "loss_value": value_loss.mean().item() * args.value_loss_coef,
            "loss_actor": actor_loss.mean().item(),
            "loss_reinforce": action_info["loss_reinforce"],
            "loss_entropy": action_info["loss_entropy"],
            "action_logprobs": to_np_array(info["action_logprobs"]).mean(),
            "action_entropies": to_np_array(info["action_entropies"]).mean(),
        }
        if "evolution_loss" in info:
            self.info["evolution/loss"] = info["evolution/loss"].mean().item()

        p.print("6.1", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        for key, value in info.items():
            if not isinstance(value[0], dict):
                if isinstance(value, torch.Tensor):
                    if value[0]!=None:   
                        self.info[key] = value.mean().item()
                else:
                    if value[0]!=None: 
                        self.info[key] = torch.mean(torch.FloatTensor(value)).item()
        p.print("6.2", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        del data_clone
        return loss

    def get_actor_loss(self, data, args, action_logprobs, action_entropies, value_preds, value_targets):
        """Get the loss for the actor model following Dreamer v2.

        Args:
            action_logprobs:  [H,B,1], logprob_k, k=0,1,...,H-1
            action_entropies: [H,B,1], entropy_k, k=0,1,...,H-1
            value_preds:      [H,B,1], torch.stack([v(z0), v(z1), ...v(z_{H-1})])
            value_targets:    [H,B,1], torch.stack([V0, V1, ... V_{H-1}])

        Returns:
            actor_loss: [B, 1], total actor loss, adding from t=0,1,...H-2
        """
        assert value_targets.shape == value_preds.shape == action_logprobs.shape == action_entropies.shape
        assert len(value_targets.shape) == 3 and value_targets.shape[-1] == 1
        value_advantage = (value_targets - value_preds).detach()
        reinforce_loss =  -(action_logprobs[:-1] * value_advantage[:-1]).sum(0)
        entropy_loss = -action_entropies[:-1].sum(0) * args.rl_eta
        actor_loss = reinforce_loss + entropy_loss
        info = {}
        info["loss_reinforce"] = reinforce_loss.mean().item()
        info["loss_entropy"] = entropy_loss.mean().item()
        return actor_loss, info

    def get_value_loss(self, data, args, state_preds, rewards,sampled_reward=None, value_target_mode="value-lambda"):
        """Get the loss for the value model following Dreamer v2.

        Args:
            state_preds:  x_k, k=0,1,...,H-1, is a list of dictionaries, each with
            {
                "x": [n_nodes, steps, feature],
                "x_pos": [n_nodes, 1],
                "x_bdd": [n_nodes, 1],
                "edge_index": [2, n_edges],
                "batch": [n_nodes],
            }
            rewards: shape of [H,B,1], reward for step k, k=0,1,...,H-1.

        Returns:
            value_loss: [B, 1], total value loss
            value_preds:   [H,B,1], torch.stack([v(z0), v(z1), ...v(z_{H-1})])
            value_targets: [H,B,1], torch.stack([V0, V1, ... V_{H-1}])
        """
        value_preds = []
        for k in range(self.horizon):
            
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                # pdb.set_trace()
                # data_pred = state_preds[k]
                if args.policy_input_feature == "velocity":
                    x_list = [state_pred["x"] for state_pred in state_preds]  # each has shape of [n_nodes, steps:1, feature]
                elif args.policy_input_feature == "coords":
                    x_list = [state_pred["history"][-1] for state_pred in state_preds]  # each has shape of [n_nodes, steps:1, feature]
                data_pred = get_2d_data_pred(
                    state_preds=x_list,
                    step=k,
                    data=data,
                    x_pos=state_preds[k]["x_pos"],
                    x_bdd=state_preds[k]["x_bdd"],
                    edge_index=state_preds[k]["edge_index"],
                    edge_attr=(state_preds[k]["edge_attr"],),
                    onehot_list=state_preds[k]["onehot_list"],
                    history=state_preds[k]["history"],
                    kinematics_list=state_preds[k]["kinematics_list"],
                    batch=state_preds[k]["batch"],
                )
                data_pred.interp_index = state_preds[k]["interp_index"]
            else:
                x_list = [state_pred["x"] for state_pred in state_preds]  # each has shape of [n_nodes, steps:1, feature]
                data_pred = get_data_pred(
                    state_preds=x_list,
                    step=k,
                    data=data,
                    x_pos=state_preds[k]["x_pos"],
                    x_bdd=state_preds[k]["x_bdd"],
                    edge_index=state_preds[k]["edge_index"],
                    batch=state_preds[k]["batch"],
                )
            # pdb.set_trace()
            value_pred = self.critic(data_pred,reward_beta=sampled_reward)  # [B, 1]
            value_preds.append(value_pred)
        value_preds = torch.stack(value_preds)  # torch.stack([v(z0), v(z1), ...v(z_{H-1})]), shape: [H,B,1]
        # pdb.set_trace()
        value_targets = self.get_value_targets(
            data=data,
            args=args,
            state_preds=state_preds,
            rewards=rewards,
            value_target_mode=value_target_mode,
            reward_beta=sampled_reward
        )  # torch.cat([V0, V1, ... V_{H-1}]), shape: [H,B,1]
        assert value_preds.shape == value_targets.shape
        if args.value_loss_type == "mse":
            value_loss = (value_preds - value_targets)[:-1].square().sum(0) / 2  # sum over from 0 to H-2, shape [B, 1]
        elif args.value_loss_type == "l1":
            value_loss = (value_preds - value_targets)[:-1].abs().sum(0) / 2  # sum over from 0 to H-2, shape [B, 1]
        else:
            raise
        return value_loss, value_preds, value_targets

    def get_value_targets(self, data, args, state_preds, rewards, reward_beta=None, stop_grad=True, value_target_mode="value-lambda"):
        """
        Compute value targets.

        Args:
            rewards: shape of [H,B,1], reward for step k, k=0,1,...,H-1.
        """
        if value_target_mode == "value-lambda":
            if stop_grad:
                with torch.no_grad():
                    return self.get_value_targets_lambda(data=data, args=args, state_preds=state_preds, rewards=rewards,reward_beta=reward_beta, step=0)
            else:
                return self.get_value_targets_lambda(data=data, args=args, state_preds=state_preds, rewards=rewards, reward_beta=reward_beta, step=0)
        elif value_target_mode == "vanilla":
            assert len(rewards.shape) == 3 and rewards.shape[2] == 1
            horizon = rewards.shape[0]
            if stop_grad:
                with torch.no_grad():
                    cum_rewards = rewards.flip(0).cumsum(0).flip(0)  # cumulative rewards from current time step to H
                    value_target = cum_rewards / torch.arange(1, horizon+1, device=cum_rewards.device).flip(0).unsqueeze(-1).unsqueeze(-1)  # denominator: shape [H,1,1]
            else:
                cum_rewards = rewards.flip(0).cumsum(0).flip(0)  # cumulative rewards from current time step to H
                value_target = cum_rewards / torch.arange(1, horizon+1, device=cum_rewards.device).flip(0).unsqueeze(-1).unsqueeze(-1)  # denominator: shape [H,1,1]
            return value_target
        else:
            raise

    def get_value_targets_lambda(self, data, args, state_preds, rewards, reward_beta=None, step=0):
        """Recursively define the target for critic.

        Args:
            state_preds:  x_k, k=0,1,...,H-1
            rewards: r_k, k=0,1,...,H-1

        Returns:
            value_targets: torch.cat([Vt, Vt+1, ... V_{H-1}])  # [H,B,1]
        """
        value_targets = []
        if (not args.use_reward_vanilla) and step == self.horizon - 1:
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                if args.policy_input_feature == "velocity":
                    x_list = [state_pred["x"] for state_pred in state_preds]  # each has shape of [n_nodes, steps:1, feature]
                elif args.policy_input_feature == "coords":
                    x_list = [state_pred["history"][-1] for state_pred in state_preds]  # each has shape of [n_nodes, steps:1, feature]
                data_pred = get_2d_data_pred(
                    state_preds=x_list,
                    step=step,
                    data=data,
                    x_pos=state_preds[step]["x_pos"],
                    x_bdd=state_preds[step]["x_bdd"],
                    edge_index=state_preds[step]["edge_index"],
                    edge_attr=(state_preds[step]["edge_attr"],),
                    onehot_list=state_preds[step]["onehot_list"],
                    history=state_preds[step]["history"],
                    kinematics_list=state_preds[step]["kinematics_list"],
                    batch=state_preds[step]["batch"],
                )
                # data_pred = state_pred[step]
                data_pred.interp_index = state_preds[step]["interp_index"]
            else:
                data_pred = get_data_pred(
                    state_preds=[state_pred["x"] for state_pred in state_preds],  # each has shape of [n_nodes, steps, feature]
                    step=step,
                    data=data,
                    x_pos=state_preds[step]["x_pos"],
                    x_bdd=state_preds[step]["x_bdd"],
                    edge_index=state_preds[step]["edge_index"],
                    batch=state_preds[step]["batch"],
                )
            value_target = rewards[step] + self.critic_target(data_pred,reward_beta=reward_beta) * args.rl_gamma
            value_targets = value_target[None]  # [1, B, 1]
        elif args.use_reward_vanilla and step == self.horizon - 1:
            value_target = rewards[step] 
            value_targets = value_target[None]  # [1, B, 1]
        else:
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                # data_pred_next = state_preds[step+1]
                if args.policy_input_feature == "velocity":
                    x_list = [state_pred["x"] for state_pred in state_preds]  # each has shape of [n_nodes, steps:1, feature]
                elif args.policy_input_feature == "coords":
                    x_list = [state_pred["history"][-1] for state_pred in state_preds]  # each has shape of [n_nodes, steps:1, feature]
                data_pred_next = get_2d_data_pred(
                    state_preds=x_list,
                    step=step+1,
                    data=data,
                    x_pos=state_preds[step+1]["x_pos"],
                    x_bdd=state_preds[step+1]["x_bdd"],
                    edge_index=state_preds[step+1]["edge_index"],
                    edge_attr=(state_preds[step+1]["edge_attr"],),
                    onehot_list=state_preds[step+1]["onehot_list"],
                    history=state_preds[step+1]["history"],
                    kinematics_list=state_preds[step+1]["kinematics_list"],
                    batch=state_preds[step+1]["batch"],
                )
                data_pred_next.interp_index = state_preds[step+1]["interp_index"]
            else:
                data_pred_next = get_data_pred(
                    state_preds=[state_pred["x"] for state_pred in state_preds],  # each has shape of [n_nodes, steps, feature]
                    step=step+1,
                    data=data,
                    x_pos=state_preds[step+1]["x_pos"],
                    x_bdd=state_preds[step+1]["x_bdd"],
                    edge_index=state_preds[step+1]["edge_index"],
                    batch=state_preds[step+1]["batch"],
                )
            value_targets_next = self.get_value_targets_lambda(
                 data=data,
                 args=args,
                 state_preds=state_preds,
                 rewards=rewards,
                 step=step+1,
                 reward_beta=reward_beta
            )
            value_target = rewards[step] + (
                self.critic_target(data_pred_next,reward_beta=reward_beta) * (1-args.rl_lambda) +
                 value_targets_next[0] * args.rl_lambda
            ) * args.rl_gamma
            value_targets = torch.cat([value_target[None], value_targets_next], 0)
        return value_targets

    def monitor_copy_critic(self, critic_update_iterations, verbose=1):
        """Copy self.critic to self.critic_target every {critic_update_iterations} steps."""
        self.critic_iteration_count += 1
        if self.critic_iteration_count % critic_update_iterations == 0:
            self.copy_critic()
            if verbose >= 2:
                print(f"Copy self.critic to self.critic_target at iteration {self.critic_iteration_count}.")

    def soft_update(self, tau=0.05):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def copy_critic(self):
        """Copy self.critic to self.critic_target."""
        self.critic_target = load_model(self.critic.model_dict, device=self.device)

    @property
    def model_dict(self):
        model_dict = {"type": "Actor_Critic"}
        model_dict["horizon"] = self.horizon
        model_dict["actor_model_dict"] = self.actor.model_dict
        model_dict["critic_model_dict"] = self.critic.model_dict
        model_dict["critic_target_model_dict"] = self.critic_target.model_dict
        return model_dict

def batch_mse(mse, index,mean=True):
    """
    Args:
        mse: [n_nodes, 25, 1]
        index: [n_nodes]
    
    """
    x = mse
    idx = index.to(torch.int64)
    idx_unique_count = torch.unique(idx, return_counts=True)[1].type(torch.float32)
    res = torch.zeros(idx.max()+1, device=x.device).scatter_add(0, idx, x.float())
    if mean:
        res /= idx_unique_count.float()
    return res

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
            # if ((math.trunc(query[0].item())%2)!=0):
            #     if (query[0].item() - math.trunc(query[0].item()) < 0.5):
            #         query[0] = math.trunc(query[0].item())
            #     elif (query[0].item() - math.trunc(query[0].item()) > 0.5):
            #         query[0] = math.trunc(query[0].item())+1
            # if ((math.trunc(query[1].item())%2)!=0):
            #     if (query[1].item() - math.trunc(query[1].item()) < 0.5):
            #         query[1] = math.trunc(query[1].item())
            #     elif (query[1].item() - math.trunc(query[1].item()) > 0.5):
            #         query[1] = math.trunc(query[1].item())+1
            # if (query[0].item() - math.trunc(query[0].item())) < 0.5:
            #     query[0] += 1e-15
            # elif (query[0].item() - math.trunc(query[0].item())) >= 0.5:
            #     query[0] -= 1e-15
            # if (query[1].item() - math.trunc(query[1].item())) < 0.5:
            #     query[1] += 1e-15
            # elif (query[1].item() - math.trunc(query[1].item())) >= 0.5:
            #     query[1] -= 1e-15     
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

def get_loss_2d_refined(data_pred, data_gt, current_step,opt_evl=False):
    if not opt_evl:
        # data_pred = data_pred.clone()
        ybatch = ((data_gt.reind_yfeatures["n0"][current_step][:,0] - data_gt.yfeatures["n0"][current_step][:,0])/2).T
        vers = (data_pred.history[-1] + 2*data_pred.batch.repeat((data_pred.history[-1].shape[1], 1)).T)
        faces = data_pred.xfaces.T
        tarvers = data_gt.reind_yfeatures["n0"][current_step][:,:2].clone().detach().cpu().numpy()
        # p.print("9991", precision="millisecond", is_silent=False, avg_window=1)
        interpolate_out = generate_barycentric_interpolated_data(vers.clone().detach(), faces, vers, tarvers)
        # p.print("9992", precision="millisecond", is_silent=False, avg_window=1)
        gt = (interpolate_out - 2*ybatch.repeat((interpolate_out.shape[-1], 1)).T)[:,3:]
        mse = (data_gt.yfeatures["n0"][current_step][:,3:] - gt).square()
        return batch_mse(torch.sqrt(mse.sum(dim=1)), ybatch)[:,None],batch_mse((mse.sum(dim=1)), ybatch, mean=False)[:,None], None
    else:
        vers = data_gt.reind_yfeatures["n0"][current_step]
        faces = torch.tensor(data_gt.yface_list["n0"][current_step])    
        tarvers = data_pred.history[-1][:,:2] + 2*data_pred.batch.repeat((data_pred.history[-1].shape[1], 1)).T[:,:2]
        if isinstance(tarvers, torch.Tensor):
            tarvers = tarvers.detach().cpu().numpy()
        # Interpolation and compute kinematics
        interp_gtar = generate_barycentric_interpolated_data(vers, faces, vers, tarvers)
        interp_gtar_coords = (interp_gtar - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]
        # mse = (data_pred.history[-1][:,3:] - gt).square()
        
        # temporary lines
        # vers = data_gt['history']['n0'][-1]+ 2*data_gt.batch['n0'].repeat((data_gt['history']['n0'][-1].shape[1], 1)).T
        # faces = data_gt.xfaces["n0"].T.detach().cpu()
        # tarvers = data_pred.history[-1][:,:2] + 2*data_pred.batch.repeat((data_pred.history[-1].shape[1], 1)).T[:,:2]
        # tarvers = tarvers.detach().cpu().numpy()
        # interpolate_out = generate_barycentric_interpolated_data(vers, faces, vers, tarvers)
        # interp_gt_coords = (interpolate_out - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]

        # tarvers = data_gt.reind_yfeatures["n0"][current_step]
        # tarfaces = torch.tensor(data_gt.yface_list["n0"][current_step]) 
        # interpolate_tar = generate_barycentric_interpolated_data(tarvers, tarfaces, tarvers, interpolate_out[:,:2].detach().cpu().numpy())
        # interp_gtar_coords = (interpolate_tar - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]

        # if current_step == 0:
        #     #pdb.set_trace()
        #     vers = data_gt['history']['n0'][-1]+ 2*data_gt.batch['n0'].repeat((data_gt['history']['n0'][-1].shape[1], 1)).T
        #     faces = data_gt.xfaces["n0"].T.detach().cpu()
        #     tarvers = data_pred.history[-1][:,:2] + 2*data_pred.batch.repeat((data_pred.history[-1].shape[1], 1)).T[:,:2]
        #     tarvers = tarvers.detach().cpu().numpy()
        #     interpolate_out = generate_barycentric_interpolated_data(vers, faces, vers, tarvers)
        #     interp_gt_coords = (interpolate_out - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]

        #     tarvers = data_gt.reind_yfeatures["n0"][current_step]
        #     tarfaces = torch.tensor(data_gt.yface_list["n0"][current_step]) 
        #     interpolate_tar = generate_barycentric_interpolated_data(tarvers, tarfaces, tarvers, interpolate_out[:,:2].detach().cpu().numpy())
        #     interp_gtar_coords = (interpolate_tar - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]

        #     pastvers = data_gt['history']['n0'][0]+ 2*data_gt.batch_history['n0'].repeat((data_gt['history']['n0'][0].shape[1], 1)).T
        #     pastfaces = torch.tensor(data_gt.xface_list["n0"][0]) 
        #     interpolate_past = generate_barycentric_interpolated_data(pastvers, pastfaces, pastvers, interpolate_out[:,:2].detach().cpu().numpy())
        #     interp_gtpast_coords = (interpolate_past - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]
        # elif current_step == 1:
        #     #pdb.set_trace()
        #     vers = data_gt.reind_yfeatures["n0"][current_step-1]
        #     faces = torch.tensor(data_gt.yface_list["n0"][current_step-1])    
        #     tarvers = data_pred.history[-1][:,:2] + 2*data_pred.batch.repeat((data_pred.history[-1].shape[1], 1)).T[:,:2]
        #     if isinstance(tarvers, torch.Tensor):
        #         tarvers = tarvers.detach().cpu().numpy()
        #     # Interpolation and compute kinematics
        #     interpolate_out = generate_barycentric_interpolated_data(vers, faces, vers, tarvers)
        #     interp_gt_coords  = (interpolate_out - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]
        #     # interp_gt_coords = (interpolate_out - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]

        #     tarvers = data_gt.reind_yfeatures["n0"][current_step]
        #     tarfaces = torch.tensor(data_gt.yface_list["n0"][current_step]) 
        #     interpolate_tar = generate_barycentric_interpolated_data(tarvers, tarfaces, tarvers, interpolate_out[:,:2].detach().cpu().numpy())
        #     interp_gtar_coords = (interpolate_tar - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]

        #     pastvers = data_gt['history']['n0'][-1]+ 2*data_gt.batch['n0'].repeat((data_gt['history']['n0'][-1].shape[1], 1)).T
        #     pastfaces = data_gt.xfaces["n0"].T.detach().cpu() 
        #     interpolate_past = generate_barycentric_interpolated_data(pastvers, pastfaces, pastvers, interpolate_out[:,:2].detach().cpu().numpy())
        #     interp_gtpast_coords = (interpolate_past - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]
        # else:
        #     #pdb.set_trace()
        #     vers = data_gt.reind_yfeatures["n0"][current_step-1]
        #     faces = torch.tensor(data_gt.yface_list["n0"][current_step-1])    
        #     tarvers = data_pred.history[-1][:,:2] + 2*data_pred.batch.repeat((data_pred.history[-1].shape[1], 1)).T[:,:2]
        #     if isinstance(tarvers, torch.Tensor):
        #         tarvers = tarvers.detach().cpu().numpy()
        #     # Interpolation and compute kinematics
        #     interpolate_out = generate_barycentric_interpolated_data(vers, faces, vers, tarvers)
        #     interp_gt_coords  = (interpolate_out - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]
        #     # interp_gt_coords = (interpolate_out - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]

        #     tarvers = data_gt.reind_yfeatures["n0"][current_step]
        #     tarfaces = torch.tensor(data_gt.yface_list["n0"][current_step]) 
        #     interpolate_tar = generate_barycentric_interpolated_data(tarvers, tarfaces, tarvers, interpolate_out[:,:2].detach().cpu().numpy())
        #     interp_gtar_coords = (interpolate_tar - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]

        #     pastvers = data_gt.reind_yfeatures["n0"][current_step-2]
        #     pastfaces = torch.tensor(data_gt.yface_list["n0"][current_step-2]) 
        #     interpolate_past = generate_barycentric_interpolated_data(pastvers, pastfaces, pastvers, interpolate_out[:,:2].detach().cpu().numpy())
        #     interp_gtpast_coords = (interpolate_past - 2*data_pred.batch.repeat((data_pred.history[-1].shape[-1], 1)).T)[:,3:]
        # ((data_pred.x - gt)-(data_pred.history[-1][:,3:] - interp_gtar_coords)).max()
        # gt = interp_gtar_coords -2*interp_gt_coords + interp_gtpast_coords
        # mse = (data_pred.x - gt).square()
        mse = (data_pred.history[-1][:,3:] - interp_gtar_coords).square()
        # pdb.set_trace()
        return batch_mse(torch.sqrt(mse.sum(dim=1)), data_pred.batch)[:,None],batch_mse((mse.sum(dim=1)), data_pred.batch, mean=True)[:,None], interp_gtar_coords


def get_loss_refined(data_pred, data_gt, current_step,):
    #here the implementation assume 1d case
    x_pos = data_pred.node_pos['n0']
    batch = data_pred.batch
    pred = data_pred.node_feature['n0']
    x_pos_gt = data_gt.node_pos['n0']
    gt = data_gt.node_label['n0'][:,current_step*pred.shape[1]:(current_step+1)*pred.shape[1],:]

    batch_gt = data_gt.batch
    batch = data_pred.batch

    x_pos_incremented = x_pos + batch[:,None]*500
    x_pos_incremented_idx = torch.argsort(x_pos_incremented[:,0])
    x_pos_incremented = x_pos_incremented[x_pos_incremented_idx]
    x_pos_gt_incremented = x_pos_gt + batch_gt[:,None]*500

    x_pos_gt_incremented = x_pos_gt_incremented.permute(1,0).repeat([pred.shape[1],1])
    x_pos_incremented = x_pos_incremented.permute(1,0).repeat([pred.shape[1],1])
    fnc = Interp1d_torch()
    pred_interp = fnc(x_pos_incremented,pred[x_pos_incremented_idx,:,0].permute(1,0),x_pos_gt_incremented)
    pred_interp = pred_interp.permute(1,0).unsqueeze(-1)
    assert pred_interp.shape==gt.shape

    mse = (pred_interp - gt).square() #[num,25,1]
    return batch_mse(mse.sum(dim=[-2,-1]), batch_gt)[:,None]

def get_interp(data,args):
    data = deepcopy(data)
    data_clone = data.clone()   
    if args.rl_data_dropout.startswith("node"):
        nx = dict(to_tuple_shape(data.original_shape))["n0"][0]
        sample_idx = np.arange(1, nx-1)
        data = get_data_dropout(data, dropout_mode=args.rl_data_dropout, sample_idx=sample_idx)
    elif args.rl_data_dropout.startswith("uniform"):
        data = get_data_dropout(data, dropout_mode=args.rl_data_dropout)
    elif args.rl_data_dropout == "None":
        pass
    else:
        raise

    x_pos_fine = data_clone.node_pos['n0']
    x_pos = data.node_pos['n0']
    batch = data.batch
    batch_fine = data_clone.batch
    x = data.node_feature['n0']
    
    x_pos_incremented = x_pos + batch[:,None]*10
    x_pos_fine_incremented = x_pos_fine + batch_fine[:,None]*10

    x_pos_incremented = x_pos_incremented.permute(1,0).repeat([x.shape[1],1])
    x_pos_fine_incremented = x_pos_fine_incremented.permute(1,0).repeat([x.shape[1],1])
    fnc = Interp1d_torch()
    pred_interp = fnc(x_pos_incremented,x[:,:,0].permute(1,0),x_pos_fine_incremented)
    pred_interp = pred_interp.permute(1,0).unsqueeze(-1)
    data_clone.node_feature['n0'] = pred_interp
    del data
    return data_clone


def full_forward(
    evolution_model,
    evolution_model_alt,
    policy,
    data,
    pred_steps,
    args,
    data_finegrain=None,
    is_alt_remeshing=True,
    step_num=None,
    sampled_reward=None,
    opt_evl=False,
    data_fine=None,
    mode="train",
    data008=None,
    is_gc_collect=True,
    heuristic=False,
):
    """Evolve the state forward to multiple steps using both the evolution model and the remeshing policy.

    Notations:
        f: evolution model,
        xk: state  at time step k
        mk: mesh   at time step k
        ak: action at time step k
        rk: reward at time step k
        logprob_k: logprob for the action "ak" at time step k
        entropy_k: entropy for the action "ak" at time step k

    Pipeline:
        Updated:

            z0=(x0,m0);   --a0--> z1'=(x0,m1), logprob_0, entropy_0 --f--> 
                          (z0=(x0,m0)--f --> z1''=(x1'',m0);loss0'' )

            z1=(x1,m1);loss0,r0=loss0''-loss0 --a1--> z2'=(x1,m2), logprob_1, entropy_1 --f--> 
                                 (z1=(x1,m1)--f --> z2''=(x2'',m1);loss1'')

            z2=(x2,m2);loss1,r1=loss1''-loss1 --a2--> z3'=(x2,m3), logprob_2, entropy_2 --f--> 
            ...

            z_{H-1}...;r_{H-2}=loss_{H-2}''-loss_{H-2}--a_{H-1}--> z_H'=(x_{H-1},m_H), logprob_{H-1}, entropy_{H-1} --f--> 
            
            z_H...;r_H=loss_{H-1}''-loss_{H-1}

    The argument "data" is z0'. Firstly evolution model f will be called, to obtain z0=(x0,m0), and reward r0,
    then the policy is called, obtaining the z1'=(x0,m1) and the corresponding 
        logprob and entropy. Then iterate.
    The value function v(z0)=r0 + gamma*r1 + gamma**2*r2+... will operate on z0, v(z1) on z1, ..., v(z_{H-1}) on z_{H-1}
    The value target V0, V1, ....V_{H-1} will be at the same time steps as v(z0), v(z1), ...v(z_{H-1})

    Returns:
        data: the Data object after the multiple steps forward
        info: dictionary containing:
            rewards/reward_preds: [H, B, 1]
    """
    info = {"prob": []}
    if not isinstance(pred_steps, list) and not isinstance(pred_steps, np.ndarray):
        pred_steps = [pred_steps]
    horizon = max(pred_steps)
    if opt_evl: horizon = 3
    info = {}
    if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
        tindices = np.array(data["hist_indices"]["n0"][0]).astype('int64')
        tfweights = data["hist_weights"]["n0"][0].to(dtype=torch.float32)
        hist_wfeature = torch.matmul(tfweights, data["history"]["n0"][0][tindices,:]).diagonal().T[:,3:]
        velocity = data["history"]["n0"][-1][:,3:] - hist_wfeature

        state_preds = Attr_Dict({
            # "x": data["node_feature"]["n0"],
            "x": velocity,
            "x_pos": data["x_pos"]["n0"],
            "edge_index": data["edge_index"][("n0","0","n0")],
            "xfaces": data["xfaces"]["n0"],
            "x_bdd": data["x_bdd"]["n0"], 
            "original_shape": data["original_shape"],
            "dyn_dims": data["dyn_dims"],
            "compute_func": data["compute_func"],
            "grid_keys": data["grid_keys"],
            "part_keys": data["part_keys"], 
            "time_step": data["time_step"], 
            "sim_id": data["sim_id"],
            "time_interval": data["time_interval"], 
            "cushion_input": data["cushion_input"],
            "bary_weights": data["bary_weights"]["n0"],
            "bary_indices": data["bary_indices"]["n0"],
            "hist_weights": data["hist_weights"]["n0"],
            "hist_indices": data["hist_indices"]["n0"],
            "yedge_index": data["yedge_index"]["n0"],
            "y_back": data["y_back"]["n0"],
            "y_tar": data["y_tar"]["n0"],
            "yface_list": data["yface_list"]["n0"],
            "history": data["history"]["n0"],
            "yfeatures": data["yfeatures"]["n0"],
            "node_dim": data["node_dim"]["n0"],
            "xface_list": data["xface_list"]["n0"],
            "reind_yfeatures": data["reind_yfeatures"]["n0"],
            "batch": data.batch["n0"],
            "batch_history": data["batch_history"]["n0"],
        })
        if "edge_attr" in data:
            state_preds.edge_attr = data["edge_attr"]["n0"][0]
        if "onehot_list" in data:
            state_preds.onehot_list = data["onehot_list"]["n0"]
            state_preds.kinematics_list = data["kinematics_list"]["n0"]   
        state_preds.x = torch.cat([state_preds.x.flatten(start_dim=1)], -1)
        state_preds.interp_index=0        
    else:
        state_preds = Attr_Dict({
            "x": data.node_feature["n0"],  # e.g. shape [2000, 25, 1]
            "x_pos": data.node_pos["n0"],  # e.g. shape [2000, 1]
            "x_bdd": data.x_bdd["n0"],  # e.g. shape [2000, 1]
            "edge_index": data.edge_index[("n0", "0", "n0")],
            "batch": data.batch,
        })
    if args.test_data_interp and step_num%args.wandb_step_plot==0:
        data_interp = get_interp(data_finegrain, args)
        state_preds_interp_alt = Attr_Dict({
            "x": data_interp.node_feature["n0"],  # e.g. shape [2000, 25, 1]
            "x_pos": data_interp.node_pos["n0"],  # e.g. shape [2000, 1]
            "x_bdd": data_interp.x_bdd["n0"],  # e.g. shape [2000, 1]
            "edge_index": data_interp.edge_index[("n0", "0", "n0")],
            "batch": data_interp.batch,
        })
        record_data(info, [state_preds_interp_alt], ["state_preds_interp_alt"])  # (x_k, x_pos_k, x_bdd_k), 0

    if opt_evl and args.rl_finetune_evalution_mode in ["policy:fine","policy:alt:fine"]:
        if args.dataset.startswith("m"):
            data_fine = data_finegrain.clone()
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                state_preds_fine = data_finegrain.clone()
                state_preds_fine.interp_index=0   
            else:
                state_preds_fine = Attr_Dict({
                    "x": data_finegrain.node_feature["n0"],  # e.g. shape [2000, 25, 1]
                    "x_pos": data_finegrain.node_pos["n0"],  # e.g. shape [2000, 1]
                    "x_bdd": data_finegrain.x_bdd["n0"],  # e.g. shape [2000, 1]
                    "edge_index": data_finegrain.edge_index[("n0", "0", "n0")],
                    "batch": data_finegrain.batch,
                })
            record_data(info, [state_preds_fine], ["state_preds_fine"])  # (x_k, x_pos_k, x_bdd_k), 0
        elif args.dataset.startswith("a"):
            tindices = np.array(data_fine["hist_indices"]["n0"][0]).astype('int64')
            tfweights = data_fine["hist_weights"]["n0"][0].to(dtype=torch.float32)
            hist_wfeature = torch.matmul(tfweights, data_fine["history"]["n0"][0][tindices,:]).diagonal().T[:,3:]
            velocity = data_fine["history"]["n0"][-1][:,3:] - hist_wfeature
            
            data_finegrain_fine = data_fine.clone()
            # data_finegrain_fine["node_feature"]["n0"] = velocity

            state_preds_fine = Attr_Dict({
                # "x": data_fine["node_feature"]["n0"],
                "x": velocity,
                "x_pos": data_fine["x_pos"]["n0"],
                "edge_index": data_fine["edge_index"][("n0","0","n0")],
                "xfaces": data_fine["xfaces"]["n0"],
                "x_bdd": data_fine["x_bdd"]["n0"], 
                "original_shape": data_fine["original_shape"],
                "dyn_dims": data_fine["dyn_dims"],
                "compute_func": data_fine["compute_func"],
                "grid_keys": data_fine["grid_keys"],
                "part_keys": data_fine["part_keys"], 
                "time_step": data_fine["time_step"], 
                "sim_id": data_fine["sim_id"],
                "time_interval": data_fine["time_interval"], 
                "cushion_input": data_fine["cushion_input"],
                "bary_weights": data_fine["bary_weights"]["n0"],
                "bary_indices": data_fine["bary_indices"]["n0"],
                "hist_weights": data_fine["hist_weights"]["n0"],
                "hist_indices": data_fine["hist_indices"]["n0"],
                "yedge_index": data_fine["yedge_index"]["n0"],
                "y_back": data_fine["y_back"]["n0"],
                "y_tar": data_fine["y_tar"]["n0"],
                "yface_list": data_fine["yface_list"]["n0"],
                "history": data_fine["history"]["n0"],
                "yfeatures": data_fine["yfeatures"]["n0"],
                "node_dim": data_fine["node_dim"]["n0"],
                "xface_list": data_fine["xface_list"]["n0"],
                "reind_yfeatures": data_fine["reind_yfeatures"]["n0"],
                "batch": data_fine.batch["n0"],
                "batch_history": data_fine["batch_history"]["n0"],
            })
            if "edge_attr" in data_fine:
                state_preds_fine.edge_attr = data_fine["edge_attr"]["n0"][0]
            if "onehot_list" in data_fine:
                state_preds_fine.onehot_list = data_fine["onehot_list"]["n0"]
                state_preds_fine.kinematics_list = data_fine["kinematics_list"]["n0"]   
            state_preds_fine.x = torch.cat([state_preds_fine.x.flatten(start_dim=1)], -1)
            state_preds_fine.interp_index=0   
            record_data(info, [state_preds_fine], ["state_preds_fine"])  # (x_k, x_pos_k, x_bdd_k), 0
    
    record_data(info, [state_preds], ["state_preds"])  # (x_k, x_pos_k, x_bdd_k), 0
    record_data(info, [state_preds], ["state_preds_alt"])  # (x_k, x_pos_k, x_bdd_k), 0
    data_not_remeshed_gt = data.clone()
    if mode.startswith("test_gt_remesh"):
        data_008_not_remeshed_gt = data008.clone()
        data_heuristic = data.clone().detach()
        data_heuristic_gt_evl = data.clone().detach()
    p.print("2.21", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
    for k in range(horizon):  # k=0,1,...H-1
        ##################################################
        # Take action, zk=(xk,mk);   --ak--> z_{k+1}'=(xk,m_{k+1}), logprob_k, entropy_k
        ##################################################
        p.print("2.211", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        if not (mode=="test_remeshing"):
            if is_alt_remeshing:
                data_not_remeshed = copy_data(data,detach=True)
            else:
                if "data_alt" in locals():
                    data_not_remeshed = copy_data(data_alt,detach=True)
                else:
                    assert k == 0
                    data_not_remeshed = copy_data(data,detach=True)

                if mode.startswith("test_gt_remesh"):
                    if "data_alt_gt_mesh" in locals():
                        data_not_remeshed_gt_mesh = copy_data(data_alt_gt_mesh,detach=True)
                    else:
                        assert k == 0
                        data_not_remeshed_gt_mesh = copy_data(data,detach=True)
                    
                    if "data_alt_008" in locals():
                        data_not_remeshed_008 = copy_data(data_alt_008,detach=True)
                    else:
                        assert k == 0
                        data_not_remeshed_008 = copy_data(data008,detach=True)

                    if "data_alt_008_gt_evl" in locals():
                        data_not_remeshed_008_gt_evl = copy_data(data_alt_008_gt_evl,detach=True)
                    else:
                        assert k == 0
                        data_not_remeshed_008_gt_evl = copy_data(data008,detach=True)

                    if "data_alt_gt_evl" in locals():
                        data_not_remeshed_gt_evl = copy_data(data_alt_gt_evl,detach=True)
                    else:
                        assert k == 0
                        data_not_remeshed_gt_evl = copy_data(data,detach=True)

                    if "data_alt_gt_mesh_gt_evl" in locals():
                        data_not_remeshed_gt_mesh_gt_evl = copy_data(data_alt_gt_mesh_gt_evl,detach=True)
                    else:
                        assert k == 0
                        data_not_remeshed_gt_mesh_gt_evl = copy_data(data,detach=True)

                    if heuristic:
                        if "data_heuristic" in locals():
                            data_remeshed_heuristic = copy_data(data_heuristic,detach=True)
                        else:
                            assert k == 0
                            data_remeshed_heuristic = copy_data(data,detach=True)

                        if "data_heuristic_gt_evl" in locals():
                            data_remeshed_heuristic_gt_evl = copy_data(data_heuristic_gt_evl,detach=True)
                        else:
                            assert k == 0
                            data_remeshed_heuristic_gt_evl = copy_data(data,detach=True)
                
                if args.test_data_interp and step_num%args.wandb_step_plot==0:
                    if "data_interp_alt" in locals():
                        data_interp_alt_not_remeshed = copy_data(data_interp_alt,detach=True)
                    else:
                        assert k==0
                        data_interp_alt_not_remeshed = copy_data(data_interp,detach=True)
        p.print("2.212", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        if (opt_evl and args.evl_stop_gradient and args.dataset.startswith("a")) or args.stop_all_gradient:
            data = detach_data(data)
            data_not_remeshed = detach_data(data_not_remeshed)
            if heuristic:
                data_heuristic = detach_data(data_heuristic)
                data_heuristic_gt_evl = detach_data(data_heuristic_gt_evl)
            if mode.startswith("test_gt_remesh"):
                data_not_remeshed_008 = detach_data(data_not_remeshed_008)
                data_not_remeshed_008_gt_evl = detach_data(data_not_remeshed_008_gt_evl)
                data_not_remeshed_gt_mesh = detach_data(data_not_remeshed_gt_mesh)
                data_not_remeshed_gt_mesh_gt_evl= detach_data(data_not_remeshed_gt_mesh_gt_evl)
                data_not_remeshed_gt_evl= detach_data(data_not_remeshed_gt_evl)
         
            if "data_fine" in locals() and opt_evl:
                data_fine = detach_data(data_fine)
            if args.fine_tune_gt_input and k!=0 and opt_evl:
                # pdb.set_trace()
                data.history[-1][:,3:] = data_gt.detach()
                data.x = data.history[-1][:,3:] - data.history[0][:,3:]
                data_not_remeshed.history[-1][:,3:] = data_alt_gt.detach()
                data_not_remeshed.x = data_not_remeshed.history[-1][:,3:] - data_not_remeshed.history[0][:,3:]
                if "data_fine" in locals() and opt_evl:
                    data_fine.history[-1][:,3:] = data_fine_gt.detach()
                    data_fine.x = data_fine.history[-1][:,3:] - data_fine.history[0][:,3:]

        p.print("2.213", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        p.print("2.22", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        # data_clone = data.clone()
        data_remeshed, action_logprobs_dict, action_entropies_dict, action_probs = policy.act(data,reward_beta=sampled_reward,interp_index=k, is_timing=args.is_timing,is_eval_sample=args.is_eval_sample,debug=args.debug,data_gt=data_finegrain if args.debug else None,policy_input_feature=args.policy_input_feature)
        # pdb.set_trace()
        if heuristic:
            data_remeshed_heuristic, action_logprobs_dict_heuristic, action_entropies_dict_heuristic, action_probs_heuristic = policy.act(data_heuristic,reward_beta=sampled_reward,interp_index=k, is_timing=args.is_timing,is_eval_sample=args.is_eval_sample,debug=args.debug,data_gt=data_finegrain if args.debug else None,policy_input_feature=args.policy_input_feature,heuristic=True)
            
            data_remeshed_heuristic_gt_evl, action_logprobs_dict_heuristic, action_entropies_dict_heuristic, action_probs_heuristic = policy.act(data_heuristic_gt_evl,reward_beta=sampled_reward,interp_index=k, is_timing=args.is_timing,is_eval_sample=args.is_eval_sample,debug=args.debug,data_gt=data_finegrain if args.debug else None,policy_input_feature=args.policy_input_feature,heuristic=True)
            

        p.print("2.23", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        
        action_logprobs = torch.stack([value for key, value in action_logprobs_dict.items()], -1).sum(-1, keepdims=True)  # [B, 1]
        action_entropies = torch.stack([value for key, value in action_entropies_dict.items()], -1).sum(-1, keepdims=True)  # [B, 1]
        record_data(info, [action_logprobs, action_entropies, action_probs],
                          ["action_logprobs", "action_entropies", "action_probs"])  # (logprob_k, entropy_k, p_k), k=0,1,2,...H-1
        
        record_data(info, [elm.detach().cpu() for elm in action_logprobs_dict.values()],list(action_logprobs_dict.keys()))
        record_data(info, [elm.detach().cpu() for elm in action_entropies_dict.values()],list(action_entropies_dict.keys()))
        
        p.print("2.24", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        ##################################################
        # Evolution with new mesh: z_{k+1}'=(x_k,m_{k+1}) --f-> z_{k+1}=(x_{k+1},m_{k+1});loss_k
        ##################################################
        t_start = time.time()
        if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
            # data_remeshed_clone = data_remeshed.clone()
            _, _, data = evolution_model.one_step_interpolation_forward(data_remeshed, interp_index=k,is_timing=args.is_timing,use_remeshing=True,opt_evl=opt_evl,debug=args.debug,data_gt=data_finegrain if args.debug else None,noise_amp_val=args.noise_amp)
            if heuristic:
                data_remeshed_heuristic = detach_data(data_remeshed_heuristic)
                with torch.no_grad():
                    _, _, data_heuristic = evolution_model.one_step_interpolation_forward(data_remeshed_heuristic, interp_index=k,is_timing=args.is_timing,use_remeshing=True,opt_evl=opt_evl,debug=args.debug,data_gt=data_finegrain if args.debug else None,noise_amp_val=0)

                    _, _, data_heuristic_gt_evl = evolution_model_alt.one_step_interpolation_forward(data_remeshed_heuristic_gt_evl, interp_index=k,is_timing=args.is_timing,use_remeshing=True,opt_evl=opt_evl,debug=args.debug,data_gt=data_finegrain if args.debug else None,noise_amp_val=0)
        else:            
            data, info_k = evolution_model(data_remeshed, pred_steps=1, returns_data=True)
            
            if args.connect_bdd:
                head = torch.where(data.node_pos["n0"]==data.node_pos["n0"].min())[0]
                end = torch.where(data.node_pos["n0"]==data.node_pos["n0"].max())[0]
                end_features = (data.node_feature["n0"][head,:].clone()*0.5+data.node_feature["n0"][end,:].clone()*0.5)
                try:
                    data.node_feature["n0"][end,:]=end_features
                    data.node_feature["n0"][head,:]=end_features
                except:
                    pdb.set_trace()


        t_end = time.time()
        t_evolve = t_end - t_start
        p.print("2.25", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
            velocity = data.history[-1][:,3:] - data.history[0][:,3:]
            state_preds = Attr_Dict({
                # "x": data.x.clone().detach(),
                "x": velocity.clone().detach(),
                "x_pos": data.x_pos,
                "edge_index": data.edge_index,
                "xfaces": data.xfaces,
                "x_bdd": torch.empty((data.x.shape[0],0), dtype=torch.float32, device=data.x.device), 
                "original_shape": data.original_shape,
                "dyn_dims": data.dyn_dims,
                "compute_func": data.compute_func,
                "grid_keys": data.grid_keys,
                "part_keys": data.part_keys, 
                "time_step": data.time_step, 
                "sim_id": data.sim_id,
                "time_interval": data.time_interval, 
                "cushion_input": data.cushion_input,
                "bary_weights": data.bary_weights,
                "bary_indices": data.bary_indices,
                "hist_weights": data.hist_weights,
                "hist_indices": data.hist_indices,
                "yedge_index": data.yedge_index,
                "y_back": data.y_back,
                "y_tar": data.y_tar,
                "yface_list": data.yface_list,
                "history": [elm.clone().detach() for elm in data.history],
                "yfeatures": data.yfeatures,
                "node_dim": data.node_dim,
                "xface_list": data.xface_list,
                "reind_yfeatures": data.reind_yfeatures,
                "batch": data.batch,
                "batch_history": data.batch_history,
            })
            if "edge_attr" in data:
                state_preds.edge_attr = data.edge_attr.clone().detach()
            if "onehot_list" in data:
                state_preds.onehot_list = data.onehot_list
                state_preds.kinematics_list = data.kinematics_list 
            state_preds.interp_index=k     
        else:
            state_preds = Attr_Dict({
                "x": data.node_feature["n0"],  # e.g. shape [2000, 25, 1]
                "x_pos": data.node_pos["n0"],  # e.g. shape [2000, 1]
                "x_bdd": data.x_bdd["n0"],  # e.g. shape [2000, 1]
                "edge_index": data.edge_index[("n0", "0", "n0")],
                "batch": data.batch,
            })
        
        record_data(info, [state_preds, t_evolve], ["state_preds", "t_evolve"])

        if heuristic:
            velocity = data_heuristic.history[-1][:,3:] - data_heuristic.history[0][:,3:]
            state_preds_heuristic = Attr_Dict({
                # "x": data_heuristic.x.clone().detach(),
                "x": velocity.clone().detach(),
                "x_pos": data_heuristic.x_pos,
                "edge_index": data_heuristic.edge_index,
                "xfaces": data_heuristic.xfaces,
                "x_bdd": torch.empty((data_heuristic.x.shape[0],0), dtype=torch.float32, device=data_heuristic.x.device), 
                "original_shape": data_heuristic.original_shape,
                "dyn_dims": data_heuristic.dyn_dims,
                "compute_func": data_heuristic.compute_func,
                "grid_keys": data_heuristic.grid_keys,
                "part_keys": data_heuristic.part_keys, 
                "time_step": data_heuristic.time_step, 
                "sim_id": data_heuristic.sim_id,
                "time_interval": data_heuristic.time_interval, 
                "cushion_input": data_heuristic.cushion_input,
                "bary_weights": data_heuristic.bary_weights,
                "bary_indices": data_heuristic.bary_indices,
                "hist_weights": data_heuristic.hist_weights,
                "hist_indices": data_heuristic.hist_indices,
                "yedge_index": data_heuristic.yedge_index,
                "y_back": data_heuristic.y_back,
                "y_tar": data_heuristic.y_tar,
                "yface_list": data_heuristic.yface_list,
                "history": [elm.clone().detach() for elm in data_heuristic.history],
                "yfeatures": data_heuristic.yfeatures,
                "node_dim": data_heuristic.node_dim,
                "xface_list": data_heuristic.xface_list,
                "reind_yfeatures": data_heuristic.reind_yfeatures,
                "batch": data_heuristic.batch,
                "batch_history": data_heuristic.batch_history,
            })
            if "edge_attr" in data_heuristic:
                state_preds_heuristic.edge_attr = data_heuristic.edge_attr.clone().detach()
            if "onehot_list" in data_heuristic:
                state_preds_heuristic.onehot_list = data_heuristic.onehot_list
                state_preds_heuristic.kinematics_list = data_heuristic.kinematics_list 
            state_preds_heuristic.interp_index=k     
            record_data(info, [state_preds_heuristic, t_evolve], ["state_preds_heuristic", "t_evolve"])

            velocity = data_heuristic_gt_evl.history[-1][:,3:] - data_heuristic_gt_evl.history[0][:,3:]
            state_preds_heuristic_gt_evl = Attr_Dict({
                # "x": data_heuristic_gt_evl.x.clone().detach(),
                "x": velocity.clone().detach(),
                "x_pos": data_heuristic_gt_evl.x_pos,
                "edge_index": data_heuristic_gt_evl.edge_index,
                "xfaces": data_heuristic_gt_evl.xfaces,
                "x_bdd": torch.empty((data_heuristic_gt_evl.x.shape[0],0), dtype=torch.float32, device=data_heuristic_gt_evl.x.device), 
                "original_shape": data_heuristic_gt_evl.original_shape,
                "dyn_dims": data_heuristic_gt_evl.dyn_dims,
                "compute_func": data_heuristic_gt_evl.compute_func,
                "grid_keys": data_heuristic_gt_evl.grid_keys,
                "part_keys": data_heuristic_gt_evl.part_keys, 
                "time_step": data_heuristic_gt_evl.time_step, 
                "sim_id": data_heuristic_gt_evl.sim_id,
                "time_interval": data_heuristic_gt_evl.time_interval, 
                "cushion_input": data_heuristic_gt_evl.cushion_input,
                "bary_weights": data_heuristic_gt_evl.bary_weights,
                "bary_indices": data_heuristic_gt_evl.bary_indices,
                "hist_weights": data_heuristic_gt_evl.hist_weights,
                "hist_indices": data_heuristic_gt_evl.hist_indices,
                "yedge_index": data_heuristic_gt_evl.yedge_index,
                "y_back": data_heuristic_gt_evl.y_back,
                "y_tar": data_heuristic_gt_evl.y_tar,
                "yface_list": data_heuristic_gt_evl.yface_list,
                "history": [elm.clone().detach() for elm in data_heuristic_gt_evl.history],
                "yfeatures": data_heuristic_gt_evl.yfeatures,
                "node_dim": data_heuristic_gt_evl.node_dim,
                "xface_list": data_heuristic_gt_evl.xface_list,
                "reind_yfeatures": data_heuristic_gt_evl.reind_yfeatures,
                "batch": data_heuristic_gt_evl.batch,
                "batch_history": data_heuristic_gt_evl.batch_history,
            })
            if "edge_attr" in data_heuristic_gt_evl:
                state_preds_heuristic_gt_evl.edge_attr = data_heuristic_gt_evl.edge_attr.clone().detach()
            if "onehot_list" in data_heuristic_gt_evl:
                state_preds_heuristic_gt_evl.onehot_list = data_heuristic_gt_evl.onehot_list
                state_preds_heuristic_gt_evl.kinematics_list = data_heuristic_gt_evl.kinematics_list 
            state_preds_heuristic_gt_evl.interp_index=k     
            record_data(info, [state_preds_heuristic_gt_evl, t_evolve], ["state_preds_heuristic_gt_evl", "t_evolve"])
        ##################################################
        # Evolution with original mesh: z_{k+1}'=(x_k,m_k) --f-> z_{k+1}''=(x_{k+1}'',m_k);loss_k''
        ##################################################
        p.print("2.26", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        t_start_alt = time.time()

        if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
            if mode.startswith("test_gt_remesh"):
                _, _, data_alt_gt_mesh = evolution_model.one_step_interpolation_forward(data_not_remeshed_gt_mesh, interp_index=k,is_timing=args.is_timing,use_remeshing=False,opt_evl=False,mode=mode,noise_amp_val=0, debug=True,data_gt=data_not_remeshed_gt.clone())
                _, _, data_alt_gt_mesh_gt_evl = evolution_model_alt.one_step_interpolation_forward(data_not_remeshed_gt_mesh_gt_evl, interp_index=k,is_timing=args.is_timing,use_remeshing=False,opt_evl=False,mode=mode,noise_amp_val=0, debug=True,data_gt=data_not_remeshed_gt.clone())
                _, _, data_alt = evolution_model.one_step_interpolation_forward(data_not_remeshed, interp_index=k,is_timing=args.is_timing,use_remeshing=False,opt_evl=False,mode=mode,noise_amp_val=0)
                _, _, data_alt_gt_evl = evolution_model_alt.one_step_interpolation_forward(data_not_remeshed_gt_evl, interp_index=k,is_timing=args.is_timing,use_remeshing=False,opt_evl=False,mode=mode,noise_amp_val=0)
                _, _, data_alt_008_gt_evl = evolution_model_alt.one_step_interpolation_forward(data_not_remeshed_008_gt_evl, interp_index=k,is_timing=args.is_timing,use_remeshing=False,opt_evl=False,mode=mode,noise_amp_val=0, debug=True,data_gt=data_008_not_remeshed_gt.clone())
                _, _, data_alt_008 = evolution_model.one_step_interpolation_forward(data_not_remeshed_008, interp_index=k,is_timing=args.is_timing,use_remeshing=False,opt_evl=False,mode=mode,noise_amp_val=0, debug=True,data_gt=data_008_not_remeshed_gt.clone())
            elif not (mode=="test_remeshing"):
                # data_not_remeshed_clone = data_not_remeshed.clone()
                _, _, data_alt = evolution_model_alt.one_step_interpolation_forward(data_not_remeshed, interp_index=k,is_timing=args.is_timing,use_remeshing=False,opt_evl=False,mode=mode,noise_amp_val=args.noise_amp)
                # pdb.set_trace()
        else:
            data_alt, info_k_alt = evolution_model(data_not_remeshed, pred_steps=1, returns_data=True)
        t_end_alt = time.time()
        t_evolve_alt = t_end_alt - t_start_alt
        if not (mode=="test_remeshing"):
            p.print("2.27", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
            if len(dict(to_tuple_shape(data_alt.original_shape))["n0"]) == 0:
                velocity = data_alt.history[-1][:,3:] - data_alt.history[0][:,3:]
                state_preds_alt = Attr_Dict({
                    # "x": data_alt.x.clone().detach(),
                    "x": velocity.clone().detach(),
                    "x_pos": data_alt.x_pos,
                    "edge_index": data_alt.edge_index,
                    "xfaces": data_alt.xfaces,
                    "x_bdd":  torch.empty((data_alt.x.shape[0],0), dtype=torch.float32, device=data_alt.x.device), 
                    "original_shape": data_alt.original_shape,
                    "dyn_dims": data_alt.dyn_dims,
                    "compute_func": data_alt.compute_func,
                    "grid_keys": data_alt.grid_keys,
                    "part_keys": data_alt.part_keys, 
                    "time_step": data_alt.time_step, 
                    "sim_id": data_alt.sim_id,
                    "time_interval": data_alt.time_interval, 
                    "cushion_input": data_alt.cushion_input,
                    "bary_weights": data_alt.bary_weights,
                    "bary_indices": data_alt.bary_indices,
                    "hist_weights": data_alt.hist_weights,
                    "hist_indices": data_alt.hist_indices,
                    "yedge_index": data_alt.yedge_index,
                    "y_back": data_alt.y_back,
                    "y_tar": data_alt.y_tar,
                    "yface_list": data_alt.yface_list,
                    "history": [elm.clone().detach() for elm in data_alt.history],
                    "yfeatures": data_alt.yfeatures,
                    "node_dim": data_alt.node_dim,
                    "xface_list": data_alt.xface_list,
                    "reind_yfeatures": data_alt.reind_yfeatures,
                    "batch": data_alt.batch,
                    "batch_history": data_alt.batch_history,
                })
                if "edge_attr" in data_alt:
                    state_preds_alt.edge_attr = data_alt.edge_attr
                if "onehot_list" in data_alt:
                    state_preds_alt.onehot_list = data_alt.onehot_list
                    state_preds_alt.kinematics_list = data_alt.kinematics_list 
                state_preds_alt.interp_index = k
            else:
                state_preds_alt = Attr_Dict({
                    "x": data_alt.node_feature["n0"],  # e.g. shape [2000, 25, 1]
                    "x_pos": data_alt.node_pos["n0"],  # e.g. shape [2000, 1]
                    "x_bdd": data_alt.x_bdd["n0"],  # e.g. shape [2000, 1]
                    "edge_index": data_alt.edge_index[("n0", "0", "n0")],
                    "batch": data_alt.batch,
                })
            record_data(info, [state_preds_alt, t_evolve_alt], ["state_preds_alt", "t_evolve_alt"])
        if mode.startswith("test_gt_remesh"):
            velocity = data_alt_gt_evl.history[-1][:,3:] - data_alt_gt_evl.history[0][:,3:]
            state_preds_alt_gt_evl = Attr_Dict({
                # "x": data_alt_gt_evl.x.clone().detach(),
                "x": velocity.clone().detach(),
                "x_pos": data_alt_gt_evl.x_pos,
                "edge_index": data_alt_gt_evl.edge_index,
                "xfaces": data_alt_gt_evl.xfaces,
                "x_bdd":  torch.empty((data_alt_gt_evl.x.shape[0],0), dtype=torch.float32, device=data_alt_gt_evl.x.device), 
                "original_shape": data_alt_gt_evl.original_shape,
                "dyn_dims": data_alt_gt_evl.dyn_dims,
                "compute_func": data_alt_gt_evl.compute_func,
                "grid_keys": data_alt_gt_evl.grid_keys,
                "part_keys": data_alt_gt_evl.part_keys, 
                "time_step": data_alt_gt_evl.time_step, 
                "sim_id": data_alt_gt_evl.sim_id,
                "time_interval": data_alt_gt_evl.time_interval, 
                "cushion_input": data_alt_gt_evl.cushion_input,
                "bary_weights": data_alt_gt_evl.bary_weights,
                "bary_indices": data_alt_gt_evl.bary_indices,
                "hist_weights": data_alt_gt_evl.hist_weights,
                "hist_indices": data_alt_gt_evl.hist_indices,
                "yedge_index": data_alt_gt_evl.yedge_index,
                "y_back": data_alt_gt_evl.y_back,
                "y_tar": data_alt_gt_evl.y_tar,
                "yface_list": data_alt_gt_evl.yface_list,
                "history": [elm.clone().detach() for elm in data_alt_gt_evl.history],
                "yfeatures": data_alt_gt_evl.yfeatures,
                "node_dim": data_alt_gt_evl.node_dim,
                "xface_list": data_alt_gt_evl.xface_list,
                "reind_yfeatures": data_alt_gt_evl.reind_yfeatures,
                "batch": data_alt_gt_evl.batch,
                "batch_history": data_alt_gt_evl.batch_history,
            })
            if "edge_attr" in data_alt_gt_evl:
                state_preds_alt_gt_evl.edge_attr = data_alt_gt_evl.edge_attr
            if "onehot_list" in data_alt_gt_evl:
                state_preds_alt_gt_evl.onehot_list = data_alt_gt_evl.onehot_list
                state_preds_alt_gt_evl.kinematics_list = data_alt_gt_evl.kinematics_list 
            state_preds_alt_gt_evl.interp_index = k

            record_data(info, [state_preds_alt_gt_evl, t_evolve_alt], ["state_preds_alt_gt_evl", "t_evolve_alt"])

            velocity = data_alt_gt_mesh.history[-1][:,3:] - data_alt_gt_mesh.history[0][:,3:]
            state_preds_alt_gt_mesh = Attr_Dict({
                # "x": data_alt_gt_mesh.x.clone().detach(),
                "x": velocity.clone().detach(),
                "x_pos": data_alt_gt_mesh.x_pos,
                "edge_index": data_alt_gt_mesh.edge_index,
                "xfaces": data_alt_gt_mesh.xfaces,
                "x_bdd":  torch.empty((data_alt_gt_mesh.x.shape[0],0), dtype=torch.float32, device=data_alt_gt_mesh.x.device), 
                "original_shape": data_alt_gt_mesh.original_shape,
                "dyn_dims": data_alt_gt_mesh.dyn_dims,
                "compute_func": data_alt_gt_mesh.compute_func,
                "grid_keys": data_alt_gt_mesh.grid_keys,
                "part_keys": data_alt_gt_mesh.part_keys, 
                "time_step": data_alt_gt_mesh.time_step, 
                "sim_id": data_alt_gt_mesh.sim_id,
                "time_interval": data_alt_gt_mesh.time_interval, 
                "cushion_input": data_alt_gt_mesh.cushion_input,
                "bary_weights": data_alt_gt_mesh.bary_weights,
                "bary_indices": data_alt_gt_mesh.bary_indices,
                "hist_weights": data_alt_gt_mesh.hist_weights,
                "hist_indices": data_alt_gt_mesh.hist_indices,
                "yedge_index": data_alt_gt_mesh.yedge_index,
                "y_back": data_alt_gt_mesh.y_back,
                "y_tar": data_alt_gt_mesh.y_tar,
                "yface_list": data_alt_gt_mesh.yface_list,
                "history": [elm.clone().detach() for elm in data_alt_gt_mesh.history],
                "yfeatures": data_alt_gt_mesh.yfeatures,
                "node_dim": data_alt_gt_mesh.node_dim,
                "xface_list": data_alt_gt_mesh.xface_list,
                "reind_yfeatures": data_alt_gt_mesh.reind_yfeatures,
                "batch": data_alt_gt_mesh.batch,
                "batch_history": data_alt_gt_mesh.batch_history,
            })
            if "edge_attr" in data_alt_gt_mesh:
                state_preds_alt_gt_mesh.edge_attr = data_alt_gt_mesh.edge_attr
            if "onehot_list" in data_alt_gt_mesh:
                state_preds_alt_gt_mesh.onehot_list = data_alt_gt_mesh.onehot_list
                state_preds_alt_gt_mesh.kinematics_list = data_alt_gt_mesh.kinematics_list 
            state_preds_alt_gt_mesh.interp_index = k
            record_data(info, [state_preds_alt_gt_mesh, t_evolve_alt], ["state_preds_alt_gt_mesh", "t_evolve_alt"])
            
            velocity = data_alt_gt_mesh_gt_evl.history[-1][:,3:] - data_alt_gt_mesh_gt_evl.history[0][:,3:]
            state_preds_alt_gt_mesh_gt_evl = Attr_Dict({
                # "x": data_alt_gt_mesh_gt_evl.x.clone().detach(),
                "x": velocity.clone().detach(),
                "x_pos": data_alt_gt_mesh_gt_evl.x_pos,
                "edge_index": data_alt_gt_mesh_gt_evl.edge_index,
                "xfaces": data_alt_gt_mesh_gt_evl.xfaces,
                "x_bdd":  torch.empty((data_alt_gt_mesh_gt_evl.x.shape[0],0), dtype=torch.float32, device=data_alt_gt_mesh_gt_evl.x.device), 
                "original_shape": data_alt_gt_mesh_gt_evl.original_shape,
                "dyn_dims": data_alt_gt_mesh_gt_evl.dyn_dims,
                "compute_func": data_alt_gt_mesh_gt_evl.compute_func,
                "grid_keys": data_alt_gt_mesh_gt_evl.grid_keys,
                "part_keys": data_alt_gt_mesh_gt_evl.part_keys, 
                "time_step": data_alt_gt_mesh_gt_evl.time_step, 
                "sim_id": data_alt_gt_mesh_gt_evl.sim_id,
                "time_interval": data_alt_gt_mesh_gt_evl.time_interval, 
                "cushion_input": data_alt_gt_mesh_gt_evl.cushion_input,
                "bary_weights": data_alt_gt_mesh_gt_evl.bary_weights,
                "bary_indices": data_alt_gt_mesh_gt_evl.bary_indices,
                "hist_weights": data_alt_gt_mesh_gt_evl.hist_weights,
                "hist_indices": data_alt_gt_mesh_gt_evl.hist_indices,
                "yedge_index": data_alt_gt_mesh_gt_evl.yedge_index,
                "y_back": data_alt_gt_mesh_gt_evl.y_back,
                "y_tar": data_alt_gt_mesh_gt_evl.y_tar,
                "yface_list": data_alt_gt_mesh_gt_evl.yface_list,
                "history": [elm.clone().detach() for elm in data_alt_gt_mesh_gt_evl.history],
                "yfeatures": data_alt_gt_mesh_gt_evl.yfeatures,
                "node_dim": data_alt_gt_mesh_gt_evl.node_dim,
                "xface_list": data_alt_gt_mesh_gt_evl.xface_list,
                "reind_yfeatures": data_alt_gt_mesh_gt_evl.reind_yfeatures,
                "batch": data_alt_gt_mesh_gt_evl.batch,
                "batch_history": data_alt_gt_mesh_gt_evl.batch_history,
            })
            if "edge_attr" in data_alt_gt_mesh_gt_evl:
                state_preds_alt_gt_mesh_gt_evl.edge_attr = data_alt_gt_mesh_gt_evl.edge_attr
            if "onehot_list" in data_alt_gt_mesh_gt_evl:
                state_preds_alt_gt_mesh_gt_evl.onehot_list = data_alt_gt_mesh_gt_evl.onehot_list
                state_preds_alt_gt_mesh_gt_evl.kinematics_list = data_alt_gt_mesh_gt_evl.kinematics_list 
            state_preds_alt_gt_mesh_gt_evl.interp_index = k
            record_data(info, [state_preds_alt_gt_mesh_gt_evl, t_evolve_alt], ["state_preds_alt_gt_mesh_gt_evl", "t_evolve_alt"])


            velocity = data_alt_008_gt_evl.history[-1][:,3:] - data_alt_008_gt_evl.history[0][:,3:]
            state_preds_alt_008_gt_evl = Attr_Dict({
                # "x": data_alt_008_gt_evl.x.clone().detach(),
                "x": velocity.clone().detach(),
                "x_pos": data_alt_008_gt_evl.x_pos,
                "edge_index": data_alt_008_gt_evl.edge_index,
                "xfaces": data_alt_008_gt_evl.xfaces,
                "x_bdd":  torch.empty((data_alt_008_gt_evl.x.shape[0],0), dtype=torch.float32, device=data_alt_008_gt_evl.x.device), 
                "original_shape": data_alt_008_gt_evl.original_shape,
                "dyn_dims": data_alt_008_gt_evl.dyn_dims,
                "compute_func": data_alt_008_gt_evl.compute_func,
                "grid_keys": data_alt_008_gt_evl.grid_keys,
                "part_keys": data_alt_008_gt_evl.part_keys, 
                "time_step": data_alt_008_gt_evl.time_step, 
                "sim_id": data_alt_008_gt_evl.sim_id,
                "time_interval": data_alt_008_gt_evl.time_interval, 
                "cushion_input": data_alt_008_gt_evl.cushion_input,
                "bary_weights": data_alt_008_gt_evl.bary_weights,
                "bary_indices": data_alt_008_gt_evl.bary_indices,
                "hist_weights": data_alt_008_gt_evl.hist_weights,
                "hist_indices": data_alt_008_gt_evl.hist_indices,
                "yedge_index": data_alt_008_gt_evl.yedge_index,
                "y_back": data_alt_008_gt_evl.y_back,
                "y_tar": data_alt_008_gt_evl.y_tar,
                "yface_list": data_alt_008_gt_evl.yface_list,
                "history": [elm.clone().detach() for elm in data_alt_008_gt_evl.history],
                "yfeatures": data_alt_008_gt_evl.yfeatures,
                "node_dim": data_alt_008_gt_evl.node_dim,
                "xface_list": data_alt_008_gt_evl.xface_list,
                "reind_yfeatures": data_alt_008_gt_evl.reind_yfeatures,
                "batch": data_alt_008_gt_evl.batch,
                "batch_history": data_alt_008_gt_evl.batch_history,
            })
            if "edge_attr" in data_alt_008_gt_evl:
                state_preds_alt_008_gt_evl.edge_attr = data_alt_008_gt_evl.edge_attr
            if "onehot_list" in data_alt_008_gt_evl:
                state_preds_alt_008_gt_evl.onehot_list = data_alt_008_gt_evl.onehot_list
                state_preds_alt_008_gt_evl.kinematics_list = data_alt_008_gt_evl.kinematics_list 
            state_preds_alt_008_gt_evl.interp_index = k

            record_data(info, [state_preds_alt_008_gt_evl, t_evolve_alt], ["state_preds_alt_008_gt_evl", "t_evolve_alt"])


            velocity = data_alt_008.history[-1][:,3:] - data_alt_008.history[0][:,3:]
            state_preds_alt_008 = Attr_Dict({
                # "x": data_alt_008.x.clone().detach(),
                "x": velocity.clone().detach(),
                "x_pos": data_alt_008.x_pos,
                "edge_index": data_alt_008.edge_index,
                "xfaces": data_alt_008.xfaces,
                "x_bdd":  torch.empty((data_alt_008.x.shape[0],0), dtype=torch.float32, device=data_alt_008.x.device), 
                "original_shape": data_alt_008.original_shape,
                "dyn_dims": data_alt_008.dyn_dims,
                "compute_func": data_alt_008.compute_func,
                "grid_keys": data_alt_008.grid_keys,
                "part_keys": data_alt_008.part_keys, 
                "time_step": data_alt_008.time_step, 
                "sim_id": data_alt_008.sim_id,
                "time_interval": data_alt_008.time_interval, 
                "cushion_input": data_alt_008.cushion_input,
                "bary_weights": data_alt_008.bary_weights,
                "bary_indices": data_alt_008.bary_indices,
                "hist_weights": data_alt_008.hist_weights,
                "hist_indices": data_alt_008.hist_indices,
                "yedge_index": data_alt_008.yedge_index,
                "y_back": data_alt_008.y_back,
                "y_tar": data_alt_008.y_tar,
                "yface_list": data_alt_008.yface_list,
                "history": [elm.clone().detach() for elm in data_alt_008.history],
                "yfeatures": data_alt_008.yfeatures,
                "node_dim": data_alt_008.node_dim,
                "xface_list": data_alt_008.xface_list,
                "reind_yfeatures": data_alt_008.reind_yfeatures,
                "batch": data_alt_008.batch,
                "batch_history": data_alt_008.batch_history,
            })
            if "edge_attr" in data_alt_008:
                state_preds_alt_008.edge_attr = data_alt_008.edge_attr
            if "onehot_list" in data_alt_008:
                state_preds_alt_008.onehot_list = data_alt_008.onehot_list
                state_preds_alt_008.kinematics_list = data_alt_008.kinematics_list 
            state_preds_alt_008.interp_index = k

            record_data(info, [state_preds_alt_008, t_evolve_alt], ["state_preds_alt_008", "t_evolve_alt"])

        ##################################################
        # Evolution with reinterp fine grid mesh [idea lower bdd of mse]
        ##################################################
        p.print("2.28", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        if args.test_data_interp and step_num%args.wandb_step_plot==0:
            t_start_interp_alt = time.time()
            data_interp_alt, info_k_interp_alt = evolution_model(data_interp_alt_not_remeshed, pred_steps=1, returns_data=True)
            t_end_interp_alt = time.time()
            t_evolve_interp_alt = t_end_interp_alt - t_start_interp_alt
            state_preds_interp_alt = Attr_Dict({
                "x": data_interp_alt.node_feature["n0"],  # e.g. shape [2000, 25, 1]
                "x_pos": data_interp_alt.node_pos["n0"],  # e.g. shape [2000, 1]
                "x_bdd": data_interp_alt.x_bdd["n0"],  # e.g. shape [2000, 1]
                "edge_index": data_interp_alt.edge_index[("n0", "0", "n0")],
                "batch": data_interp_alt.batch,
            })
            record_data(info, [state_preds_interp_alt, t_evolve_interp_alt], ["state_preds_interp_alt", "t_evolve_interp_alt"])
            
        if opt_evl and args.rl_finetune_evalution_mode in ["policy:fine","policy:alt:fine"]:    
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                t_start_fine = time.time()
                _, _, data_fine = evolution_model.one_step_interpolation_forward(data_fine, interp_index=k,use_remeshing=False,changing_mesh=True,opt_evl=opt_evl,noise_amp_val=args.noise_amp)
                t_end_fine = time.time()
                t_evolve_fine = t_end_fine - t_start_fine
                # state_preds_fine = data_fine.clone()
                
                velocity = data_fine.history[-1][:,3:] - data_fine.history[0][:,3:]
                state_preds_fine = Attr_Dict({
                    # "x": data_fine.x.clone().detach(),
                    "x": velocity.clone().detach(),
                    "x_pos": data_fine.x_pos,
                    "edge_index": data_fine.edge_index,
                    "xfaces": data_fine.xfaces,
                    "x_bdd":  torch.empty((data_fine.x.shape[0],0), dtype=torch.float32, device=data_fine.x.device), 
                    "original_shape": data_fine.original_shape,
                    "dyn_dims": data_fine.dyn_dims,
                    "compute_func": data_fine.compute_func,
                    "grid_keys": data_fine.grid_keys,
                    "part_keys": data_fine.part_keys, 
                    "time_step": data_fine.time_step, 
                    "sim_id": data_fine.sim_id,
                    "time_interval": data_fine.time_interval, 
                    "cushion_input": data_fine.cushion_input,
                    "bary_weights": data_fine.bary_weights,
                    "bary_indices": data_fine.bary_indices,
                    "hist_weights": data_fine.hist_weights,
                    "hist_indices": data_fine.hist_indices,
                    "yedge_index": data_fine.yedge_index,
                    "y_back": data_fine.y_back,
                    "y_tar": data_fine.y_tar,
                    "yface_list": data_fine.yface_list,
                    "history": [elm.clone().detach() for elm in data_fine.history],
                    "yfeatures": data_fine.yfeatures,
                    "node_dim": data_fine.node_dim,
                    "xface_list": data_fine.xface_list,
                    "reind_yfeatures": data_fine.reind_yfeatures,
                    "batch": data_fine.batch,
                    "batch_history": data_fine.batch_history,
                })
                if "edge_attr" in data_fine:
                    state_preds_fine.edge_attr = data_fine.edge_attr
                if "onehot_list" in data_fine:
                    state_preds_fine.onehot_list = data_fine.onehot_list
                    state_preds_fine.kinematics_list = data_fine.kinematics_list 

                record_data(info, [state_preds_fine, t_evolve_fine], ["state_preds_fine", "t_evolve_fine"])

            else:
                t_start_fine = time.time()
                data_fine, info_k_fine = evolution_model(data_fine, pred_steps=1, returns_data=True)
                t_end_fine = time.time()
                t_evolve_fine = t_end_fine - t_start_fine
                state_preds_fine = Attr_Dict({
                    "x": data_fine.node_feature["n0"],  # e.g. shape [2000, 25, 1]
                    "x_pos": data_fine.node_pos["n0"],  # e.g. shape [2000, 1]
                    "x_bdd": data_fine.x_bdd["n0"],  # e.g. shape [2000, 1]
                    "edge_index": data_fine.edge_index[("n0", "0", "n0")],
                    "batch": data_fine.batch,
                })
                record_data(info, [state_preds_fine, t_evolve_fine], ["state_preds_fine", "t_evolve_fine"])

        p.print("2.29", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
        if args.reward_src == "env":
            # Using the environment to obtain the reward:
            if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                # pdb.set_trace()
                loss, loss_mse, data_gt = get_loss_2d_refined(data, data_finegrain, k, opt_evl=opt_evl)
                if mode.startswith("test_gt_remesh"):
                    loss_alt, loss_alt_mse, _ = get_loss_2d_refined(data_alt, data_finegrain, k, opt_evl=opt_evl)
                    record_data(info, [loss_alt_mse], ["evolution/loss_alt_mse"])
                    record_data(info, [loss_alt], ["evolution/loss_alt"])
                    loss_alt_gt_mesh, loss_alt_gt_mesh_mse, _ = get_loss_2d_refined(data_alt_gt_mesh, data_finegrain, k, opt_evl=opt_evl)
                    record_data(info, [loss_alt_gt_mesh_mse], ["evolution/loss_alt_gt_mesh_mse"])
                    record_data(info, [loss_alt_gt_mesh], ["evolution/loss_alt_gt_mesh"])
                    loss_alt_gt_evl, loss_alt_gt_evl_mse, _ = get_loss_2d_refined(data_alt_gt_evl, data_finegrain, k, opt_evl=opt_evl)
                    record_data(info, [loss_alt_gt_evl_mse], ["evolution/loss_alt_gt_evl_mse"])
                    record_data(info, [loss_alt_gt_evl], ["evolution/loss_alt_gt_evl"])
                    loss_alt_gt_mesh_gt_evl, loss_alt_gt_mesh_gt_evl_mse, _ = get_loss_2d_refined(data_alt_gt_mesh_gt_evl, data_finegrain, k, opt_evl=opt_evl)
                    record_data(info, [loss_alt_gt_mesh_gt_evl_mse], ["evolution/loss_alt_gt_mesh_gt_evl_mse"])
                    record_data(info, [loss_alt_gt_mesh_gt_evl], ["evolution/loss_alt_gt_mesh_gt_evl"])

                    loss_alt_008, loss_alt_008_mse, _ = get_loss_2d_refined(data_alt_008, data_finegrain, k, opt_evl=opt_evl)
                    record_data(info, [loss_alt_008_mse], ["evolution/loss_alt_008_mse"])
                    record_data(info, [loss_alt_008], ["evolution/loss_alt_008"])

                    loss_alt_008_gt_evl, loss_alt_008_gt_evl_mse, _ = get_loss_2d_refined(data_alt_008_gt_evl, data_finegrain, k, opt_evl=opt_evl)
                    record_data(info, [loss_alt_008_gt_evl_mse], ["evolution/loss_alt_008_gt_evl_mse"])
                    record_data(info, [loss_alt_008_gt_evl], ["evolution/loss_alt_008_gt_evl"])
                    if heuristic:
                        loss_heuristic, loss_heuristic_mse, _ = get_loss_2d_refined(data_heuristic, data_finegrain, k, opt_evl=opt_evl)
                        record_data(info, [loss_heuristic_mse], ["evolution/loss_heuristic_mse"])
                        record_data(info, [loss_heuristic], ["evolution/loss_heuristic"])

                        loss_heuristic_gt_evl, loss_heuristic_gt_evl_mse, _ = get_loss_2d_refined(data_heuristic_gt_evl, data_finegrain, k, opt_evl=opt_evl)
                        record_data(info, [loss_heuristic_gt_evl_mse], ["evolution/loss_heuristic_gt_evl_mse"])
                        record_data(info, [loss_heuristic_gt_evl], ["evolution/loss_heuristic_gt_evl"])

                elif not (mode=="test_remeshing"): 
                    loss_alt, loss_alt_mse, data_alt_gt = get_loss_2d_refined(data_alt, data_finegrain, k, opt_evl=opt_evl)
                    record_data(info, [loss_alt_mse], ["evolution/loss_alt_mse"])
                    record_data(info, [loss_alt], ["evolution/loss_alt"])
                
                        
                record_data(info, [loss_mse], ["evolution/loss_mse"])
                record_data(info, [loss], ["evolution/loss"])
                
            else:
                loss = get_loss_refined(data, data_finegrain, k)
                loss_alt = get_loss_refined(data_alt, data_finegrain, k)  
                record_data(info, [loss_alt], ["evolution/loss_alt"])
                record_data(info, [loss], ["evolution/loss"])
            if not (mode=="test_remeshing"):     
                reward, reward_info = get_reward_batch(
                        loss=loss,
                        state=state_preds,
                        time=t_evolve,
                        loss_alt=loss_alt,
                        state_alt=state_preds_alt,
                        time_alt=t_evolve_alt,
                        reward_mode=args.reward_mode,
                        # reward_beta=args.reward_beta,
                        reward_loss_coef=args.reward_loss_coef,
                        reward_beta=sampled_reward,
                    )
            else:
                reward, reward_info = get_reward_batch(
                        loss=loss,
                        state=state_preds,
                        time=t_evolve,
                        loss_alt=loss,
                        state_alt=state_preds,
                        time_alt=t_evolve,
                        reward_mode=args.reward_mode,
                        # reward_beta=args.reward_beta,
                        reward_loss_coef=args.reward_loss_coef,
                        reward_beta=sampled_reward,
                    )

            record_data(info, [reward], ["rewards"])
            record_data(info, list(reward_info.values()), list(reward_info.keys()))
            if opt_evl and (args.rl_finetune_evalution_mode in ["policy:fine","policy:alt:fine"]):  
                if len(dict(to_tuple_shape(data.original_shape))["n0"]) == 0:
                    # pdb.set_trace()
                    loss_fine, loss_fine_mse, data_fine_gt = get_loss_2d_refined(data_fine, data_finegrain_fine, k, opt_evl=opt_evl)
                    # print("k",k,"loss_fine",loss_fine)
                    record_data(info, [loss_fine], ["evolution/loss_fine"])
                    record_data(info, [loss_fine_mse], ["evolution/loss_fine_mse"])
                else:
                    loss_fine = get_loss_refined(data_fine, data_finegrain, k)  
                    record_data(info, [loss_fine], ["evolution/loss_fine"])

            if args.test_data_interp and step_num%args.wandb_step_plot==0:
                loss_interp_alt = get_loss_refined(data_interp_alt, data_finegrain, k)
                reward_interp_alt, reward_interp_alt_info = get_reward_batch(
                    loss=loss_interp_alt,
                    state=state_preds_interp_alt,
                    time=t_evolve_interp_alt,
                    loss_alt=loss_alt,
                    state_alt=state_preds_alt,
                    time_alt=t_evolve_alt,
                    reward_mode=args.reward_mode,
                    # reward_beta=args.reward_beta,
                    reward_loss_coef=args.reward_loss_coef,
                    prefix="interp_",
                    reward_beta=sampled_reward,
                )
                record_data(info, [loss_interp_alt], ["evolution/loss_interp_alt"])
                record_data(info, [reward_interp_alt], ["rewards_interp_alt"])
                record_data(info, list(reward_interp_alt_info.values()), list(reward_interp_alt_info.keys()))
        elif args.reward_src == "pred":
            # Using predicted reward:
            record_data(info, [info_k["rewards"]], ["reward_preds"])
        else:
            raise

    p.print("2.210", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
    info["action_logprobs"] = torch.stack(info["action_logprobs"])     # [H, B, 1]
    info["action_entropies"] = torch.stack(info["action_entropies"])   # [H, B, 1]
    for key in [
        "reward_preds", "rewards", "rewards_interp_alt",
        "evolution/loss", "evolution/loss_alt", "evolution/loss_interp_alt", "evolution/loss_fine",
        "evolution/loss_mse", "evolution/loss_alt_mse", "evolution/loss_interp_alt_mse", "evolution/loss_fine_mse",
        "evolution/loss_alt_gt_mesh_mse","evolution/loss_alt_gt_evl_mse","evolution/loss_alt_gt_mesh_gt_evl_mse",
        "evolution/loss_alt_008_mse","evolution/loss_alt_008_gt_evl_mse","evolution/loss_heuristic_mse","evolution/loss_heuristic_gt_evl_mse",
        "r/lossdiff", "r/statediff", "r/timediff",
        "v/lossdiff", "v/statediff", "v/timediff",
        "v/beta", "interp_v/beta",
        "interp_r/lossdiff", "interp_r/statediff", "interp_r/timediff",
        "interp_v/lossdiff", "interp_v/statediff", "interp_v/timediff", 
        "v/state_size", "interp_v/state_size",
        "logpK_split","logpK_coarse","logpK_flip",
        "ksplit_entropy","kcoarse_entropy","kflip_entropy",
        "split_edge_remesh_logp","split_edge_remesh_entropy",
        "coarse_logp","coarse_entropy",
        "split_edge_remesh_prob", "coarse_prob","flip_edge_remesh_prob",
        "split/k_entropy","coarse/k_entropy",
        "split/k_logp","coarse/k_logp",
        "split/logp","coarse/logp",
        "split/entropy", "coarse/entropy",
    ]:
        if key in info:
            info[key] = torch.stack(info[key])  # [H, B, 1]
    p.print("2.211", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
    if "data_fine" in locals() and data_fine!=None:
        data_fine.to("cpu")
        del data_fine
    if "data_finegrain" in locals():
        data_finegrain.to("cpu")
        del data_finegrain
    if "data_finegrain_fine" in locals():
        data_finegrain_fine.to("cpu")
        del data_finegrain_fine
    if "data_interp" in locals():
        data_interp.to("cpu")
        del data_interp
    if is_gc_collect and step_num % 100 == 0:
        gc.collect()
    p.print("2.212", precision="millisecond", is_silent=args.is_timing<2, avg_window=1)
    return data, info


# # EBM:

# ### Model definition:

# In[ ]:


class ConservEBM(nn.Module):
    def __init__(
        self,
        mode="Siamese-4-sum",
        net_mode="cnn",
        combine_mode="catdiff",
        in_channels=10,
        input_shape=None,
        channel_base=128,
        is_spec_norm=True,
        act_name="elu",
    ):
        super(ConservEBM, self).__init__()
        self.mode = mode
        self.net_mode = net_mode
        self.combine_mode = combine_mode
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.pos_dim = len(input_shape)
        self.channel_base = channel_base
        self.aggr_mode = self.mode.split("-")[2]
        self.is_spec_norm = is_spec_norm
        self.act_name = act_name
        if self.mode.startswith("Siamese"):
            output_size = eval(self.mode.split("-")[1])
            if self.combine_mode == "concat":
                combine_output_size = output_size * 2
            elif self.combine_mode == "catdiff":
                combine_output_size = output_size * 3
            elif self.combine_mode in ["diff", "mse"]:
                combine_output_size = output_size
            else:
                raise
            if self.net_mode == "mlp":
                self.net = MLP(input_size=in_channels*int(np.prod(input_shape)),
                               n_neurons=32,
                               n_layers=3,
                               output_size=output_size,
                               act_name=self.act_name,
                               last_layer_linear=True,
                              )
            elif self.net_mode == "cnn":
                if is_spec_norm:
                    self.conv1 = spectral_norm(nn.Conv2d(in_channels, channel_base, 3, padding=1), std=1)
                else:
                    self.conv1 = nn.Conv2d(in_channels, channel_base, 3, padding=1)
                self.blocks_branch = nn.ModuleList([
                    ResBlock(channel_base, channel_base, downsample=True, is_spec_norm=is_spec_norm),
                    ResBlock(channel_base, channel_base, is_spec_norm=is_spec_norm),
                    ResBlock(channel_base, channel_base*2, downsample=True, is_spec_norm=is_spec_norm),
                    ResBlock(channel_base*2, channel_base*2, is_spec_norm=is_spec_norm),
                    ResBlock(channel_base*2, channel_base*2, downsample=True, is_spec_norm=is_spec_norm),
                    ResBlock(channel_base*2, channel_base*2, is_spec_norm=is_spec_norm),
                ])
                self.linear = nn.Linear(channel_base*2, output_size)
            else:
                raise
            net_combine_n_layers = eval(self.combine_mode.split("-")[1]) if len(self.combine_mode.split("-")) > 1 else 2
            self.net_combine = MLP(
                input_size=combine_output_size,
                n_neurons=combine_output_size,
                n_layers=net_combine_n_layers,
                output_size=1,
                act_name=self.act_name,
                last_layer_linear=False,
            )


    def forward(self, input, future, mask=None):
        """Given input and future pair, return a single energy value for each pair.

        Args:
            input:  [B, n_nodes, C]
            future: [B, n_nodes, C]
            mask:   [1, n_nodes, 1]. Default None.

            where B is the batch size, C is the number of channels, and n_nodes 
            is the number of nodes in one example/state. At one input state, there
            can be multiple nodes, for example in 2D grid where at each time, there 
            are H*W nodes. In pendulum, there is only one node at each state. The nodes
            is denoting the spatial degree of freedom. Remember that in the dataset, 
            each data.x has dimension of [n_nodes, time_steps, feature_size]. This 
            representation is universal, no matter how complicated the system is.
        """

        if mask is not None:
            assert len(mask.shape) == 3 and mask.shape[0] == 1 and mask.shape[2] == 1  # mask: [1, n_nodes, 1]
            input = input * mask
            future = future * mask
        if self.mode.startswith("Siamese"):
            if self.net_mode == "mlp":
                input = input.view(input.shape[0], -1)
                future = future.view(future.shape[0], -1)
                out_0 = self.net(input)
                out_1 = self.net(future)

            elif self.net_mode == "cnn":
                def aggr_fn(out, aggr_mode):
                    if aggr_mode == "sum":
                        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
                    elif aggr_mode == "max":
                        out = out.view(out.shape[0], out.shape[1], -1).max(2)[0]
                    else:
                        raise
                    return out

                # Input branch:
                input = input.reshape(input.shape[0], *self.input_shape, input.shape[-1])  # input: previous: [B, n_nodes:H*W, C]; after: [B, H, W, C]
                input = input.permute(0,3,1,2)
                out_0 = F.leaky_relu(self.conv1(input), negative_slope=0.2)

                for i, block in enumerate(self.blocks_branch):
                    out_0 = block(out_0)
                out_0 = F.relu(out_0)
                out_0 = aggr_fn(out_0, self.aggr_mode)  # [B, channel_base]
                out_0 = self.linear(out_0)

                # Future branch:
                future = future.reshape(future.shape[0], *self.input_shape, future.shape[-1])
                future = future.permute(0,3,1,2)
                out_1 = F.leaky_relu(self.conv1(future), negative_slope=0.2)
                for i, block in enumerate(self.blocks_branch):
                    out_1 = block(out_1)
                out_1 = F.relu(out_1)
                out_1 = aggr_fn(out_1, self.aggr_mode)  # [B, channel_base]
                out_1 = self.linear(out_1)
            else:
                raise

            if self.combine_mode == "concat":
                out = self.net_combine(torch.cat([out_0, out_1], 1))
            elif self.combine_mode == "catdiff":
                out = self.net_combine(torch.cat([out_0, out_1, out_1-out_0], 1))
            elif self.combine_mode == "diff":
                out = self.net_combine(out_1-out_0)
            elif self.combine_mode == "mse":
                out = self.net_combine((out_1-out_0).square())
            else:
                raise
        else:
            raise
        return out, (out_0, out_1)


    @property
    def model_dict(self):
        model_dict = {"type": "ConservEBM"}
        model_dict["mode"] = self.mode
        model_dict["net_mode"] = self.net_mode
        model_dict["combine_mode"] = self.combine_mode
        model_dict["in_channels"] = self.in_channels
        model_dict["input_shape"] = self.input_shape
        model_dict["channel_base"] = self.channel_base
        model_dict["is_spec_norm"] = self.is_spec_norm
        model_dict["act_name"] = self.act_name
        model_dict["state_dict"] = to_cpu(self.state_dict())
        return model_dict


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, input_size, output_size):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(input_size + 1, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


################################################################
# 2d fourier layers
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, input_size, output_size):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(input_size + 2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, input_size, output_size):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(input_size + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # After: [B, C, H, W, D]
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)  # [B, C, H, W, D]
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # After: [B, H, W, D, C]. pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class FNOModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        input_shape,
        modes=None,
        width=None,
        loss_type="lp",
        temporal_bundle_steps=1,
        static_encoder_type="None",
        static_latent_size=0,
    ):
        super().__init__()
        self.input_size = input_size  # steps*feature_size
        self.output_size = output_size
        self.input_shape = input_shape
        self.loss_type = loss_type
        self.temporal_bundle_steps = temporal_bundle_steps
        self.static_encoder_type = static_encoder_type
        self.static_latent_size = static_latent_size
        self.pos_dims = len(input_shape)
        static_latent_size_core = self.static_latent_size if self.static_encoder_type.startswith("param") else 0
        if self.pos_dims == 1:
            self.modes = 16 if modes is None else modes
            self.width = 64 if width is None else width
            self.model = FNO1d(self.modes, width=self.width, input_size=self.input_size+static_latent_size_core, output_size=self.output_size)
        elif self.pos_dims == 2:
            self.modes = 12 if modes is None else modes
            self.width = 20 if width is None else width
            self.model = FNO2d(self.modes, self.modes, width=self.width, input_size=self.input_size+static_latent_size_core, output_size=self.output_size)
        elif self.pos_dims == 3:
            self.modes = 8 if modes is None else modes
            self.width = 20 if width is None else width
            self.model = FNO3d(self.modes, self.modes, self.modes, width=self.width, input_size=self.input_size+static_latent_size_core, output_size=self.output_size)
        else:
            raise Exception("self.pos_dims can only be 1, 2 or 3!")


    def forward(
        self,
        data,
        pred_steps=1,
        **kwargs
    ):
        if not isinstance(pred_steps, list) and not isinstance(pred_steps, np.ndarray):
            pred_steps = [pred_steps]
        max_pred_steps = max(pred_steps)
        original_shape = dict(to_tuple_shape(data.original_shape))
        n_pos = np.array(original_shape["n0"]).prod()
        batch_size = data.node_feature["n0"].shape[0] // n_pos

        info = {}
        x = data.node_feature["n0"]
        x = x.reshape(batch_size, *self.input_shape, *x.shape[-2:])  # [B, (H, W, D), steps, feature_size]
        time_steps = x.shape[-2]
        x_feature_size = x.shape[-1]
        dyn_dims = dict(to_tuple_shape(data.dyn_dims))["n0"]
        static_dims = x_feature_size - dyn_dims
        if static_dims > 0:
            static_features = x[...,-self.temporal_bundle_steps:,:-dyn_dims]  # [B, (H, W, D), 1, static_dims]
        pos_dims = len(self.input_shape)
        assert pos_dims in [1,2,3]
        x = x.flatten(start_dim=1+pos_dims) # x: [B, (H, W, D), steps*feature_size]  , [20,64,64,10]
        if self.static_encoder_type.startswith("param"):
            static_param = data.param["n0"].reshape(batch_size, -1)  # [B, self.static_latent_size]
            static_param_expand = static_param
            for jj in range(pos_dims):
                static_param_expand = static_param_expand[...,None,:]
            static_param_expand = static_param_expand.expand(batch_size, *self.input_shape, static_param_expand.shape[-1])  # [B, (H, W, D), static_dims]
        preds = {"n0": []}

        is_multistep_detach = kwargs["is_multistep_detach"] if "is_multistep_detach" in kwargs else False

        for k in range(1, max_pred_steps + 1):
            # x: [B, (H, W, D), steps*feature_size]
            if self.static_encoder_type.startswith("param"):
                pred = self.model(torch.cat([static_param_expand, x], -1))
            else:
                pred = self.model(x)  # pred: [B, (H, W, D), temporal_bundle_steps*dyn_dims], x: [B, (H, W, D), steps*feature_size+static_latent_size]
            x_reshape = x.reshape(*x.shape[:1+pos_dims], time_steps, x_feature_size)  # [B, (H, W, D), steps, feature_size]
            pred_reshape = pred.reshape(*pred.shape[:1+pos_dims], self.temporal_bundle_steps, dyn_dims)  # [B, (H, W, D), self.temporal_bundle_steps, dyn_dims]
            if k in pred_steps:
                preds["n0"].append(pred_reshape)
            if static_dims > 0:
                pred_reshape = torch.cat([static_features, pred_reshape], -1)  # [B, (H, W, D), 1, x_feature_size]
            new_x_reshape = torch.cat([x_reshape, pred_reshape], -2)[...,-time_steps:,:]   # [B, H, W, input_steps, x_feature_size]
            x = new_x_reshape.flatten(start_dim=1+pos_dims)  # x:   # [B, (H, W, D), input_steps*x_feature_size]
            if is_multistep_detach:
                x = x.detach()
        # Before concat, each element of preds["n0"] has shape of [B, (H, W, D), self.temporal_bundle_steps, dyn_dims]
        preds["n0"] = torch.cat(preds["n0"], -2)
        preds["n0"] = preds["n0"].reshape(-1, len(pred_steps) * self.temporal_bundle_steps, dyn_dims)  # [B*n_nodes_B, pred_steps * self.temporal_bundle_steps, dyn_dims]
        return preds, info


    def get_loss(self, data, args, is_rollout=False, **kwargs):
        multi_step_dict = parse_multi_step(args.multi_step)
        preds, info = self(
            data,
            pred_steps=list(multi_step_dict.keys()),
            is_multistep_detach=args.is_multistep_detach,
        )

        original_shape = dict(to_tuple_shape(data.original_shape))
        n_pos = np.array(original_shape["n0"]).prod()
        batch_size = data.node_feature["n0"].shape[0] // n_pos

        # Prediction loss:
        self.info = {}
        loss = 0
        for pred_idx, k in enumerate(multi_step_dict):
            pred_idx_list = np.arange(pred_idx*args.temporal_bundle_steps, (pred_idx+1)*args.temporal_bundle_steps).tolist()
            y_idx_list = np.arange((k-1)*args.temporal_bundle_steps, k*args.temporal_bundle_steps).tolist()
            loss_k = loss_op(
                preds, data.node_label, data.mask,
                pred_idx=pred_idx_list,
                y_idx=y_idx_list,
                loss_type=args.loss_type,
                batch_size=batch_size,
                is_y_variable_length=args.is_y_variable_length,
                **kwargs
            )
            loss = loss + loss_k
        self.info["loss_pred"] = to_np_array(loss)
        return loss

    @property
    def model_dict(self):
        model_dict = {"type": "FNOModel"}
        model_dict["input_size"] = self.input_size
        model_dict["output_size"] = self.output_size
        model_dict["input_shape"] = self.input_shape
        model_dict["width"] = self.width
        model_dict["modes"] = self.modes
        model_dict["loss_type"] = self.loss_type
        model_dict["temporal_bundle_steps"] = self.temporal_bundle_steps
        model_dict["static_encoder_type"] = self.static_encoder_type
        model_dict["static_latent_size"] = self.static_latent_size
        model_dict["state_dict"] = to_cpu(self.state_dict())
        return model_dict





def get_loss(model, data, args, is_rollout=False, **kwargs):
    if args.algo == "contrast":
        return get_loss_contrast(model, data, args, is_rollout=False, **kwargs)
    else:
        raise


# ### Load model:

# In[ ]:


def load_model(model_dict, device, multi_gpu=False, **kwargs):
    """Load saved model using model_dict."""
    def process_state_dict(state_dict):
        """Deal with SpectralNorm:"""
        keys_to_delete = []
        for key in state_dict:
            if key.endswith("weight_bar"):
                keys_to_delete.append(key[:-4])
        for key in keys_to_delete:
            if key in state_dict:
                state_dict.pop(key)
        return state_dict

    model_type = model_dict["type"]

    if model_type in ["GNNRemesher", "Value_Model_Summation"]:
        model = GNNRemesher(
            input_size=model_dict["input_size"],
            output_size=model_dict["output_size"],
            edge_dim=model_dict["edge_dim"],
            sizing_field_dim=model_dict["sizing_field_dim"],
            reward_dim=model_dict["reward_dim"] if "reward_dim" in model_dict else 0,
            nmax=model_dict["nmax"],
            latent_dim=model_dict["latent_dim"],
            num_steps=model_dict["num_steps"],
            layer_norm=model_dict["layer_norm"],
            act_name=model_dict["act_name"],
            var=model_dict["var"],
            batch_norm=model_dict["batch_norm"],
            normalize=model_dict["normalize"],
            diffMLP=model_dict["diffMLP"],
            checkpoints=model_dict["checkpoints"],
            samplemode=model_dict["samplemode"],
            use_encoder=model_dict["use_encoder"],
            edge_attr=model_dict["edge_attr"],
            is_split_test=model_dict["is_split_test"],
            is_flip_test=model_dict["is_flip_test"],
            is_coarsen_test=model_dict["is_coarsen_test"],
            skip_split=model_dict["skip_split"],
            skip_flip=model_dict["skip_flip"],
            is_y_diff=model_dict["is_y_diff"] if "is_y_diff" in model_dict else False,
        )
    elif model_type == "GNNRemesherPolicy":
        model = GNNRemesherPolicy(
            input_size=model_dict["input_size"],
            output_size=model_dict["output_size"],
            edge_dim=model_dict["edge_dim"],
            sizing_field_dim=model_dict["sizing_field_dim"],
            reward_dim=model_dict["reward_dim"] if "reward_dim" in model_dict else 0,
            nmax=model_dict["nmax"],
            latent_dim=model_dict["latent_dim"],
            num_steps=model_dict["num_steps"],
            layer_norm=model_dict["layer_norm"],
            act_name=model_dict["act_name"],
            var=model_dict["var"],
            batch_norm=model_dict["batch_norm"],
            normalize=model_dict["normalize"],
            diffMLP=model_dict["diffMLP"],
            checkpoints=model_dict["checkpoints"],
            samplemode=model_dict["samplemode"],
            use_encoder=model_dict["use_encoder"],
            edge_attr=model_dict["edge_attr"],
            is_split_test=model_dict["is_split_test"],
            is_flip_test=model_dict["is_flip_test"],
            is_coarsen_test=model_dict["is_coarsen_test"],
            skip_split=model_dict["skip_split"],
            skip_flip=model_dict["skip_flip"],
            is_y_diff=model_dict["is_y_diff"] if "is_y_diff" in model_dict else False,
            edge_threshold=model_dict["edge_threshold"],
            noise_amp=model_dict["noise_amp"],
            correction_rate=model_dict["correction_rate"],
        )
    elif model_type == "Value_Model":
        model = Value_Model(
            input_size=model_dict["input_size"],
            output_size=model_dict["output_size"],
            edge_dim=model_dict["edge_dim"],
            latent_dim=model_dict["latent_dim"],
            num_pool=model_dict["num_pool"],
            act_name=model_dict["act_name"],
            act_name_final=model_dict["act_name_final"],
            layer_norm=model_dict["layer_norm"],
            batch_norm=model_dict["batch_norm"],            
            num_steps=model_dict["num_steps"],
            pooling_type=model_dict["pooling_type"],
            edge_attr=model_dict["edge_attr"],
            use_pos=model_dict["use_pos"],
            reward_condition=model_dict["reward_condition"]
        )
    elif model_type == "Actor_Critic":
        critic = load_model(model_dict["critic_model_dict"], device)
        if "critic_target_model_dict" in model_dict.keys():
            critic_target = load_model(model_dict["critic_target_model_dict"], device)
        else:
            critic_target = None 
        actor = load_model(model_dict["actor_model_dict"], device)
        model = Actor_Critic(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            horizon=model_dict["horizon"],
        )
    elif model_type == "GNNPolicySizing":
        model = GNNPolicySizing(
            dataset=model_dict["dataset"],
            input_size=model_dict["input_size"],
            output_size=model_dict["output_size"],
            sizing_field_dim=model_dict["sizing_field_dim"],
            nmax=model_dict["nmax"],
            batch_size=model_dict["batch_size"],
            edge_dim=model_dict["edge_dim"],
            latent_dim=model_dict["latent_dim"],
            num_steps=model_dict["num_steps"],
            layer_norm=model_dict["layer_norm"],
            batch_norm=model_dict["batch_norm"],
            edge_attr=model_dict["edge_attr"],
            edge_threshold=model_dict["edge_threshold"],
            skip_split=model_dict["skip_split"],
            skip_flip=model_dict["skip_flip"],
            act_name=model_dict["act_name"],
            is_split_test=model_dict["is_split_test"],
            is_flip_test=model_dict["is_flip_test"],
            is_coarsen_test=model_dict["is_coarsen_test"],
            samplemode=model_dict["samplemode"],
            rescale=model_dict["rescale"],
            min_edge_size=model_dict["min_edge_size"],
            is_single_action=model_dict["is_single_action"],
        )
    elif model_type == "GNNPolicyAgent":
        model = GNNPolicyAgent(
            input_size=model_dict["input_size"],
            nmax=model_dict["nmax"],
            offset_coarse=model_dict["offset_coarse"],
            offset_split=model_dict["offset_split"],
            edge_dim=model_dict["edge_dim"],
            latent_dim=model_dict["latent_dim"],
            num_steps=model_dict["num_steps"],
            layer_norm=model_dict["layer_norm"],
            is_single_action=model_dict["is_single_action"],
            batch_norm=model_dict["batch_norm"],
            edge_attr=model_dict["edge_attr"],
            edge_threshold=model_dict["edge_threshold"],
            skip_split=model_dict["skip_split"],
            skip_flip=model_dict["skip_flip"],
            skip_coarse=model_dict["skip_coarse"],
            dataset=model_dict["dataset"],
            act_name=model_dict["act_name"],
            is_split_test=model_dict["is_split_test"],
            is_flip_test=model_dict["is_flip_test"],
            is_coarsen_test=model_dict["is_coarsen_test"],
            samplemode=model_dict["samplemode"],
            rescale=model_dict["rescale"],
            min_edge_size=model_dict["min_edge_size"],
            top_k_action=model_dict["top_k_action"],
            reward_condition=model_dict["reward_condition"]
        )
    elif model_type == "GNNPolicyAgent_Sampling":
        model = GNNPolicyAgent_Sampling(
            input_size=model_dict["input_size"],
            nmax=model_dict["nmax"],
            edge_dim=model_dict["edge_dim"],
            latent_dim=model_dict["latent_dim"],
            edge_attr=model_dict["edge_attr"],
            layer_norm=model_dict["layer_norm"],
            batch_norm=model_dict["batch_norm"],
            num_steps=model_dict["num_steps"],
            val_layer_norm=model_dict["val_layer_norm"],
            val_batch_norm=model_dict["val_batch_norm"],
            val_num_steps=model_dict["val_num_steps"],
            val_pooling_type=model_dict["val_pooling_type"],
            use_pos=model_dict["use_pos"],
            final_ratio=model_dict["final_ratio"],
            edge_threshold=model_dict["edge_threshold"],
            final_pool=model_dict["final_pool"],
            val_act_name=model_dict["val_act_name"],
            val_act_name_final=model_dict["val_act_name_final"],
            processor_aggr = model_dict["processor_aggr"],

            rl_num_steps=model_dict["rl_num_steps"],
            rl_layer_norm=model_dict["rl_layer_norm"],
            rl_batch_norm=model_dict["rl_batch_norm"],
            skip_split=model_dict["skip_split"],
            skip_flip=model_dict["skip_flip"],
            skip_coarse=model_dict["skip_coarse"],
            act_name=model_dict["act_name"],
            dataset=model_dict["dataset"],
            samplemode=model_dict["samplemode"],

            min_edge_size=model_dict["min_edge_size"],
            rescale=model_dict["rescale"],
            batch_size=model_dict["batch_size"],
            is_single_action=model_dict["is_single_action"],

            top_k_action=model_dict["top_k_action"],
            max_action=model_dict["max_action"],
            reward_condition=model_dict["reward_condition"],
            offset_split=model_dict["offset_split"],

            offset_coarse=model_dict["offset_coarse"],
            kaction_pooling_type=model_dict["kaction_pooling_type"],

            share_processor=model_dict["share_processor"]
        )
    elif model_type == "FNOModel":
        model = FNOModel(
            input_size=model_dict["input_size"],
            output_size=model_dict["output_size"],
            input_shape=model_dict["input_shape"],
            modes=model_dict["modes"],
            width=model_dict["width"],
            loss_type=model_dict["loss_type"],
            temporal_bundle_steps=model_dict["temporal_bundle_steps"] if "temporal_bundle_steps" in model_dict else 1,
            static_encoder_type=model_dict["static_encoder_type"] if "static_encoder_type" in model_dict else "None",
            static_latent_size=model_dict["static_latent_size"] if "static_latent_size" in model_dict else 0,
        )
    else:
        raise Exception("model type {} is not supported!".format(model_type)) 

    if model_type not in ["Actor_Critic"]:
        model.load_state_dict(model_dict["state_dict"])
    model.to(device)
    return model


# ### Get model:

# In[ ]:


def get_model(
    args,
    data_eg,
    device,
):
    """Get model as specified by args."""
    if len(dict(data_eg.original_shape)["n0"]) != 0:
        original_shape = to_tuple_shape(data_eg.original_shape)
        pos_dims = get_pos_dims_dict(original_shape)
    grid_keys = to_tuple_shape(data_eg.grid_keys)
    part_keys = to_tuple_shape(data_eg.part_keys)
    dyn_dims = dict(to_tuple_shape(data_eg.dyn_dims))
    static_input_size = {"n0": 0}
    if "node_feature" in data_eg:
        # The data object contains the actual data:
        output_size = {key: data_eg.node_label[key].shape[-1] + 1 if key in part_keys else data_eg.node_label[key].shape[-1] for key in data_eg.node_label}
        input_size = {key: np.prod(data_eg.node_feature[key].shape[-2:]) for key in data_eg.node_feature}
        if args.static_encoder_type != "None":
            if args.static_encoder_type.startswith("param"):
                static_input_size = {key: data_eg.param[key].shape[-1] for key in input_size}
            else:
                static_input_size = {key: input_size[key]-dyn_dims[key] for key in input_size}
    else:
        # The data object contains necessary information for JIT loading:
        static_dims = dict(to_tuple_shape(data_eg.static_dims))
        compute_func = dict(to_tuple_shape(data_eg.compute_func))
        input_size = {key: compute_func[key][0] + static_dims[key] + dyn_dims[key] for key in dyn_dims}
        output_size = deepcopy(dyn_dims)
        if args.static_encoder_type != "None":
            static_input_size = static_dims
    for key in data_eg.grid_keys:
        if args.use_grads:
            input_size[key] += dyn_dims[key] * pos_dims[key]
        # if args.use_pos:
        #     input_size[key] += pos_dims[key]
    if args.algo.startswith("fno"):
        """fno-w20-m12: fno with width=20 and modes=12. Default"""
        algo_dict = {}
        for ele in args.algo.split("-")[1:]:
            algo_dict[ele[0]] = int(ele[1:])
        if "w" not in algo_dict:
            algo_dict["w"] = None
        if "m" not in algo_dict:
            algo_dict["m"] = None
        model = FNOModel(
            input_size=input_size["n0"],
            output_size=output_size["n0"] * args.temporal_bundle_steps,
            input_shape=dict(original_shape)["n0"],
            modes=algo_dict["m"],
            width=algo_dict["w"],
            loss_type=args.loss_type,
            temporal_bundle_steps=args.temporal_bundle_steps,
            static_encoder_type=args.static_encoder_type,
            static_latent_size=args.static_latent_size,
        ).to(device)
    elif args.algo.startswith("Value_Model"):
        if len(dict(data_eg.original_shape)["n0"]) == 1:
            model = Value_Model(
                input_size=input_size["n0"] + (1 if not args.is_1d_periodic else 0) + (1 if args.use_pos else 0) + (3 if args.dataset.startswith("mppde1dh") else 0),  # +1 for x_bdd (boundary nodes), +1 for use_pos, +3 for parameters
                output_size=1,
                edge_dim=1 if args.dataset.startswith("mppde1de") else 2,
                latent_dim=args.value_latent_size,
                num_pool=args.value_num_pool,
                act_name=args.value_act_name,
                act_name_final=args.value_act_name_final,
                layer_norm=args.value_layer_norm,
                batch_norm=args.value_batch_norm,
                num_steps=args.value_num_steps,
                pooling_type=args.value_pooling_type,
                edge_attr=args.edge_attr,
                use_pos=args.use_pos,
                reward_condition=args.reward_condition,
            ).to(device)
        elif len(dict(data_eg.original_shape)["n0"]) == 0:
            model = Value_Model(
                input_size=input_size["n0"], 
                output_size=4,
                edge_dim=4*input_size["n0"],
                latent_dim=args.value_latent_size,
                num_pool=args.value_num_pool,
                act_name=args.value_act_name,
                act_name_final=args.value_act_name_final,
                layer_norm=args.value_layer_norm,
                batch_norm=args.value_batch_norm,
                num_steps=args.value_num_steps,
                pooling_type=args.value_pooling_type,
                edge_attr=args.edge_attr,
                use_pos=args.use_pos,
                reward_condition=args.reward_condition
            ).to(device)
    elif args.algo.startswith("GNNPolicySizing"):
        if len(dict(data_eg.original_shape)["n0"]) == 1:
            model = GNNPolicySizing(
                input_size=input_size["n0"] + (1 if not args.is_1d_periodic else 0) + (1 if args.use_pos else 0) + (3 if args.dataset.startswith("mppde1dh") else 0),  # +1 for x_bdd (boundary nodes), +1 for use_pos, +3 for parameters
                output_size=1 if args.dataset.startswith("mppde1d") else 4,
                sizing_field_dim=1 if args.dataset.startswith("mppde1d") else 4,
                edge_dim=1 if args.dataset.startswith("mppde1de") else 2,
                nmax=torch.tensor(50000),
                latent_dim=args.latent_size,
                num_steps=args.n_layers,
                act_name=args.act_name,
                layer_norm=args.layer_norm,
                dataset=args.dataset,
                batch_norm=args.actor_batch_norm,
                edge_attr=args.edge_attr,
                edge_threshold=args.edge_threshold,
                is_split_test=False,
                is_flip_test=False,
                is_coarsen_test=False,
                batch_size=args.batch_size,
                skip_split=args.skip_split,
                skip_coarse=args.skip_coarse,
                samplemode="random",
                rescale=args.rescale,
                min_edge_size=args.min_edge_size,
                is_single_action=args.is_single_action,
            ).to(device)
        elif len(dict(data_eg.original_shape)["n0"]) == 0:
            model = GNNPolicySizing(
                input_size=input_size["n0"],  # +1 for x_bdd (boundary nodes), +1 for use_pos
                output_size=4,
                edge_dim=4*input_size["n0"],
                sizing_field_dim=4,
                nmax=torch.tensor(50000),
                batch_size=args.batch_size,
                latent_dim=args.latent_size,
                num_steps=args.n_layers,
                act_name=args.act_name,
                layer_norm=args.layer_norm,
                batch_norm=args.actor_batch_norm,
                edge_attr=args.edge_attr,
                dataset=args.dataset,
                edge_threshold=args.edge_threshold,
                is_split_test=False,
                is_flip_test=False,
                is_coarsen_test=False,
                skip_split=False,
                skip_flip=False,
                samplemode="random",
                rescale=args.rescale,
                min_edge_size=args.min_edge_size,
                is_single_action=args.is_single_action,
            ).to(device)
    elif args.algo.startswith("GNNPolicyAgent"):
        if len(dict(data_eg.original_shape)["n0"]) == 1:
            model = GNNPolicyAgent(
                input_size=input_size["n0"] + (1 if not args.is_1d_periodic else 0) + (1 if args.use_pos else 0) + (3 if args.dataset.startswith("mppde1dh") else 0),  # +1 for x_bdd (boundary nodes), +1 for use_pos, +3 for parameters
                edge_dim=1 if args.dataset.startswith("mppde1de") else 2,
                dataset=args.dataset,
                output_size=1,
                nmax=torch.tensor(50000),
                latent_dim=args.latent_size,
                num_steps=args.n_layers,
                act_name=args.act_name,
                batch_size=args.batch_size,
                layer_norm=args.layer_norm,
                batch_norm=args.actor_batch_norm,
                edge_attr=args.edge_attr,
                edge_threshold=args.edge_threshold,
                is_split_test=False,
                is_flip_test=False,
                is_coarsen_test=False,
                skip_split=args.skip_split,
                skip_coarse=args.skip_coarse,
                samplemode="random",
                rescale=args.rescale,
                offset_coarse=args.offset_coarse,
                offset_split=args.offset_split,
                min_edge_size=args.min_edge_size,
                is_single_action=args.is_single_action,
                top_k_action=args.top_k_action,
                reward_condition=args.reward_condition
            ).to(device)
        elif len(dict(data_eg.original_shape)["n0"]) == 0:
            model = GNNPolicyAgent(
                output_size=1,
                dataset=args.dataset,
                input_size=input_size["n0"],  # +1 for x_bdd (boundary nodes), +1 for use_pos
                edge_dim=4*input_size["n0"],
                sizing_field_dim=4,
                nmax=torch.tensor(50000),
                latent_dim=args.latent_size,
                batch_size=args.batch_size,
                num_steps=args.n_layers,
                offset_coarse=args.offset_coarse,
                offset_split=args.offset_split,
                act_name=args.act_name,
                layer_norm=args.layer_norm,
                batch_norm=args.actor_batch_norm,
                edge_attr=args.edge_attr,
                edge_threshold=args.edge_threshold,
                is_split_test=False,
                is_flip_test=False,
                is_coarsen_test=False,
                skip_split=False,
                skip_flip=False,
                skip_coarse=False,
                samplemode="random",
                rescale=args.rescale,
                min_edge_size=args.min_edge_size,
                is_single_action=args.is_single_action,
                top_k_action=args.top_k_action,
                reward_condition=args.reward_condition
            ).to(device)
    elif args.algo.startswith("gnnremesher"):
        """
        algo:
            gnnremesher-evolution
            gnnremesher-evolution+reward:16: the gnn has evolution and reward, where the reward's latent dimension is 32
        """
        training_mode = args.algo.split("-")[-1]
        if len(args.algo.split("-")) == 1:
            training_mode = "evolution"
        training_items = training_mode.split("+")
        reward_dim = 0
        for item in training_items:
            if "reward" in item:
                if len(item.split(":")) == 2:
                    reward_dim = int(item.split(":")[1])
                else:
                    reward_dim = args.latent_size
        diffMLP = True if args.algo.startswith("gnnremesher^diff") else False
        if len(dict(data_eg.original_shape)["n0"]) == 1:
            model = GNNRemesher(
                input_size=input_size["n0"]+(1 if not args.is_1d_periodic else 0)+(1 if args.use_pos else 0) + (3 if args.dataset.startswith("mppde1dh") else 0),  # +1 for x_bdd (boundary nodes), +1 for use_pos, +3 for parameters
                output_size=output_size["n0"] * args.temporal_bundle_steps,
                edge_dim=1 if args.dataset.startswith("mppde1de") else 2,
                sizing_field_dim=0,
                reward_dim=reward_dim,
                latent_dim=args.latent_size,
                num_steps=args.n_layers,
                nmax=torch.tensor(50000),
                use_encoder=True,
                act_name=args.act_name,
                diffMLP=diffMLP,
                edge_attr=args.edge_attr,
                is_split_test=False,
                is_flip_test=False,
                is_coarsen_test=False,
                skip_split=False,
                skip_flip=False,
                is_y_diff=args.is_y_diff,
                batch_norm=args.batch_norm,
                layer_norm=args.layer_norm,
            ).to(device)
            assert args.edge_attr == (False if args.dataset.split("-")[0] == "mppde1d" else True)
        elif len(dict(data_eg.original_shape)["n0"]) == 0:
            model = GNNRemesherPolicy(
                input_size=(2*input_size[key]+data_eg.onehot_list["n0"][0].shape[-1] if "onehot_list" in data_eg else input_size["n0"]), 
                output_size=output_size["n0"] * args.temporal_bundle_steps,
                edge_dim=4*output_size["n0"],
                sizing_field_dim=4,
                reward_dim=reward_dim,
                latent_dim=args.latent_size,
                num_steps=args.n_layers,
                nmax=torch.tensor(50000),
                use_encoder=True,
                act_name=args.act_name,
                diffMLP=diffMLP,
                edge_attr=args.edge_attr,
                is_split_test=False,
                is_flip_test=False,
                is_coarsen_test=False,
                skip_split=False,
                skip_flip=False,
                is_y_diff=args.is_y_diff,
                batch_norm=args.batch_norm,
                layer_norm=args.layer_norm,
                edge_threshold=args.edge_threshold,
                noise_amp=args.noise_amp,
                correction_rate=args.correction_rate,
            ).to(device)
        else:
            raise
    elif args.algo.startswith("rlgnnremesher"):
        reward_dim = 0
        gnn_mode_list = args.algo.lower().split("-")[0].split("^")  # e.g. args.algo = rlgnnremesher^diff^sizing
        diffMLP = "diff" in gnn_mode_list
        args_copy = deepcopy(args)
        if "sizing" in gnn_mode_list:
            args_copy.algo = "GNNPolicySizing"
        elif "agent" in gnn_mode_list:
            args_copy.algo = "GNNPolicyAgent"
        else:
            raise
        assert not (len(dict(data_eg.original_shape)["n0"]) == 1) or (args_copy.edge_attr == (False if args_copy.dataset.split("-")[0] == "mppde1d" else True))
        policy = get_model(args_copy, data_eg=data_eg, device=device)
        
        if len(dict(data_eg.original_shape)["n0"]) == 0:
            value_model = Value_Model(
                input_size=input_size["n0"],#(2*input_size[key]+data_eg.onehot_list["n0"][0].shape[-1] if "onehot_list" in data_eg else input_size["n0"]),
                output_size=1,
                edge_dim=4*output_size["n0"],
                latent_dim=args.value_latent_size,
                num_pool=args.value_num_pool,
                act_name=args.value_act_name,
                act_name_final=args.value_act_name_final,
                layer_norm=args.value_layer_norm,
                batch_norm=args.value_batch_norm,
                num_steps=args.value_num_steps,
                pooling_type=args.value_pooling_type,
                edge_attr=args.edge_attr,
                use_pos=args.use_pos,
            )            
        else:
            value_model = Value_Model(
                input_size=input_size["n0"]+(1 if not args.is_1d_periodic else 0)+(1 if args.use_pos else 0) + (3 if args.dataset.startswith("mppde1dh") else 0),  # +1 for x_bdd (boundary nodes), +1 for use_pos, +3 for parameters
                output_size=1,
                edge_dim=1 if args.dataset.startswith("mppde1de") else 2,
                latent_dim=args.value_latent_size,
                num_pool=args.value_num_pool,
                act_name=args.value_act_name,
                act_name_final=args.value_act_name_final,
                layer_norm=args.value_layer_norm,
                batch_norm=args.value_batch_norm,
                num_steps=args.value_num_steps,
                pooling_type=args.value_pooling_type,
                edge_attr=args.edge_attr,
                use_pos=args.use_pos,
            )

        model = Actor_Critic(
            actor=policy,
            critic=value_model,
            horizon=args.rl_horizon,
        ).to(device)

    elif args.algo.startswith("srlgnnremesher"):
        reward_dim = 0
        gnn_mode_list = args.algo.lower().split("-")[0].split("^")  # e.g. args.algo = rlgnnremesher^diff^sizing
        diffMLP = "diff" in gnn_mode_list
        args_copy = deepcopy(args)
        args_copy.algo = "GNNPolicyAgent_Sampling"
        assert not (len(dict(data_eg.original_shape)["n0"]) == 1) or (args_copy.edge_attr == (False if args_copy.dataset.split("-")[0] == "mppde1d" else True))
        
        if len(dict(data_eg.original_shape)["n0"]) == 1:
            print("sampling loading")
            sampler_model = GNNPolicyAgent_Sampling(
                # input_size=input_size["n0"] + 1 + (1 if args.use_pos else 0) + (3 if args.dataset.startswith("mppde1dh") else 0),  # +1 for x_bdd (boundary nodes), +1 for use_pos, +3 for parameters
                input_size=25+(1 if not args.is_1d_periodic else 0),
                edge_dim=1 if args.dataset.startswith("mppde1de") else 2,
                nmax=torch.tensor(50000),
                latent_dim=args.latent_size,
                edge_attr=args.edge_attr,
                layer_norm=args.layer_norm,
                batch_norm=args.batch_norm,
                num_steps=3,
                val_layer_norm=args.value_layer_norm,
                val_batch_norm=args.value_batch_norm,
                val_num_steps=args.value_num_steps,
                val_pooling_type=args.value_pooling_type,
                use_pos=args.use_pos,
                edge_threshold=args.edge_threshold,
                final_pool=args.value_pooling_type,
                val_act_name=args.value_act_name,
                val_act_name_final=args.value_act_name_final,

                rl_num_steps=3,
                rl_layer_norm=args.layer_norm,
                rl_batch_norm=args.batch_norm,
                skip_split=args.skip_split,
                skip_flip=False,
                skip_coarse=args.skip_coarse,
                act_name=args.act_name,
                dataset=args.dataset,
                samplemode="random",

                min_edge_size=args.min_edge_size,
                rescale=args.rescale,
                batch_size=args.batch_size,
                is_single_action=args.is_single_action,

                top_k_action=args.top_k_action,
                max_action=args.max_action,
                reward_condition=args.reward_condition,
                offset_split=args.offset_split,

                offset_coarse=args.offset_coarse,
                kaction_pooling_type=args.kaction_pooling_type,
                processor_aggr=args.processor_aggr,
            ).to(device)
        elif len(dict(data_eg.original_shape)["n0"]) == 0:
            # pdb.set_trace()
            input_size_node= (8 if args.policy_input_feature=="velocity" else 6)
            sampler_model = GNNPolicyAgent_Sampling(
                input_size= input_size_node,  # +1 for x_bdd (boundary nodes), +1 for use_pos
                output_size=1,
                edge_dim=4*input_size["n0"],
                nmax=torch.tensor(50000),
                latent_dim=args.latent_size,
                edge_attr=args.edge_attr,
                layer_norm=args.layer_norm,
                batch_norm=args.batch_norm,
                num_steps=3,
                val_layer_norm=args.value_layer_norm,
                val_batch_norm=args.value_batch_norm,
                val_num_steps=args.value_num_steps,
                val_pooling_type=args.value_pooling_type,
                use_pos=args.use_pos,
                edge_threshold=args.edge_threshold,
                final_pool=args.value_pooling_type,
                val_act_name=args.value_act_name,
                val_act_name_final=args.value_act_name_final,

                rl_num_steps=3,
                rl_layer_norm=args.layer_norm,
                rl_batch_norm=args.batch_norm,
                skip_split=args.skip_split,
                skip_flip=args.skip_flip,
                skip_coarse=args.skip_coarse,
                act_name=args.act_name,
                dataset=args.dataset,
                samplemode="random",

                min_edge_size=args.min_edge_size,
                rescale=args.rescale,
                batch_size=args.batch_size,
                is_single_action=args.is_single_action,

                top_k_action=args.top_k_action,
                max_action=args.max_action,
                reward_condition=args.reward_condition,
                offset_split=args.offset_split,

                offset_coarse=args.offset_coarse,
                kaction_pooling_type=args.kaction_pooling_type,
                share_processor = args.share_processor,
                processor_aggr=args.processor_aggr,
            ).to(device) 
            # pdb.set_trace()
        model = Actor_Critic(
            actor=sampler_model,
            critic=sampler_model.value_model,
            horizon=args.rl_horizon,
        ).to(device)
    else:
        raise Exception("Algo {} is not supported!".format(args.algo))
    return model




# # Training and evaluation:

# ### Test:

# In[ ]:


def build_optimizer(args, params, is_disc=False, is_ebm=False, separate_params=None):
    weight_decay = args.weight_decay
    if (not is_disc) and (not is_ebm):
        lr = args.lr
    elif is_disc:
        # The lr is for discriminator:
        assert not is_ebm
        lr = args.lr if args.disc_lr == -1 else args.disc_lr
    elif is_ebm:
        assert not is_disc
        lr = args.lr if args.ebm_lr == -1 else args.ebm_lr
    else:
        raise
    if separate_params is not None:
        filter_fn = separate_params
    else:
        filter_fn = filter(lambda p: p.requires_grad, params)

    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(filter_fn, lr=lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=lr, weight_decay=weight_decay)

    if args.lr_scheduler_type == "rop":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_scheduler_factor, verbose=True)
    elif args.lr_scheduler_type == "cos":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min_cos if hasattr(args, "lr_min_cos") else 0)
    elif args.lr_scheduler_type == "cos-re":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_scheduler_T0, T_mult=args.lr_scheduler_T_mult)
    elif args.lr_scheduler_type == "None":
        scheduler = None
    elif args.lr_scheduler_type.startswith("steplr"):
        """Example: "steplr-s100-g0.5" means step_size of 100 epochs and gamma decay of 0.5 (multiply 0.5 every 100 epochs)"""
        scheduler_dict = {}
        for item in args.lr_scheduler_type.split("-")[1:]:
            if item[0] == "s":
                scheduler_dict[item[0]] = int(item[1:])
            elif item[0] == "g":
                scheduler_dict[item[0]] = float(item[1:])
            else:
                raise
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_dict["s"], gamma=scheduler_dict["g"])
    else:
        raise
    return optimizer, scheduler


def build_optimizer_special(args, model, lr_special_dict):
    """
    Example of optim_list:
    optim.SGD([
            {'params': model.base.parameters()},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], lr=1e-2, momentum=0.9
    )
    """
    weight_decay = args.weight_decay
    lr = args.lr
    optim_list = []
    for key in list(dict(model.named_modules())):
        if key == "" or "." in key:
            # remove e.g., '', 'module1.module2'
            continue
        elif key in lr_special_dict:
            lr_item = lr_special_dict[key]
        else:
            lr_item = lr
        optim_list.append({"params": getattr(model, key).parameters(), "lr": lr_item})
    if args.opt == 'adam':
        optimizer = optim.Adam(optim_list, lr=lr, weight_decay=weight_decay)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(optim_list, lr=lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(optim_list, lr=lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(optim_list, lr=lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(optim_list, lr=lr, weight_decay=weight_decay)

    if args.lr_scheduler_type == "rop":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_scheduler_factor, verbose=True)
    elif args.lr_scheduler_type == "cos":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min_cos if hasattr(args, "lr_min_cos") else 0)
    elif args.lr_scheduler_type == "cos-re":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_scheduler_T0, T_mult=args.lr_scheduler_T_mult)
    elif args.lr_scheduler_type == "None":
        scheduler = None
    elif args.lr_scheduler_type.startswith("steplr"):
        """Example: "steplr-s100-g0.5" means step_size of 100 epochs and gamma decay of 0.5 (multiply 0.5 every 100 epochs)"""
        scheduler_dict = {}
        for item in args.lr_scheduler_type.split("-")[1:]:
            if item[0] == "s":
                scheduler_dict[item[0]] = int(item[1:])
            elif item[0] == "g":
                scheduler_dict[item[0]] = float(item[1:])
            else:
                raise
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_dict["s"], gamma=scheduler_dict["g"])
    else:
        raise
    return optimizer, scheduler


def test(data_loader, model, device, args, **kwargs):
    model.eval()
    count = 0
    total_loss = 0
    multi_step_dict = parse_multi_step(args.multi_step)
    args = deepcopy(args)
    info = {}
    keys_pop = []
    for key, value in kwargs.items():
        if (isinstance(value, Number) or isinstance(value, str)) and key != "current_epoch" and key != "current_minibatch":
            setattr(args, key, value)
            keys_pop.append(key)
    for key in keys_pop:
        kwargs.pop(key)
    # Compute loss:
    for data in tqdm(data_loader):
        with torch.no_grad():
            batch_size = get_batch_size(data)
            data = data.to(device)
            if is_diagnose(loc="test:1", filename=args.filename):
                pdb.set_trace()
            args = deepcopy(args)
            args.zero_weight = 1.0
            loss = model.get_loss(data, args, is_rollout=True, **kwargs).item()
            keys, values = get_keys_values(model.info, exclude=["pred"])
            record_data(info, values, keys)
            total_loss = total_loss + loss
            count += 1
    for key, item in info.items():
        info[key] = np.mean(item)
    if count == 0:
        return None, info
    else:
        return total_loss / count, info


    
def unittest_model(model, data, args, device, use_grads=True, use_pos=False, is_mesh=False, test_cases="all", algo="contrastive", **kwargs):
    """Test if the loaded model is exactly the same as the original model."""
    if test_cases == "all":
        test_cases = ["model_dict", "pred", "loss"]
    if not isinstance(test_cases, list):
        test_cases = [test_cases]

    def dataparallel_compat(model_dict):
        model_dict_copy = {}
        for (k, v) in model_dict.items():
            model_dict_copy[k.replace("module.", "")] = v
        return model_dict_copy

    with torch.no_grad():
        data = deepcopy(data).to(device)
        model.eval()
        multi_gpu=len(args.gpuid.split(",")) > 1
        #pdb.set_trace()
        model2 = load_model(get_model_dict(model), device, multi_gpu=multi_gpu)
        # model2.type(list(model.parameters())[0].dtype)
        model2.eval()

        if "model_dict" in test_cases:
            model_dict = get_model_dict(model)
            model_dict2 = get_model_dict(model2)
            diff_sum = 0
            # All other key: values must match:
            # pdb.set_trace()
            check_same_model_dict(model_dict, model_dict2)
            # The state_dict must match:
            if model_dict["type"] not in ["Actor_Critic"]:
                for k, v in model_dict["state_dict"].items():
                    v2 = model_dict2["state_dict"][k]
                    diff_sum += (v - v2).abs().max()
                assert diff_sum == 0, "The model_dict of the loaded model is not the same as the original model!"

        if "pred" in test_cases and model_dict["type"] not in ["Actor_Critic"]:
            # Evaluate the difference for three times:
            #pdb.set_trace()
            if is_mesh:
                data_c = deepcopy(data)
                pred, _ = model.interpolation_hist_forward(data_c, use_grads=use_grads, use_pos=use_pos)
                data_c = deepcopy(data)
                pred2, _ = model2.interpolation_hist_forward(data_c, use_grads=use_grads, use_pos=use_pos)
                pred["n0"] = pred["n0"][0]
                pred2["n0"] = pred2["n0"][0]
            else:
                data_c = deepcopy(data)
                pred, _ = model(data_c, use_grads=use_grads, use_pos=use_pos)
                data_c = deepcopy(data)
                pred2, _ = model2(data_c, use_grads=use_grads, use_pos=use_pos)

            max_diff = 0.0
            #pdb.set_trace()
            try:
                max_diff = max([(pred[key] - pred2[key]).abs().max().item() for key in pred])
            except:
                max_diff = max([(pred.node_feature[nt] - pred2.node_feature[nt]).abs().max().item() for nt in pred.node_feature])

            if is_mesh:
                data_c = deepcopy(data)
                pred_2, _ = model.interpolation_hist_forward(data_c, use_grads=use_grads, use_pos=use_pos)
                data_c = deepcopy(data)
                pred2_2, _ = model2.interpolation_hist_forward(data_c, use_grads=use_grads, use_pos=use_pos)
                pred_2["n0"] = pred_2["n0"][0]
                pred2_2["n0"] = pred2_2["n0"][0]
            else:
                data_c = deepcopy(data)
                pred_2, _ = model(data_c, use_grads=use_grads, use_pos=use_pos)
                data_c = deepcopy(data)
                pred2_2, _ = model2(data_c, use_grads=use_grads, use_pos=use_pos)

            max_diff_2 = 0.0
            try:
                max_diff_2 = max([(pred_2[key] - pred2_2[key]).abs().max().item() for key in pred])
            except:
                max_diff_2 = max([(pred_2.node_feature[nt] - pred2_2.node_feature[nt]).abs().max().item() for nt in pred.node_feature])

            if is_mesh:
                if max_diff < 9e-2 and max_diff_2 < 9e-2:
                    print("\nThe maximum difference between the predictions are {:.4e} and {:.4e}, within error tolerance.\n".format(max_diff, max_diff_2))
                else:
                    raise Exception("\nThe loaded model for 2d mesh is not exactly the same as the original model! The maximum difference between the predictions are {:.4e} and {:.4e}.\n".format(max_diff, max_diff_2))                
            else:
                if max_diff < 8e-5 and max_diff_2 < 8e-5:
                    print("\nThe maximum difference between the predictions for 2d meshes are {:.4e} and {:.4e}, within error tolerance.\n".format(max_diff, max_diff_2))
                else:
                    raise Exception("\nThe loaded model is not exactly the same as the original model! The maximum difference between the predictions are {:.4e} and {:.4e}.\n".format(max_diff, max_diff_2))

        if "one_step_interpolation" in test_cases:
            data_c = deepcopy(data)
            set_seed(42)
            pred, info, _ = model.one_step_interpolation_forward(data_c, 0)
            data_cc = deepcopy(data)
            set_seed(42)
            pred2, info2, _ = model.one_step_interpolation_forward(data_cc, 0)
            max_diff = 0.0
            max_diff = max([(pred["n0"][0] - pred2["n0"][0]).abs().max().item() for key in pred])
            if max_diff < 8e-5:
                print("\nThe output shape:{}".format(pred["n0"][0].shape))
                print("\nThe maximum difference between the predictions are {:.4e}, within error tolerance.\n".format(max_diff))
            else:
                raise Exception("\nThe loaded model is not exactly the same as the original model! The maximum difference between the predictions are {:.4e}.\n".format(max_diff))     

                    
        if "value_model" in test_cases:
            data_c = deepcopy(data)
            set_seed(42)
            pred = model(data_c, use_grads=use_grads, use_pos=use_pos)
            data_c = deepcopy(data)
            set_seed(42)
            pred2 = model2(data_c, use_grads=use_grads, use_pos=use_pos)

            max_diff = 0.0
            max_diff = max([(pred - pred2).abs().max().item() for key in pred])
            if max_diff < 8e-5:
                print("\nThe output shape:{}".format(pred.shape))
                print("\nThe maximum difference between the predictions are {:.4e}, within error tolerance.\n".format(max_diff))
            else:
                raise Exception("\nThe loaded model is not exactly the same as the original model! The maximum difference between the predictions are {:.4e}.\n".format(max_diff))     
            set_seed(args.seed)

        if "gnnpolicysizing" in test_cases:
            
            data_c = deepcopy(data)
            set_seed(42)
            pred,prob,entropy,dict1 = model.remeshing_forward_GNN(data_c, use_grads=use_grads, use_pos=use_pos)
            data_c = deepcopy(data)
            set_seed(42)
            pred2,prob2,entropy2,dict2 = model2.remeshing_forward_GNN(data_c, use_grads=use_grads, use_pos=use_pos)

            max_diff = 0.0
            max_diff = max([(prob[key] - prob2[key]).abs().max().item() for key in prob.keys()])
            print("max_diff, prob",max_diff)
            max_diff +=  max([(entropy[key] - entropy2[key]).abs().max().item() for key in prob.keys()])
            print("max_diff+entropy, prob",max_diff)
            if max_diff < 5e-5:
                print("\nThe maximum difference between the predictions are {:.4e}, within error tolerance.\n".format(max_diff))
            else:
                raise Exception("\nThe loaded model is not exactly the same as the original model! The maximum difference between the predictions are {:.4e}.\n".format(max_diff))  
            set_seed(args.seed)
                
        # Get difference in loss components:
        if "loss" in test_cases:
            # kwargs2 = deepcopy(kwargs)
            kwargs2 = kwargs
            if "discriminator" in kwargs:
                kwargs2["discriminator"] = load_model(get_model_dict(kwargs["discriminator"]), device)
            if "ebm" in kwargs:
                kwargs2["ebm"] = load_model(get_model_dict(kwargs["ebm"]), device)
            set_seed(42)
            # model.type(torch.float64)
            # kwargs["evolution_model"].type(torch.float64)
            loss1 = model.get_loss(
                deepcopy(data), args,
                current_epoch=0,
                current_minibatch=0,
                **kwargs
            )
            info1 = deepcopy(model.info)
            set_seed(42)
            # model2.type(torch.float64)
            # kwargs2["evolution_model"].type(torch.float64)
            loss2 = model2.get_loss(
                deepcopy(data), args,
                current_epoch=0,
                current_minibatch=0,
                **kwargs2
            )
            info2 = deepcopy(model2.info)
            for key in info1:
                if key not in [
                    "loss_reward", "t_evolve", "t_evolve_alt", "t_evolve_interp_alt",
                    "loss_value", "loss_actor", "loss_reinforce",
                    "r_timediff", "r_timediff_alt", "r_timediff_interp_alt", "interp_r_timediff",
                    "v_timediff", "v_timediff_alt", "v_timediff_interp_alt", "interp_v_timediff",
                    "v_beta", "interp_v_beta", "v/timediff", "r/timediff",
                ]:
                    print("{} \t{:.4e}\t{:.4e}\tdiff: {:.4e}".format("{}:".format(key).ljust(16), info1[key], info2[key], abs(info1[key] - info2[key])))
                    if is_mesh:
                        if abs(info1[key] - info2[key]) > 1e-3:
                            raise Exception("{} for the loaded model differs by {:.4e}, which is more than 8e-5.".format(key, abs(info1[key] - info2[key])))
                    else:
                        if abs(info1[key] - info2[key]) > 8e-6:
                            if args.uncertainty_mode == "None":
                                raise Exception("{} for the loaded model differs by {:.4e}, which is more than 8e-6.".format(key, abs(info1[key] - info2[key])))
                            elif len(args.uncertainty_mode.split("^")) > 1 and args.uncertainty_mode.split("^")[1] == "samplefull" and key in ["loss_pred"]:
                                continue
                            else:
                                rel_error = abs(info1[key] - info2[key]) / abs(info1[key] + info2[key]) * 2
                                if rel_error > 1e-5:
                                    raise Exception("{} for the loaded model differs by a relative of {:.4e}, which is more than 8e-6.".format(key, rel_error))
            set_seed(args.seed)
        print(get_time(), "Unittest passed for test cases {}!".format(test_cases))