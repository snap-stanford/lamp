import argparse
from collections import OrderedDict
import datetime
import gc
import matplotlib
import matplotlib.pylab as plt
from numbers import Number
import numpy as np
import pandas as pd
pd.options.display.max_rows = 1500
pd.options.display.max_columns = 200
pd.options.display.width = 1000
pd.set_option('max_colwidth', 400)
import pdb
import pickle
import pprint as pp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from deepsnap.batch import Batch as deepsnap_Batch
import xarray as xr

import sys, os
from argparser import arg_parse
from lamp.datasets.load_dataset import load_data
from lamp.gnns import GNNRemesher, Value_Model_Summation, Value_Model, GNNRemesherPolicy, GNNPolicySizing, GNNPolicyAgent, get_reward_batch, get_data_dropout, GNNPolicyAgent_Sampling
from lamp.models import get_model, load_model, unittest_model, build_optimizer, test

from lamp.pytorch_net.util import Attr_Dict, Batch, filter_filename, pload, pdump, Printer, get_time, init_args, update_args, clip_grad, set_seed, update_dict, filter_kwargs, plot_vectors, plot_matrices, make_dir, get_pdict, to_np_array, record_data, make_dir, str2bool, get_filename_short, print_banner, get_num_params, ddeepcopy as deepcopy, write_to_config
from lamp.utils import p, update_legacy_default_hyperparam, EXP_PATH, seed_everything

def plot_fig(info, data, index=20):
        vers_gt = data.reind_yfeatures["n0"][index].detach().cpu().numpy()
        faces_gt = data.yface_list["n0"][index]

        faces_gt = np.stack(faces_gt)
        batch_0_idx = np.where(vers_gt[faces_gt[:,0],0]<1.5)
        faces_gt = np.stack(faces_gt)[batch_0_idx]

        batch_0_idx = np.where(vers_gt[:,0]<1.5)
        vers_gt = vers_gt[batch_0_idx][:,:]

        plot0 = info['state_preds'][0]
        plot20 = info['state_preds'][index]
        plot20alt_gt_evl = info['state_preds_alt_gt_evl'][index]
        plot20alt_gt_mesh_gt_evl = info['state_preds_alt_gt_mesh_gt_evl'][index]   
        plot20alt_008_gt_evl = info['state_preds_alt_008_gt_evl'][index]
        plot20heuristic_gt_evl = info['state_preds_heuristic_gt_evl'][index]


        from matplotlib.backends.backend_pdf import PdfPages
        fig = plt.figure(figsize=(30,6))
        n = 2
        ax0= fig.add_subplot(n,7,1,projection='3d')
        ax0.set_axis_off()
        ax0.set_xlim([-0.6, 0.6])
        ax0.set_ylim([-0.6, 0.6])
        ax0.set_zlim([-0.5, 0.1])
        ax0.view_init(30, 10)
        ax0.plot_trisurf(plot0['history'][-1][:,3].detach().cpu().numpy(),plot0['history'][-1][:,4].detach().cpu().numpy(), plot0['xfaces'].detach().cpu().numpy().T, plot0['history'][-1][:,5].detach().cpu().numpy(), shade=True, linewidth = 1., edgecolor = 'black', color=(9/255,237/255,249/255,1))  
        plt.title("plot0")

        ax0= fig.add_subplot(n,7,2,projection='3d')
        ax0.set_axis_off()
        ax0.set_xlim([-0.6, 0.6])
        ax0.set_ylim([-0.6, 0.6])
        ax0.set_zlim([-0.5, 0.1])
        ax0.view_init(30, 10)
        ax0.plot_trisurf(plot20alt_gt_evl['history'][-1][:,3].detach().cpu().numpy(),plot20alt_gt_evl['history'][-1][:,4].detach().cpu().numpy(), plot20alt_gt_evl['xfaces'].detach().cpu().numpy().T, plot20alt_gt_evl['history'][-1][:,5].detach().cpu().numpy(), shade=True, linewidth = 1., edgecolor = 'black', color=(9/255,237/255,249/255,1))  
        plt.title("plot20alt_gt_evl")

        ax0= fig.add_subplot(n,7,3,projection='3d')
        ax0.set_axis_off()
        ax0.set_xlim([-0.6, 0.6])
        ax0.set_ylim([-0.6, 0.6])
        ax0.set_zlim([-0.5, 0.1])
        ax0.view_init(30, 10)
        ax0.plot_trisurf(plot20alt_gt_mesh_gt_evl['history'][-1][:,3].detach().cpu().numpy(),plot20alt_gt_mesh_gt_evl['history'][-1][:,4].detach().cpu().numpy(), plot20alt_gt_mesh_gt_evl['xfaces'].detach().cpu().numpy().T, plot20alt_gt_mesh_gt_evl['history'][-1][:,5].detach().cpu().numpy(), shade=True, linewidth = 1., edgecolor = 'black', color=(9/255,237/255,249/255,1))  
        plt.title("plot20alt_gt_mesh_gt_evl")

        ax0= fig.add_subplot(n,7,4,projection='3d')
        ax0.set_axis_off()
        ax0.set_xlim([-0.6, 0.6])
        ax0.set_ylim([-0.6, 0.6])
        ax0.set_zlim([-0.5, 0.1])
        ax0.view_init(30, 10)
        ax0.plot_trisurf(plot20alt_008_gt_evl['history'][-1][:,3].detach().cpu().numpy(),plot20alt_008_gt_evl['history'][-1][:,4].detach().cpu().numpy(), plot20alt_008_gt_evl['xfaces'].detach().cpu().numpy().T, plot20alt_008_gt_evl['history'][-1][:,5].detach().cpu().numpy(), shade=True, linewidth = 1., edgecolor = 'black', color=(9/255,237/255,249/255,1))  
        plt.title("plot20alt_008_gt_evl")

        ax0= fig.add_subplot(n,7,5,projection='3d')
        ax0.set_axis_off()
        ax0.set_xlim([-0.6, 0.6])
        ax0.set_ylim([-0.6, 0.6])
        ax0.set_zlim([-0.5, 0.1])
        ax0.view_init(30, 10)
        ax0.plot_trisurf(plot20heuristic_gt_evl['history'][-1][:,3].detach().cpu().numpy(),plot20heuristic_gt_evl['history'][-1][:,4].detach().cpu().numpy(), plot20heuristic_gt_evl['xfaces'].detach().cpu().numpy().T, plot20heuristic_gt_evl['history'][-1][:,5].detach().cpu().numpy(), shade=True, linewidth = 1., edgecolor = 'black', color=(9/255,237/255,249/255,1))  
        plt.title("plot20heuristic_gt_evl")
        

        ax0= fig.add_subplot(n,7,6,projection='3d')
        ax0.set_axis_off()
        ax0.set_xlim([-0.6, 0.6])
        ax0.set_ylim([-0.6, 0.6])
        ax0.set_zlim([-0.5, 0.1])
        ax0.view_init(30, 10)
        ax0.plot_trisurf(plot20['history'][-1][:,3].detach().cpu().numpy(),plot20['history'][-1][:,4].detach().cpu().numpy(), plot20['xfaces'].detach().cpu().numpy().T, plot20['history'][-1][:,5].detach().cpu().numpy(), shade=True, linewidth = 1., edgecolor = 'black', color=(9/255,237/255,249/255,1))  
        plt.title("plot20_policy")
        
          
        ax0= fig.add_subplot(n,7,7,projection='3d')
        ax0.set_axis_off()
        ax0.set_xlim([-0.6, 0.6])
        ax0.set_ylim([-0.6, 0.6])
        ax0.set_zlim([-0.5, 0.1])
        ax0.view_init(30, 10)
        ax0.plot_trisurf(vers_gt[:,3],vers_gt[:,4],faces_gt, vers_gt[:,5], shade=True, linewidth = 1., edgecolor = 'black', color=(9/255,237/255,249/255,1))  
        plt.title("plot20gt")



        ax0= fig.add_subplot(n,7,7+1)
        ax0.set_axis_off()
        ax0.triplot(plot0['history'][-1][:,0].detach().cpu().numpy(),plot0['history'][-1][:,1].detach().cpu().numpy(), plot0['xfaces'].detach().cpu().numpy().T)  
        plt.title("plot0")

        ax0= fig.add_subplot(n,7,7+2)
        ax0.set_axis_off()
        ax0.triplot(plot20alt_gt_evl['history'][-1][:,0].detach().cpu().numpy(),plot20alt_gt_evl['history'][-1][:,1].detach().cpu().numpy(), plot20alt_gt_evl['xfaces'].detach().cpu().numpy().T,)  
        plt.title("plot20alt_gt_evl")


        ax0= fig.add_subplot(n,7,7+3)
        ax0.set_axis_off()
        ax0.triplot(plot20alt_gt_mesh_gt_evl['history'][-1][:,0].detach().cpu().numpy(),plot20alt_gt_mesh_gt_evl['history'][-1][:,1].detach().cpu().numpy(), plot20alt_gt_mesh_gt_evl['xfaces'].detach().cpu().numpy().T,)  
        plt.title("plot20alt_gt_mesh_gt_evl")

        ax0= fig.add_subplot(n,7,7+4)
        ax0.set_axis_off()
        ax0.triplot(plot20alt_008_gt_evl['history'][-1][:,0].detach().cpu().numpy(),plot20alt_008_gt_evl['history'][-1][:,1].detach().cpu().numpy(), plot20alt_008_gt_evl['xfaces'].detach().cpu().numpy().T,)  
        plt.title("plot20alt_008_gt_evl")

        ax0= fig.add_subplot(n,7,7+5)
        ax0.set_axis_off()
        ax0.triplot(plot20heuristic_gt_evl['history'][-1][:,0].detach().cpu().numpy(),plot20heuristic_gt_evl['history'][-1][:,1].detach().cpu().numpy(), plot20heuristic_gt_evl['xfaces'].detach().cpu().numpy().T,)  
        plt.title("plot20heuristic_gt_evl")
        

        ax0= fig.add_subplot(n,7,7+6)
        ax0.set_axis_off()
        ax0.triplot(plot20['history'][-1][:,0].detach().cpu().numpy(),plot20['history'][-1][:,1].detach().cpu().numpy(), plot20['xfaces'].detach().cpu().numpy().T,)  
        plt.title("plot20_policy")
        
          
        ax0= fig.add_subplot(n,7,7+7)
        ax0.set_axis_off()
        ax0.triplot(vers_gt[:,0],vers_gt[:,1],faces_gt)  
        plt.title("plot20gt")
        


# Analysis:
def get_results_2d(
    all_hash,
    mode="best",
    exclude_idx=(None,),
    dirname=None,
    suffix="",device=None
):
    """
    Perform analysis on the 2D cloth' benchmark.

    Args:
        all_hash: a list of hashes which indicates the experiments to load for analysis
        mode: choose from "best" (load the best model with lowest validation loss) or an integer, 
            e.g. -1 (last saved model), -2 (second last saved model)
        dirname: if not None, will use the dirnaem provided. E.g. tailin-1d_2022-7-27
        suffix: suffix for saving the analysis result.
    """
    
    isplot = True
    df_dict_list = []
    dirname_start = "2d_rl_reproduce_2023-02-26" if dirname is None else dirname
    for hash_str in all_hash:
        df_dict = {}
        df_dict["hash"] = hash_str
        is_found = False
        for dirname_core in [
             dirname_start,
            ]:
            filename = filter_filename(EXP_PATH + dirname_core, include=hash_str)
            if len(filename) == 1:
                is_found = True
                break
        if not is_found:
            print(f"hash {hash_str} does not exist in {dirname}! Please pass in the correct dirname.")
            continue
        dirname = EXP_PATH + dirname_core
        if not dirname.endswith("/"):
            dirname += "/"
        
        try:
            data_record = pload(dirname + filename[0])
        except Exception as e:
            print(f"error {e} in hash_str {hash_str}")
            continue

        args = init_args(update_legacy_default_hyperparam(data_record["args"]))
        args.filename = filename
        if not("processor_aggr" in data_record["model_dict"][-1]["actor_model_dict"].keys()):
            data_record["model_dict"][-1]["actor_model_dict"]["processor_aggr"] = "max"
            if "best_model_dict" in data_record.keys():
                data_record["best_model_dict"]["actor_model_dict"]["processor_aggr"] = "max"
        model = load_model(data_record["best_model_dict"], device=device)
        evolution = load_model(data_record["best_evolution_model_dict"], device=device)
        print("Load the model with best validation loss.")
        model.eval()
        evolution.eval()

        # Load test dataset:
        args_test = deepcopy(args)
        args_test.dataset="arcsimmesh_square_annotated_coarse_minlen008_interp_500"
        args_test.multi_step = "1^20"
        args_test.n_train = "-1"
        args_test.is_train=False
        args_test.use_fineres_data=False
        args_test.show_missing_files = False
        args_test.input_steps=2
        args_test.time_interval=2
        args_test.val_batch_size = 1
        args_test.device = device
        args.device = device
        (train_dataset, test_dataset), (trian_loader, _, test_loader) = load_data(args_test)
        print(len(test_loader))
        
        args_test_gt_008 = deepcopy(args)
        args_test_gt_008.dataset="arcsimmesh_square_annotated_coarse_minlen008_interp_500_gt_500"
        args_test_gt_008.n_train = "-1"
        args_test_gt_008.multi_step = "1^20"
        args_test_gt_008.is_train=False
        args_test_gt_008.use_fineres_data=False
        args_test_gt_008.show_missing_files = False
        args_test_gt_008.input_steps=2
        args_test_gt_008.time_interval=2
        args_test_gt_008.val_batch_size = 1
        args_test_gt_008.device = device
        args.device = device
        (_, _), (_, _, test_loader_008) = load_data(args_test_gt_008)
        print(len(test_loader_008))

        args.noise_amp = 0
    return model, evolution, test_loader, test_loader_008, args

def get_eval(model, test_loader, test_loader_008,evolution_model,beta=0,break_index=2,hashva=None,name=None,best_evolution_model=None,args=None):
    dicts = {}
    min_evolution_loss_rmse_average = 1000
    if True:
        evolution_loss_mse = 0
        evolution_loss_alt_gt_evl_mse = 0
        evolution_loss_alt_gt_mesh_gt_evl_mse = 0
        evolution_loss_alt_008_gt_evl_mse = 0
        evolution_loss_heuristic_gt_evl_mse = 0


        r_statediff = 0
        v_statediff = 0
        rewards = 0
        count = 0
        actual_count = 0
        state_size = 0
        state_size_remeshed = 0
        args.pred_steps=20
        
        list_evolution_loss_mse = []
        list_evolution_loss_mse_alt_gt_evl = []
        list_evolution_loss_mse_alt_gt_mesh_gt_evl = []
        list_evolution_loss_mse_alt_008_gt_evl = []
        list_evolution_loss_heuristic_gt_evl = []

        list_state_size_remeshed = []
        list_all_nodes = []
        list_all_nodes_remesh = []
        kwargs = {}
        kwargs["evolution_model_alt"] = best_evolution_model

        if not(os.path.exists("./results/LAMP_2d")):
                        os.mkdir("./results/LAMP_2d") 
        with torch.no_grad():
            for j, (data,data008) in enumerate(zip(test_loader,test_loader_008)):
                # if (j%91==0 and j>0) or j==0:
                if data.time_step in [10,30,50]:
                    count += 1
                    actual_count +=1
                    if data.__class__.__name__ == "Attr_Dict":
                        data = data.to(args.device)
                        data008 = data008.to(args.device)
                    else:
                        data.to(args.device)
                        data008.to(args.device)
                    info, data_clone = model.get_loss(
                        data,
                        args,
                        wandb=None,
                        opt_evl=False,
                        step_num=0,
                        mode="test_gt_remesh_heursitc",
                        beta=beta,
                        is_gc_collect=False,
                        evolution_model=evolution_model,
                        data008 = data008,
                        **kwargs
                    )
                    state_size_elm = 0
                    state_size_elm_list = []
                    for elem in data_clone.reind_yfeatures["n0"]:
                        state_size_elm += elem.shape[0]
                        state_size_elm_list.append(elem.shape[0])
                    evolution_loss_mse += (info['evolution/loss_mse'].sum().item())
                    state_size += state_size_elm
                    evolution_loss_alt_gt_evl_mse += (info['evolution/loss_alt_gt_evl_mse'].sum().item())
                    evolution_loss_alt_gt_mesh_gt_evl_mse += (info['evolution/loss_alt_gt_mesh_gt_evl_mse'].sum().item())
                    evolution_loss_alt_008_gt_evl_mse += (info['evolution/loss_alt_008_gt_evl_mse'].sum().item())
                    evolution_loss_heuristic_gt_evl_mse += (info['evolution/loss_heuristic_gt_evl_mse'].sum().item())


                    state_size_remeshed += info['v/state_size'].sum().item()
                    
                    list_evolution_loss_mse.append(info['evolution/loss_mse'])
                    list_evolution_loss_mse_alt_gt_evl.append(info['evolution/loss_alt_gt_evl_mse'])
                    list_evolution_loss_mse_alt_gt_mesh_gt_evl.append(info['evolution/loss_alt_gt_mesh_gt_evl_mse'])
                    list_evolution_loss_mse_alt_008_gt_evl.append(info['evolution/loss_alt_008_gt_evl_mse'])
                    list_evolution_loss_heuristic_gt_evl.append(info['evolution/loss_heuristic_gt_evl_mse'])

                    list_all_nodes.append(state_size_elm_list)
                    list_all_nodes_remesh.append(info['v/state_size'])
                    
                    print(info['evolution/loss_mse'].shape[0])
                    print("index",actual_count,"loss", (info['evolution/loss_mse'].sum().item()/state_size_elm),"Gt state_size_elm",state_size_elm,"remeshed_state_size",info['v/state_size'].sum().item())       
                    print((info['v/state_size'].reshape(-1))[::4])
                    print("current running rms alt gt evl:",np.sqrt(evolution_loss_alt_gt_evl_mse/state_size))
                    print("current running rms alt gt mesh gt evl :",np.sqrt(evolution_loss_alt_gt_mesh_gt_evl_mse/state_size))
                    print("current running rms alt 008 gt evl:",np.sqrt(evolution_loss_alt_008_gt_evl_mse/state_size))
                    print("current running rms heursitc gt evl:",np.sqrt(evolution_loss_heuristic_gt_evl_mse/state_size))
                    print("current running rms:",np.sqrt(evolution_loss_mse/state_size))

                    
                    
                    if not(os.path.exists("./results/LAMP_2d/{}".format(name))):
                        os.mkdir("./results/LAMP_2d/{}".format(name)) 
                    if not(os.path.exists("./results/LAMP_2d/{}/{}".format(name,count))):
                        os.mkdir("./results/LAMP_2d/{}/{}".format(name,count)) 
                    if not(os.path.exists("./results/LAMP_2d/{}/{}/{}".format(name,count,data.time_step))):
                        os.mkdir("./results/LAMP_2d/{}/{}/{}".format(name,count,data.time_step)) 
                    np.save("./results/LAMP_2d/{}/{}/{}/{}_{}.npy".format(name,count,data.time_step,name,actual_count),{"loss_mse":info['evolution/loss_mse'].detach().cpu().numpy(),"loss_alt_gt_evl_mse":info['evolution/loss_alt_gt_evl_mse'].detach().cpu().numpy,"loss_alt_gt_mesh_gt_evl_mse":info['evolution/loss_alt_gt_mesh_gt_evl_mse'].detach().cpu().numpy(),"loss_alt_008_gt_evl_mse":info['evolution/loss_alt_008_gt_evl_mse'].detach().cpu().numpy(),"loss_heuristic_gt_evl_mse":info['evolution/loss_heuristic_gt_evl_mse'].detach().cpu().numpy(),"state_size_elm_list":state_size_elm_list,"state_size":info['v/state_size'].detach().cpu().numpy()})
                    np.save("./results/LAMP_2d/{}/{}/{}/{}_{}_all.npy".format(name,count,data.time_step,name,actual_count),{"info":info})
                    for index in range(5,20,2):
                        plot_fig(info, data_clone,index=index)
                        plt.savefig("./results/LAMP_2d/{}/{}/{}/{}_{}_{}.png".format(name,count,data.time_step,name,actual_count,index))
                        plt.close()
                    torch.cuda.empty_cache()
                    with open('./results/LAMP_2d/{}/summary.txt'.format(name), 'a') as f:
                        f.write("current running rms alt gt evl: {}\n".format(np.sqrt(evolution_loss_alt_gt_evl_mse/state_size)))
                        f.write("current running rms alt gt mesh gt evl : {}\n".format(np.sqrt(evolution_loss_alt_gt_mesh_gt_evl_mse/state_size)))
                        f.write("current running rms alt 008 gt evl: {}\n".format(np.sqrt(evolution_loss_alt_008_gt_evl_mse/state_size)))
                        f.write("current running rms heursitc gt evl: {}\n".format(np.sqrt(evolution_loss_heuristic_gt_evl_mse/state_size)))
                        f.write("current running rms: {}\n".format(np.sqrt(evolution_loss_mse/state_size)))

                    if count>=break_index:
                        break
                    # break
        evolution_loss_rmse_average =  np.sqrt(evolution_loss_mse/state_size)
        evolution_loss_alt_gt_mesh_gt_evl_mse_average =  np.sqrt(evolution_loss_alt_gt_mesh_gt_evl_mse/state_size)
        evolution_loss_alt_gt_evl_mse_average =  np.sqrt(evolution_loss_alt_gt_evl_mse/state_size)
        evolution_loss_alt_008_gt_evl_mse_average =  np.sqrt(evolution_loss_alt_008_gt_evl_mse/state_size)
        evolution_loss_heuristic_gt_evl_mse_average =  np.sqrt(evolution_loss_heuristic_gt_evl_mse/state_size)


        state_size_remeshed = state_size_remeshed/actual_count
        print("count",count)
        print("{}, {}, evolution_loss_alt_gt_mesh_gt_evl_mse_average".format(hashva,beta),evolution_loss_alt_gt_mesh_gt_evl_mse_average)
        print("{}, {}, evolution_loss_alt_gt_evl_mse_average".format(hashva,beta),evolution_loss_alt_gt_evl_mse_average)
        print("{}, {}, evolution_loss_alt_008_gt_evl_mse_average".format(hashva,beta),evolution_loss_alt_008_gt_evl_mse_average)
        print("{}, {}, evolution_loss_heuristic_gt_evl_mse_average".format(hashva,beta),evolution_loss_heuristic_gt_evl_mse_average)
        print("{}, {}, evolution_loss_rmse_average".format(hashva,beta),evolution_loss_rmse_average, "state_size_remeshed", state_size_remeshed)
        with open('./results/LAMP_2d/{}/summary.txt'.format(name), 'a') as f:
            f.write("count {}".format(count))
            f.write("{}, {}, evolution_loss_alt_gt_mesh_gt_evl_mse_average {}\n".format(hashva,beta,evolution_loss_alt_gt_mesh_gt_evl_mse_average))
            f.write("{}, {}, evolution_loss_alt_gt_evl_mse_average {}\n".format(hashva,beta,evolution_loss_alt_gt_evl_mse_average))
            f.write("{}, {}, evolution_loss_alt_008_gt_evl_mse_average {}\n".format(hashva,beta,evolution_loss_alt_008_gt_evl_mse_average))
            f.write("{}, {}, evolution_loss_heuristic_gt_evl_mse_average {}\n".format(hashva,beta,evolution_loss_heuristic_gt_evl_mse_average))
            f.write("{}, {}, evolution_loss_rmse_average {}\n".format(hashva,beta,evolution_loss_rmse_average))

        dicts= [info, data_clone, evolution_loss_rmse_average,state_size_remeshed,list_evolution_loss_mse,list_evolution_loss_mse_alt,list_all_nodes,list_all_nodes_remesh]
        torch.cuda.empty_cache()
    return info, data_clone, dicts, min_evolution_loss_rmse_average


def run_hash(all_hashes,dirname=None,evo_dirname="evo-2d_2023_02_18",evo_hash="9UQLIKKc_ampere1",gpu=0):

    p = Printer()

    all_hashes = [all_hashes]
    print(all_hashes)
    load_best_model = False
    device = "cuda:{}".format(gpu)
    beta = 0
    constrain_edge_size = True

    mode = "best"
    evo_dirname = os.path.join(EXP_PATH, evo_dirname)
    if not dirname.endswith("/"):
        evo_dirname += "/"
    all_dict = {}
    isplot = False
    seed_everything(42)
    filename = filter_filename(evo_dirname, include=evo_hash)
    print(filename)
    try:
        data_record = pload(evo_dirname + filename[0])
    except Exception as e:
        print(f"error {e} in evo_hash {evo_hash}")
    p.print(f"Hash {evo_hash}, best model at epoch {data_record['best_epoch']}:", banner_size=160)
    args = init_args(update_legacy_default_hyperparam(data_record["args"]))
    args.filename = filename
    data_record['best_model_dict']['type'] = 'GNNRemesherPolicy'
    data_record['best_model_dict']['noise_amp'] = 0.
    data_record['best_model_dict']["correction_rate"] = 0.
    data_record['best_model_dict']["batch_size"] = 16
    best_gnn_evl_model = load_model(data_record["best_model_dict"], device=device)
    print("Load the model with best validation loss.")
    best_gnn_evl_model.eval()
    
    value = {}
    min_evolution_loss_rmse_average = 1000
    min_hash = None

    for hashva in all_hashes:
        seed_everything(42)
        model, evolution_model, test_loader, test_loader_008, args = get_results_2d([hashva],
                                                        mode=-1,
                                                        exclude_idx=(None,),
                                                        dirname=dirname,
                                                        suffix="",device=device)
        model.to(args.device)
        evolution_model.to(args.device)
        if load_best_model:
            evolution_model = best_gnn_evl_model.eval().to(args.device)
        if constrain_edge_size:
            model.actor.min_edge_size=max(0.04,model.actor.min_edge_size)
        model.eval()
        evolution_model.eval()
        
        name = "{}_{}_{}_{}_004".format(hashva,load_best_model,constrain_edge_size,beta)
        seed_everything(42)
        info, data, dicts, minval = get_eval(model, test_loader,test_loader_008, evolution_model,break_index=50*3,hashva=hashva,beta=beta,name=name,best_evolution_model=best_gnn_evl_model.eval(),args=args)

        if minval<min_evolution_loss_rmse_average:
            min_evolution_loss_rmse_average = minval
            min_hash = hashva
        print("current min rmse is {}, hash is {}".format(min_evolution_loss_rmse_average, min_hash))


import configargparse
p = configargparse.ArgumentParser()
p.add_argument('--hasval',type=str,default="Rf6n5sRM_ampere4")
p.add_argument('--dirname',type=str,default="2d_rl_reproduce_2023-02-26")
p.add_argument('--evo_dirname',type=str,default="evo-2d_2023_02_18")
p.add_argument('--evo_hash',type=str,default="9UQLIKKc_ampere1")
p.add_argument('--gpu',type=int,default=7)
opt = p.parse_args()
torch.cuda.set_device("cuda:{}".format(opt.gpu))
print("enter run hash")
run_hash(opt.hasval,gpu=opt.gpu,dirname=opt.dirname,evo_dirname=opt.evo_dirname,evo_hash=opt.evo_hash)