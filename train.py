import argparse
import datetime
from deepsnap.batch import Batch as deepsnap_Batch
import gc
import numpy as np
import pdb
import pickle
import pprint as pp
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import time

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from lamp.argparser import arg_parse
from lamp.models import load_data, get_model, load_model, unittest_model, build_optimizer, test
from lamp.gnns import GNNRemesher
from lamp.datasets.arcsimmesh_dataset import ArcsimMesh
from lamp.pytorch_net.util import Attr_Dict, Batch, filter_filename, pload, pdump, Printer, get_time, init_args, update_args, clip_grad, set_seed, update_dict, filter_kwargs, plot_vectors, plot_matrices, make_dir, get_pdict, to_np_array, record_data, make_dir, Early_Stopping, str2bool, get_filename_short, print_banner, get_num_params, ddeepcopy as deepcopy, write_to_config
from lamp.utils import EXP_PATH, MeshBatch
from lamp.utils import p, update_legacy_default_hyperparam, get_grad_norm, loss_op_core, get_model_dict, get_elements, is_diagnose, get_keys_values, loss_op, to_tuple_shape, parse_multi_step, get_device, seed_everything


# In[ ]:

def find_hash_and_load(all_hash,mode=-1,exclude_idx=(None,),dirname=None,suffix=""):
    isplot = True
    df_dict_list = []
    dirname_start = "tailin-rl_2022-9-22/" if dirname is None else dirname
    for hash_str in all_hash:
        df_dict = {}
        df_dict["hash"] = hash_str
        # Load model:
        is_found = False
        for dirname_core in [
             dirname_start,
             "tailin-rl_2022-9-22/",
             "qq-rl_2022-11-5/",
             "qq-rl_2022-9-26/",
             "multiscale_cloth_2022-9-26/",
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
            # p.print(f"Hash {hash_str}, best model at epoch {data_record['best_epoch']}:", banner_size=100)
            print(f"error {e} in hash_str {hash_str}")
            continue
    return data_record

args = arg_parse()
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    is_jupyter = True
    args.exp_id = "tailin-test"
    args.date_time = "8-27"
    # args.date_time = "{}-{}".format(datetime.datetime.now().month, datetime.datetime.now().day)

    # Train:
    args.epochs = 200
    args.contrastive_rel_coef = 0
    args.n_conv_blocks = 6
    args.latent_noise_amp = 1e-5
    args.multi_step = "1"
    args.latent_multi_step = "1^2^3^4"
    args.latent_loss_normalize_mode = "targetindi"
    args.channel_mode = "exp-16"
    args.batch_size = 20
    args.val_batch_size = 20
    args.reg_type = "None"
    args.reg_coef = 1e-4
    args.is_reg_anneal = True
    args.lr_scheduler_type = "cos"
    args.id = "test2"
    args.n_workers = 0
    args.plot_interval = 50
    args.temporal_bundle_steps = 1

    ##################################
    # RL algorithm:
    ##################################
    # args.algo chooses from "gnnremesher-evolution(+reward:32)", "rlgnnremesher^sizing", "rlgnnremesher^agent"
    args.algo = "rlgnnremesher^agent"
    if args.algo.startswith("rlgnnremesher"):
        args.rl_coefs = "None"
        args.rl_horizon = 4
        args.reward_mode = "lossdiff+statediff"
        args.reward_beta = "1"
        args.reward_src = "env"
        args.rl_lambda = 0.95
        args.rl_gamma = 0.99
        args.rl_rho = 1.
        args.rl_eta = 1e-4
        args.rl_critic_update_iterations = 10
        args.rl_data_dropout = "node:0-0.3:0.5"

        args.value_latent_size = 32 
        args.value_num_pool = 1
        args.value_act_name = "elu"
        args.value_act_name_final = "linear"
        args.value_layer_norm = False
        args.value_batch_norm = False
        args.value_num_steps = 3
        args.value_pooling_type = "global_mean_pool"
        args.value_target_mode = "value-lambda"

        args.load_dirname = "tailin-multi_2022-8-27"
        args.load_filename = "IHvBKQ8K_ampere3"

    ##################################
    # Dataset and model:
    ##################################
    # args.dataset = "mppde1df-E2-100"
    # args.dataset = "arcsimmesh_square"
    args.dataset = "arcsimmesh_square_annotated"
    if args.dataset.startswith("mppde1d"):
        args.latent_size = 64
        args.act_name = "elu"
        args.use_grads = False
        args.n_train = "-1"
        args.epochs = 2000
        args.use_pos = False
        args.latent_size = 64
        args.contrastive_rel_coef = 0
        args.is_prioritized_dropout = False
        args.input_steps = 1
        args.multi_step = "1^2:0.1^3:0.1^4:0.1"
        args.temporal_bundle_steps = 25
        args.n_train = ":100"
        args.epochs = 2000
        args.test_interval = 100
        args.save_interval = 100

        args.data_dropout = "node:0-0.1:0.1"
        args.use_pos = False
        args.rl_coefs = "reward:0.1"

        # Data:
        args.time_interval = 1
        args.dataset_split_type = "random"
        args.train_fraction = 1

        # Model:
        args.evolution_type = "mlp-3-elu-2"
        args.forward_type = "Euler"
        args.act_name = "elu"

        args.gpuid = "7"
        args.is_unittest = True
    
    elif args.dataset.startswith("arcsimmesh"):
        args.exp_id = "takashi-2dtest"
        args.date_time = "{}-{}".format(datetime.datetime.now().month, datetime.datetime.now().day)
        args.algo = "gnnremesher-evolution"
        args.encoder_type = "cnn-s"
        args.evo_conv_type = "cnn"
        args.decoder_type = "cnn-tr"
        args.padding_mode = "zeros"
        args.n_conv_layers_latent = 3
        args.n_conv_blocks = 4
        args.n_latent_levs = 1
        args.is_latent_flatten = True
        args.latent_size = 16
        args.act_name = "elu"
        args.decoder_act_name = "rational"
        args.use_grads = False
        args.n_train = "-1"
        args.use_pos = False
        args.contrastive_rel_coef = 0
        args.is_prioritized_dropout = False
        args.input_steps = 2
        args.multi_step = "1"
        args.latent_multi_step = "1"
        args.temporal_bundle_steps = 1
        # args.static_encoder_type = "param-2-elu"
        args.static_latent_size = 16
        args.n_train = ":100"
        args.epochs = 20
        args.test_interval = 10
        args.save_interval = 10

        #args.data_dropout = "node:0-0.4"
        args.use_pos = False
        # args.load_dirname = "tailin-multi_2022-8-27"
        # args.load_filename = "mppde1de-E2-400-nt-1000-nx-400_train_-1_algo_gnnremesher_ebm_False_ebmt_cd_enc_cnn-s_evo_cnn_act_elu_hid_16_lo_mse_recef_1_conef_1_nconv_4_nlat_2_clat_1_lf_True_reg_None_id_0_Hash_ufU6+M1A_ampere3.p"
        args.load_filename = "IHvBKQ8K_ampere3"
        args.rl_coefs = "reward:0.1"

        # Data:
        args.time_interval = 1
        args.dataset_split_type = "random"
        args.train_fraction = 1

        # Model:
        args.evolution_type = "mlp-3-elu-2"
        args.forward_type = "Euler"
        args.act_name = "elu"
        args.is_mesh = True
        args.edge_attr=True
        # args.edge_threshold=0.000001
        args.edge_threshold=0.

        args.gpuid = "3"
        args.is_unittest = True

except:
    is_jupyter = False

if args.dataset.startswith("mppde1d"):
    if args.dataset.endswith("-40"):
        args.output_padding_str = "0-0-0-0"
    elif args.dataset.endswith("-50"):
        args.output_padding_str = "1-0-1-0"
    elif args.dataset.endswith("-100"):
        args.output_padding_str = "1-1-0-0"


# # 2. Load data and model:

# In[ ]:


set_seed(args.seed)
(dataset_train_val, dataset_test), (train_loader, val_loader, test_loader) = load_data(args)
p.print(f"Minibatches for train: {len(train_loader)}")
p.print(f"Minibatches for val: {len(val_loader)}")
p.print(f"Minibatches for test: {len(test_loader)}")
args_test8 = deepcopy(args)
if args.dataset.startswith("m"):
    args_test8.multi_step = "1^8"
    args_test8.pred_steps = 8
    args_test8.is_train = False 
else:
    args_test8.multi_step = "1^75"
    args_test8.pred_steps = 75
    args_test8.is_train = False 
args_test8.batch_size = 1
args_test8.val_batch_size = 1
# seed_everything(42)
(_, dataset_test8), (_, val_loader8, test_loader8) = load_data(args_test8)

p.print(f"Minibatches for val8: {len(val_loader8)}")
p.print(f"Minibatches for test8: {len(test_loader8)}")
val_loader8 = None
device = get_device(args)
args.device = device
args_test8.device = device

data = deepcopy(dataset_test[0])
epoch = 0

if args.rl_is_finetune_evolution and args.dataset.startswith("a"):
    args_testfine = deepcopy(args)
    args_testfine.use_fineres_data = True
    (dataset_train_val_fine,_), (train_loader_fine, _, _) = load_data(args_testfine)
    p.print(f"Minibatches for trainfine: {len(train_loader_fine)}")
else:
    args_testfine=None
    
model = get_model(args, data, device)
if args.algo.startswith("rlgnnremesher") or args.algo.startswith("srlgnnremesher"):
    device = get_device(args)
    loaded_dirname = EXP_PATH + args.load_dirname
    filenames = filter_filename(loaded_dirname, include=[args.load_filename])
    assert len(filenames) == 1
    loaded_filename = os.path.join(loaded_dirname, filenames[0])
    data_record_load = pload(loaded_filename)

    args_load = init_args(update_legacy_default_hyperparam(data_record_load["args"]))
    args_load.multi_step = args.multi_step
    evolution_model = load_model(data_record_load["model_dict"][-1], device)
    if args.fix_alt_evolution_model:
        evolution_model_alt = load_model(data_record_load["model_dict"][-1], device)
        evolution_model_alt.to(device)
        evolution_model_alt.eval()
    else:
        evolution_model_alt = None
    if not args.rl_is_finetune_evolution:
        evolution_model.eval()


if args.load_hash!="None":
    data_record_load = find_hash_and_load([args.load_hash])
    model.actor.load_state_dict(data_record_load["model_dict"][-1]["actor_model_dict"]["state_dict"])
    model.critic.load_state_dict(data_record_load["model_dict"][-1]["critic_model_dict"]["state_dict"])
    model.critic_target.load_state_dict(data_record_load["model_dict"][-1]["critic_model_dict"]["state_dict"])
    # model = load_model(data_record_load["model_dict"][-1],device=device)
    evolution_model = load_model(data_record_load["evolution_model_dict"][-1],device=device)
    if not args.rl_is_finetune_evolution:
        evolution_model.eval()

# # 3. Training:

# In[ ]:


if args.algo.startswith("rlgnnremesher"):
    separate_params = [
        {'params': model.actor.parameters(), 'lr': args.actor_lr},
        {'params': model.critic.parameters(), 'lr': args.value_lr},
    ]
    opt, scheduler = build_optimizer(
        args, params=None,
        separate_params=separate_params,
    )
elif args.algo.startswith("srlgnnremesher"):
    separate_params = [ {'params': model.parameters(), 'lr': args.actor_lr},]
    opt, scheduler = build_optimizer(args, params=None,separate_params=separate_params,)
else:
    opt, scheduler = build_optimizer(args, model.parameters())

if args.rl_is_finetune_evolution:
    opt_params = [{'params': evolution_model.parameters(), 'lr': args.lr}]
    opt_evolution, opt_scheduler = build_optimizer(
        args, params=None,
        separate_params=opt_params,
    )
        
n_params_model = get_num_params(model)
p.print("n_params_model: {}".format(n_params_model), end="")
machine_name = os.uname()[1].split('.')[0]
data_record = {"n_params_model": n_params_model, "args": update_dict(args.__dict__, "machine_name", machine_name),
               "best_train_model_dict": [], "best_train_loss": [], "best_train_loss_history":[]}
early_stopping = Early_Stopping(patience=args.early_stopping_patience)

short_str_dict = {
    "dataset": "",
    "n_train": "train",
    "algo": "algo",
    "is_ebm": "ebm",
    "ebm_train_mode": "ebmt",
    "encoder_type": "enc",
    "evo_conv_type": "evo",
    "act_name": "act",
    "latent_size": "hid",
    "loss_type": "lo",
    "recons_coef": "recef",
    "consistency_coef": "conef",
    "n_conv_blocks": "nconv",
    "n_latent_levs": "nlat",
    "n_conv_layers_latent": "clat",
    "is_latent_flatten": "lf",
    "reg_type": "reg",
    "gpuid": "gpu",
    "id": "id",
}

filename_short = get_filename_short(
    short_str_dict.keys(),
    short_str_dict,
    args_dict=args.__dict__,
)
filename = EXP_PATH + "{}_{}/".format(args.exp_id, args.date_time) + filename_short[:-2] + "_{}.p".format(machine_name)
write_to_config(args, filename)
args.filename = filename
kwargs = {}
if args.algo.startswith("rlgnnremesher") or args.algo.startswith("srlgnnremesher"):
    kwargs["evolution_model"] = evolution_model
    kwargs["evolution_model_alt"] = evolution_model_alt
p.print(filename, banner_size=100)
# print(model)
make_dir(filename)
best_val_loss = np.Inf
if args.load_filename != "None":
    val_loss = np.Inf
collate_fn = deepsnap_Batch.collate() if data.__class__.__name__ == "HeteroGraph" else MeshBatch(
    is_absorb_batch=True, is_collate_tuple=True).collate() if args.dataset.startswith("arcsimmesh") else Batch(
    is_absorb_batch=True, is_collate_tuple=True).collate()
if args.is_unittest:
    unittest_model(model,
                   collate_fn([data, data]), args, device, use_grads=args.use_grads, use_pos=args.use_pos, is_mesh=args.is_mesh,
                   test_cases="all" if not (args.dataset.startswith("PIL") or args.dataset.startswith("PHIL")) else "model_dict", algo=args.algo,
                   **kwargs
                  )
if args.is_tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(EXP_PATH + '/log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
pp.pprint(args.__dict__)

if args.wandb:
    import wandb
    args.wandb_project_name = name="_".join([args.wandb_project_name,args.algo])
    wandb.init(project=args.wandb_project_name, entity="multi-scale", name="_".join(["beta_",args.reward_beta]+filename.split("_")[-2:])[:-2] + f'_ntrain_{args.n_train}{"_" + args.id if args.id != "0" else ""}', 
               config={"name": args.exp_id})
    wandb.watch(model, log_freq=1000, log="all")
    wandb.watch(evolution_model, log_freq=1000, log="all")
    wandb.config=vars(args)
else:
    wandb = None
step_num = 0
opt_actor = None
opt_evl = None

# if (args.algo.startswith("rlgnnremesher") or args.algo.startswith("srlgnnremesher")) and args.wandb:
#     model.get_tested(
#         test_loader8,
#         args_test8,
#         current_epoch=0,
#         current_minibatch=0,
#         wandb=wandb,
#         step_num=step_num,
#         **kwargs
#     )

if args.load_hash!="None":
    if "last_optimizer_dict" in data_record_load.keys():
        opt.load_state_dict(data_record_load["last_optimizer_dict"])
    if "last_evolution_optimizer_dict" in data_record_load.keys():
        opt_evolution.load_state_dict(data_record_load["last_evolution_optimizer_dict"])
    if "last_scheduler_dict" in data_record_load.keys():
        opt_scheduler.load_state_dict(data_record_load["last_scheduler_dict"])



while epoch < args.epochs:
    total_loss = 0
    count = 0
    
    model.train()
    train_info = {}
    best_train_loss = np.Inf
    last_few_losses = []
    num_losses = 20
    t_start = time.time()
    
    if args.rl_is_finetune_evolution and args.dataset.startswith("a"):
        train_loader_fine_iterator = iter(train_loader_fine)
    for j, data in enumerate(train_loader):
        if args.rl_is_finetune_evolution and args.dataset.startswith("a"):
            try:
                data_fine = next(train_loader_fine_iterator)
            except StopIteration:
                train_loader_fine_iterator = iter(train_loader_fine)
                data_fine = next(dataloader_iterator)
        else:
            data_fine=None
        
        t_end = time.time()
        if args.verbose >= 2 and j % 100 == 0:
            p.print(f"Data loading time: {t_end - t_start:.6f}")
        if data.__class__.__name__ == "Attr_Dict":
            data = data.to(device)
            if args.rl_is_finetune_evolution and args.dataset.startswith("a"): data_fine = data_fine.to(device)
        else:
            data.to(device)
            if args.rl_is_finetune_evolution and args.dataset.startswith("a"): data_fine.to(device)
        opt.zero_grad()
        if args.rl_is_finetune_evolution:
            opt_evolution.zero_grad()
            if args.actor_critic_step==None:
                opt_actor=True 
                opt_evl = True
            else:
                if step_num%(args.actor_critic_step+args.evolution_steps)<args.actor_critic_step: 
                    opt_actor=True 
                    opt_evl = False
                    evolution_model.eval()
                    model.train()
                else:
                    opt_actor=False
                    opt_evl = True
                    evolution_model.train()
                    model.eval()
        else:
            opt_evl = False
            opt_actor = True
            
        if (not(opt_evl)):
            data_fine = None
        
            
        args_core = update_args(args, "multi_step", "1") if epoch < args.multi_step_start_epoch else args
        loss = model.get_loss(
            data,
            args_core,
            current_epoch=epoch,
            current_minibatch=j,
            wandb=wandb,
            step_num=step_num,
            opt_evl=opt_evl,
            opt_actor=opt_actor,
            data_fine=data_fine,
            **kwargs
        )
        if is_diagnose(loc="1", filename=filename):
            pdb.set_trace()
        if not args.test_reward_random_sample:
            p.print("7", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)
            loss.backward()
            p.print("8", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)

            if args.wandb and step_num % args.wandb_step == 0:
                grad_norm = get_grad_norm(model)
                wandb.log({"train_grad_norm": grad_norm})
            if args.max_grad_norm != -1:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.wandb and step_num % args.wandb_step == 0:
                    grad_norm = get_grad_norm(model)
                    wandb.log({"train_grad_norm_clipped": grad_norm})

            p.print("8.1", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)
            if not args.rl_is_finetune_evolution:
                opt.step()
                p.print("8.2a", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)

            else:
                if opt_actor:
                    opt.step()
                    p.print("8.2a", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)

                if opt_evl:
                    opt_evolution.step()
                    p.print("8.2b", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)        

            total_loss = total_loss + loss.item()
            count += 1
            if args.algo.startswith("rlgnnremesher") or args.algo.startswith("srlgnnremesher"):
                if not args.soft_update:
                    model.monitor_copy_critic(args.rl_critic_update_iterations, args.verbose)
                else:
                    model.soft_update()
            if args.is_tensorboard:
                writer.add_scalar("loss", total_loss, epoch)
            p.print("8.3", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)
            data.to("cpu")
            del loss
            del data
            if j % 100 == 0:
                p.print("8.31", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)
                gc.collect()
                p.print("8.32", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)
            p.print("8.4", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)
            keys, values = get_keys_values(model.info, exclude=["pred"])
            record_data(train_info, values, keys)
            t_start = time.time()
            
            if args.save_iterations != -1 and step_num % args.save_iterations == 0 and args.save_iterations:
                record_data(data_record, [model.model_dict, step_num], ["model_dict_step", "step_num"])
                p.print("9", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)
                if "evolution_model" in locals():
                    record_data(data_record, [evolution_model.model_dict], ["evolution_model_dict_step"])
                with open(filename, "wb") as f:
                    pickle.dump(data_record, f)
        step_num = step_num + 1
    p.print("10", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)
    for key, item in train_info.items():
        train_info[key] = np.mean(item)
    train_loss = total_loss / max(count,1)
    if epoch % args.test_interval == 0 or epoch == args.epochs - 1:

        if (args.algo.startswith("rlgnnremesher") or args.algo.startswith("srlgnnremesher")) and args.wandb:
            test_loss, test_info = model.get_tested(
                test_loader8,
                args_test8,
                current_epoch=epoch,
                current_minibatch=j,
                wandb=wandb,
                step_num=step_num,
                **kwargs
            )
            val_loss, val_info = None, None
        else:
            val_loss, val_info = test(
                val_loader, model, device, args,
                density_coef=0, current_epoch=epoch, current_minibatch=0,
                **kwargs
            )
            test_loss, test_info = test(
                test_loader, model, device, args,
                density_coef=0, current_epoch=epoch, current_minibatch=0,
                **kwargs
            )
        if val_loss is None:
            val_loss = test_loss
            val_info = test_info
            is_val_loss = False
        else:
            is_val_loss = True
        to_stop = early_stopping.monitor(val_loss)
        gc.collect()
    p.print("11", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)

    if is_diagnose(loc="2", filename=filename):
        pdb.set_trace()
    if args.lr_scheduler_type == "rop":
        scheduler.step(val_loss)
        kwargs["scheduler_disc"].step(model.info["loss_disc"])
    elif args.lr_scheduler_type == "None":
        pass
    else:
        scheduler.step()
        if args.disc_coef > 0 or args.disc_coef == -1:
            kwargs["scheduler_disc"].step()
    record_data(data_record, [epoch, train_loss], ["epoch", "train_loss"])
    record_data(data_record, list(train_info.values()), ["{}_tr".format(key) for key in train_info])
    p.print("Epoch {:03d}:     Train: {:.4e}  for exp {}".format(epoch + 1, train_loss, filename.split("_")[-2]), end="")
    if epoch % args.test_interval == 0 or epoch == args.epochs - 1:
        record_data(data_record, [epoch], ["test_epoch"])
        if is_val_loss:
            record_data(data_record, [val_loss], ["val_loss"])
            record_data(data_record, list(val_info.values()), ["{}_val".format(key) for key in val_info])
            p.print("       Val: {:.6f}\n".format(val_loss))
            for key, loss_ele in val_info.items():
                print("        {}: {:.6f}\n".format(key.split("loss_")[-1], loss_ele))
        if test_loss is not None:
            record_data(data_record, [test_loss], ["test_loss"])
            record_data(data_record, list(test_info.values()), ["{}_te".format(key) for key in test_info])
            p.print("       Test: {:.6f}".format(test_loss))
            for key, loss_ele in test_info.items():
                print("        {}: {:.6f}\n".format(key.split("loss_")[-1], loss_ele))
    print()
    for jj, (key, value) in enumerate(train_info.items()):
        if jj % 2 == 0:
            print(f"{key}_tr:   \t{np.mean(value):.6f}", end="")
        else:
            print(f"\t{key}_tr:   \t{np.mean(value):.6f}  ")
    print("\n")
    if is_diagnose(loc="3", filename=filename):
        pdb.set_trace()
    if epoch % args.save_interval == 0 and epoch >= 0:
        p.print(filename)
        record_data(data_record, [epoch, get_model_dict(model)], ["save_epoch", "model_dict"])
        if "evolution_model" in locals():
            record_data(data_record, [evolution_model.model_dict], ["evolution_model_dict"])
        with open(filename, "wb") as f:
            pickle.dump(data_record, f)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        data_record["best_model_dict"] = get_model_dict(model)
        data_record["best_optimizer_dict"] = opt.state_dict()
        data_record["best_scheduler_dict"] = scheduler.state_dict() if scheduler is not None else None
        if "evolution_model" in locals():
            data_record["best_evolution_model_dict"] = evolution_model.model_dict
            if args.rl_is_finetune_evolution:
                data_record["best_evolution_optimizer_dict"] = opt_evolution.state_dict()
        data_record["best_epoch"] = epoch
    data_record["last_model_dict"] = get_model_dict(model)
    data_record["last_optimizer_dict"] = opt.state_dict()
    data_record["last_scheduler_dict"] = scheduler.state_dict() if scheduler is not None else None
    data_record["last_epoch"] = epoch
    if "evolution_model" in locals():
        data_record["last_evolution_model_dict"] = evolution_model.model_dict
        if args.rl_is_finetune_evolution:
            data_record["last_evolution_optimizer_dict"] = opt_evolution.state_dict()
    p.print("12", precision="millisecond", is_silent=args.is_timing<1, avg_window=1)

    pdump(data_record, filename)
    if "to_stop" in locals() and to_stop:
        p.print("Early-stop at epoch {}.".format(epoch))
        break
    epoch += 1
record_data(data_record, [epoch, get_model_dict(model)], ["save_epoch", "model_dict"])
if "evolution_model" in locals():
    record_data(data_record, [evolution_model.model_dict], ["evolution_model_dict"])
pdump(data_record, filename)

