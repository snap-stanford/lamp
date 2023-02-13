# LAMP: Learning Controllable Adaptive Simulation for Multi-resolution Physics

[Paper](https://openreview.net/forum?id=PbfgkZ2HdbE) | [Poster](https://github.com/snap-stanford/lamp/blob/master/assets/lamp_poster.pdf) | [Slide](https://docs.google.com/presentation/d/1cMRGe2qNIrzSNRTUtbsVUod_PvyhDHcHzEa8wfxiQsw/edit?usp=sharing) | [Project Page](https://snap.stanford.edu/lamp/)

This is the official repo for the paper [Learning Controllable Adaptive Simulation for Multi-resolution Physics](https://openreview.net/forum?id=PbfgkZ2HdbE) (Tailin Wu*, Takashi Maruyama*, Qingqing Zhao*, Gordon Wetzstein, Jure Leskovec, ICLR 2023 **spotlight**). It is the first fully DL-based surrogate model that jointly learns the evolution model, and optimizes spatial resolutions to reduce computational cost, learned via reinforcement learning. We demonstrate that LAMP is able to adaptively trade-off computation to improve long-term prediction error, by performing spatial refinement and coarsening of the mesh. LAMP outperforms state-of-the-art (SOTA) deep learning surrogate models, with up to 39.3% error reduction for 1D nonlinear PDEs, and outperforms SOTA MeshGraphNets + Adaptive Mesh Refinement in 2D mesh-based simulations.

<a href="url"><img src="https://github.com/snap-stanford/lamp/blob/master/assets/lamp_architecture.png" align="center" width="700" ></a>

# Installation

1. First clone the directory. Then run the following command to initialize the submodules:

```code
git submodule init; git submodule update
```

2. Install dependencies.

First, create a new environment using [conda](https://docs.conda.io/en/latest/miniconda.html) (with python >= 3.7). Then install pytorch, torch-geometric and other dependencies as follows (the repository is run with the following dependencies. Other version of torch-geometric or deepsnap may work but there is no guarentee.)

Install pytorch (replace "cu113" with appropriate cuda version. For example, cuda11.1 will use "cu111"):
```code
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Install torch-geometric. Run the following command:
```code
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install torch-geometric==1.7.2
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
```

Install other dependencies:
```code
pip install -r requirements.txt
```

Then set up wandb, following [this link](https://docs.wandb.ai/quickstart).

# Dataset

The dataset files can be downloaded via [this link](https://drive.google.com/drive/folders/1ld5I86mPC7wWTxPhbCtG2AcH0vLW3o25?usp=share_link). 
* To run 1D experiment, download the files under "mppde1d_data/" in the link into the "data/mppde1d_data/" folder in the local repo. 
* To run 2D mesh-based experiment, download the files under "arcsimmesh_data/" in the link into the "data/arcsimmesh_data/" folder in the local repo.


# Training

Below we provide example commands for training LAMP. For all the commands that reproduce the experiments in the paper, see the [results/README.md](https://github.com/snap-stanford/lamp/tree/master/results).

## 1D nonlinear PDE:

First, pre-train the evolution model for 1D:

```code
python train.py --exp_id=evo-1d --date_time=2023-01-01 --dataset=mppde1df-E2-100-nt-250-nx-200 --time_interval=1 --data_dropout=node:0-0.3:0.1 --latent_size=64 --n_train=-1 --save_interval=5 --test_interval=5 --algo=gnnremesher --rl_coefs=None --input_steps=1 --act_name=silu --multi_step=1^2:0.1^3:0.1^4:0.1 --temporal_bundle_steps=25 --use_grads=False --is_y_diff=False --loss_type=mse --batch_size=16 --val_batch_size=16 --epochs=50 --opt=adam --weight_decay=0 --seed=0 --id=0 --verbose=1 --n_workers=0 --gpuid=0
```

The learned model will be saved under `./results/{--exp_id}_{--date_time}/`, where the `{--exp_id}` and `{--date_time}` are specified in the above command. The filename has the format of `*{hash}_{machine_name}.p`, e.g. "mppde1df-E2-100-nt-250-nx-200_train_-1_algo_gnnremesher_..._Hash_mhkVkAaz_ampere3.p", then the `{hash}` is `mhkVkAaz` and `{machine_name}` is `ampere3`, where the `{hash}` is uniquely determined by **all** the argument settings in the [argparser.py](https://github.com/snap-stanford/lamp/blob/master/argparser.py) (therefore, as long as any argument setting is different, the filename will be different and will not overwrite each other).

Then, jointly training the remeshing model via reinforcement learning (RL) and the evolution model. The `--load_dirname` below should use folder name `{exp_id}_{date_time}` where the evolution model is located (as specified above), and the `--load_filename` should use part of the filename that can uniquely identify this model file, and should include the `{hash}` of this model.

```code
python train.py --load_dirname=evo-1d_2023-01-01 --load_filename=qvQry9QJ --exp_id=rl-1d --wandb_project_name=rl-1d_2023-01-02 --wandb=True --date_time=2023-01-02 --dataset=mppde1df-E2-100-nt-250-nx-200 --algo=srlgnnremesher --time_interval=1 --data_dropout=None --latent_size=64 --n_train=-1 --input_steps=1 --act_name=elu --multi_step=1^2:0.1^3:0.1^4:0.1 --temporal_bundle_steps=25 --use_grads=False --is_y_diff=False --loss_type=mse --batch_size=128 --val_batch_size=128 --epochs=100 --opt=adam --weight_decay=0 --seed=0 --verbose=1 --n_workers=0 --gpuid=7 --reward_mode=lossdiff+statediff --reward_beta=0-0.5 --rl_data_dropout=uniform:2 --min_edge_size=0.0014 --is_1d_periodic=False --is_normalize_pos=True --rl_horizon=4 --reward_loss_coef=5 --rl_eta=1e-2 --actor_lr=5e-4 --value_lr=1e-4 --value_num_pool=1 --value_pooling_type=global_mean_pool --value_latent_size=32 --value_batch_norm=False --actor_batch_norm=True --rescale=10 --edge_attr=True --rl_gamma=0.9 --value_loss_coef=0.5 --max_grad_norm=2 --is_single_action=False --value_target_mode=vanilla --wandb_step_plot=100 --wandb_step=20 --id=0 --save_iteration=1000 --save_interval=1 --test_interval=1 --gpuid=7 --lr=1e-4 --actor_critic_step=200 --evolution_steps=200 --reward_condition=True --max_action=20 --rl_is_finetune_evolution=True --rl_finetune_evalution_mode=policy:fine --id=0
```

### 2D mesh-based simulation:

First, pre-train the evolution model for 2D:

```code

```

Then, jointly training the remeshing model via RL and the evolution model:

```code

```

# Analysis

* For 1D experiments, to analyze the pretrained evolution model for LAMP and the baselines, use [analysis_1D_evo.ipynb](https://github.com/snap-stanford/lamp/blob/master/analysis_1d_evo.ipynb).

* For 1D experiments, to analyze the full model for LAMP and the baselines, use [analysis_1D_full.ipynb](https://github.com/snap-stanford/lamp/blob/master/analysis_1d_full.ipynb).

* For 2D experiments, to analyze the full model for LAMP and the baselines, use [analysis_2D_full.ipynb](https://github.com/snap-stanford/lamp/blob/master/analysis_2d_full.ipynb).


# Related Projects:

* [LE-PDE](https://github.com/snap-stanford/le_pde) (NeurIPS 2022): Accelerate the simulation and inverse optimization of PDEs. Compared to state-of-the-art deep learning-based surrogate models (e.g., FNO, MP-PDE), it is up to 15x improvement in speed, while achieving competitive accuracy.

# Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{wu2023learning,
title={Learning Controllable Adaptive Simulation for Multi-resolution Physics},
author={Tailin Wu and Takashi Maruyama and Qingqing Zhao and Gordon Wetzstein and Jure Leskovec},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=PbfgkZ2HdbE}
}
```
