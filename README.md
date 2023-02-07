## LAMP: Learning Controllable Adaptive Simulation for Multi-resolution Physics

[Paper](https://openreview.net/forum?id=PbfgkZ2HdbE) | [Poster](https://github.com/snap-stanford/lamp/blob/master/assets/lamp_poster.pdf) | [Slide](https://docs.google.com/presentation/d/1cMRGe2qNIrzSNRTUtbsVUod_PvyhDHcHzEa8wfxiQsw/edit?usp=sharing) | [Project Page](https://snap.stanford.edu/lamp/)

This is the official repo for the paper [Learning Controllable Adaptive Simulation for Multi-resolution Physics](https://arxiv.org/abs/2206.07681) (Tailin Wu*, Takashi Maruyama*, Qingqing Zhao*, Gordon Wetzstein, Jure Leskovec, ICLR 2023 spotlight). It is the first fully DL-based surrogate model that jointly learns the evolution model, and optimizes spatial resolutions to reduce computational cost, learned via reinforcement learning. We demonstrate that LAMP is able to adaptively trade-off computation to improve long-term prediction error, by performing spatial refinement and coarsening of the mesh. LAMP outperforms state-of-the-art (SOTA) deep learning surrogate models, with up to 39.3% error reduction for 1D nonlinear PDEs, and outperforms SOTA MeshGraphNets + Adaptive Mesh Refinement in 2D mesh-based simulations.

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
# Dataset


# Training
