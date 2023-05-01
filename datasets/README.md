# Dataset description

## Attribute discription

### ArcsimMesh (in arcsimmesh_dataset.py)

The ArcsimMesh dataset consists of multiple Data instances, each Data contains the state (M_t) of the mesh-based system (including past few steps' state, and a few steps of the future to be used for computing the loss. Specifically, each Data comprises the following attributes:

* **node_feature**: 3d coordinates of vertices, shape=[#vertices, 1, dimension of node feature]
* **node_dim**: tuple of the number of nodes in meshes of ground-truth trajectory
* **edge_attr**: edge attributes for edges in a mesh, shape=[#edges, dimension of edge feature]
* **node_label:** 3d coordinates of vertices in the ground-truth mesh at next time step, shape=[#vertices, dimension of node feature]
* **yfeatures**: tuple of node features (concatenation of 2d and 3d coordinates) in meshes of ground-truth trajectory, length of tuple is length of trajectory and each element in the tuple has shape of [#vertices, dimension of node feature]
* **reind_yfeatures**: same as yfeatures, for future use
* **bary_indices**: tuple of indices of nodes for barycentric interpolation, length of tuple is length of trajectory and each element in the tuple has shape of [#vertices, 3]
* **bary_weights**: tuple of barycentric weights for each node for barycentric interpolation, length of tuple is length of trajectory and each element in the tuple has shape of [#vertices, 3]
* **hist_indices**: list of node indices of meshes in history, used to adjust input for evolution model by barycentric interpolation. Length of list is same as the number of input meshes and each element in the tuple has shape of [#vertices, dimension of node feature]
* **hist_weights**: barycentric weights to adjust history nodes for the input of evolution model. Length of tuple is same as length of the number of input meshes and each element in the tuple has shape of [#vertices, 3]
* **history**: list of history meshes used for input to evolution model. The length of list is the number of input meshes, and each element is of shape [#vertices, dimension of node feature]
* **xface_list**: list of faces, length of the list is length of trajectory and each element in the tuple has shape of [#faces, 3]
* **y_tar**: tuple of interpolated nodes of future gt meshes, length of tuple is length of trajectory and each element in the tuple has shape of [#vertices, 3]
* **y_back**: tuple of hessian of interpolated 3d coordinates of nodes in future gt meshes, length of tuple is length of trajectory and each element in the tuple has shape of [#vertices, 3]
* **x_pos**: 2d coordinates corresponding to node_feature, shape=[#vertices, 2]
* **edge_index**: edge index
* **xfaces**: faces of a mesh, shape=[#faces, 3]
* **yface_list**: list of faces of future meshes, length of tuple is length of trajectory and each element in the tuple has shape of [#faces, 3]
* **yedge_index**: list of edge indices in future meshes, the length of the list is same as the length of trajectory and each element is of shape of corresponding edge index
* **x_bdd**: data structure for batch creation
* **original_shape**: data structure for batch creation
* **dyn_dims**: dimension of dynamic node feature
* **compute_func**: data structure for sorting tensor
* **grid_keys** : data structure for sorting tensor
* **part_keys**: data structure for sorting tensor
* **time_step**: time step of the current mesh in a trajectory
* **sim_id**: index of trajectory
* **time_interval**: time interval for one step
* **cushion_input**: length of history


See below for the further details:

[Paper](https://openreview.net/forum?id=PbfgkZ2HdbE) | [Poster](https://github.com/snap-stanford/lamp/blob/master/assets/lamp_poster.pdf) | [Slides](https://docs.google.com/presentation/d/1cMRGe2qNIrzSNRTUtbsVUod_PvyhDHcHzEa8wfxiQsw/edit?usp=sharing) | [Project Page](https://snap.stanford.edu/lamp/)
