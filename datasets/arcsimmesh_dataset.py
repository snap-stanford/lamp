#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
import torch
from torch_geometric.data import Dataset, Data
# import dolfin
import pdb
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from lamp.utils import PDE_PATH, ARCSIMMESH_PATH
from lamp.pytorch_net.util import plot_matrices, Attr_Dict
from lamp.utils import get_root_dir, to_tuple_shape



class ArcsimMesh(Dataset):
    """ Generate pytorch tensor of cloth data from original 
    data generated by arcsim simulator.

    Attributes:
        dataset: name of dataset.
        input_steps: number of input to be used. 
        output_steps: number of output to be predicted.
        time_interval: step between input and output.
        is_train: indicate data is for training or not.
        show_missing_files: missing files to load data.
        is_traj: indicate data consists of trajectories.
        traj_len: length of trajectory.
        transform: option for transform data.
        pre_transform: option for transform data.
    """
    def __init__(
        self,
        dataset="arcsimmesh_square",
        input_steps=1,
        output_steps=1,
        time_interval=1,
        is_shifted_data=False,
        is_only_folding=True,
        use_fineres_data=False,
        is_train=True,
        show_missing_files=False,
        # n_init_smoke=1,
        is_traj=False,
        traj_len=None,
        transform=None,
        pre_transform=None,
    ):
        self.dataset = dataset
        if self.dataset.startswith("arcsimmesh"):
            self.root = os.path.join(PDE_PATH, ARCSIMMESH_PATH)
        else:
            raise
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.time_interval = time_interval
        # self.n_init_smoke = n_init_smoke
        # self.is_y_diff = is_y_diff
        self.is_train = is_train
        self.is_shifted_data = is_shifted_data
        self.is_only_folding=is_only_folding
        self.use_fineres_data = use_fineres_data
        
        #self.t_cushion_input = 0
        #self.t_cushion_output = 0
        self.t_cushion_input = self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1
        self.t_cushion_output = self.output_steps * self.time_interval if self.output_steps * self.time_interval > 1 else 1
        if self.dataset == "arcsimmesh_square":
            self.n_simu = 500 if self.is_train else 50
            self.time_stamps = 326
            self.t_cushion_input = 40
            self.t_cushion_output = 40
            self.original_shape = ()
            self.dyn_dims = 9  # velo_X, velo_Y, density
        elif self.dataset == "arcsimmesh_square_annotated":
            self.n_simu = 1000 if self.is_train else 50
            self.time_stamps = 326
            # self.t_cushion_input = 40
            # self.t_cushion_output = 40
            if self.is_shifted_data:
                self.t_cushion_input = self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1
                self.t_cushion_output = self.output_steps * self.time_interval if self.output_steps * self.time_interval > 1 else 1
            else:
                self.t_cushion_input = 20
                self.t_cushion_output = 20
            self.original_shape = ()
            self.dyn_dims = 9  # velo_X, velo_Y, density
        elif "interp" in self.dataset:
            if self.dataset == "arcsimmesh_square_annotated_interp_50":
                self.n_simu = 50
            elif self.dataset == "arcsimmesh_square_annotated_interp_500":
                self.n_simu = 500 if self.is_train else 50
            elif self.dataset == "arcsimmesh_square_annotated_coarse_minlen004_interp_500":
                self.n_simu = 500 if self.is_train else 50
            elif self.dataset == "arcsimmesh_square_annotated_coarse_minlen008_interp_500":
                self.n_simu = 500 if self.is_train else 50
            elif self.dataset == "arcsimmesh_square_annotated_coarse_minlen008_interp_500_gt_500":
                self.n_simu = 500 if self.is_train else 50
            if self.is_only_folding:
                self.time_stamps = 180
            else:
                self.time_stamps = 326
            self.t_cushion_input = self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1
            self.t_cushion_output = self.output_steps * self.time_interval if self.output_steps * self.time_interval > 1 else 1
            self.original_shape = ()
            self.dyn_dims = 9  
        else:
            raise
        self.is_traj = is_traj
        self.traj_len = traj_len
        if self.dataset == "arcsimmesh_square":
            if self.is_train: 
                self.dirname="arcsim_square_trajectories_train"
            else:
                self.dirname="arcsim_square_trajectories_test"
        elif self.dataset == "arcsimmesh_square_annotated":
            if self.is_train: 
                self.dirname="arcsim_square_trajectories_train"
            else:
                if self.is_shifted_data:
                    self.dirname="arcsim_square_trajectories_test_shifted" 
                else:
                    self.dirname="arcsim_square_trajectories_test"  
        elif self.dataset == "arcsimmesh_square_annotated_interp_50":
            if self.use_fineres_data:
                self.dirname = "arcsim_square_trajectories_test_shifted"
            else:
                self.dirname = "arcsim_square_trajectories_interp_50"
            self.dirname_finereso = "arcsim_square_trajectories_test_shifted"
        elif self.dataset == "arcsimmesh_square_annotated_interp_500":
            if self.use_fineres_data:
                self.dirname = "arcsim_square_trajectories_train"
            else:
                self.dirname = "arcsim_square_trajectories_interp_500"
            self.dirname_finereso = "arcsim_square_trajectories_train"
        elif self.dataset == "arcsimmesh_square_annotated_coarse_minlen004_interp_500":
            if self.use_fineres_data:
                self.dirname = "arcsim_square_trajectories_train"
            else:
                self.dirname = "output_square_coarse_minlen004_interp_500"
            self.dirname_finereso = "arcsim_square_trajectories_train"
        elif self.dataset == "arcsimmesh_square_annotated_coarse_minlen008_interp_500":
            # pdb.set_trace()
            if self.use_fineres_data:
                if self.is_train: 
                    self.dirname = "arcsim_square_trajectories_train"
                else:
                    if self.is_shifted_data:
                        self.dirname="arcsim_square_trajectories_test_shifted" 
                    else:
                        self.dirname="arcsim_square_trajectories_test"
            else:
                if self.is_train: 
                    self.dirname = "output_square_coarse_minlen008_interp_500"
                    self.dirname_finereso = "arcsim_square_trajectories_train"
                else:
                    self.dirname = "output_square_coarse_minlen008_test_interp_50"
                    self.dirname_finereso = "arcsim_square_trajectories_test"
            #self.dirname_finereso = "arcsim_square_trajectories_train"
        elif self.dataset == "arcsimmesh_square_annotated_coarse_minlen008_interp_500_gt_500":
            if self.use_fineres_data:
                if self.is_train: 
                    self.dirname = "arcsim_square_trajectories_train"
                else:
                    if self.is_shifted_data:
                        self.dirname="arcsim_square_trajectories_test_shifted" 
                    else:
                        self.dirname="arcsim_square_trajectories_test"
            else:
                if self.is_train: 
                    self.dirname = "output_square_coarse_minlen008_interp_500"
                    self.dirname_finereso = "output_square_coarse_minlen008_interp_500"
                else:
                    self.dirname = "output_square_coarse_minlen008_test_interp_50"
                    self.dirname_finereso = "output_square_coarse_minlen008_test_interp_50"
            # self.dirname_finereso = "output_square_coarse_minlen008_interp_500"
        else:
            raise
        self.show_missing_files = show_missing_files
        self.time_stamps_effective = (self.time_stamps - self.t_cushion_input - self.t_cushion_output) // self.time_interval
        print("root: ", self.root)
        super(ArcsimMesh, self).__init__(self.root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.root + self.dirname]

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dirname)

    @property
    def processed_dir_finereso(self):
        return os.path.join(self.root, self.dirname_finereso)


    @property
    #??????????????? WHAT IS 1000 ??????????????????#
    def processed_file_names(self):
        return ["traj_{:06d}/{:04d}_00.obj".format(k, i) for k in range(self.n_simu) for i in range(1000, 1000 + self.time_stamps)]

    def _process(self):
        import warnings
        from typing import Any, List
        from torch_geometric.data.makedirs import makedirs
        def _repr(obj: Any) -> str:
            if obj is None:
                return 'None'
                return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

        def files_exist(files: List[str]) -> bool:
            # NOTE: We return `False` in case `files` is empty, leading to a
            # re-processing of files on every instantiation.
            return len(files) != 0 and all([os.path.exists(f) for f in files])

        f = os.path.join(self.processed_dir, 'pre_transform.pt')
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"sure to delete '{self.processed_dir}' first")

        f = os.path.join(self.processed_dir, 'pre_filter.pt')
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in the "
                "pre-processed version of this dataset. If you want to make "
                "use of another pre-fitering technique, make sure to delete "
                "'{self.processed_dir}' first")

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        if self.show_missing_files:
            missing_files = [file for file in self.processed_paths if not os.path.exists(file)]
            print("Missing files:")
            pp.pprint(sorted(missing_files))

        print('Processing...')

        makedirs(self.processed_dir)
        self.process()

        path = os.path.join(self.processed_dir, 'pre_transform.pt')
        if not os.path.isfile(path):
            torch.save(_repr(self.pre_transform), path)
        path = os.path.join(self.processed_dir, 'pre_filter.pt')
        if not os.path.isfile(path):
            torch.save(_repr(self.pre_filter), path)

        print('Done!')

    def yield_f_file(self, traj_id, time_id, dataset="arcsimmesh_square"):
        """ Generate data of faces.
        Args:
            traj_id: index of trajectory folder.
            time_id: index of time step.
            
        Returns:
            list of faces.        
        """
        if not dataset == "arcsimmesh_square":
            f = open(os.path.join(self.processed_dir_finereso, "traj_{:06d}/{:04d}_00.obj".format(traj_id, time_id)))
        else:
            f = open(os.path.join(self.processed_dir, "traj_{:06d}/{:04d}_00.obj".format(traj_id, time_id)))
        buf = f.read()
        f.close()
        for b in buf.split('\n'):
            if b.startswith('f '):
                triangles = b.split(' ')[1:]
                # -1 as .obj is base 1 but the Data class expects base 0 indices
                yield [int(t.split("/")[0]) - 1 for t in triangles]

    def yield_v_file(self, traj_id, time_id, dataset="arcsimmesh_square"):
        """ Generate data of vertices.
        Args:
            traj_id: index of trajectory folder.
            time_id: index of time step.
            
        Returns:
            list of vertices. Each vertex is represented by 3d vector.       
        """
        if not dataset == "arcsimmesh_square":
            f = open(os.path.join(self.processed_dir_finereso, "traj_{:06d}/{:04d}_00.obj".format(traj_id, time_id)))
        else:
            f = open(os.path.join(self.processed_dir, "traj_{:06d}/{:04d}_00.obj".format(traj_id, time_id)))
        buf = f.read()
        f.close()
        for b in buf.split('\n'):
            if b.startswith('v '):
                yield [np.float(x) for x in b.split(" ")[1:]]

    def yield_ny_file(self, traj_id, time_id, dataset="arcsimmesh_square"):
        """ Generate data of 2d mesh as 3d subspace.
        Args:
            traj_id: index of trajectory folder.
            time_id: index of time step.
            
        Returns:
            list of 3d vectors. All the vectors are essentially in a 2d space:
            The 3rd corrdinate is 0.
        """
        if not dataset == "arcsimmesh_square":
            f = open(os.path.join(self.processed_dir_finereso, "traj_{:06d}/{:04d}_00.obj".format(traj_id, time_id)))
        else:
            f = open(os.path.join(self.processed_dir, "traj_{:06d}/{:04d}_00.obj".format(traj_id, time_id)))
        buf = f.read()
        f.close()
        for b in buf.split('\n'):
            if b.startswith('ny '):
                yield [np.float(x) for x in b.split(" ")[1:]]

    def yield_nv_file(self, traj_id, time_id, dataset="arcsimmesh_square"):
        """ Generate data of normal vectors.
        Args:
            traj_id: index of trajectory folder.
            time_id: index of time step.
            
        Returns:
            list of 3d normal vectors.
        """
        if not dataset == "arcsimmesh_square":
            f = open(os.path.join(self.processed_dir_finereso, "traj_{:06d}/{:04d}_00.obj".format(traj_id, time_id)))
        else:
            f = open(os.path.join(self.processed_dir, "traj_{:06d}/{:04d}_00.obj".format(traj_id, time_id)))
        buf = f.read()
        f.close()
        for b in buf.split('\n'):
            if b.startswith('nv '):
                yield [np.float(x) for x in b.split(" ")[1:]]
                
    def get_edge_index(self, traj_id, time_id, dataset="arcsimmesh_square"):
        """ Generate edge index of a mesh.
        Args:
            traj_id: index of trajectory folder.
            time_id: index of time step.
            
        Returns:
            edge_index: `torch.tensor` of shape [2*number of edgex, 2].
        """
        if not dataset == "arcsimmesh_square":
            edge_index_filename = os.path.join(self.processed_dir_finereso, "edge_index_{:06d}/{:04d}_00.obj".format(traj_id, time_id))
        else:
            edge_index_filename = os.path.join(self.processed_dir, "edge_index_{:06d}_{:04d}.p".format(traj_id, time_id))
        if os.path.isfile(edge_index_filename) and os.path.isfile(mask_valid_filename):
            edge_index = pickle.load(open(edge_index_filename, "rb"))
            return edge_index # , mask_valid

        edge_set = set()
        # Remove redundant edges by using set
        for v in self.yield_f_file(traj_id, time_id, dataset=dataset):
            edge_set.add((v[0], v[1]))
            edge_set.add((v[1], v[0]))
            edge_set.add((v[0], v[2]))
            edge_set.add((v[2], v[0]))
            edge_set.add((v[1], v[2]))
            edge_set.add((v[2], v[1]))

        edge_list = [list(edge_tup) for edge_tup in list(edge_set)]
        edge_index = torch.LongTensor(edge_list).T
        # pickle.dump(edge_index, open(edge_index_filename, "wb"))
        return edge_index        

    def get_vpos(self, traj_id, time_id, dataset="arcsimmesh_square"):
        """ Generate numpy array of 3d world coordinates.
        Args:
            traj_id: index of trajectory folder.
            time_id: index of time step.
            
        Returns:
            `numpy.array` of shape [num of vertices, 3].
        """
        vertices = []
        for v in self.yield_v_file(traj_id, time_id, dataset=dataset):
            vertices.append(v)

        return np.array(vertices)
                 
    def get_mesh_vpos(self, traj_id, time_id, dataset="arcsimmesh_square"):
        """ Generate numpy array of mesh data as 2d supspace in 3d space.
        Args:
            traj_id: index of trajectory folder.
            time_id: index of time step.
            
        Returns:
            `numpy.array` of shape [num of vertices, 3].
        """
        vertices = []
        for v in self.yield_ny_file(traj_id, time_id, dataset=dataset):
            vertices.append(v)

        return np.array(vertices)
                 
    def process(self):
        # Does not have effect for now:
        if self.is_train:
            pass
            #for i in range(self.n_simu):
            #    os.system('python {}/solver_in_the_loop/karman-2d/karman.py -o karman-fdt-hires-set -r 128 -l 100 --re `echo $(( 10000 * 2**({}+4) ))` --gpu "-1" --seed 0 --thumb;'.format(get_root_dir(), i))
        else:
            pass
            #for i in range(self.n_simu):
            #    os.system('python {}/solver_in_the_loop/karman-2d/karman.py -o karman-fdt-hires-testset -r 128 -l 100 --re `echo $(( 10000 * 2**({}+3) * 3 ))` --gpu "-1" --seed 0 --thumb'.format(get_root_dir(), i))

    def len(self):
        # pdb.set_trace()
        if self.is_traj:
            if self.traj_len is None:
                return self.n_simu
            else:
                time_stamps_effective = (self.time_stamps - self.t_cushion_input) // self.traj_len
                return self.n_simu * time_stamps_effective
        else:
            return ((self.time_stamps - self.t_cushion_input - self.t_cushion_output) // self.time_interval) * self.n_simu
        

    def generate_baryweight(self, tarvers, vers, ofaces):
        import dolfin
        
        ofaces = np.array(ofaces)
        tmesh = dolfin.Mesh()
        editor = dolfin.MeshEditor()
        editor.open(tmesh, 'triangle', 2, 2)
        editor.init_vertices(vers.shape[0])
        for i in range(vers.shape[0]):
            editor.add_vertex(i, vers[i])
        editor.init_cells(ofaces.shape[0])
        for f in range(ofaces.shape[0]):
            editor.add_cell(f, ofaces[f])
        editor.close()
        
        bvh_tree = tmesh.bounding_box_tree()
        
        faces = []
        weights = []
        for query in tarvers:
            face = bvh_tree.compute_first_entity_collision(dolfin.Point(query))
            while (tmesh.num_cells() <= face):
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
            faces.append(face)
            face_coords = tmesh.coordinates()[tmesh.cells()[face]]
            mat = face_coords.T[:,[0,1]] - face_coords.T[:,[2,2]]
            const = query - face_coords[2,:]
            weight = np.linalg.solve(mat, const)
            final_weights = np.concatenate([weight, np.ones(1) - weight.sum()], axis=-1)
            weights.append(final_weights)
        return faces, weights, tmesh

    def generate_barycentric_interpolated_data(self, vers, ofaces, refvec, tarvers):
        faces, weights, mesh = self.generate_baryweight(tarvers, vers, ofaces)
        indices = mesh.cells()[faces].astype('int64')
        fweights = torch.tensor(np.array(weights)).to(torch.float32)
        return torch.matmul(fweights, refvec[indices,:]).diagonal().T

    def get(self, idx):
        """ Transform original mesh data to pytorch data.
                
        Args:
            idx: original identification number of data folder and file.
            
        Returns:
            data: `torch.tensor` of mesh data. `data.x` is `torch.tensor` of vertex features in
            a mesh and the shape is [number of nodes, 9]. Each feature consists of 2d coordinates,
            world coordinates, and normals. `data.y` is `torch.tensor` of vertex features for
            target data. The shape is [number of nodes, 9], but can be different from `data.x`.
            `data.edge_index` is the edge indices of mesh data. It is of `torch.tensor` and the 
            shape is [2, 2*number of edges]. `data.xfaces` is `torch.tensor` of faces of a mesh.
            The shape is [3, number of faces]. Each column corresponds to a face of a mesh and 
            represented by its vertices.
        """        
        sim_id, time_id = divmod(idx, self.time_stamps_effective)
        
        # pdb.set_trace()
        xfaces = torch.FloatTensor([face for face in self.yield_f_file(sim_id, time_id * self.time_interval + self.t_cushion_input)]).T        
        xworldp_list = []
        xmeshp_list = []
        xface_list = []
        for j in range(-self.input_steps * self.time_interval, 0, self.time_interval):
            xworldp_list.append(self.get_vpos(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j))
            xmeshp_list.append(self.get_mesh_vpos(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j))  
            xface_list.append([face for face in self.yield_f_file(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j)])
                
        yworldp_list = []
        ymeshp_list = []
        yface_list = []
        yedge_index = []
        if hasattr(self, 'dirname_finereso'):
            dataset_name = self.dirname_finereso
        else:
            dataset_name = "arcsimmesh_square"
        for j in range(0, self.output_steps * self.time_interval, self.time_interval):
            yworldp_list.append(self.get_vpos(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j, dataset=dataset_name))
            ymeshp_list.append(self.get_mesh_vpos(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j, dataset=dataset_name))
            yface_list.append([face for face in self.yield_f_file(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j, dataset=dataset_name)])
            yedge_index.append(self.get_edge_index(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j, dataset=dataset_name))        
        # if hasattr(self, 'dirname_finereso'):
        #     for j in range(0, self.output_steps * self.time_interval, self.time_interval):
        #         yworldp_list.append(self.get_vpos(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j, dataset="arcsim_square_trajectories_test_shifted"))
        #         ymeshp_list.append(self.get_mesh_vpos(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j, dataset="arcsim_square_trajectories_test_shifted"))
        #         yface_list.append([face for face in self.yield_f_file(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j, dataset="arcsim_square_trajectories_test_shifted")])
        #         yedge_index.append(self.get_edge_index(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j, dataset="arcsim_square_trajectories_test_shifted"))
        # else:
        #     for j in range(0, self.output_steps * self.time_interval, self.time_interval):
        #         yworldp_list.append(self.get_vpos(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j))
        #         ymeshp_list.append(self.get_mesh_vpos(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j))
        #         yface_list.append([face for face in self.yield_f_file(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j)])
        #         yedge_index.append(self.get_edge_index(sim_id, time_id * self.time_interval + self.t_cushion_input + self.time_interval + j))
                
        # # Query vertices
        # anker_2dvers = xmeshp_list[-1][:,:2]
        
        # Generate Hisotry 
        history = []
        shift_history = []
        shift_faces = []
        for i in range(len(xmeshp_list)):
            # tarmesh = self.to_dolfin_mesh(xmeshp_list[i][:,:2], xface_list[i])
            # tar_bvhtree = tarmesh.bounding_box_tree()
            refvec = torch.cat(
                [torch.tensor(xmeshp_list[i]), torch.tensor(xworldp_list[i])],
                dim=-1).to(torch.float32)
            history.append(refvec)
            shift_history.append(refvec)
            shift_faces.append(xface_list[i])
        x_feature = torch.cat([torch.tensor(xmeshp_list[-1]), torch.tensor(xworldp_list[-1])], dim=-1).to(torch.float32)#[:,3:]
        x_catpos = torch.stack([x_feature[:,3:]], dim=-2).to(torch.float32)

        # Generate barycentric interpolated meshes for edge index
        # Will be used for labels
        targets = []
        yfeatures = []
        for i in range(len(ymeshp_list)):
            #tarmesh = self.to_dolfin_mesh(ymeshp_list[i][:,:2], yface_list[i])
            #tar_bvhtree = tarmesh.bounding_box_tree()
            yrefvec = torch.cat(
                [torch.tensor(ymeshp_list[i]), torch.tensor(yworldp_list[i])],
                dim=-1).to(torch.float32)
            yfeatures.append(yrefvec)
            shift_history.append(yrefvec)
            shift_faces.append(yface_list[i])
            if i == 0:
                past_queries = xmeshp_list[-1][:,:2]
            else:
                past_queries = ymeshp_list[i-1][:,:2]
            # pdb.set_trace()
            targets.append(self.generate_barycentric_interpolated_data(ymeshp_list[i][:,:2], yface_list[i], yrefvec, past_queries))
        x_edge_index = self.get_edge_index(sim_id, time_id * self.time_interval + self.t_cushion_input)

        hist_indices = []
        hist_weights = []
        hist_pos = []
        node_dims = []
        for k in range(1, len(shift_history)):
            vers = shift_history[k-1].cpu().numpy()
            faces = np.array(shift_faces[k-1]).astype(np.int64)
            tarvers = shift_history[k].cpu().numpy()
            node_dims.append(torch.tensor([[tarvers.shape[0]]]))
            faces, weights, mesh = self.generate_baryweight(tarvers[:,:2], vers[:,:2], faces)
            bvhtree = mesh.bounding_box_tree()
            indices = mesh.cells()[faces].astype('int64')
            fweights = torch.tensor(np.array(weights), dtype=torch.float32)
            hist_indices.append(indices)
            hist_weights.append(fweights)
            # pdb.set_trace()
            hist_pos.append(torch.matmul(fweights, torch.tensor(vers[indices,:], dtype=torch.float32)).diagonal().T[:,3:])
        
        bary_indices = []
        bary_weights = []
        for k in range(1, len(ymeshp_list)+1):
            # pdb.set_trace()
            if k-1 == 0:
                vers = x_feature.cpu().numpy()
                faces = xfaces.T.cpu().numpy().astype(np.int64)
            else:
                vers = yfeatures[k-2].cpu().numpy()
                faces = np.array(yface_list[k-2]).astype(np.int64)
            tarvers = yfeatures[k-1].cpu().numpy()
            faces, weights, mesh = self.generate_baryweight(tarvers[:,:2], vers[:,:2], faces)
            bvhtree = mesh.bounding_box_tree()
            indices = mesh.cells()[faces].astype('int64')
            fweights = torch.tensor(np.array(weights), dtype=torch.float32)
            bary_indices.append(indices)
            bary_weights.append(fweights)
            
        hessian_history = []
        for i in range(len(targets)):
            # pdb.set_trace()
            if i == 0:
                hessian_history.append(targets[i][:, 3:] - 2 * x_feature[:, 3:] + hist_pos[i])
            else:
                hessian_history.append(targets[i][:, 3:] - 2 * yfeatures[i-1][:, 3:] + hist_pos[i])
                
        # edge attributes
        ## take subtraction between node features
        x_receiver = torch.gather(x_feature, 0, x_edge_index[0,:].unsqueeze(-1).repeat(1,x_feature.shape[1]))
        x_sender = torch.gather(x_feature, 0, x_edge_index[1,:].unsqueeze(-1).repeat(1,x_feature.shape[1]))
        relative_pos = x_receiver - x_sender
        # pdb.set_trace()
        edge_attr = torch.cat([relative_pos, relative_pos.abs()], dim=-1).to(torch.float32)

        # Construct data:
        data = Attr_Dict(
            node_feature={"n0": x_catpos.clone(),},
            node_dim={"n0": tuple(node_dims),},
            edge_attr={"n0": (edge_attr,),},
            node_label={"n0": yfeatures[0][:,3:],},
            yfeatures={"n0": tuple(yfeatures),},
            reind_yfeatures={"n0": tuple(yfeatures),},
            # shift_history={"n0": tuple(shift_history),}
            bary_indices={"n0": tuple(bary_indices),},
            bary_weights={"n0": tuple(bary_weights),},
            hist_indices={"n0": tuple(hist_indices),},
            hist_weights={"n0": tuple(hist_weights),},            
            history={"n0": tuple(history),},
            xface_list={"n0": tuple(xface_list),},
            # y_back={"n0": tuple(targets)},
            y_tar={"n0": tuple(targets)},
            y_back={"n0": tuple(hessian_history)},
            # x_pos={"n0": torch.empty(x_catpos.shape[0],0)},
            x_pos={"n0": x_catpos.clone()[:,:2],},
            edge_index={("n0", "0", "n0"): x_edge_index},
            xfaces={"n0": xfaces.to(torch.int64)},
            yface_list={"n0": tuple(yface_list)},
            yedge_index={"n0": tuple(yedge_index)},
            x_bdd = {"n0": torch.empty(x_catpos.shape[0],0)},
            original_shape=(("n0", self.original_shape),),
            dyn_dims=(("n0", self.dyn_dims),),
            #dyn_dims=self.dyn_dims,
            compute_func=(0, None),
            grid_keys=("n0",),
            part_keys=(),
            time_step=time_id,
            sim_id=sim_id,
            time_interval=self.time_interval,
            cushion_input=self.t_cushion_input,
        )
        
        if self.dataset.startswith("arcsimmesh_square_annotated"):
            onehot_list = []
            kinematics_list = []
            for k in range(len(ymeshp_list)):
                if k == 0:
                    onehot = torch.zeros((x_feature.shape[0],2), dtype=torch.float32)
                    kinematics = torch.zeros((x_feature.shape[0],3), dtype=torch.float32)
                    kinematics[:4] = targets[k][:4, 3:] - x_feature[:4,3:]
                else:
                    onehot = torch.zeros((ymeshp_list[k-1].shape[0],2), dtype=torch.float32)
                    kinematics = torch.zeros((ymeshp_list[k-1].shape[0],3), dtype=torch.float32)
                    kinematics[:4] = targets[k][:4, 3:] - yfeatures[k-1][:4,3:]
                onehot[:4] = torch.tensor([1., 0.])
                onehot[4:] = torch.tensor([0., 1.])
                onehot_list.append(onehot)
                kinematics[4:] = torch.tensor([0., 0., 0.])
                kinematics_list.append(kinematics)
            
            additional_ymeshnode = ymeshp_list[len(ymeshp_list)-1]
            addonehot = torch.zeros((additional_ymeshnode.shape[0],2), dtype=torch.float32)
            addonehot[:4] = torch.tensor([1., 0.])
            addonehot[4:] = torch.tensor([0., 1.])
            onehot_list.append(addonehot)
            
            data.onehot_list = {"n0": tuple(onehot_list),}
            data.kinematics_list = {"n0": tuple(kinematics_list),}
            
        return data


# In[106]:


def get_2d_data_pred(state_preds, step, data, **kwargs):
    """Get a new mesh Data from the state_preds at step "step" (e.g. 0,1,2....).
    Here we assume that the mesh does not change.

    Args:
        state_preds: has shape of [n_nodes, n_steps, feature_size:temporal_bundle_steps]
        data: a Deepsnap Data object
        **kwargs: keys for which need to use the value instead of data's.
    """
    is_deepsnap = hasattr(data, "node_feature")
    is_list = isinstance(state_preds, list)
    # if is_list:
    #     assert len(state_preds[0].shape) == 3
    #     state_pred = state_preds[step].reshape(state_preds[step].shape[0], -1, data.node_feature["n0"].shape[-1])
    # else:
    #     assert isinstance(state_preds, torch.Tensor)
    #     state_pred = state_preds[...,step,:].reshape(state_preds.shape[0], state_preds.shape[-1], 1)
    state_pred = state_preds[step]
        
    data_pred = Attr_Dict(
        node_feature={"n0": state_pred},
        node_label={"n0": None},
        x_pos={"n0": kwargs["x_pos"] if "x_pos" in kwargs else data.node_pos["n0"] if is_deepsnap else data.x_pos},
        x_bdd={"n0": kwargs["x_bdd"] if "x_bdd" in kwargs else data.x_bdd["n0"] if is_deepsnap else data.x_bdd},
        xfaces={"n0": data.xfaces["n0"],},
        edge_index={("n0","0","n0"): kwargs["edge_index"] if "edge_index" in kwargs else data.edge_index[("n0","0","n0")] if is_deepsnap else data.edge_index},
        # mask={"n0": data.mask["n0"] if is_deepsnap else data.mask},
        # param={"n0": data.param["n0"][:1] if is_deepsnap else data.param[:1]},
        original_shape=to_tuple_shape(data.original_shape),
        dyn_dims=to_tuple_shape(data.dyn_dims),
        compute_func=to_tuple_shape(data.compute_func),
        grid_keys=("n0",),
        part_keys=(),
        # dataset=to_tuple_shape(data.dataset),
        batch=kwargs["batch"] if "batch" in kwargs else data.batch ,
        time_step=data.time_step, 
        sim_id=data.sim_id,
        time_interval={"n0": data["time_interval"]}, 
        cushion_input={"n0": data["cushion_input"]},
        bary_weights={"n0": data["bary_weights"]},
        bary_indices={"n0": data["bary_indices"]},
        hist_weights={"n0": data["hist_weights"]},
        hist_indices={"n0": data["hist_indices"]},
        yedge_index={"n0": data["yedge_index"]},
        y_back={"n0": data["y_back"]},
        y_tar={"n0": data["y_tar"]},
        yface_list={"n0": data["yface_list"]["n0"]},
        history={"n0": data["history"]["n0"],},
        yfeatures={"n0": data["yfeatures"]["n0"],},
        node_dim={"n0": data["node_dim"]["n0"],},
        xface_list={"n0": data["xface_list"]["n0"],},
        reind_yfeatures={"n0": data["reind_yfeatures"]["n0"],},
        batch_history={"n0": data["batch_history"]["n0"],},
    )
    if "edge_attr" in data:
        data_pred.edge_attr = {"n0": kwargs["edge_attr"] if "edge_attr" in kwargs else data.edge_attr["n0"] if is_deepsnap else data.edge_attr}
    if "onehot_list" in data:
        data_pred.onehot_list = {"n0": data["onehot_list"]["n0"],}
        data_pred.kinematics_list = {"n0": data["kinematics_list"]["n0"],}   
    
    return data_pred


# In[47]:


if __name__ == "__main__":
    dataset = ArcsimMesh(
        dataset="arcsimmesh_square",
        input_steps=1,
        output_steps=1,
        time_interval=1,
        is_train=True,
        show_missing_files=False,
    )


# In[107]:


# if __name__ == "__main__":
#     dataset = ArcsimMesh(
#         dataset="arcsimmesh_square_annotated_coarse_minlen008_interp_500",
#         input_steps=2,
#         output_steps=20,
#         time_interval=2,
#         is_train=False,
#         use_fineres_data=False,
#         show_missing_files=False,
#     )


# In[108]:


# dataset.len()


# In[109]:


# dataset.get(4349)["node_feature"]["n0"][:10,:]
# dataset.get(43499)["node_feature"]["n0"][:10,:]
# dataset.get(43500)["node_feature"]["n0"][:10,:]


# In[110]:


# dataset.get(0)["node_feature"]["n0"][:10,:]
#dataset.get(33999)["node_feature"]["n0"][:10,:]
#dataset.get(43500)["node_feature"]["n0"][:10,:]


# In[111]:


# import matplotlib.pyplot as plt
# import numpy as np

# outmesh = dataset.get(60)
# print(pos0.shape)
# pos0 = outmesh["node_feature"]["n0"][:,0,:].detach().cpu().numpy()
# faces0 = outmesh["xfaces"]["n0"].permute(1,0).detach().cpu().numpy()

# fig = plt.figure(figsize=(18,12))

# ax0= fig.add_subplot(121, projection='3d')
# #ax0.set_title('after evolution + remeshing')
# ax0.set_xlim([-0.6, 0.6])
# ax0.set_ylim([-0.6, 0.6])
# ax0.set_zlim([-0.4, 0.2])
# ax0.view_init(50, 20)
# print(pos0.shape)
# print(faces0.shape)
# ax0.plot_trisurf(pos0[:, 0], pos0[:, 1], faces0, pos0[:, 2], shade=True, linewidth = 0.3, edgecolor = 'grey', color="ghostwhite")

# ax0= fig.add_subplot(122, projection='3d')
# #ax0.set_title('after evolution + remeshing')
# ax0.set_xlim([-0.6, 0.6])
# ax0.set_ylim([-0.6, 0.6])
# ax0.set_zlim([-0.4, 0.2])
# ax0.view_init(10, 20)
# print(pos0.shape)
# print(faces0.shape)
# ax0.plot_trisurf(pos0[:, 0], pos0[:, 1], faces0, pos0[:, 2], shade=True, linewidth = 0.3, edgecolor = 'grey', color="ghostwhite")
# plt.show()


# In[112]:


# if __name__ == "__main__":
#     dataset_train = ArcsimMesh(
#         dataset="arcsimmesh_square_annotated_coarse_minlen008_interp_500",
#         input_steps=2,
#         output_steps=20,
#         time_interval=2,
#         is_train=True,
#         use_fineres_data=False,
#         show_missing_files=False,
#     )


# In[113]:


# import matplotlib.pyplot as plt
# import numpy as np

# outmesh = dataset_train.get(60)
# print(pos0.shape)
# pos0 = outmesh["node_feature"]["n0"][:,0,:].detach().cpu().numpy()
# faces0 = outmesh["xfaces"]["n0"].permute(1,0).detach().cpu().numpy()

# fig = plt.figure(figsize=(18,12))

# ax0= fig.add_subplot(121, projection='3d')
# #ax0.set_title('after evolution + remeshing')
# ax0.set_xlim([-0.6, 0.6])
# ax0.set_ylim([-0.6, 0.6])
# ax0.set_zlim([-0.4, 0.2])
# ax0.view_init(50, 20)
# print(pos0.shape)
# print(faces0.shape)
# ax0.plot_trisurf(pos0[:, 0], pos0[:, 1], faces0, pos0[:, 2], shade=True, linewidth = 0.3, edgecolor = 'grey', color="ghostwhite")

# ax0= fig.add_subplot(122, projection='3d')
# #ax0.set_title('after evolution + remeshing')
# ax0.set_xlim([-0.6, 0.6])
# ax0.set_ylim([-0.6, 0.6])
# ax0.set_zlim([-0.4, 0.2])
# ax0.view_init(10, 20)
# print(pos0.shape)
# print(faces0.shape)
# ax0.plot_trisurf(pos0[:, 0], pos0[:, 1], faces0, pos0[:, 2], shade=True, linewidth = 0.3, edgecolor = 'grey', color="ghostwhite")
# plt.show()


# In[114]:


# dataset[0]["yfeatures"]["n0"][0] - dataset_train[0]["yfeatures"]["n0"][0]


# In[ ]:




