#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import math
import numpy as np
from torch import nn
# from torchmeta.modules import MetaModule
from collections import OrderedDict
import copy
import torch.nn.functional as F


def get_conv_func(pos_dim, *args, **kwargs):
    if "reg_type_list" in kwargs:
        reg_type_list = kwargs.pop("reg_type_list")
    else:
        reg_type_list = None
    if pos_dim == 1:
        conv = nn.Conv1d(*args, **kwargs)
    elif pos_dim == 2:
        conv = nn.Conv2d(*args, **kwargs)
    elif pos_dim == 3:
        conv = nn.Conv3d(*args, **kwargs)
    else:
        raise Exception("The pos_dim can only be 1, 2 or 3!")
    if reg_type_list is not None:
        if "snn" in reg_type_list:
            conv = SpectralNorm(conv)
        elif "snr" in reg_type_list:
            conv = SpectralNormReg(conv)
    return conv


def get_conv_trans_func(pos_dim, *args, **kwargs):
    if "reg_type_list" in kwargs:
        reg_type_list = kwargs.pop("reg_type_list")
    else:
        reg_type_list = None
    if pos_dim == 1:
        conv_trans = nn.ConvTranspose1d(*args, **kwargs)
    elif pos_dim == 2:
        conv_trans = nn.ConvTranspose2d(*args, **kwargs)
    elif pos_dim == 3:
        conv_trans = nn.ConvTranspose3d(*args, **kwargs)
    else:
        raise Exception("The pos_dim can only be 1, 2 or 3!")
     # The weight's output dim=1 for ConvTranspose
    if reg_type_list is not None:
        if "snn" in reg_type_list:
            conv_trans = SpectralNorm(conv_trans, dim=1)
        elif "snr" in reg_type_list:
            conv_trans = SpectralNormReg(conv_trans, dim=1)
    return conv_trans


# ### Spectral Norm:

# In[ ]:


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, dim=0):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.dim = dim
        if not self._made_params():
            self._make_params()

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        w_mat = self.reshape_weight_to_matrix(w)

        height = w_mat.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w_mat.data), u.data))
            u.data = l2normalize(torch.mv(w_mat.data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w_mat.mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        w_mat = self.reshape_weight_to_matrix(w)

        height = w_mat.shape[0]
        width = w_mat.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        if self.training:
            self._update_u_v()
        else:
            setattr(self.module, self.name, getattr(self.module, self.name + "_bar") / 1)
        return self.module.forward(*args)


# ### SpectralNormReg:

# In[ ]:


class SpectralNormReg(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, dim=0):
        super(SpectralNormReg, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.dim = dim
        if not self._made_params():
            self._make_params()

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_snreg(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        w_mat = self.reshape_weight_to_matrix(w)

        height = w_mat.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w_mat.data), u.data))
            u.data = l2normalize(torch.mv(w_mat.data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w_mat.mv(v))
        self.snreg = sigma.square() / 2
        setattr(self.module, self.name, w / 1)  # Here the " / 1" is to prevent state_dict() to record self.module.weight

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        w_mat = self.reshape_weight_to_matrix(w)

        height = w_mat.shape[0]
        width = w_mat.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self.compute_snreg()
        return self.module.forward(*args)


# ### Hessian regularization:

# In[ ]:


def get_Hessian_penalty(
    G,
    z,
    mode,
    k=2,
    epsilon=0.1,
    reduction=torch.max,
    return_separately=False,
    G_z=None,
    is_nondimensionalize=False,
    **G_kwargs
):
    """
    Adapted from https://github.com/wpeebles/hessian_penalty/ (Peebles et al. 2020).
    Note: If you want to regularize multiple network activations simultaneously, you need to
    make sure the function G you pass to hessian_penalty returns a list of those activations when it's called with
    G(z, **G_kwargs). Otherwise, if G returns a tensor the Hessian Penalty will only be computed for the final
    output of G.
    
    Args:
        G: Function that maps input z to either a tensor or a list of tensors (activations)
        z: Input to G that the Hessian Penalty will be computed with respect to
        mode: choose from "Hdiag", "Hoff" or "Hall", specifying the scope of Hessian values to perform sum square on. 
                "Hall" will be the sum of "Hdiag" (for diagonal elements) and "Hoff" (for off-diagonal elements).
        k: Number of Hessian directions to sample (must be >= 2)
        epsilon: Amount to blur G before estimating Hessian (must be > 0)
        reduction: Many-to-one function to reduce each pixel/neuron's individual hessian penalty into a final loss
        return_separately: If False, hessian penalties for each activation output by G are automatically summed into
                              a final loss. If True, the hessian penalties for each layer will be returned in a list
                              instead. If G outputs a single tensor, setting this to True will produce a length-1
                              list.
    :param G_z: [Optional small speed-up] If you have already computed G(z, **G_kwargs) for the current training
                iteration, then you can provide it here to reduce the number of forward passes of this method by 1
    :param G_kwargs: Additional inputs to G besides the z vector. For example, in BigGAN you
                     would pass the class label into this function via y=<class_label_tensor>
    :return: A differentiable scalar (the hessian penalty), or a list of hessian penalties if return_separately is True
    """
    if G_z is None:
        G_z = G(z, **G_kwargs)
    rademacher_size = torch.Size((k, *z.size()))  # (k, N, z.size())
    if mode == "Hall":
        loss_diag = get_Hessian_penalty(G=G, z=z, mode="Hdiag", k=k, epsilon=epsilon, reduction=reduction, return_separately=return_separately, G_z=G_z, **G_kwargs)
        loss_offdiag = get_Hessian_penalty(G=G, z=z, mode="Hoff", k=k, epsilon=epsilon, reduction=reduction, return_separately=return_separately, G_z=G_z, **G_kwargs)
        if return_separately:
            loss = []
            for loss_i_diag, loss_i_offdiag in zip(loss_diag, loss_offdiag):
                loss.append(loss_i_diag + loss_i_offdiag)
        else:
            loss = loss_diag + loss_offdiag
        return loss
    elif mode == "Hdiag":
        xs = epsilon * complex_rademacher(rademacher_size, device=z.device)
    elif mode == "Hoff":
        xs = epsilon * rademacher(rademacher_size, device=z.device)
    else:
        raise
    second_orders = []

    if mode == "Hdiag" and isinstance(G, nn.Module):
        # Use the complex64 dtype:
        dtype_ori = next(iter(G.parameters())).dtype
        G.type(torch.complex64)
    if isinstance(G, nn.Module):
        G_wrapper = get_listified_fun(G)
        G_z = listity_tensor(G_z)
    else:
        G_wrapper = G

    for x in xs:  # Iterate over each (N, z.size()) tensor in xs
        central_second_order = multi_layer_second_directional_derivative(G_wrapper, z, x, G_z, epsilon, **G_kwargs)
        second_orders.append(central_second_order)  # Appends a tensor with shape equal to G(z).size()
    loss = multi_stack_metric_and_reduce(second_orders, mode, reduction, return_separately)  # (k, G(z).size()) --> scalar

    if mode == "Hdiag" and isinstance(G, nn.Module):
        # Revert back to original dtype:
        G.type(dtype_ori)

    if is_nondimensionalize:
        # Multiply a factor ||z||_2^2 so that the result is dimensionless:
        factor = z.square().mean()
        if return_separately:
            loss = [ele * factor for ele in loss]
        else:
            loss = loss * factor
    return loss


def listity_tensor(tensor):
    """Turn the output features (except for the first batch dimension) of a function into a list

    Args:
        tensor: has shape [B, d1, d2, ...]
    """
    batch_size = tensor.shape[0]
    shape = tensor.shape[1:]
    tensor_reshape = tensor.reshape(batch_size, -1)
    tensor_listify = [tensor_reshape[:, i] for i in range(tensor_reshape.shape[1])]
    return tensor_listify


def get_listified_fun(G):
    def fun(z, **Gkwargs):
        G_out = G(z, **Gkwargs)
        return listity_tensor(G_out)
    return fun


def rademacher(shape, device='cpu'):
    """Creates a random tensor of size [shape] under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    return torch.randint(2, size=shape, device=device).float() * 2 - 1


def complex_rademacher(shape, device='cpu'):
    """Creates a random tensor of size [shape] with (P(x=1) == P(x=-1) == P(x=1j) == P(x=-1j) == 0.25)"""
    collection = torch.from_numpy(np.array([1., -1, 1j, -1j])).type(torch.complex64).to(device)
    x = x.randint(4, size=shape, device=device)  # Creates random tensor of 0, 1, 2, 3
    return collection[x]  # Map tensor of 0, 1, 2, 3 to 1, -1, 1j, -1j


def multi_layer_second_directional_derivative(G, z, x, G_z, epsilon, **G_kwargs):
    """Estimates the second directional derivative of G w.r.t. its input at z in the direction x"""
    G_to_x = G(z + x, **G_kwargs)
    G_from_x = G(z - x, **G_kwargs)

    G_to_x = listify(G_to_x)
    G_from_x = listify(G_from_x)
    G_z = listify(G_z)

    eps_sqr = epsilon ** 2
    sdd = [(G2x - 2 * G_z_base + Gfx) / eps_sqr for G2x, G_z_base, Gfx in zip(G_to_x, G_z, G_from_x)]
    return sdd


def stack_metric_and_reduce(list_of_activations, mode, reduction=torch.max):
    """Equation (5) from the paper."""
    second_orders = torch.stack(list_of_activations)  # (k, N, C, H, W)
    if mode == "Hoff":
        tensor = torch.var(second_orders, dim=0, unbiased=True) / 2  # (N, C, H, W)
    elif mode == "Hdiag":
        tensor = torch.mean((second_orders ** 2).real, dim=0)
    else:
        raise
    penalty = reduction(tensor)  # (1,) (scalar)
    return penalty


def multi_stack_metric_and_reduce(sdds, mode, reduction=torch.max, return_separately=False):
    """Iterate over all activations to be regularized, then apply Equation (5) to each."""
    sum_of_penalties = 0 if not return_separately else []
    for activ_n in zip(*sdds):
        penalty = stack_metric_and_reduce(activ_n, mode, reduction)
        sum_of_penalties += penalty if not return_separately else [penalty]
    return sum_of_penalties


def listify(x):
    """If x is already a list, do nothing. Otherwise, wrap x in a list."""
    if isinstance(x, list):
        return x
    else:
        return [x]


def _test_hessian_penalty(mode, k=100):
    """
    A simple multi-layer test to verify the implementation.
    Function: G(z) = [z_0 * z_1 + z0 ** 2, z_0**2 * z_1 + 2 * z1 ** 2]
    The Hessian for the first value is [
        [2, 1],
        [1, 0],
    ]
    so the offdiagonal sum square is 2
    the diagonal sum square is 4.
    
    The Hessian for the second function value is: [
        [2 * z_1, 2 * z_0],
        [2 * z_0, 4],
    ]
    so the offdiagonal sum square is 8 * z_0**2
    the diagonal sum square is 16 + 4 * z_1**2.
    Ground Truth Hessian Penalty: [2, 8 * z_0**2]
    """
    batch_size = 10
    nz = 2
    z = torch.randn(batch_size, nz)
    def reduction(x): return x.abs().mean()
    def G(z): return [z[:, 0] * z[:, 1] + z[:, 0] ** 2, (z[:, 0] ** 2) * z[:, 1] + 2 * z[:, 1] ** 2]
    if mode == "Hdiag":
        ground_truth = [4, 16 + 4 * (z[:, 1] ** 2).mean().item()]
    elif mode == "Hoff":
        ground_truth = [2, reduction(8 * z[:, 0] ** 2).item()]
    elif mode == "Hall":
        ground_truth = [4+2, 16 + 4 * (z[:, 1] ** 2).mean().item() + reduction(8 * z[:, 0] ** 2).item()]
    else:
        raise
    # In this simple example, we use k=100 to reduce variance, but when applied to neural networks
    # you will probably want to use a small k (e.g., k=2) due to memory considerations.
    predicted = get_Hessian_penalty(G, z, mode=mode, G_z=None, k=k, reduction=reduction, return_separately=True)
    predicted = [p.item() for p in predicted]
    print('Ground Truth: %s' % ground_truth)
    print('Approximation: %s' % predicted)  # This should be close to ground_truth, but not exactly correct
    print('Difference: %s' % [str(100 * abs(p - gt) / gt) + '%' for p, gt in zip(predicted, ground_truth)])


## >>> functions for the MeshGraphNets:

def init_weights_requ(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1/math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277)/math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def init_weights_uniform(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def sine_init(m, w0=60):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input)/w0, np.sqrt(6/num_input)/w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1/num_input, 1/num_input)


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.
    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """
    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None

        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names

            else:
                key_escape = re.escape(key)
                key_re = re.compile(r'^{0}\.(.+)'.format(key_escape))

                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r'\1', k) for k in all_names if key_re.match(k) is not None]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn('Module `{0}` has no parameter corresponding to the '
                          'submodule named `{1}` in the dictionary `params` '
                          'provided as an argument to `forward()`. Using the '
                          'default parameters for this submodule. The list of '
                          'the parameters in `params`: [{2}].'.format(
                          self.__class__.__name__, key, ', '.join(all_names)),
                          stacklevel=2)
            return None

        return OrderedDict([(name, params[f'{key}.{name}']) for name in names])


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape)-2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class FirstSine(nn.Module):
    def __init__(self, w0=60):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class Sine(nn.Module):
    def __init__(self, w0=60):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class ReQU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace)

    def forward(self, input):
        # return torch.sin(np.sqrt(256)*input)
        return .5*self.relu(input)**2


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input)-self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input*torch.sigmoid(input)


def layer_factory(layer_type):
    layer_dict =         {
         'relu': (nn.ReLU(inplace=True), init_weights_normal),
         'leakyrelu': (nn.LeakyReLU(inplace=True), init_weights_normal),
         'requ': (ReQU(inplace=False), init_weights_requ),
         'sigmoid': (nn.Sigmoid(), None),
         'fsine': (Sine(), first_layer_sine_init),
         'sine': (Sine(), sine_init),
         'tanh': (nn.Tanh(), init_weights_xavier),
         'selu': (nn.SELU(inplace=True), init_weights_selu),
         'gelu': (nn.GELU(), init_weights_selu),
         'swish': (Swish(), init_weights_selu),
         'softplus': (nn.Softplus(), init_weights_normal),
         'msoftplus': (MSoftplus(), init_weights_normal),
         'elu': (nn.ELU(), init_weights_elu),
         'silu': (nn.SiLU(), init_weights_selu),
        }
    return layer_dict[layer_type]


class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=2, gaussian_pe=False, gaussian_variance=0.1):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(2*np.pi*gaussian_variance * torch.randn((num_encoding_functions*2), input_dim),
                                                 requires_grad=False)

        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1/self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.
        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).
        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx]*func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)



class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features,
                 outermost_linear=False, outmost_nonlinearity=None, nonlinearity='relu',
                 weight_init=None, w0=60, set_bias=None,
                 dropout=0.0, layer_norm=False,latent_dim=64,skip_connect=None):
        super().__init__()

        self.skip_connect = skip_connect
        self.latent_dim = latent_dim
        self.first_layer_init = None
        self.dropout = dropout

        if outmost_nonlinearity==None:
            outmost_nonlinearity = nonlinearity

        # Create hidden features list
        if not isinstance(hidden_features, list):
            num_hidden_features = hidden_features
            hidden_features = []
            for i in range(num_hidden_layers+1):
                hidden_features.append(num_hidden_features)
        else:
            num_hidden_layers = len(hidden_features)-1
        #print(f"net_size={hidden_features}")

        # Create the net
        #print(f"num_layers={len(hidden_features)}")
        if isinstance(nonlinearity, list):
            print(f"num_non_lin={len(nonlinearity)}")
            assert len(hidden_features) == len(nonlinearity), "Num hidden layers needs to "                                                               "match the length of the list of non-linearities"

            self.net = []
            self.net.append(nn.Sequential(
                nn.Linear(in_features, hidden_features[0]),
                layer_factory(nonlinearity[0])[0]
            ))
            for i in range(num_hidden_layers):
                if self.skip_connect==None:
                    self.net.append(nn.Sequential(
                        nn.Linear(hidden_features[i], hidden_features[i+1]),
                        layer_factory(nonlinearity[i+1])[0]
                    ))
                else:
                    if i+1 in self.skip_connect:
                        self.net.append(nn.Sequential(
                        nn.Linear(hidden_features[i]+self.latent_dim, hidden_features[i+1]),
                        layer_factory(nonlinearity[i+1])[0]
                    ))
                    else:
                        self.net.append(nn.Sequential(
                            nn.Linear(hidden_features[i], hidden_features[i+1]),
                            layer_factory(nonlinearity[i+1])[0]
                        ))

            if outermost_linear:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    layer_factory(nonlinearity[-1])[0]
                ))
        elif isinstance(nonlinearity, str):
            nl, weight_init = layer_factory(nonlinearity)
            outmost_nl, _ = layer_factory(outmost_nonlinearity)
            if(nonlinearity == 'sine'):
                first_nl = FirstSine()
                self.first_layer_init = first_layer_sine_init
            else:
                first_nl = nl

            if weight_init is not None:
                self.weight_init = weight_init

            self.net = []
            self.net.append(nn.Sequential(
                nn.Linear(in_features, hidden_features[0]),
                first_nl
            ))

            for i in range(num_hidden_layers):
                if(self.dropout > 0):
                    self.net.append(nn.Dropout(self.dropout))
                if self.skip_connect == None:
                    self.net.append(nn.Sequential(
                        nn.Linear(hidden_features[i], hidden_features[i+1]),
                        copy.deepcopy(nl)
                    ))
                else:
                    if i+1 in self.skip_connect:
                        self.net.append(nn.Sequential(
                        nn.Linear(hidden_features[i]+self.latent_dim, hidden_features[i+1]),
                        copy.deepcopy(nl)
                    ))
                    else:
                        self.net.append(nn.Sequential(
                            nn.Linear(hidden_features[i], hidden_features[i+1]),
                            copy.deepcopy(nl)
                        ))

            if (self.dropout > 0):
                self.net.append(nn.Dropout(self.dropout))
            if outermost_linear:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    copy.deepcopy(outmost_nl)
                ))
            if layer_norm:
                self.net.append(nn.LayerNorm([out_features]))

        self.net = nn.Sequential(*self.net)

        if isinstance(nonlinearity, list):
            for layer_num, layer_name in enumerate(nonlinearity):
                self.net[layer_num].apply(layer_factory(layer_name)[1])
        elif isinstance(nonlinearity, str):
            if self.weight_init is not None:
                self.net.apply(self.weight_init)

            if self.first_layer_init is not None:
                self.net[0].apply(self.first_layer_init)

        if set_bias is not None:
            self.net[-1][0].bias.data = set_bias * torch.ones_like(self.net[-1][0].bias.data)

    def forward(self, coords, batch_vecs=None):
        if self.skip_connect == None:
            output = self.net(coords)
        else:
            input = coords
            for i in range(len(self.net)):
                output = self.net[i](input)
                if i+1 in self.skip_connect:
                    input = torch.cat([batch_vecs, output], dim=-1)
                else:
                    input = output
        return output


class CoordinateNet_autodecoder(nn.Module):
    '''A autodecoder network'''
    def __init__(self, latent_size=64, out_features=1, nl='sine', in_features=64+2,
                 hidden_features=256, num_hidden_layers=3, num_pe_fns=6,
                 w0=60,use_pe=False,skip_connect=None,dataset_size=100,
                 outmost_nonlinearity=None,outermost_linear=True):
        super().__init__()

        self.nl = nl
        self.use_pe = use_pe
        self.latent_size = latent_size
        self.lat_vecs = torch.nn.Embedding(dataset_size, self.latent_size)
        torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, 1/ math.sqrt(self.latent_size))

        if self.nl != 'sine' and use_pe:
            in_features = 2 * (2*num_pe_fns + 1)+latent_size

        if self.use_pe:
            self.pe = PositionalEncoding(num_encoding_functions=num_pe_fns)
        self.decoder = FCBlock(in_features=in_features,
                           out_features=out_features,
                           num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features,
                           outermost_linear=outermost_linear,
                           nonlinearity=nl,
                           w0=w0,skip_connect=skip_connect,latent_dim=latent_size,outmost_nonlinearity=outmost_nonlinearity)
        self.mean =  torch.mean(torch.mean(self.lat_vecs.weight.data.detach(), dim=1)).cuda()
        self.varience =  torch.mean(torch.var(self.lat_vecs.weight.data.detach(), dim=1)).cuda()
    

    def forward(self, model_input,latent=None):
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        if latent==None:
            batch_vecs = self.lat_vecs(model_input['idx']).unsqueeze(1).repeat(1,coords.shape[1],1)
        else:
            batch_vecs = latent.unsqueeze(1).repeat(1,coords.shape[1],1)

        if self.nl != 'sine' and self.use_pe:
            coords_pe = self.pe(coords)
            input = torch.cat([batch_vecs, coords_pe], dim=-1)
            output = self.decoder(input,batch_vecs)
        else:
            input = torch.cat([batch_vecs, coords], dim=-1)
            output = self.decoder(input,batch_vecs)
     
        return {'model_in': coords, 'model_out': output,'batch_vecs': batch_vecs, 'meta': model_input}