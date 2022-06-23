import pickle
import numpy as np
import torch
from torch import nn
from array import array
import json
from pathlib import Path
from scipy.io import loadmat


class MorphableBase:
    @property
    def num_vertices(self):
        return self.shape_mean.shape[0]

    def vertices(self, shape_coeffs=None, select_index=slice(None)) -> np.ndarray:
        v = self.shape_mean[select_index]
        if shape_coeffs is not None:
            v = v + self.shape_basis[select_index] @ shape_coeffs
        return v

    @property
    def num_shape_coeffs(self):
        return self.shape_basis.shape[-1]

    def reduce_num_shape_coeffs(self, n: int):
        self.shape_basis = self.shape_basis[..., :n]

class WithExpression(MorphableBase):
    @property
    def num_expression_coeffs(self):
        return self.expression_basis.shape[-1]

    def reduce_num_expression_coeffs(self, n: int):
        self.expression_basis = self.expression_basis[..., :n]

    def vertices(self, shape_coeffs=None, expression_coeffs=None, select_index=slice(None)) -> np.ndarray:
        v = super().vertices(shape_coeffs=shape_coeffs, select_index=select_index)
        v += self.expression_mean[select_index]
        if expression_coeffs is not None:
            v += self.expression_basis[select_index] @ expression_coeffs
        return v


class WithVertexColor(MorphableBase):
    @property
    def num_color_coeffs(self):
        return self.color_basis.shape[-1]

    def reduce_num_color_coeffs(self, n: int):
        self.color_basis = self.color_basis[..., :n]

    def vertex_color(self, color_coeffs, select_index=slice(None)) -> np.ndarray:
        return self.color_mean[select_index] + self.color_basis[select_index] @ color_coeffs



def filter_tri(indices: np.ndarray, filtered_vertices_idx: np.ndarray):
    mask = np.all(np.isin(indices, filtered_vertices_idx), axis=-1)
    indices = indices[mask]

    old_to_new = np.full(filtered_vertices_idx.max() + 1, -1)
    old_to_new[filtered_vertices_idx] = np.arange(filtered_vertices_idx.shape[0])
    return old_to_new[indices]


class BFM_base(WithExpression, WithVertexColor, MorphableBase):
    pass

class BFM_mat(BFM_base):
    '''
    Read .mat files from BFM09

    Note the length unit in .mat files is μm, so scale by 1e-3 to mm by default, to be consistent with .h5 files
    '''
    def __init__(self, config, color=True, expression=True, geo_scale=1e-3):
        front = 'front' in config and config['front']
        if front and not expression:
            raise NotImplementedError()

        bfm_dir = Path(config['dir'])

        bfm = loadmat((bfm_dir / '01_MorphableModel.mat').open('rb'))

        self.indices = (bfm['tl'] - 1)[..., ::-1]

        self.shape_mean = bfm['shapeMU'].reshape(-1, 3) * geo_scale
        self.shape_basis = bfm['shapePC'].reshape(self.num_vertices, 3, -1)
        self.shape_std = bfm['shapeEV'].squeeze() * geo_scale

        if color:
            self.color_mean = bfm['texMU'].reshape(-1, 3) / 256
            self.color_basis = bfm['texPC'].reshape(self.num_vertices, 3, -1)
            self.color_std = bfm['texEV'].squeeze() / 256

        if expression:
            idx = (loadmat((bfm_dir / 'BFM_exp_idx.mat').open('rb'))['trimIndex'] - 1).squeeze()

            with (bfm_dir / 'Exp_Pca.bin').open('rb') as exp_bin:
                n_vertex = idx.shape[0]
                exp_dim = array('i')
                exp_dim.fromfile(exp_bin, 1)
                exp_dim = exp_dim[0]
                expMU = array('f')
                expPC = array('f')
                expMU.fromfile(exp_bin, 3 * n_vertex)
                expPC.fromfile(exp_bin, 3 * exp_dim * n_vertex)

            self.expression_mean = np.array(expMU).reshape(-1, 3) * geo_scale
            expPC = np.array(expPC).reshape(exp_dim, -1, 3)
            self.expression_basis = np.moveaxis(expPC, 0, -1)
            self.expression_std = np.loadtxt(bfm_dir / 'std_exp.txt') * geo_scale

            if front:  # from Deep3DFaceRecon_pytorch
                front_idx = loadmat((bfm_dir / 'BFM_front_idx.mat').open('rb'))['idx'].astype(int).squeeze() - 1 # 选出脸的部分，去掉耳朵
                self.expression_mean = self.expression_mean[front_idx]
                self.expression_basis = self.expression_basis[front_idx]
                idx = idx[front_idx]


            self.indices = filter_tri(self.indices, idx)
            self.shape_mean = self.shape_mean[idx]
            self.shape_basis = self.shape_basis[idx]

            if color:
                self.color_mean = self.color_mean[idx]
                self.color_basis = self.color_basis[idx]
            self.uv_indices = None
            if 'uvPath' in config:  # from OSTeC
                self.uv = np.load(config['uvPath'])
            else:
                self.uv = None

    def reduce_num_shape_coeffs(self, n: int):
        super().reduce_num_shape_coeffs(n)
        self.shape_std = self.shape_std[:n]

    def reduce_num_expression_coeffs(self, n: int):
        super().reduce_num_expression_coeffs(n)
        self.expression_std = self.expression_std[:n]

    def reduce_num_color_coeffs(self, n: int):
        super().reduce_num_color_coeffs(n)
        self.color_std = self.color_std[:n]



class Morphable(nn.Module):
    def __init__(self, morphable: MorphableBase, shape=True, expression=True):
        super().__init__()

        mean = 0
        basis = []
        if shape:
            mean = mean + morphable.shape_mean
            basis.append(morphable.shape_basis * morphable.shape_std)
        if expression:
            mean = mean + morphable.expression_mean
            basis.append(morphable.expression_basis * morphable.expression_std)

        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32), persistent=False)
        self.register_buffer('basis', torch.tensor(np.concatenate(basis, axis=-1), dtype=torch.float32).permute(2,0,1), persistent=False)

    def select_vertices(self, idx):
        self.mean = self.mean[idx]
        self.basis = self.basis[idx]

    def scale(self, s):
        self.mean *= s
        self.basis *= s

    @property
    def num_coeffs(self):
        return self.basis.size(0)

    @property
    def num_vertices(self):
        return self.mean.size(0)

    def forward(self, coeffs: torch.Tensor):
        return self.mean + torch.einsum("fc,cvk->fvk", coeffs, self.basis)


if __name__ == '__main__':
    BFM_mat({'dir': "/root/python/audio2face/FaceFormer/BFM/", "front": True})
    
