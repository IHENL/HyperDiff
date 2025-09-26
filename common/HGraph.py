import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

import random
from common.hypergraph import *

  
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channel_first=False):
        """
        :param channel_first: if True, during forward the tensor shape is [B, C, T, J] and fc layers are performed with
                              1x1 convolutions.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        if channel_first:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class HGraphBlock(nn.Module):
    """
    Implementation of AGFormer block.
    """
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5,
                neighbour_num=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.mixer = HyperGCN(dim, dim,
                                num_nodes=17,
                                neighbour_num=neighbour_num,
                                use_partscale=True,
                                use_bodyscale=True,
                                connections=None, dataset='h36m' # h36m
                                )

        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        """
        x: tensor with shape [B, 1, J, C]
        mixer + mlp
        """
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x)))

            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HGblock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, 
                 neighbour_num=4,
                 ):
        super().__init__()
        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim
        
        self.graph_spatial = HGraphBlock(dim, mlp_ratio, act_layer, drop, drop_path, 
                                        use_layer_scale, layer_scale_init_value,
                                        neighbour_num=neighbour_num)

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.hierarchical:
            B, T, J, C = x.shape
            x_attn, x_graph = x[..., :C // 2], x[..., C // 2:]
            x_graph = self.graph_spatial(x_graph + x_attn)
        else:
 
            x_graph = self.graph_spatial(x)

        x = x_graph

        return x


def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, 
                  attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  use_layer_scale=True, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False, 
                 neighbour_num=4):
    """
    generates MotionAGFormer layers
    """
    layers = []
    for i in range(n_layers):
        layers.append(HGblock(dim=dim,
                            mlp_ratio=mlp_ratio,
                            act_layer=act_layer,
                            attn_drop=attn_drop,
                            drop=drop_rate,
                            drop_path=drop_path_rate,                           
                            use_layer_scale=use_layer_scale,
                            layer_scale_init_value=layer_scale_init_value,                            
                            use_adaptive_fusion=use_adaptive_fusion,
                            hierarchical=hierarchical,                          
                            neighbour_num=neighbour_num,
                            ))

    layers = nn.Sequential(*layers)

    return layers

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class HGraph(nn.Module):
    """
    Hgmamba, the main class of our model.
    """
    def __init__(self, n_layers, dim_in=5, dim_feat=128, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 hierarchical=False, num_joints=17,
                 neighbour_num=4, is_train=True):
        """
        :param n_layers: Number of layers.
        :param dim_in: Input dimension.
        :param dim_feat: Feature dimension.
        :param dim_out: output dimension. For 3D pose lifting it is set to 3
        :param mlp_ratio: MLP ratio.
        :param act_layer: Activation layer.
        :param drop: Dropout rate.
        :param drop_path: Stochastic drop probability.
        :param use_layer_scale: Whether to use layer scaling or not.
        :param layer_scale_init_value: Layer scale init value in case of using layer scaling.
        :param use_adaptive_fusion: Whether to use adaptive fusion or not.        
        :param hierarchical: Whether to use hierarchical structure or not.
        :param num_joints: Number of joints.      
        :param neighbour_num: Number of neighbors for temporal GCN similarity.
        """
        super().__init__()

        self.Spatial_patch_to_embedding = nn.Linear(dim_in, dim_feat)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.is_train=is_train
        self.norm = nn.LayerNorm(dim_feat)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim_feat),
            nn.Linear(dim_feat, dim_feat*2),
            nn.GELU(),
            nn.Linear(dim_feat*2, dim_feat),
        )

        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,                                 
                                    use_layer_scale=use_layer_scale,                                  
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    neighbour_num=neighbour_num,
                                    )


        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

    def S_forward(self, x_2d, x_3d, t):
        if self.is_train:
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
            # x = rearrange(x, 'b f n c  -> (b f) n c', )
            ### now x is [batch_size, receptive frames, joint_num, 2 channels]
            x = self.Spatial_patch_to_embedding(x)
            x = self.norm(x)  # norm in proj
            x += self.Spatial_pos_embed
            time_embed = self.time_mlp(t)[:, None, None, :].repeat(1,f,n,1)
            # time_embed = rearrange(time_embed, 'b f n c  -> (b f) n c', )
            x += time_embed
        else:
            # x_2d_graph = x_2d_graph[:,None].repeat(1,x_3d.shape[1],1,1,1)
            x_2d = x_2d[:,None].repeat(1,x_3d.shape[1],1,1,1)
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, h, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
            # x = rearrange(x, 'b h f n c  -> (b h f) n c', )
            x = rearrange(x, 'b h f n c  -> (b h) f n c', )
            x = self.Spatial_patch_to_embedding(x)
            x = self.norm(x)  # norm in proj
            x += self.Spatial_pos_embed
            time_embed = self.time_mlp(t)[:, None, None, None, :].repeat(1, h, f, n, 1)
            time_embed = rearrange(time_embed, 'b h f n c  -> (b h) f n c', )
            # time_embed = rearrange(time_embed, 'b h f n c  -> (b h f) n c', )
            x += time_embed

        return x

    def forward(self, x_2d, x_3d, t):
        """
        :param x: tensor with shape [B, T, J, C] (T=243, J=17, C=3)
        :param return_rep: Returns motion representation feature volume (In case of using this as backbone)
        """
        if self.is_train:
            b, f, n, c = x_2d.shape
        else:
            b, h, f, n, c = x_3d.shape

        x = self.S_forward(x_2d, x_3d, t)
        
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.rep_logit(x)
        
        x = self.head(x)

        if self.is_train:
            x = x.view(b, f, n, -1)
        else:
            x = x.view(b, h, f, n, -1)

        return x


