import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math
import numpy as np
from typing import Optional, List
from torch import nn, Tensor
from .multihead_attention import MultiheadAttention
from ltr.models.layers.normalization import InstanceL2Norm
import ltr.admin.settings as ws_settings
import matplotlib.pyplot as plt

import pdb
class HelixEmbedding(nn.Module):
    def __init__(self, demb):
        super(HelixEmbedding, self).__init__()
        self.demb = demb
        self.pitch = 1/(2*math.pi) ##螺旋线的螺距为1,v/w=1/2pi
        inv_freq = 1 / (100 ** (torch.arange(0.0, demb, 1.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, R_matrix, A_matrix, bsz=None):
        R_press = R_matrix.squeeze(dim=2)
        # R_line = R_press.squeeze()
        # A_press = torch.flatten(A_matrix)
        # A_dim = A_press.repeat(self.demb, 1).permute(1, 0)
        A_dim = A_matrix.repeat(1, 1, self.demb, 1, 1)#.permute(1, 0)

        # inv = torch.ger(R_press, self.inv_freq)
        inv = torch.einsum('bnwh,c->bncwh', [R_press, self.inv_freq])
        inv_with_phase = inv + A_dim
        inv_with_v = inv * self.pitch
        helixembed_real = inv_with_v.mul(inv_with_phase.cos())
        helixembed_imag = inv_with_v.mul(inv_with_phase.sin())
        if bsz is not None:
            return helixembed_real[:, None, :].expand(-1, bsz, -1), helixembed_imag[:, None, :].expand(-1, bsz, -1)
        else:
            # return helixembed_real[:, None, :], helixembed_imag[:, None, :]
            return helixembed_real, helixembed_imag

class PositionEmbeddingSine(nn.Module):
    def __init__(self, demb):
        super(PositionEmbeddingSine, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, feat):
        num_imgs, batch, dim, h, w = feat.shape
        # Normlization

        # P = torch.ones(h*w).to(self.device)
        P_seq = pos_seq.cumsum(0)
        sinusoid_inp = torch.ger(P_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = pos_emb.expand(num_imgs, batch, -1, -1).reshape(num_imgs, batch, h, w, dim).permute(0,1,4,2,3)
        # pos_emb = pos_emb.reshape(num_imgs, batch, dim, h, w) #

        return pos_emb
        # if bsz is not None:
        #     return pos_emb[:, None, :].expand(-1, bsz, -1)
        # else:
        #     return pos_emb[:, None, :]

    # def __init__(self, num_pos_feats, temperature=10000, normalize=False, scale=None):
    #     super(HelixEmbedding, self).__init__()
    #     self.num_pos_feats = num_pos_feats
    #     self.temperature = temperature
    #     self.normalize = normalize
    #     if scale is not None and normalize is False:
    #         raise ValueError("normalize should be True if scale is passed")
    #     if scale is None:
    #         scale = 2 * math.pi
    #     self.scale = scale

    # def forward(self, feats, mask):
    #     y_embed = mask.cumsum(1,dtype=torch.float32)
    #     x_embed = mask.cumsum(2,dtype=torch.float32)
    #     pos = torch.cat((y_embed, x_embed), dim=3)
    #     return pos
class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.0, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        # self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, pos_seq, feat):
        num_imgs, batch, dim, h, w = feat.shape
        inp = feat.permute(0,1,3,4,2).reshape(-1, dim)

        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))
            # core_out = core_out.reshape(num_imgs, -1, *feat.shape[-3:])
            ##### residual connection
            output = core_out + inp
            output = output.reshape(num_imgs, batch, h, w, dim).permute(0,1,4,2,3)

        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)
            # core_out = core_out.reshape(num_imgs, -1, *feat.shape[-3:])

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)
            output = output.reshape(num_imgs, batch, h, w, dim).permute(0,1,4,2,3)

        return output

class TransformerEmbed(nn.Module):
    def __init__(self, d_model=512, nhead=1, num_layers=1, dim_feedforward=2048, 
                 activation="relu", pos_embed_type='v0'):
        super().__init__()
        multihead_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        # FFN_conv = nn.Conv2d()  # do not use feed-forward network
        self.encoder = TransformerEmbedEncoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_encoder_layers=num_layers)
        self.decoder = TransformerEmbedDecoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_decoder_layers=num_layers)
        settings = ws_settings.Settings()
        # BaseTrainer.update_settings(settings)
        self.device = getattr(settings, 'device', None)
        if self.device is None:
            # self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")
            self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        if pos_embed_type in ('v1', 'sine'):
            self.pos_emb = PositionEmbeddingSine(d_model)
        elif pos_embed_type in ('v2', 'learned'):
            self.pos_emb = PositionwiseFF(d_model, d_model, pre_lnorm=False)
        else:
            print('v0 are used')#pos_embed_type

    def forward(self, train_feat, test_feat, train_label):
        num_img_train = train_feat.shape[0]
        num_img_test = test_feat.shape[0]
        train_pos_seq = torch.ones(train_feat.shape[3]*train_feat.shape[4]).to(self.device)
        test_pos_seq = torch.ones(test_feat.shape[3]*test_feat.shape[4]).to(self.device)
        train_pos = self.pos_emb(train_pos_seq, train_feat)
        test_pos = self.pos_emb(test_pos_seq, test_feat)
        ## encoder
        # encoded_memory, _ = self.encoder(train_feat, pos=None)
        encoded_memory, _ = self.encoder(train_feat, pos_embed=train_pos)

        ## decoder
        for i in range(num_img_train):
            # _, cur_encoded_feat = self.decoder(train_feat[i,...].unsqueeze(0), memory=encoded_memory, pos=train_label, query_pos=None)
            _, cur_encoded_feat = self.decoder(train_feat[i,...].unsqueeze(0), pos_embed=train_pos[i,...].unsqueeze(0), memory=encoded_memory, pos=train_label)
            if i == 0:
                encoded_feat = cur_encoded_feat
            else:
                encoded_feat = torch.cat((encoded_feat, cur_encoded_feat), 0)
        
        for i in range(num_img_test):
            # _, cur_decoded_feat = self.decoder(test_feat[i,...].unsqueeze(0), memory=encoded_memory, pos=train_label, query_pos=None)
            _, cur_decoded_feat = self.decoder(test_feat[i,...].unsqueeze(0), pos_embed=test_pos[i,...].unsqueeze(0), memory=encoded_memory, pos=train_label)
            if i == 0:
                decoded_feat = cur_decoded_feat
            else:
                decoded_feat = torch.cat((decoded_feat, cur_decoded_feat), 0)

        return encoded_feat, decoded_feat


class TransformerEmbedEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def with_pos_train_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed*0.001

    def instance_norm(self, src, input_shape):
        num_imgs, batch, dim, h, w = input_shape
        # Normlization
        src = src.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)
        src = src.reshape(-1, batch, dim)
        return src

    def forward(self, src, input_shape, pos_embed: Optional[Tensor] = None):
        if pos_embed is not None:
            src = self.with_pos_train_embed(src, pos_embed)
        # query = key = value = src
        query = src  #src_shape:(7260,1,512) #(num_imgs*wh, batch, dim)
        key = src
        value = src
        ###### tgt_visual ####
        # plt.cla()
        # with torch.no_grad():
        #     src_copy = src.cpu().numpy()
        # plt.imshow(src_copy[:, 1, :])
        # plt.axis('off')
        # plt.axis('equal')
        # src_path = '/data1/lxt/2021projects/work-2021/Hyper-para/Fig7_draw/pos_map/src_map.jpg'
        # srcmap = plt.gcf()
        # srcmap.savefig(src_path)
        ##########################
        # self-attention
        src2 = self.self_attn(query=query, key=key, value=value)
        ########
        # plt.cla()
        # with torch.no_grad():
        #     src2_copy_2 = src2.cpu().numpy()
        # plt.imshow(src2_copy_2[:, 1, :])
        # plt.axis('off')
        # plt.axis('equal')
        # src2_path = '/data1/lxt/2021projects/work-2021/Hyper-para/Fig7_draw/pos_map/src_2_map.jpg'
        # src2map = plt.gcf()
        # src2map.savefig(src2_path)
        ########
        src = src + src2
        src = self.instance_norm(src, input_shape)
        #######
        # plt.cla()
        # with torch.no_grad():
        #     src2_copy = src.cpu().numpy()
        # plt.imshow(src2_copy[:, 1, :])
        # plt.axis('off')
        # plt.axis('equal')
        # src22_path = '/data1/lxt/2021projects/work-2021/Hyper-para/Fig7_draw/pos_map/src_22_map.jpg'
        # src22map = plt.gcf()
        # src22map.savefig(src22_path)
        # #######
        return src


class TransformerEmbedEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_encoder_layers=6, activation="relu"):
        super().__init__()
        encoder_layer = TransformerEmbedEncoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)


    def forward(self, src, pos_embed: Optional[Tensor] = None):
        assert src.dim() == 5, 'Expect 5 dimensional inputs'
        src_shape = src.shape
        num_imgs, batch, dim, h, w = src.shape #

        src = src.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)   #(num_imgs, wh, batch, dim)
        src = src.reshape(-1, batch, dim)                              #(num_imgs*wh, batch, dim)

        # if pos is not None:
        #     pos = pos.view(num_imgs, batch, 1, -1).permute(0,3,1,2)   #(num_imgs, wh, batch, 1)
        #     pos = pos.reshape(-1, batch, 1)                           #(num_imgs*wh, batch, 1)
        if pos_embed is not None:
            pos_embed = pos_embed.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)   #(num_imgs, wh, batch, dim)
            pos_embed = pos_embed.reshape(-1, batch, dim)                           #(num_imgs*wh, batch, dim)

        output = src

        for layer in self.layers:
            output = layer(output, input_shape=src_shape, pos_embed=pos_embed)

        # [L,B,D] -> [B,D,L]
        output_feat = output.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        output_feat = output_feat.reshape(-1, dim, h, w)
        return output, output_feat

class TransformerEmbedDecoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn    #multihead_attn = MultiheadAttention(feature_dim=d_model=512, n_head=1, key_feature_dim=128)
        self.cross_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)

        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor * pos

    def with_pos_test_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos*0.001

    def instance_norm(self, src, input_shape):
        num_imgs, batch, dim, h, w = input_shape
        # Normlization
        src = src.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)
        src = src.reshape(-1, batch, dim)
        return src

    # def forward(self, tgt, memory, input_shape, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
    def forward(self, tgt, pos_embed, memory, input_shape, pos: Optional[Tensor] = None): #, query_pos: Optional[Tensor] = None
        # self-attention
        if pos_embed is not None:
            ###### pos_map_visual ####
            # plt.cla()
            # pos_embed_copy = pos_embed.cpu().numpy()
            # plt.imshow(pos_embed_copy[:, 1, :])
            # plt.axis('off')
            # plt.axis('equal')
            # pos_path = '/data1/lxt/2021projects/work-2021/Hyper-para/Fig7_draw/pos_map/pos_embed_map.jpg'
            # Posmap = plt.gcf()
            # Posmap.savefig(pos_path)
            ##########################
            ###### tgt_visual ####
            # plt.cla()
            # with torch.no_grad():
            #     tgt_copy = tgt.cpu().numpy()
            # plt.imshow(tgt_copy[:, 1, :])
            # plt.axis('off')
            # plt.axis('equal')
            # tgt_path = '/data1/lxt/2021projects/work-2021/Hyper-para/Fig7_draw/pos_map/tgt_map.jpg'
            # tgtmap = plt.gcf()
            # tgtmap.savefig(tgt_path)
            ##########################
            tgt = self.with_pos_test_embed(tgt, pos_embed)
            ########
            # plt.cla()
            # with torch.no_grad():
            #     tgt_copy_2 = tgt.cpu().numpy()
            # plt.imshow(tgt_copy_2[:, 1, :])
            # plt.axis('off')
            # plt.axis('equal')
            # tgt2_path = '/data1/lxt/2021projects/work-2021/Hyper-para/Fig7_draw/pos_map/tgt_2_map.jpg'
            # tgt2map = plt.gcf()
            # tgt2map.savefig(tgt2_path)
            ########

        query = tgt
        key = tgt
        value = tgt
        
        tgt2 = self.self_attn(query=query, key=key, value=value)  #tgt2 = tgt_self_atten
        tgt = tgt + tgt2                                          #tgt22 = tgt_self_res_atten
        tgt = self.instance_norm(tgt, input_shape)    #tgt.shape:(484,1,512) # tgt222 = tgt_self_res_norm_atten
    
        mask = self.cross_attn(query=tgt, key=memory, value=pos)    #mask = tgt222 & memory & pos  #mask.shape:(484,1,512)  pos(7260,1,512)??
        tgt2 = tgt * mask                                           #mask * tgt222 = mask * tgt222
        tgt2 = self.instance_norm(tgt2, input_shape)                #norm(mask * tgt222)

        tgt3 = self.cross_attn(query=tgt, key=memory, value=memory*pos)   #tgt3 =tgt222 & memory & memory*pos     #tgt3.shape:(484,1,512)
        tgt4 = tgt + tgt3                                                 #tgt4 = tgt3 + tgt222
        tgt4 = self.instance_norm(tgt4, input_shape)                      #norm(tgt4)
    
        tgt = tgt2 + tgt4                                                 #out = norm (norm(mask & tgt222)+norm(tgt3 + tgt222))
        tgt = self.instance_norm(tgt, input_shape)
        return tgt


class TransformerEmbedDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_decoder_layers=6, activation="relu"):
        super().__init__()
        decoder_layer = TransformerEmbedDecoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        # self.post1 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        # self.activation = _get_activation_fn(activation)
        # self.post2 = nn.Conv2d(d_model, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, tgt, pos_embed, memory, pos: Optional[Tensor] = None): #query_pos: Optional[Tensor] = None
        assert tgt.dim() == 5, 'Expect 5 dimensional inputs'
        tgt_shape = tgt.shape
        num_imgs, batch, dim, h, w = tgt.shape

        if pos is not None:
            num_pos, batch, h, w = pos.shape
            pos = pos.view(num_pos, batch, 1, -1).permute(0,3,1,2)
            pos = pos.reshape(-1, batch, 1)
            pos = pos.repeat(1, 1, dim)

        tgt = tgt.view(num_imgs, batch, dim, -1).permute(0,3,1,2)
        tgt = tgt.reshape(-1, batch, dim)

        if pos_embed is not None:
            pos_embed = pos_embed.view(num_imgs, batch, dim, -1).permute(0,3,1,2)
            pos_embed = pos_embed.reshape(-1, batch, dim)
        output = tgt
        
        for layer in self.layers:
            output = layer(output, pos_embed, memory, input_shape=tgt_shape, pos=pos)  # diff in input from encoder, query_pos=query_pos

        # [L,B,D] -> [B,D,L]
        output_feat = output.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        output_feat = output_feat.reshape(-1, dim, h, w)
        # output = self.post2(self.activation(self.post1(output)))
        return output, output_feat


def _get_clones(module, N):
    # return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    return nn.ModuleList([module for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


