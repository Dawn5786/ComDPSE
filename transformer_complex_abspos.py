import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math
import numpy as np
from typing import Optional, List
from torch import nn, Tensor
from .multihead_attention import MultiheadAttention
from .multihead_attention_complex import ComMultiheadAttention
from .multihead_attention_complex_simple import ComSimMultiheadAttention
from ltr.models.layers.normalization import InstanceL2Norm
from ltr.trainers import BaseTrainer
import ltr.admin.settings as ws_settings

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

# class Transformer(nn.Module):
class AbsComTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=1, num_layers=1, dim_feedforward=2048,
                 activation="relu"):
        super().__init__()
        # multihead_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        # multihead_attn = ComMultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        multihead_attn = ComSimMultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        enhead_attn = ComSimMultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=d_model)
        # FFN_conv = nn.Conv2d()  # do not use feed-forward network
        # self.encoder = TransformerEncoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_encoder_layers=num_layers)
        self.encoder = ComTransformerEncoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_encoder_layers=num_layers)
        # self.decoder = TransformerDecoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_decoder_layers=num_layers)
        self.decoder = ComTransformerDecoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_decoder_layers=num_layers)
        self.compress = nn.Conv2d(2*d_model, d_model, kernel_size=1, stride=1, bias=False)
        self.helix_emb = HelixEmbedding(d_model)
        settings = ws_settings.Settings()
        # BaseTrainer.update_settings(settings)
        # self.device = getattr(settings, 'device', None)
        # if self.device is None:
        #     self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")
        self.drop_path_prob = 0.0

    # def forward(self, train_feat, test_feat, train_label):
    # adding input  # the trasformer module in classifier need abspos adding inputs (test_bb, test_label)
    # def forward(self, train_feat, test_feat, train_label):
    def forward(self, train_feat, test_feat, train_X_pos_matrixs, train_Y_pos_matrixs,
                test_X_pos_matrixs, test_Y_pos_matrixs, train_label, test_label, train_anno_center, test_anno_center, train_bb, test_bb):
        num_img_train = train_feat.shape[0]
        num_img_test = test_feat.shape[0]

        train_feat_shape = train_feat.shape
        #num_imgs, batch, dim, h, w = train_feat.shape
        train_feat_real = train_feat
        train_feat_imag = train_feat - train_feat
        # train_feat_imag = torch.zeros(train_feat_shape).to(self.device)
        # train_feat_imag = train_feat_imag.to(self.device)
        #train_feat_imag = torch.zeros([num_imgs, batch, dim, h, w])
        # train_R_matrixs, train_Angle_matrixs = self.pos_to_polar(train_X_pos_matrixs, train_Y_pos_matrixs)
        train_X_pos_abs, train_Y_pos_abs = self.pos_to_sample(train_X_pos_matrixs, train_Y_pos_matrixs, train_anno_center, train_feat.shape)
        train_R_matrixs, train_Angle_matrixs = self.pos_to_polar(train_X_pos_abs, train_Y_pos_abs)
        # train_embed_real, train_embed_imag = self.polar_to_com_embedd(train_R_matrixs, train_Angle_matrixs)
        train_embed_real, train_embed_imag = self.helix_emb(train_R_matrixs, train_Angle_matrixs)
        train_FP_real = train_feat_real + train_embed_real
        train_FP_imag = train_feat_imag + train_embed_imag

        test_feat_shape = test_feat.shape
        #num_imgs, batch, dim, h, w = train_feat.shape
        test_feat_real = test_feat
        test_feat_imag = test_feat - test_feat
        # test_feat_imag = torch.zeros(test_feat_shape).to(self.device)
        # test_feat_imag = test_feat_imag.to(self.device)
        test_X_pos_abs, test_Y_pos_abs = self.pos_to_sample(test_X_pos_matrixs, test_Y_pos_matrixs, test_anno_center, test_feat.shape)
        test_R_matrixs, test_Angle_matrixs = self.pos_to_polar(test_X_pos_abs, test_Y_pos_abs)
        # test_embed_real, test_embed_imag = self.polar_to_com_embedd(test_R_matrixs, test_Angle_matrixs)
        test_embed_real, test_embed_imag = self.helix_emb(test_R_matrixs, test_Angle_matrixs)
        test_FP_real = test_feat_real + test_embed_real
        test_FP_imag = test_feat_imag + test_embed_imag

        train_label_shape = train_label.shape
        train_label_real = train_label
        train_label_imag = train_label -train_label
        # train_label_imag = torch.zeros(train_label_shape).to(self.device)
        # train_label_imag = train_label_imag.to(self.device)

        ## encoder
        # encoded_memory, _ = self.encoder(train_feat, pos=None)
        # encoded_memory_real, encoded_memory_imag = self.encoder(train_feat_real, train_feat_imag, pos=None)
        encoded_memory_real, encoded_memory_imag = self.encoder(train_FP_real, train_FP_imag, pos=None)  #[1452,4,512]

        ## decoder
        for i in range(num_img_train):
            # _, cur_encoded_feat = self.decoder(train_feat[i,...].unsqueeze(0), memory=encoded_memory, pos=train_label, query_pos=None)
            # _, _, cur_encoded_feat_real, cur_encoded_feat_imag, _ = self.decoder(train_feat_real[i,...].unsqueeze(0), train_feat_imag[i,...].unsqueeze(0), memory_real=encoded_memory_real, memory_imag=encoded_memory_imag, pos_real=train_label_real, pos_imag=train_label_imag, query_pos=None)
            _, _, cur_encoded_feat_real, cur_encoded_feat_imag, _ = self.decoder(train_FP_real[i,...].unsqueeze(0), train_FP_imag[i,...].unsqueeze(0), memory_real=encoded_memory_real, memory_imag=encoded_memory_imag, pos_real=train_label_real, pos_imag=train_label_imag, query_pos=None)
            if i == 0:
                # encoded_feat = cur_encoded_feat
                encoded_feat_real = cur_encoded_feat_real
                encoded_feat_imag = cur_encoded_feat_imag
            else:
                # encoded_feat = torch.cat((encoded_feat, cur_encoded_feat), 0)
                encoded_feat_real = torch.cat((encoded_feat_real, cur_encoded_feat_real), 0)  #[4,512,22,22] [4,512,22,22]->[8,512,22,22]
                encoded_feat_imag = torch.cat((encoded_feat_imag, cur_encoded_feat_imag), 0)

        encoded_feat = torch.cat([encoded_feat_real, encoded_feat_imag], 1) #dim维concat
        encoded_feat_compress = self.compress(encoded_feat) #[12,1024,22,22]->[12,512,22,22]

        for i in range(num_img_test):
            # _, cur_decoded_feat = self.decoder(test_feat[i,...].unsqueeze(0), memory=encoded_memory, pos=train_label, query_pos=None)
            # _, _, cur_decoded_feat_real, cur_decoded_feat_imag, _ = self.decoder(test_feat_real[i,...].unsqueeze(0), test_feat_imag[i,...].unsqueeze(0), memory_real=encoded_memory_real, memory_imag=encoded_memory_imag, pos_real=train_label_real, pos_imag=train_label_imag, query_pos=None)
            _, _, cur_decoded_feat_real, cur_decoded_feat_imag, _ = self.decoder(test_FP_real[i,...].unsqueeze(0), test_FP_imag[i,...].unsqueeze(0), memory_real=encoded_memory_real, memory_imag=encoded_memory_imag, pos_real=train_label_real, pos_imag=train_label_imag, query_pos=None)
            if i == 0:
                # decoded_feat = cur_decoded_feat
                decoded_feat_real = cur_decoded_feat_real
                decoded_feat_imag = cur_decoded_feat_imag
            else:
                # decoded_feat = torch.cat((decoded_feat, cur_decoded_feat), 0)
                decoded_feat_real = torch.cat((decoded_feat_real, cur_decoded_feat_real), 0)
                decoded_feat_imag = torch.cat((decoded_feat_imag, cur_decoded_feat_imag), 0)

        decoded_feat = torch.cat([decoded_feat_real, decoded_feat_imag], 1) #dim维concat
        decoded_feat_compress = self.compress(decoded_feat)

        # return encoded_feat, decoded_feat
        return encoded_feat_compress, decoded_feat_compress

    def pos_to_sample(self, Xpos_matrixs, Ypos_matrixs, anno_center, out_size):

        num_imgs, batch, dim, h, w = out_size
        N = Xpos_matrixs.shape[-1]
        fact = N/h  #16
        interval = fact * torch.arange(h)  # 16 * [0,1,2,...21]
        mid_left = math.floor((fact - 1) / 2)
        mid_right = math.ceil((fact - 1) / 2)
        Index_1 = mid_left + interval
        Index_2 = mid_right + interval

        Xpos_sample = (Xpos_matrixs[:, :, :, 0:h, Index_1.long()] + Xpos_matrixs[:, :, :, 0:h, Index_2.long()])/2
        Ypos_sample = (Ypos_matrixs[:, :, :, Index_1.long(), 0:h] + Xpos_matrixs[:, :, :, Index_2.long(), 0:h])/2

        X_center_shift = anno_center[:, :, 0].unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(num_imgs, batch, 1, h, w)
        Y_center_shift = anno_center[:, :, 1].unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(num_imgs, batch, 1, h, w)

        X_pos_abs = (Xpos_sample + X_center_shift)/(2*N)
        Y_pos_abs = (Ypos_sample + Y_center_shift)/(2*N)
        #归一化
        # R_matrixs = Xpos_matrixs
        # Angle_matrixs = Ypos_matrixs
        return X_pos_abs, Y_pos_abs #Angle_matrixs

    def pos_to_polar(self, X_pos, Y_pos):
        R_matrixs = X_pos.mul(X_pos) + Y_pos.mul(Y_pos)
        R_matrixs = R_matrixs.sqrt()
        Angle_matrixs = torch.atan2(Y_pos, X_pos)
        return R_matrixs, Angle_matrixs

    def polar_to_com_embedd(self, R_matrixs, Angle_matrixs):

        embed_real = R_matrixs
        embed_imag = Angle_matrixs
        return embed_real, embed_imag

# class TransformerEncoderLayer(nn.Module):
class ComTransformerEncoderLayer(nn.Module):
    def __init__(self, enhead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = enhead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

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

    # def forward(self, src, input_shape, pos: Optional[Tensor] = None):
    def forward(self, src_real, src_imag, input_shape, pos: Optional[Tensor] = None):

        #if src.shape(-1)
        # query = key = value = src
        # query = src  #src_shape:(7260,1,512) #(num_imgs*wh, batch, dim)
        # key = src
        # value = src

        query_real = src_real  #src_shape:(7260,1,512) #(num_imgs*wh, batch, dim)
        key_real = src_real
        value_real = src_real
        query_imag = src_imag  #src_shape:(7260,1,512) #(num_imgs*wh, batch, dim)
        key_imag = src_imag
        value_imag = src_imag

        # self-attention
        # src2_real = self.self_attn(query=query_real, key=key_real, value=value_real)
        # src_real = src_real + src2_real
        # src_real = self.instance_norm(src_real, input_shape)
        #
        # src2_imag = self.self_attn(query=query_imag, key=key_imag, value=value_imag)
        # src_imag = src_imag + src2_imag
        # src_imag = self.instance_norm(src_imag, input_shape)

        src2_real, src2_imag = self.self_attn(query_real=query_real, query_imag=query_imag, key_real=key_real, key_imag=key_imag, value_real=value_real, value_imag=value_imag)
        src_real = src_real + src2_real
        src_imag = src_imag + src2_imag
        src_real = self.instance_norm(src_real, input_shape)
        src_imag = self.instance_norm(src_imag, input_shape)
        # src = torch.cat([src_real, src_imag], 2) #dim维concat

        return src_real, src_imag


# class TransformerEncoder(nn.Module):
class ComTransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_encoder_layers=6, activation="relu"):
        super().__init__()
        # encoder_layer = TransformerEncoderLayer(multihead_attn, FFN, d_model)
        encoder_layer = ComTransformerEncoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)


    # def forward(self, src, pos: Optional[Tensor] = None):
    def forward(self, src_real, src_imag, pos: Optional[Tensor] = None):
        assert src_real.dim() == 5, 'Expect 5 dimensional inputs'
        src_real_shape = src_real.shape
        num_imgs, batch, dim, h, w = src_real.shape #

        src_real = src_real.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)   #(num_imgs, wh, batch, dim)
        src_imag = src_imag.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)   #(num_imgs, wh, batch, dim)
        src_real = src_real.reshape(-1, batch, dim)                              #(num_imgs*wh, batch, dim)
        src_imag = src_imag.reshape(-1, batch, dim)                              #(num_imgs*wh, batch, dim)

        if pos is not None:
            pos = pos.view(num_imgs, batch, 1, -1).permute(0,3,1,2)   #(num_imgs, wh, batch, 1)
            pos = pos.reshape(-1, batch, 1)                           #(num_imgs*wh, batch, 1)

        # output = src 换成input
        # input = torch.cat([src_real, src_imag], 2)


        for layer in self.layers:
            # output = layer(output, input_shape=src_shape, pos=pos)
            output_real, output_imag = layer(src_real, src_imag, input_shape=src_real_shape, pos=pos)

        # [L,B,D] -> [B,D,L]
        # output_feat = output.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        # output_feat = output_feat.reshape(-1, dim, h, w)
        output_feat_real = output_real.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        output_feat_real = output_feat_real.reshape(-1, dim, h, w)
        output_feat_imag = output_imag.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        output_feat_imag = output_feat_imag.reshape(-1, dim, h, w)

        output = torch.cat([output_feat_real, output_feat_imag], 1) #dim维concat

        # return output_real, output_imag, output_feat_real, output_feat_imag, output
        return output_real, output_imag  #, output_feat_real, output_feat_imag, output


# class TransformerDecoderLayer(nn.Module):
class ComTransformerDecoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn  #multihead_attn = MultiheadAttention(feature_dim=d_model=512, n_head=1, key_feature_dim=128)
        # self.cross_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        # self.cross_attn = ComMultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        self.cross_attn = ComSimMultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)

        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor * pos

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
    def forward(self, tgt_real, tgt_imag, memory_real, memory_imag, input_shape, pos_real: Optional[Tensor] = None, pos_imag: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        # self-attention
        # query = tgt
        # key = tgt
        # value = tgt
        query_real = tgt_real
        key_real = tgt_real
        value_real = tgt_real
        query_imag = tgt_imag
        key_imag = tgt_imag
        value_imag = tgt_imag

        # tgt2 = self.self_attn(query=query, key=key, value=value)  #tgt2 = tgt_self_atten
        # tgt = tgt + tgt2                                          #tgt22 = tgt_self_res_atten
        # tgt = self.instance_norm(tgt, input_shape)    #tgt.shape:(484,1,512) # tgt222 = tgt_self_res_norm_atten
        #
        # mask = self.cross_attn(query=tgt, key=memory, value=pos)    #mask = tgt222 & memory & pos  #mask.shape:(484,1,512)
        # tgt2 = tgt * mask                                           #mask & tgt222 = mask * tgt222
        # tgt2 = self.instance_norm(tgt2, input_shape)                #norm(mask & tgt222)
        #
        # tgt3 = self.cross_attn(query=tgt, key=memory, value=memory*pos)   #tgt3 =tgt222 & memory & memory*pos     #tgt3.shape:(484,1,512)
        # tgt4 = tgt + tgt3                                                 #tgt4 = tgt3 + tgt222
        # tgt4 = self.instance_norm(tgt4, input_shape)                      #norm(tgt4)
        #
        # tgt = tgt2 + tgt4                                                 #out = norm (norm(mask & tgt222)+norm(tgt3 + tgt222))
        # tgt = self.instance_norm(tgt, input_shape)

        tgt2_real, tgt2_imag = self.self_attn(query_real=query_real, query_imag=query_imag, key_real=key_real, key_imag=key_imag, value_real=value_real, value_imag=value_imag)  #tgt2 = tgt_self_atten
        tgt_real = tgt_real + tgt2_real                                          #tgt22 = tgt_self_res_atten
        tgt_imag = tgt_imag + tgt2_imag                                          #tgt22 = tgt_self_res_atten
        tgt_real = self.instance_norm(tgt_real, input_shape)    #tgt.shape:(484,1,512) # tgt222 = tgt_self_res_norm_atten
        tgt_imag = self.instance_norm(tgt_imag, input_shape)    #tgt.shape:(484,1,512) # tgt222 = tgt_self_res_norm_atten

        # mask = self.cross_attn(query=tgt, key=memory, value=pos)    #mask = tgt222 & memory & pos  #mask.shape:(484,1,512)
        mask_real, mask_imag = self.cross_attn(query_real=tgt_real, query_imag=tgt_imag, key_real=memory_real, key_imag=memory_imag, value_real=pos_real, value_imag=pos_imag)    #mask = tgt222 & memory & pos  #mask.shape:(484,1,512)
        # tgt2 = tgt * mask                                           #mask & tgt222 = mask * tgt222
        tgt2_real = tgt_real * mask_real                                           #mask & tgt222 = mask * tgt222
        tgt2_imag = tgt_imag * mask_imag                                         #mask & tgt222 = mask * tgt222
        tgt2_real = self.instance_norm(tgt2_real, input_shape)                #norm(mask & tgt222)
        tgt2_imag = self.instance_norm(tgt2_imag, input_shape)                #norm(mask & tgt222)

        # tgt3 = self.cross_attn(query=tgt, key=memory, value=memory*pos)   #tgt3 =tgt222 & memory & memory*pos     #tgt3.shape:(484,1,512)
        tgt3_real, tgt3_imag = self.cross_attn(query_real=tgt_real, query_imag=tgt_imag, key_real=memory_real, key_imag=memory_imag, value_real=memory_real*pos_real, value_imag=memory_imag*pos_imag)   #tgt3 =tgt222 & memory & memory*pos     #tgt3.shape:(484,1,512)
        tgt4_real = tgt_real + tgt3_real                                                 #tgt4 = tgt3 + tgt222
        tgt4_imag = tgt_imag + tgt3_imag                                                 #tgt4 = tgt3 + tgt222
        tgt4_real = self.instance_norm(tgt4_real, input_shape)                      #norm(tgt4)
        tgt4_imag = self.instance_norm(tgt4_imag, input_shape)                      #norm(tgt4)

        tgt_real = tgt2_real + tgt4_real                                               #out = norm (norm(mask & tgt222)+norm(tgt3 + tgt222))
        tgt_imag = tgt2_imag + tgt4_imag                                                 #out = norm (norm(mask & tgt222)+norm(tgt3 + tgt222))
        tgt_real = self.instance_norm(tgt_real, input_shape)
        tgt_imag = self.instance_norm(tgt_imag, input_shape)

        # return tgt
        return tgt_real, tgt_imag


# class TransformerDecoder(nn.Module):
class ComTransformerDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_decoder_layers=6, activation="relu"):
        super().__init__()
        # decoder_layer = TransformerDecoderLayer(multihead_attn, FFN, d_model)
        decoder_layer = ComTransformerDecoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        # self.post1 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        # self.activation = _get_activation_fn(activation)
        # self.post2 = nn.Conv2d(d_model, 1, kernel_size=3, stride=1, padding=1)


    # def forward(self, tgt, memory, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
    def forward(self, tgt_real, tgt_imag, memory_real, memory_imag, pos_real: Optional[Tensor] = None, pos_imag: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        assert tgt_real.dim() == 5, 'Expect 5 dimensional inputs'
        tgt_real_shape = tgt_real.shape
        num_imgs, batch, dim, h, w = tgt_real.shape

        if pos_real is not None:
            num_pos, batch, h, w = pos_real.shape
            pos_real = pos_real.view(num_pos, batch, 1, -1).permute(0,3,1,2)
            pos_real = pos_real.reshape(-1, batch, 1)
            pos_real = pos_real.repeat(1, 1, dim) ##扩增pos维度至dim?
            pos_imag = pos_imag.view(num_pos, batch, 1, -1).permute(0,3,1,2)
            pos_imag = pos_imag.reshape(-1, batch, 1)
            pos_imag = pos_imag.repeat(1, 1, dim)

        # tgt = tgt.view(num_imgs, batch, dim, -1).permute(0,3,1,2)
        # tgt = tgt.reshape(-1, batch, dim)
        tgt_real = tgt_real.view(num_imgs, batch, dim, -1).permute(0,3,1,2)
        tgt_real = tgt_real.reshape(-1, batch, dim)
        tgt_imag = tgt_imag.view(num_imgs, batch, dim, -1).permute(0,3,1,2)
        tgt_imag = tgt_imag.reshape(-1, batch, dim)

        # memory_real = memory_real.view()

        # output = tgt  换成input
        input = torch.cat([tgt_real, tgt_imag], 2)

        for layer in self.layers:
            # output = layer(output, memory, input_shape=tgt_shape, pos=pos, query_pos=query_pos)  # diff in input from encoder
            output_real, output_imag = layer(tgt_real, tgt_imag, memory_real, memory_imag, input_shape=tgt_real_shape, pos_real=pos_real, pos_imag=pos_imag, query_pos=query_pos)  # diff in input from encoder

        # [L,B,D] -> [B,D,L]
        # output_feat = output.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        # output_feat = output_feat.reshape(-1, dim, h, w)
        output_feat_real = output_real.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        output_feat_real = output_feat_real.reshape(-1, dim, h, w)
        output_feat_imag = output_imag.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
        output_feat_imag = output_feat_imag.reshape(-1, dim, h, w)
        output = torch.cat([output_feat_real, output_feat_imag], 1) #dim维concat
        # output = self.post2(self.activation(self.post1(output)))
        # return output, output_feat
        return output_real, output_imag, output_feat_real, output_feat_imag, output


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


