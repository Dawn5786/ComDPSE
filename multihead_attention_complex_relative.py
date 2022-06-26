import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from ltr.models.layers.normalization import InstanceL2Norm
import numpy as np
import pdb
import re


# class MultiheadAttention(nn.Module):
# class ComSimMultiheadAttention(nn.Module):
class RelaComMultiheadAttention(nn.Module):
    # def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64):  #MultiheadAttention(feature_dim=d_model=512, n_head=1, key_feature_dim=128)
    def __init__(self, feature_dim=64, n_head=8, key_feature_dim=64):  #MultiheadAttention(feature_dim=d_model=512, n_head=1, key_feature_dim=128)
        # super(MultiheadAttention, self).__init__()
        # super(ComSimMultiheadAttention, self).__init__()
        super(RelaComMultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        self.linear = nn.Linear(feature_dim, feature_dim)
        for N in range(self.Nh):
            # self.head.append(RelationUnit(feature_dim, key_feature_dim))    #(512, 128)
            # self.head.append(self.linear)
            # self.head.append(ComSimRelationUnit(feature_dim, key_feature_dim))    #(512, 128)
            self.head.append(RelaComRelationUnit(feature_dim, key_feature_dim))    #(512, 128)
        # self.out_conv = nn.Linear(n_head*key_feature_dim, feature_dim)  # bias=False


    # def forward(self, query=None, key=None, value=None):
    # def forward(self, query_real=None, query_imag=None, key_real=None, key_imag=None, value_real=None, value_imag=None):
    def forward(self, query_real=None, query_imag=None, key_real=None, key_imag=None, value_real=None, value_imag=None,
                emb_real=None, emb_imag=None, uu=None, vv=None):
        isFirst = True
        for N in range(self.Nh):
            if(isFirst):
                # concat = self.head[N](query, key, value)
                # concat_real, concat_imag = self.head[N](query_real, query_imag, key_real, key_imag, value_real, value_imag)
                concat_real, concat_imag = self.head[N](query_real, query_imag, key_real, key_imag, value_real, value_imag,
                                                        emb_real, emb_imag)   #[3,484,4,512], uu, vv
                isFirst = False
            else:
                # concat = torch.cat((concat, self.head[N](query, key, value)), -1)
                # cur_real, cur_imag = self.head[N](query_real, query_imag, key_real, key_imag, value_real, value_imag)
                cur_real, cur_imag = self.head[N](query_real, query_imag, key_real, key_imag, value_real, value_imag,
                                                  emb_real, emb_imag, uu, vv)
                concat_real = torch.cat((concat_real, cur_real), -1)
                concat_imag = torch.cat((concat_imag, cur_imag), -1)
        # output = self.out_conv(concat)
        # output = concat
        output_real = concat_real
        output_imag = concat_imag
        # return output
        return output_real, output_imag  #[3,484,4,512]


# class RelationUnit(nn.Module):
# class ComSimRelationUnit(nn.Module):
class RelaComRelationUnit(nn.Module):
    # def __init__(self, feature_dim=512, key_feature_dim=64):    #(512, 128)
    def __init__(self, feature_dim=64, key_feature_dim=64):    #(512, 128)
        # super(RelationUnit, self).__init__()
        # super(ComSimRelationUnit, self).__init__()
        super(RelaComRelationUnit, self).__init__()
        self.temp = 30
        # self.WK = nn.Linear(feature_dim, key_feature_dim)  # bias=False
        self.WK_real = nn.Linear(feature_dim, feature_dim)  # bias=False
        self.WK_imag = nn.Linear(feature_dim, feature_dim)  # bias=False

        self.WK_R_real = nn.Linear(feature_dim, feature_dim)  # bias=False
        self.WK_R_imag = nn.Linear(feature_dim, feature_dim)  # bias=False
        # self.WQ = nn.Linear(feature_dim, key_feature_dim)
        # self.WV = nn.Linear(feature_dim, feature_dim)
        self.WQ_real = nn.Linear(feature_dim, feature_dim)
        self.WQ_imag = nn.Linear(feature_dim, feature_dim)

        self.insnorm_w_k_real = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=False)
        self.insnorm_w_k_imag = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=False)
        self.insnorm_w_kr_real = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=False)
        self.insnorm_w_kr_imag = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=False)

        self.insnorm_ww_real = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=True)
        self.insnorm_ww_imag = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=True)
        self.insnorm_wr_real = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=True)
        self.insnorm_wr_imag = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=True)
        # Init weights
        # for m in self.WK.modules():
        #     m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
        #     # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        for m in self.WK_real.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WK_imag.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WK_R_real.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WK_R_imag.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        '''
        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        '''
        # for m in self.WV.modules():
        #     m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
        #     # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

        # for m in self.WV_real.modules():
        for m in self.WQ_real.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        # for m in self.W_imag.modules():
        for m in self.WQ_imag.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()


    def bias_generate(self, feat_shape):
        self.ef_bias_real = nn.Parameter(torch.Tensor(feat_shape)).cuda()
        self.ef_bias_imag = nn.Parameter(torch.Tensor(feat_shape)).cuda()
        self.er_bias_real = nn.Parameter(torch.Tensor(feat_shape)).cuda()
        self.er_bias_imag = nn.Parameter(torch.Tensor(feat_shape)).cuda()
        return self.ef_bias_real, self.ef_bias_imag, self.er_bias_real, self.er_bias_imag

    # def instanceNorm(self, src, input_shape):
    #
    #     num_imgs, batch, dim, h, w = input_shape
    #     # Normlization
    #     src = src.reshape(num_imgs, h, w, batch, dim).permute(0,3,4,1,2)
    #     src = src.reshape(-1, dim, h, w)
    #     src = self.norm(src)
    #     # reshape back
    #     src = src.reshape(num_imgs, batch, dim, -1).permute(0,3,1,2)
    #     src = src.reshape(-1, batch, dim)
    #     return src

    # def forward(self, query=None, key=None, value=None):
    # def forward(self, query_real=None, query_imag=None, key_real=None, key_imag=None, value_real=None, value_imag=None):
    def forward(self, query_real=None, query_imag=None, key_real=None, key_imag=None, value_real=None, value_imag=None,
                emb_real=None, emb_imag=None):  #, uu=None, vv=None
        # w_k = self.WK(key)
        # w_k = F.normalize(w_k, p=2, dim=-1)
        # w_k = w_k.permute(1,2,0) # Batch, Dim, Len_1
        #
        # w_q = self.WK(query)
        # w_q = F.normalize(w_q, p=2, dim=-1)
        # w_q = w_q.permute(1,0,2) # Batch, Len_2, Dim
        #
        # dot_prod = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1
        # affinity = F.softmax(dot_prod*self.temp, dim=-1)
        #
        # w_v = value.permute(1,0,2) # Batch, Len_1, Dim
        # output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
        # output = output.permute(1,0,2)
        feat_shape = query_real.shape
        # ef_bias_real, ef_bias_imag, er_bias_real, er_bias_imag = self.bias_generate(feat_shape)
        w_k_11 = self.WK_real(key_real)  #key_real:[22*22*3,4,512]->[1452,4,512]  WK:[512,512]  w_k_1:[1452,4,512]       ##[3, 22*22, 4, 512]
        w_k_12 = self.WK_imag(key_real)
        w_k_21 = self.WK_real(key_imag)
        w_k_22 = self.WK_imag(key_imag)

        w_k_real = w_k_11 - w_k_22
        w_k_imag = w_k_12 + w_k_21
        w_k_real = self.insnorm_w_k_real(w_k_real.permute(0,3,1,2)).permute(0,2,3,1)#+ ef_bias_real   ##[3,484,4,512]
        w_k_imag = self.insnorm_w_k_real(w_k_imag.permute(0,3,1,2)).permute(0,2,3,1)#+ ef_bias_real   ##[3,484,4,512]


        w_kr_11 = self.WK_R_real(emb_real) #emb_real:[3,4,512,484,484]    WK_R:[] #[3,484,484,4,512]    ##[3,484,484,4,512]
        w_kr_12 = self.WK_R_imag(emb_real)
        w_kr_21 = self.WK_R_real(emb_imag)
        w_kr_22 = self.WK_R_imag(emb_imag)

        w_kr_real = w_kr_11 - w_kr_22
        w_kr_imag = w_kr_12 + w_kr_21
        # w_kr_real = self.insnorm_w_kr_real(w_kr_real.permute(0,3,1,2)).permute(0,2,3,1)#+ ef_bias_real   ##[3,484,4,512]
        # w_kr_imag = self.insnorm_w_kr_real(w_kr_imag.permute(0,3,1,2)).permute(0,2,3,1)#+ ef_bias_real   ##[3,484,4,512]

       # w_k = torch.sqrt(w_k_real*w_k_real+w_k_imag*w_k_imag)
        # w_k_real = F.normalize(w_k_real, p=2, dim=-1)
        # w_k_imag = F.normalize(w_k_imag, p=2, dim=-1)
        # w_k = w_k.permute(1,2,0) # Batch, Dim, Len_1
        # w_k_real = w_k_real.permute(1, 2, 0) # Batch, Dim, Len_1
        # w_k_imag = w_k_imag.permute(1, 2, 0) # Batch, Dim, Len_1
        # w_k_real = w_k_real.permute(1, 2, 0) # Batch, Dim, Len_1
        # w_k_imag = w_k_imag.permute(1, 2, 0) # Batch, Dim, Len_1
        # w_kr_real = w_kr_real.permute(1, 2, 0)
        # w_kr_imag = w_kr_imag.permute(1, 2, 0)

        w_q_11 = self.WQ_real(query_real)    ##[3,484,4,512]
        w_q_12 = self.WQ_imag(query_real)
        w_q_21 = self.WQ_real(query_imag)
        w_q_22 = self.WQ_imag(query_imag)

        w_q_real = w_q_11 - w_q_22
        w_q_imag = w_q_12 + w_q_21

        # ww_head_q_real = w_q_real + uu
        # ww_head_q_imag = w_q_imag + uu
        ww_head_q_real = self.insnorm_ww_real(w_q_real.permute(0,3,1,2)).permute(0,2,3,1)#+ ef_bias_real   ##[3,484,4,512]
        ww_head_q_imag = self.insnorm_ww_imag(w_q_imag.permute(0,3,1,2)).permute(0,2,3,1)#+ ef_bias_imag
        # ww_head_q_real = ww_head_q_real.permute(1,0,2)
        # ww_head_q_imag = ww_head_q_imag.permute(1,0,2)

        wr_head_q_real = self.insnorm_wr_real(w_q_real.permute(0,3,1,2)).permute(0,2,3,1) #+ er_bias_real   ##[3,484,4,512]
        wr_head_q_imag = self.insnorm_wr_imag(w_q_imag.permute(0,3,1,2)).permute(0,2,3,1) #+ er_bias_imag
        # wr_head_q_real = ww_head_q_real.permute(1,0,2)
        # wr_head_q_imag = ww_head_q_imag.permute(1,0,2)
        # w_q_real = F.normalize(w_q_real, p=2, dim=-1)
        # w_q_imag = F.normalize(w_q_imag, p=2, dim=-1)
        # w_q_real = w_q_real.permute(1, 0, 2) # Batch, Len_2, Dim
        # w_q_imag = w_q_imag.permute(1, 0, 2) # Batch, Len_2, Dim

        # dot_prod = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1
        # dot_prod_real = torch.einsum('bdp,bqd->bqp', (w_k_real, w_q_real))-torch.einsum('bdp,bqd->bqp', (w_k_imag, w_q_imag)) # Batch, Len_2, Len_1
        # dot_prod_imag = torch.einsum('bdp,bqd->bqp', (w_k_real, w_q_imag))+torch.einsum('bdp,bqd->bqp', (w_k_imag, w_q_real))
        # AC_real = torch.einsum('bdp,bqd->bqp', (w_k_real, ww_head_q_real))-torch.einsum('bdp,bqd->bqp', (w_k_imag, ww_head_q_imag)) # Batch, Len_2, Len_1
        # AC_imag = torch.einsum('bdp,bqd->bqp', (w_k_real, ww_head_q_imag))+torch.einsum('bdp,bqd->bqp', (w_k_imag, ww_head_q_real))
        # BD_real = torch.einsum('bdp,bqd->bqp', (w_kr_real, wr_head_q_real))-torch.einsum('bdp,bqd->bqp', (w_kr_imag, wr_head_q_imag)) # Batch, Len_2, Len_1
        # BD_imag = torch.einsum('bdp,bqd->bqp', (w_kr_real, wr_head_q_imag))+torch.einsum('bdp,bqd->bqp', (w_kr_imag, wr_head_q_real))
        #ww_q[3,484,4,512] w_k[3,484,4,512]   wr_q[3,484,4,512] w_kr[3,484,484,4,512]
        AC_real = torch.einsum('npbd,nqbd->nqpb', (w_k_real, ww_head_q_real))-torch.einsum('npbd,nqbd->nqpb', (w_k_imag, ww_head_q_imag)) # Batch, Len_2, Len_1
        AC_imag = torch.einsum('npbd,nqbd->nqpb', (w_k_real, ww_head_q_imag))+torch.einsum('npbd,nqbd->nqpb', (w_k_imag, ww_head_q_real))
        BD_real = torch.einsum('nqpbd,nqbd->nqpb', (w_kr_real, wr_head_q_real))-torch.einsum('npqbd,nqbd->nqpb', (w_kr_imag, wr_head_q_imag)) # Batch, Len_2, Len_1
        BD_imag = torch.einsum('nqpbd,nqbd->nqpb', (w_kr_real, wr_head_q_imag))+torch.einsum('npqbd,nqbd->nqpb', (w_kr_imag, wr_head_q_real))

        AC = AC_real * AC_real + AC_imag * AC_imag
        AC = torch.sqrt(AC)  #[3,484,484,4]

        BD = BD_real * BD_real + BD_imag * BD_imag
        BD = torch.sqrt(BD)

        affinity = AC + BD

        # dot_prod = dot_prod_real * dot_prod_real + dot_prod_imag * dot_prod_imag
        # dot_prod = torch.sqrt(dot_prod)
        # affinity = F.softmax(dot_prod*self.temp, dim=-1)
        affinity = F.softmax(affinity*self.temp, dim=-1)

        # w_v = value.permute(1,0,2) # Batch, Len_1, Dim
        # output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
        # output = output.permute(1,0,2)

        # w_v_real = value_real.permute(1, 0, 2) # Batch, Len_1, Dim
        # w_v_imag = value_imag.permute(1, 0, 2) # Batch, Len_1, Dim
        # w_v_real = value_real.permute(1, 0, 2) # Batch, Len_1, Dim
        # w_v_imag = value_imag.permute(1, 0, 2) # Batch, Len_1, Dim

        output_real = torch.einsum('nqpb,npbd->nqbd', (affinity,value_real))     #[3,484,4,512]
        output_imag = torch.einsum('nqpb,npbd->nqbd', (affinity,value_imag))
        # output_real = torch.bmm(affinity, w_v_real) # Batch, Len_2, Dim
        # output_real = output_real.permute(1, 0, 2)
        # output_imag = torch.bmm(affinity, w_v_imag) # Batch, Len_2, Dim
        # output_imag = output_imag.permute(1, 0, 2)

        # return output
        return output_real, output_imag


class ComMultiheadAttention(nn.Module):
    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64):  #MultiheadAttention(feature_dim=d_model=512, n_head=1, key_feature_dim=128)
        # super(MultiheadAttention, self).__init__()
        # super(ComSimMultiheadAttention, self).__init__()
        super(ComMultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        self.linear = nn.Linear(feature_dim, feature_dim)
        for N in range(self.Nh):
            # self.head.append(RelationUnit(feature_dim, key_feature_dim))    #(512, 128)
            # self.head.append(self.linear)
            # self.head.append(ComSimRelationUnit(feature_dim, key_feature_dim))    #(512, 128)
            self.head.append(ComSimRelationUnit(feature_dim, key_feature_dim))    #(512, 128)
        # self.out_conv = nn.Linear(n_head*key_feature_dim, feature_dim)  # bias=False


    # def forward(self, query=None, key=None, value=None):
    # def forward(self, query_real=None, query_imag=None, key_real=None, key_imag=None, value_real=None, value_imag=None):
    def forward(self, query_real=None, query_imag=None, key_real=None, key_imag=None, value_real=None, value_imag=None):
        isFirst = True
        for N in range(self.Nh):
            if(isFirst):
                # concat = self.head[N](query, key, value)
                # concat_real, concat_imag = self.head[N](query_real, query_imag, key_real, key_imag, value_real, value_imag)
                concat_real, concat_imag = self.head[N](query_real, query_imag, key_real, key_imag, value_real, value_imag)   #[3,484,4,512]
                isFirst = False
            else:
                # concat = torch.cat((concat, self.head[N](query, key, value)), -1)
                # cur_real, cur_imag = self.head[N](query_real, query_imag, key_real, key_imag, value_real, value_imag)
                cur_real, cur_imag = self.head[N](query_real, query_imag, key_real, key_imag, value_real, value_imag)
                concat_real = torch.cat((concat_real, cur_real), -1)
                concat_imag = torch.cat((concat_imag, cur_imag), -1)
        # output = self.out_conv(concat)
        # output = concat
        output_real = concat_real
        output_imag = concat_imag
        # return output
        return output_real, output_imag  #[3,484,4,512]

class ComSimRelationUnit(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64):  # (512, 128)
        # super(RelationUnit, self).__init__()
        super(ComSimRelationUnit, self).__init__()
        self.temp = 30
        # self.WK = nn.Linear(feature_dim, key_feature_dim)  # bias=False
        self.WK_real = nn.Linear(feature_dim, feature_dim)  # bias=False
        self.WK_imag = nn.Linear(feature_dim, feature_dim)  # bias=False
        # self.WQ = nn.Linear(feature_dim, key_feature_dim)
        # self.WV = nn.Linear(feature_dim, feature_dim)
        self.WV_real = nn.Linear(feature_dim, feature_dim)
        self.WV_imag = nn.Linear(feature_dim, feature_dim)
        self.insnorm_w_k_real = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=False)
        self.insnorm_w_k_imag = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=False)
        self.insnorm_w_kr_real = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=False)
        self.insnorm_w_kr_imag = nn.InstanceNorm2d(feature_dim, eps=1e-05, affine=False)
        # self.LN = nn.LayerNorm(input.size()[1:])  # input.size()[1:]为torch.Size([3, 2, 2])
        # Init weights
        # for m in self.WK.modules():
        #     m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
        #     # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

        for m in self.WK_real.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WK_imag.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        '''
        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        '''

        # for m in self.WV.modules():
        #     m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
        #     # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

        for m in self.WV_real.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WV_imag.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    # def forward(self, query=None, key=None, value=None):
    def forward(self, query_real=None, query_imag=None, key_real=None, key_imag=None, value_real=None, value_imag=None): #[484,4,512]  [3*484,4,512],  [3*484,4,512]
        # w_k = self.WK(key)
        # w_k = F.normalize(w_k, p=2, dim=-1)
        # w_k = w_k.permute(1,2,0) # Batch, Dim, Len_1
        #
        # w_q = self.WK(query)
        # w_q = F.normalize(w_q, p=2, dim=-1)
        # w_q = w_q.permute(1,0,2) # Batch, Len_2, Dim
        #
        # dot_prod = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1
        # affinity = F.softmax(dot_prod*self.temp, dim=-1)
        #
        # w_v = value.permute(1,0,2) # Batch, Len_1, Dim
        # output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
        # output = output.permute(1,0,2)

        w_k_11 = self.WK_real(key_real)   #[3*484,4,512]
        w_k_12 = self.WK_imag(key_real)
        w_k_21 = self.WK_real(key_imag)
        w_k_22 = self.WK_imag(key_imag)

        w_k_real = w_k_11 - w_k_22
        w_k_imag = w_k_12 + w_k_21
        # w_k = torch.sqrt(w_k_real*w_k_real+w_k_imag*w_k_imag)
        # w_k_real = F.normalize(w_k_real, p=2, dim=-1)
        # w_k_imag = F.normalize(w_k_imag, p=2, dim=-1)
        # w_k = w_k.permute(1,2,0) # Batch, Dim, Len_1
        w_k_real = w_k_real.permute(1, 2, 0)  # Batch, Dim, Len_1
        w_k_imag = w_k_imag.permute(1, 2, 0)  # Batch, Dim, Len_1

        w_q_11 = self.WV_real(query_real)     #[1*484,4,512]
        w_q_12 = self.WV_imag(query_real)
        w_q_21 = self.WV_real(query_imag)
        w_q_22 = self.WV_imag(query_imag)

        w_q_real = w_q_11 - w_q_22
        w_q_imag = w_q_12 + w_q_21

        # w_q_real = F.normalize(w_q_real, p=2, dim=-1)
        # w_q_imag = F.normalize(w_q_imag, p=2, dim=-1)
        w_q_real = w_q_real.permute(1, 0, 2)  # Batch, Len_2, Dim
        w_q_imag = w_q_imag.permute(1, 0, 2)  # Batch, Len_2, Dim

        # dot_prod = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1
        dot_prod_real = torch.einsum('bdp,bqd->bqp', (w_k_real, w_q_real)) - torch.einsum('bdp,bqd->bqp', (w_k_imag, w_q_imag))  # Batch, Len_2, Len_1
        dot_prod_imag = torch.einsum('bdp,bqd->bqp', (w_k_real, w_q_imag)) + torch.einsum('bdp,bqd->bqp', (w_k_imag, w_q_real))

        dot_prod = dot_prod_real * dot_prod_real + dot_prod_imag * dot_prod_imag
        dot_prod = torch.sqrt(dot_prod)
        affinity = F.softmax(dot_prod * self.temp, dim=-1) #[1,]

        # w_v = value.permute(1,0,2) # Batch, Len_1, Dim
        # output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
        # output = output.permute(1,0,2)

        w_v_real = value_real.permute(1, 0, 2)  # Batch, Len_1, Dim     [3,484,4,512]
        w_v_imag = value_imag.permute(1, 0, 2)  # Batch, Len_1, Dim

        output_real = torch.bmm(affinity, w_v_real)  # Batch, Len_2, Dim
        output_real = output_real.permute(1, 0, 2)
        output_imag = torch.bmm(affinity, w_v_imag)  # Batch, Len_2, Dim
        output_imag = output_imag.permute(1, 0, 2)

        # return output
        return output_real, output_imag
