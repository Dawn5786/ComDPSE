import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import pdb
import re


# class MultiheadAttention(nn.Module):
class ComSimMultiheadAttention(nn.Module):
    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64):  #MultiheadAttention(feature_dim=d_model=512, n_head=1, key_feature_dim=128)
        # super(MultiheadAttention, self).__init__()
        super(ComSimMultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        self.linear = nn.Linear(feature_dim, feature_dim)
        for N in range(self.Nh):
            # self.head.append(RelationUnit(feature_dim, key_feature_dim))    #(512, 128)
            # self.head.append(self.linear)
            self.head.append(ComSimRelationUnit(feature_dim, key_feature_dim))    #(512, 128)
        # self.out_conv = nn.Linear(n_head*key_feature_dim, feature_dim)  # bias=False

    # def forward(self, query=None, key=None, value=None):
    def forward(self, query_real=None, query_imag=None, key_real=None, key_imag=None, value_real=None, value_imag=None):
        isFirst = True
        for N in range(self.Nh):
            if(isFirst):
                # concat = self.head[N](query, key, value)
                concat_real, concat_imag = self.head[N](query_real, query_imag, key_real, key_imag, value_real, value_imag)
                isFirst = False
            else:
                # concat = torch.cat((concat, self.head[N](query, key, value)), -1)
                cur_real, cur_imag = self.head[N](query_real, query_imag, key_real, key_imag, value_real, value_imag)
                concat_real = torch.cat((concat_real, cur_real), -1)
                concat_imag = torch.cat((concat_imag, cur_imag), -1)
        # output = self.out_conv(concat)
        # output = concat
        output_real = concat_real
        output_imag = concat_imag
        # return output
        return output_real, output_imag


# class RelationUnit(nn.Module):
class ComSimRelationUnit(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64):    #(512, 128)
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
    def forward(self, query_real=None, query_imag=None, key_real=None, key_imag=None, value_real=None, value_imag=None):
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

        w_k_11 = self.WK_real(key_real)
        w_k_12 = self.WK_imag(key_real)
        w_k_21 = self.WK_real(key_imag)
        w_k_22 = self.WK_imag(key_imag)

        w_k_real = w_k_11 - w_k_22
        w_k_imag = w_k_12 + w_k_21
        # w_k = torch.sqrt(w_k_real*w_k_real+w_k_imag*w_k_imag)
        # w_k_real = F.normalize(w_k_real, p=2, dim=-1)
        # w_k_imag = F.normalize(w_k_imag, p=2, dim=-1)
        # w_k = w_k.permute(1,2,0) # Batch, Dim, Len_1
        w_k_real = w_k_real.permute(1, 2, 0) # Batch, Dim, Len_1
        w_k_imag = w_k_imag.permute(1, 2, 0) # Batch, Dim, Len_1

        w_q_11 = self.WV_real(query_real)
        w_q_12 = self.WV_imag(query_real)
        w_q_21 = self.WV_real(query_imag)
        w_q_22 = self.WV_imag(query_imag)

        w_q_real = w_q_11 - w_q_22
        w_q_imag = w_q_12 + w_q_21

        # w_q_real = F.normalize(w_q_real, p=2, dim=-1)
        # w_q_imag = F.normalize(w_q_imag, p=2, dim=-1)
        w_q_real = w_q_real.permute(1, 0, 2) # Batch, Len_2, Dim
        w_q_imag = w_q_imag.permute(1, 0, 2) # Batch, Len_2, Dim

        # dot_prod = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1
        dot_prod_real = torch.einsum('bdp,bqd->bqp',(w_k_real, w_q_real))-torch.einsum('bdp,bqd->bqp',(w_k_imag, w_q_imag)) # Batch, Len_2, Len_1
        dot_prod_imag = torch.einsum('bdp,bqd->bqp',(w_k_real, w_q_imag))+torch.einsum('bdp,bqd->bqp',(w_k_imag, w_q_real))

        dot_prod = dot_prod_real * dot_prod_real + dot_prod_imag * dot_prod_imag
        dot_prod = torch.sqrt(dot_prod)
        affinity = F.softmax(dot_prod*self.temp, dim=-1)

        # w_v = value.permute(1,0,2) # Batch, Len_1, Dim
        # output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
        # output = output.permute(1,0,2)

        w_v_real = value_real.permute(1,0,2) # Batch, Len_1, Dim
        w_v_imag = value_imag.permute(1,0,2) # Batch, Len_1, Dim

        output_real = torch.bmm(affinity, w_v_real) # Batch, Len_2, Dim
        output_real = output_real.permute(1,0,2)
        output_imag = torch.bmm(affinity, w_v_imag) # Batch, Len_2, Dim
        output_imag = output_imag.permute(1,0,2)

        # return output
        return output_real, output_imag

