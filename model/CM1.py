import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import time
import json
import torch.nn.init as init
from torch.autograd import Function
from typing import Dict, List, Union
from importlib import import_module
import math
from asteroid_filterbanks import Encoder, ParamSincFB



class Bottle2neck(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=(2,3),
        dilation=1,
        scale=8,
        pool=(1,3),
    ):
        super().__init__()

        self.pool = pool
        width = int(math.floor(planes / scale))
        # print("width",width)  4
        self.conv1 = nn.Conv2d(in_channels=inplanes,
                               out_channels=width * scale,
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale - 1

        convs = []
        bns = []
        # num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width,
                    width,
                    kernel_size=(1,3),
                    # dilation=dilation,
                    padding=(0, 1),
                )
            )
            bns.append(nn.BatchNorm2d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(in_channels=width * scale,
                               out_channels=planes,
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.mp = nn.MaxPool2d(self.pool)
        self.afms = AFMS(planes)

        if inplanes != planes:  # if change in number of filters
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels=inplanes,
                       out_channels=planes,
                       padding=(0, 1),
                       kernel_size=(1, 3),
                       stride=1)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        # print("spx[0].shape", spx[0].shape)   torch.Size([1, 4, 24, 21490])

        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
                # print("sp.shape",sp.shape)   torch.Size([1, 4, 24, 21490])
            else:

                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            # print("sp.shape", sp.shape)  torch.Size([1, 4, 24, 21490])
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out += residual
        if self.mp:
            out = self.mp(out)
        out = self.afms(out)

        return out

class AFMS(nn.Module):
    """
    Alpha-Feature map scaling, added to the output of each residual block[1,2].

    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] AMFS    : https://www.koreascience.or.kr/article/JAKO202029757857763.page
    """

    def __init__(self, nb_dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print("x.shape", x.shape)  torch.Size([1, 32, 23, 7163])
        y = F.adaptive_avg_pool2d(x, (1,1)).view(x.size(0), -1)
        # print("y.shape", y.shape) torch.Size([1, 32])
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), 1, -1)
        # print("y.shape", y.shape) torch.Size([1, 32, 1, 1])
        # print("self.alpha.shape", self.alpha.shape) torch.Size([32, 1, 1])
        x = x + self.alpha
        # print("x.shape", x.shape)  torch.Size([1, 32, 23, 7163])
        x = x * y
        # print("x.shape", x.shape)  torch.Size([1, 32, 23, 7163])
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k, in_dim, p):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)

        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        n_nodes = max(int(h.size(1) * k), 2)
        n_feat = h.size(2)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(
                self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))  # self.mp = nn.MaxPool2d((1,4))
        self.afms = AFMS(nb_filts[1])

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        # print('out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        #out = self.afms(out)
        return out


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        self.preE = False
        self.mff = True
        self.gff = True
        self.d_args = d_args
        filts = d_args["filts"]
        self.conv_time = CONV(out_channels=filts[0],
                              kernel_size=d_args["first_conv"],
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        if self.mff == True:
            self.Tlayer1 = Residual_block(nb_filts=filts[1], first=True)  # 1 32
            self.Tlayer2 = Residual_block(nb_filts=filts[2])  # 32 32
            self.Tlayer3 = Residual_block(nb_filts=filts[3])  # 32 64
            self.Tlayer4 = Residual_block(nb_filts=filts[4])  # 64 64
            self.Tlayer5 = Residual_block(nb_filts=filts[4])
            self.Tlayer6 = Residual_block(nb_filts=filts[4])

            self.Slayer1 = Residual_block(nb_filts=filts[1], first=True)  # 1 32
            self.Slayer2 = Residual_block(nb_filts=filts[2])  # 32 32
            self.Slayer3 = Residual_block(nb_filts=filts[3])  # 32 64
            self.Slayer4 = Residual_block(nb_filts=filts[4])  # 64 64
            self.Slayer5 = Residual_block(nb_filts=filts[4])
            self.Slayer6 = Residual_block(nb_filts=filts[4])

            self.Slayer6_1 = Bottle2neck(filts[4][0], filts[4][1], scale=8)
            self.Slayer2_3 = Bottle2neck(32, 64, scale=8, pool=(1, 9))
            self.Slayer4_5 = Bottle2neck(64, 64, scale=8, pool=(1, 9))

            self.Tlayer6_1 = Bottle2neck(filts[4][0], filts[4][1], scale=8)
            self.Tlayer2_3 = Bottle2neck(32, 64, scale=8, pool=(1, 9))
            self.Tlayer4_5 = Bottle2neck(64, 64, scale=8, pool=(1, 9))

        elif self.mff == False:
            self.encoder_T = nn.Sequential(
                nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
                nn.Sequential(Residual_block(nb_filts=filts[2])),
                nn.Sequential(Residual_block(nb_filts=filts[3])),
                nn.Sequential(Residual_block(nb_filts=filts[4])),
                nn.Sequential(Residual_block(nb_filts=filts[4])),
                nn.Sequential(Residual_block(nb_filts=filts[4])))
            self.encoder_S = nn.Sequential(
                nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
                nn.Sequential(Residual_block(nb_filts=filts[2])),
                nn.Sequential(Residual_block(nb_filts=filts[3])),
                nn.Sequential(Residual_block(nb_filts=filts[4])),
                nn.Sequential(Residual_block(nb_filts=filts[4])),
                nn.Sequential(Residual_block(nb_filts=filts[4])))
        if self.gff == True:
            self.GAT_layer_T = GraphAttentionLayer(192, 32)
            self.GAT_layer_S = GraphAttentionLayer(192, 32)
        else:    
            self.GAT_layer_T = GraphAttentionLayer(64, 32)
            self.GAT_layer_S = GraphAttentionLayer(64, 32)
        self.GAT_layer_ST = GraphAttentionLayer(32, 16)

        self.pool_S = GraphPool(0.64, 32, 0.3)
        self.pool_T = GraphPool(0.81, 32, 0.3)
        self.pool_ST = GraphPool(0.64, 16, 0.3)

        self.proj_S = nn.Linear(14, 12)
        self.proj_T = nn.Linear(23, 12)
        self.proj_ST = nn.Linear(16, 1)
        self.out_layer = nn.Linear(7, 2)

    def forward(self, x, Freq_aug=False):
        nb_samp1 = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp1, 1, len_seq)

        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        if self.mff==True:        
            Sx1 = self.Slayer1(x)
            
            Sx2_3 = self.Slayer2_3(Sx1)
            Sx2 = self.Slayer2(Sx1)
            Sx3 = self.Slayer3(Sx2)
            Smergex2_33 = (Sx2_3 + Sx3) / 2
            
            Sx4_5 = self.Slayer4_5(Smergex2_33)
            Sx4 = self.Slayer4(Smergex2_33)
            Sx5 = self.Slayer5(Sx4)
            Smergex4_55 = (Sx4_5 + Sx5) / 2
            
            Sx6 = self.Slayer6(Smergex4_55)
            Sx6_1 = self.Slayer6_1(Smergex4_55)
            Smergex6 = (Sx6 + Sx6_1) / 2
            e_S = Smergex6
        
            #e_T = e_S
        elif self.mff==False:
            e_S = self.encoder_S(x)  # (#bs, #filt, #spec, #seq)
        # print("e_S.shape", e_S.shape)  torch.Size([1, 64, 23, 29])
        ts = e_S.size()[2]
        tt = e_S.size()[3]
        if self.gff == True:
            e_S = torch.cat((e_S, torch.mean(e_S, dim=3, keepdim=True).repeat(1, 1, 1, tt),
                           torch.sqrt(torch.var(e_S, dim=3, keepdim=True).clamp(min=1e-4)).repeat(1, 1, 1, tt)), dim=1)
            #e_T = e_S
        e_S, _ = torch.max(torch.abs(e_S), dim=3)  # max along time
        # print("e_S.shape",e_S.shape)  torch.Size([1, 64, 23])
        gat_S = self.GAT_layer_S(e_S.transpose(1, 2))
        pool_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        out_S = self.proj_S(pool_S.transpose(1, 2))   #14--12

        if self.mff == True:
            Tx1 = self.Tlayer1(x)

            Tx2_3 = self.Tlayer2_3(Tx1)
            Tx2 = self.Tlayer2(Tx1)
            Tx3 = self.Tlayer3(Tx2)
            Tmergex2_33 = (Tx2_3 + Tx3) / 2

            Tx4_5 = self.Tlayer4_5(Tmergex2_33)
            Tx4 = self.Tlayer4(Tmergex2_33)
            Tx5 = self.Tlayer5(Tx4)
            Tmergex4_55 = (Tx4_5 + Tx5) / 2

            Tx6 = self.Tlayer6(Tmergex4_55)
            Tx6_1 = self.Tlayer6_1(Tmergex4_55)
            Tmergex6 = (Tx6 + Tx6_1) / 2
            e_T = Tmergex6
        elif self.mff == False:
            e_T = self.encoder_T(x)
        if self.gff == True:
            e_T = torch.cat((e_T, torch.mean(e_T, dim=2, keepdim=True).repeat(1, 1, ts, 1),
                           torch.sqrt(torch.var(e_T, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, ts, 1)),dim=1)
        e_T, _ = torch.max(torch.abs(e_T), dim=2)  # max along freq
        # print("e_T.shape", e_T.shape)  torch.Size([1, 64, 29])
        gat_T = self.GAT_layer_T(e_T.transpose(1, 2))
        pool_T = self.pool_T(gat_T)

        out_T = self.proj_T(pool_T.transpose(1, 2))   #23--12

        gat_ST = torch.mul(out_T, out_S)

        gat_ST = self.GAT_layer_ST(gat_ST.transpose(1, 2))
        pool_ST = self.pool_ST(gat_ST)
        proj_ST = self.proj_ST(pool_ST).flatten(1)
        output = self.out_layer(proj_ST)

        return proj_ST, output

if __name__ == "__main__":
    torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    with open('../config/RawGATST_baseline.conf', "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]


    def get_model(model_config: Dict, device: torch.device):
        """Define DNN model architecture"""
        module = import_module("models.{}".format(model_config["architecture"]))
        _model = getattr(module, "Model")
        model = _model(model_config).to(device)
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        print("no. model params:{}".format(nb_params))
        return model

    model1 = get_model(model_config, device)

    x = torch.randn(1, 64600).to(device)

    # flops, params = profile(model1, inputs=(x,))
    # print(f"FLOPs: {flops}, Params: {params}")
    start = time.time()
    print(start)
    last_hidden, output = model1(x)
    # print("lasthidden,shape:",last_hidden.shape)
    # classifier_out = classifier(last_hidden)
    # # output = F.softmax(output, dim=1)
    end = time.time()
    # print(classifier_out)
    print(end)
    print(end - start)
    print(output.shape)
