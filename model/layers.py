import torch
import torch.nn as nn
import torch.nn.functional as F

# from thop import profile
import numpy as np
import math
import sys
sys.path.append('../')
from model.utils import import_class, conv_branch_init, conv_init, bn_init


class Basic_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, residual, act=nn.ReLU(inplace=True), **kwargs):
        super(Basic_Layer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

        self.residual = nn.Identity() if residual else Zero_Layer()
        self.act = act

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        return x


class Basic_TCN_layer(Basic_Layer):
    def __init__(self, channels, kernel_size=9, stride=1, residual=True, **kwargs):
        super(Basic_TCN_layer, self).__init__(channels, channels, residual, **kwargs)
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(channels)

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(channels),
            )


class MS_TCN_layer(nn.Module):
    def __init__(self, channels, t_scale, kernel_size=9, stride=1, expand_ratio=0, residual=True,
                 act=nn.ReLU(inplace=True), layer_type='normal', **kwargs):
        super(MS_TCN_layer, self).__init__()
        assert channels % t_scale == 0
        pad = int((kernel_size - 1) / 2)
        self.stride = stride
        self.t_scale = t_scale
        self.act = act

        self.bn_in = nn.BatchNorm2d(channels)
        self.conv_bn = nn.ModuleList()

        for i in range(t_scale):
            if layer_type == 'normal':
                self.conv_bn.append(nn.Sequential(
                    nn.Conv2d(channels // t_scale, channels // t_scale, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1)),
                    nn.BatchNorm2d(channels // t_scale),
                ))
            elif layer_type == 'sep':
                self.conv_bn.append(
                    Temporal_Sep_Layer(channels // t_scale, kernel_size=kernel_size, stride=stride,
                                       expand_ratio=expand_ratio, residual=False, act=act)
                )
            else:
                self.conv_bn.append(nn.Sequential(
                    nn.Conv2d(channels // t_scale, channels // t_scale, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1)),
                    nn.BatchNorm2d(channels // t_scale),
                ))

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        res = self.residual(x)
        n, c, t, v = x.size()
        spx = torch.split(x, c // self.t_scale, 1)
        for i in range(self.t_scale):
            if i == 0 or self.stride != 1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.conv_bn[i](sp)
            x = sp if i == 0 else torch.cat((x, sp), 1)

        return self.act(x + res)


class Temporal_Bottleneck_Layer(nn.Module):
    def __init__(self, channels, kernel_size=9, stride=1, reduct_ratio=2, residual=True,
                 act=nn.ReLU(inplace=True), **kwargs):
        super(Temporal_Bottleneck_Layer, self).__init__()
        inner_channel = channels // reduct_ratio
        pad = int((kernel_size - 1) / 2)
        self.act = act

        self.reduct_conv = nn.Sequential(
            nn.Conv2d(channels, inner_channel, 1),
            nn.BatchNorm2d(inner_channel),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (kernel_size, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(inner_channel),
        )
        self.expand_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channels, 1),
            nn.BatchNorm2d(channels),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))
        x = self.act(self.conv(x))
        x = self.act(self.expand_conv(x) + res)
        return x


class MS_Temporal_Bottleneck_Layer(nn.Module):
    def __init__(self, channels, t_scale, kernel_size=9, stride=1, reduct_ratio=2, residual=True,
                 act=nn.ReLU(inplace=True), **kwargs):
        super(MS_Temporal_Bottleneck_Layer, self).__init__()
        inner_channel = channels // reduct_ratio
        assert inner_channel % t_scale == 0
        pad = int((kernel_size - 1) / 2)
        self.t_scale = t_scale
        self.stride = stride
        self.act = act

        self.reduct_conv = nn.Sequential(
            nn.Conv2d(channels, inner_channel, 1),
            nn.BatchNorm2d(inner_channel),
        )
        self.convs = nn.ModuleList()
        for i in range(t_scale):
            self.convs.append(nn.Sequential(
                nn.Conv2d(inner_channel // t_scale, inner_channel // t_scale, kernel_size=(kernel_size, 1),
                          padding=(pad, 0), stride=(stride, 1)),
                nn.BatchNorm2d(inner_channel // t_scale),
            ))
        self.expand_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channels, 1),
            nn.BatchNorm2d(channels),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))

        n, c, t, v = x.size()
        spx = torch.split(x, c // self.t_scale, 1)
        for i in range(self.t_scale):
            if i == 0 or self.stride != 1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.act(self.convs[i](sp))
            x = sp if i == 0 else torch.cat((x, sp), 1)

        x = self.act(self.expand_conv(x) + res)
        return x


class Temporal_Sep_Layer(nn.Module):
    def __init__(self, channels, kernel_size=9, stride=1, expand_ratio=0, residual=True,
                 act=nn.ReLU(inplace=True), **kwargs):
        super(Temporal_Sep_Layer, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.act = act

        if expand_ratio > 0:
            inner_channel = channels * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channels, inner_channel, 1),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = channels
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (kernel_size, 1), (stride, 1), (pad, 0), groups=inner_channel),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channels, 1),
            nn.BatchNorm2d(channels),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride, 1)),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


class Basic_GCN_layer(Basic_Layer):
    def __init__(self, in_channels, out_channels, A, residual, **kwargs):
        super(Basic_GCN_layer, self).__init__(in_channels, out_channels, residual, **kwargs)

        self.conv = SpatialGraphCov(in_channels, out_channels, A, **kwargs)

        if not residual:
            self.residual = Zero_Layer()
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )


class MS_GCN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, s_scale, A, expand_ratio, layer_type, residual, act, **kwargs):
        super(MS_GCN_layer, self).__init__()
        self.s_scale = s_scale
        self.act = act
        self.mode = 'basic' if in_channels == out_channels else 'stage'

        self.bn_in = nn.BatchNorm2d(in_channels)
        self.conv_bn = nn.ModuleList()

        for i in range(s_scale):
            if layer_type == 'normal':
                self.conv_bn.append(nn.Sequential(
                    SpatialGraphCov(in_channels // s_scale, out_channels // s_scale, A, **kwargs),
                    nn.BatchNorm2d(out_channels // s_scale),
                ))
            elif layer_type == 'sep':
                self.conv_bn.append(
                    Spatial_Sep_Layer(in_channels // s_scale, out_channels // s_scale, A,
                                      expand_ratio=expand_ratio, residual=False, act=act, **kwargs),
                )
            else:
                self.conv_bn.append(nn.Sequential(
                    SpatialGraphCov(in_channels // s_scale, out_channels // s_scale, A, **kwargs),
                    nn.BatchNorm2d(out_channels // s_scale),
                ))


        if not residual:
            self.residual = Zero_Layer()
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        n, c, t, v = x.size()
        x = self.bn_in(x)

        spx = torch.split(x, c // self.s_scale, 1)
        for i in range(self.s_scale):
            if i == 0 or self.mode == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.conv_bn[i](sp)
            x = sp if i == 0 else torch.cat((x, sp), 1)

        return self.act(x + res)


class Spatial_Bottleneck_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, A, reduct_ratio=2, residual=True,
                 act=nn.ReLU(inplace=True), **kwargs):
        super(Spatial_Bottleneck_Layer, self).__init__()
        inner_channel = in_channels // reduct_ratio if in_channels == out_channels else in_channels * 2 // reduct_ratio
        self.act = act

        self.reduct_conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, 1),
            nn.BatchNorm2d(inner_channel),
        )
        self.conv = nn.Sequential(
            SpatialGraphCov(inner_channel, inner_channel, A),
            nn.BatchNorm2d(inner_channel),
        )
        self.expand_conv = nn.Sequential(
            nn.Conv2d(inner_channel, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))
        x = self.act(self.conv(x))
        x = self.act(self.expand_conv(x) + res)
        return x


class MS_Spatial_Bottleneck_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, s_scale, A, reduct_ratio=2, residual=True,
                 act=nn.ReLU(inplace=True), **kwargs):
        super(MS_Spatial_Bottleneck_Layer, self).__init__()
        inner_channel = in_channels // reduct_ratio if in_channels == out_channels else in_channels * 2 // reduct_ratio
        self.act = act
        self.s_scale = s_scale
        self.mode = 'stage' if in_channels != out_channels else 'basic'

        self.reduct_conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, 1),
            nn.BatchNorm2d(inner_channel),
        )
        self.conv = nn.ModuleList()
        for i in range(s_scale):
            self.conv.append(nn.Sequential(
                SpatialGraphCov(inner_channel // s_scale, inner_channel // s_scale, A),
                nn.BatchNorm2d(inner_channel // s_scale),
            ))
        self.expand_conv = nn.Sequential(
            nn.Conv2d(inner_channel, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))

        n, c, t, v = x.size()
        spx = torch.split(x, c // self.s_scale, 1)
        for i in range(self.s_scale):
            if i == 0 or self.mode == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.act(self.conv[i](sp))
            x = sp if i == 0 else torch.cat((x, sp), 1)

        x = self.act(self.expand_conv(x) + res)
        return x


class SpatialGraphCov(nn.Module):
    def __init__(self, in_channels, out_channels, A, **kwargs):
        super(SpatialGraphCov, self).__init__()
        self.num_subset = A.shape[0]

        self.gcn = nn.Conv2d(in_channels, out_channels * A.shape[0], kernel_size=1)

        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)  # non-constraint
        A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

    def forward(self, x):
        A = self.A + self.PA  # mask A
        n, c, t, v = x.size()
        # perform gcn
        x = self.gcn(x).view(n, self.num_subset, -1, t, v)  # update
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # aggregation

        return x


class Spatial_Sep_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, A, expand_ratio=0, residual=True,
                 act=nn.ReLU(inplace=True), **kwargs):
        super(Spatial_Sep_Layer, self).__init__()

        self.act = act
        self.num_subset = A.shape[0]

        if expand_ratio > 0:
            inner_channel = in_channels * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, inner_channel, 1),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = in_channels
            self.expand_conv = None

        self.depth_gcn = nn.Conv2d(inner_channel, inner_channel * A.shape[0], kernel_size=1, groups=inner_channel)
        self.depth_bn = nn.BatchNorm2d(inner_channel)
        self.point_gcn_bn = nn.Sequential(
            nn.Conv2d(inner_channel, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)  # non-constraint
        A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        if not residual:
            self.residual = Zero_Layer()
        elif in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)

        A = self.A + self.PA  # mask A

        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))

        n, c, t, v = x.size()
        # perform gcn
        # dw_update
        x = self.depth_gcn(x).view(n, -1, self.num_subset, t, v)  # nctv -> nc'ktv
        # aggregation
        x = torch.einsum('ncktv,kvw->nctw', (x, A))
        # dw_bn
        x = self.act(self.depth_bn(x))
        # pw_gcn
        x = self.point_gcn_bn(x)

        return x + res

class Zero_Layer(nn.Module):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0


if __name__ == '__main__':
    from thop import profile
    __layers__ = {
        #'Basic_Layer': Basic_Layer,
        'Basic_TCN_layer': Basic_TCN_layer,
        'Temporal_Bottleneck_Layer': Temporal_Bottleneck_Layer,
        'Temporal_Sep_Layer': Temporal_Sep_Layer,
        'MS_TCN_layer': MS_TCN_layer,
        'MS_Temporal_Bottleneck_Layer': MS_Temporal_Bottleneck_Layer,
        'Basic_GCN_layer': Basic_GCN_layer,
        'Spatial_Bottleneck_Layer': Spatial_Bottleneck_Layer,
        'SpatialGraphCov': SpatialGraphCov,
        'Spatial_Sep_Layer': Spatial_Sep_Layer,
        'MS_GCN_layer': MS_GCN_layer,
        'MS_Spatial_Bottleneck_Layer': MS_Spatial_Bottleneck_Layer
    }
    N, C, T, V = 2, 64, 300, 25
    inputs = torch.rand(N, C, T, V)
    A = np.random.rand(3, 25, 25)
    cfgs = {
        'in_channels': 64,
        'out_channels': 128,
        'channels': 64,
        'residual': True,
        'kernel_size': 9,
        'stride': 1,
        'reduct_ratio': 1,
        'expand_ratio': 0,
        'A': A,
        't_scale': 4,
        'layer_type': 'sep',
        'act': nn.Identity(),
        's_scale': 4
    }

    for layer in __layers__.keys():
        print('{0:-^50}'.format(layer))
        model = __layers__[layer](**cfgs)
        outputs = model(inputs)
        print(layer, outputs.size())

        hereflops, params = profile(model, inputs=(inputs,), verbose=False)

        print('# GFlops is {} G'.format(hereflops / 10 ** 9))
        print('# Params is {} M'.format(sum(param.numel() for param in model.parameters()) / 10 ** 6))
