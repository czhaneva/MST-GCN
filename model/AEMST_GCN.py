import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

import sys
sys.path.append('../')
from model.layers import Basic_Layer, Basic_TCN_layer, MS_TCN_layer, Temporal_Bottleneck_Layer, \
    MS_Temporal_Bottleneck_Layer, Temporal_Sep_Layer, Basic_GCN_layer, MS_GCN_layer, Spatial_Bottleneck_Layer, \
    MS_Spatial_Bottleneck_Layer, SpatialGraphCov, Spatial_Sep_Layer
from model.activations import Activations
from model.utils import import_class, conv_branch_init, conv_init, bn_init
from model.attentions import Attention_Layer

# import model.attentions

__block_type__ = {
    'basic': (Basic_GCN_layer, Basic_TCN_layer),
    'bottle': (Spatial_Bottleneck_Layer, Temporal_Bottleneck_Layer),
    'sep': (Spatial_Sep_Layer, Temporal_Sep_Layer),
    'ms': (MS_GCN_layer, MS_TCN_layer),
    'ms_bottle': (MS_Spatial_Bottleneck_Layer, MS_Temporal_Bottleneck_Layer),
}


class Model(nn.Module):
    def __init__(self, num_class, num_point, num_person, block_args, graph, graph_args, kernel_size, block_type, atten,
                 **kwargs):
        super(Model, self).__init__()
        kwargs['act'] = Activations(kwargs['act'])
        atten = None if atten == 'None' else atten
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(num_person * block_args[0][0] * num_point)

        self.layers = nn.ModuleList()

        for i, block in enumerate(block_args):
            if i == 0:
                self.layers.append(MST_GCN_block(in_channels=block[0], out_channels=block[1], residual=block[2],
                                                 kernel_size=kernel_size, stride=block[3], A=A, block_type='basic',
                                                 atten=None, **kwargs))
            else:
                self.layers.append(MST_GCN_block(in_channels=block[0], out_channels=block[1], residual=block[2],
                                                 kernel_size=kernel_size, stride=block[3], A=A, block_type=block_type,
                                                 atten=atten, **kwargs))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_args[-1][1], num_class)

        for m in self.modules():
            if isinstance(m, SpatialGraphCov) or isinstance(m, Spatial_Sep_Layer):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        conv_branch_init(mm, self.graph.A.shape[0])
                    if isinstance(mm, nn.BatchNorm2d):
                        bn_init(mm, 1)
            elif isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # N C T V M --> N M V C T
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        features = x

        x = self.gap(x).view(N, M, -1).mean(dim=1)
        x = self.fc(x)

        return features, x


class MST_GCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, residual, kernel_size, stride, A, block_type, atten, **kwargs):
        super(MST_GCN_block, self).__init__()
        self.atten = atten
        self.msgcn = __block_type__[block_type][0](in_channels=in_channels, out_channels=out_channels, A=A,
                                                   residual=residual, **kwargs)
        self.mstcn = __block_type__[block_type][1](channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                   residual=residual, **kwargs)
        if atten is not None:
            self.att = Attention_Layer(out_channels, atten, **kwargs)

    def forward(self, x):
        return self.att(self.mstcn(self.msgcn(x))) if self.atten is not None else self.mstcn(self.msgcn(x))


if __name__ == '__main__':
    import sys
    import time

    parts = [
        np.array([5, 6, 7, 8, 22, 23]) - 1,  # left_arm
        np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
        np.array([13, 14, 15, 16]) - 1,  # left_leg
        np.array([17, 18, 19, 20]) - 1,  # right_leg
        np.array([1, 2, 3, 4, 21]) - 1  # torso
    ]

    warmup_iter = 3
    test_iter = 10
    sys.path.append('/home/chenzhan/mywork/MST-GCN/')
    from thop import profile
    basic_channels = 112
    cfgs = {
        'num_class': 2,
        'num_point': 25,
        'num_person': 1,
        'block_args': [[2, basic_channels, False, 1],
                       [basic_channels, basic_channels, True, 1], [basic_channels, basic_channels, True, 1], [basic_channels, basic_channels, True, 1],
                       [basic_channels, basic_channels*2, True, 1], [basic_channels*2, basic_channels*2, True, 1], [basic_channels*2, basic_channels*2, True, 1],
                       [basic_channels*2, basic_channels*4, True, 1], [basic_channels*4, basic_channels*4, True, 1], [basic_channels*4, basic_channels*4, True, 1]],
        'graph': 'graph.ntu_rgb_d.Graph',
        'graph_args': {'labeling_mode': 'spatial'},
        'kernel_size': 9,
        'block_type': 'ms',
        'reduct_ratio': 2,
        'expand_ratio': 0,
        't_scale': 4,
        'layer_type': 'sep',
        'act': 'relu',
        's_scale': 4,
        'atten': 'stcja',
        'bias': True,
        'parts': parts
    }

    model = Model(**cfgs)

    N, C, T, V, M = 4, 2, 16, 25, 1
    inputs = torch.rand(N, C, T, V, M)

    for i in range(warmup_iter + test_iter):
        if i == warmup_iter:
            start_time = time.time()
        outputs = model(inputs)
    end_time = time.time()

    total_time = end_time - start_time
    print('iter_with_CPU: {:.2f} s/{} iters, persample: {:.2f} s/iter '.format(
        total_time, test_iter, total_time/test_iter/N))

    print(outputs.size())

    hereflops, params = profile(model, inputs=(inputs,), verbose=False)
    print('# GFlops is {} G'.format(hereflops / 10 ** 9 / N))
    print('# Params is {} M'.format(sum(param.numel() for param in model.parameters()) / 10 ** 6))




