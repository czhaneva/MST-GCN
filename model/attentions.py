import torch
from torch import nn
import sys
sys.path.append('../')

from model.activations import HardSwish

class Attention_Layer(nn.Module):
    def __init__(self, out_channel, att_type, act, **kwargs):
        super(Attention_Layer, self).__init__()

        __attention = {
            'stja': ST_Joint_Att,
            'stcja': STC_Joint_Att,
            'st1ja': ST_Joint_1_Att,
            'stc1ja': STC_Joint_1_Att,
            'pa': Part_Att,
            'ca': Channel_Att,
            'fa': Frame_Att,
            'ja': Joint_Att,
        }

        self.att = __attention[att_type](channel=out_channel, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act

    def forward(self, x):
        res = x
        x = x * self.att(x)
        return self.act(self.bn(x) + res)

class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            HardSwish(inplace=True),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att

class ST_Joint_1_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_1_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            HardSwish(inplace=True),
        )
        self.conv_t = nn.Conv2d(inner_channel, 1, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, 1, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att


class STC_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(STC_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            HardSwish(inplace=True),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_c = nn.Sequential(
            nn.Conv2d(inner_channel, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)  # N, C, T, 1
        x_v = x.mean(2, keepdims=True).transpose(2, 3)  # N, C, 1, V --> N, C, V, 1
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))  # N, C, T+V, 1
        x_t, x_v = torch.split(x_att, [T, V], dim=2)    # N, C, T, 1; N, C, V, 1
        x_c = (self.pooling(x_t) + self.pooling(x_v)) * 0.5     # N, C, 1, 1
        x_t_att = self.conv_t(x_t).sigmoid()            # N, C, T, 1
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()    # N, C, V, 1 --> N, C, 1, V
        x_c_att = self.conv_c(x_c)
        x_att = x_t_att * x_v_att * x_c_att
        return x_att

class STC_Joint_1_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(STC_Joint_1_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            HardSwish(inplace=True),
        )
        self.conv_t = nn.Conv2d(inner_channel, 1, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, 1, kernel_size=1)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_c = nn.Sequential(
            nn.Conv2d(inner_channel, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)  # N, C, T, 1
        x_v = x.mean(2, keepdims=True).transpose(2, 3)  # N, C, 1, V --> N, C, V, 1
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))  # N, C, T+V, 1
        x_t, x_v = torch.split(x_att, [T, V], dim=2)    # N, C, T, 1; N, C, V, 1
        x_c = (self.pooling(x_t) + self.pooling(x_v)) * 0.5     # N, C, 1, 1
        x_t_att = self.conv_t(x_t).sigmoid()            # N, C, T, 1
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()    # N, C, V, 1 --> N, C, 1, V
        x_c_att = self.conv_c(x_c)
        x_att = x_t_att * x_v_att * x_c_att
        return x_att


class Part_Att(nn.Module):
    def __init__(self, channel, parts, reduct_ratio, bias, **kwargs):
        super(Part_Att, self).__init__()

        self.parts = parts
        self.joints = nn.Parameter(self.get_corr_joints(), requires_grad=False)
        inner_channel = channel // reduct_ratio

        self.softmax = nn.Softmax(dim=3)
        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, channel*len(self.parts), kernel_size=1, bias=bias),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x_att = self.softmax(self.fcn(x).view(N, C, 1, len(self.parts)))
        x_att = x_att.index_select(3, self.joints).expand_as(x)
        return x_att

    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [j for i in range(num_joints) for j in range(len(self.parts)) if i in self.parts[j]]
        return torch.LongTensor(joints)


class Channel_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fcn(x)


class Frame_Att(nn.Module):
    def __init__(self, **kwargs):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=(9,1), padding=(4,0))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=2).transpose(1, 2)
        return self.conv(x)


class Joint_Att(nn.Module):
    def __init__(self, parts, **kwargs):
        super(Joint_Att, self).__init__()

        num_joint = sum([len(part) for part in parts])

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_joint, num_joint//2, kernel_size=1),
            nn.BatchNorm2d(num_joint//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_joint//2, num_joint, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fcn(x.transpose(1, 3)).transpose(1, 3)


if __name__ == '__main__':
    N, C, T, V = 2, 64, 300, 25
    x = torch.rand(N, C, T, V)
    import numpy as np

    parts = [
        np.array([5, 6, 7, 8, 22, 23]) - 1,  # left_arm
        np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
        np.array([13, 14, 15, 16]) - 1,  # left_leg
        np.array([17, 18, 19, 20]) - 1,  # right_leg
        np.array([1, 2, 3, 4, 21]) - 1  # torso
    ]

    att_type = ['stja', 'stcja', 'st1ja', 'stc1ja', 'pa', 'ca', 'fa', 'ja']

    for att in att_type:
        model = Attention_Layer(C, att, nn.ReLU(inplace=True), bias=True, reduct_ratio=2, parts=parts)
        y = model(x)
        print('{}: {}'.format(att, y.size()))
