import torch 
import torch.nn as nn

from model import common

def make_model(args, parent=False):
    return BELCSCNet(args)



class LCSC_Unit(nn.Module):
    def __init__(self, channels, rate, kernel_size):
        super(LCSC_Unit, self).__init__()
        self.channels = channels
        self.rate = rate
        self.kernel_size = kernel_size
        self.nonlinear_filters = int(self.channels * self.rate)
        self.linear_filters = self.channels - self.nonlinear_filters
        self.nonlinear_conv = nn.Conv2d(self.channels, self.nonlinear_filters, self.kernel_size, padding=1, stride=1)
        self.linear_conv = nn.Conv2d(self.channels, self.linear_filters, 1, padding=0, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        linear_output = self.linear_conv(x)
        nonlinear_output = self.relu(x)
        nonlinear_output = self.nonlinear_conv(nonlinear_output)
        return torch.cat((linear_output, nonlinear_output), 1)

class ELCSC_Block(nn.Module):
    def __init__(self, channels, rate, unit_num, kernel_size):
        super(ELCSC_Block, self).__init__()
        self.channels = channels
        self.rate = rate
        self.unit_num = unit_num
        self.kernel_size = kernel_size
        self.LCSC_units = nn.ModuleList()
        for i in range(self.unit_num):
            self.LCSC_units.append(LCSC_Unit(self.channels, self.rate, self.kernel_size))
        self.concat_conv = nn.Conv2d(self.channels*2, self.channels, 1, padding=0, stride=1)

    def forward(self, x):
        init_feature = x
        for i in range(self.unit_num):
            x = self.LCSC_units[i](x)

        x = torch.cat((init_feature, x), 1)
        x = self.concat_conv(x)
        return x




class BELCSCNet(nn.Module):
    def __init__(self, args):
        super(BELCSCNet, self).__init__()
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.default_conv = common.default_conv
        self.use_add = args.lcsc_use_add
        self.channels = args.channels
        self.rate_list = args.rate_list
        self.len_list = args.len_list
        self.kernel_size = args.kernel_size
        self.scale = args.scale[0]
        self.upscale = args.upscale_manner
        self.len = len(self.rate_list)
        self.init_conv = nn.Conv2d(3, self.channels, self.kernel_size, padding=1, stride=1)
        self.ELCSC_blocks = nn.ModuleList()
        for i in range(self.len):
            self.ELCSC_blocks.append(ELCSC_Block(self.channels, self.rate_list[i], self.len_list[i], self.kernel_size))

        assert self.upscale == 'espcn' or self.upscale == 'deconv', "upscaling manner should be espcn or deconv"

        if self.upscale = 'espcn':
            self.up_part = nn.Sequential(
                    common.Upsampler(self.default_conv, self.scale, self.channels, act=False),
                    self.default_conv(self.channels, 3, self.kernel_size)
                    )
        else:
            self.up_part = nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=self.scale),
                    nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(self.channels, 3, self.kernel_size, padding=1, stride=1)
                    )


    def forward(self, x):
        x = self.sub_mean(x)
        x = self.init_conv(x)
        init_feature = x
        for i in range(self.len):
            x = self.ELCSC_blocks[i](x)
        if self.use_add == True:
            x += init_feature

        x = self.up_part(x)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('up_part') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('up_part') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))








