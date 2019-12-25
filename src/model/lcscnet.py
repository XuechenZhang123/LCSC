import torch 
import torch.nn as nn

from model import common

def make_model(args, parent=False):
    return LCSCNet(args)


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

class LCSC_Block(nn.Module):
    def __init__(self, channels, rate, unit_num, kernel_size):
        super(LCSC_Block, self).__init__()
        self.channels = channels
        self.rate = rate
        self.unit_num = unit_num
        self.kernel_size = kernel_size
        self.LCSC_units = nn.ModuleList()
        for i in range(self.unit_num):
            self.LCSC_units.append(LCSC_Unit(self.channels, self.rate, self.kernel_size))

    def forward(self, x):
        for i in range(self.unit_num):
            x = self.LCSC_units[i](x)

        return x


class LCSCNet(nn.Module):
    def __init__(self, args):
        super(LCSCNet, self).__init__()
        self.channels = args.channels
        self.rate_list = args.rate_list
        self.len_list = args.len_list
        self.kernel_size = args.kernel_size
        self.scale = args.scale
        self.multi_out = args.multi_out
        self.len = len(self.rate_list)
        self.init_conv = nn.Conv2d(3, self.channels, self.kernel_size, padding=1, stride=1)
        self.LCSC_blocks = nn.ModuleList()
        for i in range(self.len):
            self.LCSC_blocks.append(LCSC_Block(self.channels, self.rate_list[i], self.len_list[i], self.kernel_size))

        self.up_part = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=1, stride=1),
                nn.ReLU(), 
                nn.Conv2d(self.channels, 3, self.kernel_size, padding=1, stride=1)
                )

        self.weight_layers = nn.ModuleList()
        for i in range(self.len-1):
            self.weight_layers.append(nn.Conv2d(6, 3, 1, padding=0, stride=1))

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.init_conv(x)
        intermediate_output = []
        for i in range(self.len):
            x = self.LCSC_blocks[i](x)
            recon = self.up_part(x)
            intermediate_output.append(recon)

        fused_output = intermediate_output[0]

        for i in range(self.len - 1):
            concat_output = torch.cat((fused_output, intermediate_output[i+1]), 1)
            merge_weight = self.weight_layers[i](concat_output)
            merge_weight = self.sigmoid(merge_weight)
            subtract_output = fused_output - intermediate_output[i+1]
            mul_output = subtract_output.mul(merge_weight)
            fused_output = mul_output + intermediate_output[i+1]

        intermediate_output.insert(0, fused_output)
      
        if self.multi_out == True:
            return intermediate_output
        else:
            return intermediate_output[0]
        



