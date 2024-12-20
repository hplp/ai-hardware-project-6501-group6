import torch
from torch import nn
import torch.nn.functional as F

def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().int()
    return q_x, scale, zero_point

def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x.float() - zero_point)

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_flag = False
        self.scale = None
        self.shift = None
        self.zero_point = None
        self.qx_minus_zeropoint = None
        self.bias_divide_scale = None

    def linear_quant(self, quantize_bit=8):
        self.weight.data, self.scale, self.zero_point = quantize_tensor(self.weight.data, num_bits=quantize_bit)
        self.quant_flag = True

    def load_quant(self, scale, shift, zero_point):
        self.scale = scale
        self.shift = shift
        self.zero_point = zero_point
        self.qx_minus_zeropoint = self.weight - self.zero_point
        self.qx_minus_zeropoint = self.qx_minus_zeropoint.round().int()
        self.bias_divide_scale = (self.bias * 256) / (self.scale / 2 ** self.shift)
        self.bias_divide_scale = self.bias_divide_scale.round().int()
        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            return (F.linear(x, self.qx_minus_zeropoint, self.bias_divide_scale) * self.scale) >> self.shift
        else:
            return F.linear(x, self.weight, self.bias)

class QuantAvePool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride, padding=0):
        super(QuantAvePool2d, self).__init__(kernel_size, stride, padding)
        self.quant_flag = False

    def pool_quant(self, quantize_bit=8):
        self.quant_flag = True

    def load_quant(self):
        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            return F.avg_pool2d(x.float(), self.kernel_size, self.stride, self.padding).round().int()
        else:
            return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QuantConv2d, self).__init__(in_channels, out_channels,
                                          kernel_size, stride, padding, dilation, groups, bias)
        self.quant_flag = False
        self.scale = None
        self.shift = None
        self.zero_point = None
        self.qx_minus_zeropoint = None
        self.bias_divide_scale = None

    def conv_quant(self, quantize_bit=8):
        self.weight.data, self.scale, self.zero_point = quantize_tensor(self.weight.data, num_bits=quantize_bit)
        self.quant_flag = True

    def load_quant(self, scale, shift, zero_point):
        # true_scale = scale >> shift
        self.scale = scale
        self.shift = shift
        self.zero_point = zero_point
        self.qx_minus_zeropoint = self.weight - self.zero_point
        self.qx_minus_zeropoint = self.qx_minus_zeropoint.round().int()
        self.bias_divide_scale = (self.bias * 256) / (self.scale / 2 ** self.shift)
        self.bias_divide_scale = self.bias_divide_scale.round().int()
        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            return (F.conv2d(x, self.qx_minus_zeropoint, self.bias_divide_scale, self.stride,
                            self.padding, self.dilation, self.groups) * self.scale) >> self.shift
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.c1 = QuantConv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Relu = nn.ReLU()
        self.s2 = QuantAvePool2d(kernel_size=2, stride=2)
        self.c3 = QuantConv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = QuantAvePool2d(kernel_size=2, stride=2)
        self.c5 = QuantConv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = QuantLinear(120, 84)
        self.output = QuantLinear(84, 10)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.Relu(self.c1(x))
        x = self.s2(x)
        x = self.Relu(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x

    def linear_quant(self, quantize_bit=8):
        self.c1.conv_quant(quantize_bit)
        self.s2.pool_quant(quantize_bit)
        self.c3.conv_quant(quantize_bit)
        self.s4.pool_quant(quantize_bit)
        self.c5.conv_quant(quantize_bit)
        self.f6.linear_quant(quantize_bit)
        self.output.linear_quant(quantize_bit)

    def load_quant(self, c1_sc: int, c1_sh: int, c1_zp: int, c3_sc: int, c3_sh: int, c3_zp: int,
                   c5_sc: int, c5_sh: int, c5_zp: int, f6_sc: int, f6_sh: int, f6_zp: int,
                   out_sc: int, out_sh: int, out_zp: int):
        self.c1.load_quant(c1_sc, c1_sh, c1_zp)
        self.s2.load_quant()
        self.c3.load_quant(c3_sc, c3_sh, c3_zp)
        self.s4.load_quant()
        self.c5.load_quant(c5_sc, c5_sh, c5_zp)
        self.f6.load_quant(f6_sc, f6_sh, f6_zp)
        self.output.load_quant(out_sc, out_sh, out_zp)

if __name__ == "__main__":
    x = torch.rand([1, 1, 28, 28]).round().int()
    model = LeNet()
    model.linear_quant()
    model.eval()
    with torch.no_grad():
        model.load_quant(26, 2, 90, 26, 2, 90, 26, 2, 90, 26, 2, 90, 26, 2, 90)
        y = model(x)