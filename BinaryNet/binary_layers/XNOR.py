import torch.nn as nn
import torch


class BinWeightFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor):
        if weight.dim() == 4:
            # [out, in, kh, kw]
            negMean = weight.mean(dim=1, keepdim=True).mul(-1).expand_as(weight)
        elif weight.dim() == 2:
            # [out, in]
            negMean = weight.mean(dim=1, keepdim=True).mul(-1).expand_as(weight)

        w_centered = weight + negMean
        w_clamped  = w_centered.clamp_(-1.0, 1.0)

        if weight.dim() == 4:
            n = weight[0].numel()
            m = (w_clamped.norm(p=1, dim=3, keepdim=True)
                           .sum(2, keepdim=True)
                           .sum(1, keepdim=True)
                           .div(n))
        else:
            n = weight.size(1)
            m = w_clamped.norm(p=1, dim=1, keepdim=True).div(n)

        w_bin = w_clamped.sign()

        ctx.save_for_backward(w_clamped)
        ctx.n = n
        return w_bin, m

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, scale_grad:torch.Tensor):
        (w_clamped,) = ctx.saved_tensors
        n = ctx.n
        s = w_clamped.size()

        g = grad_out
        if w_clamped.dim() == 4:
            m = (w_clamped.norm(p=1, dim=3, keepdim=True)
                              .sum(2, keepdim=True)
                              .sum(1, keepdim=True)
                              .div(n)).expand(s)
        else:
            m = w_clamped.norm(p=1, dim=1, keepdim=True).div(n).expand(s)

        m = m.clone()
        m[w_clamped.lt(-1.0)] = 0
        m[w_clamped.gt( 1.0)] = 0

        grad = m * g

        m_add = w_clamped.sign() * g
        if w_clamped.dim() == 4:
            m_add = (m_add.sum(3, keepdim=True)
                           .sum(2, keepdim=True)
                           .sum(1, keepdim=True)
                           .div(n)
                           .expand(s))
        else:
            m_add = m_add.sum(1, keepdim=True).div(n).expand(s)

        m_add = m_add * w_clamped.sign()
        grad = (grad + m_add) * (1.0 - 1.0 / s[1]) * n
        #grad = grad * 1e9
        return grad
    
class BinActive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d_Impl(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x_bin = BinActive.apply(x)
        w_bin, scale = BinWeightFn.apply(self.weight)
        y = nn.functional.conv2d(x_bin, w_bin, self.bias,
                                 self.stride, self.padding,
                                 self.dilation, self.groups)
        return y * scale.view(1, -1, 1, 1)
    
class BinLinear_Impl(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        x_bin = BinActive.apply(x)
        w_bin, scale = BinWeightFn.apply(self.weight)
        y = nn.functional.linear(x_bin, w_bin, self.bias)
        return y * scale.view(1, -1)

class BinConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.1, affine=True)
        self.conv = BinConv2d_Impl(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=groups)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class BinLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features, eps=1e-4, momentum=0.1, affine=True)
        self.linear = BinLinear_Impl(in_features, out_features, bias=bias)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.bn(x)
        x = self.linear(x)
        x = self.relu(x)
        return x