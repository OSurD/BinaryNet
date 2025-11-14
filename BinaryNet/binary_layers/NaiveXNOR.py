import torch
import torch.nn as nn

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        mask = (x.abs() <= 1).to(g.dtype)
        return g * mask

class NaiveXNORLinear(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(NaiveXNORLinear, self).__init__()
        self.fc = nn.Linear(ch_in, ch_out, bias=True)

    def forward(self, input_x):
        quantized_weight = SignSTE.apply(self.fc.weight)
        quintized_input = SignSTE.apply(input_x)
        out = nn.functional.linear(quintized_input, quantized_weight)
        alpha = self.fc.weight.abs().mean()
        betta = input_x.abs().mean()
        return out*alpha*betta + self.fc.bias
    
class NaiveXNORConv2d(nn.Module):
    def __init__(self, ch_in,ch_out, kernel=3, padding=0, stride=1):
        super(NaiveXNORConv2d, self).__init__()
        self.padding = padding
        self.stride = stride
        self.conv = nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=False)
        k = torch.ones((1, 1, kernel, kernel))/(kernel**2) #< Now consider only constants
        self.k = nn.parameter.Buffer(k, persistent=False)

    def forward(self, input_x):
        # A, k - BinActiv
        A = input_x.abs().mean(axis=1, keepdim=True)
        K = nn.functional.conv2d(A, self.k, None, self.stride, self.padding)
        alpha = self.conv.weight.abs().mean()

        quantized_weight = SignSTE.apply(self.conv.weight)
        quintized_input = SignSTE.apply(input_x)
        out = nn.functional.conv2d(quintized_input, quantized_weight, None, self.stride, self.padding)
        return out*K*alpha