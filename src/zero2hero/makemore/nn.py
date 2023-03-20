
# Manual definition of NN modules similar to torch.nn
import torch

class Linaer:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out) / fan_in**0.5) # kaiming init
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        z = x @ self.weight
        if self.bias:
            z += self.bias
        return z

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias else [])

class Tanh:
    def __call__(self, x):
        return torch.tanh(x)
    def parameters(self):
        return []

class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # trainable parameters
        self.bn_gain = torch.ones(dim)
        self.bn_bias = torch.zeros(dim)
        # buffers
        self.run_mean = torch.zeros(dim)
        self.run_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            z_mean = x.mean(axis=0, keepdim=True)
            z_var = x.var(axis=0, keepdim=True)
        else:
            z_mean = self.run_mean
            z_var = self.run_svar

        z = (x - z_mean) / torch.sqrt(z_var+ self.eps)
        out = self.bn_gain * z + self.bn_bias

        if self.training:
            with torch.no_grad():
                self.run_mean = (1 - self.momentum) * self.run_mean + self.momentum * z_mean
                self.run_var  = (1 - self.momentum) * self.run_var  + self.momentum * z_var

        return out

    def parameters(self):
        return [self.bn_gain, self.bn_bias,]


