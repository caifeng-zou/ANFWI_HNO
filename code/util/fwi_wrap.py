import torch.nn as nn


class auto_diff_FWI(nn.Module):
    def __init__(self, vs_init, vp_init):
        super().__init__()
        self.vs = nn.Parameter(vs_init, requires_grad=True)
        self.vp = nn.Parameter(vp_init, requires_grad=True)
    
    def forward(self, x, model):
        x[..., 0] =  self.vp
        x[..., 1] =  self.vs
        out = model(x)
        return out