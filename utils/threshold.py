import torch
import torch.nn as nn

class hard_threshold(nn.Module):
    def __init__(self):
        super(hard_threshold, self).__init__()
    
    def forward(self, input, var_th, type='down'):
        if type=='down':
            return torch.where(input < var_th, 1, 0)
        else:
            return torch.where(input >= var_th, 1, 0)

class soft_threshold(nn.Module):
    def __init__(self):
        super(soft_threshold, self).__init__()
    
    def forward(self, input, var_th, type='down'):
        if type=='down':
            return torch.sigmoid(-(input - var_th))
        else:
            return torch.sigmoid(input - var_th)

class diff_threshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, var_th, type='down'):
        with torch.enable_grad():
            ST = soft_threshold()
            HT = hard_threshold()

            output = ST(input, var_th, type)
            ctx.save_for_backward(input, output)

        return HT(input, var_th, type)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        output.backward(grad_output, retain_graph=True)
        return input.grad
