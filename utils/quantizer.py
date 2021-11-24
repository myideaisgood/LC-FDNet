import torch
import torch.nn as nn

def custom_round(input):
    
    floor = torch.floor(input)

    down_res = input - floor
    up_res = 1- down_res

    sign = torch.where(down_res>=0.5, 1, -1)

    down_idx = torch.where(sign==-1, 1, 0)
    up_idx = torch.where(sign==1, 1, 0)

    output = input + down_idx * sign * down_res + up_idx * sign * up_res

    return output 

class hard_quantizer(nn.Module):
    def __init__(self):
        super(hard_quantizer, self).__init__()
    
    def forward(self, input):
        return custom_round(input)

class soft_quantizer(nn.Module):
    def __init__(self):
        super(soft_quantizer, self).__init__()
    
    def forward(self, input):
        return input

class diff_quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        with torch.enable_grad():
            SQ = soft_quantizer()
            HQ = hard_quantizer()

            output = SQ(input)
            ctx.save_for_backward(input, output)

        return HQ(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        output.backward(grad_output, retain_graph=True)
        return input.grad
