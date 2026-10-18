import torch 
import triton
import triton.language as tl


device= torch.device(f'cuda:{torch.cuda.current_device()}')
