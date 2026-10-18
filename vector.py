import torch 
import triton
import triton.language as tl


device= torch.device(f'cuda:{torch.cuda.current_device()}')
import time 

def add_vector_pseudocode_wrapper(a,b):
    n_elements=len(a)
    grid=(n_elements,)
    output=torch.empty_like(a)

    vector_add_pseudocode(a,b,output,grid)