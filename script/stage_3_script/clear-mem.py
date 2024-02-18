import torch, gc

torch._C._cuda_clearCublasWorkspaces()
torch._dynamo.reset()
gc.collect()
torch.cuda.empty_cache()