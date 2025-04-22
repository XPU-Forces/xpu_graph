import torch
import torch_npu

from torch.library import Library, impl
npu_def = Library("torch_npu_triton", "DEF")

npu_lib = Library("torch_npu_triton", "IMPL", "PrivateUse1")
npu_meta =Library("torch_npu_triton", "IMPL", "Meta")