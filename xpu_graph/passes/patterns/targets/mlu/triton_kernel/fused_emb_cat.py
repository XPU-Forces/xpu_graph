import torch                                                                                                                                                                                                                                                                    
import torch_mlu                                                                                                                                                                                                                                                                
import triton                                                                                                                                                                                                                                                                   
import triton.language as tl                                                                                                                                                                                                                                                    
from . import libentry                                                                                                                                                                                                                                                          
from .get_mlu_devinfo import get_device_properties                                                                                                                                                                                                                              
from typing import List, Tuple                                                                                                                                                                                                                                                  

@torch.library.custom_op("torch_mlu_triton::fused_emb_cat", mutates_args=())
def fused_emb_cat(
    input_tensor: torch.Tensor,
    input_offsets: torch.Tensor,
    input_dims: torch.Tensor,
    output_ids: torch.Tensor,
    output_offsets: torch.Tensor,
    output_dims: torch.Tensor,
    output_dims_list: List[int]
) -> List[torch.Tensor]:

    '''
    print(input_tensor.shape)
    print(input_offsets)
    print(input_dims)
    print(output_ids)
    print(output_offsets)
    print(output_dims)
    '''

    output = torch.ops.torch_mlu.emb_concat(
        input_tensor,
        input_offsets,
        input_dims,
        output_ids,
        output_offsets,
        output_dims,
        output_dims_list,
    )
    return output


@fused_emb_cat.register_fake
def fused_emb_cat_fake(
    input_tensor: torch.Tensor,
    input_offsets: torch.Tensor,
    input_dims: torch.Tensor,
    output_ids: torch.Tensor,
    output_offsets: torch.Tensor,
    output_dims: torch.Tensor,
    output_dims_list: List[int]
) -> List[torch.Tensor]:
    batch = input_tensor.shape[0]
    outputs = []
    for dim in output_dims_list:
        outputs.append(torch.empty(batch, dim, device=input_tensor.device, dtype=input_tensor.dtype))
    return outputs
