import torch
from typing import Iterable
import warnings

def iterable_to_cuda(input:Iterable[torch.tensor], cuda_available:bool=False) -> Iterable[torch.tensor]: 
    """This function returns an iterable containing all the tensors in the input Iterable in which the various tensors
    are sent to CUDA (if applicable). 

    Args:
        input (Iterable[torch.tensor]): Iterable of tensors to be sent to CUDA.

    Returns:
        Iterable[torch.tensor]: Iterable of tensors sent to CUDA. 
    """
    warnings.warn("This function is deprecated. Handle device logic in the main script instead.")
    
    return [tensor.to("cuda") if cuda_available else tensor for tensor in input]

