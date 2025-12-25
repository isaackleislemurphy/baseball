"""Utils relating to torch data types and loaders"""

from typing import Iterable, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def safe_convert_to_tensor(x: Union[torch.Tensor, np.ndarray, float, int]) -> torch.Tensor:
    """
    Convert a scalar, NumPy array, or Tensor into a fresh PyTorch Tensor.

    This function ensures that the resulting tensor is a deep copy of the input,
    breaking any reference links to the original object to prevent side effects.

    Parameters
    ----------
    x : Union[torch.Tensor, np.ndarray, float, int]
        The input data to convert. Can be a standard Python scalar, a NumPy scalar/array,
        or an existing PyTorch Tensor.

    Returns
    -------
    torch.Tensor
        A new PyTorch Tensor containing the data from `x`.

    Raises
    ------
    TypeError
        If the input type is not supported.
    """
    # 1.) Is it a regular old number...if yes, make it a tensor
    _np_dtypes = np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64
    if isinstance(x, (float, int) + _np_dtypes):
        return torch.tensor(x)
    # 2.) Is it a numpy array...if yes, copy and cast to tensor
    elif isinstance(x, np.ndarray):
        # will inherit the numpy dtype here
        return torch.from_numpy(x.copy())
    # 3.) Is it already a tensor? If yes, clone and send it on its way
    elif isinstance(x, torch.Tensor):
        return x.clone()
    else:
        raise TypeError("`safe_convert_to_tensor` must receive a `torch.Tensor`, `np.ndarray`, `float`, or `int`")


def make_data_loader(
    inputs: Union[np.ndarray, torch.Tensor, Iterable], batch_size: int, device: str = "cpu"
) -> DataLoader:
    """
    Create a shuffled PyTorch DataLoader from one or more input arrays.

    This utility handles the conversion of inputs to tensors, moves them to the
    specified computing `device`, and wraps them in a TensorDataset and DataLoader.

    Parameters
    ----------
    inputs : Union[np.ndarray, torch.Tensor, Iterable]
        The data to load.
        - If a single array/tensor is passed, the DataLoader yields single batches.
        - If an Iterable (list/tuple) of arrays is passed (e.g., `[X, y]`),
          the DataLoader yields tuples of batches.
    batch_size : int
        The number of samples per batch.
    device : str, default "cpu"
        The target device to move tensors to (e.g., "cpu", "cuda", "mps").

    Returns
    -------
    DataLoader
        A PyTorch DataLoader configured with `shuffle=True`.
    """
    # if it's just an array or tensor, just safely convert it and do a single list
    if isinstance(inputs, (np.ndarray, torch.Tensor)):
        inputs = [safe_convert_to_tensor(inputs)]

    # otherwise safely convert every element
    elif isinstance(inputs, Iterable):
        inputs = [safe_convert_to_tensor(item) for item in inputs]
    else:
        raise TypeError("`inputs` should be a `np.ndarray` or `torch.Tensor`")

    # make the data loader and return it
    loader = DataLoader(
        dataset=TensorDataset(*[item.to(device) for item in inputs]),
        batch_size=batch_size,
        shuffle=True,
    )

    return loader
