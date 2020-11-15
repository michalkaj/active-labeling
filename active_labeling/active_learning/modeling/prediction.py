from typing import Dict

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def predict(model: Module, dataset: Dataset, dataloader_kwargs: Dict) -> Tensor:
    if not len(dataset):
        raise ValueError("Empty dataset")

    logits = None
    device = next(model.parameters()).device
    start = 0
    for batch in tqdm(
        DataLoader(dataset, **dataloader_kwargs),
        desc='Computing logits...'
    ):
        images = batch['image'].to(device)
        end = min(start + len(images), len(dataset))
        with torch.no_grad():
            output = model(images)

        if logits is None:
            logits = torch.empty(len(dataset), *output.shape[1:], dtype=output.dtype)
        logits[start: end] = output.detach().cpu()
        start = end

    return logits
