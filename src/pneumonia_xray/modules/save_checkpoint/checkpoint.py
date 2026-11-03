import torch
from torch import nn

from pathlib import Path

def save_checkpoint(model : nn.Module, path_to_save : str, **extra):
    
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state" : model.state_dict(), **extra}, path_to_save)

def load_checkpoint(model : nn.Module, path_to_load : str):

    device = next(model.parameters()).device
    checkpoint = torch.load(path_to_load, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint