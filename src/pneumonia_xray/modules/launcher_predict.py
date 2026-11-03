import torch
from torch import nn

from torchvision.transforms import v2

from pathlib import Path

from pneumonia_xray.modules.neural_model.neural_network import NeuralNetwork
from .data_loading import get_device
from .image_prediction.prediction import predict
from .save_checkpoint.checkpoint import load_checkpoint

def launch_predict(image_size : list, images_path : str, classnames : dict, path_to_load : str):
    
    transform = v2.Compose([
        v2.Resize(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    device = get_device()
    model = NeuralNetwork().to(device=device)
    load_checkpoint(model=model, path_to_load=path_to_load)

    images_path = Path(images_path)
    path_to_load = Path(path_to_load)

    predict(model=model, images_path=images_path, transform=transform, class_names=classnames, device=device)