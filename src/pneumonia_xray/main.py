
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from .modules.data_loading import fetch_data


if __name__ == "__main__":
    batch_size = 64

    fetch_data(path_train_data="C:/Users/Killian/Documents/github/Programmation/Python/Pneumonia_Xray/data/chest_xray/train", path_test_data="C:/Users/Killian/Documents/github/Programmation/Python/Pneumonia_Xray/data/chest_xray/test")
