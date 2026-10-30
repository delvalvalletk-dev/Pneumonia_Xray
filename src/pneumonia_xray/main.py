
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from .modules.data_loading import load_data,get_device
from .modules.neural_model.neural_network import NeuralNetwork
from .modules.neural_training.neural_train import train
from .modules.neural_testing.neural_test import test



BATCH_SIZE = 64
EPOCHS = 5

if __name__ == "__main__":

    train_dataloader,test_dataloader = load_data(path_train_data="C:/Users/Killian/Documents/github/Programmation/Python/Pneumonia_Xray/data/chest_xray/train", path_test_data="C:/Users/Killian/Documents/github/Programmation/Python/Pneumonia_Xray/data/chest_xray/test")
    device = get_device()

    model = NeuralNetwork().to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(EPOCHS):
        print(f"Lancement de l'époque {i}\n")
        train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        test(dataloader=test_dataloader, model=model, loss_fn=loss_fn)
        print(f"Fin de l'époque {i}\n\n")

    print(f"Fin {i}\n\n")

