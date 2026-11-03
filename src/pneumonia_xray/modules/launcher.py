import torch
import torch.nn as nn
import torch.optim as optim

from .data_loading import load_data,get_device
from .neural_model.neural_network import NeuralNetwork
from .neural_training.neural_train import train
from .neural_testing.neural_test import test

from .save_checkpoint.checkpoint import save_checkpoint,load_checkpoint

from pathlib import Path 


def launch_learning(epochs : int, learning_rate : float, path_to_data : str, path_to_save_last : str):

    train_dataloader,test_dataloader = load_data(path_to_data=path_to_data)
    device = get_device()

    # Poids de classes calculés sur l'entraînement uniquement
    subset = train_dataloader.dataset
    counts = torch.bincount(torch.tensor(subset.dataset.targets)[subset.indices])
    class_weights = (counts.sum() / (2 * counts)).float().to(device)
    
    print(f"Images par classe : {counts.tolist()} / poids : {class_weights.tolist()}")

    model = NeuralNetwork().to(device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    path_to_save_last = Path(path_to_save_last)

    if path_to_save_last.exists():
        load_checkpoint(model=model, path_to_load=path_to_save_last)

    for epoch in range(epochs):
        print(f"Lancement de l'époque {epoch}\n\n")
        
        total_loss = train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        print(f"Perte totale : {total_loss}\n")
        
        test(dataloader=test_dataloader, model=model, loss_fn=loss_fn)
        print(f"Fin de l'époque {epoch}\n\n")

        print(f"Sauvegarde de l'époque {epoch}\n\n")
        save_checkpoint(model=model, path_to_save=path_to_save_last, epoch=epoch)

    print(f"Fin\n\n")