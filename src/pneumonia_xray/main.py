import torch
import torch.nn as nn
import torch.optim as optim
from .modules.data_loading import load_data,get_device
from .modules.neural_model.neural_network import NeuralNetwork
from .modules.neural_training.neural_train import train
from .modules.neural_testing.neural_test import test

from .config import EPOCHS, LEARNING_RATE, PATH_TO_DATA

if __name__ == "__main__":


    train_dataloader,test_dataloader = load_data(path_to_data=PATH_TO_DATA)
    device = get_device()

    # Poids de classes calculés sur l'entraînement uniquement
    subset = train_dataloader.dataset
    counts = torch.bincount(torch.tensor(subset.dataset.targets)[subset.indices])
    class_weights = (counts.sum() / (2 * counts)).float().to(device)
    
    print(f"Images par classe : {counts.tolist()} / poids : {class_weights.tolist()}")

    model = NeuralNetwork().to(device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for i in range(EPOCHS):
        print(f"Lancement de l'époque {i}\n")
        
        total_loss = train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        print(f"Perte totale : {total_loss}")
        
        test(dataloader=test_dataloader, model=model, loss_fn=loss_fn)
        print(f"Fin de l'époque {i}\n\n")

    print(f"Fin {i}\n\n")




