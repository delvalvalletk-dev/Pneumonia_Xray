from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer) -> None:
    """
        Entraîne le modèle pendant une époque, soit un passage complet sur le jeu d'entraînement

        args:
            dataloader (DataLoader): paniers (batches) d'images d'entraînement et leurs étiquettes
            model      (nn.Module) : modèle à entraîner 
            loss_fn    (nn.Module) : fonction de perte (CrossEntropyLoss pondérée pour l'instant)
            optimizer  (Optimizer) : optimiseur qui met à jour les poids du modèle (Adam pour l'instant)

        returns:
            float: valeur de la perte totale lors de l'époque
    """
    size = len(dataloader.dataset)
    model.train()
    device = next(model.parameters()).device

    total_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()

        total_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss_value = loss.item()
            current = batch * len(X) + len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

    return total_loss / len(dataloader)