import torch
from torch import nn
from torch.utils.data import DataLoader


def test(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """
        Évalue le modèle sur le jeu de test (sans modifier ses poids)

        args:
            dataloader (DataLoader): paniers d'images de test et leurs étiquettes
            model      (nn.Module) : modèle à évaluer
            loss_fn    (nn.Module) : fonction de perte

        returns:
            tuple: (perte moyenne, précision entre 0 et 1, étiquettes réelles, étiquettes prédites)
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    device = next(model.parameters()).device

    model.eval()

    test_loss, correct = 0, 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predicted = pred.argmax(1)
            correct += (predicted == y).sum().item()
            all_true.append(y.cpu())
            all_pred.append(predicted.cpu())

    test_loss /= num_batches
    accuracy = correct / size
    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)

    print(f"Test Error: \n Accuracy: {100 * accuracy:>0.1f}%, Avg loss: {test_loss:>8f}")
    for label, name in enumerate(["NORMAL", "PNEUMONIA"]):
        mask = y_true == label
        recall = (y_pred[mask] == label).float().mean().item()
        print(f" Rappel {name}: {100 * recall:>0.1f}%  ({mask.sum().item()} images)")
    print()

    return test_loss, accuracy, y_true, y_pred