from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from .fetch_data import fetch_data

BATCH_SIZE = 64

def load_data(path_test_data : str, path_train_data) -> str :
    """
        Récupère les données de tests et d'entrainement dans leur chemin respéctif

        args:
            path_test_data   (str): chemin pointant vers le dossier contenant les images de test
            path_train_data  (str): chemin pointant vers le dossier contenant les images d'entrainement

        returns:
            ???
    """

    # recupere les datasets de l'entrainement et du test
    train_dataset,test_dataset = fetch_data(path_train_data=path_train_data, path_test_data=path_test_data)

    train_dataloader,test_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for x,y in train_dataloader:
        print(f"shape of X : {x.shape}\n")
        print(f"shape of Y : {y.shape}\n")
        break

    return train_dataloader,test_dataloader

