from torch.utils.data import DataLoader, Dataset
from torch import float32
from torchvision import datasets
from torchvision.transforms import v2

def fetch_data(path_test_data : str, path_train_data) -> tuple[Dataset, Dataset] :
    """
        Récupère les données de tests et d'entrainement dans leur chemin respéctif

        args:
            path_test_data   (str): chemin pointant vers le dossier contenant les images de test
            path_train_data  (str): chemin pointant vers le dossier contenant les images d'entrainement

        returns:
            tuple[Dataset, Dataset] : tuple contenant le dataset d'entrainement et le dataset de test, dans cet ordre
    """
    transform = v2.Compose([
        v2.Resize((224,224)),
        v2.ToImage(),
        v2.ToDtype(dtype=float32, scale=True)
    ])

    test_dataset  = datasets.ImageFolder(
        path_test_data,
        transform=transform
    )

    train_dataset = datasets.ImageFolder(
        path_train_data,
        transform=transform    
    )

    return train_dataset,test_dataset

