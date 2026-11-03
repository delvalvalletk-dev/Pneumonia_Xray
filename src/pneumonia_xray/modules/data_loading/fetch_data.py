from torch.utils.data import Dataset,Subset
from torch import float32
from torchvision import datasets
from torchvision.transforms import v2

from sklearn.model_selection import StratifiedGroupKFold 

from pathlib import Path
import re
import numpy as np

from pneumonia_xray.config import IMAGE_SIZE,SEED

def fetch_data(path_to_data : str) -> tuple[Dataset, Dataset] :
    """
        Récupère les données de tests et d'entrainement dans leur chemin respectif

        args:
            path_to_data   (str): chemin pointant vers le dossier contenant les images/données

        returns:
            tuple[Dataset, Dataset] : tuple contenant le dataset d'entrainement et le dataset de test, dans cet ordre
    """
    train_transform = v2.Compose([
        v2.Resize(IMAGE_SIZE),
        v2.RandomRotation(10),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.ToImage(),
        v2.ToDtype(float32, scale=True),
    ])

    test_transform = v2.Compose([   # pas d'augmentation sur le test
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(float32, scale=True),
    ])

    test_dataset  = datasets.ImageFolder(
        path_to_data,
        transform=test_transform
    )

    train_dataset = datasets.ImageFolder(
        path_to_data,
        transform=train_transform    
    )

    indexes_train, indexes_test = sort_patients_dataset(data_train=train_dataset)

    train_dataset = Subset(train_dataset, indexes_train)
    test_dataset = Subset(test_dataset, indexes_test)


    return train_dataset, test_dataset

def fetch_patient_id(path : str, label : int):
    name_patient = Path(path).name
    match = re.match(r"(person\d+)", name_patient)

    patient_id = match.group(1) if match else name_patient

    return f"{label}_{patient_id}"

def fetch_all_patient_ids(list_images : list[tuple[str,int]]) -> list[str]:
    
    return [fetch_patient_id(patient[0],patient[1]) for patient in list_images] 


def sort_patients_dataset(data_train : Dataset):
    list_train_image_pathes = [(samp[0],samp[1]) for samp in data_train.samples]

    indexes_patients_train = fetch_all_patient_ids(list_train_image_pathes)
    targets = data_train.targets
    indices_train = np.arange(len(data_train))

    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    index_train,index_test = next(splitter.split(X=indices_train, y=targets, groups=indexes_patients_train))

    return index_train,index_test


def display_dataset(data : Dataset):

    targets = data.targets
    sample = data.samples
    indexes = set(targets)

    print(f"Targets : {data.targets} \nLength : {len(data.targets)} \nLength By Class : [ {f"{'/':^3}".join([f"{index} : {targets.count(index)}" for index in indexes])} ]\n")
    print(f"First 10 Samples : \n\n{'\n'.join([f"{samp[0]} : {"Normal" if samp[1]==0 else "PNEUMONIA"}" for samp in sample[0:10]])}")

    
