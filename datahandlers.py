"""Library of classes and functions to manage data sources and sinks."""
import os
import shutil
from typing import List

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import errordefs as err
import constants as con

IMAGE_SIZE = 256  # Size of the input image (images are resized to this value)
NUM_WORKERS = 4   # Number of workers for data loading

class PartitionsData:
    """Class PartitionsData manages the directory structure after partitioning data into
    train, validation, and test sets.
    """
    def __init__(self, root_dir_path: str):
        self.__root_dir = root_dir_path


    def create(self, labels: List[str]):
        """Create the folder structure for dataset partitions (train, validation, test).
        Each partition folder has N subfolders, one per class label. These subfolders contain
        the images associated with the class label.
        """
        os.makedirs(self.__root_dir, exist_ok=True)
        for folder_name in con.partition_folders:
            folder_path = os.path.join(self.__root_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            for subfolder_name in labels:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                os.makedirs(subfolder_path, exist_ok=True)

    @property
    def labels(self):
        """Extract the list of class labels."""
        items = os.listdir(self.__root_dir)
        item_paths = [os.path.join(self.__root_dir, item) for item in items]
        folder_paths = [item_path for item_path in item_paths if os.path.isdir(item_path)]
        folder_names = [os.path.basename(folder_path) for folder_path in folder_paths]
        if len(folder_names) == 0:
            raise err.EmptyPartitionsData("The partitions folder structure has not been created.")
        return folder_names

    def add_images(self, subset: str, label: str, image_paths: List[str]):
        """Add images (defined as a list of paths) to the subset folder and the label subfolder."""
        if subset not in con.partition_folders:
            raise ValueError(f"The selected subset must be one of {con.partition_folders}")
        if label not in self.labels:
            raise ValueError(f"The selected label must be one of {self.labels}")

        dst_path = os.path.join(self.__root_dir, subset, label)
        for img_path in image_paths:
            try:
                shutil.copy(src=img_path, dst=dst_path)
            except FileNotFoundError:
                print(f"Missing file: {img_path}")
            except Exception as e:
                print(f"An exception occurred during file copy: {e}")

    @property
    def trn_path(self) -> str:
        """Get the path to the train data folder."""
        return os.path.join(self.__root_dir, con.trn_folder)

    @property
    def val_path(self) -> str:
        """Get the path to the validation data folder."""
        return os.path.join(self.__root_dir, con.val_folder)

    @property
    def tst_path(self) -> str:
        """Get the path to the test data folder."""
        return os.path.join(self.__root_dir, con.tst_folder)

    def get_label_folder_path(self, label: str, subset: str) -> str:
        """Get the path to a subfolder for a class label. The subfolder is inside a
        subset folder.
        """
        if label not in self.labels:
            raise ValueError(f"{label} not in list of labels: {self.labels}")
        if subset not in con.partition_folders:
            raise ValueError(f"{subset} not in list of partition names: {con.partition_folders}")

        return os.path.join(self.__root_dir, subset, label)


class ImageFolderDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.__dir = dir_path
        self.__transform = transform

        items = os.listdir(self.__dir)
        images = [item for item in items if item.split(".")[-1] in con.image_extensions]
        self.__image_paths = [os.path.join(self.__dir, image) for image in images]

        self.__num_images = len(images)

    def __len__(self):
        return self.__num_images

    def __getitem__(self, idx):
        img_path = self.__image_paths[idx]
        img_fname = os.path.basename(img_path)
        img_data = Image.open(img_path)
        if self.__transform:
            img_data = self.__transform(img_data)

        return {
            "image": img_data,
            "filename": img_fname
        }

# Training transforms
def get_trn_transform(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform

# Validation transforms
def get_val_transform(image_size):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return valid_transform


def get_datasets(trn_dir_path, val_dir_path):
    """ Create handlers for the train and validation datasets. Return the TRN and VAL dataset handlers
    and the associated classes.
    """
    dataset_train = datasets.ImageFolder(trn_dir_path, transform=(get_trn_transform(IMAGE_SIZE)))
    dataset_valid = datasets.ImageFolder(val_dir_path, transform=(get_val_transform(IMAGE_SIZE)))
    return dataset_train, dataset_valid, dataset_train.classes


def get_data_loaders(dataset_train, dataset_valid, batch_size):
    """ Return the loader functions for TRN and VAL datasets. The function uses the dataset handlers
    obtained using the get_datasets() function.
    """
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, valid_loader

def get_test_dataset(tst_dir_path):
    """ Create a handler for the test dataset. Return the handler and the associated classes."""
    dataset_test = datasets.ImageFolder(tst_dir_path, transform=(get_val_transform(IMAGE_SIZE)))
    return dataset_test, dataset_test.classes

def get_test_loader(dataset_test_handle, batch_size):
    """Return the loader function for the TST dataset."""
    test_loader = DataLoader(dataset_test_handle, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return test_loader
