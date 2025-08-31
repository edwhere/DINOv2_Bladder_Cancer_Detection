"""Generate TRN, VAL and TST data directories using the downloaded bladder cancer
cystoscopy raw data.

The data for the Bladder Cancer cystoscopy tissue classification project
can be downloaded from Zenodo: https://zenodo.org/records/7741476

After downloading, the data goes to a directory identified by a path. This
directory contains an 'annotations.csv' file and 4 folders for each of
the 4 tissue types:

HGC: Folder storing High Grade Carcinoma images
LGC: Folder storing Low Grade Carcinoma images
NTL: Folder storing No Tumor Lession images
NST: Folder storing Non-Suspicious Tissue images

This script takes the raw data (images) from the original directory and
creates a new directory containing 3 folders (partitions) for train, validation and
test datasets. The typical folder names are TRN, VAL, TST.

Each partition folder contains subfolders with tissue types. The subfolders
contain the corresponding images.

"""

import os
import pandas as pd
import argparse
from typing import List

import constants as con
import datahandlers as dh


class IncorrectFolderStructure(Exception):
    """Raise an error if the provided directory does not have a valid file/folder structure."""
    pass

class CystoscopyData:
    """ Manage the folder that contains the original bladder cancer cystoscopy dataset.
    The folder is identified by a root directory.
    The folder contains an annotations.csv file and 4 subfolder for tissue types: HGC, LGC, NST and NTL.
    """
    def __init__(self, root_dir_path: str):
        """Initialize and instance of this class with the path where the raw data is stored."""
        self.__root_dir = root_dir_path
        self.__tissue_types = ["HGC", "LGC", "NST", "NTL"]
        self.__annot_file = "annotations.csv"

    def is_valid_structure(self):
        """Check if the folder defined by root_dir_path actually contains subfolders for tissue types
        and an annotation.csv file.
        """
        items = os.listdir(self.__root_dir)
        item_paths = [os.path.join(self.__root_dir, item) for item in items]
        folder_names = [os.path.basename(it_path) for it_path in item_paths if os.path.isdir(it_path)]

        result1 = set(folder_names) == set(self.__tissue_types)
        result2 = os.path.isfile(os.path.join(self.__root_dir, self.__annot_file))

        return result1 and result2

    @property
    def tissue_types(self) -> List[str]:
        return self.__tissue_types

    @property
    def labels(self) -> List[str]:
        return self.__tissue_types

    def get_folder_paths(self) -> List[str]:
        """Return a list of folder paths for each folder in the raw data directory,"""
        items = os.listdir(self.__root_dir)
        folders = [it for it in items if os.path.isdir(os.path.join(self.__root_dir, it))]
        folder_paths = [os.path.join(self.__root_dir, folder) for folder in folders]
        return folder_paths

    def get_image_paths(self, folder_name: str) -> List[str]:
        """Return a list of image paths for each image stored in a folder. The folder is
        identified by its name (not path).
        """
        if folder_name not in self.__tissue_types:
            raise ValueError(f"{folder_name} is not a valid folder name.")

        folder_path = os.path.join(self.__root_dir, folder_name)
        items = os.listdir(folder_path)
        images = [it for it in items if it.split(".")[-1] in con.image_extensions]
        image_paths = [os.path.join(folder_path, img) for img in images]
        return image_paths

    def get_annotation_file_path(self) -> str:
        """Return the path for the annotations.csv file that exists in the raw data directory."""
        annot_file_path = os.path.join(self.__root_dir, self.__annot_file)
        return annot_file_path


def parse_arguments():
    """Parse command-line arguments."""
    describe=("Convert a raw dataset of bladder cancer endoscopic images into "
              "structured partitions with train, validation, and test data.")
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group("required arguments")

    required.add_argument("-inp", "--input_dir", type=str, required=True,
                          help="Path to input directory (raw data).")

    required.add_argument("-out", "--output_dir", type=str, required=True,
                          help="Path to output directory (folder containing partitions).")

    parser.add_argument("--include_NBI", action="store_true",
                        help="Include NBI images in addition to WLI images.")

    args = parser.parse_args()

    return args


def main():
    """Run the main sequence of procedures."""
    # Parse the command-line arguments.
    args = parse_arguments()

    # Create data handlers to manage input and output directories
    raw_data = CystoscopyData(args.input_dir)
    if not raw_data.is_valid_structure():
        raise IncorrectFolderStructure("Data folder does not have the required structure.")

    partitions_data = dh.PartitionsData(args.output_dir)

    # Select the data for ML operations. It can be WLI data or combined WLI and NBI data.
    annot_df = pd.read_csv(raw_data.get_annotation_file_path())
    if not args.include_NBI:
        selected_df = annot_df[annot_df["imaging type"] == 'WLI'].copy()
    else:
        selected_df = annot_df.copy()

    # Select the train, validation, and test subsets
    trn_df = selected_df[selected_df["sub_dataset"] == 'train']
    val_df = selected_df[selected_df["sub_dataset"] == 'val']
    tst_df = selected_df[selected_df["sub_dataset"] == 'test']

    # Copy images from the original directory (raw data) to the partitions directory
    for label in raw_data.labels:
        print(f"--- Label: {label} ------------------------------------------------------------")
        print("Copying train data.")
        trn_single_label_df = trn_df[trn_df["tissue type"] == label]
        trn_file_names = list(trn_single_label_df["HLY"])
        trn_file_paths = [os.path.join(args.input_dir, label, fn) for fn in trn_file_names]
        trn_file_paths = [str(file_path) for file_path in trn_file_paths]
        partitions_data.add_images(subset="TRN", label=label, image_paths=trn_file_paths)

        print("Copying validation data.")
        val_single_label_df = val_df[val_df["tissue type"] == label]
        val_file_names = list(val_single_label_df["HLY"])
        val_file_paths = [os.path.join(args.input_dir, label, fn) for fn in val_file_names]
        val_file_paths = [str(file_path) for file_path in val_file_paths]
        partitions_data.add_images(subset='VAL', label=label, image_paths=val_file_paths)

        print("Copying test data.")
        tst_single_label_df = tst_df[tst_df["tissue type"] == label]
        tst_file_names = list(tst_single_label_df["HLY"])
        tst_file_paths = [os.path.join(args.input_dir, label, fn) for fn in tst_file_names]
        tst_file_paths = [str(file_path) for file_path in tst_file_paths]
        partitions_data.add_images(subset='TST', label=label, image_paths=tst_file_paths)


if __name__ == "__main__":
    main()
