"""This script tests if the content described in the annotations.csv file exists
as part of the downloaded content directories.
"""

import os
import argparse
import pandas as pd

import errordefs as err
import datahandlers as dh

def parse_arguments():
    """Parse command-line arguments."""
    describe = "Test if content in annotations file exists in downloaded folders."
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group("required arguments")

    required.add_argument("-r", "--root_dir", type=str, required=True,
                          help="Path to root directory of downloaded content.")

    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        raise err.DirectoryNotFoundError(f"Did not find a root dir at {args.root_dir}")

    return args

def main():
    """Run the main sequence of tasks."""
    args = parse_arguments()

    content = dh.RawData(args.root_dir)
    annot_df = pd.read_csv(content.get_annotation_file_path())

    entries = annot_df.values.tolist()

    print("--- Missing files ------------------------")
    all_filepaths = []
    for entry in entries:
        filename, method, label, _ = entry
        filepath = os.path.join(args.root_dir, label, filename)
        all_filepaths.append(filepath)
        if not os.path.isfile(filepath):
            print(filepath)

    print()
    print("--- Missing annotation entries ---------------------")
    for folder_path in content.get_folder_paths():
        folder_name = os.path.basename(folder_path)
        image_path_list = content.get_image_paths(folder_name)

        for img_path in image_path_list:
            if img_path not in all_filepaths:
                print(img_path)


if __name__ == "__main__":
    main()

