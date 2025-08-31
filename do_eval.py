"""Evaluate a trained model using the test (TST) dataset.
The test (TST) dataset is normally reserved for final evaluations. The test dataset has a
root directory often named TST. This root directory contains N folder, one per class. The folder
names are class labels. These folders contain images for the corresponding class.
"""

import os
import argparse
import torch
from tqdm.auto import tqdm
from typing import Tuple
from datetime import datetime

import errordefs as err
import datahandlers as dhl
import modelbuilder as mod
import constants as con

BATCH_SIZE = 8

def parse_arguments():
    """Parse command-line arguments."""
    describe = "Evaluate a trained model using the test (TST) dataset."
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group("required")

    required.add_argument("-root", "--root_dir", type=str, required=True,
                          help="Path to root directory containing TRN, VAL, and TST folders.")

    parser.add_argument("-res", "--results_dir", type=str, default=None,
                          help="(Optional) Path to a directory that will store the results. By default, "
                               "results are displayed on screen only.")

    required.add_argument("-mod", "--model_file", type=str, required=True,
                          help="Path to a stored model file (a file with the .pth extension).")

    parser.add_argument("-bs", "--batch_size", type=int, default=BATCH_SIZE,
                        help=f"(Optional) Number of images processed concurrently in a batch. Default: {BATCH_SIZE}")

    args = parser.parse_args()

    if not os.path.isfile(args.model_file):
        raise FileNotFoundError(f"Unavailable model file at {args.model_file}")

    if args.model_file.split(".")[-1] != "pth":
        raise err.UnknownFileExtension("Wrong extension in model file.")

    if not os.path.isdir(args.root_dir):
        raise err.DirectoryNotFoundError(f"Unavailable test directory at {args.root_dir}")

    return args

def get_device() -> str:
    """Determine if the software runs on a GPU or CPU. Includes Nvidia and Apple's GPUs. If there are
    multiple Nvidia GPUs, it uses the first one.
    Return the device identifier for PyTorch operations.
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device


def evaluate_test_data(model, testloader, criterion, device: str) -> Tuple[float, float]:
    """Apply model to a test dataset to compute the loss and accuracy values. The test dataset has a
    similar directory structure as the train and validation datasets. I.e., there is a directory
    dedicated to the test dataset, often called TST. This directory contains N folders, whose names
    are class labels. Each folder contains the images associated with the class.
    Arguments:
        model: A handler that references the model being trained.
        testloader: A data loading function that brings in test data samples
        criterion: The loss function
        device: The selected device to run a validation procedure
    Return:
        A pair of numbers: loss value and accuracy.
    """
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the entire dataset.
    test_loss = valid_running_loss / counter
    test_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return test_loss, test_acc


def main():
    """Run the main sequence of tasks."""
    args = parse_arguments()

    # Create results directory
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

    # Determine target device
    target_device = get_device()

    # Get dataset handles and data loaders
    input_data = dhl.PartitionsData(root_dir_path=args.root_dir)
    tst_data_handle, labels_from_data = dhl.get_test_dataset(tst_dir_path=input_data.tst_path)
    tst_loader = dhl.get_test_loader(dataset_test_handle=tst_data_handle, batch_size=args.batch_size)

    model, labels_from_model = mod.load_stored_data(file_path=args.model_file)
    model = model.to(target_device)

    if set(labels_from_model) != set(labels_from_data):
        msg = f"Labels from data: {labels_from_data} not compatible with labels from model: {labels_from_model}"
        raise err.IncompatibleModel(msg)

    criterion = torch.nn.CrossEntropyLoss()

    tst_loss, tst_acc = evaluate_test_data(model=model, testloader=tst_loader,
                                           criterion=criterion, device=target_device)

    print("---------------------------")
    print(f"Test loss: {tst_loss}")
    print(f"Test Accuracy: {tst_acc}")
    print("---------------------------")

    if args.results_dir:
        test_results_filename = con.TEST_RESULTS_FILE_NAME
        test_results_filepath = os.path.join(args.results_dir, test_results_filename)
        model_name = os.path.basename(args.model_file)
        with open(test_results_filepath, "w") as fh:
            fh.write(f"test loss: {tst_loss}\ntest acc: {tst_acc}\nmodel: {model_name}\n")
        print(f"Evaluation results using test data saved to {test_results_filepath}")

def get_timedate_stamp() -> str:
    """Get a time stamp that can be used to add to titles or file names."""
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    return stamp


if __name__ == "__main__":
    main()
