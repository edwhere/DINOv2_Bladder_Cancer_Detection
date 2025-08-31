"""Utility functions and classes to perform operations related to DINOv2 training and
inference.
"""

import torch
import matplotlib.pyplot as plt
import os
import pandas as pd

import constants as con


class SaveBestModel2:
    """ Class to save the best model while training. If the current epoch's validation loss is less
    than the previous least loss, then save the model state.
    """

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, out_dir, name, labels):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'storage_type': "dv2_best",
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'labels': labels,
            }, str(os.path.join(out_dir, 'best_' + name + '.pth')))


def save_model2(epochs, model, optimizer, criterion, out_dir, name, labels):
    """ Function to save the trained model to storage."""
    torch.save({
        'storage_type': 'dv2_final',
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'labels': labels,
    }, str(os.path.join(out_dir, name + '.pth')))


def save_learn_curves(train_acc, valid_acc, train_loss, valid_loss, out_dir):
    """Save accuracy and loss curves to a CSV file."""
    results = {
        "trn_acc": train_acc,
        "val_acc": valid_acc,
        "trn_loss": train_loss,
        "val_loss": valid_loss
    }

    results_df = pd.DataFrame(results)
    results_file_path = os.path.join(out_dir, con.RESULTS_FILE_NAME)
    results_df.to_csv(results_file_path)
    print(f"Learning curves data saved to {results_file_path}")

def save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir):
    """ Plot accuracy and loss functions and save the figures.
    """
    xvals = list(range(1, len(train_acc) + 1))

    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(xvals, train_acc, color='tab:blue', linestyle='-', label='train accuracy')
    plt.plot(xvals, valid_acc, color='tab:red', linestyle='-', label='validation accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(xvals)
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(xvals, train_loss, color='tab:blue', linestyle='-', label='train loss')
    plt.plot(xvals, valid_loss, color='tab:red', linestyle='-', label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(xvals)
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

    print(f"Learning curves figures saved to directory {out_dir}")
