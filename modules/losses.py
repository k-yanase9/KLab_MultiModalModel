import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class LossCounter():
    def __init__(self):
        self.losses = {'train':[], 'val':[]}

    def add(self, phase, loss):
        self.losses[phase].append(loss)

    def plot_loss(self, result_dir):
        # Plot the loss values.
        plt.plot(self.losses['train'], label='Train')
        plt.plot(self.losses['val'], label='Val')

        # Set the title and axis labels.
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Show the plot.
        plt.savefig(os.path.join(result_dir, "loss.png"))

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        #To Do: FocalLossのサンプル数を計算
        sample_size = None
        return focal_loss,sample_size
