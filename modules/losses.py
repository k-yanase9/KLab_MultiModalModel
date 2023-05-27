import os
import matplotlib.pyplot as plt

class LossCounter():
    def __init__(self, train_loader_len, val_loader_len):
        self.loader_len = {'train':train_loader_len, 'val':val_loader_len}
        self.losses = {'train':[], 'val':[]}
        self.total_loss = {'train':0.0, 'val':0.0}

    def add_loss(self, phase, loss):
        self.total_loss[phase] += loss

    def count_and_get_loss(self):
        for phase in ['train', 'val']:
            self.losses[phase].append(self.total_loss[phase] / self.loader_len[phase])
            self.total_loss[phase] = 0.0
        return self.losses['train'][-1], self.losses['val'][-1]

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